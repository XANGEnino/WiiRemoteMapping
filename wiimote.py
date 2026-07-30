"""Communication with a Wii remote (Bluetooth HID) via hidapi.

The setup sequence mirrors how Dolphin talks to real Wii remotes
(InputCommon/ControllerInterface/Wiimote/WiimoteController.cpp):

* Every configuration report is sent with the acknowledgement flag and
  re-sent until the remote confirms it with an ack report (0x22) — a
  single unacknowledged write is easily lost on the Windows Bluetooth
  stack, leaving the remote stuck in its buttons-only default mode.
* The accelerometer is normalized with the zero/+1g calibration block
  stored in the remote's EEPROM instead of assumed constants.
* A status report (0x20) the remote sends on its own — extension
  plugged/unplugged, and some remotes emit one right after connecting —
  stops all data reporting, so the reporting mode is negotiated again
  whenever one arrives.
* A MotionPlus gyroscope (built into RVL-CNT-01-TR remotes, a dongle on
  older ones) is probed for and activated when present: its angular
  rates are streamed as ("gyro", ...) events so the viewer can show yaw,
  which the accelerometer alone cannot sense.  A remote without one
  simply gets an error back from the probe and stays accelerometer-only.

Protocol reference: https://wiibrew.org/wiki/Wiimote
"""

import threading
import time

import hid

from motion import SwingDetector

VENDOR_ID = 0x057E
PRODUCT_IDS = (0x0306, 0x0330)  # RVL-CNT-01 and RVL-CNT-01-TR

# Output report IDs.
REPORT_LEDS = 0x11
REPORT_MODE = 0x12
REPORT_REQUEST_STATUS = 0x15
REPORT_WRITE_DATA = 0x16
REPORT_READ_DATA = 0x17

# Input report IDs.
REPORT_STATUS = 0x20
REPORT_READ_REPLY = 0x21
REPORT_ACK = 0x22
REPORT_CORE_BUTTONS = 0x30
REPORT_CORE_ACCEL = 0x31  # core buttons + accelerometer
REPORT_CORE_ACCEL_EXT = 0x35  # + 16 extension bytes (MotionPlus gyro)

# Data reports whose bytes 3-5 carry the accelerometer.
ACCEL_REPORTS = (0x31, 0x33, 0x35, 0x37)

# Flags in the first payload byte of output reports: bit 1 asks the
# remote to answer with an ack report (0x22); bit 2 of the mode report
# keeps reports streaming even when nothing changes (steady accel data).
FLAG_ACK = 0x02
FLAG_CONTINUOUS = 0x04

LED_PLAYER_1 = 0x10

# Accelerometer calibration block in EEPROM: zero point and +1g reading
# for each axis (plus volume/checksum bytes we don't use).
CAL_ADDR = 0x16
CAL_SIZE = 10
# Typical values (WiiBrew) used when the stored block is unreadable.
CAL_FALLBACK = ([0x80] * 3, [26] * 3)

# MotionPlus control registers.  While dormant it answers on the 0xa6
# bus: writing 0x55 to 0xf0 wakes it, then 0x04 to 0xfe activates it as
# the extension (it re-appears on the normal 0xa4 bus and data reports
# start carrying its gyro).  A remote without one errors on the first
# write.  Addresses are (space-prefixed) 24-bit, big-endian.
MP_INIT_ADDR = (0xA6, 0x00, 0xF0)
MP_ACTIVATE_ADDR = (0xA6, 0x00, 0xFE)

# Gyro conversion (WiiBrew): a still remote reads ~8063 counts on each
# axis; in slow (high-resolution) mode there are 8192/595 counts per
# deg/s, and fast mode is 2000/440 times less sensitive.
MP_ZERO = 8063.0
MP_SLOW_SCALE = 8192.0 / 595.0
MP_FAST_SCALE = MP_SLOW_SCALE * 440.0 / 2000.0

RETRY_S = 1.0   # resend an unacknowledged setup report after this long
MAX_TRIES = 5   # then stop waiting for a confirmation and move on

# (byte index in report 0x30 payload, bit mask, button name)
BUTTON_BITS = [
    (0, 0x01, "Left"),
    (0, 0x02, "Right"),
    (0, 0x04, "Down"),
    (0, 0x08, "Up"),
    (0, 0x10, "Plus"),
    (1, 0x01, "Two"),
    (1, 0x02, "One"),
    (1, 0x04, "B"),
    (1, 0x08, "A"),
    (1, 0x10, "Minus"),
    (1, 0x80, "Home"),
]

BUTTONS = [name for _, _, name in BUTTON_BITS]


class Wiimote:
    """Reads core button state and pushes events into a queue.

    Events put on the queue:
        ("status", "connected" | "disconnected")
        ("button", <name>, <pressed: bool>)
        ("gesture", <swing name>)          # momentary; tap the mapped key
        ("accel", x, y, z, deviation)      # in g; only while emit_accel is set
        ("gyro", yaw, pitch, roll, dt)     # deg/s + seconds since previous
                                           # sample; MotionPlus remotes only,
                                           # and only while emit_accel is set
    """

    def __init__(self, event_queue):
        self.emit_accel = False  # stream ("accel", ...) events (~100/s)
        self._queue = event_queue
        self._device = None
        self._thread = None
        self._stop = threading.Event()
        self._pressed = frozenset()
        self._swing = SwingDetector()
        self._reset_setup()

    @property
    def connected(self):
        return self._device is not None

    def connect(self):
        """Try to open the remote and start the reader thread.

        Returns True on success. Safe to call again after a disconnect.
        """
        if self.connected:
            return True
        candidates = [
            d for d in hid.enumerate()
            if d["vendor_id"] == VENDOR_ID and d["product_id"] in PRODUCT_IDS
        ]
        for info in candidates:
            device = hid.device()
            try:
                device.open_path(info["path"])
                self._device = device
                # Probe the link; also the first step of the setup chain.
                self._write(REPORT_REQUEST_STATUS, 0x00)
            except (OSError, ValueError):
                device.close()
                self._device = None
                continue
            self._pressed = frozenset()
            self._swing.reset()
            self._reset_setup()
            self._pending = ("status", time.monotonic())
            self._tries["status"] = 1
            self._stop.clear()
            self._thread = threading.Thread(target=self._read_loop, daemon=True)
            self._thread.start()
            self._queue.put(("status", "connected"))
            return True
        return False

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._close_device()

    def _close_device(self):
        if self._device is not None:
            try:
                self._device.close()
            except OSError:
                pass
            self._device = None

    def _write(self, *report):
        data = bytes(report)
        try:
            written = self._device.write(data)
        except OSError:
            written = -1
        if written <= 0:
            # Microsoft Bluetooth stack quirk (mainly -TR remotes): writes
            # must match the full output report length, so retry padded.
            self._device.write(data.ljust(22, b"\x00"))

    # ---- Dolphin-style setup task chain ----

    def _reset_setup(self):
        """Forget all negotiated state (fresh connection)."""
        self._ext_connected = None  # unknown until the first status report
        self._leds_ok = False
        self._mode_ok = False
        self._calibration = None    # ([zero x,y,z], [counts per +1g x,y,z])
        self._mp_state = "unknown"  # unknown | activating | active | absent
        self._mp_bias = [MP_ZERO] * 3   # per-axis gyro zero, adapted at rest
        self._mp_last = 0.0             # monotonic time of the last gyro frame
        self._pending = None        # (task name, monotonic time sent)
        self._tries = {}

    def _next_task(self):
        # Same order as Dolphin's RunTasks: status, LEDs, reporting
        # mode, accelerometer calibration, then the MotionPlus probe.
        if self._ext_connected is None:
            return "status"
        if not self._leds_ok:
            return "leds"
        if not self._mode_ok:
            return "mode"
        if self._calibration is None:
            return "calibration"
        if self._mp_state == "unknown":
            return "mp_init"
        if self._mp_state == "activating":
            return "mp_activate"
        return None

    def _run_tasks(self):
        now = time.monotonic()
        if self._pending is not None and now - self._pending[1] < RETRY_S:
            return
        while True:
            task = self._next_task()
            if task is None:
                self._pending = None
                return
            if self._tries.get(task, 0) < MAX_TRIES:
                break
            # No confirmation after several sends (some third-party
            # remotes skip acks): assume it took and keep going.
            self._assume_done(task)
        self._tries[task] = self._tries.get(task, 0) + 1
        self._pending = (task, now)
        try:
            self._send(task)
        except OSError:
            pass  # the read loop notices a dead link and reports it

    def _send(self, task):
        if task == "status":
            self._write(REPORT_REQUEST_STATUS, 0x00)
        elif task == "leds":
            self._write(REPORT_LEDS, LED_PLAYER_1 | FLAG_ACK)
        elif task == "mode":
            mode = (REPORT_CORE_ACCEL_EXT if self._mp_state == "active"
                    else REPORT_CORE_ACCEL)
            self._write(REPORT_MODE, FLAG_ACK | FLAG_CONTINUOUS, mode)
        elif task == "calibration":
            # Read CAL_SIZE bytes from EEPROM address CAL_ADDR
            # (space, 24-bit address, 16-bit size, all big-endian).
            self._write(REPORT_READ_DATA, 0x00,
                        0x00, 0x00, CAL_ADDR, 0x00, CAL_SIZE)
        elif task == "mp_init":
            self._write_register(MP_INIT_ADDR, 0x55)
        elif task == "mp_activate":
            self._write_register(MP_ACTIVATE_ADDR, 0x04)

    def _write_register(self, addr, value):
        # 0x16: space flags, 24-bit address, size, then 16 data bytes.
        # 0x04 selects the control registers instead of EEPROM.
        self._write(REPORT_WRITE_DATA, 0x04, *addr, 0x01, value,
                    *([0x00] * 15))

    def _assume_done(self, task):
        if task == "status":
            self._ext_connected = False
        elif task == "leds":
            self._leds_ok = True
        elif task == "mode":
            self._mode_ok = True
        elif task == "calibration":
            self._calibration = CAL_FALLBACK
        elif task in ("mp_init", "mp_activate"):
            # No ack: assume there is no MotionPlus to set up.
            self._mp_state = "absent"

    # ---- input reports ----

    def _read_loop(self):
        while not self._stop.is_set():
            try:
                data = self._device.read(32, timeout_ms=100)
            except (OSError, ValueError):
                break
            if data:
                self._handle_report(data)
            self._run_tasks()
        if not self._stop.is_set():
            # Read failed: the remote powered off or went out of range.
            self._close_device()
            self._release_all()
            self._queue.put(("status", "disconnected"))

    def _handle_report(self, data):
        rid = data[0]
        # Button state rides along in every report we handle: status,
        # read replies and acks all start with the two button bytes.
        wanted = (0x30 <= rid <= 0x37
                  or rid in (REPORT_STATUS, REPORT_READ_REPLY, REPORT_ACK))
        if wanted and len(data) >= 3:
            self._handle_buttons(data[1], data[2])
        if rid == REPORT_STATUS and len(data) >= 7:
            self._handle_status(data)
        elif rid == REPORT_READ_REPLY and len(data) >= 6:
            self._handle_read_reply(data)
        elif rid == REPORT_ACK and len(data) >= 5:
            self._handle_ack(data)
        elif rid in ACCEL_REPORTS and len(data) >= 6:
            self._handle_accel(data[3], data[4], data[5])
            if rid == REPORT_CORE_ACCEL_EXT and len(data) >= 12:
                self._handle_motionplus(data[6:12])

    def _handle_status(self, data):
        solicited = self._pending is not None and self._pending[0] == "status"
        if solicited:
            self._pending = None
        was = self._ext_connected
        self._ext_connected = bool(data[3] & 0x02)
        if self._mp_state == "active" and not self._ext_connected:
            # The MotionPlus dongle was unplugged: probe again — a failed
            # probe just settles back to accelerometer-only.
            self._mp_state = "unknown"
            self._tries.pop("mp_init", None)
            self._tries.pop("mp_activate", None)
        # The remote stops sending data reports after a status report it
        # sent on its own (extension plugged/unplugged): negotiate the
        # reporting mode again, as Dolphin does on extension port events.
        if not solicited or (was is not None and was != self._ext_connected):
            self._mode_ok = False
            self._tries.pop("mode", None)

    def _handle_ack(self, data):
        acked_report, error = data[3], data[4]
        if acked_report == REPORT_WRITE_DATA:
            self._handle_write_ack(error)
            return
        if acked_report == REPORT_LEDS:
            task = "leds"
            if error == 0:
                self._leds_ok = True
        elif acked_report == REPORT_MODE:
            task = "mode"
            if error == 0:
                self._mode_ok = True
        else:
            return
        if self._pending is not None and self._pending[0] == task:
            # On success move straight to the next task; on an error
            # ack hold off for RETRY_S before resending.
            self._pending = None if error == 0 else (task, time.monotonic())

    def _handle_write_ack(self, error):
        """Acks for the MotionPlus register writes (the only writes made)."""
        task = self._pending[0] if self._pending is not None else None
        if task == "mp_init":
            # An error (7) means nothing answered at the MotionPlus
            # address: a plain remote without a gyro.
            self._mp_state = "activating" if error == 0 else "absent"
            self._pending = None
        elif task == "mp_activate":
            if error == 0:
                self._mp_state = "active"
                # Re-negotiate reporting so data reports carry the
                # extension bytes holding the gyro.
                self._mode_ok = False
                self._tries.pop("mode", None)
            else:
                self._mp_state = "absent"
            self._pending = None

    def _handle_read_reply(self, data):
        address = (data[4] << 8) | data[5]
        if address != CAL_ADDR:
            return
        error = data[3] & 0x0F
        if self._pending is not None and self._pending[0] == "calibration":
            self._pending = (None if error == 0
                             else ("calibration", time.monotonic()))
        if error or len(data) < 6 + CAL_SIZE:
            return
        block = data[6:6 + CAL_SIZE]
        zero = list(block[0:3])
        gain = [block[4 + i] - zero[i] for i in range(3)]
        # Sanity-check the stored block (Dolphin verifies its checksum);
        # a real remote reads ~26 counts per g on each axis.
        if all(10 <= g <= 100 for g in gain):
            self._calibration = (zero, gain)
        else:
            self._calibration = CAL_FALLBACK

    def _handle_buttons(self, byte0, byte1):
        state = [byte0, byte1]
        now = frozenset(
            name for idx, mask, name in BUTTON_BITS if state[idx] & mask
        )
        for name in now - self._pressed:
            self._queue.put(("button", name, True))
        for name in self._pressed - now:
            self._queue.put(("button", name, False))
        self._pressed = now

    def _handle_accel(self, x, y, z):
        # Accelerometer data arriving proves the reporting mode took
        # hold, even if the ack report itself was lost.
        if not self._mode_ok:
            self._mode_ok = True
            if self._pending is not None and self._pending[0] == "mode":
                self._pending = None
        if self._calibration is None:
            return  # like Dolphin, wait for calibration before using accel
        zero, gain = self._calibration
        # The high 8 bits of each axis are plenty to detect a swing; the
        # low bits (tucked into the button bytes) don't affect our masks.
        gx = (x - zero[0]) / gain[0]
        gy = (y - zero[1]) / gain[1]
        gz = (z - zero[2]) / gain[2]
        swing = self._swing.feed(gx, gy, gz)
        if swing is not None:
            self._queue.put(("gesture", swing))
        if self.emit_accel:
            self._queue.put(("accel", gx, gy, gz, self._swing.deviation))

    def _handle_motionplus(self, ext):
        """Decode one 6-byte MotionPlus frame into deg/s and queue it."""
        # Bits 1:0 of byte 5 are (1, 0) in every MotionPlus data frame;
        # anything else is a passthrough or garbage frame.
        if (ext[5] & 0x03) != 0x02:
            return
        raws = (
            ext[0] | ((ext[3] & 0xFC) << 6),  # yaw    (about body Z)
            ext[2] | ((ext[5] & 0xFC) << 6),  # pitch  (about body X)
            ext[1] | ((ext[4] & 0xFC) << 6),  # roll   (about body Y)
        )
        slow = (ext[3] & 0x02, ext[3] & 0x01, ext[4] & 0x02)
        # While the remote rests, follow the zero point: gyro bias moves
        # with temperature, and any error here integrates into yaw drift.
        if self._swing.deviation < 0.05:
            for i, raw in enumerate(raws):
                if abs(raw - self._mp_bias[i]) < 80:
                    self._mp_bias[i] += 0.02 * (raw - self._mp_bias[i])
        yaw, pitch, roll = (
            (raw - self._mp_bias[i])
            / (MP_SLOW_SCALE if slow[i] else MP_FAST_SCALE)
            for i, raw in enumerate(raws)
        )
        now = time.monotonic()
        dt = min(now - self._mp_last, 0.1) if self._mp_last else 0.0
        self._mp_last = now
        if self.emit_accel:
            self._queue.put(("gyro", yaw, pitch, roll, dt))

    def _release_all(self):
        for name in self._pressed:
            self._queue.put(("button", name, False))
        self._pressed = frozenset()
