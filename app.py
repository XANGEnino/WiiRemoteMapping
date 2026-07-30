"""Wii Remote -> keyboard mapper GUI."""

import queue
import threading
import tkinter as tk
from tkinter import ttk

import config
import keysender
from recorder import SwingRecorder
from tennis import TennisPractice
from viewer import MotionViewer
from wiimote import Wiimote

DISPLAY_ORDER = [
    "A", "B", "One", "Two", "Plus", "Minus", "Home",
    "Up", "Down", "Left", "Right",
    "SwingUp", "SwingDown", "SwingLeft", "SwingRight",
]

BUTTON_LABELS = {
    "One": "1", "Two": "2", "Plus": "+", "Minus": "−",
    "Up": "D-pad Up", "Down": "D-pad Down",
    "Left": "D-pad Left", "Right": "D-pad Right",
    "SwingUp": "Swing Up", "SwingDown": "Swing Down",
    "SwingLeft": "Swing Left", "SwingRight": "Swing Right",
}

# How long a swing's indicator stays lit (ms) — swings are momentary.
FLASH_MS = 150
# The compass arrow holds longer so the last direction stays readable.
ARROW_HOLD_MS = 3000

IDLE_COLOR = "#d9d9d9"
ACTIVE_COLOR = "#4caf50"


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Wii Remote Mapper")
        self.root.resizable(False, False)

        self.mappings = config.load_mappings()
        self.events = queue.Queue()
        self.wiimote = Wiimote(self.events)
        self.capturing = None  # button name currently being remapped
        self.held_keys = {}  # wiimote button -> key name currently held down
        self.viewer = None  # MotionViewer window, when open
        self.tennis = None  # TennisPractice window, when open
        self.recorder = None  # SwingRecorder window, when open

        self._build_ui()
        self.root.bind("<KeyPress>", self._on_keyboard)
        self.root.bind("<KeyRelease>", self._on_key_release)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._reconnect()
        self.root.after(16, self._poll_events)

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=(10, 10, 10, 5))
        top.pack(fill="x")
        self.status_label = ttk.Label(top, text="Connecting...")
        self.status_label.pack(side="left")
        self.reconnect_btn = ttk.Button(
            top, text="Reconnect", command=self._reconnect
        )
        self.reconnect_btn.pack(side="right")
        ttk.Button(
            top, text="Motion Viewer", command=self._open_viewer
        ).pack(side="right", padx=(0, 6))
        ttk.Button(
            top, text="Tennis Practice", command=self._open_tennis
        ).pack(side="right", padx=(0, 6))
        ttk.Button(
            top, text="Swing Recorder", command=self._open_recorder
        ).pack(side="right", padx=(0, 6))

        body = ttk.Frame(self.root)
        body.pack(fill="both")

        table = ttk.Frame(body, padding=(10, 5, 10, 10))
        table.pack(side="left", fill="both")

        self._build_compass(body)

        self.indicators = {}
        self.key_labels = {}
        self.set_buttons = {}
        for row, name in enumerate(DISPLAY_ORDER):
            indicator = tk.Frame(
                table, width=14, height=14, bg=IDLE_COLOR,
                relief="sunken", borderwidth=1,
            )
            indicator.grid(row=row, column=0, padx=(0, 8), pady=3)
            self.indicators[name] = indicator

            ttk.Label(
                table, text=BUTTON_LABELS.get(name, name), width=12
            ).grid(row=row, column=1, sticky="w")

            key_label = ttk.Label(
                table, text=self.mappings[name], width=18,
                relief="groove", anchor="center",
            )
            key_label.grid(row=row, column=2, padx=8, pady=3)
            self.key_labels[name] = key_label

            set_btn = ttk.Button(
                table, text="Set",
                command=lambda n=name: self._toggle_capture(n),
            )
            set_btn.grid(row=row, column=3)
            self.set_buttons[name] = set_btn

    def _build_compass(self, parent):
        """A little compass whose arrow lights up in the swung direction."""
        size, center, reach = 360, 180, 132
        frame = ttk.Frame(parent, padding=(0, 5, 10, 10))
        frame.pack(side="right", anchor="n")
        ttk.Label(frame, text="Last swing").pack()
        canvas = tk.Canvas(
            frame, width=size, height=size,
            highlightthickness=1, highlightbackground=IDLE_COLOR,
        )
        canvas.pack(pady=(4, 0))
        canvas.create_oval(center - 9, center - 9, center + 9, center + 9,
                           fill=IDLE_COLOR, outline="")
        tips = {
            "SwingUp": (center, center - reach),
            "SwingDown": (center, center + reach),
            "SwingLeft": (center - reach, center),
            "SwingRight": (center + reach, center),
        }
        self.arrows = {}
        for name, (tx, ty) in tips.items():
            self.arrows[name] = canvas.create_line(
                center, center, tx, ty,
                width=12, fill=IDLE_COLOR, arrow="last",
                arrowshape=(33, 39, 15), capstyle="round",
            )
        self.compass = canvas
        self._lit_arrow = None
        self._arrow_reset_job = None

    def _flash_arrow(self, name):
        arrow = self.arrows.get(name)
        if arrow is None:
            return
        if self._arrow_reset_job is not None:
            self.root.after_cancel(self._arrow_reset_job)
        if self._lit_arrow is not None and self._lit_arrow != arrow:
            self.compass.itemconfig(self._lit_arrow, fill=IDLE_COLOR)
        self._lit_arrow = arrow
        self.compass.itemconfig(arrow, fill=ACTIVE_COLOR)
        self._arrow_reset_job = self.root.after(
            ARROW_HOLD_MS, self._reset_arrow
        )

    def _reset_arrow(self):
        self._arrow_reset_job = None
        if self._lit_arrow is not None:
            self.compass.itemconfig(self._lit_arrow, fill=IDLE_COLOR)
            self._lit_arrow = None

    # ---- remapping ----

    def _toggle_capture(self, name):
        if self.capturing == name:
            self._end_capture()
            return
        if self.capturing is not None:
            self._end_capture()
        self.capturing = name
        self.key_labels[name].config(text="press a key...")
        self.set_buttons[name].config(text="Cancel")
        self.root.focus_set()

    def _end_capture(self):
        name, self.capturing = self.capturing, None
        if name is not None:
            self.key_labels[name].config(text=self.mappings[name])
            self.set_buttons[name].config(text="Set")

    def _on_keyboard(self, event):
        if self.capturing is None:
            return
        key_name = keysender.keysym_to_name(event.keysym, event.char)
        if key_name is None:
            return
        if key_name in keysender.MODIFIERS:
            # Combo in progress — wait for a non-modifier key to finish it.
            held = keysender.modifiers_from_state(event.state)
            if key_name not in held:
                held.append(key_name)
            self.key_labels[self.capturing].config(
                text="+".join(held) + "+..."
            )
            return
        combo = keysender.modifiers_from_state(event.state)
        combo.append(key_name)
        self._assign("+".join(combo))

    def _on_key_release(self, event):
        if self.capturing is None:
            return
        key_name = keysender.keysym_to_name(event.keysym, event.char)
        if key_name not in keysender.MODIFIERS:
            return
        # Modifier released without completing a combo: alone, it becomes
        # the mapping itself; with others still held, the combo continues.
        held = [
            m for m in keysender.modifiers_from_state(event.state)
            if m != key_name
        ]
        if held:
            self.key_labels[self.capturing].config(
                text="+".join(held) + "+..."
            )
        else:
            self._assign(key_name)

    def _assign(self, name):
        self.mappings[self.capturing] = name
        config.save_mappings(self.mappings)
        self._end_capture()

    # ---- wiimote events ----

    def _poll_events(self):
        try:
            while True:
                event = self.events.get_nowait()
                if event[0] == "status":
                    self._on_status(event[1])
                elif event[0] == "gesture":
                    self._on_gesture(event[1])
                elif event[0] == "accel":
                    if self.viewer is not None:
                        self.viewer.feed_accel(*event[1:])
                    if self.tennis is not None:
                        self.tennis.feed_accel(*event[1:])
                    if self.recorder is not None:
                        self.recorder.feed_accel(*event[1:])
                elif event[0] == "gyro":
                    if self.viewer is not None:
                        self.viewer.feed_gyro(*event[1:])
                    if self.recorder is not None:
                        self.recorder.feed_gyro(*event[1:])
                else:
                    self._on_button(event[1], event[2])
        except queue.Empty:
            pass
        self.root.after(16, self._poll_events)

    def _on_status(self, status):
        if status == "connected":
            self.status_label.config(text="Connected", foreground="#2e7d32")
        else:
            self.status_label.config(
                text="Disconnected — press a button on the remote, "
                     "then click Reconnect",
                foreground="#c62828",
            )
        self.reconnect_btn.config(state="normal")

    def _on_button(self, name, pressed):
        self.indicators[name].config(
            bg=ACTIVE_COLOR if pressed else IDLE_COLOR
        )
        # While the recorder is open, holding A records a swing instead
        # of typing its mapped key.  While tennis practice is open (and
        # the recorder isn't), A feeds a ball and the D-pad picks the
        # forehand/backhand side — again instead of typing.
        to_recorder = self.recorder is not None and name == "A"
        to_tennis = (not to_recorder and self.tennis is not None
                     and name in ("A", "Left", "Right"))
        if pressed:
            if to_recorder:
                self.recorder.on_record_button(True)
            elif to_tennis:
                self.tennis.on_button(name)
            elif self.capturing is None:
                # While remapping, don't type: our own synthetic key
                # events would be captured as the new mapping.
                key = self.mappings[name]
                self.held_keys[name] = key
                keysender.press(key)
        else:
            # Release any held key even when the recorder took over A
            # mid-hold, so no key stays stuck down.
            key = self.held_keys.pop(name, None)
            if key is not None:
                keysender.release(key)
            if to_recorder:
                self.recorder.on_record_button(False)

    def _on_gesture(self, name):
        if self.viewer is not None:
            self.viewer.mark_gesture(name)
        # While remapping, don't fire — the user is assigning a key, not playing.
        if self.capturing is not None:
            return
        if self.tennis is None and self.recorder is None:
            keysender.tap(self.mappings[name])
        # else: tennis runs its own stroke detector, and the recorder is
        # capturing practice swings — either way a swing is not a keyboard
        # tap, and typing into whichever window has focus would be chaos.
        indicator = self.indicators[name]
        indicator.config(bg=ACTIVE_COLOR)
        self.root.after(FLASH_MS, lambda: indicator.config(bg=IDLE_COLOR))
        self._flash_arrow(name)

    # ---- motion viewer & tennis practice ----

    def _open_viewer(self):
        if self.viewer is not None:
            self.viewer.lift()
            self.viewer.focus_set()
            return
        self.viewer = MotionViewer(self.root, on_close=self._on_viewer_close)
        self._update_streaming()

    def _on_viewer_close(self):
        self.viewer = None
        self._update_streaming()

    def _open_tennis(self):
        if self.tennis is not None:
            self.tennis.lift()
            self.tennis.focus_set()
            return
        self.tennis = TennisPractice(self.root, on_close=self._on_tennis_close)
        self._update_streaming()

    def _on_tennis_close(self):
        self.tennis = None
        self._update_streaming()

    def _open_recorder(self):
        if self.recorder is not None:
            self.recorder.lift()
            self.recorder.focus_set()
            return
        self.recorder = SwingRecorder(
            self.root, on_close=self._on_recorder_close
        )
        self._update_streaming()

    def _on_recorder_close(self):
        self.recorder = None
        self._update_streaming()

    def _update_streaming(self):
        """Stream accel events only while a window that uses them is open."""
        self.wiimote.emit_accel = (
            self.viewer is not None
            or self.tennis is not None
            or self.recorder is not None
        )

    # ---- connection ----

    def _reconnect(self):
        if self.wiimote.connected:
            return
        self.status_label.config(text="Connecting...", foreground="")
        self.reconnect_btn.config(state="disabled")

        def worker():
            if not self.wiimote.connect():
                self.events.put(("status", "disconnected"))

        threading.Thread(target=worker, daemon=True).start()

    def _on_close(self):
        for key in self.held_keys.values():
            keysender.release(key)
        self.wiimote.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
