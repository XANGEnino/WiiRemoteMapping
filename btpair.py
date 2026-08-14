"""Bluetooth discovery and pairing for Wii remotes (Windows).

Windows only exposes a HID interface — the thing hidapi can open — for a
Wii remote that is paired *and* has its HID service switched on.  Dolphin
does that work itself instead of leaving it to the Settings app, which is
why remotes connect while Dolphin is running and not otherwise.  This is
that same work, ported from Dolphin's Windows backend
(Core/HW/WiimoteReal/IOWin.cpp, class WiimoteScannerWindows):

* Remembered-but-unauthenticated remotes are removed first.  Windows
  keeps a device entry around after a remote powers off, and an
  unauthenticated entry can never reconnect — deleting it is what lets
  the next inquiry find the remote again, instead of the user having to
  remove it by hand in Settings.
* A Bluetooth inquiry looks for devices named "Nintendo RVL-CNT...",
  i.e. a remote in discovery mode (blinking after sync or 1+2).
* Pairing uses the *host radio's* address as the pass key — the sync
  button method.  A remote paired that way stores the host and seeks
  reconnection on any button press later, which is what makes plain
  scanning (`scan()`) enough from then on.
* Finally the HID service is enabled, and only then does the remote
  appear to hidapi.

All of it is ctypes over bthprops.cpl, the Windows Bluetooth API DLL.
"""

import ctypes
import threading
from ctypes import wintypes

BLUETOOTH_MAX_NAME_SIZE = 248

ERROR_SUCCESS = 0
ERROR_MORE_DATA = 234

BLUETOOTH_SERVICE_ENABLE = 0x01

# Dolphin's DEFAULT_INQUIRY_LENGTH: an inquiry runs for this many
# 1.28-second slices, so ~3.8s per sweep.
DEFAULT_INQUIRY_LENGTH = 3
# Dolphin's ITERATION_COUNT — "Windows isn't so cooperative", one pass
# often misses a remote that is plainly in discovery mode.
PAIR_ITERATIONS = 3

# Devices Dolphin accepts as a remote (IsWiimoteName); third-party
# remotes spoof this name because the Wii itself only checks the name.
WIIMOTE_NAME_PREFIX = "Nintendo RVL-CNT"

# Which address the remote expects as the pass key.  Pressing sync means
# the host's address, pressing 1+2 means the remote's own.  Guessing
# wrong makes Windows drop the remote immediately, and Dolphin considers
# the 1+2 method effectively pointless, so sync is the default.
AUTH_SYNC_BUTTON = "sync"
AUTH_ONE_TWO = "one_two"


class _BluetoothAddress(ctypes.Union):
    _fields_ = [
        ("ull", ctypes.c_ulonglong),
        ("rgBytes", ctypes.c_ubyte * 6),
    ]


class _SystemTime(ctypes.Structure):
    _fields_ = [
        (name, wintypes.WORD) for name in (
            "wYear", "wMonth", "wDayOfWeek", "wDay",
            "wHour", "wMinute", "wSecond", "wMilliseconds",
        )
    ]


class _Guid(ctypes.Structure):
    _fields_ = [
        ("Data1", wintypes.DWORD),
        ("Data2", wintypes.WORD),
        ("Data3", wintypes.WORD),
        ("Data4", ctypes.c_ubyte * 8),
    ]


class _FindRadioParams(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD)]


class _RadioInfo(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("address", _BluetoothAddress),
        ("szName", ctypes.c_wchar * BLUETOOTH_MAX_NAME_SIZE),
        ("ulClassofDevice", wintypes.ULONG),
        ("lmpSubversion", ctypes.c_ushort),
        ("manufacturer", ctypes.c_ushort),
    ]


class _DeviceInfo(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("Address", _BluetoothAddress),
        ("ulClassofDevice", wintypes.ULONG),
        ("fConnected", wintypes.BOOL),
        ("fRemembered", wintypes.BOOL),
        ("fAuthenticated", wintypes.BOOL),
        ("stLastSeen", _SystemTime),
        ("stLastUsed", _SystemTime),
        ("szName", ctypes.c_wchar * BLUETOOTH_MAX_NAME_SIZE),
    ]


class _DeviceSearchParams(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("fReturnAuthenticated", wintypes.BOOL),
        ("fReturnRemembered", wintypes.BOOL),
        ("fReturnUnknown", wintypes.BOOL),
        ("fReturnConnected", wintypes.BOOL),
        ("fIssueInquiry", wintypes.BOOL),
        ("cTimeoutMultiplier", ctypes.c_ubyte),
        ("hRadio", wintypes.HANDLE),
    ]


# {00001124-0000-1000-8000-00805F9B34FB}
HID_SERVICE_UUID = _Guid(
    0x00001124, 0x0000, 0x1000,
    (ctypes.c_ubyte * 8)(0x80, 0x00, 0x00, 0x80, 0x5F, 0x9B, 0x34, 0xFB),
)

_PWCHAR = ctypes.POINTER(ctypes.c_wchar)


def _load_bluetooth_api():
    """Open bthprops.cpl and declare the calls we use, or give up."""
    try:
        dll = ctypes.WinDLL("bthprops.cpl")
    except (OSError, AttributeError):
        # Not Windows, or a machine with no Bluetooth stack installed.
        return None
    try:
        dll.BluetoothFindFirstRadio.argtypes = [
            ctypes.POINTER(_FindRadioParams), ctypes.POINTER(wintypes.HANDLE)
        ]
        dll.BluetoothFindFirstRadio.restype = wintypes.HANDLE
        dll.BluetoothFindNextRadio.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(wintypes.HANDLE)
        ]
        dll.BluetoothFindNextRadio.restype = wintypes.BOOL
        dll.BluetoothFindRadioClose.argtypes = [wintypes.HANDLE]
        dll.BluetoothFindRadioClose.restype = wintypes.BOOL

        dll.BluetoothGetRadioInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(_RadioInfo)
        ]
        dll.BluetoothGetRadioInfo.restype = wintypes.DWORD

        dll.BluetoothFindFirstDevice.argtypes = [
            ctypes.POINTER(_DeviceSearchParams), ctypes.POINTER(_DeviceInfo)
        ]
        dll.BluetoothFindFirstDevice.restype = wintypes.HANDLE
        dll.BluetoothFindNextDevice.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(_DeviceInfo)
        ]
        dll.BluetoothFindNextDevice.restype = wintypes.BOOL
        dll.BluetoothFindDeviceClose.argtypes = [wintypes.HANDLE]
        dll.BluetoothFindDeviceClose.restype = wintypes.BOOL

        dll.BluetoothRemoveDevice.argtypes = [
            ctypes.POINTER(_BluetoothAddress)
        ]
        dll.BluetoothRemoveDevice.restype = wintypes.DWORD

        dll.BluetoothSetServiceState.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(_DeviceInfo),
            ctypes.POINTER(_Guid), wintypes.DWORD,
        ]
        dll.BluetoothSetServiceState.restype = wintypes.DWORD

        dll.BluetoothAuthenticateDevice.argtypes = [
            wintypes.HWND, wintypes.HANDLE, ctypes.POINTER(_DeviceInfo),
            _PWCHAR, wintypes.ULONG,
        ]
        dll.BluetoothAuthenticateDevice.restype = wintypes.DWORD

        dll.BluetoothEnumerateInstalledServices.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(_DeviceInfo),
            ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(_Guid),
        ]
        dll.BluetoothEnumerateInstalledServices.restype = wintypes.DWORD
    except AttributeError:
        return None
    return dll


_bt = _load_bluetooth_api()

try:
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.CloseHandle.restype = wintypes.BOOL
except (OSError, AttributeError):
    _kernel32 = None


def available():
    """Is the Windows Bluetooth API usable on this machine?"""
    return _bt is not None and _kernel32 is not None


# ---- enumeration (Dolphin's EnumerateRadios / EnumerateBluetoothDevices) ----

def _enumerate_radios(visit):
    params = _FindRadioParams(ctypes.sizeof(_FindRadioParams))
    radio = wintypes.HANDLE()
    find = _bt.BluetoothFindFirstRadio(ctypes.byref(params),
                                       ctypes.byref(radio))
    if not find:
        return
    try:
        while True:
            try:
                visit(radio)
            finally:
                _kernel32.CloseHandle(radio)
            if not _bt.BluetoothFindNextRadio(find, ctypes.byref(radio)):
                break
    finally:
        _bt.BluetoothFindRadioClose(find)


def _enumerate_devices(inquiry_length, visit):
    """Walk every Bluetooth device on every radio.

    An inquiry_length of 0 lists only devices Windows already knows
    about; anything higher makes the radio actively search the air for
    that many 1.28-second slices.
    """
    def per_radio(radio):
        radio_info = _RadioInfo()
        radio_info.dwSize = ctypes.sizeof(_RadioInfo)
        if _bt.BluetoothGetRadioInfo(radio,
                                     ctypes.byref(radio_info)) != ERROR_SUCCESS:
            return
        params = _DeviceSearchParams(
            dwSize=ctypes.sizeof(_DeviceSearchParams),
            fReturnAuthenticated=True,
            fReturnRemembered=True,
            fReturnUnknown=True,
            fReturnConnected=True,
            fIssueInquiry=inquiry_length > 0,
            cTimeoutMultiplier=inquiry_length,
            hRadio=radio,
        )
        btdi = _DeviceInfo()
        btdi.dwSize = ctypes.sizeof(_DeviceInfo)
        find = _bt.BluetoothFindFirstDevice(ctypes.byref(params),
                                            ctypes.byref(btdi))
        if not find:
            return  # ERROR_NO_MORE_ITEMS: this radio sees nothing
        try:
            while True:
                visit(radio, radio_info, btdi)
                if not _bt.BluetoothFindNextDevice(find, ctypes.byref(btdi)):
                    break
        finally:
            _bt.BluetoothFindDeviceClose(find)

    _enumerate_radios(per_radio)


def _enumerate_wiimotes(inquiry_length, visit):
    def per_device(radio, radio_info, btdi):
        if btdi.szName.startswith(WIIMOTE_NAME_PREFIX):
            visit(radio, radio_info, btdi)

    _enumerate_devices(inquiry_length, per_device)


# ---- pairing ----

def _authenticate(radio, radio_info, btdi, auth_method):
    """Pair one remote, using an address as the legacy pairing pass key."""
    address = (radio_info.address if auth_method == AUTH_SYNC_BUTTON
               else btdi.Address)
    # The pass key is the six raw address bytes, in stored order (the
    # reverse of how an address is displayed), one byte per WCHAR.
    pass_key = (ctypes.c_wchar * 6)()
    for i, byte in enumerate(address.rgBytes):
        pass_key[i] = chr(byte)
    result = _bt.BluetoothAuthenticateDevice(
        None, radio, ctypes.byref(btdi),
        ctypes.cast(pass_key, _PWCHAR), len(pass_key),
    )
    if result != ERROR_SUCCESS:
        # Usually ERROR_NO_MORE_ITEMS or ERROR_GEN_FAILURE — the remote
        # stopped advertising, or wanted the other pass key.
        return False
    # Enumerating the services afterwards is what makes the remote
    # remember the pairing (Dolphin does the same and ignores the list).
    count = wintypes.DWORD(0)
    result = _bt.BluetoothEnumerateInstalledServices(
        radio, ctypes.byref(btdi), ctypes.byref(count), None
    )
    return result in (ERROR_SUCCESS, ERROR_MORE_DATA)


def discover_and_pair(inquiry_length, auth_method=None):
    """Enable the HID service on every remote found, pairing if asked.

    Returns how many remotes were switched on.  Without an auth_method
    this only revives remotes Windows already knows — the cheap sweep
    continuous scanning repeats.
    """
    count = 0

    def visit(radio, radio_info, btdi):
        nonlocal count
        if btdi.fConnected:
            return
        if not btdi.fAuthenticated and auth_method is not None:
            _authenticate(radio, radio_info, btdi, auth_method)
        result = _bt.BluetoothSetServiceState(
            radio, ctypes.byref(btdi), ctypes.byref(HID_SERVICE_UUID),
            BLUETOOTH_SERVICE_ENABLE,
        )
        if result == ERROR_SUCCESS:
            count += 1

    _enumerate_wiimotes(inquiry_length, visit)
    return count


def remove_unusable():
    """Delete remembered remotes Windows can no longer reconnect.

    Windows hangs on to a device entry after a remote powers off.  A
    remembered *and* authenticated one reconnects on any button press,
    but a remembered unauthenticated one is stuck: it blocks the address
    without ever coming back.  Dropping it lets the next inquiry pair the
    remote afresh.
    """
    count = 0

    def visit(radio, radio_info, btdi):
        nonlocal count
        if btdi.fRemembered and not btdi.fConnected and not btdi.fAuthenticated:
            if _bt.BluetoothRemoveDevice(
                    ctypes.byref(btdi.Address)) == ERROR_SUCCESS:
                count += 1

    _enumerate_wiimotes(0, visit)
    return count


# Two inquiries running on one radio at the same time just get in each
# other's way, so scan() and pair() take turns.  Dolphin sidesteps this
# by keeping all of it on its single scanning thread.
_radio_lock = threading.Lock()


def scan():
    """One scanning sweep — Dolphin's WiimoteScannerWindows::FindNewWiimotes.

    Cheap enough to repeat: it pairs nothing, it just clears out dead
    entries and switches the HID service back on for remotes the host
    already knows.  Takes roughly four seconds (the inquiry), and does
    nothing at all if the radio is already busy with a pair().
    """
    if not available() or not _radio_lock.acquire(blocking=False):
        return 0
    try:
        remove_unusable()
        return discover_and_pair(DEFAULT_INQUIRY_LENGTH)
    finally:
        _radio_lock.release()


def pair(auth_method=AUTH_SYNC_BUTTON):
    """Pair remotes in discovery mode — Dolphin's FindAndAuthenticateWiimotes.

    Needed once per remote: hold 1+2 (or press sync under the battery
    cover) so it advertises, then call this.  Runs the inquiry several
    times because a single pass frequently misses.  Takes on the order of
    ten seconds.
    """
    if not available():
        return 0
    with _radio_lock:
        remove_unusable()
        return sum(
            discover_and_pair(DEFAULT_INQUIRY_LENGTH, auth_method)
            for _ in range(PAIR_ITERATIONS)
        )
