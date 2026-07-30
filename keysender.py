"""Simulated keyboard output via pynput."""

import sys

from pynput.keyboard import Controller, Key

_controller = Controller()

# Names that act as modifiers in a combo like "ctrl+shift+z".
MODIFIERS = ("ctrl", "alt", "shift", "cmd")

# tkinter KeyPress state bits. Alt is 0x20000 on Windows; 0x8 (Mod1)
# elsewhere — 0x8 means NumLock on Windows, so it can't be used there.
_STATE_MASKS = (
    ("ctrl", 0x4),
    ("alt", 0x20000 if sys.platform == "win32" else 0x8),
    ("shift", 0x1),
)


def modifiers_from_state(state):
    """Modifier names currently held, from a tkinter event's state field."""
    return [name for name, mask in _STATE_MASKS if state & mask]

# tkinter keysyms that differ from pynput Key names
KEYSYM_TO_KEY = {
    "Return": "enter",
    "Escape": "esc",
    "BackSpace": "backspace",
    "Prior": "page_up",
    "Next": "page_down",
    "Caps_Lock": "caps_lock",
    "Control_L": "ctrl",
    "Control_R": "ctrl",
    "Shift_L": "shift",
    "Shift_R": "shift",
    "Alt_L": "alt",
    "Alt_R": "alt",
    "Win_L": "cmd",
    "Win_R": "cmd",
}


def keysym_to_name(keysym, char):
    """Convert a tkinter key event to a storable key name, or None."""
    if keysym in KEYSYM_TO_KEY:
        return KEYSYM_TO_KEY[keysym]
    if len(keysym) == 1:
        return keysym.lower()
    lowered = keysym.lower()
    if hasattr(Key, lowered):
        return lowered  # space, left, f1, home, tab, delete, ...
    if char and char.isprintable() and len(char) == 1:
        return char.lower()
    return None


def _resolve(name):
    if len(name) == 1:
        return name
    return getattr(Key, name, None)


def _parse(name):
    """Resolve "ctrl+shift+z" into a list of pynput keys, or None.

    A trailing empty part means the base key is the literal "+"
    (e.g. "ctrl++"), as is a bare "+".
    """
    if name is None:
        return None
    parts = name.split("+")
    if parts[-1] == "":
        parts = [p for p in parts if p] + ["+"]
    keys = [_resolve(p) for p in parts]
    if any(key is None for key in keys):
        return None  # skip the whole combo rather than leave keys stuck
    return keys


def press(name):
    for key in _parse(name) or ():
        _controller.press(key)


def release(name):
    for key in reversed(_parse(name) or ()):
        _controller.release(key)


def tap(name):
    """Press and immediately release — for momentary gestures like swings."""
    keys = _parse(name)
    if keys is None:
        return
    for key in keys:
        _controller.press(key)
    for key in reversed(keys):
        _controller.release(key)
