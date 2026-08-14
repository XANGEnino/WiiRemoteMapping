"""Load/save button-to-key mappings and app settings as JSON next to the app."""

import json
from pathlib import Path

MAPPINGS_FILE = Path(__file__).resolve().parent / "mappings.json"
SETTINGS_FILE = Path(__file__).resolve().parent / "settings.json"

# Anki-friendly defaults: space = show answer / Good, 1-4 = answer buttons
DEFAULT_MAPPINGS = {
    "A": "space",
    "B": "1",
    "One": "2",
    "Two": "3",
    "Plus": "enter",
    "Minus": "backspace",
    "Home": "esc",
    "Up": "up",
    "Down": "down",
    "Left": "left",
    "Right": "right",
    # Motion swings — tapped once per swing. Anki grade defaults.
    "SwingUp": "1",     # Again
    "SwingDown": "2",   # Hard
    "SwingLeft": "3",   # Good / normal
    "SwingRight": "4",  # Easy
}


def load_mappings():
    mappings = dict(DEFAULT_MAPPINGS)
    try:
        saved = json.loads(MAPPINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return mappings
    if isinstance(saved, dict):
        for button, key in saved.items():
            if button in mappings and isinstance(key, str):
                mappings[button] = key
    return mappings


def save_mappings(mappings):
    MAPPINGS_FILE.write_text(
        json.dumps(mappings, indent=2), encoding="utf-8"
    )


DEFAULT_SETTINGS = {
    # Off by default: an ordinary hand movement is easily mistaken for a
    # swing, so the swing bindings only fire once switched on.
    "motion_enabled": False,
    # On by default: without it a remote only connects if something else
    # (Dolphin, say) is sweeping the Bluetooth radio for it.
    "continuous_scanning": True,
}


def load_settings():
    settings = dict(DEFAULT_SETTINGS)
    try:
        saved = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return settings
    if isinstance(saved, dict):
        for name, value in saved.items():
            if name in settings and isinstance(value, type(settings[name])):
                settings[name] = value
    return settings


def save_settings(settings):
    SETTINGS_FILE.write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )
