# Wii Remote Mapper

Maps the buttons of a Bluetooth-paired Wii remote (Nintendo RVL-CNT-01) to
keyboard keys on Windows. Built to control Anki, works with any app.

## Setup

```
py -m pip install -r requirements.txt
```

## Run

Make sure the Wii remote is connected via Bluetooth (press a button on it to
wake it up), then:

```
py app.py
```

The remote's player-1 LED lights up when the app connects.

## Usage

- Each row shows a Wiimote button, a green indicator that lights while the
  button is held, and the keyboard key it sends.
- The last four rows are **motion swings** — swing the remote up, down, left,
  or right (like grading an Anki card with a flick). A swing taps its key once;
  its indicator flashes green. Forward/back jabs are ignored.
- Click **Set** next to a button or swing, then press the keyboard key you want
  to assign. Combinations work too: hold the modifiers and press the final key
  (e.g. Ctrl+Z is stored as `ctrl+z`). Releasing a modifier without pressing
  another key assigns the modifier by itself. Click **Cancel** to abort.
- Mappings are saved automatically to `mappings.json`.
- If the remote powers off (it sleeps when idle), press any button on it and
  click **Reconnect**.
- **Motion Viewer** opens a window with a live 3D model of the remote and a
  rolling graph of swing acceleration. Pitch and roll come from the
  accelerometer. Yaw (turning around the vertical axis) uses the MotionPlus
  gyroscope, which the app activates automatically when present (built into
  `RVL-CNT-01-TR` remotes; a plug-in dongle on older ones) — gyro headings
  drift slowly, so click the 3D model to re-center. Without a MotionPlus, yaw
  is physically invisible to the accelerometer and is shown as unavailable.
  The graph plots the deviation from gravity in g with
  the swing-trigger threshold marked — useful for Wii Sports golf: games fire
  the stroke on an acceleration spike, so if the trace crosses the line during
  your backswing (jerky start or abrupt stop), that's why the character swings
  early. Practice keeping the backswing under the line.
- **Tennis Practice** opens a Wii Sports-style wall-rally game. Any swing
  serves the ball at the practice wall; it bounces back (with a ground
  bounce) and must be returned as it reaches the highlighted zone at the
  bottom of the court. The game uses its own stroke recognition, not the
  four directional flicks: it integrates the whole sweeping arc of a
  racket stroke, so the ball goes wherever you swing — swinging across
  your body angles the shot to that side, a clearly upward stroke is a
  lob, a clearly downward one a smash — and the speed of the remote
  during the stroke sets the ball's pace, so faster swings come back
  faster. Every return also speeds the rally up, so long rallies train
  exactly the timing Wii tennis wants. An on-court player runs to meet
  the return, and their racket mirrors the remote live — it tilts with
  roll, rises with pitch, sways as your hand moves, and sweeps through
  a full stroke when a swing is detected — so you can see exactly what
  the game sees. To help with timing, a ring collapses onto the racket
  as the ball comes back: orange means the ball is in reach, green is
  the sweet spot — swing as the green ring meets the racket. While the
  game window is open, swings play strokes instead of typing their
  mapped keys (buttons still work normally).
- **Swing Recorder** captures labeled swing samples for building better
  stroke recognition. Pick the stroke class (topspin, backspin, no-spin,
  smash, dropshot, or lob) and the placement (forehand or backhand, aimed
  cross, straight, or center), then hold the remote's **A** button while
  you swing: recording starts when A goes down and is saved when you
  release it, with a short tail so the end-of-stroke deceleration is
  kept. Accel and MotionPlus gyro (when present) are recorded and
  appended to `recordings/swings.jsonl`; record as many samples per
  stroke as you like. Stroke labels show their total sample count;
  placement labels show the count for the selected stroke. **Discard
  last** removes a bad capture. While the recorder is open, A doesn't
  type its mapped key and swings don't type theirs.
- Motion setup follows Dolphin's real-Wiimote handshake: the reporting mode
  is sent with an acknowledgement flag and retried until the remote confirms
  it, the accelerometer is calibrated from the remote's own stored zero/+1g
  values, and reporting is automatically restored after events (like plugging
  or unplugging a Nunchuk) that silently stop it.

## Default mappings (Anki-friendly)

| Wiimote | Key | Anki action |
|---------|-----|-------------|
| A | space | Show answer / Good |
| B | 1 | Again |
| 1 | 2 | Hard |
| 2 | 3 | Good |
| + | enter | Show answer / Good |
| − | backspace | |
| Home | esc | Back to deck list |
| D-pad | arrow keys | |
| Swing up | 1 | Again |
| Swing down | 2 | Hard |
| Swing left | 3 | Good |
| Swing right | 4 | Easy |
