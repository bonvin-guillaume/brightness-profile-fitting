# brightness_profile_fitting

Inspect vertical brightness profiles of a Sony JPEG.

## Setup

```bash
python3.11 -m venv venv
venv/bin/pip install pillow numpy matplotlib
```

## Run

```bash
venv/bin/python brightness_profile_fitting.py
```

Opens an interactive plot of `LYR-Sony-151221_090202.jpg` in grayscale. Type values in the boxes (Enter to apply):

- `x` &mdash; column index
- `y0`, `y1` &mdash; row range (end exclusive)

The cyan line marks column `x`, the magenta dashed lines mark the row band, the red curve is the brightness profile, and the green `+` marks the brightest pixel in the slice (also shown in the title).
