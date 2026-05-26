from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import TextBox
from PIL import Image

SONY_IMAGE_PATH = Path(__file__).resolve().parent / "LYR-Sony-151221_090202.jpg"
SONY_GRAYSCALE_PATH = SONY_IMAGE_PATH.with_name(
    f"{SONY_IMAGE_PATH.stem}_grayscale{SONY_IMAGE_PATH.suffix}"
)


def load_sony_image() -> np.ndarray:
    """Load the Sony JPEG as an RGB uint8 array with shape (height, width, 3)."""
    with Image.open(SONY_IMAGE_PATH) as img:
        return np.asarray(img.convert("RGB"))


def load_sony_grayscale() -> np.ndarray:
    """Load the Sony JPEG as a grayscale uint8 array with shape (height, width)."""
    with Image.open(SONY_IMAGE_PATH) as img:
        return np.asarray(img.convert("L"))


def save_sony_grayscale(path: Path = SONY_GRAYSCALE_PATH) -> Path:
    """Convert the Sony JPEG to grayscale and save it."""
    with Image.open(SONY_IMAGE_PATH) as img:
        img.convert("L").save(path)
    return path


def plot_vertical_slice_overlay(
    image: np.ndarray,
    x: int,
    y: tuple[int, int],
    *,
    profile_width_fraction: float = 0.12,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Show grayscale image with profile[:, x] over rows y[0]:y[1] (end exclusive)."""
    y0, y1 = y
    profile = image[:, x]
    rows = np.arange(image.shape[0])
    profile_width = profile_width_fraction * image.shape[1]

    profile_slice = profile[y0:y1]
    rows_slice = rows[y0:y1]

    p = profile_slice.astype(np.float64)
    p = (p - p.min()) / (p.max() - p.min() + 1e-9)
    x_curve = x + profile_width * p

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(image, cmap="gray", origin="upper")
    ax.axvline(x, color="cyan", linewidth=1, alpha=0.9, label=f"column x={x}")
    ax.axhline(y0, color="magenta", linewidth=1, alpha=0.7, linestyle="--")
    ax.axhline(y1 - 1, color="magenta", linewidth=1, alpha=0.7, linestyle="--")
    ax.plot(
        x_curve,
        rows_slice,
        color="red",
        linewidth=1.2,
        label=f"profile, rows {y0}:{y1}",
    )
    ax.set_xlim(-0.5, image.shape[1] - 0.5)
    ax.set_ylim(image.shape[0] - 0.5, -0.5)
    ax.set_title(f"Vertical slice at column {x}, rows {y0}:{y1}")
    ax.legend(loc="upper right")
    return ax


def plot_vertical_slice_interactive(
    image: np.ndarray,
    x: int | None = None,
    y: tuple[int, int] | None = None,
    *,
    profile_width_fraction: float = 0.12,
) -> tuple[plt.Figure, tuple[TextBox, TextBox, TextBox]]:
    """Interactive overlay: text boxes accept column x and row range (y0, y1)."""
    height, width = image.shape[:2]
    if x is None:
        x = width // 2
    if y is None:
        y = (height // 4, 3 * height // 4)
    y0, y1 = y

    profile_width = profile_width_fraction * width

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.subplots_adjust(bottom=0.18)

    ax.imshow(image, cmap="gray", origin="upper")
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)

    vline = ax.axvline(x, color="cyan", linewidth=1, alpha=0.9)
    hline_top = ax.axhline(y0, color="magenta", linewidth=1, alpha=0.7, linestyle="--")
    hline_bot = ax.axhline(y1 - 1, color="magenta", linewidth=1, alpha=0.7, linestyle="--")
    (curve,) = ax.plot([], [], color="red", linewidth=1)
    (peak_marker,) = ax.plot(
        [], [], color="lime", marker="+", markersize=14, markeredgewidth=2, linestyle="none"
    )

    ax_x = fig.add_axes((0.15, 0.06, 0.12, 0.05))
    ax_y0 = fig.add_axes((0.45, 0.06, 0.12, 0.05))
    ax_y1 = fig.add_axes((0.75, 0.06, 0.12, 0.05))
    tb_x = TextBox(ax_x, "x ", initial=str(int(x)))
    tb_y0 = TextBox(ax_y0, "y0 ", initial=str(int(y0)))
    tb_y1 = TextBox(ax_y1, "y1 ", initial=str(int(y1)))

    state = {"x": int(x), "y0": int(y0), "y1": int(y1)}

    def update(xi: int, y0i: int, y1i: int) -> None:
        profile = image[y0i:y1i, xi].astype(np.float64)
        rows_slice = np.arange(y0i, y1i)
        p = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)
        x_curve = xi + profile_width * p

        peak_idx = int(np.argmax(profile))
        y_peak = y0i + peak_idx
        peak_value = profile[peak_idx]

        vline.set_xdata([xi, xi])
        hline_top.set_ydata([y0i, y0i])
        hline_bot.set_ydata([y1i - 1, y1i - 1])
        curve.set_data(x_curve, rows_slice)
        peak_marker.set_data([xi], [y_peak])
        ax.set_title(
            f"Vertical slice at column {xi}, rows {y0i}:{y1i}\n"
            f"Brightest pixel: (x={xi}, y={y_peak}), value={peak_value:.0f}"
        )
        fig.canvas.draw_idle()

    def parse_and_apply(_text: str | None = None) -> None:
        try:
            xi = int(tb_x.text)
            y0i = int(tb_y0.text)
            y1i = int(tb_y1.text)
        except ValueError:
            ax.set_title("invalid input: x, y0, y1 must be integers")
            fig.canvas.draw_idle()
            return

        xi = max(0, min(width - 1, xi))
        y0i = max(0, min(height - 1, y0i))
        y1i = max(y0i + 1, min(height, y1i))

        state["x"], state["y0"], state["y1"] = xi, y0i, y1i
        if tb_x.text != str(xi):
            tb_x.set_val(str(xi))
        if tb_y0.text != str(y0i):
            tb_y0.set_val(str(y0i))
        if tb_y1.text != str(y1i):
            tb_y1.set_val(str(y1i))

        update(xi, y0i, y1i)

    tb_x.on_submit(parse_and_apply)
    tb_y0.on_submit(parse_and_apply)
    tb_y1.on_submit(parse_and_apply)

    fig._textboxes = (tb_x, tb_y0, tb_y1)

    update(state["x"], state["y0"], state["y1"])
    return fig, (tb_x, tb_y0, tb_y1)


sony_image = load_sony_image()
sony_grayscale = load_sony_grayscale()

if __name__ == "__main__":
    # save_sony_grayscale()
    x = sony_grayscale.shape[1] // 2 + 100
    y = (630, 1000)
    profile = sony_grayscale[:, x]
    print(f"Loaded {SONY_IMAGE_PATH.name}: shape={sony_image.shape}, dtype={sony_image.dtype}")

    print(f"Vertical slice at x={x}: profile shape={profile.shape}")

    plot_vertical_slice_interactive(sony_grayscale, x=x, y=y)
    plt.show()
