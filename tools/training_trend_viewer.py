from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from tkinter import filedialog, messagebox

import tkinter as tk

from PIL import Image, ImageDraw, ImageFont, ImageTk

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    DND_AVAILABLE = True
except ImportError:
    DND_FILES = None
    TkinterDnD = None
    DND_AVAILABLE = False


WIDTH = 1500
HEIGHT = 860
PADDING = 70
PANEL_GAP = 110
MOVING_AVERAGE_WINDOW = 20
FRAME_THRESHOLDS = (600, 1200, 1500, 1800, 2000)
SCATTER_BINS = 18


# Convert a value to float and keep bad values harmless.
def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# Convert a value to int and keep bad values harmless.
def safe_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


# Read one PPO or DQN training CSV file.
def read_training_log(path: Path) -> list[dict[str, float]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for raw_row in reader:
            rows.append(
                {
                    "episode": safe_int(raw_row.get("episode")),
                    "global_step": safe_float(raw_row.get("global_step")),
                    "total_frame_steps": safe_float(raw_row.get("total_frame_steps")),
                    "frame_steps": safe_float(raw_row.get("frame_steps")),
                    "decision_steps": safe_float(raw_row.get("decision_steps")),
                    "episode_reward": safe_float(raw_row.get("episode_reward")),
                    "policy_loss": safe_float(raw_row.get("policy_loss")),
                    "value_loss": safe_float(raw_row.get("value_loss")),
                    "entropy": safe_float(raw_row.get("entropy")),
                    "approx_kl": safe_float(raw_row.get("approx_kl")),
                    "hp": safe_float(raw_row.get("hp")),
                    "collisions": safe_float(raw_row.get("collisions")),
                }
            )
    return [row for row in rows if row["episode"] > 0]


# Compute a trailing moving average.
def moving_average(values: list[float], window: int) -> list[float]:
    if not values:
        return []

    smoothed = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        current_window = min(index + 1, window)
        smoothed.append(running_sum / current_window)
    return smoothed


# Return a compact summary for one group of rows.
def summarize_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}

    frames = [row["frame_steps"] for row in rows]
    rewards = [row["episode_reward"] for row in rows]
    summary = {
        "n": len(rows),
        "mean_frame": sum(frames) / len(frames),
        "max_frame": max(frames),
        "min_frame": min(frames),
        "mean_reward": sum(rewards) / len(rewards),
    }
    for threshold in FRAME_THRESHOLDS:
        summary[f"hit_{threshold}"] = sum(frame >= threshold for frame in frames)
    return summary


# Build rolling window summaries for the text panel.
def build_summary_text(rows: list[dict[str, float]]) -> list[str]:
    total = summarize_rows(rows)
    recent_50 = summarize_rows(rows[-50:])
    recent_20 = summarize_rows(rows[-20:])
    last_episode = int(rows[-1]["episode"]) if rows else 0

    lines = [f"episodes: {len(rows)}    last episode: {last_episode}"]
    for name, summary in (("all", total), ("last 50", recent_50), ("last 20", recent_20)):
        if not summary:
            continue
        lines.append(
            f"{name:<7} mean={summary['mean_frame']:.1f}  max={summary['max_frame']:.0f}  "
            f"min={summary['min_frame']:.0f}  reward={summary['mean_reward']:.3f}"
        )
        lines.append(
            f"{'':<7} >=1200:{summary['hit_1200']:.0f}  >=1500:{summary['hit_1500']:.0f}  "
            f">=1800:{summary['hit_1800']:.0f}  >=2000:{summary['hit_2000']:.0f}"
        )
    return lines


# Find a readable font on Windows or fall back to Pillow default.
def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


# Scale data coordinates into the chart rectangle.
def scale_points(
    xs: list[float],
    ys: list[float],
    rect: tuple[int, int, int, int],
    y_min: float,
    y_max: float,
    x_min: float | None = None,
    x_max: float | None = None,
) -> list[tuple[int, int]]:
    left, top, right, bottom = rect
    data_x_min = min(xs) if x_min is None else x_min
    data_x_max = max(xs) if x_max is None else x_max
    if math.isclose(data_x_min, data_x_max):
        data_x_max = data_x_min + 1.0
    if math.isclose(y_min, y_max):
        y_max = y_min + 1.0

    points = []
    for x, y in zip(xs, ys):
        px = left + int((x - data_x_min) / (data_x_max - data_x_min) * (right - left))
        py = bottom - int((y - y_min) / (y_max - y_min) * (bottom - top))
        points.append((px, py))
    return points


# Return the data range with a little padding.
def padded_range(values: list[float], fixed_min: float | None = None, fixed_max: float | None = None) -> tuple[float, float]:
    data_min = min(values) if fixed_min is None else fixed_min
    data_max = max(values) if fixed_max is None else fixed_max
    value_range = data_max - data_min
    padding = max(value_range * 0.08, abs(data_max) * 0.08, abs(data_min) * 0.08, 0.001)
    if fixed_min is None:
        data_min -= padding
    if fixed_max is None:
        data_max += padding
    if math.isclose(data_min, data_max):
        data_max = data_min + 1.0
    return data_min, data_max


# Draw grid lines and axis labels for one chart.
def draw_axes(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    title: str,
    y_min: float,
    y_max: float,
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> None:
    left, top, right, bottom = rect
    draw.rectangle(rect, outline=(40, 40, 40), width=2)
    draw.text((left, top - 30), title, fill=(20, 20, 20), font=font)
    value_span = max(abs(y_min), abs(y_max), abs(y_max - y_min))

    for index in range(5):
        ratio = index / 4.0
        y = top + int(ratio * (bottom - top))
        value = y_max - ratio * (y_max - y_min)
        draw.line((left, y, right, y), fill=(225, 225, 225), width=1)
        if value_span < 0.01:
            label = f"{value:.4f}"
        elif value_span < 1.0:
            label = f"{value:.3f}"
        else:
            label = f"{value:.1f}"
        draw.text((left - 64, y - 9), label, fill=(90, 90, 90), font=small_font)


# Draw x-axis labels for one chart.
def draw_x_labels(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    x_min: float,
    x_max: float,
    small_font: ImageFont.ImageFont,
    tick_count: int = 5,
) -> None:
    left, top, right, bottom = rect
    tick_count = max(2, tick_count)
    for index in range(tick_count):
        ratio = index / (tick_count - 1)
        x = left + int(ratio * (right - left))
        value = x_min + ratio * (x_max - x_min)
        draw.line((x, bottom, x, bottom + 5), fill=(90, 90, 90), width=1)
        if abs(value) >= 10000:
            label = f"{value / 1000:.0f}k"
        else:
            label = f"{value:.0f}"
        draw.text((x - 24, bottom + 8), label, fill=(90, 90, 90), font=small_font)


# Draw one line chart with raw and smoothed values.
def draw_line_chart(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    episodes: list[float],
    values: list[float],
    title: str,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    if not values:
        return

    smooth_values = moving_average(values, MOVING_AVERAGE_WINDOW)
    data_min, data_max = padded_range(values + smooth_values, y_min, y_max)

    draw_axes(draw, rect, title, data_min, data_max, font, small_font)
    draw_x_labels(draw, rect, min(episodes), max(episodes), small_font)
    raw_points = scale_points(episodes, values, rect, data_min, data_max)
    smooth_points = scale_points(episodes, smooth_values, rect, data_min, data_max)

    if len(raw_points) > 1:
        draw.line(raw_points, fill=(195, 205, 215), width=2)
    if len(smooth_points) > 1:
        draw.line(smooth_points, fill=color, width=5)

    left, top, right, bottom = rect
    draw.text((right - 210, top + 8), f"MA{MOVING_AVERAGE_WINDOW}", fill=color, font=small_font)


# Draw threshold lines on the frame chart.
def draw_frame_thresholds(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    y_min: float,
    y_max: float,
    small_font: ImageFont.ImageFont,
) -> None:
    left, top, right, bottom = rect
    for threshold in FRAME_THRESHOLDS[1:]:
        if not (y_min <= threshold <= y_max):
            continue
        y = bottom - int((threshold - y_min) / (y_max - y_min) * (bottom - top))
        draw.line((left, y, right, y), fill=(230, 170, 170), width=2)
        draw.text((right - 76, y - 18), str(threshold), fill=(150, 70, 70), font=small_font)


# Draw a frame-step chart with useful survival thresholds.
def draw_frame_chart(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    episodes: list[float],
    frames: list[float],
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> None:
    y_max = max(max(frames), max(FRAME_THRESHOLDS)) + 120.0
    y_min = 0.0
    draw_line_chart(draw, rect, episodes, frames, "frame_steps / episode", (35, 110, 210), font, small_font, y_min, y_max)
    draw_frame_thresholds(draw, rect, y_min - max(1.0, y_max * 0.08), y_max + max(1.0, y_max * 0.08), small_font)


# Build average reward points for total-step bins.
def binned_reward_curve(total_steps: list[float], rewards: list[float], bins: int) -> tuple[list[float], list[float]]:
    if not total_steps or not rewards:
        return [], []

    step_min = min(total_steps)
    step_max = max(total_steps)
    if math.isclose(step_min, step_max):
        return [step_min], [sum(rewards) / len(rewards)]

    bucket_sums = [0.0 for _ in range(bins)]
    bucket_counts = [0 for _ in range(bins)]
    for step, reward in zip(total_steps, rewards):
        bucket = int((step - step_min) / (step_max - step_min) * bins)
        bucket = min(bins - 1, max(0, bucket))
        bucket_sums[bucket] += reward
        bucket_counts[bucket] += 1

    xs = []
    ys = []
    bucket_width = (step_max - step_min) / bins
    for index, count in enumerate(bucket_counts):
        if count <= 0:
            continue
        xs.append(step_min + (index + 0.5) * bucket_width)
        ys.append(bucket_sums[index] / count)
    return xs, ys


# Draw reward as a function of total training steps.
def draw_reward_step_chart(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    total_steps: list[float],
    rewards: list[float],
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> None:
    if not total_steps or not rewards:
        return

    x_min = min(total_steps)
    x_max = max(total_steps)
    if math.isclose(x_min, x_max):
        x_max = x_min + 1.0
    y_min, y_max = padded_range(rewards)
    draw_axes(draw, rect, "reward / total_steps", y_min, y_max, font, small_font)
    draw_x_labels(draw, rect, x_min, x_max, small_font, tick_count=9)

    points = scale_points(total_steps, rewards, rect, y_min, y_max, x_min, x_max)
    for x, y in points:
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(190, 205, 220))

    curve_xs, curve_ys = binned_reward_curve(total_steps, rewards, SCATTER_BINS)
    curve_points = scale_points(curve_xs, curve_ys, rect, y_min, y_max, x_min, x_max) if curve_xs else []
    if len(curve_points) > 1:
        draw.line(curve_points, fill=(210, 65, 45), width=5)
    for x, y in curve_points:
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(210, 65, 45))


# Draw a small text report inside the trend image.
def draw_summary_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    lines: list[str],
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> None:
    left, top, right, bottom = rect
    draw.rectangle(rect, fill=(247, 248, 250), outline=(60, 60, 60), width=2)
    draw.text((left + 18, top + 14), "summary", fill=(20, 20, 20), font=font)
    y = top + 52
    for line in lines:
        draw.text((left + 18, y), line, fill=(45, 45, 45), font=small_font)
        y += 26


# Create and save one training trend image.
def draw_training_trend(csv_path: Path, rows: list[dict[str, float]]) -> Path:
    output_dir = csv_path.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{csv_path.stem}_trend.png"

    image = Image.new("RGB", (WIDTH, HEIGHT), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    title_font = load_font(30, bold=True)
    chart_font = load_font(22, bold=True)
    small_font = load_font(18)

    episodes = [row["episode"] for row in rows]
    total_steps = [row["total_frame_steps"] for row in rows]
    if max(total_steps) <= 0.0:
        running_steps = 0.0
        total_steps = []
        for row in rows:
            running_steps += row["frame_steps"]
            total_steps.append(running_steps)
    frames = [row["frame_steps"] for row in rows]
    rewards = [row["episode_reward"] for row in rows]

    draw.text((PADDING, 24), f"Training trend: {csv_path.name}", fill=(15, 15, 15), font=title_font)

    panel_width = (WIDTH - PADDING * 2 - PANEL_GAP) // 2
    panel_height = 275
    left_x = PADDING
    right_x = PADDING + panel_width + PANEL_GAP
    row_1_y = 105
    row_2_y = row_1_y + panel_height + 90
    wide_panel_width = WIDTH - PADDING * 2

    draw_frame_chart(draw, (left_x, row_1_y, left_x + panel_width, row_1_y + panel_height), episodes, frames, chart_font, small_font)
    draw_line_chart(
        draw,
        (right_x, row_1_y, right_x + panel_width, row_1_y + panel_height),
        episodes,
        rewards,
        "reward / episode",
        (35, 140, 80),
        chart_font,
        small_font,
    )
    draw_reward_step_chart(
        draw,
        (left_x, row_2_y, left_x + wide_panel_width, row_2_y + panel_height),
        total_steps,
        rewards,
        chart_font,
        small_font,
    )

    image.save(output_path)
    return output_path


# Generate a trend image from one CSV path.
def generate_from_csv(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() != ".csv":
        raise ValueError("Please choose a CSV log file.")

    rows = read_training_log(path)
    if not rows:
        raise ValueError("No valid training rows were found.")
    return draw_training_trend(path, rows)


# Parse file paths dropped onto the Tk window.
def parse_dropped_paths(root: tk.Tk, data: str) -> list[Path]:
    try:
        parts = root.tk.splitlist(data)
    except tk.TclError:
        parts = [data]
    return [Path(part.strip()) for part in parts if part.strip()]


class TrendViewer:
    # Create the GUI window.
    def __init__(self) -> None:
        root_class = TkinterDnD.Tk if DND_AVAILABLE and TkinterDnD is not None else tk.Tk
        self.root = root_class()
        self.root.title("LunAI Training Trend Viewer")
        self.root.geometry("760x560")
        self.preview_image: ImageTk.PhotoImage | None = None

        self.status = tk.StringVar()
        self.status.set("Drop a training CSV here." if DND_AVAILABLE else "Click Open CSV. Drag-drop needs tkinterdnd2.")

        self._build_widgets()
        self._enable_drag_drop()

    # Build labels, buttons, and preview area.
    def _build_widgets(self) -> None:
        self.drop_label = tk.Label(
            self.root,
            text="Drop training CSV here\nor click Open CSV",
            relief="ridge",
            bd=3,
            width=58,
            height=6,
            font=("Segoe UI", 16),
            bg="#f2f5f8",
        )
        self.drop_label.pack(padx=18, pady=18, fill="x")

        button_frame = tk.Frame(self.root)
        button_frame.pack(fill="x", padx=18)

        open_button = tk.Button(button_frame, text="Open CSV", command=self.open_file)
        open_button.pack(side="left")

        self.status_label = tk.Label(self.root, textvariable=self.status, anchor="w", justify="left")
        self.status_label.pack(fill="x", padx=18, pady=10)

        self.preview_label = tk.Label(self.root, bg="#ffffff", relief="sunken", bd=1)
        self.preview_label.pack(padx=18, pady=8, fill="both", expand=True)

    # Enable file drag and drop when tkinterdnd2 is installed.
    def _enable_drag_drop(self) -> None:
        if not DND_AVAILABLE or DND_FILES is None:
            return
        self.drop_label.drop_target_register(DND_FILES)
        self.drop_label.dnd_bind("<<Drop>>", self.on_drop)
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind("<<Drop>>", self.on_drop)

    # Open a CSV file through the normal file picker.
    def open_file(self) -> None:
        filename = filedialog.askopenfilename(
            title="Choose training CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if filename:
            self.process_path(Path(filename))

    # Process files dropped onto the window.
    def on_drop(self, event: tk.Event) -> None:
        paths = parse_dropped_paths(self.root, event.data)
        if paths:
            self.process_path(paths[0])

    # Generate a trend image and update the preview.
    def process_path(self, path: Path) -> None:
        try:
            output_path = generate_from_csv(path)
        except Exception as exc:
            messagebox.showerror("Trend generation failed", str(exc))
            self.status.set(f"Failed: {exc}")
            return

        self.status.set(f"Saved: {output_path}")
        self.show_preview(output_path)

    # Show the generated image in the GUI.
    def show_preview(self, image_path: Path) -> None:
        image = Image.open(image_path)
        image.thumbnail((700, 330))
        self.preview_image = ImageTk.PhotoImage(image)
        self.preview_label.configure(image=self.preview_image)

    # Start the Tk event loop.
    def run(self) -> None:
        self.root.mainloop()


# Start the GUI or generate directly from command line arguments.
def main() -> None:
    if len(sys.argv) > 1:
        for argument in sys.argv[1:]:
            output_path = generate_from_csv(Path(argument))
            print(f"Saved: {output_path}")
        return

    viewer = TrendViewer()
    viewer.run()


if __name__ == "__main__":
    main()
