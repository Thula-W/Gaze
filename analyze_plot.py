import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import os

def load_data(path):
    df = pd.read_csv(path)
    return df


def smooth(series, window=5):
    """Simple moving-average low-pass filter."""
    return uniform_filter1d(series, size=window)


def plot(fileName):
    df = load_data(f'logs/{fileName}.csv')

    df["avg_ear_smooth"] = smooth(df["avg_ear"].values, window=5)
    df["gaze_avg"] = (df["gaze_ratio_left"] + df["gaze_ratio_right"]) / 2
    df["gaze_avg_smooth"] = smooth(df["gaze_avg"].values, window=5)

    total_blinks = int(df["blink"].sum())
    duration = df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]
    blink_rate_per_min = total_blinks / (duration / 60) if duration > 0 else 0

    # crude "gaze stability" metric: lower std = more fixated/stable
    gaze_stability = df["gaze_avg_smooth"].std()

    print("---- Summary ----")
    print(f"Duration: {duration:.1f}s")
    print(f"Total blinks: {total_blinks}")
    print(f"Blink rate: {blink_rate_per_min:.1f} per minute")
    print(f"Gaze stability (std dev, lower = steadier): {gaze_stability:.4f}")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(df["timestamp"], df["avg_ear"], alpha=0.3, label="raw EAR")
    axes[0].plot(df["timestamp"], df["avg_ear_smooth"], label="smoothed EAR")
    axes[0].axhline(0.21, color="red", linestyle="--", label="blink threshold")
    axes[0].set_ylabel("EAR")
    axes[0].set_title("Eye Aspect Ratio over time (blink detection)")
    axes[0].legend()

    axes[1].plot(df["timestamp"], df["gaze_avg"], alpha=0.3, label="raw gaze ratio")
    axes[1].plot(df["timestamp"], df["gaze_avg_smooth"], label="smoothed gaze ratio")
    axes[1].set_ylabel("Gaze ratio (0=inner, 1=outer)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("Horizontal gaze position over time")
    axes[1].legend()

    os.makedirs("plots", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"plots/{fileName}_analysis.png", dpi=150)
    print(f"Saved plot to {fileName}_analysis.png")
    plt.show()

