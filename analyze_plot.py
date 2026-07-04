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

    df["v_gaze_avg"] = (df["vertical_gaze_ratio_left"] + df["vertical_gaze_ratio_right"]) / 2
    df["v_gaze_avg_smooth"] = smooth(df["v_gaze_avg"].values, window=5)

    total_blinks = int(df["blink"].sum())
    duration = df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]
    blink_rate_per_min = total_blinks / (duration / 60) if duration > 0 else 0

    # crude "gaze stability" metric: lower std = more fixated/stable
    gaze_stability = df["gaze_avg_smooth"].std()
    v_gaze_stability = df["v_gaze_avg_smooth"].std()

    print("---- Summary ----")
    print(f"Duration: {duration:.1f}s")
    print(f"Total blinks: {total_blinks}")
    print(f"Blink rate: {blink_rate_per_min:.1f} per minute")
    print(f"Gaze stability (std dev, lower = steadier): {gaze_stability:.4f}")
    print(f"Vertical gaze stability (std dev, lower = steadier): {v_gaze_stability:.4f}")

    has_attention = "attention_score" in df.columns
    num_plots = 4 if has_attention else 3

    # Dynamically scale the figure size based on the number of plots
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 2.5 * num_plots), sharex=True)

    ax_idx = 0

    if has_attention:
        df["attention_score_smooth"] = smooth(df["attention_score"].values, window=5)
        axes[ax_idx].plot(df["timestamp"], df["attention_score"], alpha=0.3, label="raw attention", color="tab:purple")
        axes[ax_idx].plot(df["timestamp"], df["attention_score_smooth"], label="smoothed attention", color="indigo")
        axes[ax_idx].set_ylabel("Attention Score")
        axes[ax_idx].set_title("Attention Score over time")
        axes[ax_idx].legend()
        ax_idx += 1

    axes[ax_idx].plot(df["timestamp"], df["avg_ear"], alpha=0.3, label="raw EAR")
    axes[ax_idx].plot(df["timestamp"], df["avg_ear_smooth"], label="smoothed EAR")
    axes[ax_idx].axhline(0.21, color="red", linestyle="--", label="blink threshold")
    axes[ax_idx].set_ylabel("EAR")
    axes[ax_idx].set_title("Eye Aspect Ratio over time (blink detection)")
    axes[ax_idx].legend()
    ax_idx += 1

    axes[ax_idx].plot(df["timestamp"], df["gaze_avg"], alpha=0.3, label="raw gaze ratio")
    axes[ax_idx].plot(df["timestamp"], df["gaze_avg_smooth"], label="smoothed gaze ratio")
    axes[ax_idx].set_ylabel("Gaze ratio (0=Left, 1=Right)")
    axes[ax_idx].set_title("Horizontal gaze position over time")
    axes[ax_idx].legend()
    ax_idx += 1

    axes[ax_idx].plot(df["timestamp"], df["v_gaze_avg"], alpha=0.3, label="raw vertical gaze")
    axes[ax_idx].plot(df["timestamp"], df["v_gaze_avg_smooth"], label="smoothed vertical", color="tab:green")
    axes[ax_idx].set_ylabel("V-Gaze (0=Up, 1=Down)")
    axes[ax_idx].set_title("Vertical gaze position over time")
    axes[ax_idx].legend()

    axes[ax_idx].set_xlabel("Time (s)")

    os.makedirs("plots", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"plots/{fileName}_analysis.png", dpi=150)
    print(f"Saved plot to {fileName}_analysis.png")
    plt.show(block=False)
    plt.pause(10) 
    plt.close()

