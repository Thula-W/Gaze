import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import os

# Must match the calibration/distraction split used in attention_experiment.py
CALIBRATION_END_SEC = 10.0
DISTRACTION_END_SEC = 25.0


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

    gaze_stability = df["gaze_avg_smooth"].std()
    v_gaze_stability = df["v_gaze_avg_smooth"].std()

    has_attention = "attention_score" in df.columns
    has_head_pose = "head_yaw_rad" in df.columns and "combined_h" in df.columns

    # ---- split into calibration / distraction phases for phase-based stats ----
    calib_mask = df["timestamp"] < CALIBRATION_END_SEC
    distract_mask = (df["timestamp"] >= CALIBRATION_END_SEC) & (df["timestamp"] < DISTRACTION_END_SEC)

    calib_blinks = int(df.loc[calib_mask, "blink"].sum())
    distract_blinks = int(df.loc[distract_mask, "blink"].sum())
    calib_dur = df.loc[calib_mask, "timestamp"].max() - df.loc[calib_mask, "timestamp"].min() if calib_mask.any() else 0
    distract_dur = df.loc[distract_mask, "timestamp"].max() - df.loc[distract_mask, "timestamp"].min() if distract_mask.any() else 0
    calib_blink_rate = calib_blinks / (calib_dur / 60) if calib_dur > 0 else 0
    distract_blink_rate = distract_blinks / (distract_dur / 60) if distract_dur > 0 else 0

    print("---- Summary ----")
    print(f"Duration: {duration:.1f}s")
    print(f"Total blinks: {total_blinks}")
    print(f"Blink rate: {blink_rate_per_min:.1f} per minute")
    print(f"  Calibration phase blink rate: {calib_blink_rate:.1f}/min")
    print(f"  Distraction phase blink rate: {distract_blink_rate:.1f}/min")
    print(f"Gaze stability (std dev, lower = steadier): {gaze_stability:.4f}")
    print(f"Vertical gaze stability (std dev, lower = steadier): {v_gaze_stability:.4f}")

    if has_attention:
        avg_attention = df.loc[distract_mask, "attention_score"].mean()
        print(f"Average attention score (distraction phase): {avg_attention:.1f}")

    os.makedirs("plots", exist_ok=True)

    # =========================================================================
    # FIGURE 1: main time-series panel
    # =========================================================================
    num_plots = 3  # EAR, horizontal gaze, vertical gaze
    if has_attention:
        num_plots += 1
    if has_head_pose:
        num_plots += 2  # head yaw/pitch panel + validation panel

    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 2.5 * num_plots), sharex=True)
    if num_plots == 1:
        axes = [axes]

    ax_idx = 0

    if has_attention:
        df["attention_score_smooth"] = smooth(df["attention_score"].values, window=5)
        axes[ax_idx].plot(df["timestamp"], df["attention_score"], alpha=0.3, label="raw attention", color="tab:purple")
        axes[ax_idx].plot(df["timestamp"], df["attention_score_smooth"], label="smoothed attention", color="indigo")
        axes[ax_idx].axvline(CALIBRATION_END_SEC, color="red", linestyle=":", label="calibration -> distraction")
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
    axes[ax_idx].axhline(0.5, color="red", linestyle="--", label="center")
    axes[ax_idx].set_ylabel("Gaze ratio (0=Left, 1=Right)")
    axes[ax_idx].set_title("Horizontal gaze position over time (eye-in-head only)")
    axes[ax_idx].legend()
    ax_idx += 1

    axes[ax_idx].plot(df["timestamp"], df["v_gaze_avg"], alpha=0.3, label="raw vertical gaze")
    axes[ax_idx].plot(df["timestamp"], df["v_gaze_avg_smooth"], label="smoothed vertical", color="tab:green")
    axes[ax_idx].axhline(0.5, color="red", linestyle="--", label="center")
    axes[ax_idx].set_ylabel("V-Gaze (0=Up, 1=Down)")
    axes[ax_idx].set_title("Vertical gaze position over time (eye-in-head only)")
    axes[ax_idx].legend()
    ax_idx += 1

    if has_head_pose:
        head_yaw_deg = np.degrees(df["head_yaw_rad"].values)
        head_pitch_deg = np.degrees(df["head_pitch_rad"].values)

        axes[ax_idx].plot(df["timestamp"], head_yaw_deg, label="head yaw (deg)", color="tab:orange")
        axes[ax_idx].plot(df["timestamp"], head_pitch_deg, label="head pitch (deg)", color="tab:brown")
        axes[ax_idx].set_ylabel("Degrees")
        axes[ax_idx].set_title("Head orientation over time")
        axes[ax_idx].legend()
        ax_idx += 1

        # Validation panel: eye-only vs head-only vs combined (horizontal axis)
        eye_h_centered = df["gaze_avg"] - 0.5
        axes[ax_idx].plot(df["timestamp"], eye_h_centered, alpha=0.5, label="eye-in-head only (h)")
        axes[ax_idx].plot(df["timestamp"], df["head_yaw_rad"], alpha=0.5, label="head yaw only (rad)")
        axes[ax_idx].plot(df["timestamp"], df["combined_h"], label="combined estimate (h)", color="black", linewidth=2)
        axes[ax_idx].set_ylabel("Signal value")
        axes[ax_idx].set_title("Validation: eye-only vs head-only vs combined gaze (horizontal)")
        axes[ax_idx].legend()
        ax_idx += 1

    axes[ax_idx - 1].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig(f"plots/{fileName}_analysis.png", dpi=150)
    print(f"Saved plot to plots/{fileName}_analysis.png")

    # =========================================================================
    # FIGURE 2: combined gaze trajectory vs. anchor (2D, not a time series)
    # =========================================================================
    if has_head_pose:
        calib_h = df.loc[calib_mask, "combined_h"]
        calib_v = df.loc[calib_mask, "combined_v"]
        distract_h = df.loc[distract_mask, "combined_h"]
        distract_v = df.loc[distract_mask, "combined_v"]

        anchor_h = calib_h.mean() if len(calib_h) else 0.0
        anchor_v = calib_v.mean() if len(calib_v) else 0.0
        # approximate personalized threshold the same way attention_experiment.py does
        h_std = calib_h.std() if len(calib_h) else 0.15
        v_std = calib_v.std() if len(calib_v) else 0.15
        radius = 3 * max(0.03, np.sqrt(h_std ** 2 + v_std ** 2))

        fig2, ax2 = plt.subplots(figsize=(7, 7))
        ax2.scatter(distract_h, distract_v, s=8, alpha=0.4, color="tab:red", label="distraction phase gaze")
        ax2.scatter(calib_h, calib_v, s=8, alpha=0.4, color="tab:blue", label="calibration phase gaze")
        ax2.scatter([anchor_h], [anchor_v], marker="*", s=250, color="black", label="anchor (calibrated target)", zorder=5)

        circle = plt.Circle((anchor_h, anchor_v), radius, fill=False, color="black",
                             linestyle="--", label="personalized threshold")
        ax2.add_patch(circle)

        ax2.set_xlabel("Combined horizontal gaze")
        ax2.set_ylabel("Combined vertical gaze")
        ax2.set_title("Combined gaze trajectory vs. calibrated anchor")
        ax2.legend()
        ax2.set_aspect("equal", adjustable="datalim")

        plt.tight_layout()
        plt.savefig(f"plots/{fileName}_trajectory.png", dpi=150)
        print(f"Saved plot to plots/{fileName}_trajectory.png")

    # =========================================================================
    # FIGURE 3: blink rate, calibration vs distraction phase
    # =========================================================================
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    ax3.bar(["Calibration", "Distraction"], [calib_blink_rate, distract_blink_rate],
            color=["tab:blue", "tab:red"])
    ax3.set_ylabel("Blink rate (per minute)")
    ax3.set_title("Blink rate by phase")
    plt.tight_layout()
    plt.savefig(f"plots/{fileName}_blink_rate.png", dpi=150)
    print(f"Saved plot to plots/{fileName}_blink_rate.png")

    plt.show(block=False)
    plt.pause(40)
    plt.close("all")


# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.ndimage import uniform_filter1d
# import os

# def load_data(path):
#     df = pd.read_csv(path)
#     return df


# def smooth(series, window=5):
#     """Simple moving-average low-pass filter."""
#     return uniform_filter1d(series, size=window)


# def plot(fileName):
#     df = load_data(f'logs/{fileName}.csv')

#     df["avg_ear_smooth"] = smooth(df["avg_ear"].values, window=5)
#     df["gaze_avg"] = (df["gaze_ratio_left"] + df["gaze_ratio_right"]) / 2
#     df["gaze_avg_smooth"] = smooth(df["gaze_avg"].values, window=5)

#     df["v_gaze_avg"] = (df["vertical_gaze_ratio_left"] + df["vertical_gaze_ratio_right"]) / 2
#     df["v_gaze_avg_smooth"] = smooth(df["v_gaze_avg"].values, window=5)

#     total_blinks = int(df["blink"].sum())
#     duration = df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]
#     blink_rate_per_min = total_blinks / (duration / 60) if duration > 0 else 0

#     # crude "gaze stability" metric: lower std = more fixated/stable
#     gaze_stability = df["gaze_avg_smooth"].std()
#     v_gaze_stability = df["v_gaze_avg_smooth"].std()

#     print("---- Summary ----")
#     print(f"Duration: {duration:.1f}s")
#     print(f"Total blinks: {total_blinks}")
#     print(f"Blink rate: {blink_rate_per_min:.1f} per minute")
#     print(f"Gaze stability (std dev, lower = steadier): {gaze_stability:.4f}")
#     print(f"Vertical gaze stability (std dev, lower = steadier): {v_gaze_stability:.4f}")

#     has_attention = "attention_score" in df.columns
#     num_plots = 4 if has_attention else 3

#     # Dynamically scale the figure size based on the number of plots
#     fig, axes = plt.subplots(num_plots, 1, figsize=(10, 2.5 * num_plots), sharex=True)

#     ax_idx = 0

#     if has_attention:
#         df["attention_score_smooth"] = smooth(df["attention_score"].values, window=5)
#         axes[ax_idx].plot(df["timestamp"], df["attention_score"], alpha=0.3, label="raw attention", color="tab:purple")
#         axes[ax_idx].plot(df["timestamp"], df["attention_score_smooth"], label="smoothed attention", color="indigo")
#         axes[ax_idx].set_ylabel("Attention Score")
#         axes[ax_idx].set_title("Attention Score over time")
#         axes[ax_idx].legend()
#         ax_idx += 1
        
#     axes[ax_idx].plot(df["timestamp"], df["avg_ear"], alpha=0.3, label="raw EAR")
#     axes[ax_idx].plot(df["timestamp"], df["avg_ear_smooth"], label="smoothed EAR")
#     axes[ax_idx].axhline(0.21, color="red", linestyle="--", label="blink threshold")
#     axes[ax_idx].set_ylabel("EAR")
#     axes[ax_idx].set_title("Eye Aspect Ratio over time (blink detection)")
#     axes[ax_idx].legend()
#     ax_idx += 1

#     axes[ax_idx].plot(df["timestamp"], df["gaze_avg"], alpha=0.3, label="raw gaze ratio")
#     axes[ax_idx].plot(df["timestamp"], df["gaze_avg_smooth"], label="smoothed gaze ratio")
#     axes[ax_idx].axhline(0.5, color="red", linestyle="--", label="Gaze threshold")
#     axes[ax_idx].set_ylabel("Gaze ratio (0 =Left, 1 =Right)")
#     axes[ax_idx].set_title("Horizontal gaze position over time")
#     axes[ax_idx].legend()
#     ax_idx += 1

#     axes[ax_idx].plot(df["timestamp"], df["v_gaze_avg"], alpha=0.3, label="raw vertical gaze")
#     axes[ax_idx].plot(df["timestamp"], df["v_gaze_avg_smooth"], label="smoothed vertical", color="tab:green")
#     axes[ax_idx].axhline(0.5, color="red", linestyle="--", label="Vertical gaze threshold")
#     axes[ax_idx].set_ylabel("V-Gaze (0=Up, 1=Down)")
#     axes[ax_idx].set_title("Vertical gaze position over time")
#     axes[ax_idx].legend()

#     axes[ax_idx].set_xlabel("Time (s)")

#     os.makedirs("plots", exist_ok=True)
#     plt.tight_layout()
#     plt.savefig(f"plots/{fileName}_analysis.png", dpi=150)
#     print(f"Saved plot to {fileName}_analysis.png")
#     plt.show(block=False)
#     plt.pause(40) 
#     plt.close()

