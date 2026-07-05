import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import os

from config import (CALIBRATION_END_SEC, DISTRACTION_END_SEC, DISTRACTOR_INTERVAL_SEC)

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

    if has_head_pose:
        calib_h = df.loc[calib_mask, "combined_h"]
        calib_v = df.loc[calib_mask, "combined_v"]
        distract_h = df.loc[distract_mask, "combined_h"]
        distract_v = df.loc[distract_mask, "combined_v"]

        anchor_h = calib_h.mean() if len(calib_h) else 0.0
        anchor_v = calib_v.mean() if len(calib_v) else 0.0
        h_std = calib_h.std() if len(calib_h) else 0.15
        v_std = calib_v.std() if len(calib_v) else 0.15
        radius = 3 * max(0.03, np.sqrt(h_std ** 2 + v_std ** 2))

    if has_attention:
        avg_attention = df.loc[distract_mask, "attention_score"].mean()
        print(f"Average attention score (distraction phase): {avg_attention:.1f}")

    os.makedirs("plots", exist_ok=True)

    num_plots = 3  # EAR, horizontal gaze, vertical gaze
    if has_attention:
        num_plots += 1
    if has_head_pose:
        num_plots += 3

    # Ensure we have at least enough rows to neatly split the right column plots
    grid_rows = max(num_plots, 4)
    
    # Width ratios: 65% for time-series timelines, 35% for summary blocks
    fig = plt.figure(figsize=(16, 2.5 * grid_rows))
    gs = fig.add_gridspec(grid_rows, 2, width_ratios=[2, 1.1])

    # Generate Left Side (Time Series) Axes
    axes = []
    for i in range(num_plots):
        if i == 0:
            ax = fig.add_subplot(gs[i, 0])
        else:
            # sharex links timeline zoom/pans together
            ax = fig.add_subplot(gs[i, 0], sharex=axes[0])
        axes.append(ax)

    ax_idx = 0

    # 1. Attention Plot
    if has_attention:
        df["attention_score_smooth"] = smooth(df["attention_score"].values, window=5)
        axes[ax_idx].plot(df["timestamp"], df["attention_score"], alpha=0.3, label="raw attention")
        axes[ax_idx].plot(df["timestamp"], df["attention_score_smooth"], label="smoothed attention", color="indigo")
        axes[ax_idx].axvline(CALIBRATION_END_SEC, color="red", linestyle=":", label="calibration -> distraction")
        axes[ax_idx].set_ylabel("Attention Score")
        axes[ax_idx].set_title("Attention Score over time")
        axes[ax_idx].legend()
        ax_idx += 1

    if has_head_pose:
        axes[ax_idx].plot(df["timestamp"], df["combined_h"], label="combined horizontal gaze", color="navy", linewidth=1.5)
        axes[ax_idx].axhline(anchor_h, color="blue", linestyle=":", linewidth=2, label=f"anchor ({anchor_h:.2f})")
        axes[ax_idx].set_ylabel("H-Gaze Value")
        axes[ax_idx].set_title("Combined Horizontal Gaze over time vs. Anchor")
        axes[ax_idx].legend()
        ax_idx += 1

        # 2c. New Plot: Combined Vertical Gaze Timeline
        axes[ax_idx].plot(df["timestamp"], df["combined_v"], label="combined vertical gaze", color="darkgreen", linewidth=1.5)
        axes[ax_idx].axhline(anchor_v, color="green", linestyle=":", linewidth=2, label=f"anchor ({anchor_v:.2f})")
        axes[ax_idx].set_ylabel("V-Gaze Value")
        axes[ax_idx].set_title("Combined Vertical Gaze over time vs. Anchor")
        axes[ax_idx].legend()
        ax_idx += 1

        head_yaw_deg = np.degrees(df["head_yaw_rad"].values)
        head_pitch_deg = np.degrees(df["head_pitch_rad"].values)

        axes[ax_idx].plot(df["timestamp"], head_yaw_deg, label="head yaw (- left , + right)", color="black")
        axes[ax_idx].plot(df["timestamp"], head_pitch_deg, label="head pitch (- up , + down)", color="brown")
        axes[ax_idx].set_ylabel("Degrees")
        axes[ax_idx].set_title("Head orientation over time")
        axes[ax_idx].legend()
        ax_idx += 1

    # 2. EAR Plot
    axes[ax_idx].plot(df["timestamp"], df["avg_ear"], alpha=0.3, label="raw EAR")
    axes[ax_idx].plot(df["timestamp"], df["avg_ear_smooth"], label="smoothed EAR")
    axes[ax_idx].axhline(0.21, color="red", linestyle="--", label="blink threshold")
    axes[ax_idx].set_ylabel("EAR")
    axes[ax_idx].set_title("Eye Aspect Ratio over time (blink detection)")
    axes[ax_idx].legend()
    ax_idx += 1

    # 3. Horizontal Gaze Plot
    axes[ax_idx].plot(df["timestamp"], df["gaze_avg"], alpha=0.3, label="raw gaze ratio")
    axes[ax_idx].plot(df["timestamp"], df["gaze_avg_smooth"], label="smoothed gaze ratio", color='royalblue')
    axes[ax_idx].axhline(0.5, color="red", linestyle="--", label="center")
    axes[ax_idx].set_ylabel("Gaze ratio (0=Left, 1=Right)")
    axes[ax_idx].set_title("Horizontal gaze position over time (eye-in-head only)")
    axes[ax_idx].legend()
    ax_idx += 1

    # 4. Vertical Gaze Plot
    axes[ax_idx].plot(df["timestamp"], df["v_gaze_avg"], alpha=0.3, label="raw vertical gaze")
    axes[ax_idx].plot(df["timestamp"], df["v_gaze_avg_smooth"], label="smoothed vertical", color="mediumseagreen")
    axes[ax_idx].axhline(0.5, color="red", linestyle="--", label="center")
    axes[ax_idx].set_ylabel("V-Gaze (0=Up, 1=Down)")
    axes[ax_idx].set_title("Vertical gaze position over time (eye-in-head only)")
    axes[ax_idx].legend()
    ax_idx += 1

    # Apply X-axis label only to the very last active time-series plot
    axes[ax_idx - 1].set_xlabel("Time (s)")

    # =========================================================================
    # RIGHT SIDE: SUMMARY PLOTS 
    # =========================================================================

    one_third = grid_rows // 3
    two_thirds = 2 * one_third

    # Top Plot: Trajectory Map
    if has_head_pose:
        ax_trajectory = fig.add_subplot(gs[:one_third, 1])
        ax_trajectory.scatter(distract_h, distract_v, s=8, alpha=0.4, color="tab:red", label="distraction phase")
        ax_trajectory.scatter(calib_h, calib_v, s=8, alpha=0.4, color="tab:blue", label="calibration phase")
        ax_trajectory.scatter([anchor_h], [anchor_v], marker="*", s=250, color="black", label="anchor target", zorder=5)

        circle = plt.Circle((anchor_h, anchor_v), radius, fill=False, color="black", linestyle="--", label="threshold")
        ax_trajectory.add_patch(circle)

        ax_trajectory.set_xlabel("Combined horizontal gaze")
        ax_trajectory.set_ylabel("Combined vertical gaze")
        ax_trajectory.set_title("Combined Gaze Trajectory vs. Anchor")
        ax_trajectory.legend()
        ax_trajectory.set_aspect("equal", adjustable="datalim")

    # Middle Plot: Block-wise Attention Score
    ax_block_attention = fig.add_subplot(gs[one_third:two_thirds, 1])
    if has_attention and distract_mask.any():
        # Segment attention scores into intervals after calibration ends
        distract_df = df[distract_mask].copy()
        distract_df["block"] = ((distract_df["timestamp"] - CALIBRATION_END_SEC) // DISTRACTOR_INTERVAL_SEC).astype(int) + 1
        
        # Calculate mean attention per block group
        block_stats = distract_df.groupby("block")["attention_score"].mean()
        block_labels = [f"B{b}\n({(b-1)*int(DISTRACTOR_INTERVAL_SEC)}s-{(b)*int(DISTRACTOR_INTERVAL_SEC)}s)" for b in block_stats.index]
        
        # Plot attention bars
        bars = ax_block_attention.bar(block_labels, block_stats.values, color="indigo", width=0.5, alpha=0.85)
        ax_block_attention.set_ylabel("Avg Attention Score")
        ax_block_attention.set_ylim(0, 105) # Assuming attention operates on a standard 0-100 scale
        ax_block_attention.set_title("Average Attention per Distraction Block")
        ax_block_attention.grid(axis='y', linestyle=':', alpha=0.6)
        
        # Add values on top of the bars
        for bar in bars:
            yval = bar.get_height()
            ax_block_attention.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}", ha='center', va='bottom', fontsize=9)
    else:
        ax_block_attention.text(0.5, 0.5, "Attention data unavailable\nor calibration phase running", ha='center', va='center')
        ax_block_attention.set_title("Average Attention per Distraction Block")

    # Bottom Plot: Blink Rate Chart
    ax_blink = fig.add_subplot(gs[two_thirds:, 1])
    ax_blink.bar(["Calibration", "Distraction"], [calib_blink_rate, distract_blink_rate], color=["tab:blue", "tab:red"], width=0.5)
    ax_blink.set_ylabel("Blink rate (per minute)")
    ax_blink.set_title("Blink Rate Comparison by Phase")
    ax_blink.grid(axis='y', linestyle=':', alpha=0.6)


    # Save and Show Consolidated Plot
    plt.tight_layout()
    plt.savefig(f"plots/{fileName}_combined_report.png", dpi=150)
    print(f"Saved complete single dashboard to plots/{fileName}_combined_report.png")

    plt.show(block=False)
    plt.pause(40)
    plt.close("all")