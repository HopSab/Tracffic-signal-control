"""Utility helpers for saving outputs, plotting, and printing summary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import pandas as pd

# Use non-interactive backend so plotting also works in terminals/headless mode.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_output_dir(path: Path) -> Path:
    """Create output directory if missing."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def save_tables(
    counts_df: pd.DataFrame,
    recommendations_df: pd.DataFrame,
    output_dir: Path,
    counts_csv_name: str,
    recommendation_csv_name: str,
    recommendation_json_name: str,
) -> dict[str, Path]:
    """Save count and recommendation tables."""

    counts_csv_path = output_dir / counts_csv_name
    recommendations_csv_path = output_dir / recommendation_csv_name
    recommendations_json_path = output_dir / recommendation_json_name

    counts_df.to_csv(counts_csv_path, index=False)
    recommendations_df.to_csv(recommendations_csv_path, index=False)
    recommendations_json_path.write_text(
        recommendations_df.to_json(orient="records", indent=2),
        encoding="utf-8",
    )

    return {
        "counts_csv": counts_csv_path,
        "recommendations_csv": recommendations_csv_path,
        "recommendations_json": recommendations_json_path,
    }


def save_summary_json(summary: dict[str, Any], output_dir: Path, summary_name: str) -> Path:
    """Save run summary JSON."""

    summary_path = output_dir / summary_name
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def save_final_frame(frame: Any, output_dir: Path, frame_name: str) -> Path | None:
    """Save final annotated frame snapshot."""

    if frame is None:
        return None
    frame_path = output_dir / frame_name
    ok = cv2.imwrite(str(frame_path), frame)
    return frame_path if ok else None


def plot_vehicle_counts_over_time(
    counts_df: pd.DataFrame, region_names: list[str], output_dir: Path, output_name: str
) -> Path | None:
    """Plot total and per-region counts over time."""

    if counts_df.empty or "frame" not in counts_df:
        return None

    plt.figure(figsize=(12, 6))
    plt.plot(counts_df["frame"], counts_df["total_count"], label="Total Count", linewidth=2.2)
    for region_name in region_names:
        if region_name in counts_df:
            plt.plot(counts_df["frame"], counts_df[region_name], label=region_name, alpha=0.85)

    plt.title("Vehicle Count Over Time")
    plt.xlabel("Frame")
    plt.ylabel("Vehicle Count")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / output_name
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_region_density_over_time(
    counts_df: pd.DataFrame, region_names: list[str], output_dir: Path, output_name: str
) -> Path | None:
    """Plot rolling density (moving average counts) per region."""

    density_cols = [f"{region_name}_density" for region_name in region_names]
    if counts_df.empty or not all(col in counts_df for col in density_cols):
        return None

    plt.figure(figsize=(12, 6))
    for region_name in region_names:
        col = f"{region_name}_density"
        plt.plot(counts_df["frame"], counts_df[col], label=region_name, linewidth=2)

    plt.title("Per-Region Traffic Density (Rolling)")
    plt.xlabel("Frame")
    plt.ylabel("Density (moving average count)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / output_name
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def print_summary(summary: dict[str, Any]) -> None:
    """Print end-of-run summary in student-friendly format."""

    print("\n========== Final Summary ==========")
    print(f"Processed frames            : {summary.get('processed_frames', 0)}")
    print(f"Processed seconds           : {summary.get('processed_seconds', 0)}")
    print(f"Total vehicles detected     : {summary.get('total_detected_vehicles', 0)}")
    print(f"Busiest region              : {summary.get('busiest_region', 'N/A')}")
    print(f"Recommendation changes      : {summary.get('recommendation_changes', 0)}")

    avg_keys = [key for key in summary if key.startswith("avg_")]
    if avg_keys:
        print("Average count per region    :")
        for key in sorted(avg_keys):
            print(f"  - {key.replace('avg_', '')}: {summary[key]}")
    print("===================================\n")
