"""Entry point for smart_traffic_mini_project.

Pipeline:
1. Read video
2. Detect vehicles with YOLOv8
3. Count vehicles by region
4. Recommend green signal using simple rule logic
5. Save outputs (video/csv/json/plots/summary)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import AppConfig, build_config, validate_config
from detector import YOLOVehicleDetector
from utils import (
    ensure_output_dir,
    plot_region_density_over_time,
    plot_vehicle_counts_over_time,
    print_summary,
    save_final_frame,
    save_summary_json,
    save_tables,
)
from video_processor import VideoProcessor


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Smart Traffic Mini Project")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config file.")
    parser.add_argument("--input", type=str, default=None, help="Input traffic video path.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory.")

    parser.add_argument("--model", type=str, default=None, help="YOLO model path (example: yolov8n.pt).")
    parser.add_argument("--confidence", type=float, default=None, help="Detection confidence threshold.")
    parser.add_argument("--frame-skip", type=int, default=None, help="Run detection every Nth frame.")
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Process only first N seconds (optional).",
    )
    parser.add_argument(
        "--region-file",
        type=str,
        default=None,
        help="Optional JSON file defining custom regions.",
    )
    parser.add_argument(
        "--min-green-seconds",
        type=float,
        default=None,
        help="Minimum duration for green recommendation before switch.",
    )
    parser.add_argument(
        "--tie-margin",
        type=int,
        default=None,
        help="If counts are close (within margin), apply fair rotation.",
    )
    parser.add_argument(
        "--cpu",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force CPU inference.",
    )

    parser.add_argument(
        "--show-window",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show processed video window while running.",
    )
    parser.add_argument(
        "--save-final-frame",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save final annotated frame snapshot.",
    )

    parser.add_argument("--output-video-name", type=str, default=None, help="Output processed video name.")
    parser.add_argument("--counts-csv-name", type=str, default=None, help="Vehicle counts CSV filename.")
    parser.add_argument(
        "--recommendation-csv-name",
        type=str,
        default=None,
        help="Recommendation CSV filename.",
    )
    parser.add_argument(
        "--recommendation-json-name",
        type=str,
        default=None,
        help="Recommendation JSON filename.",
    )
    parser.add_argument("--summary-json-name", type=str, default=None, help="Summary JSON filename.")
    parser.add_argument("--final-frame-name", type=str, default=None, help="Final frame image filename.")
    parser.add_argument("--plot-counts-name", type=str, default=None, help="Counts plot PNG filename.")
    parser.add_argument("--plot-density-name", type=str, default=None, help="Density plot PNG filename.")
    parser.add_argument(
        "--log-interval-frames",
        type=int,
        default=None,
        help="Print logs after every N frames.",
    )
    return parser.parse_args()


def run_pipeline(config: AppConfig) -> int:
    """Run full mini project pipeline."""

    if not config.input_video.exists():
        print(f"Error: Input video not found: {config.input_video}")
        print("Tip: Provide a valid path with --input <video_path>")
        return 1

    ensure_output_dir(config.output_dir)
    validate_config(config)

    print("Running Smart Traffic Mini Project with settings:")
    print(f"  Input video        : {config.input_video}")
    print(f"  Output directory   : {config.output_dir}")
    print(f"  YOLO model         : {config.model_path}")
    print(f"  Confidence         : {config.confidence_threshold}")
    print(f"  Frame skip         : {config.frame_skip}")
    print(f"  Show window        : {config.show_window}")
    print(f"  Max seconds        : {config.max_seconds}")
    print(f"  Min green seconds  : {config.min_green_seconds}")
    print(f"  Tie margin         : {config.tie_margin}")
    print("")

    detector = YOLOVehicleDetector(
        model_path=config.model_path,
        confidence_threshold=config.confidence_threshold,
        allowed_classes=config.vehicle_classes,
        force_cpu=config.force_cpu,
    )
    processor = VideoProcessor(config=config, detector=detector)
    result = processor.process()

    table_paths = save_tables(
        counts_df=result.counts_df,
        recommendations_df=result.recommendations_df,
        output_dir=config.output_dir,
        counts_csv_name=config.counts_csv_name,
        recommendation_csv_name=config.recommendation_csv_name,
        recommendation_json_name=config.recommendation_json_name,
    )
    summary_path = save_summary_json(result.summary, config.output_dir, config.summary_json_name)

    counts_plot_path = plot_vehicle_counts_over_time(
        counts_df=result.counts_df,
        region_names=result.region_names,
        output_dir=config.output_dir,
        output_name=config.plot_counts_name,
    )
    density_plot_path = plot_region_density_over_time(
        counts_df=result.counts_df,
        region_names=result.region_names,
        output_dir=config.output_dir,
        output_name=config.plot_density_name,
    )

    frame_path = None
    if config.save_final_frame:
        frame_path = save_final_frame(result.final_frame, config.output_dir, config.final_frame_name)

    print_summary(result.summary)
    print("Saved outputs:")
    print(f"  Processed video            : {result.output_video_path}")
    print(f"  Vehicle counts CSV         : {table_paths['counts_csv']}")
    print(f"  Recommendations CSV        : {table_paths['recommendations_csv']}")
    print(f"  Recommendations JSON       : {table_paths['recommendations_json']}")
    print(f"  Summary JSON               : {summary_path}")
    if counts_plot_path is not None:
        print(f"  Counts plot                : {counts_plot_path}")
    if density_plot_path is not None:
        print(f"  Density plot               : {density_plot_path}")
    if frame_path is not None:
        print(f"  Final frame snapshot       : {frame_path}")
    return 0


def main() -> int:
    """CLI entrypoint."""

    args = parse_args()
    try:
        config = build_config(args)
        return run_pipeline(config)
    except Exception as exc:  # noqa: BLE001 - keep error handling simple for college project
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
