"""Project configuration helpers for the smart traffic mini project.

This module keeps all user-tunable values in one place so students can
experiment without editing many files.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class RegionSpec:
    """Represents one counting region on the traffic video frame."""

    name: str
    points: list[tuple[int, int]]
    color: tuple[int, int, int]


@dataclass(slots=True)
class AppConfig:
    """Runtime configuration for the mini project pipeline."""

    input_video: Path = Path("input/traffic_signal_video.mp4")
    output_dir: Path = Path("sample_output")

    output_video_name: str = "processed_output.mp4"
    counts_csv_name: str = "vehicle_counts.csv"
    recommendation_csv_name: str = "signal_recommendations.csv"
    recommendation_json_name: str = "signal_recommendations.json"
    summary_json_name: str = "run_summary.json"

    model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.35
    frame_skip: int = 1
    show_window: bool = True
    force_cpu: bool = False
    max_seconds: float | None = None

    region_file: Path | None = None
    min_green_seconds: float = 8.0
    tie_margin: int = 1

    save_final_frame: bool = True
    final_frame_name: str = "final_frame.jpg"
    plot_counts_name: str = "vehicle_counts_over_time.png"
    plot_density_name: str = "region_density_over_time.png"

    log_interval_frames: int = 30
    vehicle_classes: tuple[str, ...] = ("car", "bus", "truck", "motorcycle")


def _pick(cli_value: Any, json_value: Any, default_value: Any) -> Any:
    """Pick value priority: CLI > JSON config > hardcoded default."""

    if cli_value is not None:
        return cli_value
    if json_value is not None:
        return json_value
    return default_value


def load_json_config(config_path: Path | None) -> dict[str, Any]:
    """Load optional JSON config file."""

    if config_path is None:
        return {}
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Config file not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in config file: {config_path}") from exc

    if not isinstance(data, dict):
        raise ValueError("Config JSON must be an object at the top level.")
    return data


def build_config(args: argparse.Namespace) -> AppConfig:
    """Build AppConfig by combining CLI values and optional JSON config."""

    config_json_path = Path(args.config) if args.config is not None else None
    json_cfg = load_json_config(config_json_path)
    defaults = AppConfig()

    show_window = _pick(args.show_window, json_cfg.get("show_window"), defaults.show_window)
    save_final_frame = _pick(
        args.save_final_frame, json_cfg.get("save_final_frame"), defaults.save_final_frame
    )

    cfg = AppConfig(
        input_video=Path(_pick(args.input, json_cfg.get("input_video"), defaults.input_video)),
        output_dir=Path(_pick(args.output_dir, json_cfg.get("output_dir"), defaults.output_dir)),
        output_video_name=_pick(
            args.output_video_name, json_cfg.get("output_video_name"), defaults.output_video_name
        ),
        counts_csv_name=_pick(
            args.counts_csv_name, json_cfg.get("counts_csv_name"), defaults.counts_csv_name
        ),
        recommendation_csv_name=_pick(
            args.recommendation_csv_name,
            json_cfg.get("recommendation_csv_name"),
            defaults.recommendation_csv_name,
        ),
        recommendation_json_name=_pick(
            args.recommendation_json_name,
            json_cfg.get("recommendation_json_name"),
            defaults.recommendation_json_name,
        ),
        summary_json_name=_pick(
            args.summary_json_name, json_cfg.get("summary_json_name"), defaults.summary_json_name
        ),
        model_path=str(_pick(args.model, json_cfg.get("model_path"), defaults.model_path)),
        confidence_threshold=float(
            _pick(args.confidence, json_cfg.get("confidence_threshold"), defaults.confidence_threshold)
        ),
        frame_skip=int(_pick(args.frame_skip, json_cfg.get("frame_skip"), defaults.frame_skip)),
        show_window=bool(show_window),
        force_cpu=bool(_pick(args.cpu, json_cfg.get("force_cpu"), defaults.force_cpu)),
        max_seconds=(
            float(_pick(args.max_seconds, json_cfg.get("max_seconds"), defaults.max_seconds))
            if _pick(args.max_seconds, json_cfg.get("max_seconds"), defaults.max_seconds)
            is not None
            else None
        ),
        region_file=(
            Path(_pick(args.region_file, json_cfg.get("region_file"), defaults.region_file))
            if _pick(args.region_file, json_cfg.get("region_file"), defaults.region_file) is not None
            else None
        ),
        min_green_seconds=float(
            _pick(args.min_green_seconds, json_cfg.get("min_green_seconds"), defaults.min_green_seconds)
        ),
        tie_margin=int(_pick(args.tie_margin, json_cfg.get("tie_margin"), defaults.tie_margin)),
        save_final_frame=bool(save_final_frame),
        final_frame_name=_pick(
            args.final_frame_name, json_cfg.get("final_frame_name"), defaults.final_frame_name
        ),
        plot_counts_name=_pick(
            args.plot_counts_name, json_cfg.get("plot_counts_name"), defaults.plot_counts_name
        ),
        plot_density_name=_pick(
            args.plot_density_name, json_cfg.get("plot_density_name"), defaults.plot_density_name
        ),
        log_interval_frames=int(
            _pick(args.log_interval_frames, json_cfg.get("log_interval_frames"), defaults.log_interval_frames)
        ),
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: AppConfig) -> None:
    """Simple validation so runtime failures are easier to understand."""

    if not 0.0 <= cfg.confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0.0 and 1.0")
    if cfg.frame_skip < 1:
        raise ValueError("frame_skip must be >= 1")
    if cfg.min_green_seconds <= 0:
        raise ValueError("min_green_seconds must be > 0")
    if cfg.tie_margin < 0:
        raise ValueError("tie_margin must be >= 0")
    if cfg.max_seconds is not None and cfg.max_seconds <= 0:
        raise ValueError("max_seconds must be > 0 when provided")


def default_regions(frame_width: int, frame_height: int) -> list[RegionSpec]:
    """Create 4 simple rectangular regions (Lane_A ... Lane_D).

    Assumption:
    - Camera is fixed and intersection mostly spans the frame.
    - These are demo-friendly defaults. For real videos, prefer custom regions.
    """

    half_w = frame_width // 2
    half_h = frame_height // 2
    return [
        RegionSpec(
            name="Lane_A",
            points=[(0, 0), (half_w, 0), (half_w, half_h), (0, half_h)],
            color=(0, 255, 0),
        ),
        RegionSpec(
            name="Lane_B",
            points=[(half_w, 0), (frame_width, 0), (frame_width, half_h), (half_w, half_h)],
            color=(255, 255, 0),
        ),
        RegionSpec(
            name="Lane_C",
            points=[(0, half_h), (half_w, half_h), (half_w, frame_height), (0, frame_height)],
            color=(255, 0, 255),
        ),
        RegionSpec(
            name="Lane_D",
            points=[
                (half_w, half_h),
                (frame_width, half_h),
                (frame_width, frame_height),
                (half_w, frame_height),
            ],
            color=(0, 165, 255),
        ),
    ]


def load_regions(region_file: Path | None, frame_width: int, frame_height: int) -> list[RegionSpec]:
    """Load user-defined regions from JSON, else fallback to default regions."""

    if region_file is None:
        return default_regions(frame_width, frame_height)

    data = load_json_config(region_file)
    raw_regions = data.get("regions")
    if not isinstance(raw_regions, list) or len(raw_regions) == 0:
        raise ValueError("Region file must contain a non-empty 'regions' list.")

    palette: list[tuple[int, int, int]] = [(0, 255, 0), (255, 255, 0), (255, 0, 255), (0, 165, 255)]
    parsed: list[RegionSpec] = []

    for idx, item in enumerate(raw_regions):
        if not isinstance(item, dict):
            raise ValueError("Each region must be a JSON object.")
        name = str(item.get("name", f"Lane_{idx + 1}"))
        raw_points = item.get("points")
        if not isinstance(raw_points, list) or len(raw_points) < 3:
            raise ValueError(f"Region '{name}' must have at least 3 points.")

        points: list[tuple[int, int]] = []
        for point in raw_points:
            if (
                not isinstance(point, list | tuple)
                or len(point) != 2
                or not isinstance(point[0], int | float)
                or not isinstance(point[1], int | float)
            ):
                raise ValueError(f"Invalid point in region '{name}'.")
            x = int(max(0, min(frame_width - 1, int(point[0]))))
            y = int(max(0, min(frame_height - 1, int(point[1]))))
            points.append((x, y))

        raw_color = item.get("color")
        if (
            isinstance(raw_color, list | tuple)
            and len(raw_color) == 3
            and all(isinstance(c, int | float) for c in raw_color)
        ):
            color = (int(raw_color[0]), int(raw_color[1]), int(raw_color[2]))
        else:
            color = palette[idx % len(palette)]

        parsed.append(RegionSpec(name=name, points=points, color=color))

    return parsed
