"""Video processing pipeline: read frames, detect vehicles, analyze traffic, recommend signal."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from config import AppConfig, load_regions
from detector import Detection, YOLOVehicleDetector
from signal_logic import SignalLogic
from traffic_analyzer import TrafficAnalyzer


@dataclass(slots=True)
class ProcessingResult:
    """Outputs returned after full video processing."""

    counts_df: pd.DataFrame
    recommendations_df: pd.DataFrame
    summary: dict[str, int | float | str]
    output_video_path: Path
    final_frame: np.ndarray | None
    region_names: list[str]


class VideoProcessor:
    """Runs the main loop on a single video file."""

    def __init__(self, config: AppConfig, detector: YOLOVehicleDetector) -> None:
        self.config = config
        self.detector = detector

    @staticmethod
    def _draw_status_bar(
        frame: np.ndarray,
        frame_index: int,
        total_count: int,
        recommended_region: str,
    ) -> np.ndarray:
        """Overlay quick status text for demo clarity."""

        cv2.putText(
            frame,
            f"Frame: {frame_index}",
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Vehicles in regions: {total_count}",
            (10, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Recommended GREEN: {recommended_region}",
            (10, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return frame

    def process(self) -> ProcessingResult:
        """Process video and return all output tables + summary."""

        cap = cv2.VideoCapture(str(self.config.input_video))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open input video: {self.config.input_video}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        output_video_path = self.config.output_dir / self.config.output_video_name
        writer = cv2.VideoWriter(
            str(output_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not create output video: {output_video_path}")

        regions = load_regions(self.config.region_file, width, height)
        analyzer = TrafficAnalyzer(regions=regions, rolling_window_frames=max(3, int(fps)))
        signal_logic = SignalLogic(
            region_names=analyzer.region_names,
            min_green_frames=max(1, int(self.config.min_green_seconds * fps)),
            tie_margin=self.config.tie_margin,
        )

        max_frames = int(self.config.max_seconds * fps) if self.config.max_seconds is not None else None
        frame_index = 0
        last_detections: list[Detection] = []
        final_frame: np.ndarray | None = None

        print("Starting video processing...")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and frame_index >= max_frames:
                break

            run_detection = frame_index % self.config.frame_skip == 0
            if run_detection:
                detections = self.detector.detect(frame)
                last_detections = detections
            else:
                detections = last_detections

            timestamp_s = frame_index / fps
            analysis = analyzer.analyze_frame(frame_index, timestamp_s, detections)
            decision = signal_logic.recommend(frame_index, timestamp_s, analysis.region_counts)

            annotated = frame.copy()
            analyzer.draw_regions(annotated, analysis.region_counts)
            self.detector.draw_detections(annotated, detections)
            self._draw_status_bar(
                annotated,
                frame_index=frame_index,
                total_count=analysis.total_count,
                recommended_region=decision.recommended_green_region,
            )

            writer.write(annotated)
            final_frame = annotated

            if self.config.show_window:
                cv2.imshow("Smart Traffic Mini Project", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Stopped by user (pressed q).")
                    break

            if frame_index % self.config.log_interval_frames == 0:
                print(
                    f"frame={frame_index:05d} "
                    f"vehicles={len(detections):02d} "
                    f"recommended={decision.recommended_green_region}"
                )

            frame_index += 1

        cap.release()
        writer.release()
        cv2.destroyAllWindows()

        counts_df = analyzer.to_dataframe()
        recommendations_df = signal_logic.to_dataframe()

        busiest_region = "N/A"
        if not counts_df.empty:
            region_totals = {
                region: float(counts_df[region].sum()) for region in analyzer.region_names if region in counts_df
            }
            if region_totals:
                busiest_region = max(region_totals, key=region_totals.get)

        summary: dict[str, int | float | str] = {
            "processed_frames": frame_index,
            "video_fps": round(float(fps), 2),
            "processed_seconds": round(frame_index / fps, 2),
            "total_detected_vehicles": int(analyzer.total_detected_vehicles),
            "busiest_region": busiest_region,
            "recommendation_changes": int(signal_logic.recommendation_change_count()),
        }
        for region_name in analyzer.region_names:
            if not counts_df.empty and region_name in counts_df:
                summary[f"avg_{region_name}"] = round(float(counts_df[region_name].mean()), 3)
            else:
                summary[f"avg_{region_name}"] = 0.0

        return ProcessingResult(
            counts_df=counts_df,
            recommendations_df=recommendations_df,
            summary=summary,
            output_video_path=output_video_path,
            final_frame=final_frame,
            region_names=analyzer.region_names,
        )
