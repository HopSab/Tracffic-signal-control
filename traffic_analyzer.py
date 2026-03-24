"""Traffic analysis module for region-wise counting and density estimation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd

from config import RegionSpec
from detector import Detection


@dataclass(slots=True)
class FrameAnalysis:
    """Per-frame analysis data."""

    frame_index: int
    timestamp_s: float
    region_counts: dict[str, int]
    total_count: int


class TrafficAnalyzer:
    """Assign vehicles to regions and track counts over time."""

    def __init__(self, regions: list[RegionSpec], rolling_window_frames: int = 30) -> None:
        if len(regions) == 0:
            raise ValueError("At least one region is required.")

        self.regions = regions
        self.region_names = [region.name for region in regions]
        self.rolling_window_frames = max(1, rolling_window_frames)

        self.records: list[dict[str, float | int]] = []
        self.total_detected_vehicles = 0
        self._rolling: dict[str, deque[int]] = {
            name: deque(maxlen=self.rolling_window_frames) for name in self.region_names
        }

    def _point_region(self, point: tuple[int, int]) -> str | None:
        """Return region name for a point, else None."""

        x, y = point
        for region in self.regions:
            polygon = np.array(region.points, dtype=np.int32)
            inside = cv2.pointPolygonTest(polygon, (float(x), float(y)), False)
            if inside >= 0:
                return region.name
        return None

    def analyze_frame(
        self, frame_index: int, timestamp_s: float, detections: list[Detection]
    ) -> FrameAnalysis:
        """Count vehicles per region and update rolling density."""

        region_counts: dict[str, int] = {name: 0 for name in self.region_names}

        for detection in detections:
            region_name = self._point_region(detection.center)
            if region_name is not None:
                region_counts[region_name] += 1

        total_count = int(sum(region_counts.values()))
        self.total_detected_vehicles += total_count

        for region_name in self.region_names:
            self._rolling[region_name].append(region_counts[region_name])

        record: dict[str, float | int] = {
            "frame": frame_index,
            "timestamp_s": round(timestamp_s, 2),
            "total_count": total_count,
        }
        for region_name in self.region_names:
            record[region_name] = region_counts[region_name]
            rolling_density = sum(self._rolling[region_name]) / len(self._rolling[region_name])
            record[f"{region_name}_density"] = round(rolling_density, 3)

        self.records.append(record)

        return FrameAnalysis(
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            region_counts=region_counts,
            total_count=total_count,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all frame records to DataFrame."""

        return pd.DataFrame(self.records)

    def draw_regions(self, frame: np.ndarray, region_counts: dict[str, int]) -> np.ndarray:
        """Overlay region outlines and current counts on frame."""

        for region in self.regions:
            polygon = np.array(region.points, dtype=np.int32)
            cv2.polylines(frame, [polygon], isClosed=True, color=region.color, thickness=2)

            anchor_x = min(point[0] for point in region.points) + 8
            anchor_y = min(point[1] for point in region.points) + 24

            label = f"{region.name}: {region_counts.get(region.name, 0)}"
            cv2.putText(
                frame,
                label,
                (anchor_x, anchor_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                region.color,
                2,
                cv2.LINE_AA,
            )
        return frame
