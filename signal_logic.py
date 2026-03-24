"""Simple rule-based traffic signal recommendation logic."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class SignalDecision:
    """Decision made for a frame/time step."""

    frame_index: int
    timestamp_s: float
    recommended_green_region: str
    reason: str
    counts: dict[str, int]


class SignalLogic:
    """Recommend which region should get green signal."""

    def __init__(self, region_names: list[str], min_green_frames: int = 30, tie_margin: int = 1) -> None:
        if len(region_names) == 0:
            raise ValueError("SignalLogic requires at least one region name.")

        self.region_names = region_names
        self.min_green_frames = max(1, min_green_frames)
        self.tie_margin = max(0, tie_margin)

        self.current_green_region: str | None = None
        self.last_switch_frame = 0
        self.rotation_index = 0
        self.decisions: list[SignalDecision] = []

    def _pick_fair_region(self, candidates: list[str]) -> str:
        """Choose among tied regions using round-robin fairness."""

        if len(candidates) == 1:
            return candidates[0]

        for _ in range(len(self.region_names)):
            candidate = self.region_names[self.rotation_index % len(self.region_names)]
            self.rotation_index += 1
            if candidate in candidates:
                return candidate
        return sorted(candidates)[0]

    def recommend(self, frame_index: int, timestamp_s: float, counts: dict[str, int]) -> SignalDecision:
        """Return rule-based green recommendation for this frame."""

        missing_regions = [region for region in self.region_names if region not in counts]
        if missing_regions:
            raise ValueError(f"Counts missing required regions: {missing_regions}")

        if self.current_green_region is None:
            self.current_green_region = max(self.region_names, key=lambda name: counts[name])
            self.last_switch_frame = frame_index
            reason = "Initial choice: selected region with highest vehicle count."
        else:
            elapsed = frame_index - self.last_switch_frame
            if elapsed < self.min_green_frames:
                reason = (
                    f"Minimum green time active ({elapsed}/{self.min_green_frames} frames), "
                    f"keeping {self.current_green_region}."
                )
            else:
                max_count = max(counts.values()) if counts else 0
                candidates = [
                    name for name in self.region_names if max_count - counts[name] <= self.tie_margin
                ]
                selected = self._pick_fair_region(candidates)
                if selected != self.current_green_region:
                    reason = (
                        f"Switched from {self.current_green_region} to {selected} "
                        f"due to higher/competitive traffic load."
                    )
                    self.current_green_region = selected
                    self.last_switch_frame = frame_index
                else:
                    reason = f"Retained {self.current_green_region}; still among highest demand."

        decision = SignalDecision(
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            recommended_green_region=self.current_green_region,
            reason=reason,
            counts=dict(counts),
        )
        self.decisions.append(decision)
        return decision

    def to_dataframe(self) -> pd.DataFrame:
        """Convert decisions to DataFrame."""

        rows: list[dict[str, str | int | float]] = []
        for decision in self.decisions:
            row: dict[str, str | int | float] = {
                "frame": decision.frame_index,
                "timestamp_s": round(decision.timestamp_s, 2),
                "recommended_green_region": decision.recommended_green_region,
                "reason": decision.reason,
            }
            for region_name, count in decision.counts.items():
                row[region_name] = count
            rows.append(row)
        return pd.DataFrame(rows)

    def recommendation_change_count(self) -> int:
        """Count how many times recommendation changed."""

        if len(self.decisions) < 2:
            return 0
        changes = 0
        prev = self.decisions[0].recommended_green_region
        for decision in self.decisions[1:]:
            if decision.recommended_green_region != prev:
                changes += 1
            prev = decision.recommended_green_region
        return changes
