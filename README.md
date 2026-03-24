# smart_traffic_mini_project

A small 6th-semester mini project that takes CCTV/security camera traffic video as input, detects vehicles, estimates traffic density by region, and recommends which region should get GREEN signal.

## 1. Project Objective

Build a simple and presentable pipeline:
- Input traffic video from one intersection
- Detect vehicles using YOLOv8 + OpenCV
- Count vehicles in predefined road regions (Lane_A, Lane_B, Lane_C, Lane_D by default)
- Recommend traffic signal direction using easy rule-based logic
- Save outputs (video, CSV/JSON, plots, summary)

This project is intentionally simple for college demo/viva and laptop execution.

## 2. Input

Required:
- Traffic CCTV/security camera video (`.mp4`, `.avi`, etc.)

Optional:
- Region JSON file (`--region-file`) to customize lane/road polygons
- JSON config file (`--config`) for easy parameter setup
- Confidence threshold (`--confidence`)
- Frame skipping (`--frame-skip`)
- Limit processing to first N seconds (`--max-seconds`)

## 3. Processing Steps

1. Read video frames using OpenCV
2. Detect vehicles (car, bus, truck, motorcycle) using YOLOv8
3. Assign each detected vehicle to a region using bounding-box center point
4. Count vehicles per region per frame
5. Compute simple rolling density per region
6. Recommend GREEN region using rule-based signal logic:
   - highest count gets priority
   - if counts are close, rotate fairly
   - keep minimum green duration before switching

## 4. Output

Saved in `sample_output/` (or your `--output-dir`):
- Processed video with bounding boxes + region overlays
- `vehicle_counts.csv` (frame-wise counts + density)
- `signal_recommendations.csv`
- `signal_recommendations.json`
- `run_summary.json`
- `vehicle_counts_over_time.png`
- `region_density_over_time.png`
- Final frame snapshot image (optional)

Console output also prints:
- frame logs
- current recommendation
- final summary

## 5. Expected Output (Example)

- “Lane_B has 14 vehicles and others are lower, so GREEN is recommended for Lane_B.”
- “If Lane_B and Lane_C are close, system rotates fairly after minimum green time.”
- “Output video shows detected vehicles and current lane counts.”

## 6. How to Run

### Install

```bash
pip install -r requirements.txt
```

### Basic run

```bash
python main.py --input input_video.mp4
```

### Example with custom settings

```bash
python main.py \
  --input input_video.mp4 \
  --output-dir sample_output \
  --confidence 0.35 \
  --frame-skip 2 \
  --max-seconds 60 \
  --min-green-seconds 8 \
  --tie-margin 1
```

### Force CPU

```bash
python main.py --input input_video.mp4 --cpu
```

### Headless mode (no popup window)

```bash
python main.py --input input_video.mp4 --no-show-window
```

## 7. Region Configuration (Optional)

Use `region_config_example.json` format:

```json
{
  "regions": [
    { "name": "East", "points": [[100, 100], [300, 100], [300, 300], [100, 300]] },
    { "name": "West", "points": [[320, 100], [600, 100], [600, 300], [320, 300]] }
  ]
}
```

Run with:

```bash
python main.py --input input_video.mp4 --region-file region_config_example.json
```

## 8. Project Files

- `main.py`: entrypoint, argument parsing, full pipeline orchestration
- `config.py`: app settings + default/custom region loading
- `detector.py`: YOLOv8 detection wrapper
- `video_processor.py`: frame loop, integration of detection + counting + recommendation
- `traffic_analyzer.py`: region-wise counting and density estimation
- `signal_logic.py`: rule-based traffic signal recommendation
- `utils.py`: save CSV/JSON, plotting, summary printing
- `requirements.txt`: dependencies
- `project_config_example.json`: optional full config sample
- `region_config_example.json`: optional region polygon sample
- `sample_output/`: generated outputs

## 9. Limitations

- Single intersection only
- No vehicle tracking ID (only per-frame counting)
- Depends on camera angle and region quality
- Not optimized for real-time city-scale deployment
- Rule-based recommendation (not RL/advanced optimization)

## 10. Future Scope

- Better lane mapping for real roads
- Add lightweight tracking to reduce double counting
- Add adaptive timing (based on flow trend)
- Compare with fixed-time baseline in simulation

