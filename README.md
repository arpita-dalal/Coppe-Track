# Coppe-Track

Coppe-Track is a Python-based video processing and tracking pipeline designed to track subjects (such as copepods) through complex topographical environments. It uses a combination of frame extraction, inversion, topographical gating, and MOG2-based background subtraction to generate precise positional trajectories.

## Requirements

The project relies on standard Python computer vision and data science libraries:
- `opencv-python`
- `numpy`
- `pandas`
- `pillow`
- `matplotlib`

You can install all dependencies via:
```bash
pip install -r requirements.txt
```
*(Alternatively, you can run `bash install_dependencies.sh`)*

## Project Structure

```
Coppe-Track/
├── data/
│   ├── input/       # Place your source video files here (.mov, .mp4, etc.)
│   ├── cache/       # Intermediate frames are stored here during processing
│   └── output/      # Final CSV position data and trajectory plots
├── logs/            # Execution logs
├── src/             # Core Python modules
│   ├── main.py
│   ├── image_processor.py
│   ├── tracking_engine.py
│   └── video_loader.py
├── runner.py        # Interactive CLI entry point
└── compare_tracking.py
```

## How to Run

1. Place your video file (e.g., `video.mp4`) into the `data/input/` directory.
2. Run the interactive pipeline via the terminal:

```bash
python runner.py
```

3. Follow the on-screen prompts to select your video and choose which pipeline stage to start from.

### Pipeline Stages

The pipeline is divided into the following sequential stages:
1. **Frame Extraction:** Extracts raw frames from the input video.
2. **Image Inversion:** Inverts the frame colors to optimize tracking contrast.
3. **Masking and Cropping:** Allows you to interactively define a circular mask to isolate the region of interest.
4. **Topographical Generation:** Generates Topographical Maps to calculate spatial complexity metrics.
5. **Final Tracking (MOG2):** Uses OpenCV's MOG2 background subtractor combined with topographical gating (Sharpness Index and Area filtering) to generate the subject's tracking data.
6. **Tracking Refinement:** An optional secondary pass to refine coordinates based on localized search parameters.

Outputs, including the CSV of coordinate positions and a trajectory plot, will be saved to the `data/output/` folder.
