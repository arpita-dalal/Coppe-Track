import os
import logging
from datetime import datetime

# Import modules from the src directory
from src import video_loader
from src import image_processor
from src import tracking_engine

def setup_logging(log_dir):
    """Sets up recording logs to a file."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"pipeline_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info("Pipeline initialized (Simplified Mode).")

def run_coppe_track_pipeline(video_filename, start_step=1):
    """
    Orchestrates the simplified Coppe-Track workflow.
    Stages: 1=Extraction, 2=Inversion, 3=Masking, 4=Refinement, 5=Tracking
    """
    # Define project root and paths
    project_src = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(project_src)
    input_path = os.path.join(project_root, "data", "input", video_filename)
    cache_dir = os.path.join(project_root, "data", "cache")
    output_dir = os.path.join(project_root, "data", "output")
    log_dir = os.path.join(project_root, "logs")

    setup_logging(log_dir)

    video_name = os.path.splitext(video_filename)[0]

    if not os.path.exists(input_path):
        logging.error(f"Input video not found: {input_path}")
        return

    # Sub-directories in cache
    frames_dir = os.path.join(cache_dir, "01_raw_frames")
    inverted_dir = os.path.join(cache_dir, "02_inverted_frames")
    masked_dir = os.path.join(cache_dir, "03_masked_frames")
    topo_full_dir = os.path.join(cache_dir, "04_topo_full")
    topo_refined_dir = os.path.join(project_root, "Topo refined frames")

    # Step 1: Frame Extraction
    if start_step <= 1:
        logging.info("Step 1: Extracting frames...")
        if os.path.exists(frames_dir):
            import shutil
            shutil.rmtree(frames_dir)
        video_loader.extract_frames(input_path, frames_dir)

    # Step 2: Image Inversion
    if start_step <= 2:
        logging.info("Step 2: Inverting images...")
        image_processor.invert_images(frames_dir, inverted_dir)

    # Step 3: Masking and Cropping
    if start_step <= 3:
        logging.info("Step 3: Defining circular mask...")
        sample_frame = os.path.join(inverted_dir, sorted(os.listdir(inverted_dir))[0])
        center, radius = image_processor.select_circle_interactively(sample_frame)
        
        if center is None or radius is None:
            logging.error("Mask selection failed. Aborting.")
            return

        logging.info(f"Applying circular mask with center {center} and radius {radius:.2f}...")
        image_processor.apply_circular_mask(inverted_dir, masked_dir, center, radius)

    # Step 4: Topographical Generation (Pass 0)
    if start_step <= 4:
        logging.info("Step 4: Generating Topographical Maps (Full & Refined)...")
        os.makedirs(topo_full_dir, exist_ok=True)
        os.makedirs(topo_refined_dir, exist_ok=True)

        files = sorted([f for f in os.listdir(masked_dir) if f.lower().endswith(('.jpg', '.jpeg'))])
        if not files:
            logging.warning("No masked frames found for Stage 4.")
            return
        
        topo_csv = os.path.join(output_dir, f"{video_name}_topo_metrics.csv")
        
        tracking_engine.generate_topography_pass(files, masked_dir, topo_full_dir, topo_refined_dir, output_csv=topo_csv)

    # Step 5: Final Tracking (Simplified MOG2 logic)
    if start_step <= 5:
        logging.info("Step 5: MOG2 Tracking & Kinematic Analysis...")
        output_csv = os.path.join(output_dir, f"{video_name}_positions.csv")
        output_plot = os.path.join(output_dir, f"{video_name}_plot.png")

        # Gating Mode Selection
        print("\nSelect Topographical Gating Mode:")
        print("  1. Sharpness Index (TSI) Only")
        print("  2. Refined Area Only")
        print("  3. Both (Recommended)")
        g_choice = input("Choice [1-3, default 3]: ").strip()
        g_mode = { "1": "tsi", "2": "area", "3": "both" }.get(g_choice, "both")

        # Optional Threshold Customization
        g_thresholds = {"tsi_low": -0.7, "tsi_high": -0.35, "area_low": 2.47, "vol_limit": 0.1}
        custom = input("\nCustomize thresholds? (y/n, default n): ").strip().lower()
        if custom == 'y':
            try:
                t_low = input(f"  LogTSI Lower Limit (default {g_thresholds['tsi_low']}): ").strip()
                if t_low: g_thresholds['tsi_low'] = float(t_low)
                t_high = input(f"  LogTSI Upper Limit (default {g_thresholds['tsi_high']}): ").strip()
                if t_high: g_thresholds['tsi_high'] = float(t_high)
                a_low = input(f"  LogArea Lower Limit (default {g_thresholds['area_low']}): ").strip()
                if a_low: g_thresholds['area_low'] = float(a_low)
                v_lim = input(f"  Volatility Jump Limit (default {g_thresholds['vol_limit']}): ").strip()
                if v_lim: g_thresholds['vol_limit'] = float(v_lim)
            except ValueError:
                print("Invalid input. Using defaults.")

        # Call the simplified tracking engine pointing to topo isolates
        topo_csv = os.path.join(output_dir, f"{video_name}_topo_metrics.csv")
        tracking_engine.track_copepod_mog2(topo_refined_dir, output_csv, output_plot, masked_dir=masked_dir, frame_interval=1, topo_metrics_csv=topo_csv, gating_mode=g_mode, thresholds=g_thresholds)

    # Step 6: Tracking Refinement (Pass 2)
    if start_step == 6:
        logging.info("Step 6: Tracking Refinement (Pass 2 - Localized Search)...")
        video_name = os.path.splitext(video_filename)[0]
        input_csv = os.path.join(output_dir, f"{video_name}_positions.csv")
        output_csv = os.path.join(output_dir, f"{video_name}_refined_positions.csv")
        output_plot = os.path.join(output_dir, f"{video_name}_refined_plot.png")
        
        topo_csv = os.path.join(output_dir, f"{video_name}_topo_metrics.csv")
        if os.path.exists(input_csv):
            # Reuse g_mode/thresholds if defined, else ask
            if 'g_mode' not in locals():
                print("\nSelect Topographical Gating Mode for Refinement:")
                print("  1. Sharpness (TSI), 2. Area, 3. Both")
                g_choice = input("Choice [1-3, default 3]: ").strip()
                g_mode = { "1": "tsi", "2": "area", "3": "both" }.get(g_choice, "both")
                g_thresholds = {"tsi_low": -0.7, "tsi_high": -0.35, "area_low": 2.47, "vol_limit": 0.1}
                custom = input("\nCustomize thresholds? (y/n, default n): ").strip().lower()
                if custom == 'y':
                    try:
                        t_low = input(f"  LogTSI Lower Limit (default {g_thresholds['tsi_low']}): ").strip()
                        if t_low: g_thresholds['tsi_low'] = float(t_low)
                        t_high = input(f"  LogTSI Upper Limit (default {g_thresholds['tsi_high']}): ").strip()
                        if t_high: g_thresholds['tsi_high'] = float(t_high)
                        a_low = input(f"  LogArea Lower Limit (default {g_thresholds['area_low']}): ").strip()
                        if a_low: g_thresholds['area_low'] = float(a_low)
                        v_lim = input(f"  Volatility Jump Limit (default {g_thresholds['vol_limit']}): ").strip()
                        if v_lim: g_thresholds['vol_limit'] = float(v_lim)
                    except ValueError:
                        print("Invalid input. Using defaults.")

            tracking_engine.track_copepod_refined(topo_refined_dir, input_csv, output_csv, output_plot, masked_dir=masked_dir, frame_interval=1, topo_metrics_csv=topo_csv, gating_mode=g_mode, thresholds=g_thresholds)
        else:
            logging.error(f"Cannot perform refinement: {input_csv} not found.")

    logging.info(f"Simplified Pipeline finished successfully for {video_filename}")

if __name__ == "__main__":
    target_video = "T10_R3.mov"
    run_coppe_track_pipeline(target_video, start_step=1)
