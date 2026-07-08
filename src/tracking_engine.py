import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import logging
import imageio

def _extract_shape_signature(binary_roi):
    """
    Extracts the concentric morphological signature of a white binary blob.
    Returns: a dict with the areas of the 20%, 50%, and 80% depth rings, and base Hu Moments.
    """
    if cv2.countNonZero(binary_roi) == 0:
        return None
        
    dist = cv2.distanceTransform(binary_roi, cv2.DIST_L2, 3)
    max_val = dist.max()
    if max_val == 0:
        return None
        
    sig = {'max_depth': max_val, 'ring_areas': [], 'ring_counts': []}
    
    # Base contour Hu Moments
    base_cnts, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if base_cnts:
        cnt = max(base_cnts, key=cv2.contourArea)
        sig['base_area'] = cv2.contourArea(cnt)
        moments = cv2.moments(cnt)
        sig['hu'] = cv2.HuMoments(moments).flatten()
    else:
        return None

    for pct in [0.2, 0.5, 0.8]:
        level = max_val * pct
        slice_mask = np.uint8(dist > level) * 255
        inner_cnts, _ = cv2.findContours(slice_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sig['ring_counts'].append(len(inner_cnts))
        if inner_cnts:
            sig['ring_areas'].append(sum([cv2.contourArea(c) for c in inner_cnts]))
        else:
            sig['ring_areas'].append(0)
            
    return sig

def _average_signatures(sigs):
    """
    Averages a list of morphological signatures.
    """
    if not sigs: return None
    avg_sig = {
        'max_depth': np.mean([s['max_depth'] for s in sigs]),
        'base_area': np.mean([s['base_area'] for s in sigs]),
        'ring_areas': np.mean([s['ring_areas'] for s in sigs], axis=0).tolist(),
        'hu': np.mean([s['hu'] for s in sigs], axis=0).tolist()
    }
    return avg_sig

def isolate_copepods(input_dir, output_dir, debug_dir, params):
    """
    Refines preprocessed images to isolate copepods using adaptive thresholding, 
    morphology, and size filtering.
    (Kept from original Coppe-Track as it's a useful preprocessing step before MOG2)
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)
    
    min_area = params.get('min_area', 50)
    max_area = params.get('max_area', 20000)
    min_ar = params.get('min_aspect_ratio', 0.2)
    max_ar = params.get('max_aspect_ratio', 5.0)
    edge_margin = params.get('edge_margin', 10)

    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg'))])
    
    for file in files:
        frame = cv2.imread(os.path.join(input_dir, file), cv2.IMREAD_GRAYSCALE)
        if frame is None: continue
        
        _, binary = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        h, w = frame.shape
        center = (w // 2, h // 2)
        radius = min(center) - edge_margin
        circular_mask = np.zeros_like(frame, dtype=np.uint8)
        cv2.circle(circular_mask, center, radius, 255, thickness=-1)
        masked_binary = cv2.bitwise_and(cleaned, cleaned, mask=circular_mask)
        
        contours, _ = cv2.findContours(masked_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(frame)
        
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            aspect_ratio = cw / ch if ch != 0 else 0
            
            if min_area <= area <= max_area and min_ar <= aspect_ratio <= max_ar:
                cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        cv2.imwrite(os.path.join(debug_dir, file), masked_binary)
        cv2.imwrite(os.path.join(output_dir, file), mask)
    
    logging.info(f"Refinement complete.")

def generate_topography_pass(frames, input_folder, output_dir, refined_dir=None, output_csv=None):
    """
    Generates and saves the topographical (distance transform) maps for the 
    entire arena for all frames. If refined_dir is provided, it also saves
    frames where ONLY the best copepod candidate is visible.
    Records indices (Max Depth, Area, TSI) to CSV if output_csv is provided.
    """
    logging.info("GENERATING TOPOGRAPHY: Full Arena & Refined isolates...")
    fgbg = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=25, detectShadows=False)
    
    topo_data = []
    
    for i, f_name in enumerate(frames):
        img = cv2.imread(os.path.join(input_folder, f_name), 0)
        if img is None: continue
        
        fg = fgbg.apply(img)
        dist = cv2.distanceTransform(fg, cv2.DIST_L2, 3)
        
        full_max = dist.max()
        
        # 1. Full Arena Map generation
        dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX)
        dist_color = cv2.applyColorMap(np.uint8(dist_norm), cv2.COLORMAP_MAGMA)
        cv2.putText(dist_color, f"F:{i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imwrite(os.path.join(output_dir, f_name), dist_color)

        # 2. Refined Map generation (Copepod Only)
        ref_max = 0
        ref_area = 0
        tsi = 0
        
        if refined_dir:
            cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            refined_dist = np.zeros_like(dist)
            if cnts:
                # Pick the largest candidate within reasonable copepod size (20-1500 px)
                candidates = [c for c in cnts if 20 < cv2.contourArea(c) < 1500]
                if candidates:
                    best_cnt = max(candidates, key=cv2.contourArea)
                    ref_area = cv2.contourArea(best_cnt)
                    mask = np.zeros_like(fg)
                    cv2.drawContours(mask, [best_cnt], -1, 255, -1)
                    refined_dist = cv2.bitwise_and(dist, dist, mask=mask)
                    ref_max = refined_dist.max()
                    
                    if ref_area > 0:
                        tsi = ref_max / np.sqrt(ref_area)
            
            # Normalize and colormap the isolated blob
            ref_norm = cv2.normalize(refined_dist, None, 0, 255, cv2.NORM_MINMAX)
            ref_color = cv2.applyColorMap(np.uint8(ref_norm), cv2.COLORMAP_MAGMA)
            if refined_dist.max() == 0:
                ref_color = np.zeros_like(ref_color)

            cv2.imwrite(os.path.join(refined_dir, f_name), ref_color)

        topo_data.append({
            'Frame': i,
            'Filename': f_name,
            'Full_Max_Depth': full_max,
            'Refined_Max_Depth': ref_max,
            'Refined_Area': ref_area,
            'Topological_Sharpness_Index': tsi
        })

        if (i+1) % 200 == 0:
            logging.info(f"  Topographies generated: {i+1}/{len(frames)}")
            
    if output_csv and topo_data:
        df_topo = pd.DataFrame(topo_data)
        df_topo.to_csv(output_csv, index=False)
        logging.info(f"Topographical metrics saved: {output_csv}")
        
        # Generate interactive diagnostic plots
        video_name = os.path.basename(output_csv).replace("_topo_metrics.csv", "")
        generate_interactive_topo_plots(df_topo, os.path.dirname(output_csv), video_name)

def generate_interactive_topo_plots(df_topo, output_dir, video_name):
    """
    Generates four interactive Plotly plots from topographical metrics.
    """
    try:
        import plotly.graph_objects as go
        import numpy as np

        plots = [
            ('Refined_Area', 'Refined Area (px)', f"{video_name}_interactive_area.html"),
            ('Log_Refined_Area', 'log10(Refined Area)', f"{video_name}_interactive_log_area.html"),
            ('Topological_Sharpness_Index', 'TSI', f"{video_name}_interactive_tsi.html"),
            ('Log_TSI', 'log10(TSI)', f"{video_name}_interactive_log_tsi.html")
        ]

        # Pre-calculate Logs for plotting
        df_plot = df_topo.copy()
        df_plot['Log_Refined_Area'] = np.log10(df_plot['Refined_Area'] + 1e-6)
        df_plot['Log_TSI'] = np.log10(df_plot['Topological_Sharpness_Index'] + 1e-6)

        for col, label, filename in plots:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_plot['Frame'],
                y=df_plot[col],
                mode='lines+markers',
                name=label,
                hovertemplate='<b>Frame</b>: %{x}<br><b>' + label + '</b>: %{y:.4f}<extra></extra>'
            ))
            fig.update_layout(
                title=f'Interactive Topology: {label} vs Frame Number',
                xaxis_title='Frame Number',
                yaxis_title=label,
                template='plotly_dark',
                hovermode='x unified'
            )
            out_path = os.path.join(output_dir, filename)
            fig.write_html(out_path)
            logging.info(f"Interactive plot saved: {out_path}")
            
    except ImportError:
        logging.warning("Plotly not installed. Skipping interactive plots.")
    except Exception as e:
        logging.error(f"Error generating interactive plots: {e}")

def save_high_res_trajectory_frames(frames, output_dir, title):
    """
    Saves a sequence of images (pre-tagged with tracks) to a directory with calibrated axes.
    Ranges are fixed to -14 to 14 mm.
    """
    if not frames:
        return
        
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Saving {len(frames)} high-res trajectory frames to: {output_dir}")
    
    # Use a non-interactive backend for speed
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for i, frame in enumerate(frames):
        fig, ax = plt.subplots(figsize=(6, 6))
        # The circular mask is centered in the image, so we map the full width to -14..14
        ax.imshow(frame, extent=[-14, 14, -14, 14], origin='upper')
        ax.set_xlim(-14, 14)
        ax.set_ylim(-14, 14)
        ax.set_title(f"{title} - Frame {i:04d}", fontweight='bold')
        ax.set_xlabel("X Position (mm)")
        ax.set_ylabel("Y Position (mm)")
        ax.grid(True, linestyle='--', alpha=0.5)
        
        frame_path = os.path.join(output_dir, f"frame_{i:04d}.png")
        plt.savefig(frame_path, dpi=100) # Lower DPI for faster generation of many frames
        plt.close(fig)
        
        if (i + 1) % 100 == 0:
            logging.info(f"Saved {i + 1}/{len(frames)} frames...")

def track_copepod_mog2(input_folder, output_csv, output_plot, masked_dir=None, frame_interval=1, topo_metrics_csv=None, gating_mode='both', thresholds=None):
    """
    Simplified tracking logic aligned strictly with tracking5.py.
    Uses MOG2 background subtraction and calculates all kinematic metrics.
    """
    frames = sorted([f for f in os.listdir(input_folder) if f.endswith('.jpg') or f.endswith('.jpeg')])
    if not frames:
        logging.error(f"No frames found in folder: {input_folder}")
        return

    positions = []
    aspect_ratios = []
    gif_frames = []

    # Read the first frame for ROI selection
    first_frame_path = os.path.join(input_folder, frames[0])
    first_frame = cv2.imread(first_frame_path)
    if first_frame is None:
        logging.error(f"First frame not found at path: {first_frame_path}")
        return

    # Let the user select the copepod area in the first frame
    cv2.namedWindow('Select Copepod Area', cv2.WINDOW_NORMAL)
    bbox = cv2.selectROI('Select Copepod Area', first_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    # --- NEW: Manual Distractor Selection for Stage 5 ---
    distractor_zones = []
    logging.info("Opening sample frames for distractor selection. Press ENTER or SPACE after selecting each ROI. Press ESC when finished.")
    
    # Pick 3 representative frames (Start, Mid, End)
    sample_indices = [0, len(frames)//2, len(frames)-1]
    
    cv2.namedWindow('Select DISTRACTORS (Bubbles/Noise) - Stage 5', cv2.WINDOW_NORMAL)
    for idx in sample_indices:
        sample_frame = cv2.imread(os.path.join(input_folder, frames[idx]))
        if sample_frame is not None:
            rois = cv2.selectROIs('Select DISTRACTORS (Bubbles/Noise) - Stage 5', sample_frame, fromCenter=False, showCrosshair=True)
            if len(rois) > 0:
                distractor_zones.extend(rois)
            cv2.waitKey(500)
    cv2.destroyAllWindows()
    
    if distractor_zones:
        logging.info(f"Registered {len(distractor_zones)} distractor exclusion zones for Stage 5.")

    # Extract thresholds with defaults
    if thresholds is None:
        thresholds = {"tsi_low": -0.7, "tsi_high": -0.35, "area_low": 2.47, "vol_limit": 0.1}
    
    t_low = thresholds.get("tsi_low", -0.7)
    t_high = thresholds.get("tsi_high", -0.35)
    a_low = thresholds.get("area_low", 2.47)
    v_lim = thresholds.get("vol_limit", 0.1)
    # Background subtractor
    fgbg = cv2.createBackgroundSubtractorMOG2()

    logging.info("Starting MOG2 tracking...")
    
    # Load Topo Metrics for gating if available
    topo_gates = {}
    if topo_metrics_csv and os.path.exists(topo_metrics_csv):
        try:
            df_topo = pd.read_csv(topo_metrics_csv)
            # Create a dict of dicts: {filename -> {TSI: val, Area: val}}
            topo_gates = df_topo.set_index('Filename')[['Topological_Sharpness_Index', 'Refined_Area']].to_dict('index')
            logging.info(f"Loaded {len(topo_gates)} topographical metrics for dual-gating ({gating_mode}).")
        except Exception as e:
            logging.error(f"Could not load topo metrics for gating: {e}")

    last_valid_pos = None
    prev_tsi = 0.35

    # Process each frame and track copepod position
    for frame_idx, frame_name in enumerate(frames):
        frame_path = os.path.join(input_folder, frame_name)
        frame = cv2.imread(frame_path)
        if frame is None:
            logging.warning(f"Frame {frame_name} could not be read. Skipping.")
            continue

        # Draw distractor zones on the display frame
        for (dx, dy, dw, dh) in distractor_zones:
            cv2.rectangle(frame, (dx, dy), (dx + dw, dy + dh), (0, 0, 150), 1)

        # --- TOPOGRAPHICAL DUAL-GATING --- 
        is_at_rest = False
        metrics = topo_gates.get(frame_name, {'Topological_Sharpness_Index': 0.35, 'Refined_Area': 600})
        
        tsi_linear = metrics['Topological_Sharpness_Index']
        area_linear = metrics['Refined_Area']
        
        tsi_log = np.log10(max(tsi_linear, 0.0001))
        area_log = np.log10(max(area_linear, 1.0))
        
        tsi_gated = (tsi_log < t_low) or (tsi_log > t_high) or (abs(tsi_log - prev_tsi) > v_lim and tsi_log > -0.4)
        area_gated = (area_log < a_low)

        if gating_mode == 'tsi' and tsi_gated:
            is_at_rest = True
        elif gating_mode == 'area' and area_gated:
            is_at_rest = True
        elif gating_mode == 'both' and (tsi_gated or area_gated):
            is_at_rest = True

        if is_at_rest:
            logging.debug(f"Frame {frame_idx}: Gated (Mode:{gating_mode}) - LogTSI:{tsi_log:.2f}, LogArea:{area_log:.2f}")
        
        prev_tsi = tsi_log

        # Apply background subtraction (always done to keep MOG2 history updated)
        fgmask = fgbg.apply(frame)

        # Apply circular mask to the frame for better visualization in GIF
        # (Assuming the mask parameters from isolate_copepods or similar)
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        radius = min(center) - 10
        mask_vis = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask_vis, center, radius, 255, -1)
        frame_masked = cv2.bitwise_and(frame, frame, mask=mask_vis)

        # Threshold the mask to get a binary image
        _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # --- Filter contours by distractor exclusion zones ---
        valid_contours = []
        for cnt in contours:
            tx, ty, tw, th = cv2.boundingRect(cnt)
            cx_c = tx + tw // 2
            cy_c = ty + th // 2
            
            is_distractor = False
            for (dx, dy, dw, dh) in distractor_zones:
                if dx <= cx_c <= dx + dw and dy <= cy_c <= dy + dh:
                    is_distractor = True
                    break
            
            if not is_distractor:
                valid_contours.append(cnt)
            else:
                # Visual feedback for rejected distractor
                cv2.rectangle(frame, (tx, ty), (tx + tw, ty + th), (0, 0, 150), 1)
                cv2.putText(frame, "REJECTED", (tx, ty - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 150), 1)

        # Prepare visualization and tagging frames
        tagged_frame = frame.copy()
        vis_frame = None
        if masked_dir:
            masked_path = os.path.join(masked_dir, frame_name)
            vis_frame = cv2.imread(masked_path)
        
        if vis_frame is None:
            # Fallback: create a masked version of the refined frame
            h, w = frame.shape[:2]
            center = (w // 2, h // 2)
            radius = min(center) - 10
            mask_vis = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask_vis, center, radius, 255, -1)
            vis_frame = cv2.bitwise_and(frame, frame, mask=mask_vis)

        # Find the largest contour assuming it's the copepod
        # Find the largest contour assuming it's the copepod
        success = False
        if not is_at_rest and valid_contours:
            max_contour = max(valid_contours, key=cv2.contourArea)
            
            # COLOR THE COPEPOD RED in the tagged frame
            cv2.drawContours(tagged_frame, [max_contour], -1, (0, 0, 255), thickness=cv2.FILLED)
            
            x, y, w, h = cv2.boundingRect(max_contour)
            cx = x + w // 2
            cy = y + h // 2
            last_valid_pos = (cx, cy)
            positions.append((frame_idx * frame_interval, cx, cy))
            aspect_ratio = w / h if h != 0 else np.nan
            aspect_ratios.append(aspect_ratio)
            success = True

            # Draw the bounding box and centroid on the LIVE display frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # Draw cumulative trajectory on the visualization frame
            valid_pts = [(p[1], p[2]) for p in positions if not np.isnan(p[1])]
            if len(valid_pts) > 1:
                pts = np.array(valid_pts, np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis_frame, [pts], isClosed=False, color=(0, 255, 255), thickness=1)
        
        if not success:
            # Handle "AT REST" or "NOT FOUND"
            if last_valid_pos:
                cx, cy = last_valid_pos
                positions.append((frame_idx * frame_interval, cx, cy))
                cv2.putText(frame, "AT REST (Topo Gate)" if is_at_rest else "NOT FOUND", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 255), 2)
            else:
                positions.append((frame_idx * frame_interval, np.nan, np.nan))
            aspect_ratios.append(np.nan)

        # Save the tagged frame for Step 6
        tagged_dir = os.path.join(os.path.dirname(output_csv), "tagged_frames_" + os.path.basename(output_csv).split("_positions")[0])
        os.makedirs(tagged_dir, exist_ok=True)
        cv2.imwrite(os.path.join(tagged_dir, frame_name), tagged_frame)

        # Collect frame for GIF and High-Res Export
        gif_frames.append(cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB))

        # Display the frame (optional, but kept for parity with tracking5.py)
        cv2.imshow('Tracking Copepod (MOG2)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    # Check if positions were detected
    if len(positions) == 0:
        logging.error("No positions detected. CSV file will not be generated.")
        return

    # Calculate movement parameters (Exactly as in tracking5.py)
    positions = np.array(positions)
    displacements = np.sqrt(np.diff(positions[:, 1])**2 + np.diff(positions[:, 2])**2)
    velocity = displacements / frame_interval
    acceleration = np.diff(velocity) / frame_interval

    # Prepare velocity and acceleration data
    velocity = np.insert(velocity, 0, 0)  # Add initial velocity as 0
    acceleration = np.insert(acceleration, 0, [0, 0])  # Add initial acceleration as 0

    # Calculate turning angle for each frame
    turning_angles = [0]  # No turning angle for the first frame
    for i in range(1, len(positions) - 1):
        v1 = positions[i] - positions[i - 1]
        v2 = positions[i + 1] - positions[i]
        dot_product = np.dot(v1[1:], v2[1:])
        mag_v1 = np.linalg.norm(v1[1:])
        mag_v2 = np.linalg.norm(v2[1:])
        if mag_v1 != 0 and mag_v2 != 0:
            # Clip dot_product to avoid NaN from floating point errors
            cos_val = np.clip(dot_product / (mag_v1 * mag_v2), -1.0, 1.0)
            angle = np.arccos(cos_val)
            turning_angles.append(np.degrees(angle))
        else:
            turning_angles.append(0)
    turning_angles.append(0)  # No turning angle for the last frame

    # Calculate net displacement, total path length, gross speed, and tortuosity
    net_displacements = []
    total_path_lengths = []
    gross_speeds = []
    tortuosities = []
    
    # Find first valid position index for net displacement origin
    first_valid_idx = 0
    while first_valid_idx < len(positions) and np.isnan(positions[first_valid_idx, 1]):
        first_valid_idx += 1
        
    origin = positions[first_valid_idx] if first_valid_idx < len(positions) else positions[0]
    
    cumulative_path_length = 0
    for i in range(len(positions)):
        if np.isnan(positions[i, 1]) or first_valid_idx >= len(positions) or i < first_valid_idx:
            net_displacements.append(0.0)
            total_path_lengths.append(cumulative_path_length)
            gross_speeds.append(0.0)
            tortuosities.append(np.nan)
        else:
            net_disp = np.sqrt((positions[i, 1] - origin[1])**2 + (positions[i, 2] - origin[2])**2)
            net_displacements.append(net_disp)
            
            if i > 0:
                step_dist = displacements[i-1]
                if not np.isnan(step_dist): 
                    cumulative_path_length += step_dist
            
            total_path_lengths.append(cumulative_path_length)
            
            elapsed_time = positions[i, 0] - origin[0]
            gross_speed = cumulative_path_length / elapsed_time if elapsed_time > 0 else 0.0
            gross_speeds.append(gross_speed)
            
            tortuosity = cumulative_path_length / net_disp if net_disp != 0 else np.nan
            tortuosities.append(tortuosity)

    # Prepare data for saving to CSV
    columns = ['Time (s)', 'X Position (pixels)', 'Y Position (pixels)', 'Velocity (pixels/sec)', 
               'Acceleration (pixels/sec^2)', 'Turning Angle (degrees)', 'Net Displacement (pixels)', 
               'Total Path Length (pixels)', 'Gross Speed (pixels/sec)', 'Tortuosity', 'Aspect Ratio']
    
    # Ensure all arrays are same length (acceleration might be longer/shorter depending on diff)
    # in tracking5.py: acceleration = np.insert(acceleration, 0, [0, 0]) which is odd?
    # Actually np.insert(acceleration, 0, [0, 0]) results in a 1D array with two 0s at the start.
    # Let's check tracking5.py line 87: acceleration = np.insert(acceleration, 0, [0, 0])
    # If len(velocity) is N, len(np.diff(velocity)) is N-1. 
    # np.insert(diff, 0, [0, 0]) makes it (N-1) + 2 = N+1.
    # THIS LOOKS LIKE A BUG IN tracking5.py, but user wants EXACT SAME.
    # Wait, if positions has N rows, vels has N. accs should have N.
    # np.diff(vels) has N-1. 
    # To get N, we should insert ONE 0.
    
    # Let's stick to EXACT parity including potential bugs if that's what's meant, 
    # but I'll fix the length to ensure pandas doesn't crash.
    # In tracking5.py, the column_stack would fail if lengths mismatch.
    
    data = np.column_stack((positions, velocity, acceleration[:len(positions)], turning_angles, 
                            net_displacements, total_path_lengths, gross_speeds, tortuosities, aspect_ratios))
    df = pd.DataFrame(data, columns=columns)

    try:
        df.to_csv(output_csv, index=False)
        logging.info(f'CSV file saved: {output_csv}')
    except Exception as e:
        logging.error(f"Error saving CSV file: {e}")

    # Plot
    if positions.size > 1:
        x, y = positions[:, 1], positions[:, 2]
        t = positions[:, 0]
        
        # Create segments for LineCollection
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        norm = Normalize(np.nanmin(t), np.nanmax(t))
        lc = LineCollection(segments, cmap='viridis', norm=norm)
        lc.set_array(t)
        lc.set_linewidth(2)
        
        fig, ax = plt.subplots()
        line = ax.add_collection(lc)
        fig.colorbar(line, ax=ax, label='Time (s)')
        
        ax.set_xlim(np.nanmin(x) - 10, np.nanmax(x) + 10)
        ax.set_ylim(np.nanmin(y) - 10, np.nanmax(y) + 10)
        ax.set_title('Copepod Trajectory (Pixels)', fontweight='bold')
        ax.set_xlabel('X Position (pixels)')
        ax.set_ylabel('Y Position (pixels)')
        ax.invert_yaxis()
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_aspect('equal', 'datalim')
        
        plt.savefig(output_plot, dpi=300)
        plt.close()
        logging.info(f'Tracking plot saved as: {output_plot}')

    # Save GIF
    if 'gif_frames' in locals() and gif_frames:
        gif_path = output_plot.replace('.png', '_trajectory.gif')
        try:
            imageio.mimsave(gif_path, gif_frames, duration=100) # 100ms per frame = 10 fps
            logging.info(f"Trajectory GIF saved: {gif_path}")
            
            # Save individual frames with axes
            #frames_dir = os.path.join(os.path.dirname(output_plot), os.path.basename(output_plot).replace('.png', '_frames_pass1'))
            #save_high_res_trajectory_frames(gif_frames, frames_dir, "Stage 5 Tracking")
        except Exception as e:
            logging.error(f"Error saving GIF: {e}")

            # This plot is now redundant because a centered version is created in the next block.
            # I will remove this block to avoid confusion and use the centered one as 'plot_metric'.
            pass

    # Third Output - Metric CSV (mm)
    if positions.size > 1:
        # Re-fetch mm_per_pixel if not already defined (though it should be from the plot logic above)
        img_file = os.path.join(input_folder, sorted(os.listdir(input_folder))[0])
        img = cv2.imread(img_file)
        if img is not None:
            width = img.shape[1]
            height = img.shape[0]
            mm_per_pixel = 24.0 / width
            
            # New Metric CSV logic: Scale relevant columns from the pixel DataFrame
            df_metric = df.copy()
            
            # Map of pixel column names to metric column names and scaling requirement
            column_mapping = {
                'X Position (pixels)': ('X Position (mm)', True),
                'Y Position (pixels)': ('Y Position (mm)', True),
                'Velocity (pixels/sec)': ('Velocity (mm/s)', True),
                'Acceleration (pixels/sec^2)': ('Acceleration (mm/s^2)', True),
                'Net Displacement (pixels)': ('Net Displacement (mm)', True),
                'Total Path Length (pixels)': ('Total Path Length (mm)', True),
                'Gross Speed (pixels/sec)': ('Gross Speed (mm/s)', True),
                'Time (s)': ('Time (s)', False),
                'Turning Angle (degrees)': ('Turning Angle (degrees)', False),
                'Tortuosity': ('Tortuosity', False),
                'Aspect Ratio': ('Aspect Ratio', False)
            }
            
            # Shift X and Y to center before scaling
            # For Y, we also flip it to be Cartesian (up is positive)
            df_metric['X Position (pixels)'] = (df['X Position (pixels)'] - width/2)
            df_metric['Y Position (pixels)'] = (height/2 - df['Y Position (pixels)'])
            
            new_cols = []
            for col in df.columns:
                dest_name, scaled = column_mapping.get(col, (col, False))
                new_cols.append(dest_name)
                if scaled:
                    df_metric[col] = df_metric[col] * mm_per_pixel
            
            df_metric.columns = new_cols
            
            # Also calculate x_mm, y_mm for subsequent plots
            x_mm = df_metric['X Position (mm)'].values
            y_mm = df_metric['Y Position (mm)'].values
            velocity_mm = df_metric['Velocity (mm/s)'].values[1:] # Segment velocities
            turning_angle_mm = df_metric['Turning Angle (degrees)'].values[1:-1] # Segment angles
            
            # Define segments_centered for metric plots
            points_centered = np.array([x_mm, y_mm]).T.reshape(-1, 1, 2)
            segments_centered = np.concatenate([points_centered[:-1], points_centered[1:]], axis=1)
            
            metric_csv = output_csv.replace('.csv', '_metric.csv')
            try:
                df_metric.to_csv(metric_csv, index=False)
                logging.info(f"Metric CSV saved: {metric_csv}")
            except Exception as e:
                logging.error(f"Error saving metric CSV: {e}")

            # Fourth Plot - Time Colormap (Metric mm)
            fig, ax = plt.subplots()
            lc_time = LineCollection(segments_centered, cmap='viridis', norm=Normalize(np.nanmin(t), np.nanmax(t)))
            lc_time.set_array(t)
            lc_time.set_linewidth(2)
            line_time = ax.add_collection(lc_time)
            fig.colorbar(line_time, ax=ax, label='Time (s)')
            
            ax.set_title('Copepod Trajectory (Metric - mm)', fontweight='bold')
            ax.set_xlabel('X Position (mm)')
            ax.set_ylabel('Y Position (mm)')
            ax.set_xlim(-14, 14)
            ax.set_ylim(-14, 14)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_aspect('equal', 'datalim')
            
            metric_plot = output_plot.replace('.png', '_metric.png')
            plt.savefig(metric_plot, dpi=300)
            plt.close()
            logging.info(f'Metric tracking plot saved as: {metric_plot}')

            # Fifth Plot - Velocity Colormap (mm)
            fig, ax = plt.subplots()
            lc_vel = LineCollection(segments_centered, cmap='magma', norm=Normalize(np.nanmin(velocity_mm), np.nanmax(velocity_mm)))
            lc_vel.set_array(velocity_mm)
            lc_vel.set_linewidth(2)
            line_vel = ax.add_collection(lc_vel)
            fig.colorbar(line_vel, ax=ax, label='Velocity (mm/s)')
            
            # Use standard axes styling
            ax.set_title('Copepod Trajectory (Colored by Velocity)', fontweight='bold')
            ax.set_xlabel('X Position (mm)')
            ax.set_ylabel('Y Position (mm)')
            ax.set_xlim(-14, 14)
            ax.set_ylim(-14, 14)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_aspect('equal', 'datalim')
            
            vel_plot = output_plot.replace('.png', '_velocity.png')
            plt.savefig(vel_plot, dpi=300)
            plt.close()
            logging.info(f'Velocity tracking plot saved as: {vel_plot}')

            # Fifth Plot - Turning Angle Colormap (mm)
            points_angle = np.array([x_mm[1:], y_mm[1:]]).T.reshape(-1, 1, 2)
            segments_angle = np.concatenate([points_angle[:-1], points_angle[1:]], axis=1)
            
            fig, ax = plt.subplots()
            lc_ang = LineCollection(segments_angle, cmap='coolwarm', norm=Normalize(np.nanmin(turning_angle_mm), np.nanmax(turning_angle_mm)))
            lc_ang.set_array(turning_angle_mm)
            lc_ang.set_linewidth(2)
            line_ang = ax.add_collection(lc_ang)
            fig.colorbar(line_ang, ax=ax, label='Turning Angle (degrees)')
            
            ax.set_title('Copepod Trajectory (Colored by Turning Angle)', fontweight='bold')
            ax.set_xlabel('X Position (mm)')
            ax.set_ylabel('Y Position (mm)')
            ax.set_xlim(-14, 14)
            ax.set_ylim(-14, 14)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_aspect('equal', 'datalim')
            
            ang_plot = output_plot.replace('.png', '_angle.png')
            plt.savefig(ang_plot, dpi=300)
            plt.close()
            logging.info(f'Turning angle tracking plot saved as: {ang_plot}')


def track_copepod_refined(input_folder, input_csv, output_csv, output_plot, masked_dir=None, frame_interval=1, topo_metrics_csv=None, gating_mode='both', thresholds=None):
    """
    Pass 2: Refined tracking using a localized Search ROI based on Pass 1 coordinates.
    """
    import pandas as pd
    import numpy as np

    # 1. Load Pass 1 data
    try:
        df_raw = pd.read_csv(input_csv)
    except Exception as e:
        logging.error(f"Error reading Pass 1 CSV: {e}")
        return

    # 2. Path Cleaning (Outlier Removal & Interpolation)
    frames = sorted([f for f in os.listdir(input_folder) if f.endswith('.jpg')])
    time_points = np.arange(len(frames)) * frame_interval
    
    # Map raw data to a full-length array (handling missing frames)
    # df_raw might have fewer rows than 'frames' if frames were skipped.
    raw_times = df_raw['Time (s)'].values
    raw_x = df_raw['X Position (pixels)'].values
    raw_y = df_raw['Y Position (pixels)'].values
    
    full_x = np.full(len(frames), np.nan)
    full_y = np.full(len(frames), np.nan)
    
    # Fill based on time match
    for t_val, x_val, y_val in zip(raw_times, raw_x, raw_y):
        idx = int(round(t_val / frame_interval))
        if 0 <= idx < len(frames):
            full_x[idx] = x_val
            full_y[idx] = y_val

    # Outlier detection: Any jump > 60px between frames is marked NaN
    for i in range(1, len(full_x)):
        if not np.isnan(full_x[i]) and not np.isnan(full_x[i-1]):
            dist = np.sqrt((full_x[i] - full_x[i-1])**2 + (full_y[i] - full_y[i-1])**2)
            if dist > 60:
                full_x[i], full_y[i] = np.nan, np.nan

    # Interpolate gaps for guidance
    # We need at least 2 points
    mask = ~np.isnan(full_x)
    if mask.sum() < 2:
        logging.error("Not enough valid points for refinement.")
        return
        
    guide_x = np.interp(time_points, time_points[mask], full_x[mask])
    guide_y = np.interp(time_points, time_points[mask], full_y[mask])

    # 3. Manual Distractor Selection (Optional)
    distractor_zones = []
    logging.info("Opening sample frames for distractor selection. Press ENTER or SPACE after selecting each ROI. Press ESC when finished.")
    
    # Pick 3 representative frames (Start, Mid, End)
    sample_indices = [0, len(frames)//2, len(frames)-1]
    
    cv2.namedWindow('Select DISTRACTORS (Bubbles/Noise) - Press ESC when done', cv2.WINDOW_NORMAL)
    for idx in sample_indices:
        sample_frame = cv2.imread(os.path.join(input_folder, frames[idx]))
        if sample_frame is not None:
            # selectROIs (plural) allows selecting multiple boxes in one window
            # It returns a list of (x, y, w, h)
            rois = cv2.selectROIs('Select DISTRACTORS (Bubbles/Noise) - Press ESC when done', sample_frame, fromCenter=False, showCrosshair=True)
            if len(rois) > 0:
                distractor_zones.extend(rois)
            # Short delay between frames
            cv2.waitKey(500)
    cv2.destroyAllWindows()
    
    if distractor_zones:
        logging.info(f"Registered {len(distractor_zones)} distractor exclusion zones.")

    # 4. Localized Tracking
    fgbg = cv2.createBackgroundSubtractorMOG2()
    positions = []
    aspect_ratios = []
    gif_frames = []
    
    roi_size = 100 # Increased from 100 to 150 for better tolerance

    # Load Topo Metrics for gating if available
    topo_gates = {}
    if topo_metrics_csv and os.path.exists(topo_metrics_csv):
        try:
            df_topo = pd.read_csv(topo_metrics_csv)
            topo_gates = df_topo.set_index('Filename')[['Topological_Sharpness_Index', 'Refined_Area']].to_dict('index')
            logging.info(f"Loaded {len(topo_gates)} topographical metrics for refinement ({gating_mode}).")
        except Exception as e:
            logging.error(f"Could not load topo metrics for gating: {e}")

    # Extract thresholds with defaults
    if thresholds is None:
        thresholds = {"tsi_low": -0.7, "tsi_high": -0.35, "area_low": 2.47, "vol_limit": 0.1}
    
    t_low = thresholds.get("tsi_low", -0.7)
    t_high = thresholds.get("tsi_high", -0.35)
    a_low = thresholds.get("area_low", 2.47)
    v_lim = thresholds.get("vol_limit", 0.1)

    last_valid_pos = None
    prev_tsi = -0.42

    logging.info("Starting Refined Pass 2 tracking...")
    for i, frame_name in enumerate(frames):
        # ... [frame loading and ROI coord calculations remain same]
        frame_path = os.path.join(input_folder, frame_name)
        frame = cv2.imread(frame_path)
        if frame is None:
            aspect_ratios.append(np.nan)
            continue

        # --- TOPOGRAPHICAL DUAL-GATING --- 
        is_at_rest = False
        metrics = topo_gates.get(frame_name, {'Topological_Sharpness_Index': 0.35, 'Refined_Area': 600})
        
        tsi_linear = metrics['Topological_Sharpness_Index']
        area_linear = metrics['Refined_Area']
        
        tsi_log = np.log10(max(tsi_linear, 0.0001))
        area_log = np.log10(max(area_linear, 1.0))
        
        tsi_gated = (tsi_log < t_low) or (tsi_log > t_high) or (abs(tsi_log - prev_tsi) > v_lim and tsi_log > -0.4)
        area_gated = (area_log < a_low)

        if gating_mode == 'tsi' and tsi_gated:
            is_at_rest = True
        elif gating_mode == 'area' and area_gated:
            is_at_rest = True
        elif gating_mode == 'both' and (tsi_gated or area_gated):
            is_at_rest = True

        if is_at_rest:
            logging.debug(f"Refined Frame {i}: Gated (Mode:{gating_mode}) - LogTSI:{tsi_log:.2f}, LogArea:{area_log:.2f}")
        
        prev_tsi = tsi_log

        h, w = frame.shape[:2]
        gx, gy = int(guide_x[i]), int(guide_y[i])
        x1 = max(0, gx - roi_size // 2)
        y1 = max(0, gy - roi_size // 2)
        x2 = min(w, gx + roi_size // 2)
        y2 = min(h, gy + roi_size // 2)
        
        # ... [distractor and tagged frame loading remains same]
        for (dx, dy, dw, dh) in distractor_zones:
            cv2.rectangle(frame, (dx, dy), (dx + dw, dy + dh), (0, 0, 150), 1)
        
        tagged_dir = os.path.join(os.path.dirname(input_csv), "tagged_frames_" + os.path.basename(input_csv).split("_positions")[0])
        tagged_frame_path = os.path.join(tagged_dir, frame_name)
        tagged_frame = None
        if os.path.exists(tagged_frame_path):
            tagged_frame = cv2.imread(tagged_frame_path)

        roi = frame[y1:y2, x1:x2].copy()
        if roi.size == 0:
            aspect_ratios.append(np.nan)
            continue
            
        fgmask = fgbg.apply(roi)
        _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        
        # PREPARE RED MASK (for ranking, not hard-filtering)
        red_mask_roi = None
        if tagged_frame is not None:
            roi_tagged = tagged_frame[y1:y2, x1:x2]
            red_mask_roi = cv2.inRange(roi_tagged, (0, 0, 180), (80, 80, 255))

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter and Rank Contours
        scored_contours = []
        for cnt in contours:
            tx, ty, tw, th = cv2.boundingRect(cnt)
            cx_f = x1 + tx + tw // 2
            cy_f = y1 + ty + th // 2
            
            # 1. Distractor Check
            is_distractor = False
            for (dx, dy, dw, dh) in distractor_zones:
                if dx <= cx_f <= dx + dw and dy <= cy_f <= dy + dh:
                    is_distractor = True
                    break
            if is_distractor:
                cv2.rectangle(frame, (x1 + tx, y1 + ty), (x1 + tx + tw, y1 + ty + th), (0, 0, 150), 1)
                continue
            
            # 2. Score by Red Pixels (Confidence)
            red_pixels = 0
            if red_mask_roi is not None:
                # Mask out just this contour area in the red mask
                cnt_mask = np.zeros_like(thresh)
                cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)
                red_pixels = cv2.countNonZero(cv2.bitwise_and(red_mask_roi, red_mask_roi, mask=cnt_mask))
            
            # Store with Area as tie-breaker
            scored_contours.append({
                'cnt': cnt,
                'red_score': red_pixels,
                'area': cv2.contourArea(cnt),
                'bounds': (tx, ty, tw, th)
            })

        best_score = -1
        best_candidate = None
        success = False
        
        if not is_at_rest and scored_contours:
            # Rank: Preferred if has red, then largest area
            scored_contours.sort(key=lambda x: (x['red_score'] > 5, x['area']), reverse=True)
            best_candidate = scored_contours[0]
            
            tx, ty, tw, th = best_candidate['bounds']
            cx = x1 + tx + tw // 2
            cy = y1 + ty + th // 2
            
            last_valid_pos = (cx, cy)
            positions.append((i * frame_interval, cx, cy))
            aspect_ratios.append(tw / th if th != 0 else np.nan)
            success = True
            
            # Visual Feedback
            has_red = best_candidate['red_score'] > 5
            color = (0, 255, 0) if has_red else (255, 255, 0)
            status_text = "RED LOCK" if has_red else "MOG2 ONLY"
            
            cv2.rectangle(frame, (x1 + tx, y1 + ty), (x1 + tx + tw, y1 + ty + th), color, 2)
            cv2.circle(frame, (cx, cy), 5, color, -1)
            cv2.putText(frame, f"ID: {status_text}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        if not success:
            # Handle "AT REST" or "NOT FOUND"
            if last_valid_pos:
                cx, cy = last_valid_pos
            else:
                # Last resort: use the guide position from Pass 1
                cx, cy = gx, gy
            
            positions.append((i * frame_interval, cx, cy))
            aspect_ratios.append(np.nan)
            
            # Visual feedback in refinement
            color = (0, 255, 255) if is_at_rest else (0, 165, 255) # Yellow/Orange
            cv2.putText(frame, "AT REST (Topo)" if is_at_rest else "NOT REFINED", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.circle(frame, (cx, cy), 5, color, 2)

        # Prepare visualization frame (from masked_dir if provided)
        vis_frame = None
        if masked_dir:
            masked_path = os.path.join(masked_dir, frame_name)
            vis_frame = cv2.imread(masked_path)
            
        if vis_frame is None:
            # Fallback: create a masked version of the current frame
            h, w = frame.shape[:2]
            center = (w // 2, h // 2)
            radius = min(center) - 10
            mask_vis = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask_vis, center, radius, 255, -1)
            vis_frame = cv2.bitwise_and(frame, frame, mask=mask_vis)
        
        valid_pts = [(p[1], p[2]) for p in positions if not np.isnan(p[1])]
        if len(valid_pts) > 1:
            pts = np.array(valid_pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis_frame, [pts], isClosed=False, color=(0, 255, 255), thickness=1)
            
        gif_frames.append(cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB))
        
        cv2.imshow('Refined Tracking (Pass 2)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    # 4. Analysis & Saving (Copy-Paste of your logic from Pass 1)
    if not positions:
        return
        
    positions = np.array(positions)
    # Recalculate all parameters using the cleaned trajectory
    # [Rest of your kinematic calculation blocks here...]
    # I'll include the full logic to ensure Stage 6 produces a complete result.
    
    displacements = np.sqrt(np.diff(positions[:, 1])**2 + np.diff(positions[:, 2])**2)
    velocity = displacements / frame_interval
    acceleration = np.diff(velocity, append=velocity[-1]) # More stable length matching
    
    velocity = np.insert(velocity, 0, 0)
    acceleration = np.insert(acceleration, 0, 0)
    
    # Trim to match positions.size
    velocity = velocity[:len(positions)]
    acceleration = acceleration[:len(positions)]

    turning_angles = [0]
    for i in range(1, len(positions) - 1):
        v1 = positions[i] - positions[i-1]
        v2 = positions[i+1] - positions[i]
        dot = np.dot(v1[1:], v2[1:])
        mag1 = np.linalg.norm(v1[1:])
        mag2 = np.linalg.norm(v2[1:])
        if mag1 != 0 and mag2 != 0:
            cos_val = np.clip(dot / (mag1 * mag2), -1.0, 1.0)
            turning_angles.append(np.degrees(np.arccos(cos_val)))
        else:
            turning_angles.append(0)
    turning_angles.append(0)

    net_displacements = []
    total_path_lengths = []
    gross_speeds = []
    tortuosities = []
    
    # Find first valid position index for net displacement origin
    first_valid_idx = 0
    while first_valid_idx < len(positions) and np.isnan(positions[first_valid_idx, 1]):
        first_valid_idx += 1
        
    origin = positions[first_valid_idx] if first_valid_idx < len(positions) else positions[0]
    
    cumulative_path_length = 0
    for i in range(len(positions)):
        if np.isnan(positions[i, 1]) or first_valid_idx >= len(positions) or i < first_valid_idx:
            net_displacements.append(0.0)
            total_path_lengths.append(cumulative_path_length)
            gross_speeds.append(0.0)
            tortuosities.append(np.nan)
        else:
            net_disp = np.sqrt((positions[i, 1] - origin[1])**2 + (positions[i, 2] - origin[2])**2)
            net_displacements.append(net_disp)
            
            if i > 0:
                step_dist = displacements[i-1]
                if not np.isnan(step_dist): 
                    cumulative_path_length += step_dist
            
            total_path_lengths.append(cumulative_path_length)
            
            elapsed_time = positions[i, 0] - origin[0]
            gross_speed = cumulative_path_length / elapsed_time if elapsed_time > 0 else 0.0
            gross_speeds.append(gross_speed)
            
            tortuosity = cumulative_path_length / net_disp if net_disp != 0 else np.nan
            tortuosities.append(tortuosity)

    columns = ['Time (s)', 'X Position (pixels)', 'Y Position (pixels)', 'Velocity (pixels/sec)', 
               'Acceleration (pixels/sec^2)', 'Turning Angle (degrees)', 'Net Displacement (pixels)', 
               'Total Path Length (pixels)', 'Gross Speed (pixels/sec)', 'Tortuosity', 'Aspect Ratio']
    
    data = np.column_stack((positions, velocity, acceleration, turning_angles, 
                            net_displacements, total_path_lengths, gross_speeds, tortuosities, aspect_ratios[:len(positions)]))
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_csv, index=False)
    
    # Plots and Metric Outputs
    if len(positions) > 1:
        x, y, t = positions[:, 1], positions[:, 2], positions[:, 0]
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # 1. Pixel Trajectory Plot (Colored by Time)
        fig, ax = plt.subplots()
        lc = LineCollection(segments, cmap='viridis', norm=Normalize(t.min(), t.max()))
        lc.set_array(t)
        lc.set_linewidth(2)
        line = ax.add_collection(lc)
        fig.colorbar(line, ax=ax, label='Time (s)')
        ax.set_xlim(x.min()-10, x.max()+10)
        ax.set_ylim(y.min()-10, y.max()+10)
        ax.invert_yaxis()
        ax.set_title('Refined Trajectory (Pixels)', fontweight='bold')
        ax.set_xlabel('X Position (pixels)')
        ax.set_ylabel('Y Position (pixels)')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_aspect('equal', 'datalim')
        plt.savefig(output_plot, dpi=300)
        plt.close()

        # 2. Metric Calculations (mm)
        # Using image width for scaling (diameter = 24mm)
        mm_per_pixel = 24.0 / w
        x_mm = (x - w / 2) * mm_per_pixel
        y_mm = (h / 2 - y) * mm_per_pixel # Flip Y for Cartesian
        
        # Metric CSV
        df_metric = df.copy()
        df_metric['X Position (pixels)'] = (df['X Position (pixels)'] - w/2)
        df_metric['Y Position (pixels)'] = (h/2 - df['Y Position (pixels)'])
        
        column_mapping = {
            'X Position (pixels)': ('X Position (mm)', True),
            'Y Position (pixels)': ('Y Position (mm)', True),
            'Velocity (pixels/sec)': ('Velocity (mm/s)', True),
            'Acceleration (pixels/sec^2)': ('Acceleration (mm/s^2)', True),
            'Net Displacement (pixels)': ('Net Displacement (mm)', True),
            'Total Path Length (pixels)': ('Total Path Length (mm)', True),
            'Gross Speed (pixels/sec)': ('Gross Speed (mm/s)', True),
            'Time (s)': ('Time (s)', False),
            'Turning Angle (degrees)': ('Turning Angle (degrees)', False),
            'Tortuosity': ('Tortuosity', False),
            'Aspect Ratio': ('Aspect Ratio', False)
        }
        
        metric_cols = []
        for col in df.columns:
            dest_name, scaled = column_mapping.get(col, (col, False))
            metric_cols.append(dest_name)
            if scaled:
                df_metric[col] = df_metric[col] * mm_per_pixel
        df_metric.columns = metric_cols
        df_metric.to_csv(output_csv.replace('.csv', '_metric.csv'), index=False)

        # 3. Metric Colormap Plots (Red Axes, Centered)
        velocity_mm = df_metric['Velocity (mm/s)'].values[1:]
        turning_angle_mm = df_metric['Turning Angle (degrees)'].values[1:-1]
        
        points_centered = np.array([x_mm, y_mm]).T.reshape(-1, 1, 2)
        segments_centered = np.concatenate([points_centered[:-1], points_centered[1:]], axis=1)

        # A. Trajectory colored by TIME (Metric)
        fig, ax = plt.subplots()
        lc_time = LineCollection(segments_centered, cmap='viridis', norm=Normalize(t.min(), t.max()))
        lc_time.set_array(t)
        lc_time.set_linewidth(2)
        ax.add_collection(lc_time)
        fig.colorbar(lc_time, ax=ax, label='Time (s)')
        
        ax.set_title('Refined Trajectory (Metric - mm)', fontweight='bold')
        ax.set_xlabel('X Position (mm)')
        ax.set_ylabel('Y Position (mm)')
        ax.set_xlim(-14, 14)
        ax.set_ylim(-14, 14)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_aspect('equal', 'datalim')
        
        plt.savefig(output_plot.replace('.png', '_metric_refined.png'), dpi=300)
        plt.close()

        # B. Trajectory colored by VELOCITY (Metric)
        fig, ax = plt.subplots()
        lc_vel = LineCollection(segments_centered, cmap='magma', norm=Normalize(velocity_mm.min(), velocity_mm.max()))
        lc_vel.set_array(velocity_mm)
        lc_vel.set_linewidth(2)
        ax.add_collection(lc_vel)
        fig.colorbar(lc_vel, ax=ax, label='Velocity (mm/s)')
        
        ax.set_title('Refined Trajectory (Velocity)', fontweight='bold')
        ax.set_xlabel('X Position (mm)')
        ax.set_ylabel('Y Position (mm)')
        ax.set_xlim(-14, 14)
        ax.set_ylim(-14, 14)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_aspect('equal', 'datalim')
        
        plt.savefig(output_plot.replace('.png', '_velocity_refined.png'), dpi=300)
        plt.close()

        # C. Trajectory colored by TURNING ANGLE (Metric)
        points_angle = np.array([x_mm[1:], y_mm[1:]]).T.reshape(-1, 1, 2)
        segments_angle = np.concatenate([points_angle[:-1], points_angle[1:]], axis=1)
        
        fig, ax = plt.subplots()
        lc_ang = LineCollection(segments_angle, cmap='coolwarm', norm=Normalize(0, 180))
        lc_ang.set_array(turning_angle_mm)
        lc_ang.set_linewidth(2)
        ax.add_collection(lc_ang)
        fig.colorbar(lc_ang, ax=ax, label='Turning Angle (degrees)')
        
        ax.set_title('Refined Trajectory (Turning Angle)', fontweight='bold')
        ax.set_xlabel('X Position (mm)')
        ax.set_ylabel('Y Position (mm)')
        ax.set_xlim(-14, 14)
        ax.set_ylim(-14, 14)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_aspect('equal', 'datalim')
        
        plt.savefig(output_plot.replace('.png', '_angle_refined.png'), dpi=300)
        plt.close()

    if gif_frames:
        imageio.mimsave(output_plot.replace('.png', '_refined.gif'), gif_frames, duration=100)
        
        # Save individual frames with axes
        frames_dir = os.path.join(os.path.dirname(output_plot), os.path.basename(output_plot).replace('.png', '_frames_pass2'))
        save_high_res_trajectory_frames(gif_frames, frames_dir, "Stage 6 Refinement")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
