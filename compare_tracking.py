import pandas as pd
import numpy as np
import os

def analyze_csv(file_path):
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path)
    cols = df.columns
    x_col = [c for c in cols if 'X Position' in c][0]
    y_col = [c for c in cols if 'Y Position' in c][0]
    vel_col = [c for c in cols if 'Velocity' in c][0]
    
    # Calculate velocity spikes (outliers)
    vel = df[vel_col].dropna()
    v_max = vel.max()
    v_mean = vel.mean()
    v_std = vel.std()
    
    # Calculate coordinate jumps (Euclidean distance between frames)
    # The 'Velocity' column might already be this, but let's be sure.
    dx = df[x_col].diff()
    dy = df[y_col].diff()
    dist = np.sqrt(dx**2 + dy**2)
    max_jump = dist.max()
    mean_jump = dist.mean()
    
    # Outlier count (where velocity > 2*std + mean)
    outlier_threshold = v_mean + 2 * v_std
    outliers = (vel > outlier_threshold).sum()
    
    # Percentiles
    v_95 = np.percentile(vel, 95)
    v_median = vel.median()
    
    # Significant jumps (> 150 pixels/sec)
    jumps_150 = (vel > 150).sum()
    
    return {
        'count': len(df),
        'max_vel': v_max,
        'mean_vel': v_mean,
        'median_vel': v_median,
        'v_95th': v_95,
        'std_vel': v_std,
        'outliers_count': outliers,
        'jumps > 150px/s': jumps_150
    }

csv1 = r"d:\ARPITA_PROJECTS\python swimming tracking\Coppe-Track\data\output\T30_R6_positions.csv"
csv2 = r"d:\ARPITA_PROJECTS\python swimming tracking\Coppe-Track\data\output\T30_R6_refined_positions.csv"

res1 = analyze_csv(csv1)
res2 = analyze_csv(csv2)

if res1 and res2:
    print("Comparison Summary:")
    print(f"{'Metric':<20} | {'Round 1':<15} | {'Refined':<15} | {'Reduction %':<15}")
    print("-" * 70)
    for key in res1.keys():
        v1 = res1[key]
        v2 = res2[key]
        if v1 != 0:
            change = ((v1 - v2) / v1) * 100
        else:
            change = 0
        
        # For "count", reduction might actually be a good thing if it was noise, 
        # but usually we want to keep points. However, refinement often prunes.
        print(f"{key:<20} | {v1:>15.4f} | {v2:>15.4f} | {change:>14.2f}%")
else:
    print("Error: One or both files could not be processed.")
