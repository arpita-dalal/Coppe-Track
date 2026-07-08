import os
import sys
import shutil
from src.main import run_coppe_track_pipeline

def main():
    print("=== Coppe-Track Runner ===")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_input_dir = os.path.join(current_dir, "data", "input")
    
    print(f"\nDefault input folder: {default_input_dir}")
    custom_dir = input("Enter a custom input folder path (or press Enter for default): ").strip()
    input_dir = custom_dir if custom_dir and os.path.isdir(custom_dir) else default_input_dir
    
    if not os.path.isdir(input_dir):
        print(f"Error: '{input_dir}' is not a valid directory.")
        return

    while True:
        video_extensions = ('.mov', '.mp4', '.avi', '.mkv')
        existing_videos = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(video_extensions)])
        
        print(f"\nAvailable videos in: {input_dir}")
        if not existing_videos:
            print("  (None)")
            break
        
        for i, vid in enumerate(existing_videos, 1):
            print(f"  {i}. {vid}")
        
        print("\nOptions:")
        print("  [Number] - Select a video to process")
        print("  q        - Quit")
        
        choice = input("\nSelect a video: ").strip()
        if choice.lower() == 'q': break
        
        if not choice.isdigit(): continue
        idx = int(choice) - 1
        if not (0 <= idx < len(existing_videos)):
            print("Invalid selection.")
            continue
            
        selected_video = existing_videos[idx]
        
        # Ingest if external
        if os.path.abspath(input_dir) != os.path.abspath(default_input_dir):
            dest_path = os.path.join(default_input_dir, selected_video)
            if not os.path.exists(dest_path):
                print(f"Ingesting video: {selected_video}")
                shutil.copy(os.path.join(input_dir, selected_video), dest_path)

        while True:
            cache_dir = os.path.join(current_dir, "data", "cache")
            output_dir = os.path.join(current_dir, "data", "output")
            video_name = os.path.splitext(selected_video)[0]
            
            print(f"\n--- Video: {selected_video} ---")
            print("Select starting stage:")
            print("  1. Start All Over (Delete Cache)")
            
            # Dynamic stage availability checking
            stage_paths = [
                os.path.join(cache_dir, "01_raw_frames"),
                os.path.join(cache_dir, "02_inverted_frames"),
                os.path.join(cache_dir, "03_masked_frames"),
                os.path.join(current_dir, "Topo refined frames")
            ]
            
            for i, path in enumerate(stage_paths, 2):
                if os.path.exists(path) and any(f.endswith('.jpg') for f in os.listdir(path)):
                    display_name = os.path.basename(path).replace('0', '').replace('_', ' ').capitalize()
                    if "Topo" in path: display_name = "Topographical isolates"
                    print(f"  {i}. Start from {display_name}")
            
            # Step 5 is always available if topo refined frames exist
            if os.path.exists(stage_paths[3]):
                print("  5. Start Topo-Driven MOG2 Tracking (Stage 5)")
                
            # Step 6 available if CSV exists
            if os.path.exists(os.path.join(output_dir, f"{video_name}_positions.csv")):
                print("  6. Tracking Refinement - Pass 2 (Stage 6)")
            
            print("\nOptions:")
            print("  b. Back to Video Selection")
            print("  q. Quit")
            
            step_choice = input("\nEnter choice: ").strip().lower()
            if step_choice == 'q': return
            if step_choice == 'b': break
            
            if not step_choice.isdigit(): continue
            start_step = int(step_choice)
            
            print(f"\n>>> Starting pipeline for: {selected_video} (Stage: {start_step})\n")
            run_coppe_track_pipeline(selected_video, start_step=start_step)
            print(f"\n>>> Stage {start_step} complete.")

if __name__ == "__main__":
    main()
