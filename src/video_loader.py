import cv2
import os
from queue import Queue
from threading import Thread
import logging

def extract_frames(video_path, output_folder, interval_sec=1):
    """
    Extracts frames from a video file at a specified interval.
    """
    os.makedirs(output_folder, exist_ok=True)
    frame_queue = Queue(maxsize=10)
    
    def read_frames(path, queue):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            logging.error(f"Could not open video: {path}")
            queue.put(None)
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * interval_sec)
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                queue.put(frame)
            frame_count += 1

        queue.put(None)
        cap.release()

    def save_frames(queue, folder):
        count = 0
        while True:
            frame = queue.get()
            if frame is None:
                break
            frame_name = f'frame{count:04d}.jpg'
            cv2.imwrite(os.path.join(folder, frame_name), frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            count += 1
        logging.info(f"Extracted {count} frames to {folder}")

    reader_thread = Thread(target=read_frames, args=(video_path, frame_queue))
    writer_thread = Thread(target=save_frames, args=(frame_queue, output_folder))

    reader_thread.start()
    writer_thread.start()

    reader_thread.join()
    writer_thread.join()

if __name__ == "__main__":
    # Test extraction
    logging.basicConfig(level=logging.INFO)
    extract_frames('data/input/T10_R3.mov', 'data/cache/frames')
