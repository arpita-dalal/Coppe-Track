import cv2
import os
import numpy as np
from PIL import Image, ImageChops
import logging

def invert_images(input_dir, output_dir):
    """
    Inverts all grayscale images in a directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    count = 0
    for file in files:
        img = cv2.imread(os.path.join(input_dir, file), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            inverted_img = 255 - img
            cv2.imwrite(os.path.join(output_dir, file), inverted_img)
            count += 1
    logging.info(f"Inverted {count} images and saved to {output_dir}")

def apply_circular_mask(input_dir, output_dir, center, radius):
    """
    Applies a circular mask and crops images to a square using OpenCV.
    """
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
    if not files:
        return

    # Integer coordinates for OpenCV
    cx, cy = int(center[0]), int(center[1])
    r = int(radius)

    count = 0
    for file in files:
        img = cv2.imread(os.path.join(input_dir, file))
        if img is None: continue
        
        # Create mask
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, -1)
        
        # Apply mask
        masked = cv2.bitwise_and(img, img, mask=mask)
        
        # Crop box: (y1:y2, x1:x2)
        y1, y2 = max(0, cy - r), min(h, cy + r)
        x1, x2 = max(0, cx - r), min(w, cx + r)
        cropped = masked[y1:y2, x1:x2]
        
        cv2.imwrite(os.path.join(output_dir, file), cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
        count += 1
    logging.info(f"Masked and cropped {count} images using OpenCV into {output_dir}")

def select_circle_interactively(image_path):
    """
    Opens a GUI for the user to select three points on the circle circumference.
    Returns (center, radius).
    """
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    img = mpimg.imread(image_path)
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.set_title("Click 3 points on the circle circumference")

    points = []

    def onclick(event):
        if event.button == 1:
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                points.append((x, y))
                ax.plot(x, y, 'ro')
                fig.canvas.draw()

                if len(points) == 3:
                    center = _calculate_circle_center(points[0], points[1], points[2])
                    if center:
                        radius = np.sqrt((center[0] - points[0][0])**2 + (center[1] - points[0][1])**2)
                        logging.info(f"Selected Circle - Center: {center}, Radius: {radius:.2f}")
                        plt.close(fig) # Automatically close the window
                    else:
                        logging.warning("Selected points are collinear. Resetting selection.")
                        points.clear()

    def _calculate_circle_center(p1, p2, p3):
        temp = p2[0]**2 + p2[1]**2
        bc = (p1[0]**2 + p1[1]**2 - temp) / 2
        cd = (temp - p3[0]**2 - p3[1]**2) / 2
        det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

        if abs(det) < 1.0e-10:
            return None

        cx = (bc * (p2[1] - p3[1]) - cd * (p1[1] - p2[1])) / det
        cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det
        return (cx, cy)

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    if len(points) == 3:
        center = _calculate_circle_center(points[0], points[1], points[2])
        radius = np.sqrt((center[0] - points[0][0])**2 + (center[1] - points[0][1])**2)
        return center, radius
    return None, None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example usage
    # invert_images('data/cache/frames', 'data/cache/inverted')
    # apply_circular_mask('data/cache/inverted', 'data/cache/masked', (472, 334), 321)
