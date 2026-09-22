import cv2
import os

video_path = "violence_video.mp4"
output_dir = "dataset/images"

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_number = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if frame_number % 10 == 0:
        cv2.imwrite(
            f"{output_dir}/frame_{frame_number:06d}.jpg",
            frame
        )

    frame_number += 1

cap.release()