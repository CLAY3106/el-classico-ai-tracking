import cv2
import time
from ultralytics import YOLO

# get the video path
VIDEO_PATH = r"C:\Users\Son\Documents\comp-vision-football\video\clip.mp4"
# output with bounding box
OUT_IMG = "detect_frame.jpg"

# create object that reads video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

# read the frames
ok, frame = cap.read()
cap.release()
if not ok:
    raise RuntimeError("Cannot read first frame from video.")

# load the model YOLO. This model receives images and returns bounding box + class + confidence
model = YOLO("yolov8n.pt")

# track time inference
t0 = time.time() # starting time
res = model.predict(frame, verbose=False) # run detection
dt = time.time() - t0 # time inference

# draw bouding box
annotated = res[0].plot() # contain boxes, class, confidence
cv2.imwrite(OUT_IMG, annotated) # save image

# print the results
print(f"Saved: {OUT_IMG}")
print(f"Inference: {dt:.3f}s | approx FPS: {1.0/dt:.1f}")
