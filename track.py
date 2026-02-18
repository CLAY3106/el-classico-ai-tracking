import csv
from ultralytics import YOLO
from src.classification.team_classifier import TeamClassifier

VIDEO_PATH = r"C:\Users\Son\Documents\comp-vision-football\video\clip.mp4"
OUT_CSV = "tracks.csv"

model = YOLO("yolov8n.pt")

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["frame", "track_id", "team_id", "cls", "conf", "x1", "y1", "x2", "y2","cx","cy"])

    frame_idx = 0
    team_classifier = TeamClassifier()
    for r in model.track(
        source=VIDEO_PATH,
        tracker="bytetrack.yaml", #use ByteTrack
        persist=True,
        imgsz=480,
        vid_stride=2,
        stream=True, # return generator (no auto-plot/save), save memory
        verbose=False # turn off log
    ):
        frame = r.orig_img
        frame_idx += 1

        # check if frame has no detection or tracker with no ID -> skip
        if r.boxes is None or r.boxes.id is None:
            continue

        boxes = r.boxes.xyxy.cpu().numpy() # contain boxes
        ids = r.boxes.id.cpu().numpy().astype(int) # tọa độ bbox dạng x1,x2,y1,y2
        clss = r.boxes.cls.cpu().numpy().astype(int) # class id
        confs = r.boxes.conf.cpu().numpy() #confidence
        # cpu().numpy(): turn tensor -> numpy for easy record
        # tensor: a multi-dimentional data structure used in ML (>=3D)

        H, W = frame.shape[:2]
        for (x1, y1, x2, y2), tid, cls, conf in zip(boxes, ids, clss, confs):
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            x1 = max(0, min(x1, W-1))
            x2 = max(0, min(x2, W))
            y1 = max(0, min(y1, H-1))
            y2 = max(0, min(y2, H))

            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            team_id = team_classifier.update(tid, crop)   
            if team_id is None:
                team_id = -1
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w.writerow([frame_idx, tid, team_id, cls, float(conf), float(x1), float(y1), float(x2), float(y2), cx, cy])
            
# return the raw tracking
print(f"Saved: {OUT_CSV}")
