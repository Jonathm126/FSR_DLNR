import cv2
from pathlib import Path
from ultralytics import YOLO

# ---------------------------
# Config
# ---------------------------
yolo_imgsz  = 864
target_h = 640
model_name  = f"s_{yolo_imgsz}_occlusion-v1"
model_chkpt = "best.pt"
pt_thresh   = 0.85
start_time  = 3 * 60 + 0
end_time    = 4 * 60 + 30
save_out    = True

# ---------------------------
# Paths
# ---------------------------
SCRIPT_DIR = Path(__file__).resolve()
BASE_DIR   = SCRIPT_DIR.parent
video_path = BASE_DIR / "disengage_videos" / "20160211_163956_603_001.mp4"
model_path = BASE_DIR / "runs" / "pose" / model_name / "weights" / model_chkpt
out_path   = BASE_DIR / "runs" / "pose" / "infer" / f"{model_name}.mp4"

# const
KP_NAMES = ["tip", "bend_1", "bend_2", "shaft"]
KP_COLORS = [
    (0, 255, 0),    # tip - green
    (0, 165, 255),  # bend_1 - orange
    (255, 0, 0),    # bend_2 - blue
    (180, 0, 255)   # shaft - purple/pink
]

# --- Load YOLO model ---
model = YOLO(model_path, task = 'pose')

# --- Open video ---
cap    = cv2.VideoCapture(str(video_path))
fps    = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
mid_x = width // 2

start_frame = int(start_time * fps)
end_frame   = int(end_time * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# output frame size
target_w = int(target_h * mid_x *2 / height) # make up for X2 width compression

# --- Output writer (matches model input resolution) ---
if save_out:
    print(f"Output frame size: ({target_w}, {target_h})")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (target_w, target_h),
    )

# --- Process frames ---
frame_idx = start_frame
while cap.isOpened() and frame_idx < end_frame:
    ret, frame = cap.read()
    if not ret:
        break

    # ---- Split stereo frame and resize ----
    left_frame = cv2.resize(frame[:, :mid_x], (target_w, target_h))

    # ---- YOLO inference (auto-resize to imgsz internally) ----
    results = model.predict(
        left_frame,
        conf    = pt_thresh,
        iou     = 0.3,
        imgsz   = yolo_imgsz, # width dimention of the model
        device  = "cuda",
        verbose = False,
    )
    annotated = results[0].orig_img.copy()

    # ---- Draw keypoints (only visible) ----
    for kp_set, conf_set in zip(results[0].keypoints.xy, results[0].keypoints.conf):
            for i, (pt, kp_conf) in enumerate(zip(kp_set, conf_set)):
                if kp_conf < pt_thresh:
                    continue

                x, y = map(int, pt)
                color = KP_COLORS[i % len(KP_COLORS)]
                cv2.circle(annotated, (x, y), 4, color, -1)
                label = f"{KP_NAMES[i]} ({kp_conf:.2f})"
                cv2.putText(
                    annotated,
                    label,
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                )
    
    # ---- Display / save ----
    cv2.imshow("YOLO Pose - Left View", annotated)
    if save_out:
        out.write(annotated)

    if cv2.waitKey(1) == 27:  # ESC
        break

    frame_idx += 1

cap.release()
if save_out:
    out.release()
cv2.destroyAllWindows()

print(f"Inference complete! Output saved to {out_path if save_out else '(not saved)'}")
