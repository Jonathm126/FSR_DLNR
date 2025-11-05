import cv2
from pathlib import Path
from ultralytics import YOLO

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve()
BASE_DIR = SCRIPT_DIR.parent
video_path = BASE_DIR / "disengage_videos" / "20160211_163956_603_001.mp4"
model_path = BASE_DIR / "runs" / "pose" / "s_640" / "weights" / "best.pt"
out_path   = BASE_DIR / "runs" / "pose" / "val3"/ "inference_segment_left.mp4"


# --- Config ---
start_time = 3*60        # seconds
end_time   = 4*60+30    # seconds
save_out = True

target_height = 640       # resize target height
KP_NAMES = ["tip", "bend_1", "bend_2", "shaft"]

# --- Load YOLO model ---
model = YOLO(model_path, task = 'pose')

# --- Open video ---
cap = cv2.VideoCapture(str(video_path))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

start_frame = int(start_time * fps)
end_frame   = int(end_time * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# --- Output writer (optional) ---
if save_out:
    out = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (1136, target_height),  # output width will be ~2×640 after resize
    )

# --- Process frames ---
frame_idx = start_frame
while cap.isOpened() and frame_idx < end_frame:
    ret, frame = cap.read()
    if not ret:
        break

    # ---- Split stereo frame ----
    mid_x = width // 2
    left_frame = frame[:, :mid_x]  # use left camera only

    # ---- Resize ----
    aspect_ratio = mid_x / height
    target_width = int(target_height * aspect_ratio)
    resized_left = cv2.resize(left_frame, (target_width, target_height))

    # ---- Duplicate horizontally (X2 width) ----
    resized_left = cv2.resize(resized_left, (target_width * 2, target_height))

    # ---- YOLO inference ----
    results = model.predict(
        resized_left,
        conf    = 0.8,
        iou     = 0.4,
        imgsz   = 640,
        device  = "cuda",
        verbose = True,
    )
    annotated = results[0].plot()

    # ---- Draw keypoint names (only visible) ----
    # ---- Draw keypoint names + confidence (only visible) ----
    for kp_set, conf_set in zip(results[0].keypoints.xy, getattr(results[0].keypoints, 'conf', [])):
        for i, (pt, kp_conf) in enumerate(zip(kp_set, conf_set)):
            x, y = map(int, pt)
            if results[0].keypoints.has_visible:
                if kp_conf > 0.5:  # visible enough
                    label = f"{KP_NAMES[i]} ({kp_conf:.2f})"
                    cv2.putText(
                        annotated,
                        label,
                        (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )
            else:
                # If model doesn't output confidences, just draw name
                cv2.putText(
                    annotated,
                    KP_NAMES[i],
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
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
