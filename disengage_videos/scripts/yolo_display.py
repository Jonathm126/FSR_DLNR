import cv2
from pathlib import Path

# Get the current script directory
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
images_dir = BASE_DIR / "images"
labels_dir = BASE_DIR / "labels"

KP_NAMES = ["tip", "bend_1", "bend_2", "shaft"]
CLASS_NAMES = ["chopper"]  # you have one class

for label_path in sorted(labels_dir.glob("*.txt")):
    img_path = images_dir / (label_path.stem + ".png")
    if not img_path.exists():
        continue
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    vals = list(map(float, open(label_path).read().split()))

    cls, xc, yc, bw, bh = vals[:5]
    kps = vals[5:]

    # Bounding box
    x1, y1 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
    x2, y2 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Put class label on top-left of box
    label_text = CLASS_NAMES[int(cls)] if int(cls) < len(CLASS_NAMES) else str(int(cls))
    (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(img, (x1, y1 - text_h - 4), (x1 + text_w + 4, y1), (0, 255, 0), -1)
    cv2.putText(img, label_text, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Keypoints
    for i in range(0, len(kps), 3):
        x, y, v = kps[i:i+3]
        color = (0, 0, 255) if v == 2 else (0, 165, 255)
        cx, cy = int(x * w), int(y * h)
        if v > 0:
            cv2.circle(img, (cx, cy), 4, color, -1)
            cv2.putText(img, KP_NAMES[i // 3], (cx+5, cy-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    cv2.imshow(f"Labels Only - {img_path.name}", img)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()  # close window before showing the next one
    if key == 27:  # Esc key
        break
cv2.destroyAllWindows()
