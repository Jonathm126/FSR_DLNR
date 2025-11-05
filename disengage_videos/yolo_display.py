import cv2
from pathlib import Path

images_dir = Path("/home/jonathan/Documents/Github/FSR_DLNR/disengage_videos/images")
labels_dir = Path("/home/jonathan/Documents/Github/FSR_DLNR/disengage_videos/labels")
KP_NAMES = ["tip", "bend_1", "bend_2", "shaft"]

for label_path in sorted(labels_dir.glob("*.txt")):
    img_path = images_dir / (label_path.stem + ".png")
    if not img_path.exists():
        continue
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    vals = list(map(float, open(label_path).read().split()))

    _, xc, yc, bw, bh = vals[:5]
    kps = vals[5:]

    # Bounding box
    x1, y1 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
    x2, y2 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Keypoints
    for i in range(0, len(kps), 3):
        x, y, v = kps[i:i+3]
        color = (0, 0, 255) if v == 2 else (0, 165, 255)
        cx, cy = int(x * w), int(y * h)
        if v > 0:
            cv2.circle(img, (cx, cy), 4, color, -1)
            cv2.putText(img, KP_NAMES[i // 3], (cx+5, cy-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    cv2.imshow("Labels Only", img)
    if cv2.waitKey(0) == 27:
        break
cv2.destroyAllWindows()
