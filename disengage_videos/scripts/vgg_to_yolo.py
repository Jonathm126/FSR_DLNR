import json
import re
from pathlib import Path
from PIL import Image

KP_NAMES = ["tip", "bend_1", "bend_2", "shaft"]

def extract_frame_index(filename):
    """
    Extracts frame index (integer) from filename like 'frame_000123_left.png'.
    Returns None if pattern not found.
    """
    m = re.search(r"frame_(\d+)", filename)
    return int(m.group(1)) if m else None

def load_vgg_json(path):
    data = json.load(open(path))
    return data.get("_via_img_metadata", data)

def convert_vgg_to_yolo(vgg_json_path, images_dir, output_dir, class_id=0, start=None, end=None):
    img_metadata = load_vgg_json(vgg_json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_id, img_info in img_metadata.items():
        filename = img_info.get("filename")
        frame_idx = extract_frame_index(filename)
        if start is not None and end is not None:
            if frame_idx is None or not (start <= frame_idx <= end):
                continue  # skip frames outside the range

        regions = img_info.get("regions", [])
        img_path = Path(images_dir) / filename
        if not img_path.exists():
            print(f"⚠️ Skipping {filename}: image not found in {images_dir}")
            continue

        # Load image to get width and height
        with Image.open(img_path) as im:
            w, h = im.size

        # default all keypoints to occluded (v=1)
        kps = {name: (0.0, 0.0, 1) for name in KP_NAMES}

        for region in regions:
            shape = region.get("shape_attributes", {})
            attrs = region.get("region_attributes", {})
            if shape.get("name") == "point":
                kp_name = attrs.get("kp_name")
                if kp_name in KP_NAMES:
                    x, y = shape["cx"], shape["cy"]
                    kps[kp_name] = (x / w, y / h, 2)  # visible

        # compute bbox of visible or occluded (v>0) keypoints
        visible_points = [(x, y) for x, y, v in kps.values() if v > 0 and (x > 0 or y > 0)]
        if visible_points:
            xs, ys = zip(*visible_points)
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
        else:
            continue  # skip if no keypoints

        # normalize bbox
        x_center = ((xmin + xmax) / 2)
        y_center = ((ymin + ymax) / 2)
        bbox_w = (xmax - xmin)
        bbox_h = (ymax - ymin)

        x_center /= w
        y_center /= h
        bbox_w /= w
        bbox_h /= h

        # Build YOLO line
        line = [class_id, x_center, y_center, bbox_w, bbox_h]
        for name in KP_NAMES:
            x, y, v = kps[name]
            line.extend([x, y, v])

        # Save .txt
        out_file = output_dir / f"{Path(filename).stem}.txt"
        with open(out_file, "w") as f:
            f.write(" ".join(map(str, line)) + "\n")

    print(f"✅ Conversion complete! YOLO keypoint labels saved to {output_dir}")

if __name__ == "__main__":
    vgg_json_path = r"C:\\Github\\FSR_DLNR\\disengage_videos\\frames_chopper_v2.json"
    images_dir = "C:\\Github\\FSR_DLNR\\disengage_videos\\frames_chopper"
    output_dir = "C:\\Github\\FSR_DLNR\\disengage_videos\\labels_chopper"
    class_id = 0
    start = 0
    end = 10

    convert_vgg_to_yolo(
        vgg_json_path=vgg_json_path,
        images_dir=images_dir,
        output_dir=output_dir,
        class_id=class_id,
        start=start,
        end=end
    )