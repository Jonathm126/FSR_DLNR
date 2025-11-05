import json
from pathlib import Path

# Define the desired keypoint order
KP_NAMES = ["tip", "bend_1", "bend_2", "shaft"]


def polyline_to_points(polyline):
    """Convert a polyline (x,y lists) into 4 ordered keypoints."""
    xs, ys = polyline["all_points_x"], polyline["all_points_y"]
    n = len(xs)
    if n < 4:
        xs += [xs[-1]] * (4 - n)
        ys += [ys[-1]] * (4 - n)
    elif n > 4:
        idxs = [int(i * (n - 1) / 3) for i in range(4)]
        xs = [xs[i] for i in idxs]
        ys = [ys[i] for i in idxs]
    return [{"name": name, "cx": int(x), "cy": int(y)} for name, x, y in zip(KP_NAMES, xs, ys)]

def clean_vgg_json(input_path, output_path):
    with open(input_path, "r") as f:
        data = json.load(f)

    # handle both old and new VGG keys
    img_metadata = data.get("_via_img_metadata", data)
    cleaned_metadata = {}

    for img_id, img_info in img_metadata.items():
        filename = img_info.get("filename")
        regions = img_info.get("regions", [])
        found_kps = {}

        for region in regions:
            shape = region.get("shape_attributes", {})
            attrs = region.get("region_attributes", {})

            if shape.get("name") == "point":
                kp_name = attrs.get("kp_name")
                if kp_name in KP_NAMES and kp_name not in found_kps:
                    found_kps[kp_name] = {
                        "shape_attributes": {"name": "point", "cx": int(shape["cx"]), "cy": int(shape["cy"])},
                        "region_attributes": {"kp_name": kp_name},
                    }

            elif shape.get("name") == "polyline":
                for kp in polyline_to_points(shape):
                    if kp["name"] not in found_kps:
                        found_kps[kp["name"]] = {
                            "shape_attributes": {"name": "point", "cx": kp["cx"], "cy": kp["cy"]},
                            "region_attributes": {"kp_name": kp["name"]},
                        }

        # Keep only complete sets of 4
        new_regions = [found_kps[kp] for kp in KP_NAMES if kp in found_kps]
        if len(new_regions) != 4:
            continue

        # Keep the structure VGG expects
        cleaned_metadata[img_id] = {
            "filename": filename,
            "size": img_info.get("size", 0),
            "regions": new_regions,
            "file_attributes": {}
        }

    cleaned_data = {
        "_via_settings": {
            "project": {"name": "cleaned_annotations"},
            "ui": {},
            "core": {"default_filepath": ""}
        },
        "_via_img_metadata": cleaned_metadata,
        "_via_attributes": {
            "region": {"kp_name": {"type": "text", "description": "", "options": {}}},
            "file": {}
        }
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(cleaned_data, f, indent=2)
    print(f"✅ Cleaned, valid VGG JSON saved to: {output_path}")
    print(f"Total valid images: {len(cleaned_metadata)}")

if __name__ == "__main__":
    input = 'disengage_videos\\frames_chopper_v1.json'
    output = 'disengage_videos\\frames_chopper_v1_clean.json'
    clean_vgg_json(input, output)
