import os
import re

# path to your folder
folder = "C:\\Github\\FSR_DLNR\\face_laser_projector\\no_markers-09_11"

# pattern for matching files like 00000_left or 00000_right
pattern = re.compile(r"(\d+)_((left|right).*)")

for filename in os.listdir(folder):
    match = pattern.match(filename)
    if match:
        number = int(match.group(1))
        if number % 10 != 0:
            os.remove(os.path.join(folder, filename))
            print(f"Deleted {filename}")

print("Done.")
