from pathlib import Path

import cv2


VIDEO = Path("data/input/bad_apple.mp4")
OUTPUT = Path("data/input/bad_apple_frames")

MAX_FRAMES = None


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    video = cv2.VideoCapture(str(VIDEO))

    if not video.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO}")

    frame_index = 0

    #while frame_index < MAX_FRAMES:
    while MAX_FRAMES is None or frame_index < MAX_FRAMES:
        success, frame = video.read()

        if not success:
            break

        path = OUTPUT / f"{frame_index:06d}.png"
        cv2.imwrite(str(path), frame)

        frame_index += 1

    video.release()

    print(f"Saved {frame_index} frames to {OUTPUT}")


if __name__ == "__main__":
    main()