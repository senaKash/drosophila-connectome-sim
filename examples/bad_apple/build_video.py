from pathlib import Path

import cv2


INPUT = Path("results/bad_apple_frames")
OUTPUT = Path("results/bad_apple_activity.mp4")

FPS = 30


def main():
    frames = sorted(INPUT.glob("*.png"))

    if not frames:
        raise RuntimeError(f"No frames found in {INPUT}")

    first = cv2.imread(str(frames[0]))
    height, width = first.shape[:2]

    writer = cv2.VideoWriter(
        str(OUTPUT),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (width, height),
    )

    for path in frames:
        frame = cv2.imread(str(path))

        if frame is None:
            raise RuntimeError(f"Cannot read {path}")

        writer.write(frame)
        print("Added:", path.name)

    writer.release()

    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()