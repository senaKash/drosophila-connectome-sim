import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from flysim.vision.columns import build_right_eye_columns
from flysim.vision.eye_geometry import build_eye_geometry
from flysim.rendering.fly_view import reconstruct_fly_view
from flysim.vision.image_input import (
    sample_image_brightness_from_geometry,
)


ANNOTATIONS_PATH = (
    "data/raw/"
    "body-annotations-male-cns-v1.0-minconf-0.5.feather"
)

IMAGE_PATH = "data/input/test.png"


def load_image(path: str) -> np.ndarray:
    """
    Р—Р°РіСЂСѓР¶Р°РµС‚ РєР°СЂС‚РёРЅРєСѓ РІ grayscale Рё РїРµСЂРµРІРѕРґРёС‚ РІ РґРёР°РїР°Р·РѕРЅ [0, 1].
    """
    image = Image.open(path).convert("L")

    return (
        np.asarray(image, dtype=np.float32)
        / 255.0
    )


def main() -> None:
    # 1. Р—Р°РіСЂСѓР¶Р°РµРј Р°РЅРЅРѕС‚Р°С†РёРё С‚РѕР»СЊРєРѕ СЃ РЅСѓР¶РЅС‹РјРё РєРѕР»РѕРЅРєР°РјРё
    annotations = pd.read_feather(
        ANNOTATIONS_PATH,
        columns=[
            "type",
            "somaSide",
            "assignedOlHex1",
            "assignedOlHex2",
        ],
    )

    # 2. РЎС‚СЂРѕРёРј right-eye visual columns
    columns = build_right_eye_columns(
        annotations
    )

    # 3. РЎС‚СЂРѕРёРј РіРµРѕРјРµС‚СЂРёСЋ РіР»Р°Р·Р°
    geometry = build_eye_geometry(
        hex1=columns["hex1"].to_numpy(),
        hex2=columns["hex2"].to_numpy(),
    )

    # 4. Р—Р°РіСЂСѓР¶Р°РµРј РёСЃС…РѕРґРЅСѓСЋ РєР°СЂС‚РёРЅРєСѓ
    image = load_image(IMAGE_PATH)

    # 5. РЎСЌРјРїР»РёСЂСѓРµРј СЏСЂРєРѕСЃС‚СЊ РёР·РѕР±СЂР°Р¶РµРЅРёСЏ РІ РЅР°РїСЂР°РІР»РµРЅРёСЏ visual columns
    samples = sample_image_brightness_from_geometry(
        image=image,
        geometry=geometry,
        blur_radius_px=18,
        fov_x_degrees=110.0,
        fov_y_degrees=90.0,
    )

    brightness = samples["brightness"].to_numpy()

    # 6. Р’РѕСЃСЃС‚Р°РЅР°РІР»РёРІР°РµРј fly POV РєР°Рє blocky-РєР°СЂС‚РёРЅРєСѓ РёР· РїСЂСЏРјРѕСѓРіРѕР»СЊРЅРёРєРѕРІ
    fly_view = reconstruct_fly_view(
        geometry=geometry,
        brightness=brightness,
        width=640,
        height=480,
        eye_aspect=1.25,
    )

    # 7. РџРµС‡Р°С‚Р°РµРј РєСЂР°С‚РєСѓСЋ СЃС‚Р°С‚РёСЃС‚РёРєСѓ
    print("FLY VIEW")
    print("--------")
    print("Visual columns:", len(brightness))
    print("Output shape:", fly_view.shape)
    print("Brightness min:", float(fly_view.min()))
    print("Brightness mean:", float(fly_view.mean()))
    print("Brightness max:", float(fly_view.max()))
    print()

    print("First 10 samples:")
    print(
        samples[
            ["hex1", "hex2", "image_x", "image_y", "brightness"]
        ].head(10).to_string(index=False)
    )

    # 8. РџРѕРєР°Р·С‹РІР°РµРј СЃР»РµРІР° РѕСЂРёРіРёРЅР°Р», СЃРїСЂР°РІР° reconstructed fly POV
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    plt.title("Original image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(
        fly_view,
        cmap="gray",
        vmin=0,
        vmax=1,
        origin="lower",
        interpolation="nearest",
    )
    plt.title("Reconstructed fly POV")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
