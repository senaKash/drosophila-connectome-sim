from pathlib import Path

from flygym.compose import NeuroMechFly
from flygym.rendering import preview_model


OUTPUT = Path("results/flygym_preview.mp4")


fly = NeuroMechFly()

fly.colorize()
camera = fly.add_tracking_camera()

model, data = fly.compile()

preview_model(
    model,
    data,
    camera,
    duration=1.0,
    camera_res=(720, 720),
    output_fps=30,
    output_path=OUTPUT,
)

print("Saved:", OUTPUT)