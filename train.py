from ultralytics import YOLO


def train():
    model = YOLO("./weights/yolo26s.pt")
    model.train(
        data="./datasets/ctafor_01/ctafor_01.yaml",
        epochs=300,
        batch=16,
        imgsz=896,
        cache=True,
        cls_pw=1.0,
        save_period=30,
        name="ctafor_01_exp",
        exist_ok=False,
        device=0,
        workers=4,
        # Augmentations
        close_mosaic=100,
        degrees=20,
        scale=(0.9, 1.5),
        shear=10,
    )


def _run():
    try:
        train()
        return 0
    except Exception:
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(_run())
