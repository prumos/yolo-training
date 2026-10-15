from ultralytics import YOLO


def train():
    model = YOLO("./weights/yolo26s.pt")
    model.train(
        data="./datasets/ctafor_01/ctafor_01.yaml",
        batch=16,
        epochs=300,
        imgsz=896,
        device=0,
        cache=True,
        patience=0,
        workers=4,
        # Optim
        # optimizer="AdamW",
        # cos_lr=True,
        # lr0=0.005,
        # lrf=0.01,
        # Loss
        # cls_pw=0.5,
        # Augmentations
        close_mosaic=50,
        degrees=20,
        scale=(0.9, 1.5),
        shear=10,
        # Project
        save_period=50,
        name="ctafor_01_exp",
        exist_ok=False,
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
