import argparse

from ultralytics import YOLO


def train(dataset: str, model: str = "weights/yolo26m.pt"):
    model = YOLO(model)
    model.train(
        data=f"datasets/{dataset}/{dataset}.yaml",
        batch=8,
        epochs=300,
        imgsz=896,
        device=0,
        cache=True,
        patience=0,
        workers=4,
        # Loss
        cls_pw=0.5,
        # Augmentations
        close_mosaic=50,
        degrees=20,
        scale=(0.9, 1.5),
        shear=10,
        # Project
        save_period=50,
        name="exp",
        project=f"{dataset}/train",
        exist_ok=False,
    )


def _get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        type=str,
        help="Target dataset.",
    )
    parser.add_argument(
        "model",
        type=str,
        default="weights/yolo26m.pt",
        help="Which YOLO model to use.",
    )
    return parser


def _run():
    try:
        parser = _get_argparser()
        args = parser.parse_args()
        train(args.dataset, args.model)
        return 0
    except Exception:
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(_run())
