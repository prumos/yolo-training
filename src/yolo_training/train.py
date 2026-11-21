import argparse
from pathlib import Path

from ultralytics import YOLO


def train(dataset: Path, model: str = "yolo26m.pt") -> None:
    model = YOLO(model)
    model.train(
        data=(dataset / f"{dataset.name}.yaml"),
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
        project=f"{dataset.name}/train",
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
        default="yolo26m.pt",
        help="Which YOLO model to use.",
    )
    return parser


def _run() -> int:
    try:
        parser = _get_argparser()
        args = parser.parse_args()
        dataset = Path(args.dataset).resolve()
        if not dataset.is_dir():
            raise ValueError("`dataset` must be a path to a directory.")
        train(dataset, args.model)
        return 0
    except Exception:
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(_run())
