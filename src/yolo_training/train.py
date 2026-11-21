import argparse
from pathlib import Path

from ultralytics import YOLO


def train(
    dataset: Path,
    run_name: str,
    model_str: str,
    cfg_file: str | None = None,
    resume: bool = False,
) -> None:
    if resume:
        model = YOLO(
            f"runs/detect/{dataset.name}/train/{run_name}/weights/last.pt"
        )
        model.train(resume=True)
        return
    model = YOLO(model_str)
    model.train(
        data=(dataset / f"{dataset.name}.yaml"),
        name=run_name,
        project=f"{dataset.name}/train",
        cfg=cfg_file,
    )


def _get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        type=str,
        help="Target dataset.",
    )
    parser.add_argument(
        "name",
        type=str,
        nargs="?",
        default="exp",
        help="Training run name.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo26m.pt",
        help="Which YOLO model to use.",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default=None,
        help="Path to a training configuration file.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training run",
    )
    return parser


def _run() -> int:
    try:
        parser = _get_argparser()
        args = parser.parse_args()
        dataset = Path(args.dataset).resolve()
        if not dataset.is_dir():
            raise ValueError("`dataset` must be a path to a directory.")
        train(
            dataset=dataset,
            run_name=args.name,
            model_str=args.model,
            cfg_file=args.cfg,
            resume=args.resume,
        )
        return 0
    except Exception:
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(_run())
