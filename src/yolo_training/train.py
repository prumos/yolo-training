import argparse
from pathlib import Path
from typing import Any

from ultralytics import YOLO


def train(
    dataset: Path,
    run_name: str,
    model_str: str,
    cfg_file: str | None = None,
    resume: bool = False,
    overrides: dict[str, Any] | None = None,
) -> None:
    overrides = {} if overrides is None else overrides
    if resume:
        model = YOLO(
            f"runs/detect/{dataset.name}/train/{run_name}/weights/last.pt"
        )
        model.train(resume=True, **overrides)
        return
    model = YOLO(model_str)
    model.train(
        data=(dataset / f"{dataset.name}.yaml"),
        name=run_name,
        project=f"{dataset.name}/train",
        cfg=cfg_file,
        **overrides,
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
        help="Resume training run.",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override YOLO training configuration parameters using "
            "KEY=VALUE pairs. Multiple overrides can be specified "
            "(e.g., --overrides epochs=100 imgsz=800 batch=16)."
        ),
    )
    return parser


def process_overrides(args: list[str]) -> dict[str, int | float | str]:
    def convert(value: str) -> int | float | str:
        for fn in (int, float):
            try:
                return fn(value)
            except ValueError:
                continue
        return value
    return {
        key: convert(value)
        for key, value in (item.split("=", 1) for item in args)
    }


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
            overrides=process_overrides(args.overrides),
        )
        return 0
    except Exception:
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(_run())
