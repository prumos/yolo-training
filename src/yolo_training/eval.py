import argparse
from pathlib import Path

import pandas as pd
from ultralytics import YOLO


def eval(dataset: Path, split: str = "val") -> None:
    results = {}
    metric_col = None
    runs_dir = Path.cwd() / "runs/detect"
    for exp in sorted((runs_dir / f"{dataset.name}/train").iterdir()):
        if not exp.is_dir():
            continue
        model = YOLO(exp / "weights/best.pt")
        metrics = model.val(
            data=(dataset / f"{dataset.name}.yaml"),
            conf=0.25,
            split=split,
            batch=8,
            name=exp.name,
            project=f"{dataset.name}/{split}",
            exist_ok=True,
            device="cuda:0",
            verbose=False,
            visualize=True,
            nms=True,
        )
        if metric_col is None:
            classes = [metrics.names[idx] for idx in sorted(metrics.names)]
            metric_col = [
                "map50",
                "map75",
                "map50-95",
                *[f"{cls_name}_map50" for cls_name in classes],
                *[f"{cls_name}_map50-95" for cls_name in classes],
            ]
            results["metric"] = metric_col
        results[exp.name] = [
            metrics.box.map50,
            metrics.box.map75,
            metrics.box.map,
            *metrics.box.ap50,
            *metrics.box.maps,
        ]
    results = pd.DataFrame(data=results)
    results["metric"] = results["metric"].str.pad(
        width=max(len(metric) for metric in results["metric"]),
    )
    results.to_csv(
        runs_dir / f"{dataset.name}/{split}/results.csv",
        float_format="%.5f",
        header=True,
        index=False,
    )


def _get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        type=str,
        help="Target dataset.",
    )
    parser.add_argument(
        "split",
        type=str,
        nargs="?",
        default="val",
        choices=["train", "val", "test"],
        help="Which data split to use.",
    )
    return parser


def _run() -> int:
    try:
        parser = _get_argparser()
        args = parser.parse_args()
        dataset = Path(args.dataset).resolve()
        if not dataset.is_dir():
            raise ValueError("`dataset` must be a path to a directory.")
        eval(dataset, args.split)
        return 0
    except Exception:
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(_run())
