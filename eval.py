from pathlib import Path

import pandas as pd
from ultralytics import YOLO


def eval():
    val_df = {}
    metric_col = None
    runs_dir = Path.cwd() / "runs/detect"
    for exp in sorted((runs_dir / "train").iterdir()):
        if not exp.is_dir():
            continue
        model = YOLO(exp / "weights/best.pt")
        metrics = model.val(
            data="./datasets/ctafor_01/ctafor_01.yaml",
            conf=0.25,
            split="val",
            batch=8,
            project="val",
            name=exp.name,
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
            val_df["metric"] = metric_col
        val_df[exp.name] = [
            metrics.box.map50,
            metrics.box.map75,
            metrics.box.map,
            *metrics.box.ap50,
            *metrics.box.maps,
        ]
    val_df = pd.DataFrame(data=val_df)
    val_df["metric"] = val_df["metric"].str.pad(
        width=max(len(metric) for metric in val_df["metric"]),
    )
    val_df.to_csv(
        runs_dir / "val/experiments_val.csv",
        float_format="%.5f",
        header=True,
        index=False,
    )
    return


def _run():
    try:
        eval()
        return 0
    except Exception:
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(_run())
