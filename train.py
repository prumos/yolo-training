import ultralytics

def train():
    model = ultralytics.YOLO("yolo26s.pt")
    model.train(
        data=".datasets/ctafor_01/ctafor_01.yaml",
        epochs=100,
        batch=0.8,
        save_period=10,
        cache=True,
        workers=4,
        project="./runs",
        name="ctafor_01_exp01",
        mosaic=0.5,
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(train())
