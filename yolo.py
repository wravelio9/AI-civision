from ultralytics import YOLO

model = YOLO("yolo26n.pt")

model.train(
    data="dataset/data.yaml",
    epochs=100,
    imgsz=640,
)

metrics = model.val(
    data="dataset/data.yaml",
    split="test"
)

# print(f"Val: ${metrics}")

model.predict(
    source="dataset/test/images",
    save=True,
    conf=0.5
)

# model.tune(
#     data="dataset/data.yaml",
#     epochs=30,
#     iterations=50,
# )