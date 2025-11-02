
from ultralytics import YOLO

dataset = 'ultralytics/cfg/datasets/sample_data.yaml'
modelroot = r"runs/detect/train4/weights/best.pt"
model = YOLO(modelroot)  # model  yaml
result = model.val(data=dataset,imgsz=1280)
