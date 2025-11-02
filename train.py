from ultralytics import YOLO

model_yaml = 'ultralytics/cfg/models/sr/yolov8_sr.yaml'
dataset = 'ultralytics/cfg/datasets/sample_data.yaml'
model = YOLO(model_yaml)  # model  yaml
result = model.train(data=dataset,epochs=500, imgsz=1280)
