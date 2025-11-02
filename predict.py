
from ultralytics import YOLO
import os

image_dir=r'\images\val'
model = YOLO(r"runs/detect/train4/weights/best.pt") # model  yaml
for name in os.listdir(image_dir):
    image_path=os.path.join(image_dir,name)
    results = model(image_path,save=True,save_conf=True,save_txt=True) # data yaml
