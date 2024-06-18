import torch
from ultralytics import YOLO
import time


start_time = time.time()
model = YOLO('models/yolov8s-cls.pt')

data_path = 'datasets/classification_yolo'
model.train(data=data_path, epochs=100, imgsz=320)

# Проверка на валидационном датасете
results = model.val(data=data_path, imgsz=320)

end_time = time.time()

elapsed_time = end_time - start_time
print(f"Training completed in {elapsed_time:.2f} seconds")