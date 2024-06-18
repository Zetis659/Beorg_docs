import time
from ultralytics import YOLO

start_time = time.time()


model = YOLO('models/yolov8n.pt') 
results = model.train(data="data.yaml", epochs=50, imgsz=320)

end_time = time.time()

elapsed_time = end_time - start_time
print(f"Training completed in {elapsed_time:.2f} seconds")