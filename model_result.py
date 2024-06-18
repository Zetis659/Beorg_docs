import cv2
from ultralytics import YOLO
import numpy as np
import time
import os

def correct_perspective(image, points):
    width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
    height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))

    dst_points = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(points, dst_points)
    corrected_image = cv2.warpPerspective(image, M, (width, height))
    return corrected_image

def detect_and_save_stamp_area(image, model_detection, output_path):
    detect_results = model_detection(image)
    
    if detect_results[0].boxes is not None:
        stamps = [box for box in detect_results[0].boxes if box.cls == 0]
        
        if len(stamps) > 0:
            stamps_sorted = sorted(stamps, key=lambda s: s.xyxy[0][2], reverse=True)
            
            last_stamp = None
            max_y2 = -1

            for stamp in stamps_sorted:
                x1, y1, x2, y2 = map(int, stamp.xyxy[0])

                if last_stamp is not None:
                    last_x1, last_y1, last_x2, last_y2 = map(int, last_stamp.xyxy[0])
                    if abs(last_x2 - x2) > 150:
                        continue

                if y2 > max_y2:
                    last_stamp = stamp
                    max_y2 = y2

            if last_stamp:
                x1, y1, x2, y2 = map(int, last_stamp.xyxy[0])
                margin = 75
                x1_margin = max(0, x1 - margin)
                y1_margin = max(0, y1 - margin)
                x2_margin = min(image.shape[1], x2 + margin)
                y2_margin = min(image.shape[0], y2 + margin)

                stamp_area = image[y1_margin:y2_margin, x1_margin:x2_margin]

                # Определяю углы для корректировки перспективы
                points = np.array([
                    [x1, y1], [x2, y1],
                    [x2, y2], [x1, y2]
                ], dtype="float32")

                # Корректирую перспективу
                corrected_stamp_area = correct_perspective(image, points)

                cv2.imwrite(output_path, corrected_stamp_area)
                print(f"Изображение зоны вокруг штампа сохранено в файл: {output_path}")

                print(f"Координаты последнего штампа: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            else:
                print("Последний штамп не найден.")
        else:
            print("Штампы не найдены.")
    else:
        print("Штампы не найдены в результатах детекции.")

def classify_and_process_image(image_path, model_classification, model_detection, output_path):
    image = cv2.imread(image_path)
    results = model_classification(image)
    probs = results[0].probs
    top = probs.top1

    if top == 0: 
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif top == 1:  
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif top == 2:  
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    detect_and_save_stamp_area(image, model_detection, output_path)

def process_images_from_folder(input_folder, output_folder, model_classification, model_detection):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_stamp{os.path.splitext(filename)[1]}")
            classify_and_process_image(image_path, model_classification, model_detection, output_path)

start_time = time.time()


# Загружаю обученные модели
model_detection = YOLO('runs/detect/train/weights/best.pt')
model_classification = YOLO('runs/classify/train/weights/best.pt')

input_folder = 'datasets/detection_yolo/images/val'
output_folder = 'results'

process_images_from_folder(input_folder, output_folder, model_classification, model_detection)

finish_time = time.time()
total_time = finish_time - start_time
print(f'Total_time: {total_time:.2f}')