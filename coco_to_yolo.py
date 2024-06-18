import os
import json
from shutil import copyfile

def convert_coco_to_yolo(coco_annotation_file, images_folder, output_image_folder, output_label_folder):
    # Открываю файл аннотаций COCO
    with open(coco_annotation_file) as f:
        coco_data = json.load(f)
    
    # Создаю словари для быстрого доступа к изображениям и категориям
    images = {img['id']: img for img in coco_data['images']}
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    annotations = coco_data['annotations']
    
    # Создаю выходные папки, если они не существуют
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)
    if not os.path.exists(output_label_folder):
        os.makedirs(output_label_folder)
        
    # Прохожу по всем аннотациям
    for ann in annotations:
        img_id = ann['image_id']
        category_id = ann['category_id']
        bbox = ann['bbox']
        
        img_info = images[img_id]
        img_width = img_info['width']
        img_height = img_info['height']
        
        # Преобразую координаты bbox в формат YOLO
        x_center = (bbox[0] + bbox[2] / 2) / img_width
        y_center = (bbox[1] + bbox[3] / 2) / img_height
        width = bbox[2] / img_width
        height = bbox[3] / img_height
        
        # Определяю имя файла изображения и метки
        img_filename = img_info['file_name']
        label_filename = img_filename.replace('.jpg', '.txt')
        
        # Записываю аннотации в файл меток в формате YOLO
        with open(os.path.join(output_label_folder, label_filename), 'a') as label_file:
            label_file.write(f"0 {x_center} {y_center} {width} {height}\n")
        
        # Копирую изображение в выходную папку
        copyfile(os.path.join(images_folder, img_filename), os.path.join(output_image_folder, img_filename))

coco_annotation_files = ['train.json', 'eval.json']
images_folder = 'images'
output_image_folders = ['datasets/detection_yolo/images/train', 'datasets/detection_yolo/images/val']
output_label_folders = ['datasets/detection_yolo/labels/train', 'datasets/detection_yolo/labels/val']

# Конвертируем аннотации для каждого файла
for coco_annotation_file, output_image_folder, output_label_folder in zip(coco_annotation_files, output_image_folders, output_label_folders):
    convert_coco_to_yolo(coco_annotation_file, images_folder, output_image_folder, output_label_folder)