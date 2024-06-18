import os
import shutil
import random

source_dir = 'datasets/classification'
destination_dir = 'datasets/classification_yolo'

if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

for folder in ['train', 'val']:
    folder_path = os.path.join(destination_dir, folder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

categories = ['inverted', 'left', 'right', 'straight']

# Обработка каждой категории
for category in categories:
    category_path = os.path.join(source_dir, category)
    images = os.listdir(category_path)

    random.shuffle(images)

    train_size = int(0.7 * len(images))

    train_images = images[:train_size]
    val_images = images[train_size:]

    train_category_path = os.path.join(destination_dir, 'train', category)
    val_category_path = os.path.join(destination_dir, 'val', category)

    if not os.path.exists(train_category_path):
        os.makedirs(train_category_path)
    if not os.path.exists(val_category_path):
        os.makedirs(val_category_path)

    for image in train_images:
        src_image_path = os.path.join(category_path, image)
        dest_image_path = os.path.join(train_category_path, image)
        shutil.copy(src_image_path, dest_image_path)

    for image in val_images:
        src_image_path = os.path.join(category_path, image)
        dest_image_path = os.path.join(val_category_path, image)
        shutil.copy(src_image_path, dest_image_path)

print('Данные успешно разделены и скопированы в новую папку.')