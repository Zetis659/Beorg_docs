## Тестовое задание на вакансию "Разработчик CV"

### Results
Результаты работы алгоритма на eval части датасета лежат в папке /results  
Для удобства изображения штампов названы точно так же, как и оригинальные изображения, но с добавлением "..._stamp.jpg"  
Изображения eval находятся в datasets/detection_yolo/images/val

### How to use
1. Преобразую координаты в формат YOLO в модуле coco_to_yolo.py
2. Обучаю модель YOLO детектировать все штампы на фото в модуле yolo_detection_fit.py
3. В ручную отбираю фотографий в разных ориентациях и делю на 4 класса: перевёрнутые, повёрнутые влево, повёрнутые в право и правильно стоящие.
4. Разбиваю данные на train и val в соотношении 70/30.
5. Обучаю модель YOLO классифицировать в каком положении находится фото.
6. Пишу алгоритм, который сначала смотрит ориентацию фото, затем переворачивает его при необходимости, далее детектирует штампы,
вычисляет последний стоящий штамп и сохраняет изображение в папку /results в модуле model_results.py

### Processing time
Время обработки папки eval, в которой находилось 660 фото, составило 64.8 секунды на GPU моего ПК.

### Specs
CPU: ryzen 7 5700X3D 8c/16t  
GPU: RX 6800 XT 16GB  
RAM: 32GB ddr4 3200 mhz  
OS: Manjaro GNOME 45.4 (wayland)  