import numpy as np
import torch
import cv2 as cv
from pathlib import Path
from ultralytics import YOLO
import os
import sys
import re
from collections import defaultdict
import timm

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)
import paddle_ocr.PaddleOCR.tools.infer.predict_rec as infer
import paddle_ocr.PaddleOCR.tools.infer.utility as utility

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

class NumberPlateOCR:
    def __init__(self, path_to_video: Path, path_car_detector: Path = "models/yolov8m.pt", 
                path_plate_detector: Path = "models/license_plate_detector.pt",
                path_ocr: Path = "/home/user/3_numbers/paddle_ocr/output/rec/rec_svtr_small_stn_en/300k_with_char_norm_val/infer_0.99_infer/", 
                path_classifier: Path = "models/model_best.pth.tar"):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_car_detector = YOLO(path_car_detector)
        self.model_plate_detector = YOLO(path_plate_detector)
        self.cap = cv.VideoCapture(str(path_to_video))
        self.model_ocr = self.init_ocr(path_ocr)
        self.create_classifier(path_classifier)
        
        # Увеличим высоту верхней области для лучшего отображения
        self.top_border_height = 150
        self.bottom_border_height = 80
        
        # Словарь для хранения результатов
        self.track_data = defaultdict(lambda: {
            "car_count": 0,
            "plate_count": 0,
            "last_plate_img": None,
            "confirmed_plate": None,
            "recognition_history": [],
            "last_position": None
        })
        
        # Параметры фильтров
        self.min_car_area = 50000       # Минимальная площадь автомобиля для обработки
        self.min_plate_area = 1000      # Минимальная площадь номера для распознавания
        self.min_plate_ratio = 2.0      # Минимальное соотношение сторон номера (ширина/высота)
        self.confirmation_threshold = 5 # Количество успешных распознаваний для подтверждения
        self.max_history = 20           # Максимальное количество хранимых распознаваний
        
        # Фиксированные позиции для отображения
        self.fixed_positions = {}

    def video_runner(self):
        frame_count = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # Создаем кадр с черными областями
            h, w = frame.shape[:2]
            bordered_frame = np.zeros(
                (h + self.top_border_height + self.bottom_border_height, w, 3),
                dtype=np.uint8
            )
            bordered_frame[self.top_border_height:self.top_border_height+h, :] = frame

            # Детекция автомобилей с трекингом
            self.car_detect_result = self.model_car_detector.track(
                frame, stream=False, save=False, imgsz=1280, classes=2, verbose=False, persist=True
            )

            # Обработка и визуализация
            self.process_frame(bordered_frame, self.top_border_height)

            # Отображение результата
            cv.imshow("License Plate Recognition", bordered_frame)
            
            # Сохраняем кадр каждые 50 фреймов
            frame_count += 1
            if frame_count % 50 == 0:
                cv.imwrite(f"output/frame_{frame_count}.jpg", bordered_frame)
                
            if cv.waitKey(1) & 0xFF == 27:
                break

        self.cap.release()
        cv.destroyAllWindows()

    def process_frame(self, bordered_frame, y_offset):
        # Проверка наличия результатов трекинга
        if len(self.car_detect_result) == 0 or self.car_detect_result[0].boxes.id is None:
            return

        current_recognitions = []
        boxes = self.car_detect_result[0].boxes
        track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else []

        for i in range(len(boxes.xyxy)):
            # Если есть ID трекера - используем, иначе -1
            track_id = track_ids[i] if i < len(track_ids) else -1
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
            
            # Обновляем данные о треке
            track_info = self.track_data[track_id]
            track_info["car_count"] += 1
            track_info["last_position"] = (x1, y1, x2, y2)

            # Рисуем рамку автомобиля
            cv.rectangle(
                bordered_frame,
                (x1, y1 + y_offset),
                (x2, y2 + y_offset),
                (0, 255, 0),
                2
            )
            
            # Определяем цвет рамки в зависимости от статуса распознавания
            border_color = (0, 255, 0)  # Зеленый - не распознан
            if track_info["confirmed_plate"]:
                border_color = (0, 165, 255)  # Оранжевый - распознан
            if track_info["plate_count"] >= self.confirmation_threshold:
                border_color = (0, 0, 255)  # Красный - подтвержден
            
            # Рисуем внутреннюю рамку с цветом статуса
            cv.rectangle(
                bordered_frame,
                (x1 + 2, y1 + y_offset + 2),
                (x2 - 2, y2 + y_offset - 2),
                border_color,
                1
            )
            
            # Подписываем ID
            cv.putText(
                bordered_frame,
                f"ID: {track_id}",
                (x1, y1 + y_offset - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # Пропускаем маленькие автомобили
            car_area = (x2 - x1) * (y2 - y1)
            if car_area < self.min_car_area:
                continue

            # Проверяем, не подтвержден ли уже номер
            if track_info["plate_count"] >= self.confirmation_threshold:
                # Используем подтвержденный номер
                plate_text = track_info["confirmed_plate"]
                current_recognitions.append((track_info["last_plate_img"], plate_text, track_id))
                continue

            # Обрезаем область автомобиля
            car_roi = bordered_frame[y1+y_offset:y2+y_offset, x1:x2]
            if car_roi.size == 0:
                continue

            # Детекция номеров
            plate_results = self.model_plate_detector(car_roi, verbose=False)
            if len(plate_results) == 0 or len(plate_results[0].boxes) == 0:
                continue

            # Обрабатываем первое обнаружение номера
            plate_box = plate_results[0].boxes.xyxy[0].cpu().numpy().astype(int)
            px1, py1, px2, py2 = plate_box
            
            # Проверяем размеры номера
            plate_width = px2 - px1
            plate_height = py2 - py1
            plate_area = plate_width * plate_height
            plate_ratio = plate_width / plate_height if plate_height > 0 else 0
            
            # Пропускаем слишком маленькие номера или с неправильным соотношением
            if (plate_area < self.min_plate_area or 
                plate_ratio < self.min_plate_ratio or
                plate_height < 10 or plate_width < 40):
                continue

            # Рисуем рамку номера
            abs_px1, abs_py1 = x1 + px1, y1 + y_offset + py1
            abs_px2, abs_py2 = x1 + px2, y1 + y_offset + py2
            cv.rectangle(
                bordered_frame,
                (abs_px1, abs_py1),
                (abs_px2, abs_py2),
                (0, 0, 255),
                2
            )

            # Распознаем номер
            plate_img = car_roi[py1:py2, px1:px2]
            if plate_img.size == 0:
                continue
                
            # Сохраняем изображение номера
            track_info["last_plate_img"] = plate_img
            
            # Распознаем текст номера
            plate_rec = self.model_ocr(np.expand_dims(plate_img, axis=0))
            plate_text = plate_rec[0][0][0] if plate_rec[0] else ""
            
            # Фильтруем невалидные номера
            if self.is_valid_plate(plate_text):
                track_info["plate_count"] += 1
                track_info["recognition_history"].append(plate_text)
                
                # Ограничиваем историю распознаваний
                if len(track_info["recognition_history"]) > self.max_history:
                    track_info["recognition_history"].pop(0)
                
                # Определяем наиболее частый результат
                if track_info["recognition_history"]:
                    plate_text = max(set(track_info["recognition_history"]), 
                                    key=track_info["recognition_history"].count)
                
                # Подтверждаем номер после нескольких успешных распознаваний
                if track_info["plate_count"] >= self.confirmation_threshold:
                    track_info["confirmed_plate"] = plate_text
                    print(f"Confirmed plate for ID {track_id}: {plate_text}")

            # Сохраняем для отображения
            current_recognitions.append((plate_img, plate_text, track_id))

        # Отображаем распознанные номера
        self.display_recognitions(bordered_frame, current_recognitions)

    def is_valid_plate(self, plate_text):
        """Проверяет, соответствует ли текст формату номерного знака"""
        if not plate_text:
            return False
            
        # Простая проверка на наличие букв и цифр
        has_letters = any(char.isalpha() for char in plate_text)
        has_digits = any(char.isdigit() for char in plate_text)
        
        # Проверка минимальной длины
        min_length = 8
        if len(plate_text) < min_length:
            return False
            
        return has_letters and has_digits

    def display_recognitions(self, frame, recognitions):
        # Верхняя область: текущие распознавания
        y_text_top = 30
        x_start_top = 10
        
        # Создаем фиксированные позиции для треков
        for plate_img, plate_text, track_id in recognitions:
            if track_id not in self.fixed_positions:
                # Назначаем новую позицию
                next_pos = len(self.fixed_positions) * 250
                if next_pos < frame.shape[1] - 300:
                    self.fixed_positions[track_id] = next_pos
                else:
                    # Если нет места, используем последнюю позицию
                    self.fixed_positions[track_id] = frame.shape[1] - 300
            
            x_pos = self.fixed_positions[track_id]
            
            # Увеличиваем размер миниатюры номера
            if plate_img is not None and plate_img.size > 0:
                small_plate = cv.resize(plate_img, (200, 50))
                frame[20:70, x_pos:x_pos+200] = small_plate
                
                # Выводим статус распознавания
                status = "UNCONFIRMED"
                color = (200, 200, 200)
                track_info = self.track_data[track_id]
                
                if track_info["confirmed_plate"]:
                    status = "CONFIRMED"
                    color = (0, 255, 0)
                    plate_text = track_info["confirmed_plate"]
                elif track_info["plate_count"] > 0:
                    status = f"RECOGNIZING ({track_info['plate_count']}/{self.confirmation_threshold})"
                    color = (0, 165, 255)
                
                # Выводим текст
                cv.putText(
                    frame,
                    f"ID {track_id}: {plate_text}",
                    (x_pos, 90),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                
                cv.putText(
                    frame,
                    status,
                    (x_pos, 120),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    1
                )

        # Нижняя область: история по track_id
        y_text_bottom = frame.shape[0] - self.bottom_border_height + 30
        x_start_bottom = 10
        
        # Отображаем только подтвержденные номера
        confirmed_plates = []
        for track_id, data in self.track_data.items():
            if data["confirmed_plate"]:
                confirmed_plates.append((track_id, data["confirmed_plate"]))
        
        # Сортируем по ID для стабильного отображения
        confirmed_plates.sort(key=lambda x: x[0])
        
        for track_id, plate_text in confirmed_plates:
            cv.putText(
                frame,
                f"ID {track_id}: {plate_text}",
                (x_start_bottom, y_text_bottom),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
            y_text_bottom += 30

    # Остальные методы без изменений
    def create_classifier(self, path_checkpoint):
        print(path_checkpoint)
        self.model_classifier = timm.create_model(model_name="convnext_pico.d1_in1k", num_classes=2, checkpoint_path=path_checkpoint).to(self.device)
        self.test_config = timm.data.transforms_factory.create_transform(
            input_size=(3, 224, 224),
            is_training=False,
            interpolation='bicubic',
            crop_pct=1.0,
            crop_border_pixels=0,
            use_prefetcher=False,
            normalize=True,
        )

    def check_image(self, image):
        img = cv.resize(image, (224, 224))
        img_tensor = (
            torch.from_numpy(img)
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            res = self.model_classifier(img_tensor).softmax(dim=1)
        print("classifier res:", res)
        return bool(res.argmax() == 0)

    def plate_detecting(self, image):
        result = self.model_plate_detector(image, verbose=False)
        res = result[0].boxes
        if len(res.cls) > 0:
            crop_image = image[int(res.xyxy[0][1]):int(res.xyxy[0][3]), int(res.xyxy[0][0]):int(res.xyxy[0][2])]
            return crop_image, [(int(res.xyxy[0][0]), int(res.xyxy[0][1])), (int(res.xyxy[0][2]), int(res.xyxy[0][3]))]
        return None, None

    def init_ocr(self, path_to_model):
        args = utility.parse_args()
        args.use_tensorrt = False
        args.rec_model_dir = path_to_model
        args.image_dir = "/home/user/3_numbers/ocr_training_prepairing/data/clear_data/images_clear_DONE"
        args.benchmark = False
        return infer.TextRecognizer(args)

def main():
    number_plate_ocr = NumberPlateOCR(Path("/home/user/3_numbers/ocr_show_work/video/53.avi"))
    number_plate_ocr.video_runner()

if __name__ == "__main__":
    main()