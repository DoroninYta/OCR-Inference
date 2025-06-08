import ultralytics
import torch
import cv2 as cv
from pathlib import Path
from ultralytics import YOLO
import os
import sys
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)
import paddle_ocr.PaddleOCR.tools.infer.predict_rec as infer
import paddle_ocr.PaddleOCR.tools.infer.utility as utility


# добавить трекинг и посмотреть как извлекать инфомрацию из results 
# рисование номера индекса трекера на видео убрать инфу о конфиденсе и меньше линии сделать 
# добавить сегментацию номеров
# добавить отрисовку номера сбоку с индексом трекера
# вспомнить и добавить распознавание номера
class NumberPlateOCR:
    def __init__(self, path_to_video: Path, path_car_detector: Path = "models/yolov8m.pt", path_plate_detector: Path = "models/license_plate_detector.pt",
                path_ocr: Path = "/home/user/3_numbers/paddle_ocr/output/rec/rec_svtr_small_stn_en/300k_with_char_norm_val/infer_0.99_infer/"):
        self.model_car_detector = YOLO(path_car_detector)
        self.model_plate_detector = YOLO(path_plate_detector)
        self.cap = cv.VideoCapture(path_to_video)
        self.model_ocr = self.init_ocr(path_ocr)
        # self.model_ocr = self.init_ocr
    def video_runner(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.car_detect_result = self.predict(frame)
                # gt_imge = self.car_detect_result[0].plot()
                # cv.imshow("Frame", gt_imge)
                # Exit on pressing esc
                if cv.waitKey(1) & 0xFF == 27:
                    break
            else:
                break
    def init_ocr(self, path_to_model):
        args = utility.parse_args()
        args.use_tensorrt = False
        # args.rec_model_dir = "/home/user/3_numbers/paddle_ocr/output/rec/rec_svtr_small_stn_en/300k_with_char_norm_val/infer_0.99_infer/"
        args.rec_model_dir = path_to_model
        args.image_dir = "/home/user/3_numbers/ocr_training_prepairing/data/clear_data/images_clear_DONE"
        args.benchmark = False
        return infer.TextRecognizer(args)
    

    def predict(self, image):
        car_detect_result = self.model_car_detector.track(image, stream=False, save=False, imgsz=1280, classes=2, show=False)
        return car_detect_result
    
def main():
    number_plate_ocr = NumberPlateOCR(Path("/home/user/3_numbers/ocr_show_work/video/53.avi"))
    number_plate_ocr.video_runner()

if __name__ == "__main__":
    main()
