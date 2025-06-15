import numpy as np
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
from collections import defaultdict
import timm

# добавить трекинг и посмотреть как извлекать инфомрацию из results 
# рисование номера индекса трекера на видео убрать инфу о конфиденсе и меньше линии сделать 
# добавить сегментацию номеров
# добавить отрисовку номера сбоку с индексом трекера
# вспомнить и добавить распознавание номера
class NumberPlateOCR:
    def __init__(self, path_to_video: Path, path_car_detector: Path = "models/yolov8m.pt", path_plate_detector: Path = "models/license_plate_detector.pt",
                path_ocr: Path = "/home/user/3_numbers/paddle_ocr/output/rec/rec_svtr_small_stn_en/300k_with_char_norm_val/infer_0.99_infer/"):
        self.device = "cuda" if torch.cuda.is_available else "cpu"
        self.model_car_detector = YOLO(path_car_detector)
        self.model_plate_detector = YOLO(path_plate_detector)
        self.cap = cv.VideoCapture(path_to_video)
        self.model_ocr = self.init_ocr(path_ocr)
        self.model_classifier = timm.create_model(model_name="convnext_pico.d1_in1k", num_classes=2, checkpoint_path="models/model_best.pth.tar").to(self.device)
        self.dict_results = defaultdict(lambda: {"car_images": [], "plate_images" : [], "plate_rec": None}) #{track_id : {images : [], plate_rec : str}}


    def video_runner(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.car_detect_result = self.predict(frame)
                self.processing_result(frame)
                # gt_imge = self.car_detect_result[0].plot()

                # cv.imshow("Frame", gt_imge)
                # Exit on pressing esc
                if cv.waitKey(1) & 0xFF == 27:
                    break
            else:
                break
    
    def processing_result(self, frame):
        result = self.car_detect_result[0].boxes
        cv.imshow("window", frame)
        for i in range(len(result.id)):
            crop_image = frame[int(result.xyxy[i][1]):int(result.xyxy[i][3]), int(result.xyxy[i][0]):int(result.xyxy[i][2])]
            h, w, _ = crop_image.shape
            if h * w > 50000:
                self.dict_results[result.id[i]]["car_images"].append(crop_image)
                plate_image, boxes = self.plate_detecting(crop_image)
                if plate_image is not None and self.check_image(plate_image) == 0:
                    crop_image_rect = cv.rectangle(crop_image, boxes[0], boxes[1], (0, 0, 255), 1)
                    self.dict_results[result.id[i]]["plate_images"].append(plate_image)
                    plate_rec = self.model_ocr(np.expand_dims(plate_image, axis=0))
                    print(plate_rec[0][0][0])
                    self.dict_results[result.id[i]]["plate_rec"] = plate_rec[0][0][0]
                    cv.imshow("car", crop_image_rect)
                    cv.waitKey(0)
                    
    def check_image(self, image):
        # img = torch.Tensor(image).unsqueeze(0).to(self.device)
        cv.imwrite("test.jpg", image)
        img = image
        print(img.shape)
        res = self.model_classifier(img).softmax(axis=1)
        print("classifier res", res)
        return res.argmax(axis=1)

    def plate_detecting(self, image):
        result = self.model_plate_detector(image)
        res = result[0].boxes
        if len(res.cls) > 0:
            crop_image = image[int(res.xyxy[0][1]):int(res.xyxy[0][3]), int(res.xyxy[0][0]):int(res.xyxy[0][2])]
            return crop_image, [(int(res.xyxy[0][0]), int(res.xyxy[0][1])), (int(res.xyxy[0][2]), int(res.xyxy[0][3]))]
        return None, None
    
    def init_ocr(self, path_to_model):
        args = utility.parse_args()
        args.use_tensorrt = True
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
