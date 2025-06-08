import ultralytics
import torch
import cv2 as cv
from pathlib import Path
from ultralytics import YOLO
class NumberPlateOCR:
    def __init__(self, path_to_video: Path,  ):
        self.model = YOLO("yolov8m.pt")
        self.cap = cv.VideoCapture(path_to_video)
    
    def  video_runner(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                result = self.predict(frame)
                gt_imge = result[0].plot()
                cv.imshow("Frame", gt_imge)
                # Exit on pressing esc
                if cv.waitKey(20) & 0xFF == 27:
                    break
            else:
                break
            
    def predict(self, image):
        results = self.model.predict(image, stream=False, save = False)
        return results
    
def main():
    number_plate_ocr = NumberPlateOCR(Path("/home/user/3_numbers/ocr_show_work/video/53.avi"))
    number_plate_ocr.video_runner()

if __name__ == "__main__":
    main()
