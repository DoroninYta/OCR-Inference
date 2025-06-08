# all proccess - load models: detector+ trcker, plate segmentor, ocr
# later append api using 
# import cv2 as cv
#Import only if not previously imported
#Import only if not previously imported
import cv2
from pathlib import Path
import argparse
# Create a Video Reader Object.
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', type=Path, default="/home/user/3_numbers/ocr_show_work/video")
    parser.add_argument('--video_name', type=str, default="3.mp4")
    parser.add_argument('--count_frames', type=int, default=6000)
    return parser.parse_args()

def func_rewrite_video(folder_path : Path, video_name : Path, count_frames):
    cap = cv2.VideoCapture(folder_path / video_name)
    if cap.isOpened() == False:
        print("Error in opening video stream or file")
    #Define the codec for the Video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #Create Video Writer Object
    writer = cv2.VideoWriter((folder_path / video_name).with_suffix(".avi"),fourcc, 30, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    frame_number = 0
    while cap.isOpened():
        frame_number += 1
        ret, frame = cap.read()
        if frame_number > count_frames:
            break
        if ret:
            writer.write(frame)
            cv2.imshow("Frame",frame)
            # Exit on pressing esc
            if cv2.waitKey(20) & 0xFF == 27:
                break
        else:
            break
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

folder_path = Path("/home/user/3_numbers/ocr_show_work/video")
video_name = "3.mp4"
count_frames = 6000
def main():
    args = parse_args()
    func_rewrite_video(args.folder_path, args.video_name, args.count_frames)

if __name__ == "__main__":
    main()