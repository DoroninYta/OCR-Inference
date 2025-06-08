# all proccess - load models: detector+ trcker, plate segmentor, ocr
# later append api using 
# import cv2 as cv
#Import only if not previously imported
#Import only if not previously imported
import cv2
# Create a Video Reader Object.
cap = cv2.VideoCapture("/home/user/3_numbers/ocr_show_work/video/3.mp4")
if cap.isOpened() == False:
    print("Error in opening video stream or file")
#Define the codec for the Video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#Create Video Writer Object
writer = cv2.VideoWriter("/home/user/3_numbers/ocr_show_work/video/3.avi",fourcc, 30, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
frame_number = 0
while cap.isOpened():
    frame_number += 1
    ret, frame = cap.read()
    if frame_number > 
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