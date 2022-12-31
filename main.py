import cv2
import numpy as np

from TraficSignClassification.YoloDetector import Detector

IMAGE_WIDTH = 150
IMAGE_HEIGHT = 150
LABEL_HEIGHT = 180
SIGN_SCREEN_SHAPE = (180, 900, 3)

detector = Detector(
    weights_file_path="model4/yolov3_training_last.weights",
    config_file_path="model4/yolov3_testing.cfg",
    classes_file_path="model4/classes.txt",
    confidence_threshold=.1,
    nms_threshold=.1
)

cap = cv2.VideoCapture(r"images/crossings2.mp4")

while cap.isOpened():
    _, img = cap.read()

    signs_screen = np.zeros(SIGN_SCREEN_SHAPE, np.uint8)

    start_x = 0
    end_x = 150

    detections = detector.detect(img)
    for detection in detections:
        x1, y1 = detection.x, detection.y
        x2, y2 = detection.x + detection.w, detection.y + detection.h
        x1, y1, x2, y2 = abs(x1), abs(y1), abs(x2), abs(y2)

        sign_area = img[y1:y2, x1:x2]
        sign_area = cv2.resize(sign_area, (150, 150))
        signs_screen[0:IMAGE_HEIGHT, start_x:end_x] = sign_area

        label_img = signs_screen[IMAGE_HEIGHT:LABEL_HEIGHT, start_x:end_x]
        cv2.putText(label_img, detection.class_name, (5, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

        cv2.rectangle(img, (x1, y1), (x2, y2), detection.color, 3)
        cv2.putText(img, f"{detection.class_name} {int(round(detection.detections_conf, 2) * 100)}%",
                    (x1, y1 - 15), cv2.FONT_HERSHEY_PLAIN, 1.5, detection.color, 2)

        start_x += IMAGE_WIDTH
        end_x += IMAGE_WIDTH

    cv2.imshow("resImage", img)
    cv2.imshow("SignScreen", signs_screen)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
