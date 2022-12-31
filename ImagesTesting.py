import glob
import cv2
import numpy as np

from TraficSignClassification.YoloDetector import Detector

detector = Detector(
    weights_file_path="model3/yolov3_training_last.weights",
    config_file_path="model4/yolov3_testing.cfg",
    classes_file_path="model4/classes.txt",
    confidence_threshold=.1,
    nms_threshold=.1
)

files = glob.glob(r"images/*.png")

for file in files:
    img = cv2.imread(file)
    signs_screen = np.zeros((150, 900, 3), np.uint8)

    start_x = 0
    end_x = 150

    detections = detector.detect(img)
    for detection in detections:
        try:
            x1, y1 = detection.x, detection.y
            x2, y2 = detection.x + detection.w, detection.y + detection.h

            x1, y1, x2, y2 = abs(x1), abs(y1), abs(x2), abs(y2)

            sign_area = img[y1:y2, x1:x2]
            sign_area = cv2.resize(sign_area, (150, 150))

            print(signs_screen.shape)
            signs_screen[0:150, start_x:end_x] = sign_area

            cv2.rectangle(img, (x1, y1), (x2, y2), detection.color, 3)
            cv2.putText(img, f"{detection.class_name} {int(round(detection.detections_conf, 2) * 100)}%",
                        (x1, y1 - 15), cv2.FONT_HERSHEY_PLAIN, 1.5, detection.color, 2)
            print(f"{detection.class_name} {int(round(detection.detections_conf, 2) * 100)}%")

            start_x += 150
            end_x += 150
        except Exception:
            pass

    cv2.imshow("resImage", img)
    cv2.imshow("SignScreen", signs_screen)
    cv2.waitKey(0)
