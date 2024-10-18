import cv2
import numpy as np

modelFile = r"face_detection\res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = r"face_detection\deploy.prototxt"

net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def get_face(img, bounds=(1,1,1,1)):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    if not isinstance(faces, tuple):
        bounds = faces[0]
    x, y, w, h = bounds
    cropped = img[y:y+h, x:x+w]
    resized = cv2.resize(cropped, (128, 128))
    return resized, bounds

cap = cv2.VideoCapture(0)


bounds = (1, 1, 1, 1)
while True:
    ret, frame = cap.read()

    if not ret:
        continue

    face, bounds = get_face(frame)

    cv2.imshow("Face", face)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


h, w = frame.shape[:2]

blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300))

net.setInput(blob)
detections = net.forward()

box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
x0, y0, x1, y1 = box.astype(int)

cropped = frame[x0:x1, y0:x1]

cv2.imshow("DL Face Detection", cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()