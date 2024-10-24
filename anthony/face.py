import cv2
import numpy as np

vid = r"C:\Users\MBEGroup\Documents\TeMoRett\data\rppg-eval\sourcedata\sub-666\sub-666_task-BBSIGIBTask_vid.avi"

## OpenCV basic face detection
deep_learning = True

if not deep_learning:

    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    def resize_square(x, y, w, h, scale):
        # Calculate the center of the square
        center_x = x + w / 2
        center_y = y + h / 2

        # Calculate the new width and height by increasing by the percentage
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Calculate the new top-left corner to keep the square centered
        new_x = int(center_x - new_w / 2)
        new_y = int(center_y - new_h / 2)

        return new_x, new_y, new_w, new_h

    def get_face(img, bounds=(1,1,1,1)):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
        if not isinstance(faces, tuple):
            dist = np.inf
            fnum = 0
            for i, face in enumerate(faces):
                face_dist = np.linalg.norm(bounds[:2] - face[:2])
                if face_dist < dist:
                    dist = face_dist
                    fnum = i

            bounds = resize_square(*faces[fnum], 1.5)
        x, y, w, h = bounds
        cropped = img[y:y+h, x:x+w]
        resized = cv2.resize(cropped, (128, 128))
        return resized, bounds

    # Begin capture
    cap = cv2.VideoCapture(0)

    bounds = (1, 1, 1, 1)
    while True:
        ret, frame = cap.read()

        if not ret:
            continue

        face, bounds = get_face(frame, bounds)

        print(f"Size: {bounds[2]}x{bounds[3]}")

        cv2.imshow("Face", face)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

## Deep learning model face detection

if deep_learning:

    def resize_box(x0, y0, x1, y1, scale, w_max, h_max):
        # Calculate the center of the box
        center_x = (x0 + x1) / 2
        center_y = (y0 + y1) / 2

        # Calculate the width and height of the rectangle
        w = abs(x1 - x0) * scale
        h = abs(y1 - y0) * scale

        # Find the largest side to use as the size of the square
        side = max(w, h)

        # Ensure that the square stays within the bounds
        # Adjust the side length if necessary to fit within the bounds
        side = min(side, w_max, h_max)

        # Calculate new top-left and bottom-right corners for the square
        new_x0 = center_x - side / 2
        new_y0 = center_y - side / 2
        new_x1 = center_x + side / 2
        new_y1 = center_y + side / 2

        # Ensure the new coordinates stay within the bounds (0, 0) and (w_max, h_max)
        if new_x0 < 0:
            new_x0 = 0
            new_x1 = side  # Adjust bottom-right corner accordingly
        elif new_x1 > w_max:
            new_x1 = w_max
            new_x0 = w_max - side

        if new_y0 < 0:
            new_y0 = 0
            new_y1 = side  # Adjust bottom-right corner accordingly
        elif new_y1 > h_max:
            new_y1 = h_max
            new_y0 = h_max - side

        return int(new_x0), int(new_y0), int(new_x1), int(new_y1)

    modelFile = r"anthony\face_detection\res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = r"anthony\face_detection\deploy.prototxt"

    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # Begin capture
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            continue

        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300))

        net.setInput(blob)
        detections = net.forward()

        box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
        # x0, y0, x1, y1 = box.astype(int)
        x0, y0, x1, y1 = resize_box(*box, 1.2, w, h)

        cropped = frame[y0:y1, x0:x1]
        resized = cv2.resize(cropped, (128, 128))

        cv2.imshow("DL Face Detection", resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()