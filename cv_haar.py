"""
Classic OpenCV car detection using a Haar cascade. Pure cv2, no deep learning.
Detects cars only (no make/model). Good for learning, weak on busy scenes.

Get a car cascade xml (e.g. 'cars.xml' / 'haarcascade_car.xml') from:
  https://github.com/andrewssobral/vehicle_detection_haarcascades
Run: python cv_haar.py image.jpg cars.xml
"""
import sys
import cv2

def detect(image_path, cascade_path="cars.xml"):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cascade = cv2.CascadeClassifier(cascade_path)
    cars = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4,
                                    minSize=(40, 40))

    for (x, y, w, h) in cars:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, "car", (x, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imwrite("output.jpg", img)
    print(f"found {len(cars)} cars -> output.jpg")

if __name__ == "__main__":
    detect(sys.argv[1] if len(sys.argv) > 1 else "cars.jpg",
           sys.argv[2] if len(sys.argv) > 2 else "cars.xml")
