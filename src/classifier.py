#import cv library
import cv2

#using haarcascades classifier
def face_class():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#detect box on cam
def detect_box(vid, classifier):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY) #gray scaled for computational efficiency
    faces = classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces

