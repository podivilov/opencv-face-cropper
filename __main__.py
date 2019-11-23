#!/usr/bin/env python3

import cv2
import sys
import os

class FaceCropper(object):
    CASCADE_PATH = "data/haarcascades/haarcascade_frontalface_default.xml"

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(self.CASCADE_PATH)

    def generate(self, image_path):
        img = cv2.imread(image_path)
        if (img is None):
            print("Can't open image file")
            return 0

        faces = self.face_cascade.detectMultiScale(img, 1.1, 3, minSize=(100, 100))
        if (faces is None):
            print('Failed to detect face')
            return 0

        facecnt = len(faces)
        print("Detected faces: %d" % facecnt)
        i = 0
        height, width = img.shape[:2]

        for (x, y, w, h) in faces:
            r = max(w, h) / 2
            centerx = x + w / 2
            centery = y + h / 2
            nx = int(centerx - r)
            ny = int(centery - r)
            nr = int(r * 2)

            faceimg = img[ny:ny+nr, nx:nx+nr]
            lastimg = cv2.resize(faceimg, (147, 180))
            i += 1
            cv2.imwrite(os.path.splitext(sys.argv[1])[0] + " (обработанный) вариант №%d.jpg" % i, lastimg)


if __name__ == '__main__':
    args = sys.argv
    argc = len(args)

    if (argc != 2):
        print('Usage: %s [image file]' % args[0])
        quit()

    detecter = FaceCropper()
    detecter.generate(args[1])
