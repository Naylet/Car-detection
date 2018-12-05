import cv2
import numpy as np
import time
import pry

import matplotlib.pyplot as plt

videos = ["surveillance.m4v", "input.mp4"]
cap = cv2.VideoCapture(videos[1])


w = cap.get(3)
h = cap.get(4)
frameArea = h * w
areaTH = frameArea / 400

kernal = np.ones((3, 3), np.uint8)


end = False  # use to quit from pry


def show(img):
    plt.imshow(img)
    plt.title('Matplotlib')
    plt.show()


def train_bg_subtractor(inst, cap, num=500):
    print('Training BG Subtractor...')
    i = 0
    while i < 500:
        _ret, frame = cap.read()
        inst.apply(frame, None, 0.001)
        i += 1
        if i >= num:
            print('Trained!')
            return cap


bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500, detectShadows=True)

# skipping 500 frames to train bg subtractor
train_bg_subtractor(bg_subtractor, cap, num=500)

while(cap.isOpened() and not end):
    ret, frame = cap.read()
    fgmask = bg_subtractor.apply(frame, None, 0.001)

    if ret:
        # Binarization // try to use Otsu Binarization ;)
        ret, imBin = cv2.threshold(fgmask, 230, 255, cv2.THRESH_BINARY)
        # OPening i.e First Erode the dilate
        mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernal)

        # Closing i.e First Dilate then Erode
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal)

        # Find Contours
        _, contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        height, width = mask.shape
        min_x, min_y = width, height
        max_x = max_y = 0

        # computes the bounding box for the contour, and draws it on the frame,
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            min_x, max_x = min(x, min_x), max(x + w, max_x)
            min_y, max_y = min(y, min_y), max(y + h, max_y)
            if w > 80 and h > 80:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if max_x - min_x > 0 and max_y - min_y > 0:
            cv2.rectangle(frame, (min_x, min_y),
                          (max_x, max_y), (255, 0, 0), 2)

        cv2.imshow('Frame', frame)
        cv2.imshow('Bin', imBin)
        cv2.imshow('Foreground', mask)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
