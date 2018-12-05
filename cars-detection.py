import cv2
import numpy as np
import time
#import pry

import matplotlib.pyplot as plt

videos = ["surveillance.m4v", "input.mp4"]
cap = cv2.VideoCapture(videos[1])


width = int(cap.get(3))
height = int(cap.get(4))

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # kernel to apply to the morphology


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
        frame = cv2.resize(frame, (0, 0), None, .5, .5)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # converts image to gray
        inst.apply(gray, None, 0.001)
        i += 1
        if i >= num:
            print('Trained!')
            return cap


bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, detectShadows=True)

# skipping 500 frames to train bg subtractor
train_bg_subtractor(bg_subtractor, cap, num=500)

while(cap.isOpened() and not end):
    ret, frame = cap.read()

    if ret:
        frame = cv2.resize(frame, (0, 0), None, .5, .5)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # converts image to gray

        fgmask = bg_subtractor.apply(gray, None, 0.001)

        # Closing i.e First Dilate then Erode
        mask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

        # Opening i.e First Erode the dilate
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Binarization // try to use Otsu Binarization ;)
        _, imBin = cv2.threshold(mask, 220, 255, cv2.THRESH_BINARY)

        # Find Contours
        _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # line created to stop counting contours
        lineypos = 225
        cv2.line(frame, (0, lineypos), (width, lineypos), (255, 0, 0), 2)

        # line y position created to count contours
        lineypos2 = 250
        cv2.line(frame, (0, lineypos2), (width, lineypos2), (0, 255, 0), 2)

        # min area for contours in case a bunch of small noise contours are created
        minarea = 300

        # max area for contours, can be quite large for buses
        maxarea = 50000

        for i in range(len(contours)):  # cycles through all contours in current frame

            if hierarchy[0, i, 3] == -1:  # using hierarchy to only count parent contours (contours not within others)

                area = cv2.contourArea(contours[i])  # area of contour

                if minarea < area < maxarea:  # area threshold for contour

                    # calculating centroids of contours
                    cnt = contours[i]
                    M = cv2.moments(cnt)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])

                    if cy > lineypos:  # filters out contours that are above line (y starts at top)

                        # gets bounding points of contour to create rectangle
                        # x,y is top left corner and w,h is width and height
                        x, y, w, h = cv2.boundingRect(cnt)

                        # creates a rectangle around contour
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                        # Prints centroid text in order to double check later on
                        #cv2.putText(frame, str(cx) + "," + str(cy), (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 0, 255), 1)

                        cv2.drawMarker(frame, (cx, cy), (0, 0, 255), cv2.MARKER_STAR, markerSize=5, thickness=1, line_type=cv2.LINE_AA)


        # height, width = mask.shape
        # min_x, min_y = width, height
        # max_x = max_y = 0
        #
        # # computes the bounding box for the contour, and draws it on the frame,
        # for contour in contours:
        #     (x, y, w, h) = cv2.boundingRect(contour)
        #     min_x, max_x = min(x, min_x), max(x + w, max_x)
        #     min_y, max_y = min(y, min_y), max(y + h, max_y)
        #     if w > 80 and h > 80:
        #         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        #
        # if max_x - min_x > 0 and max_y - min_y > 0:
        #     cv2.rectangle(frame, (min_x, min_y),
        #                   (max_x, max_y), (255, 0, 0), 2)

        cv2.imshow('Frame', frame)
        cv2.imshow('Bin', imBin)
        cv2.imshow('Foreground', mask)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
