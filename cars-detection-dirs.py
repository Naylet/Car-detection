import cv2
import numpy as np
import time
import pry

import matplotlib.pyplot as plt


from random import randint
import time


class Car:
    tracks = []

    def __init__(self, id, x, y, max_age=5):
        self.id = id
        self.x = x
        self.y = y
        self.tracks = []
        self.R = randint(0, 255)
        self.G = randint(0, 255)
        self.B = randint(0, 255)
        self.done = False
        self.is_counted = False
        self.age = 0
        self.max_age = max_age
        self.dir = None

    def getRGB(self):  # For the RGB colour
        return (self.R, self.G, self.B)
    #
    # def getTracks(self):
    #     return self.tracks
    #
    # def getId(self):  # For the ID
    #     return self.id
    #
    # def getState(self):
    #     return self.is_counted
    #
    # def getDir(self):
    #     return self.dir
    #
    # def getX(self):  # for x coordinate
    #     return self.x
    #
    # def getY(self):  # for y coordinate
    #     return self.y

    def updateCoords(self, xn, yn):
        self.age = 0
        self.tracks.append([self.x, self.y])
        self.x = xn
        self.y = yn

    def timed_out(self):
        return self.done

    def going_UP(self, mid_start, mid_end):
        if len(self.tracks) >= 2:
            if self.is_counted == '0':
                if self.tracks[-1][1] < mid_end and self.tracks[-2][1] >= mid_end:
                    self.dir = 'up'
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False

    def going_DOWN(self, mid_start, mid_end):
        if len(self.tracks) >= 2:
            if self.is_counted == False:
                if self.tracks[-1][1] > mid_start and self.tracks[-2][1] <= mid_start:
                    self.dir = 'down'
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False

    def age_one(self):
        self.age += 1
        if self.age > self.max_age:
            self.done = True
        return True


videos = ["surveillance.m4v", "input.mp4"]
cap = cv2.VideoCapture(videos[1])


width = cap.get(3)
height = cap.get(4)
frameArea = height * width
max_contour_area = frameArea / 400

kernel = np.ones((3, 3), np.uint8)


cars = []

pid = 1
max_p_age = 50
cnt_up = 0
cnt_down = 0


# Lines
line_up = int(2 * (height / 5))
line_down = int(3 * (height / 5))

up_limit = int(1 * (height / 5))
down_limit = int(4 * (height / 5))


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
        ret2, bin_img = cv2.threshold(fgmask, 220, 255, cv2.THRESH_BINARY)

        filtered_bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
        filtered_bin_img = cv2.morphologyEx(filtered_bin_img, cv2.MORPH_CLOSE, kernel)



        # old
        # # Find Contours
        # _, contours, hierarchy = cv2.findContours(
        #     filtered_bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #
        # height, width = filtered_bin_img.shape
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

        # Find Contours
        _, contours, hierarchy = cv2.findContours(
            filtered_bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_contour_area:
                ####Tracking######
                m = cv2.moments(contour)
                cx = int(m['m10'] / m['m00'])
                cy = int(m['m01'] / m['m00'])
                rect_x, rect_y, rect_width, rect_height = cv2.boundingRect(contour)
                new = True
                if cy in range(up_limit, down_limit):
                    for car in cars:
                        if abs(rect_x - car.x) <= rect_width and abs(rect_y - car.y) <= rect_height:
                            new = False
                            car.updateCoords(cx, cy)

                            if car.going_UP(line_down, line_up) and car.is_counted == False:
                                cnt_up += 1
                                car.is_counted = True
                                print("ID:", car.id, 'crossed going up at', time.strftime("%c"))
                            elif car.going_DOWN(line_down, line_up) and car.is_counted == False:
                                cnt_down += 1
                                car.is_counted = True
                                print("ID:", car.id, 'crossed going up at', time.strftime("%c"))
                            break
                        if car.is_counted:
                            if car.dir == 'down'and car.y > down_limit:
                                car.done = True
                            elif car.dir == 'up'and car.y < up_limit:
                                car.done = True
                        if car.timed_out():
                            index = cars.index(car)
                            cars.pop(index)
                            del car

                    if new:  # If nothing is detected,create new
                        p = Car(pid, cx, cy, max_p_age)
                        cars.append(p)
                        pid += 1

                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                img = cv2.rectangle(
                    frame, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 255, 0), 2)

        for car in cars:
            cv2.putText(frame, str(car.id), (car.x, car.y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, car.getRGB(), 1, cv2.LINE_AA)

        cv2.imshow('Frame', frame)
        cv2.imshow('Bin', bin_img)
        cv2.imshow('Filtered Bin', filtered_bin_img)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
