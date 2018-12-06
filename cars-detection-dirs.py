import cv2
import numpy as np
import time
#import pry
import matplotlib.pyplot as plt
from random import randint


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

    def updateCoords(self, xn, yn):
        self.age = 0
        self.tracks.append([self.x, self.y])
        self.x = xn
        self.y = yn

    def timed_out(self):
        return self.done

    def crossed_up(self, mid_start, mid_end):
        if len(self.tracks) >= 2:
            if self.is_counted == False:
                if self.tracks[-1][1] < mid_end and self.tracks[-2][1] >= mid_end:
                    self.dir = 'up'
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False

    def crossed_down(self, mid_start, mid_end):
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


videos = ["surveillance.m4v", "input.mp4", "videoplayback.mp4", "night.mp4", "night2.mp4", "counting.mp4"]
cap = cv2.VideoCapture(videos[1])


width = int(cap.get(3))
height = int(cap.get(4))
frameArea = height * width


cars = []
pid = 1
cnt_up = 0
cnt_down = 0

#parametry do sterowania
counting_line_up = int(height * 3.5 / 5)
counting_line_down = int(height * 4 / 5)
max_p_age = 5
max_contour_area = frameArea / 400

up_limit = int(counting_line_up * 5 / 6)
down_limit = int(counting_line_down * 7 / 6)

font = cv2.FONT_HERSHEY_SIMPLEX

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))


def show(img):
    plt.imshow(img)
    plt.title('Matplotlib')
    plt.show()


def filter_img(img):
    _, bin_img = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)

    # Fill any small holes
    closing = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)

    # Remove noise
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    # Dilate to merge adjacent blobs
    dilation = cv2.dilate(opening, kernel, iterations=2)

    return dilation


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

while(cap.isOpened()):
    ret, frame = cap.read()
    fgmask = bg_subtractor.apply(frame, None, 0.001)
    for car in cars:
        car.age_one()

    if ret:

        cv2.line(frame, (0, counting_line_up), (width, counting_line_up), (0, 255, 0), 2)
        cv2.line(frame, (0, counting_line_down), (width, counting_line_down), (255, 153, 51), 2)

        filtered_bin_img = filter_img(fgmask)
        time.sleep(0.02)

        _, contours, hierarchy = cv2.findContours(filtered_bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_contour_area:

                # --------------Tracking---------------
                m = cv2.moments(contour)
                cx = int(m['m10'] / m['m00'])
                cy = int(m['m01'] / m['m00'])
                new = True

                if cy in range(up_limit, down_limit):  # filters out contours that are above line (y starts at top)

                    # centroid marker
                    cv2.drawMarker(frame, (cx, cy), (0, 255, 255), cv2.MARKER_STAR, markerSize=5, thickness=3,
                                   line_type=cv2.LINE_AA)

                    # x,y is top left corner
                    rect_x, rect_y, rect_width, rect_height = cv2.boundingRect(contour)

                    # creates a rectangle around contour
                    cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 255, 0), 2)

                    for car in cars:
                        if abs(rect_x - car.x) <= rect_width and abs(rect_y - car.y) <= rect_height:
                            new = False
                            car.updateCoords(cx, cy)

                            if car.crossed_up(counting_line_down, counting_line_up):
                                cnt_up += 1
                                car.is_counted = True
                                cv2.line(frame, (0, counting_line_up), (width, counting_line_up), (0, 0, 255), 2)
                                print("ID:", car.id, 'crossed going up at', time.strftime("%c"))
                            elif car.crossed_down(counting_line_down, counting_line_up):
                                cnt_down += 1
                                car.is_counted = True
                                cv2.line(frame, (0, counting_line_down), (width, counting_line_down), (0, 0, 255), 2)

                                print("ID:", car.id, 'crossed going down at', time.strftime("%c"))
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

                    for car in cars:
                        cv2.putText(frame, str(car.id), (car.x, car.y), font, 0.3, car.getRGB(), 1, cv2.LINE_AA)

        str_up='UP: '+str(cnt_up)
        str_down='DOWN: '+str(cnt_down)

        cv2.putText(frame, str_up, (10, 40), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, str_up, (10, 40), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, str_down, (10, 90), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, str_down, (10, 90), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        cv2.line(frame, (0, up_limit), (width, up_limit), (153, 255, 102), 2)
        cv2.line(frame, (0, down_limit), (width, down_limit), (255, 204, 102), 2)

        cv2.imshow('Filtered Bin', filtered_bin_img)
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
