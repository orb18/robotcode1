#import RPi.GPIO as GPIO
import gpio as GPIO
import time
import cv2
from datetime import datetime
from threading import Thread
import math
import numpy as np


saveangles = []



#import video.

#find angles
#find delta angles
#run logic to find direction of fly.

#from direction steer robot.




def wingAverage(saveangles):
    #initialising the counters for each itteration
    evenTotal = 0
    oddTotal = 0
    turningNum = 0

    for j in range(0, 8):
        if int(saveangles[j]) > 0:
            evenTotal = evenTotal + 1

        elif int(saveangles[j]) < 0:
            oddTotal = oddTotal + 1


        #returns the value based on the majority of the angles
        if evenTotal > oddTotal:
            turningNum = 1
        elif evenTotal < oddTotal:
            turningNum = -1
        else:
            turningNum = 0

    rorbotTurning(turningNum)
#-------------------------------------------------------------------



def rorbotTurning(turningNum):
    print('start...')
    #GPIO.setmode(GPIO.BOARD)

    #GPIO.setup(32, GPIO.OUT)
    #GPIO.setup(33, GPIO.OUT)
    #GPIO.setup(35, GPIO.OUT)
    #GPIO.setup(36, GPIO.OUT)
    #GPIO.setup(37, GPIO.OUT)
    #GPIO.setup(38, GPIO.OUT)

    #pwm0 = GPIO.PWM(32, 2000)
    #pwm1 = GPIO.PWM(33, 20000)

    if turningNum == 1:

        #pwm0.start(40)
        #pwm1.start(100)
        print('turning left')
        #GPIO.output(33, GPIO.HIGH)
        #GPIO.output(37, GPIO.HIGH)
        time.sleep(3)

    elif turningNum == -1:
        print('turning right')
        #pwm0.ChangeDutyCycle(100)
        #pwm1.ChangeDutyCycle(40)
        time.sleep(3)

    else:
        print('no turn')
        #pwm0.ChangeDutyCycle(100)
        #pwm1.ChangeDutyCycle(100)
        time.sleep(3)

    #GPIO.output(33, GPIO.LOW)
    #GPIO.output(37, GPIO.LOW)
    time.sleep(3)

    #GPIO.output(37, GPIO.LOW)
    #pwm0.stop()
    #pwm1.stop()

    #GPIO.cleanup()
    print('end')




yshift = -50  # Use these parameters to adjust the position of the fly in the image. Negative means moving left, positive moves it to the right
xshift = -15

saveangles = []

def findanglel(upmost, bottommost,
               centre):  # Calculates the angle between the top and bottom of the wingbeat amplitude and the winghinge (centre)
    dy_up = (upmost[1] - centre[1])
    dx_up = (upmost[0] - centre[0])
    dy_bottom = (bottommost[1] - centre[1])
    dx_bottom = (bottommost[0] - centre[0])
    if dy_up <= 0 and dx_up < 0:
        angle_up = math.atan(dy_up / dx_up)
    elif dx_up >= 0:
        angle_up = math.pi / 2  # Avoid divide by 0 and also realistically the wing can't have negative angle
    else:
        angle_up = math.atan(-dy_up / dx_up)

    if dy_bottom > 0 and dx_bottom < 0:
        angle_bottom = math.atan(-dy_bottom / dx_bottom)
    elif dx_bottom >= 0:
        angle_bottom = math.pi / 2  # Avoid divide by 0 and also realistically the wing can't have negative angle
    else:
        angle_bottom = math.atan(dy_bottom / dx_bottom)

    if dy_up >= 0:
        angle = angle_bottom - angle_up
    elif dy_bottom <= 0:
        angle = angle_up - angle_bottom
    else:
        angle = angle_up + angle_bottom
    return angle

def findangler(upmost, bottommost, centre):  # Left wing and right wing are slightly different
    dy_up = (upmost[1] - centre[1])
    dx_up = (upmost[0] - centre[0])
    dy_bottom = (bottommost[1] - centre[1])
    dx_bottom = (bottommost[0] - centre[0])
    if dy_up < 0 and dx_up > 0:
        angle_up = math.atan(-dy_up / dx_up)
    elif dx_up <= 0:
        angle_up = math.pi / 2  # Avoid divide by 0 and also realistically the wing can't have negative angle
    else:
        angle_up = math.atan(dy_up / dx_up)

    if dy_bottom > 0 and dx_bottom > 0:
        angle_bottom = math.atan(dy_bottom / dx_bottom)
    elif dx_bottom <= 0:
        angle_bottom = math.pi / 2  # Avoid divide by 0 and also realistically the wing can't have negative angle
    else:
        angle_bottom = math.atan(-dy_bottom / dx_bottom)

    if dy_up >= 0:
        angle = angle_bottom - angle_up
    elif dy_bottom <= 0:
        angle = angle_up - angle_bottom
    else:
        angle = angle_up + angle_bottom
    return angle

class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    Don't worry too much about this, it's just using threading for speeding things up on raspberry pi
    """

    def __init__(self, src, w=1280, h=720, yshift=yshift, xshift=xshift):
        self.yshift = yshift
        self.xshift = xshift
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FPS, 100)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.mask = np.ones((int(h / 2), int(w / 2)), np.uint8) * 255
        self.mask[:, int(int(w / 2 / 2) - 50):int(int(w / 2 / 2) + 50)] = 0
        # self.mask[0:110, int(225):int(370)]=0
        # self.gray = cv2.cvtColor(self.frame[180:540, (320+self.hshift):(960+self.hshift)], cv2.COLOR_BGR2GRAY)
        for x in range(int(h / 2)):
            for y in range(int(w / 2)):
                if (x - (int(h / 2 / 2) - 5)) ** 2 + (y - (int(w / 2 / 2))) ** 2 > 200 ** 2:
                    self.mask[x, y] = 0
                elif (x - (int(h / 2 / 2))) ** 2 + (y - (int(w / 2 / 2))) ** 2 < 70 ** 2:
                    self.mask[x, y] = 0

        self.thr = cv2.bitwise_and(
            cv2.cvtColor(
                self.frame[(180 + self.yshift):(540 + self.yshift), (320 + self.xshift):(960 + self.xshift)],
                cv2.COLOR_BGR2GRAY), self.mask)

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

                self.thr = cv2.bitwise_and(cv2.cvtColor(
                    self.frame[(180 + self.yshift):(540 + self.yshift),
                    (320 + self.xshift):(960 + self.xshift)],
                    cv2.COLOR_BGR2GRAY), self.mask)

    def stop(self):
        self.stopped = True
        self.stream.release()

class CountsPerSec:
    """
    Class that tracks the number of occurrences ("counts") of an
    arbitrary event and returns the frequency in occurrences
    (counts) per second. The caller must increment the count.
    """

    def __init__(self):
        self._start_time = None
        self._num_occurrences = 0

    def start(self):
        self._start_time = datetime.now()
        return self

    def increment(self):
        self._num_occurrences += 1

    def countsPerSec(self):
        elapsed_time = (datetime.now() - self._start_time).total_seconds()
        return self._num_occurrences / elapsed_time if elapsed_time > 0 else 0

def putIterationsPerSec(frame, iterations_per_sec):
    """
    Add iterations per second text to lower-left corner of a frame. It also includes the main video processing /wingbeat analysing part
    """
    w = 1280  # int
    h = 720  # int
    global counter
    legcut = 100
    static_area_l = 2000  # preset value for static left wing from calibration, IMPORTANT as will be used as threshold for wingbeat envelope detection and calculation
    static_area_r = 1900  # preset value for static right wing from calibration
    hshift = -20
    bodycentre = (int(w / 2 / 2), int(h / 2 / 2))
    if static_area_r > static_area_l:
        static_area = static_area_l
    else:
        static_area = static_area_r
    leftangle = 0  # Variables storing angles of wingbeat
    rightangle = 0

    kernel = np.ones((2, 2), np.uint8)
    deltaangle = 0

    ret, thr = cv2.threshold(frame, 20, 255, cv2.THRESH_BINARY)  # threshold to see the wing

    hinge_l = tuple(
        (int(int(w / 2 / 2)) - 25, int(int(h / 2 / 2))))  # Manually define winghinges, done in calibration
    hinge_r = tuple((int(int(w / 2 / 2)) + 15, int(int(h / 2 / 2))))

    contours, hierarchy = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(
            contours) >= 1:  ##Note that there are many commented out part in this section, most of them should be reactivated when doing calibration
        # print(len(contours))   ## This version is only for autonomous running stage
        for i in range(len(contours)):

            cnt = contours[i]
            M = cv2.moments(cnt)
            area = cv2.contourArea(cnt)

            if area > static_area:  # Generalised threshold for a moving wing detection from experiments so that static wings won't be detected

                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])  # find the geographical centre of wingbeat(amplitude)

                if cx <= int(
                        w / 2 / 2) - 10:  # if the centre is in the left half then this area corresponds to left wingbeat(amplitude),same for right wing
                    leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])  # find the tips of the wings
                    upmost_l = tuple(cnt[cnt[:, :, 1].argmin()][0])
                    bottommost_l = tuple(cnt[cnt[:, :, 1].argmax()][0])

                    if upmost_l[1] < int(h / 2 / 2) and upmost_l[0] < (int(w / 2 / 2) - 5 - 0.6 * legcut):
                        leftangle = findanglel(upmost_l, bottommost_l, hinge_l)

                    else:
                        leftangle = findanglel(leftmost, bottommost_l, hinge_l)


                elif cx > int(w / 2 / 2) - 10 and area > static_area_r:
                    rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
                    upmost_r = tuple(cnt[cnt[:, :, 1].argmin()][0])
                    bottommost_r = tuple(cnt[cnt[:, :, 1].argmax()][0])

                    if upmost_r[1] < int(h / 2 / 2) and upmost_r[0] > (
                            int(w / 2 / 2) - 5 + 0.6 * legcut):  # Same logic for the right wing

                        rightangle = findangler(upmost_r, bottommost_r, hinge_r)
                    else:
                        rightangle = findangler(rightmost, bottommost_r, hinge_r)

        deltaangle = leftangle - rightangle  # angular wingbeat amplitude difference in radians
        if leftangle < math.radians(30) and abs(deltaangle) < 5:
            deltaangle = 0  # Not flapping, no significance
        elif rightangle < math.radians(30) and abs(deltaangle) < 5:
            deltaangle = 0

    print(deltaangle)


    saveangles[counter] = deltaangle
    counter = counter + 1
    if counter == 8:
        wingAverage(saveangles)
        counter = 0
    else:
        counter = counter + 1





def threadVideoGet(source=0):
    """
    Dedicated thread for grabbing video frames with VideoGet object.
    Main thread shows video frames.
    """

    legcut = 100

    video_getter = VideoGet(source).start()
    cps = CountsPerSec().start()
    st = time.time()
    period_goal = 0.01
    period = period_goal
    while True:  # Again the commented part is for displaying the video and calibration
        starttime = time.time()
        putIterationsPerSec(video_getter.thr, cps.countsPerSec())
        cps.increment()
        if cps._num_occurrences > 2500 or video_getter.stopped:
            print(time.time() - st)
            video_getter.stop()
            break

        ddt = time.time() - starttime

        if period - ddt < 0:
            period = ddt
        else:
            time.sleep(period - ddt)
            period = period - (time.time() - starttime - period_goal)

if __name__ == '__main__':
    threadVideoGet("LuciliaSample.avi")













#
"""""
if turningNum == 1:

    pwm0.start(40)
    pwm1.start(100)
    print('truning left')
    GPIO.output(33, GPIO.HIGH)
    GPIO.output(37, GPIO.HIGH)
    time.sleep(3)

elif turningNum == -1:
    print('truning right')
    pwm0.ChangeDutyCycle(100)
    pwm1.ChangeDutyCycle(40)
    time.sleep(3)

else:
    print('no turn')
    pwm0.ChangeDutyCycle(100)
    pwm1.ChangeDutyCycle(100)
    time.sleep(3)


GPIO.output(33, GPIO.LOW)
GPIO.output(37, GPIO.LOW)
time.sleep(3)

GPIO.output(37, GPIO.LOW)
pwm0.stop()
pwm1.stop()

GPIO.cleanup()
print('end')

"""
