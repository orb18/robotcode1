#import RPi.GPIO as GPIO
import gpio as GPIO
import time
import cv2
from datetime import datetime
from threading import Thread
import math
import numpy as np

saveaangles = [0, 0, 0, 0]

#saveangles = []


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

    for j in range(0, 4):
        #finding out how many even and odd deltaangles in the array to find out which way the fly is moving.
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
    #this is the code which will turn the robot.
    #isnt recognised on pyCharm so will need to do by trial and error with the robot itslef.
    #if the robot isnt't flying shall we get the robot to stop moving completely???


    if turningNum == 1:


        print('turning left')
        pwm0.ChangeDutyCycle(60)
        pwm1.ChangeDutyCycle(40)
        time.sleep(0.1)

    elif turningNum == -1:
        print('turning right')
        pwm0.ChangeDutyCycle(40)
        pwm1.ChangeDutyCycle(60)
        time.sleep(0.1)

    else:
        print('no turn')
        pwm0.ChangeDutyCycle(50)
        pwm1.ChangeDutyCycle(50)
        time.sleep(0.1)





yshift = -50  # Use these parameters to adjust the position of the fly in the image. Negative means moving left, positive moves it to the right
xshift = -15


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

def putIterationsPerSec(frame, iterations_per_sec,counter):
    """
    Add iterations per second text to lower-left corner of a frame. It also includes the main video processing /wingbeat analysing part
    """


    #this function will work out the delta angle by utilising other fucntions (findanglel and findangler)
    w = 1280  # int
    h = 720  # int
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

    kernel = np.ones((2, 2), np.uint8) # not sure what kernel does.
    deltaangle = 0 #initialising the deltaangle for each loop around.

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
                        leftangle = findanglel(upmost_l, bottommost_l, hinge_l) #jumps to the fuction and returns left angle

                    else:
                        leftangle = findanglel(leftmost, bottommost_l, hinge_l)


                elif cx > int(w / 2 / 2) - 10 and area > static_area_r:
                    rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
                    upmost_r = tuple(cnt[cnt[:, :, 1].argmin()][0])
                    bottommost_r = tuple(cnt[cnt[:, :, 1].argmax()][0])

                    if upmost_r[1] < int(h / 2 / 2) and upmost_r[0] > (
                            int(w / 2 / 2) - 5 + 0.6 * legcut):  # Same logic for the right wing

                        rightangle = findangler(upmost_r, bottommost_r, hinge_r) #jumps to the function and returns right angle
                    else:
                        rightangle = findangler(rightmost, bottommost_r, hinge_r)

        deltaangle = leftangle - rightangle  # angular wingbeat amplitude difference in radians


        #We may need to change this due to the sensetivity of each wing cycle
        if leftangle < math.radians(30) and abs(deltaangle) < 5:
            deltaangle = 0  # Not flapping, no significance
        elif rightangle < math.radians(30) and abs(deltaangle) < 5:
            deltaangle = 0

    print(deltaangle) # to make sure its outputing the correct angle
    #print(counter)



    saveaangles[counter] = deltaangle # saving the deltaangle to the correct array spot
    if counter == 3:
        wingAverage(saveaangles) #jumping to the processing fucntion.
        time.sleep(1) #we could put the sleep in a higher up fucntion to the robot and is only 1sec to exagerate.





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
    counter = 0


    #we need to start the robot moving.

    GPIO.setmode(GPIO.BOARD)

    GPIO.setup(32, GPIO.OUT)
    GPIO.setup(33, GPIO.OUT)
    GPIO.setup(35, GPIO.OUT)
    GPIO.setup(36, GPIO.OUT)
    GPIO.setup(37, GPIO.OUT)
    GPIO.setup(38, GPIO.OUT)

    pwm0 = GPIO.PWM(32, 20000)
    pwm1 = GPIO.PWM(33, 20000)

    pwm0.start(50)
    pwm1.start(50)


    GPIO.output(36, GPIO.HIGH)
    GPIO.output(35, GPIO.HIGH)
    GPIO.output(38, GPIO.LOW)
    GPIO.output(37, GPIO.LOW)

    #robot is now moving in a stright line.
    while True:  # Again the commented part is for displaying the video and calibration
        #this is the main while loop which will run until the video is finished.

        #the counter will go from 0 to 3 to save and average 4 seperate wing beat angles to an array later on
        counter = counter +1
        if counter == 4:
            #will need to reset the counter every 4 itterations.
            counter = 0
        starttime = time.time()
        #now go to the next function:
        putIterationsPerSec(video_getter.thr, cps.countsPerSec(),counter)
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
    #program starts here.
    #finds the attached .avi video and then goes to the threadVideoGet() function
    threadVideoGet("LuciliaSample.avi")

