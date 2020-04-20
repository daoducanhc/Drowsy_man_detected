from imutils.video import VideoStream
from imutils import face_utils
from datetime import datetime
import numpy as np
import dlib
import imutils
import time
import cv2

def distance_3d(pointA, pointB):
    return np.linalg.norm(pointA - pointB)

def eye_aspect_ratio(eye):
    a = distance_3d(eye[1], eye[5])                #1       #2
    b = distance_3d(eye[2], eye[4])        #0                      #3
                                                    #5       #4
    c = distance_3d(eye[0], eye[3])

    ear = (a + b) / (2 * c)
    return ear

def mouth_aspect_ratio(mouth):
    a = distance_3d(mouth[1], mouth[7])
    b = distance_3d(mouth[2], mouth[6])
    c = distance_3d(mouth[3], mouth[5])

    d = distance_3d(mouth[0], mouth[4])

    mar = (a + b + c) / (2 * d)
    return mar

def detect_drowsiness():

    blinkFrameCounter = 0
    yawnFrameCounter = 0
    startTime = None

    detector = cv2.CascadeClassifier("resources/haarcascade_frontalface_default.xml")
    predictor = dlib.shape_predictor("resources/shape_predictor_68_face_landmarks.dat")

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]

    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    while True:
        frame = vs.read().copy()
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        if len(rects) > 0:
            rect = sorted(rects, key=lambda r: r[2]*r[3], reverse=True)[0]

            (x, y, w, h) = rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            rect = dlib.rectangle( int(x), int(y), int(x+w), int(y+h) )
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart : lEnd]
            rightEye = shape[rStart : rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2

    #? option to draw the eye hull
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (255, 0, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (255, 0, 0), 1)
            cv2.putText(frame, f"Eye aspect ratio: {ear:.3} {blinkFrameCounter}", (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # EYES
            if ear < 0.3: # eyes closed
                blinkFrameCounter += 1

                if blinkFrameCounter > 6: # eyes closed in 6 frames
                    cv2.putText(frame, f"DROWSINESS ALERT!!! --- EYES", (50, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                blinkFrameCounter = 0

            # YAWNING
            mouth = shape[mStart : mEnd]
            mar = mouth_aspect_ratio(mouth)

    #? option to draw mouth hull
            mouthHull = cv2.convexHull(mouth)
            cv2.drawContours(frame, [mouthHull], -1, (255, 0, 0), 1)
            cv2.putText(frame, f"Mouth aspect ratio: {mar:.3} {yawnFrameCounter}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if mar > 0.65:
                yawnFrameCounter += 1
                if startTime == None:
                    startTime = datetime.now()

                # if u yawn 2-3 times in 5 minutes
                if yawnFrameCounter > 150 and (datetime.now() - startTime).seconds < 300:
                    cv2.putText(frame, f"DROWSINESS ALERT!!! --- YAWN", (50, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            elif startTime != None:
                if (datetime.now() - startTime).seconds >= 300:
                    yawnFrameCounter = 0
                    startTime = None


        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break


    cv2.destroyAllWindows()
    vs.stop()
