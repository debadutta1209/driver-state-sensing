from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import dlib
import cv2
import time

def sound_alarm(path):
    # Placeholder function to play an alarm sound
    # You can replace this with actual sound playing logic
    cv2.waitKey(1000)

def eye_aspect_ratio(eye):
    # Compute the eye aspect ratio
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Argument parsing
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", default="shape_predictor_68_face_landmarks.dat",
                help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", type=str, default="",
                help="path alarm .WAV file")
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
args = vars(ap.parse_args())

# Constants
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 48
EAR_NOT_VISIBLE_FRAMES = 30  # Number of consecutive frames with no eyes detected to trigger drowsiness alert

# Initialize variables
COUNTER = 0
EAR_NOT_VISIBLE_COUNTER = 0
ALARM_ON = False

# Initialize dlib's face detector and facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# Define eye indexes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Start the video stream
print("[INFO] starting video stream thread...")
vs = VideoStream(src=args["webcam"]).start()
vs.stream.set(cv2.CAP_PROP_FPS, 30)
time.sleep(2.0)  # Give time for the camera to warm up

# Loop over frames from the video stream
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    rects = detector(gray, 0)

    # Reset EAR_NOT_VISIBLE_COUNTER if eyes are detected
    if len(rects) > 0:
        EAR_NOT_VISIBLE_COUNTER = 0

    # Loop over the face detections
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        if ear < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                    if args["alarm"] != "":
                        t = Thread(target=sound_alarm, args=(args["alarm"],))
                        t.daemon = True
                        t.start()

                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            COUNTER = 0
            ALARM_ON = False

    # Increment EAR_NOT_VISIBLE_COUNTER if no eyes are detected
    if len(rects) == 0:
        EAR_NOT_VISIBLE_COUNTER += 1

    # Check if drowsiness alert should be triggered due to eyes not being visible
    if EAR_NOT_VISIBLE_COUNTER >= EAR_NOT_VISIBLE_FRAMES:
        if not ALARM_ON:
            ALARM_ON = True
            if args["alarm"] != "":
                t = Thread(target=sound_alarm, args=(args["alarm"],))
                t.daemon = True
                t.start()

        cv2.putText(frame, "DROWSINESS ALERT! Eyes not visible", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # If the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()
vs.stop()
