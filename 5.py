import cv2
import dlib
import numpy as np

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)


def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) != 1:
        return None

    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def annotate_landmarks(im, landmarks):
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im


def get_lip_distance(landmarks):
    if landmarks is None:
        return 0

    top_lip_pts = landmarks[50:53] + landmarks[61:64]
    bottom_lip_pts = landmarks[65:68] + landmarks[56:59]

    top_lip_mean = np.mean(top_lip_pts, axis=0)
    bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)

    lip_distance = abs(top_lip_mean[0, 1] - bottom_lip_mean[0, 1])
    return lip_distance


cap = cv2.VideoCapture(0)
yawns = 0
yawn_status = False

while True:
    ret, frame = cap.read()
    landmarks = get_landmarks(frame)

    if landmarks is not None:
        image_with_landmarks = annotate_landmarks(frame, landmarks)
        lip_distance = get_lip_distance(landmarks)

        if lip_distance > 40:
            yawn_status = True
            cv2.putText(frame, "Subject is Yawning", (50, 450),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            yawns += 1

        cv2.putText(frame, f"Yawn Count: {yawns}", (50, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)

        cv2.imshow('Live Landmarks', image_with_landmarks)
    else:
        cv2.putText(frame, "No face detected", (50, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Yawn Detection', frame)

    if cv2.waitKey(1) == 13:  # 13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows()
