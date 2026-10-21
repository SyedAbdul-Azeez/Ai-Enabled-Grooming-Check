import os
import sys
import urllib.request
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

FOREHEAD_MIN = 0.60
EAR_MIN = 0.30
CR_TOL = 12
CB_TOL = 12
Y_TOL = 45
EAR_WIDTH_FACTOR = 0.22

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "face_landmarker.task")

FOREHEAD_TOP = 10
BROW_MID = 9
FOREHEAD_LEFT = 67
FOREHEAD_RIGHT = 297
FACE_EDGE_RIGHT = 234
FACE_EDGE_LEFT = 454
CHEEK_POINTS = [50, 280]


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading face model (one time only)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")


def make_landmarker():
    ensure_model()
    base = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def pt(landmarks, idx, w, h):
    lm = landmarks[idx]
    return int(lm.x * w), int(lm.y * h)


def clamp_box(x1, y1, x2, y2, w, h):
    x1, x2 = sorted((max(0, x1), min(w - 1, x2)))
    y1, y2 = sorted((max(0, y1), min(h - 1, y2)))
    return x1, y1, x2, y2


def get_regions(landmarks, w, h):
    top = pt(landmarks, FOREHEAD_TOP, w, h)
    brow = pt(landmarks, BROW_MID, w, h)
    fl = pt(landmarks, FOREHEAD_LEFT, w, h)
    fr = pt(landmarks, FOREHEAD_RIGHT, w, h)
    er = pt(landmarks, FACE_EDGE_RIGHT, w, h)
    el = pt(landmarks, FACE_EDGE_LEFT, w, h)

    face_w = abs(el[0] - er[0])
    ear_w = int(face_w * EAR_WIDTH_FACTOR)

    face_h = abs(pt(landmarks, 152, w, h)[1] - top[1])
    forehead_top = top[1] - int(face_h * 0.06)
    forehead = clamp_box(fl[0], forehead_top, fr[0], brow[1] - 5, w, h)
    ear_top = brow[1]
    ear_bottom = brow[1] + int(face_w * 0.35)
    right_ear = clamp_box(er[0] - ear_w, ear_top, er[0], ear_bottom, w, h)
    left_ear = clamp_box(el[0], ear_top, el[0] + ear_w, ear_bottom, w, h)

    return {"Forehead": forehead, "Right ear": right_ear, "Left ear": left_ear}


def learn_skin_colour(frame, landmarks, w, h):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    samples = []
    for idx in CHEEK_POINTS:
        x, y = pt(landmarks, idx, w, h)
        patch = ycrcb[max(0, y - 6):y + 6, max(0, x - 6):x + 6]
        if patch.size:
            samples.append(patch.reshape(-1, 3))
    if not samples:
        return None
    return np.vstack(samples).mean(axis=0)


def skin_ratio(frame, box, skin_mean):
    x1, y1, x2, y2 = box
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    y_ref, cr_ref, cb_ref = skin_mean
    y = ycrcb[:, :, 0].astype(float)
    cr = ycrcb[:, :, 1].astype(float)
    cb = ycrcb[:, :, 2].astype(float)
    mask = ((np.abs(cr - cr_ref) < CR_TOL) &
            (np.abs(cb - cb_ref) < CB_TOL) &
            (np.abs(y - y_ref) < Y_TOL))
    return float(mask.mean())


def analyse(frame, landmarker):
    h, w = frame.shape[:2]
    rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        cv2.putText(frame, "No face detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame, None

    landmarks = result.face_landmarks[0]
    skin_mean = learn_skin_colour(frame, landmarks, w, h)
    if skin_mean is None:
        return frame, None

    regions = get_regions(landmarks, w, h)
    report = {}
    for name, box in regions.items():
        ratio = skin_ratio(frame, box, skin_mean)
        min_ratio = FOREHEAD_MIN if name == "Forehead" else EAR_MIN
        ok = ratio >= min_ratio
        report[name] = (ok, ratio)
        colour = (0, 200, 0) if ok else (0, 0, 255)
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(frame, f"{name} {ratio:.0%}", (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

    passed = all(ok for ok, _ in report.values())
    text = "GROOMING: PASS" if passed else "GROOMING: FAIL - hair covering"
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 200, 0) if passed else (0, 0, 255), 2)
    return frame, report


WINDOW_TITLE = "Python Hair Grooming Check AI"


def print_menu():
    print("=" * 40)
    print("   PYTHON HAIR GROOMING CHECK AI")
    print("=" * 40)
    print("   s  =  save screenshot")
    print("   q  =  quit")
    print("=" * 40)
    print("(click the camera window, then press a key)\n")


def save_screenshot(img):
    folder = os.path.dirname(os.path.abspath(__file__))
    name = datetime.now().strftime("grooming_%Y%m%d_%H%M%S.png")
    path = os.path.join(folder, name)
    cv2.imwrite(path, img)
    print("Screenshot saved:", path)


def main():
    landmarker = make_landmarker()
    print_menu()

    if len(sys.argv) > 1:
        frame = cv2.imread(sys.argv[1])
        if frame is None:
            print("Could not read image.")
            return
        out, report = analyse(frame, landmarker)
        print(report)
        cv2.imshow(WINDOW_TITLE, out)
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord("s"):
                save_screenshot(out)
            elif key == ord("q"):
                break
    else:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            out, _ = analyse(frame, landmarker)
            cv2.imshow(WINDOW_TITLE, out)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                save_screenshot(out)
            elif key == ord("q"):
                print("Goodbye!")
                break
        cap.release()

    landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
