import cv2
from math import sqrt


def is_same_centers(x1, y1, x2, y2, threshold=50):
    return abs(x1 - x2) <= threshold and abs(y1 - y2) <= threshold


def get_nearest_face(faces: list, prev_x, prev_y):
    faces = list(map(lambda x: (x[0] + x[-1] // 2, x[1] + x[2] // 2), faces))
    faces.sort(key=lambda x: sqrt((x[0] - prev_x) ** 2 + (x[-1] - prev_y) ** 2))
    return faces[0]


proportions = list(map(int, input("proportions: ").split()))
# proportions = (3, 4)
assert len(proportions) == 2

print("please wait...")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

ret, img = cap.read()
win_h, win_w = img.shape[:2]

if proportions is not None:
    alpha = 1
    frame_h = int(win_h * alpha)
    ratio = frame_h / proportions[0]
    frame_h_half, frame_w_half = frame_h // 2, int(proportions[-1] * ratio) // 2

prev_cx, prev_cy = 0, 0
prev_top, prev_bott, prev_l, prev_r = 0, win_h, 0, win_w

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if proportions is not None and len(faces) != 0:
        c_x, c_y = get_nearest_face(faces, prev_cx, prev_cy)

        if is_same_centers(c_x, c_y, prev_cx, prev_cy):
            top, bottom = prev_top, prev_bott
            left, right = prev_l, prev_r
        else:
            top = max(0, c_y - frame_h_half)
            bottom = min(win_h, c_y + frame_h_half)

            if top == 0:
                bottom = frame_h
            elif bottom == win_h:
                top = win_h - frame_h

            left = max(0, c_x - frame_w_half)
            right = min(win_w, c_x + frame_w_half)

            if left == 0:
                right = 2 * frame_w_half
            elif right == win_w:
                left = win_w - 2 * frame_w_half

            prev_cx, prev_cy = c_x, c_y
            prev_top, prev_bott = top, bottom
            prev_l, prev_r = left, right

    else:
        top, bottom = prev_top, prev_bott
        left, right = prev_l, prev_r

    img = img[top:bottom, left:right]

    cv2.imshow('img', img)

    # Wait for Esc key to stop
    k = cv2.waitKey(1) & 0xff
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(":)")
