import cv2
from math import sqrt


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def is_same_centers(c1: Point, c2: Point, threshold=50) -> bool:
    return abs(c1.x - c2.x) <= threshold and abs(c1.y - c2.y) <= threshold


def get_distance(p1: Point, p2: Point) -> float:
    return sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def get_clipped_segment(center, half_len, up_bound) -> (int, int):
    """
    :return: segment between 0 and up_bound

    """
    start = max(0, center - half_len)
    end = min(up_bound, center + half_len)
    if start == 0:
        end = 2 * half_len
    elif end == up_bound:
        start = up_bound - 2 * half_len

    return start, end


def get_nearest_face(faces: list, prev_center: Point) -> Point:
    """
    :param faces: a list of faces whose description is like (x, y, h, w)
    :param prev_center: center point of previous shot's face
    :return: center point of the nearest face to previous
    """
    # (x, y, h, w) -> (c_x, c_y)
    centers = list(map(lambda x: Point(x[0] + x[-1] // 2, x[1] + x[2] // 2), faces))
    # sort by the nearest
    centers.sort(key=lambda x: get_distance(x, prev_center))
    return centers[0]


class DBroadcaster:
    def __init__(self, id_cam, proportions=None, ratio_coeff=1.0):
        self.proportions = proportions

        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.cap = cv2.VideoCapture(id_cam)

        ret, img = self.cap.read()
        # window dimensions
        self.win_h, self.win_w = img.shape[:2]

        # evaluate frame dimensions
        if self.proportions is not None:
            self.frame_h = int(self.win_h * ratio_coeff)
            ratio = self.frame_h / proportions[0]
            self.frame_h_half = self.frame_h // 2
            self.frame_w_half = int(proportions[-1] * ratio) // 2

        self.prev_c = Point(0, 0)  # center of face
        # extreme frame points
        self.prev_left_top = Point(0, 0)
        self.prev_right_bottom = Point(self.win_w, self.win_h)

    def broadcast_crop_video(self):
        """
        detects, cuts and shows
        """
        while True:
            ret, img = self.cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            left_top, right_bottom = self.get_new_frame_points(faces, self.prev_left_top, self.prev_right_bottom)
            img = img[left_top.y:right_bottom.y, left_top.x:right_bottom.x]
            cv2.imshow('video cropped', img)
            # Wait for 'q' key to stop
            if cv2.waitKey(1) & 0xff == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def get_new_frame_points(self, faces, prev_left_top, prev_right_bottom) -> (Point, Point):
        """
        evaluate new frame coordinates
        """
        if self.proportions is not None and len(faces) != 0:
            center = get_nearest_face(faces, self.prev_c)
            if is_same_centers(center, self.prev_c):
                left_top = prev_left_top
                right_bottom = prev_right_bottom
            else:
                top, bottom = get_clipped_segment(center.y, self.frame_h_half, self.win_h)
                left, right = get_clipped_segment(center.x, self.frame_w_half, self.win_w)

                left_top = Point(left, top)
                right_bottom = Point(right, bottom)

                self.prev_c = center
                self.prev_left_top = left_top
                self.prev_right_bottom = right_bottom

        else:
            left_top = prev_left_top
            right_bottom = prev_right_bottom

        return left_top, right_bottom


user_input = input("proportions (if they don't needs - use '-'): ")
if user_input == '-':
    props = None
else:
    props = list(map(int, user_input.split()))
    assert len(props) == 2, "proportions have only 2 arguments :0\n"

print("please wait...")
streamer = DBroadcaster(0, props, 0.7)
streamer.broadcast_crop_video()
print(":)")
