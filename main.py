import cv2
from math import sqrt
from rtsp_streaming import Streamer


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def get_distance(p1: Point, p2: Point) -> float:
    return sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


class Face:
    def __init__(self, description=(0, 0, 0, 0)):
        """
        :param description: (x, y, h, w)
        """
        x, y, h, w = description
        self.h_half = h // 2
        self.w_half = w // 2
        self.center = Point(x + self.w_half, y + self.h_half)


def is_same_faces(face: Face, center_face: Face, half_h, half_w, alpha=1) -> bool:
    """
    center_face is in center of frame={height, width}
    faces are the same if the second in rectangle={alpha*height, alpha*width}
    """
    x_difference = abs(face.center.x - center_face.center.x)
    y_difference = abs(face.center.y - center_face.center.y)
    return x_difference <= alpha * half_w and y_difference <= alpha * half_h


def get_move_vector(face1: Face, face2: Face) -> (int, int):
    move_x = 4
    move_y = 2
    dx = face2.center.x - face1.center.x
    dy = face2.center.y - face1.center.y

    step_x = move_x if dx > 0 else -move_x
    step_y = move_y if dy > 0 else -move_y

    return step_x, step_y


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


def get_nearest_face(faces: list, prev_face: Face) -> Face:
    """
    :param faces: a list of faces whose description is like (x, y, h, w)
    :param prev_face: previous shot's face
    :return: center point of the nearest face to previous
    """
    # (x, y, h, w) -> Face object
    faces = list(map(lambda x: Face(x), faces))
    # sort by the nearest to prev_face
    faces.sort(key=lambda x: get_distance(x.center, prev_face.center))
    return faces[0]


class DBroadcaster:
    def __init__(self, id_cam, proportions=None, ratio_coeff=1.0):
        self.proportions = proportions

        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.cap = cv2.VideoCapture(id_cam)

        # window dimensions
        self.win_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.win_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        # evaluate frame dimensions
        if self.proportions is not None:
            self.frame_h = int(self.win_h * ratio_coeff)
            ratio = self.frame_h / proportions[0]
            self.frame_h_half = self.frame_h // 2
            self.frame_w_half = int(proportions[-1] * ratio) // 2
            self.streamer = Streamer(2 * self.frame_h_half, 2 * self.frame_w_half,
                                     int(self.cap.get(cv2.CAP_PROP_FPS)))
        else:
            self.streamer = Streamer(self.win_h, self.win_w,
                                     int(self.cap.get(cv2.CAP_PROP_FPS)))

        self.prev_face = Face()  # center of face
        # extreme frame points
        self.prev_left_top = Point(0, 0)
        self.prev_right_bottom = Point(self.win_w, self.win_h)

        self.move_cam = False
        self.num_iter_step = 0
        self.prev_dx = 0
        self.prev_dy = 0

    def broadcast_crop_video(self):
        """
        detects, cuts and shows
        """
        while True:
            ret, img = self.cap.read()
            img = img[:, ::-1]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            left_top, right_bottom = self.get_new_frame_points(faces, self.prev_left_top, self.prev_right_bottom)
            img = img[left_top.y:right_bottom.y, left_top.x:right_bottom.x]
            # cv2.imshow('video cropped', img)
            self.streamer.stream(img)
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
            face = get_nearest_face(faces, self.prev_face)
            alpha = 0.25 if self.move_cam else 0.6

            if is_same_faces(face, self.prev_face, self.frame_h_half, self.frame_w_half, alpha):
                self.move_cam = False
                left_top = prev_left_top
                right_bottom = prev_right_bottom
            else:
                if self.prev_face.center.x != 0 and self.prev_face.center.y != 0:
                    dx, dy = self.move_center(face)
                    face.center.x = self.prev_face.center.x + dx
                    face.center.y = self.prev_face.center.y + dy

                top, bottom = get_clipped_segment(face.center.y, self.frame_h_half, self.win_h)
                left, right = get_clipped_segment(face.center.x, self.frame_w_half, self.win_w)

                left_top = Point(left, top)
                right_bottom = Point(right, bottom)

                self.prev_face = face
                self.prev_left_top = left_top
                self.prev_right_bottom = right_bottom

        else:
            left_top = prev_left_top
            right_bottom = prev_right_bottom

        return left_top, right_bottom

    def move_center(self, face):
        if self.move_cam and self.num_iter_step < 25:
            self.num_iter_step += 1
            dx, dy = self.prev_dx, self.prev_dy
        else:
            self.move_cam = True
            self.num_iter_step = 0
            dx, dy = get_move_vector(self.prev_face, face)
            self.prev_dx, self.prev_dy = dx, dy

        return dx, dy


streamer = DBroadcaster(0, (3, 4), 0.7)
streamer.broadcast_crop_video()
