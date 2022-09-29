import cv2
import subprocess as sp


class Streamer:
    def __init__(self, size_h, size_w, fps):
        rtsp_server = 'rtsp://localhost:8554/dptz_stream'
        sizeStr = f'{size_h}x{size_w}'

        command = ['ffmpeg',
                   '-re',
                   '-s', sizeStr,
                   '-r', str(fps),  # rtsp fps (from input server)
                   '-i', '-',

                   # You can change ffmpeg parameter after this item.
                   '-pix_fmt', 'yuv420p',
                   '-r', '60',  # '30',  # output fps
                   '-g', '250',  # '50'
                   '-keyint_min', '25', # !!!
                   '-c:v', 'libx264',
                   '-b:v', '2M',
                   '-bufsize', '64M',
                   '-maxrate', "4M",
                   '-preset', 'veryfast',
                   '-rtsp_transport', 'tcp',
                   '-segment_times', '5',
                   '-f', 'rtsp',
                   rtsp_server]
        self.process = sp.Popen(command, stdin=sp.PIPE)

    def stream(self, frame):
        ret, frame_ = cv2.imencode('.png', frame)
        self.process.stdin.write(frame_.tobytes())
