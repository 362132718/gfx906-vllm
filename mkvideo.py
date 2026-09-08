# mkvideo.py -- synthesise a clip: a red square sliding left->right on a blue
# field. Used by the video probe; vLLM's serving API wants a real video file.
import sys
import cv2
import numpy as np

path = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
fps = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
W, H = 448, 256

def make(codec, ext):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, (W, H))
    if not vw.isOpened():
        return False
    for i in range(n):
        f = np.zeros((H, W, 3), np.uint8)
        f[:, :] = (120, 40, 20)                      # BGR blue field
        x = int(40 + i * (W - 140) / max(n - 1, 1))
        f[90:190, x:x + 100] = (45, 45, 225)        # red square
        vw.write(f)
    vw.release()
    cap = cv2.VideoCapture(path)
    got = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    ok = got >= 2
    print("codec=%s wrote=%s frames_reported=%s readable=%s" % (codec, path, got, ok))
    return ok

for codec, ext in (("mp4v", ".mp4"), ("avc1", ".mp4"), ("MJPG", ".avi")):
    p = path.rsplit(".", 1)[0] + ext
    if make(codec, ext):
        sys.exit(0)
print("NO_CODEC_WORKED")
sys.exit(1)
