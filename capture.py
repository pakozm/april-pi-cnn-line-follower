import time
import picamera
import os

WIDTH=24
HEIGHT=24

FRAMERATE=24
MAXFRAMES=FRAMERATE

images_dir=os.environ.get("HOME") + "/april-pi-cnn-line-follower/images"

def filenames():
    frame = 0
    while True:
        yield '%s/image%04d.png' % (images_dir,frame)
        frame = (frame+1) % MAXFRAMES

with picamera.PiCamera() as camera:
    camera.resolution = (640, 480)
    camera.framerate = FRAMERATE
    camera.start_preview()
    # Give the camera some warm-up time
    time.sleep(2)
    camera.capture_sequence(filenames(), format="png", use_video_port=True, resize=(WIDTH,HEIGHT))
