import time
import picamera
import os

WIDTH=40
HEIGHT=32

MAXFRAMES=200
FRAMERATE=24

images_dir=os.environ.get("HOME") + "/april-pi-cnn-line-follower/images"

def filenames():
    frame = 0
    while True:
        yield '%s/image%04d.jpg' % (images_dir,frame)
        frame = (frame+1) % MAXFRAMES

with picamera.PiCamera() as camera:
    camera.resolution = (WIDTH, HEIGHT)
    camera.framerate = FRAMERATE
    camera.start_preview()
    # Give the camera some warm-up time
    time.sleep(2)
    camera.capture_sequence(filenames(), use_video_port=True)
