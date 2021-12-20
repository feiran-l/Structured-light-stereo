import os
import time
import numpy as np
import cv2
from matplotlib import pyplot as plt


if __name__ == '__main__':
    # setups
    proj_w, proj_h = 1920, 1080
    major_screen_resolution = int(1920 / 1.5)  # resolution of the laptop's screen (e.g., 1080P laptop with 150% scaling)

    # generate patterns
    graycode = cv2.structured_light_GrayCodePattern.create(width=proj_w, height=proj_h)
    _, patterns = graycode.generate()
    black, white = graycode.getImagesForShadowMasks(np.zeros_like(patterns[0]), np.zeros_like(patterns[0]))
    patterns = patterns + [white, black]  # horizontal, vertical, black-white
    print('{} pictures in total for the projector resolution ({}, {})'.format(len(patterns), proj_w, proj_h))

    for i, x in enumerate(patterns):
        print('showing the {}-th image'.format(i))
        capname = '{}'.format(i)
        cv2.namedWindow(capname, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(capname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.moveWindow(capname, major_screen_resolution, 0)
        cv2.imshow(capname, x)
        k = cv2.waitKey(0)
        if k == ord("q"):
            cv2.destroyWindow(capname)
            continue
        else:
            time.sleep(2)
            cv2.destroyWindow(capname)
            continue