import numpy as np
import cv2 as cv

class CoordinateClicker():
    ''' Define an object to keep track of the locations that has been clicked.'''
    
    def __init__(self, img):
        self.coordinates = []
        self.img = img

    
    def click_event(self, event, x, y, flags, params):
        '''Registers left mouse button clicks and record the coordinate points. 
        Displays the image with circles at the coordinate locations.'''

        if event == cv.EVENT_LBUTTONDOWN:
            print(f'Click detected at {x},{y}')
            self.coordinates.append(np.array([x,y], dtype=np.float32))
            cv.circle(self.img, (x,y), 10, (255, 255, 0), 3)

            