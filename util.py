import numpy as np
import cv2 as cv
from bs4 import BeautifulSoup


def get_checkerboard_config():
    '''Read the checkerboard's dimensions and tile size from data/checkerboard.xml.
    '''
    with open('data/checkerboard.xml', 'r') as f:
        data = f.read()

    bs_data = BeautifulSoup(data, 'xml')
    width = int(bs_data.find('CheckerBoardWidth').text)
    height = int(bs_data.find('CheckerBoardHeight').text)
    square_size = int(bs_data.find('CheckerBoardSquareSize').text)

    return width, height, square_size


def sample_frames_from_video(vid_loc, frame_loc, N):
    '''Take N sample frames from vid_loc and store the images in frame_loc.
    '''
    frame_nums = np.random.randint(0, 3000, N)
    vidcap = cv.VideoCapture(vid_loc)
    success, image = vidcap.read()
    i = 0

    while success:        
        if i in frame_nums:
            cv.imwrite(frame_loc + 'frame%d.jpg' % i, image)
        
        success, image = vidcap.read()
        i += 1


def generate_sample_frames(num_frames):
    sample_frames_from_video('data/cam1/intrinsics.avi', 'data/cam1/frames/', num_frames)
    sample_frames_from_video('data/cam2/intrinsics.avi', 'data/cam2/frames/', num_frames)
    sample_frames_from_video('data/cam3/intrinsics.avi', 'data/cam3/frames/', num_frames)
    sample_frames_from_video('data/cam4/intrinsics.avi', 'data/cam4/frames/', num_frames)

