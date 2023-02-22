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
    '''Take N sample frames from vid_loc and store the images in frame_loc.'''

    if N == 1:
        vidcap = cv.VideoCapture(vid_loc)
        success, image = vidcap.read()
        cv.imwrite(frame_loc + 'frame.jpg', image)
    else:
        frame_nums = np.random.randint(0, 3000, N)
        vidcap = cv.VideoCapture(vid_loc)
        success, image = vidcap.read()
        i = 0

        while success:        
            if i in frame_nums:
                cv.imwrite(frame_loc + 'frame%d.jpg' % i, image)
            
            success, image = vidcap.read()
            i += 1


def generate_frames_for_intrinsics(num_frames):
    sample_frames_from_video('data/cam1/intrinsics.avi', 'data/cam1/frames_in/', num_frames)
    sample_frames_from_video('data/cam2/intrinsics.avi', 'data/cam2/frames_in/', num_frames)
    sample_frames_from_video('data/cam3/intrinsics.avi', 'data/cam3/frames_in/', num_frames)
    sample_frames_from_video('data/cam4/intrinsics.avi', 'data/cam4/frames_in/', num_frames)


def generate_frames_for_extrinsics():
    sample_frames_from_video('data/cam1/checkerboard.avi', 'data/cam1/frames_ex/', 1)
    sample_frames_from_video('data/cam2/checkerboard.avi', 'data/cam2/frames_ex/', 1)
    sample_frames_from_video('data/cam3/checkerboard.avi', 'data/cam3/frames_ex/', 1)
    sample_frames_from_video('data/cam4/checkerboard.avi', 'data/cam4/frames_ex/', 1)


def draw_axes(img, corners, imgpts):
    '''Project three dimensional axes onto the img and draw a line for each axis.'''

    corner = tuple(corners[0].ravel().astype(int))
    # bright yellow (69,233,255)
    img = cv.line(img, corner, tuple(imgpts[0].ravel().astype(int)), (0,0,255), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel().astype(int)), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel().astype(int)), (255,0,0), 5)
    return img