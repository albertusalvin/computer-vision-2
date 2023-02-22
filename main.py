import numpy as np
import cv2 as cv
import glob

import util


def calibrate_camera(objp, images, img_scale, board_dim, criteria):
    # Arrays to store object points and image points from all the images.    
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    for fname in images:
        img = cv.imread(fname)
        scale_dim = (int(img.shape[1] * img_scale / 100), int(img.shape[0] * img_scale / 100))
        img = cv.resize(img, scale_dim, interpolation = cv.INTER_AREA)  # Scales the image down
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        ret, corners = cv.findChessboardCorners(gray, board_dim, None)
        # TODO: Do we need to use manual annotation in this phase? 
        # The assignment instruction indicates that manual annotation comes in step 2, when calculating camera extrinsics.
        
        # If found, add object points, image points (after refining them)
        if ret:
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv.drawChessboardCorners(img, board_dim, corners2, True)
            cv.imshow('img', img)
            cv.waitKey(1000)
    
        print(f'Calibrating with {fname}. Autofound corners: {ret}')
    
    return cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)



if __name__ == '__main__':
    np.random.seed(10)
    num_frames = 10
    img_scale = 80
    board_width, board_height, square_size = util.get_checkerboard_config()

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((board_width * board_height, 3), np.float32)
    objp[:,:2] = np.mgrid[0:board_width, 0:board_height].T.reshape(-1,2)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001) # Termination criteria

    # Take sample frames from videos
    util.generate_frames_for_intrinsics(num_frames)
    util.generate_frames_for_extrinsics()

    calibrate_camera(objp, glob.glob('data/cam1/frames/*.jpg'), img_scale, (board_width, board_height), criteria)
    calibrate_camera(objp, glob.glob('data/cam2/frames/*.jpg'), img_scale, (board_width, board_height), criteria)
    calibrate_camera(objp, glob.glob('data/cam3/frames/*.jpg'), img_scale, (board_width, board_height), criteria)
    calibrate_camera(objp, glob.glob('data/cam4/frames/*.jpg'), img_scale, (board_width, board_height), criteria)
