import numpy as np
import cv2 as cv
import glob

import util


def calibrate_camera(images, img_scale, board_dim, criteria):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((board_dim[1] * board_dim[0], 3), np.float32)
    objp[:,:2] = np.mgrid[0:board_dim[0], 0:board_dim[1]].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.    
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    for fname in images:
        img = cv.imread(fname)
        scale_dim = (int(img.shape[1] * img_scale / 100), int(img.shape[0] * img_scale / 100))
        img = cv.resize(img, scale_dim, interpolation = cv.INTER_AREA)  # Scales the image down
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        ret, corners = cv.findChessboardCorners(gray, board_dim, None)
        print('isCornersFound?', ret)
        # TODO: Do we need to use manual annotation in this phase? The assignment instruction indicates that manual annotation comes in step 2, when calculating camera extrinsics.
        
        # If found, add object points, image points (after refining them)
        if ret:
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv.drawChessboardCorners(img, board_dim, corners2, True)
            cv.imshow('img', img)
            cv.waitKey(1000)
    
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print('MATRIX')
    print(mtx)
    return ret, mtx, dist, rvecs, tvecs
    


if __name__ == '__main__':
    np.random.seed(10)
    num_frames = 10
    img_scale = 80
    board_width, board_height, square_size = util.get_checkerboard_config()

    # Termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Take sample frames from videos
    util.sample_frames_from_video('data/cam1/intrinsics.avi', 'data/cam1/frames/', num_frames)
    util.sample_frames_from_video('data/cam2/intrinsics.avi', 'data/cam2/frames/', num_frames)
    util.sample_frames_from_video('data/cam3/intrinsics.avi', 'data/cam3/frames/', num_frames)
    util.sample_frames_from_video('data/cam4/intrinsics.avi', 'data/cam4/frames/', num_frames)

    calibrate_camera(glob.glob('data/cam1/frames/*.jpg'), img_scale, (board_width, board_height), criteria)
    calibrate_camera(glob.glob('data/cam2/frames/*.jpg'), img_scale, (board_width, board_height), criteria)
    calibrate_camera(glob.glob('data/cam3/frames/*.jpg'), img_scale, (board_width, board_height), criteria)
    calibrate_camera(glob.glob('data/cam4/frames/*.jpg'), img_scale, (board_width, board_height), criteria)