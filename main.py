import numpy as np
import cv2 as cv
import glob

import util
from coordinate_clicker import CoordinateClicker
from corner_finder2 import interpolate
from corner_finder1 import transform_perspective


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

    cams = [
        {'ins': 'data/cam1/frames_in/*.jpg', 'exs': 'data/cam1/frames_ex/frame.jpg'},
        {'ins': 'data/cam2/frames_in/*.jpg', 'exs': 'data/cam2/frames_ex/frame.jpg'},
        {'ins': 'data/cam3/frames_in/*.jpg', 'exs': 'data/cam3/frames_ex/frame.jpg'},
        {'ins': 'data/cam4/frames_in/*.jpg', 'exs': 'data/cam4/frames_ex/frame.jpg'},
    ]

    for cam in cams:
        print('======= PROCESSING CAMERA =======')

        # INTRINSICS
        ret, mtx, dist, rvecs, tvecs = calibrate_camera(objp, glob.glob(cam['ins']), img_scale, (board_width, board_height), criteria)


        # EXTRINSIS
        img = cv.imread(cam['exs'])
        dim = (int(img.shape[1] * img_scale / 100), int(img.shape[0] * img_scale / 100))
        img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        ret, corners = cv.findChessboardCorners(gray, (board_width, board_height), None)

        if not ret:
            # Find corners manually
            print("PLEASE CLICK ON THE FOUR INNER CORNERS IN THIS ORDER: TopLeft, TopRight, BottomLeft, BottomRight")
            clicker = CoordinateClicker(img)
            while len(clicker.coordinates) < 4:
                cv.imshow('img', img)
                cv.setMouseCallback('img', clicker.click_event)
                cv.waitKey(50)
            
            # linear interpolation
            corners = interpolate(clicker.coordinates, board_width, board_height)

        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)

        # VISUALIZATION
        # mark the chessboard corners
        cv.drawChessboardCorners(img, (board_width, board_height), corners2, True)

        # project 3D points to image plane and draw axis
        axis = np.float32([[4,0,0], [0,4,0], [0,0,4]]).reshape(-1,3)
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
        img = util.draw_axes(img, corners2, imgpts)      

        cv.imshow('img',img)
        cv.waitKey(5000)

    cv.destroyAllWindows()
