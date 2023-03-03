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




def substract_background(bg_video, fg_video):
    backsub = cv.createBackgroundSubtractorMOG2(history=5000, varThreshold=36, detectShadows=True)

    for vid in [bg_video, fg_video]:
        vidcap = cv.VideoCapture(vid)

        while True:
            success, frame = vidcap.read()

            if not success:
                break

            # Comparing the frame against the background frame (in this case, the first frame)
            mask = backsub.apply(frame, learningRate=0.00)

            # Frame, minus the masked out area
            new_frame = cv.bitwise_and(frame, frame, mask=mask)

            # Apply "opening" operation to the mask. 
            # It is erosion (removing the noise) followed by dilation (emphasizing the content).
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
            opening = cv.morphologyEx(mask, cv.MORPH_ERODE, kernel, iterations=1)
            opening[opening < 200] = 0      # Remove shadows

            # Find the contours (the shape boundary) in the mask
            contours, _ = cv.findContours(opening, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            # Turn to grayscale and remove shadows
            mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
            opening = cv.cvtColor(opening, cv.COLOR_GRAY2BGR)
            new_frame[mask < 200] = 0
            mask[mask < 200] = 0
            opening[opening < 200] = 0       
            #mask = cv.threshold(mask, thresh=200, maxval=255, type=cv.THRESH_BINARY)
            #opening = cv.threshold(opening, thresh=200, maxval=255, type=cv.THRESH_BINARY)

            # Draw rectangles around the found contours. Not on the mask opening frame, but on the corresponding original frame.
            for contour in contours:
                if cv.contourArea(contour) > 1000:
                    (x, y, w, h) = cv.boundingRect(contour)
                    cv.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)

            # Display the results
            hstacked_frames = np.hstack((frame, new_frame))
            hstacked_frames1 = np.hstack((mask, opening))
            vstacked_frames = np.vstack((hstacked_frames, hstacked_frames1))
            cv.imshow('Frame, new frame, mask, opening operation', vstacked_frames)
            if cv.waitKey(30) == ord("q"): 
                break
        
        vidcap.release()
    cv.destroyAllWindows()




if __name__ == '__main__':
    np.random.seed(10)
    num_frames = 10
    img_scale = 100
    board_width, board_height, square_size = util.get_checkerboard_config()

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((board_width * board_height, 3), np.float32)
    objp[:,:2] = np.mgrid[0:board_width, 0:board_height].T.reshape(-1,2)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001) # Termination criteria

    # Take sample frames from videos
    # WARN: only need to run this part once
    # util.generate_frames_for_intrinsics(num_frames)
    # util.generate_frames_for_extrinsics()
    # util.generate_frames_for_background()
    # util.generate_frames_for_foreground()

    cams = [
        {'ins': 'data/cam1/frames_in/*.jpg', 'exs': 'data/cam1/frames_ex/frame.jpg'},
        {'ins': 'data/cam2/frames_in/*.jpg', 'exs': 'data/cam2/frames_ex/frame.jpg'},
        {'ins': 'data/cam3/frames_in/*.jpg', 'exs': 'data/cam3/frames_ex/frame.jpg'},
        {'ins': 'data/cam4/frames_in/*.jpg', 'exs': 'data/cam4/frames_ex/frame.jpg'},
    ]

    for cam_idx, cam in enumerate(cams):
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

        # SAVE PARAMS
        util.save_camera_config(mtx, dist, rvecs, tvecs, f'data/cam{cam_idx + 1}/config.xml')

    cv.destroyAllWindows()
