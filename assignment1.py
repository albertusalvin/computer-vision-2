import numpy as np
import cv2 as cv
import glob

from corner_finder2 import transformation, interpolate
from corner_finder1 import transform_perspective


def interface_find_chessboard_corners(img, size, tileSize):
    '''Given a checkerboard image and its dimensions, attempt to use OpenCV's findChessboardCorners() function to detect
    the locations of its inner corners. If fail, then ask the user to click on the four corners, and interpolate the
    locations of other inner corners from the given four points.'''
    
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, size, flags=cv.CALIB_USE_INTRINSIC_GUESS)
    if ret == True:
        return ret, corners, 'automatic'
    else:
        # ask for manual input
        print("PLEASE CLICK ON THE FOUR INNER CORNERS IN THIS ORDER: TopLeft, TopRight, BottomLeft, BottomRight")

        man = ManualCornersAnnotate([])
        while len(man.corn) < 4:    # Make sure 4 corners get selected
            cv.imshow('img', img)
            cv.setMouseCallback('img', man.click_event)
            cv.waitKey(50)

        # linear interpolation
        intersections = interpolate(man.corn, size[0], size[1])
        # intersections = transform_perspective(man.corn, size[0], size[1], tileSize)
        print(f"{len(intersections)} intersections are found")
        return True, intersections, 'manual'


def draw_axes(img, corners, imgpts):
    '''Project three dimensional axes onto the img and draw a line for each axis.
    '''
    corner = tuple(corners[0].ravel().astype(int))
    # bright yellow (69,233,255)
    img = cv.line(img, corner, tuple(imgpts[0].ravel().astype(int)), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel().astype(int)), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel().astype(int)), (255,0,0), 5)
    return img


def draw_box(img, corners, imgpts):
    '''Draw a cube on the img.
    '''
    imgpts = np.int32(imgpts).reshape(-1,2)
    # floor
    img = cv.drawContours(img, [imgpts[:4]], -1, (63,158,252), -3)
    # pillars
    for i,j in zip(range(4),range(4,8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (0,0,255), 3)
    # roof
    img = cv.drawContours(img, [imgpts[4:]], -1, (0,0,255), 3)
    return img


class ManualCornersAnnotate():
    ''' Define an object to keep track of the locations that has been clicked.'''
    corn = []
    
    def __init__(self, arr):
        self.corn = arr
    
    def click_event(self, event, x, y, flags, params):
        ''' Registers left mouse button clicks and record the coordinate point. 
        Draws a circle around the click location.
        '''
        if event==cv.EVENT_LBUTTONDOWN:
            cv.circle(img, (x,y), 10, (255, 255, 0), 3)
            cv.imshow('img', img)
            self.corn.append(np.array([x,y], dtype=np.float32))


# ######################## Finding corners ########################


'''IMPORTANT: 
1. Assign the location of the calibration images to variable "calibrationImages".
2. Assign the location of the test images to variable "testImages".
3. To scale the image size for improved performance and to show the entire image fully for manual entry, 
set variable "scale_percent" to the percent of original image size. 
4. OpenCV's ability to auto-detect the corners HEAVILY depends on the dimension of the checkerboard
you pass to it. Make sure variable "numRows" & "numCols" match the exact number of INNER corners of your board.
5. Measure the side of a tile in the real world (in millimeters) and assign to variable "tileSize".
6. Set the duration (in milliseconds) for showing the detected corner points in variable "duration_show_corners".
7. Set the duration (in milliseconds) for showing the cube on each test image in variable "duration_show_cube".
'''
calibrationImages = glob.glob('images-run-1/*.jpg')
testImages = glob.glob('images-test/*.jpg')
scale_percent = 30
numRows = 9
numCols = 6
tileSize = 25
duration_show_corners = 3000
duration_show_cube = 3000

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
cell_size = 4.7

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((numCols*numRows,3), np.float32)
objp[:,:2] = np.mgrid[0:numRows, 0:numCols].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
autofound = []
manualfound = []


for fname in calibrationImages:
    print(f'Processing {fname}')
    img = cv.imread(fname)
    scale_dim = (int(img.shape[1] * scale_percent / 100), int(img.shape[0] * scale_percent / 100))
    img = cv.resize(img, scale_dim, interpolation = cv.INTER_AREA)  # Scales the image down
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # ret, corners = cv.findChessboardCorners(gray, (7,7), None)
    ret, corners, method = interface_find_chessboard_corners(img, (numRows,numCols), tileSize)
    
    # If found, add object points, image points (after refining them)
    corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
    objpoints.append(objp)
    imgpoints.append(corners2)

    if method == 'automatic':
        autofound.append(fname)
    else:
        manualfound.append(fname)

    # Draw and display the corners
    cv.drawChessboardCorners(img, (numRows,numCols), corners2, True)
    cv.imshow('img', img)
    cv.waitKey(duration_show_corners)


print(f'Auto found {len(autofound)}')
print(autofound)
print(f'Manual found {len(manualfound)}')
print(manualfound)


# ######################## Calibration part ########################


ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print('CAMERA INTRINSICS')
print(mtx)


# ######################## Testing with test images ########################


for fname in testImages:
    img = cv.imread(fname)
    dim = (int(img.shape[1] * scale_percent / 100), int(img.shape[0] * scale_percent / 100))
    img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    
    ret, corners = cv.findChessboardCorners(gray, (numRows,numCols), flags=cv.CALIB_USE_INTRINSIC_GUESS)
    
    if ret == True:
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)   # Improve the accuracy of the corner
        ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)

        print('CAMERA EXTRINSICS')
        print(rvecs)
        print(tvecs)
        
        # mark the chessboard corners
        cv.drawChessboardCorners(img, (numRows,numCols), corners2, True)

        # project 3D points to image plane and draw axis
        axis = np.float32([[4,0,0], [0,4,0], [0,0,-4]]).reshape(-1,3)
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
        img = draw_axes(img, corners2, imgpts)
        
        # project 3D points and draw a cube
        axis_box = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                        [0,0,-3], [0,3,-3], [3,3,-3], [3,0,-3]])
        imgpts_box, jac = cv.projectPoints(axis_box, rvecs, tvecs, mtx, dist)
        img = draw_box(img, corners2, imgpts_box)        

        cv.imshow('img',img)
        cv.waitKey(duration_show_cube)

cv.destroyAllWindows()

