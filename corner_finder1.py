import numpy as np
import cv2 as cv


def transform_perspective(cornersCamWorld, nRows=6, nCols=6, tileSize=25):
    '''Compare the four corner points from the camera world (2D) to their locations in the real world (3D)
    to get the perspective transformation matrix. Then invert the matrix and use it to project the
    inner corners points from their locations in the real world (3D) onto the camera world (2D).
    '''    
    # "ground truth" locations of the nRows x nCols points
    pointsRealWorld = []
    for row in range(nRows):
        pointsRealWorld.extend([[col*tileSize, row*tileSize] for col in range(nCols)])

    # ground truth locations of the corners (up-left, up-right, low-left, low-right)
    cornersRealWorld = np.array([
        [0, 0],
        [(nCols-1)*tileSize, 0],
        [0, (nRows-1)*tileSize],
        [(nCols-1)*tileSize, (nRows-1)*tileSize]
    ], dtype=np.float32)

    # sort the points by the X, then by the Y
    # the order will be, from human eyes perspective: upper-left, upper-right, lower-left, lower-right 
    cornersCamWorld = sorted(cornersCamWorld, key=lambda p: p[0])
    cornersCamWorld = sorted(cornersCamWorld, key=lambda p: p[1])
    cornersCamWorld = np.array(cornersCamWorld, dtype=np.float32)
    
    transMatrix = cv.getPerspectiveTransform(cornersCamWorld, cornersRealWorld)
    _, invTransMatrix = cv.invert(transMatrix)
    
    # use the matrix to project points from real world onto the cam world
    pointsCamWorld = []
    for p in pointsRealWorld:
        x, y, z = np.dot(invTransMatrix, np.append(p, [1]))
        pointsCamWorld.append([x/z, y/z])

    return np.array(pointsCamWorld, dtype=np.float32)


def equidistant_points(point1, point2, N):
    '''Calculate the positions of N equidistant points along a straight line between point1 and point2.
    '''
    x1, y1 = point1
    x2, y2 = point2
    dx = (x2 - x1) / (N + 1)
    dy = (y2 - y1) / (N + 1)
    points = [(x1 + i * dx, y1 + i * dy) for i in range(1, N + 1)]
    return points


def line_equation(point1, point2):
    '''Given a straight line between point1 and point2, calculate the line equation in y = mx + b form
    and return the coefficients m and b .
    '''
    x1, y1 = point1
    x2, y2 = point2
    try:
        m = (y2 - y1) / (x2 - x1)
        b = -m * x1 + y1
        return (m, b)
    except ZeroDivisionError:
        # x1 == x2, a vertical line
        return (x2, "vertical")


def intersection_point(line1, line2):
    '''Given two straight lines, each represented in (m, b) as in y = mx + b, find the intersection point.
    '''
    m1, b1 = line1
    m2, b2 = line2
    if b1 == "vertical" and b2 == "vertical" and m1 != m2:
        # vertical lines on 2 different X's
        return "parallel vertical lines"
    elif b1 == "vertical":
        yCross = m2 * m1 + b2
        return (m1, yCross)
    elif b2 == "vertical":
        yCross = m1 * m2 + b1
        return (m2, yCross)
    elif m1 == m2:
        # parallel lines
        return "parallel lines"
    else:
        xCross = (b1 - b2) / (m2 - m1)
        yCross = ((m2 * b1) - (m1 * b2)) / (m2 - m1)
        return (xCross, yCross)


def grid_intersections(corners, N):
    '''
    Finding the intersection points of a grid of size (N+1, N+1) in a rectangle.

    Params:
    - corners: array of (x,y), denoting the 4 corners of the rectangle. It is assumed that the array only contains 4 elements.
    - N: number of equidistant points to be placed along the rectangle edges. The number of gaps, therefore, is N+1.

    Returns:
    Array of (x,y); all the intersection points.
    '''
    # sort the points by the X, then by the Y
    # the order will be, from human eyes perspective: upper-left, upper-right, lower-left, lower-right 
    corners = sorted(corners, key=lambda p: p[0])
    corners = sorted(corners, key=lambda p: p[1])
    
    # the points along the sides
    topPoints = equidistant_points(corners[0], corners[1], N)
    bottomPoints = equidistant_points(corners[2], corners[3], N)
    leftPoints = equidistant_points(corners[0], corners[2], N)
    rightPoints = equidistant_points(corners[1], corners[3], N)

    # the lines
    verticalLines = [line_equation(topPoints[i], bottomPoints[i]) for i in range(N)]
    horizontalLines = [line_equation(leftPoints[i], rightPoints[i]) for i in range(N)]

    # the intersections
    intersections = []
    for i in range(len(verticalLines)):
        for j in range(len(horizontalLines)):
            intersections.append(intersection_point(verticalLines[i], horizontalLines[j]))
    
    # include points along the edges and the 4 corners
    intersections.extend(topPoints)
    intersections.extend(bottomPoints)
    intersections.extend(leftPoints)
    intersections.extend(rightPoints)
    intersections.extend(corners)

    # sort the points by the X, then by the Y
    intersections = sorted(intersections, key=lambda p: p[0])
    intersections = sorted(intersections, key=lambda p: p[1])
    
    return np.array(intersections, dtype=np.float32)
