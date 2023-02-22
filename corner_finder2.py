import numpy as np

def interpolate(points, n, m): 
    '''Interpolate the locations of inner corners points, 
    given the four corner points.
    '''
    dx1 = points[2] - points[0]     # Define line vector from the bottom left to top left
    dx2 = points[3] - points[1]     # Define line vector from the bottom right to top right

    corners = []
    for i in np.linspace(0, 1, m):
        corners += add_corners(dx1*i + points[0], dx2*i + points[1], n)
    return np.array(corners, np.float32)


def add_corners(pointa, pointb, n):
    '''It takes two points and draws a line inbetween them, 
    then it evenly spaces n points on this line, these are the corner points.
    '''
    d = pointa - pointb # Define line vector
    corners = []
    for i in np.linspace(0,1,n):    # Will result in a list of form [0,..,1] of length n where is element is evenly spaced apart
        corners.append(i*d+pointb)  # Trace a certain distance from the line and select to get a point
    return corners


class transformation():
    '''Below is our attempt to create a transformation matric to find the corner points. 
    The idea was to overcome the issue of squares closer to the camera being bigger then those further away.
    But we where unable to get it to work as accurate as the naive interpolation function.
    '''
    transform_to: np.ndarray
    transform_from: np.ndarray

    
    def create_matrix(self, points):
        '''Takes a list of points in form x y ordered by corer 0 0, 1 0, 0 1, 1 1 . 
        And creates the transformation matricies for them.
        '''
        points = np.array([np.append(point, [1]) for point in points])
        solutions_x = np.array([0, 1, 0, 1])
        solutions_y = np.array([0, 0, 1, 1])
        val_x, res, r, s = np.linalg.lstsq(points, solutions_x, rcond=None)
        val_y, res, r, s = np.linalg.lstsq(points, solutions_y, rcond=None)
        self.transform_to = np.matrix([val_x, val_y, [0, 0, 1]])
        self.transform_from = np.linalg.inv(self.transform_to)
    
    def interpolate_points_outer(self, n):
        corners = []
        for j in np.linspace(0, 1, n):
            for i in np.linspace(0, 1, n):
                chess_view = np.array([i, j, 1])
                camera_view = np.dot(self.transform_from, chess_view)
                camera_view = np.squeeze(np.asarray(camera_view))
                corners.append(camera_view.astype(np.float32)[:-1])
        return np.array(corners)

    def interpolate_points_inner(self, n):
        corners = []
        for j in range(n):
            for i in range(n):
                chess_view = np.array([i, j, 1])
                camera_view = np.dot(self.transform_from, chess_view)
                camera_view = np.squeeze(np.asarray(camera_view))
                corners.append(camera_view.astype(np.float32)[:-1])
        return np.array(corners)

