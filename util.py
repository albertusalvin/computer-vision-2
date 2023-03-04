import numpy as np
import cv2 as cv
from bs4 import BeautifulSoup
import xml.etree.cElementTree as ET
from xml.dom import minidom

def get_camera_rotation_matrix(cam):
    K, dist_coeffs, rvec, tvec = get_camera_config(cam)
    # Compute the rotation matrix from the rotation vector using the Rodrigues formula
    R, _ = cv.Rodrigues(rvec)

    # Construct the extrinsic matrix using the rotation and translation vectors
    T_mat = np.eye(4)
    T_mat[:3, :3] = R
    T_mat[:3, 3] = tvec.reshape(3)

    # Compute the camera matrix with the extrinsic matrix and intrinsic matrix
    K_ext = np.dot(K, T_mat[:3, :])

    # # Compute the projection matrix using the camera matrix and distortion coefficients
    # P_mat = np.dot(K_ext, np.hstack([R, tvec.reshape(3, 1)]))
    # P_mat = np.dot(P_mat, np.hstack([np.eye(3), np.zeros((3, 1))]))

    # The rotation matrix is the upper-left 3x3 submatrix of the extrinsic matrix
    R_mat = np.eye(4)
    R_mat[:3, :3] = T_mat[:3, :3]
    return R_mat

def get_camera_config(cam):
    '''Read the config from for a camera
    '''
    with open('computervision2/data/cam%s/config.xml' % cam, 'r') as f:
        data = f.read()

    soup = BeautifulSoup(data, 'xml')
    camera_matrix = convert_xml_to_matrix(soup, "CameraMatrix")
    distortion_coeffs = convert_xml_to_matrix(soup, "DistortionCoeffs")
    rotation_vector = convert_xml_to_matrix(soup, "RotationVector")
    translation_vector = convert_xml_to_matrix(soup, "TranslationVector")
    return camera_matrix, distortion_coeffs, rotation_vector, translation_vector

def convert_xml_to_matrix(soup, name):
    # Find the camera matrix element in the XML file
    cam_matrix = soup.find(name)

    # Extract the rows and columns from the XML file
    rows = int(cam_matrix.find('rows').string)
    cols = int(cam_matrix.find('cols').string)

    # Create NumPy matrix with the correct dimensions
    matrix = np.zeros((rows, cols))

    # Extract the matrix values from the XML file
    values = cam_matrix.find('data').string.split()
    print(values)
    for i, value in enumerate(values):
        matrix[i // cols, i % cols] = float(value)

    return matrix

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


def save_camera_config(mtx, dist, rvecs, tvecs, cam):
    mtx_str = f"\n{'{:.5f}'.format(mtx[0][0])} {'{:.5f}'.format(mtx[0][1])} {'{:.5f}'.format(mtx[0][2])}\n"\
            f"{'{:.5f}'.format(mtx[1][0])} {'{:.5f}'.format(mtx[1][1])} {'{:.5f}'.format(mtx[1][2])}\n"\
            f"{'{:.5f}'.format(mtx[2][0])} {'{:.5f}'.format(mtx[2][1])} {'{:.5f}'.format(mtx[2][2])}"
    
    dst_str = f"\n{'{:.5f}'.format(dist[0][0])}\n"\
            f"{'{:.5f}'.format(dist[0][1])}\n"\
            f"{'{:.5f}'.format(dist[0][2])}\n"\
            f"{'{:.5f}'.format(dist[0][3])}\n"\
            f"{'{:.5f}'.format(dist[0][4])}"

    rvec_str = f"\n{'{:.5f}'.format(rvecs[0][0])}\n"\
                f"{'{:.5f}'.format(rvecs[1][0])}\n"\
                f"{'{:.5f}'.format(rvecs[2][0])}"

    tvec_str = f"\n{'{:.5f}'.format(tvecs[0][0])}\n"\
                f"{'{:.5f}'.format(tvecs[1][0])}\n"\
                f"{'{:.5f}'.format(tvecs[2][0])}"

    root = ET.Element('opencv_storage')
    cam_mtx = ET.SubElement(root, 'CameraMatrix', type_id='opencv-matrix')
    ET.SubElement(cam_mtx, 'rows').text = '3'
    ET.SubElement(cam_mtx, 'cols').text = '3'
    ET.SubElement(cam_mtx, 'dt').text = 'f'
    ET.SubElement(cam_mtx, 'data').text = mtx_str

    dist_coef = ET.SubElement(root, 'DistortionCoeffs', type_id='opencv-matrix')
    ET.SubElement(dist_coef, 'rows').text = '5'
    ET.SubElement(dist_coef, 'cols').text = '1'
    ET.SubElement(dist_coef, 'dt').text = 'f'
    ET.SubElement(dist_coef, 'data').text = dst_str

    rot_vec = ET.SubElement(root, 'RotationVector', type_id='opencv-matrix')
    ET.SubElement(rot_vec, 'rows').text = '3'
    ET.SubElement(rot_vec, 'cols').text = '1'
    ET.SubElement(rot_vec, 'dt').text = 'f'
    ET.SubElement(rot_vec, 'data').text = rvec_str

    tran_vec = ET.SubElement(root, 'TranslationVector', type_id='opencv-matrix')
    ET.SubElement(tran_vec, 'rows').text = '3'
    ET.SubElement(tran_vec, 'cols').text = '1'
    ET.SubElement(tran_vec, 'dt').text = 'f'    
    ET.SubElement(tran_vec, 'data').text = tvec_str

    tree = ET.ElementTree(root)
    ET.indent(tree, space='    ', level=0)
    tree.write('data/cam%s/config.xml' % cam, encoding='utf-8')


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


def generate_frames_for_background():
    sample_frames_from_video('data/cam1/background.avi', 'data/cam1/frames_bg/', 1)
    sample_frames_from_video('data/cam2/background.avi', 'data/cam2/frames_bg/', 1)
    sample_frames_from_video('data/cam3/background.avi', 'data/cam3/frames_bg/', 1)
    sample_frames_from_video('data/cam4/background.avi', 'data/cam4/frames_bg/', 1)

def generate_frames_for_foreground():
    sample_frames_from_video('data/cam1/video.avi', 'data/cam1/frames_fg/', 1)
    sample_frames_from_video('data/cam2/video.avi', 'data/cam2/frames_fg/', 1)
    sample_frames_from_video('data/cam3/video.avi', 'data/cam3/frames_fg/', 1)
    sample_frames_from_video('data/cam4/video.avi', 'data/cam4/frames_fg/', 1)


def draw_axes(img, corners, imgpts):
    '''Project three dimensional axes onto the img and draw a line for each axis.'''

    corner = tuple(corners[0].ravel().astype(int))
    # bright yellow (69,233,255)
    img = cv.line(img, corner, tuple(imgpts[0].ravel().astype(int)), (0,0,255), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel().astype(int)), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel().astype(int)), (255,0,0), 5)
    return img