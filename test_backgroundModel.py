import numpy as np
import cv2 as cv
from tqdm import tqdm

# Credit: https://dontrepeatyourself.org/post/how-to-remove-background-with-opencv/


def substract_background(bg_video, fg_video):
    backsub = cv.createBackgroundSubtractorMOG2()

    for vid in [bg_video, fg_video]:
        vidcap = cv.VideoCapture(vid)

        while True:
            success, frame = vidcap.read()

            if not success:
                break

            # Comparing the frame against the background frame (in this case, the first frame)
            mask = backsub.apply(frame, learningRate=0)

            # Frame, minus the masked out area
            new_frame = cv.bitwise_and(frame, frame, mask=mask)

            # Apply "opening" operation to the mask. It is erosion (removing the noise) followed by dilation (emphasizing the content)
            kernel = np.ones((5,5), np.uint8)
            opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)
            
            # Find the contours (the shape boundary) in the mask
            contours, _ = cv.findContours(opening, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            
            # Draw rectangles around the found contours. Not on the mask opening frame, but on the corresponding original frame.
            for contour in contours:
                if cv.contourArea(contour) > 1000:
                    (x, y, w, h) = cv.boundingRect(contour)
                    cv.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)

            # Display the results
            mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
            opening = cv.cvtColor(opening, cv.COLOR_GRAY2BGR)
            hstacked_frames = np.hstack((frame, new_frame))
            hstacked_frames1 = np.hstack((mask, opening))
            vstacked_frames = np.vstack((hstacked_frames, hstacked_frames1))
            cv.imshow('Frame, new frame, mask, opening operation', vstacked_frames)
            if cv.waitKey(30) == ord("q"): 
                break
        
        vidcap.release()
    cv.destroyAllWindows()


def background_averaging(bg_video):
    frames = []
    vidcap = cv.VideoCapture(bg_video)
    success, frame = vidcap.read()

    while success:
        frames.append(frame)
        success, frame = vidcap.read()
    
    frames = np.array(frames)
    bg_avg_img = np.full((frames.shape[1], frames.shape[2], frames.shape[3]), 255, dtype=np.uint8)

    for row in tqdm(range(frames.shape[1])):
        for col in range(frames.shape[2]):
            for channel in range(frames.shape[3]):
                mean = np.mean(frames[:, row, col, channel])
                bg_avg_img[row][col][channel] = mean

    print('Showing averaged background')
    cv.imshow('averaged-background', bg_avg_img)
    cv.waitKey(10000)
    cv.destroyAllWindows()    


def background_modeling_gaussian(bg_video):
    '''Create model of background frame from bg_video by calculating the Gaussian distribution
    of each channel of the pixels.'''

    frames = []
    vidcap = cv.VideoCapture(bg_video)
    success, frame = vidcap.read()

    while success:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        frames.append(frame)
        success, frame = vidcap.read()
    
    frames = np.array(frames)

    # Array of pixel values distribution. For each channel, compute the mean and std-dev.
    # shape[1] = width, shape[2] = height, shape[3] = num channels, 2 = mean and std-dev
    bg_dist = np.zeros((frames.shape[1], frames.shape[2], frames.shape[3], 2))
    print('Calculating background pixel value distribution ...')

    for row in tqdm(range(frames.shape[1])):
        for col in range(frames.shape[2]):
            for channel in range(frames.shape[3]):
                mean = np.mean(frames[:, row, col, channel])
                std = np.std(frames[:, row, col, channel])
                bg_dist[row][col][channel][0] = int(mean - abs(std))    # lower bound
                bg_dist[row][col][channel][1] = int(mean + abs(std))    # upper bound

    return bg_dist


def compare_foreground_background(bg_model, fg_video):
    vidcap = cv.VideoCapture(fg_video)
    success, frame = vidcap.read()

    if not success:
        print(f'Unable to read {fg_video}')
    elif (bg_model.shape[0] != frame.shape[0]
        or bg_model.shape[1] != frame.shape[1]
        or bg_model.shape[2] != frame.shape[2]):
        print('Mismatching shapes between the background and foreground frames.')
    else:
        print('Creating foreground mask ...')
        mask = np.zeros((frame.shape[0], frame.shape[1], frame.shape[2]), dtype=np.uint8)
        
        while success:
            print('Comparing frame ...')
            # compare every pixel in the frame against the bg_model
            frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            for row in tqdm(range(frame.shape[0])):
                for col in range(frame.shape[1]):
                    in_treshold = True                    
                    
                    for chan in range(frame.shape[2]):
                        in_treshold_ = (frame[row][col][chan] >= bg_model[row][col][chan][0] 
                                    and frame[row][col][chan] <= bg_model[row][col][chan][1])
                        in_treshold = in_treshold and in_treshold_
                    
                    if in_treshold:
                        mask[row][col] = [0 for i in range(frame.shape[2])]
                    else:
                        mask[row][col] = [255 for i in range(frame.shape[2])]
            
            print('SHOWING MASk NOW ...')
            cv.imshow('mask-window', mask)
            cv.waitKey(2000)

            success, frame = vidcap.read()
        
        cv.destroyAllWindows()


def create_masking():
    # 350,55,100
    img = cv.imread('testimage.jpg')
    lower_bound = np.array([14,14,14])
    upper_bound = np.array([16,16,16])

    imgmask = cv.inRange(img, lower_bound, upper_bound)
    cv.imshow('image-window', imgmask)
    cv.waitKey(15000)
    cv.destroyAllWindows()


if __name__ == '__main__':
    substract_background('data/cam1/background.avi', 'data/cam1/video.avi')

