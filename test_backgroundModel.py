import numpy as np
import cv2 as cv


def bg_model_gaussian(bg_video):
    '''Create model of background frame from bg_video by calculating the Gaussian distribution
    of each channel of the pixels.'''

    frames = []
    vidcap = cv.VideoCapture(bg_video)
    success, frame = vidcap.read()

    while success:
        frames.append(frame)
        success, frame = vidcap.read()
    
    frames = np.array(frames)

    # Array of pixel values distribution. For each channel, compute the mean and std-dev.
    # shape[1] = width, shape[2] = height, shape[3] = num channels, 2 = mean and std-dev
    bg_dist = np.zeros((frames.shape[1], frames.shape[2], frames.shape[3], 2))

    for row in range(frames.shape[1]):
        for col in range(frames.shape[2]):
            for channel in range(frames.shape[3]):
                bg_dist[row][col][channel][0] = np.mean(frames[:, row, col, channel])
                bg_dist[row][col][channel][1] = np.std(frames[:, row, col, channel])

    return bg_dist


def bg_model_simple_average(bg_video):
    '''Create model of background frame by averaging frames from bg_video.'''

    vidcap = cv.VideoCapture(bg_video)
    success, image = vidcap.read()
    if not success:
        print(f'Unable to read {bg_video}')
    else:
        avg_val = np.float32(image)
        n = 0

    while success:
        success, image = vidcap.read()
        if success:
            cv.accumulateWeighted(image, avg_val, 0.01)
            n += 1
            print(f'Processing {n} frames')
                
    
    print(f'Done processing {n} frames.')
    
    avg_frame = cv.convertScaleAbs(avg_val)
    cv.imshow('average-frame', avg_frame)
    cv.waitKey(10000)
    vidcap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    create_background_model('data/cam1/background.avi')

