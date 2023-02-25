import numpy as np
import cv2 as cv


def create_background_model(bg_video):
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

