import os

import cv2
import fire
import numpy as np

drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1


class Selector:
    """
    WIP
    """
    def __init__(self, pickled_vis_file):
        self.pickled_vis_file = pickled_vis_file
        self.img = None

    def select_region(self, event, x, y, flags, param):
        global ix, iy, drawing, mode

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                if mode == True:
                    cv2.rectangle(self.img, (ix, iy), (x, y), (0, 255, 0), -1)
                else:
                    cv2.circle(self.img, (x, y), 5, (0, 0, 255), -1)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if mode == True:
                cv2.rectangle(self.img, (ix, iy), (x, y), (0, 255, 0), -1)
            else:
                cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

    def test(self):

        cv2.namedWindow('test')
        cv2.setMouseCallback('test', select_region)

        while 1:
            cv2.imshow('test', img)
            if cv2.waitKey(20) & 0xFF == 27:
                break

        cv2.destroyAllWindows()


if __name__ == '__main__':
    fire.Fire(Selector)
    '''{
        'select': Selector
        # 'test': unittest.main()
    })
    '''