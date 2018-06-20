import cv2
class opencv(object):
    """class about computer vision """
    def __init__(self,com_index=0):
        self._com = com_index

    def camera(self):
        cap = cv2.VideoCapture(self._com)
        while cap.isOpened():
            ret,frame= cap.read()
            if frame is not None:
                cv2.imshow('camera',frame)
                key = cv2.waitKey(1) & 0XFF
                if key == 27:
                    break
            else:
                print('No image captured!!!')
                break
        print('Closing and quiting...')
        cv2.destroyAllWindows()
        cap.release()
