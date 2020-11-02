import cv2
import os
import numpy as np

if __name__ == '__main__':
    path = 'C:/Users/ACER/Desktop/Semestre10/Imagenes/Presentaciones/Semana 12'
    image_name = 'vtest.avi'
    path_file = os.path.join(path, image_name)

    camera = cv2.VideoCapture(path_file)
    ret = True

    while ret:
        ret, frame = camera.read()
        if ret:
            ################################################################
            ######### 1. Generate Gaussian and Laplacian Pyramid ###########
            ################################################################

            # Generate Gaussian Pyramid for the frame
            G = np.array(frame.copy())
            GPyramid = [G]
            for i in range(5):
                G = cv2.pyrDown(G)
                GPyramid.append(G)

            # Show Gaussian Pyramid
            for i in range(0, len(GPyramid)):
                img = GPyramid[i]
                cv2.imshow('Gaussian Pyramid frame', img)
                cv2.waitKey(0)

            # Generate Laplacian Pyramid
            LPyramid = []
            for i in range(5, 0, -1):
                GE = cv2.pyrUp(GPyramid[i])
                L = cv2.subtract(GPyramid[i - 1], GE)
                LPyramid.append(L)

            # Show Laplacian Pyramid
            for i in range(0, len(LPyramid)):
                img = LPyramid[i]
                cv2.imshow('Laplacian Pyramid frame', img)
                cv2.waitKey(0)

            ################################################################
            ######### 2. Apply bandPass filtering:   #######################
            ################################################################







