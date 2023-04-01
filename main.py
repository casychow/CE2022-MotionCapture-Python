# Tutorial from https://www.computervision.zone/lessons/video-lessons-4/

import cv2
from cvzone.PoseModule import PoseDetector
# PoseDetector has 33 landmarks in which we will be extracting
# 33 landmarks * 3 axis (x,y,z) = 99 parameters
# eg. x1,y1,z1,x2,y2,z2,x3,y3,z3

cap = cv2.VideoCapture("Video.mp4")
detector = PoseDetector()  # no parameters are necessary rn
posList = []  # position list

while True:
    success, img = cap.read()
    img = detector.findPose(img)

    # lmList = landmark list
    # bboxInfo = bounding box info
    lmList, bboxInfo = detector.findPosition(img)

    # if pose is detected and there is an overall bounding box for the person
    if bboxInfo:
        # landmark string contains all 33 points of our pose
        lmString = ""
        for lm in lmList:
            # format of lm --> [lm #, x coord, y coord, z coord]
            # opencv starts the y coord at the top left corner of the image
            # unity starts the y coord at the bottom left corner of the image
            # we will feed this file into unity so we will calculate the correct y coord for unity to use:
                # img.shape[0] - lm[2]
            lmString += f'{lm[1]},{img.shape[0] - lm[2]},{lm[3]},'
        posList.append(lmString)

    print(len(posList))

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)  # delay of 1ms
    if key == ord('s'):
        with open("AnimationFile.txt", "w") as f:
            f.writelines(["%s\n" % item for item in posList])
