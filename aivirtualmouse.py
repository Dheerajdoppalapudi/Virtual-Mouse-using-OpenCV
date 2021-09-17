import cv2
import time
import numpy as np
import OpenCV.handtrackingmodule as htm
import autopy

wCam, hCam = 640, 480
frameR = 100
smoothening = 5

plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0
cTime = 1

detector = htm.handDetector(maxHands=1)

wScr, hScr = autopy.screen.size()
print(wScr, hScr)

while True:
    # 1. find hand landmarks
    # 2. get the tip of index and middle fingers
    # 3. check which fingers are up
    # 4. only index finger in moving mode
    # 5. convert coordinates (your screen values)
    # 6. smoothen values
    # 7. move mouse
    # 8. when both index and middle fingers up we are in clicking mode
    # 9. find the distance b/w fingers
    # 10. click if distance is short
    # 11. frame rate
    # 12 Display

    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        #print(x1, y1, x2, y2)

        fingers = detector.fingersUp()
        #print(fingers)

        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        if fingers[1] == 1 and fingers[2] == 0:
            x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))

            clocX = plocX + (x3-plocX)/smoothening
            clocY = plocY + (y3 - plocY)/smoothening

            autopy.mouse.move(wScr-clocX,clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        if fingers[1] == 1 and fingers[2] == 1:
            length, img, info = detector.findDistance(8, 12, img)
            print(length)
            if length < 40:
                cv2.circle(img, (info[4], info[5]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2)
    img = cv2.flip(img, 1)
    cv2.imshow("Image", img)
    cv2.waitKey(1)