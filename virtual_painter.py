import cv2
import numpy as np
import os
import hand_tracking_module as htm

# Brush and eraser thickness
brushThickness = 15
eraserThickness = 50

# Load heaqqder images
folderPath = "header"
myList = os.listdir(folderPath)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

header = overlayList[0]
drawColor = (255, 0, 255)

# Video capture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

detector = htm.handDetector(detectionCon=0.85)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    # 1. Import image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 2. Find hand landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]  # Index finger tip
        x2, y2 = lmList[12][1:] # Middle finger tip

        # 3. Fingers state
        fingers = []
        tipIds = [4, 8, 12, 16, 20]
        for id in range(0, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # 4. Selection mode
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            if y1 < 125:
                imgWidth = 1280
                sectionWidth = imgWidth // len(overlayList)
                selectedSection = x1 // sectionWidth
                header = overlayList[selectedSection]

                # Assign colors
                if selectedSection == len(overlayList) - 1:
                    drawColor = (0, 0, 0)  # Eraser
                elif selectedSection == 0:
                    drawColor = (255, 0, 255)
                elif selectedSection == 1:
                    drawColor = (255, 0, 0)
                elif selectedSection == 2:
                    drawColor = (0, 255, 0)

            cv2.rectangle(img, (x1, y1-25), (x2, y2+25),
                          drawColor, cv2.FILLED)

        # 5. Drawing mode
        if fingers[1] and fingers[2] == 0:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                # Eraser → clears strokes on canvas
                cv2.line(imgCanvas, (xp, yp), (x1, y1),
                         (0, 0, 0), eraserThickness)
            else:
                cv2.line(imgCanvas, (xp, yp), (x1, y1),
                         drawColor, brushThickness)

            xp, yp = x1, y1

    # 6. Merge canvas and main image
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

    img = cv2.bitwise_and(img, imgInv)   # remove strokes from webcam
    img = cv2.bitwise_or(img, imgCanvas) # add strokes back

    # 7. Add header
    img[0:125, 0:1280] = header

    cv2.imshow("Virtual Painter", img)
    cv2.imshow("Canvas", imgCanvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()