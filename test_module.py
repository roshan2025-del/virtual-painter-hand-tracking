import cv2
import hand_tracking_module as htm

detector = htm.handDetector(detectionCon=0.7)

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    
    if len(lmList) != 0:
        print(lmList[4])  # Example: prints thumb tip coordinates

    cv2.imshow("Hand Module Test", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()