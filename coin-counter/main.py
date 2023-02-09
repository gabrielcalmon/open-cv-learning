import cv2 as cv
import cvzone
#print (cv.__version__)

cap = cv.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)


while True:
    sucess, img = cap.read()
    cv.imshow('webcam',img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()