import cv2 as cv2
import cvzone
import numpy as np
#print (cv2.__version__)

def preProcessing(imgOriginal):
    # filtra a imagem inicial para algo que possa ser processado
    imgPre = cv2.GaussianBlur(imgOriginal, (5,5), 3)

    # # Create the sharpening kernel
    # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    # # Apply the sharpening kernel to the image using filter2D
    # imgPre = cv2.filter2D(imgPre, -1, kernel)

    thresh1 = cv2.getTrackbarPos("Threshould1", "Settings")
    thresh2 = cv2.getTrackbarPos("Threshould2", "Settings")
    imgPre = cv2.Canny(imgPre, thresh1, thresh2) # os threshoulds vao de 0-255; identifica as bordas
    return imgPre

# uma funcao de callback que nao faz nada
def empty(value):
    pass

cap = cv2.VideoCapture(0)    # recebe imagem da webcam
cap.set(3, 640)
cap.set(4, 480)

cv2.namedWindow("Settings")
cv2.resizeWindow("Settings", 640, 240)
cv2.createTrackbar("Threshould1", "Settings", 50, 255, empty)
cv2.createTrackbar("Threshould2", "Settings", 100, 255, empty)

while True:
    sucess, img = cap.read()
    if not sucess:
        break
    
    imgPre = preProcessing(img)

    imgStack = cvzone.stackImages([img, imgPre], 2, 1)
    cv2.imshow('webcam',imgStack)     # abre uma janela com a imagem
    if cv2.waitKey(1) & 0xFF == ord('q'): # condicao de saida do loop
        break
        

cap.release()
cv2.destroyAllWindows()