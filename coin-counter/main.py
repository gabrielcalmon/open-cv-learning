import cv2 as cv2
import cvzone
import numpy as np
from cvzone.ColorModule import ColorFinder
#print (cv2.__version__)

myColorFinder = ColorFinder(False)
# Custom Orange Color
hsvVals = {'hmin': 10, 'smin': 55, 'vmin': 215, 'hmax': 42, 'smax': 255, 'vmax': 255}

# filtra a imagem inicial para algo que possa ser processado
def preProcessing(imgOriginal):
    #---BLUR---
    imgPre = cv2.GaussianBlur(imgOriginal, (5,5), 3)

    #---CANNY---
    thresh1 = cv2.getTrackbarPos("Threshould1", "Settings")
    thresh2 = cv2.getTrackbarPos("Threshould2", "Settings")
    imgPre = cv2.Canny(imgPre, thresh1, thresh2) # os threshoulds vao de 0-255; identifica as bordas

    #---DILATE & MORPH---
    kernel = np.ones((3, 3), np.uint8)
    imgPre = cv2.dilate(imgPre, kernel, iterations=1)
    imgPre = cv2.morphologyEx(imgPre, cv2.MORPH_CLOSE, kernel)
    return imgPre

# uma funcao de callback que nao faz nada
def empty(value):
    pass

cap = cv2.VideoCapture(0)    # recebe imagem da webcam
cap.set(3, 640)
cap.set(4, 480)

cv2.namedWindow("Settings")
cv2.resizeWindow("Settings", 640, 240)
cv2.createTrackbar("Threshould1", "Settings", 240, 255, empty)
cv2.createTrackbar("Threshould2", "Settings", 150, 255, empty)

# sizeMax = {
#     "10": 1200
#     "25": 1850
#     "50": 1500
#     "100": 2200 }    #10, 25, 50 e 100

while True:
    sucess, img = cap.read()
    if not sucess:
        break
    
    imgPre = preProcessing(img)
    imgContours, conFound = cvzone.findContours(img, imgPre, minArea=800, filter=8)

    totalMoney = 0
    if conFound:
        for contour in conFound:
            peri = cv2.arcLength(contour['cnt'], True)
            approx = cv2.approxPolyDP(contour['cnt'], 0.02 * peri, True)
            if len(approx)>5:   # um poligono com mais de 5 lados sera considerado um circulo
                print(contour['area'])
                area = contour['area']
                if area <= 1300:
                    totalMoney += 0.10
                elif 1300<area<=1600:
                    totalMoney += 0.50
                elif 1600<area<=1950:
                    totalMoney += 0.25
                elif 1950<area<=2200:
                    totalMoney += 1
                
        #print(f'R$: {totalMoney}')
    
    imgStack = cvzone.stackImages([img, imgPre, imgContours], 2, 1)
    cvzone.putTextRect(imgStack, f'R$: {totalMoney}', (50,50))
    cv2.imshow('webcam',imgStack)     # abre uma janela com a imagem
    if cv2.waitKey(1) & 0xFF == ord('q'): # condicao de saida do loop
        break
        

cap.release()
cv2.destroyAllWindows()