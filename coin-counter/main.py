import cv2 as cv2
import cvzone
import numpy as np
from cvzone.ColorModule import ColorFinder
#print (cv2.__version__)

myColorFinder = ColorFinder(False) # habilita o trackbar para identificacao da cor manualmente
# Custom Color
# hsvVals = {'hmin': 58, 'smin': 13, 'vmin': 116, 'hmax': 164, 'smax': 34, 'vmax': 174}
hsvVals = {'hmin': 16, 'smin': 45, 'vmin': 89, 'hmax': 24, 'smax': 151, 'vmax': 178}
# filtra a imagem inicial para algo que possa ser processado e classificado
def preProcessing(imgOriginal):
    #---BLUR---
    # o blur ajuda a diminuir os ruidos
    imgPre = cv2.GaussianBlur(imgOriginal, (5,5), 3)

    #---CANNY---
    thresh1 = cv2.getTrackbarPos("Threshould1", "Settings") # seletores para os threshoulds
    thresh2 = cv2.getTrackbarPos("Threshould2", "Settings")
    imgPre = cv2.Canny(imgPre, thresh1, thresh2) # os threshoulds vao de 0-255; identifica as bordas e as desenha em branco

    #---DILATE & MORPH---
    kernel = np.ones((3, 3), np.uint8)  # matriz de 1's
    imgPre = cv2.dilate(imgPre, kernel, iterations=1)   #dilata as linha brancas de contorno
    imgPre = cv2.morphologyEx(imgPre, cv2.MORPH_CLOSE, kernel)  # fecha os poligonos com gaps
    return imgPre  # imagem apos aplicacao de todos os filtros

# uma funcao de callback que nao faz nada
def empty(value):
    pass

cap = cv2.VideoCapture(0)    # recebe imagem da webcam
cap.set(3, 640)              # estabelece a resolucao de captura
cap.set(4, 480)

# cria tab com os trackbars
cv2.namedWindow("Settings")
cv2.resizeWindow("Settings", 640, 240)
cv2.createTrackbar("Threshould1", "Settings", 240, 255, empty)
cv2.createTrackbar("Threshould2", "Settings", 150, 255, empty)

# loop main()
while True:
    sucess, img = cap.read()
    if not sucess:  # se houve erro na captura da imagem o processo e interrompido
        break
    
    imgPre = preProcessing(img)
    imgContours, conFound = cvzone.findContours(img, imgPre, minArea=800, filter=8)
    # marca a imagem original com os identificadores provenientes de imgPre, gerando uma nova imagem

    totalMoney = 0  # variavel para contagem das moedas
    if conFound:
        for count, contour in enumerate(conFound):  # retorna, respectivamente, a posicao e cada elemento de conFound
            peri = cv2.arcLength(contour['cnt'], True)
            approx = cv2.approxPolyDP(contour['cnt'], 0.02 * peri, True)

            if len(approx)>5:   # um poligono com mais de 5 lados sera considerado um circulo aqui
                # print(contour['area'])
                area = contour['area']  # obtem o valor da area do poligono
                
                # Recorta cada moeda em imagens cortadas individuais
                x, y, w, h = contour['bbox']
                imgCrop = img[y:y+h, x:x+w]
                cv2.imshow(str(count), imgCrop)

                imgColor, mask = myColorFinder.update(img, hsvVals) # cria uma mascara com base num filtro de cor
                whitePixelCount = cv2.countNonZero(mask) # conta todos os valores nao nulos da mascara
                print(whitePixelCount)

                # classifica as moedas da menor para a maior, conforme o tamanho
                # (valores obtidos empiricamente para o setup construído)
                if area <= 1300:
                    totalMoney += 0.10
                elif 1300<area<=1600:
                    totalMoney += 0.50
                elif 1600<area<=1950:
                    totalMoney += 0.25
                elif 1950<area<=2200:
                    totalMoney += 1
                
        #print(f'R$: {totalMoney}')
    
    imgStack = cvzone.stackImages([img, imgPre, imgContours], 2, 1) # apresenta varias imagens numa mesma janela
    cvzone.putTextRect(imgStack, f'R$: {totalMoney}', (50,50))  # escreve o valor em R$ calculado
    cv2.imshow('webcam',imgStack)     # abre uma janela com as imagens selecionadas 
    cv2.imshow('colors',imgColor)
    if cv2.waitKey(1) & 0xFF == ord('q'): # condicao de saida do loop
        break
        

cap.release()
cv2.destroyAllWindows()