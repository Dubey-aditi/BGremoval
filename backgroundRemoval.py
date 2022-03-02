import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os 

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(cv2.CAP_PROP_FPS, 60)
segmentor = SelfiSegmentation()
fpsReader = cvzone.FPS()
#imgBg = cv2.imread("backgroundRemoval/bg6.jpg")

listImg = os.listdir("backgroundRemoval")
imgList = []
for imgPath in listImg:
    img = cv2.imread(f'backgroundRemoval/{imgPath}')
    imgList.append(img)
print(len(imgList))  

indexImg = 0

while True:
    success, img = cap.read()
    imgOut = segmentor.removeBG(img, imgList[indexImg], threshold=0.9)

    imgStacked = cvzone.stackImages([img, imgOut], 2, 1)
    _, imgStacked = fpsReader.update(imgStacked, color=(0, 255, 0))

    cv2.imshow("Image", imgStacked)
    #cv2.imshow("ImageOut", imgOut)

    key = cv2.waitKey(1)
    if key == ord('j'):
        indexImg -=1
    elif key == ord('l'):
        indexImg +=1
    elif key == ord('q'):
        break

