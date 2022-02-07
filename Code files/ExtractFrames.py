import cv2

# Opens the Video file
cap= cv2.VideoCapture('C:/Users/eliet/OneDrive/Desktop/WhatsApp Video 2021-04-19 at 13.57.11 (2).mp4')
i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    cv2.imwrite('kang'+str(i)+'.jpg',frame)
    i+=1

cap.release()
cv2.destroyAllWindows()