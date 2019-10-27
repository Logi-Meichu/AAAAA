import numpy as np
import cv2

cap = cv2.VideoCapture(1)
cap.set(3,500)
cap.set(4,500)
ret1, frame = cap.read()
shape = frame.shape

x_ = int(shape[1] / 5)
y_ = int(shape[0] / 5)
x_w = int(shape[1] * 4/5)
y_h = int(shape[0] * 4/5)
markColor=(0,255,0)

while(True):
    ret0 ,frame0 = cap.read()
    ret1, frame1 = cap.read()
    ret2, frame2 = cap.read()
    gray1 = cv2.cvtColor(frame1 , cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2 , cv2.COLOR_BGR2GRAY)
    blur1 = cv2.GaussianBlur(gray1,(7,7),0)
    blur2 = cv2.GaussianBlur(gray2,(5,5),0)

    d = cv2.absdiff(blur1, blur2)
    ret, th = cv2.threshold( d, 10, 255, cv2.THRESH_BINARY )
    dilated=cv2.dilate(th, None, iterations=1)


    im2,contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    areas = [cv2.contourArea(c) for c in contours]
    try:
        max_index = np.argmax(areas)
        cnt=contours[max_index]
    except:
        pass
    x,y,w,h = cv2.boundingRect(cnt)

    frame3 = cv2.drawContours(frame1, cnt, -1, markColor, 2)
    crop = frame3[y_:y_h,x_:x_w]
    frame2[y_:y_h,x_:x_w] = crop[: ,:]
    cv2.rectangle(frame2 ,(x_ ,y_),(x_w ,y_h) ,(0,0,255),3)

    #cv2.imshow('orgin', frame0)
    #cv2.imshow('after', frame2)
    
    frame4 = cv2.hconcat([ frame0, frame2 ])
    cv2.imshow('demo', frame4)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()