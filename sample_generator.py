import cv2

cam=cv2.VideoCapture(0, cv2.CAP_DSHOW)

cam.set(3,640)
cam.set(4,480)

detector=cv2.CascadeClassifier('C:/Users/HP/Desktop/vs code/Jarvis/haarcascade_frontalface_default.xml')
face_id=input("enter a numeric user ID here: ")
print("taking samples, look at camera:")
count=0

while True:
    ret, img =cam.read()
    converted_image=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces= detector.detectMultiScale(converted_image,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        count+=1

        cv2.imwrite("C:/Users/HP/Desktop/vs code/ABC/pic_samples/face." + str(face_id) + '.' + str(count) + ".jpg", converted_image[y:y+h,x:x+w])
        # To capture & Save images into the datasets folder

        cv2.imshow('image', img) #Used to display an image in a window

    k = cv2.waitKey(100) & 0xff # Waits for a pressed key
    if k == 27: # Press 'ESC' to stop
        break
    elif count >= 10: # Take 50 sample (More sample --> More accuracy)
         break

print("Samples taken now closing the program....")
cam.release()
cv2.destroyAllWindows()
