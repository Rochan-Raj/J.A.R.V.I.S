import pyttsx3
import speech_recognition as sr
import datetime
import wikipedia
import webbrowser
import os
import sys
import cv2
import pyautogui as p
import pyaudio


engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voices',voices[1].id)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()
def wishme(datetime):
    hour= int(datetime.datetime.now().hour)
    if hour>=0 and hour<12:
        speak("good morning")
    elif hour>=12 and hour<18:
        speak("good afternoon")
    else:
        speak("good evening")
    speak("hey Rochan ,jarvis here. please let me know how can i help you")

def takecommand():
    r=sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold=1
        audio=r.listen(source)
    try:
        print("recognizing...")
        query=r.recognize_google(audio,language="en-in")
        print('user said:',query)

    except Exception as e:
        print(e)
        speak("say that again please...")
        return "none"
    return query

#if __name__=='__main__':
def taskexecution():
        global speak
        p.press('esc')
        speak("verification successful") 
        wishme(datetime)
        speak("welcome back rochan sir")
        #if 1:
        while True:
            query = takecommand().lower()
            if 'wikipedia' in query:
                speak("%tb searching wikipedia.....please wait for a while")
                query=query.replace("wikipedia","")
                results=wikipedia.summary(query, sentences=5)
                print(results)
                speak=("%tb according to wikipedia")
                speak(results)
            elif "headlights" in query:
                webbrowser.open("youtube.com/watch?v=kyLuzKbgXAs")
                break
            elif "open youtube" in query:
                webbrowser.open("youtube.com")
            elif "open google" in query:
                webbrowser.open("google.com")
            elif "open command prompt" in query:
                os.system('%tbstart cmd')
            elif 'open stackoverflow' in query:
                webbrowser.open(" stackoverflow.com")
            elif 'open calender' in query:
                webbrowser.open('calendar.com')
            elif 'time' in query:
                strTime=datetime.datetime.now().strftime("%h:%m:%s")
                speak(f"%tb sir the time is{strTime}")
            elif 'no thanks' in query:
                speak("%tb thank u sir for using me. have a good day.")
                break
        sys.exit()

if __name__=="__main__":
    recognizer = cv2.face.LBPHFaceRecognizer_create() # Local Binary Patterns Histograms
    recognizer.read('C:/Users/HP/Desktop/vs code/Jarvis/trainer/trainer.yml')   #load trained model
    cascadePath = "C:/Users/HP/Desktop/vs code/Jarvis/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath) #initializing haar cascade for object detection approach

    font = cv2.FONT_HERSHEY_SIMPLEX #denotes the font type


    id = 2 #number of persons you want to Recognize


    names = ['','rochan']  #names, leave first empty bcz counter starts from 0


    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW) #cv2.CAP_DSHOW to remove warning
    cam.set(3, 640) # set video FrameWidht
    cam.set(4, 480) # set video FrameHeight

    # Define min window size to be recognized as a face
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)

    # flag = True

    while True:

        ret, img =cam.read() #read the frames using the above created object

        converted_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  #The function converts an input image from one color space to another

        faces = faceCascade.detectMultiScale( 
            converted_image,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
           )

        for(x,y,w,h) in faces:

            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2) #used to draw a rectangle on any image

            id, accuracy = recognizer.predict(converted_image[y:y+h,x:x+w]) #to predict on every single image

            # Check if accuracy is less them 100 ==> "0" is perfect match 
            if (accuracy < 100):
                id = names[id]
                accuracy = "  {0}%".format(round(100 - accuracy))
                taskexecution()
            else:
                id = "unknown"
                accuracy = "  {0}%".format(round(100 - accuracy))

            cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
            cv2.putText(img, str(accuracy), (x+5,y+h-5), font, 1, (255,255,0), 1)  

        cv2.imshow('camera',img) 

        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break

    # Do a bit of cleanup
    print("Thanks for using this program, have a good day.")
    cam.release()
    cv2.destroyAllWindows()
