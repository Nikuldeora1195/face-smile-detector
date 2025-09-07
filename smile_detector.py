import cv2



# Use built-in haarcascade path
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")



# Open the camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    faces = face_cascade.detectMultiScale(gray, 1.1 ,5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(150,55,100),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1 ,5)
        if len(eyes) > 0:
            cv2.putText(frame, 'Eyes Detected', (x-10, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36,255,12), 2)
        for (ex, ey, ew, eh) in eyes:
           cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 40, 100), 2)
           
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.4 ,25)
        if len(smiles)>0:
            cv2.putText(frame, 'Smiling', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,155,222), 2)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (50, 155, 55), 2)
    cv2.imshow('Face Detector',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
