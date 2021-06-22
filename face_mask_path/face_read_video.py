import cv2
import tkinter as tk
from PIL import ImageTk, Image

root = tk.Tk()
root.rowconfigure(0, weight=1)
root.columnconfigure(0, weight=1)
lmain = tk.Label(root)
lmain.grid()

# Initialize the camera with index 0
cap = cv2.VideoCapture(0)
# Check that we have camera access
# This check is not included in all further examples
if not cap.isOpened():
    lmain.config(text="Unable to open camera: please grant appropriate permission in Pydroid permissions plugin and relaunch.\nIf this doesn't work, ensure that your device supports Camera NDK API: it is required that your device supports non-legacy Camera2 API.", wraplength=lmain.winfo_screenwidth())
    root.mainloop()
else:
    # You can set the desired resolution here
    cap = cv2.VideoCapture('face_video.mp4') 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')  
  
def refresh():
    global imgtk
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
  
    # Detect the faces  
    faces = face_cascade.detectMultiScale(gray, 1.1, 4) 
    for (x, y, w, h) in faces:  
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 6)  
    if not ret:
        # Error capturing frame, try next time
        lmain.after(0, refresh)
        return
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
  
    # Detect the faces  
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)  
  
    # Draw the rectangle around each face  
    
  
   
    w = lmain.winfo_screenwidth()
    h = lmain.winfo_screenheight()
    cw = cv2image.shape[0]
    ch = cv2image.shape[1]
    # In portrait, image is rotated
    cw, ch = ch, cw
    if (w > h) != (cw > ch):
        # In landscape, we have to rotate it
        cw, ch = ch, cw
        # Note that image can be upside-down, then use clockwise rotation
        cv2image = cv2.rotate(cv2image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # Keep aspect ratio
    w = min(cw * h / ch, w)
    h = min(ch * w / cw, h)
    w, h = int(w), int(h)
    # Resize to fill the whole screen
    cv2image = cv2.resize(cv2image, (w, h), interpolation=cv2.INTER_LINEAR)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.configure(image=imgtk)
    lmain.update()
    lmain.after(0, refresh)


refresh()
root.mainloop()
