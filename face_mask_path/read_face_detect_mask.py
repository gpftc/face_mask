import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import image


#cascade path file for eyes and frontal face
import os
casc_path_face = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
casc_path_eyes = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_eye_tree_eyeglasses.xml"
#%matplotlib inline  # if you are running this code in Jupyter notebook

# reads image 'opencv-logo.png' as grayscale
path_image = 'face_with_mask.jpg'
img = cv2.imread(path_image, 1) 
#im=cv2.flip(img,1,1) 
mymodel=load_model('mymodel.h5')
#im=cv2.flip(im,1,1) 



resized=cv2.resize(img,(150,150))
normalized=resized/255.0
reshaped=np.reshape(normalized,(1,150,150,3))
reshaped = np.vstack([reshaped])
pred=mymodel.predict(reshaped)[0][0]
pred=float(pred)
	    
print(pred)
#import cv2  
cv2.putText(img,f'MASCARA {round(pred,4)*100}% ',(20,180),cv2.FONT_HERSHEY_SIMPLEX,2.5,(50,250,50),6)


    	 	#cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 4) 
    	
 

# Load the cascade  
face_cascade = cv2.CascadeClassifier(casc_path_face)  
eye_cascade = cv2.CascadeClassifier(casc_path_eyes)
# Read the input image  
#img = cv2.imread('test.jpg')  
  
# Convert into grayscale  
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
  
# Detect faces  
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Draw rectangle around the faces  
for (x, y, w, h) in faces:
	    face_img = img[y:y+h, x:x+w]
	    
	    resized=cv2.resize(face_img,(150,150))
	    normalized=resized/255.0
	    reshaped=np.reshape(normalized,(1,150,150,3))
	    reshaped = np.vstack([reshaped])
	    pred=mymodel.predict(reshaped)[0][0]
	    pred=float(pred)
	    
	    if pred>=.9:
	    	cv2.rectangle(img, (x, y), (x + w, y + h), (50,250,50), 4) 
	    	cv2.putText(img,f'MASCARA {round(pred,2)*100}% ',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,3,(50,250,50),6)
	    else:
    	 	cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 4) 
    	 	cv2.putText(img,f'SEM MASCARA  {round(pred,2)*100}%',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,3,(255,0,0),6)
 

    	 
# Display the output  
#cv2.imshow('image', img)  
#cv2.waitKey()  



plt.title(f'{path_image}')

plt.imshow(img,cmap='gray')
plt.savefig(f'tested_{path_image}',dpi=900)
plt.show()