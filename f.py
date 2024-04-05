from textblob import  TextBlob

import cv2
import numpy as np
import tensorflow as tf
import os
#from playsound import playsound
# Load your trained model
model = tf.keras.models.load_model('CNN.model')
DATADIR = 'dataset'
from gtts import gTTS
import os
import pyttsx3
import os
import time
engine = pyttsx3.init()
engine.setProperty("rate", 150)

CATEGORIES = os.listdir(DATADIR)  # Get all subfolder names as categories
IMG_SIZE = 50
ui=0
# Open a webcam feed
cap = cv2.VideoCapture(0) 

# Define the dimensions of the box
box_size = 300

# Create a window for displaying the hand image
cv2.namedWindow('Hand Image')
det=['hi how are you','i dont know','what is your name','who are you','what is this','where are you','how are you','i am hungry','i am ironman','i love you','i hate you','i am sick','i am sleeping','i am thirsty','i am in home','thankyou','hi how are you','i dont know','what is your name','who are you','what is this','where are you','how are you','i am hungry','i am ironman','i love you','i hate you','i am sick','i am sleeping','i am thirsty','i am in home','thankyou']

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Extract the box region for hand
    hand_box = gray_frame[0:box_size, 0:box_size]

    resized_frame = cv2.resize(hand_box, (IMG_SIZE, IMG_SIZE))
    prepared_image = np.array(resized_frame).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    import soundfile as sf
    prediction = model.predict(prepared_image)
    predicted_class = np.argmax(prediction)
    predicted_category = CATEGORIES[predicted_class]
    
    if "unknown" not in predicted_category:
    
        m=TextBlob(det[np.argmax(prediction)])
        print(m)
        import win32com.client
        speaker = win32com.client.Dispatch("SAPI.SpVoice")
        speaker.Speak(m)
        #output_file = str(ui) + 'output.mp3'
        #sf.write(output_file, speaker.AudioOutputStream.Buffer, speaker.AudioOutputStream.Format.FormatTag)



    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, predicted_category, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Draw the box on the main frame
    cv2.rectangle(frame, (0, 0), (box_size, box_size), (0, 255, 0), 2)

    # Show the hand image in a separate window
    cv2.imshow('Hand Image', resized_frame)

    cv2.imshow('Hand Sign Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close the 'Hand Image' window
cv2.destroyWindow('Hand Image')

cap.release()
cv2.destroyAllWindows()
