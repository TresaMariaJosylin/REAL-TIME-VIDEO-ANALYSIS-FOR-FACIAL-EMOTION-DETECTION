import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tkinter import *
from PIL import Image, ImageTk
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class UserNameWindow:
    def __init__(self, root, cap):
        self.root = root
        self.root.title("Emotion Detection App")

        # Open and display the background image
        bg_image = Image.open(r"C:\Users\hp\Desktop\python_project_gui\Emotion-detection-master\Emotion-detection-master\imgs\RR-Institute-of-Technology-Bangalore-1-4.png")  # Replace "background_image.png" with the path to your image
        bg_photo = ImageTk.PhotoImage(bg_image)
        bg_label = Label(root, image=bg_photo)
        bg_label.image = bg_photo
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        self.label = Label(root, text="Enter Your Name:",font=("Arial", 25))
        self.label.place(relx=0.5, rely=0.15, anchor=CENTER)

        self.entry = Entry(root,font=("Arial", 20), width=20)
        self.entry.place(relx=0.5, rely=0.25, anchor=CENTER)

        self.button = Button(root, text="Enter", command=self.on_enter,font=("Arial", 20))
        self.button.place(relx=0.5, rely=0.35, anchor=CENTER)

        self.cap = cap

    def on_enter(self):
        username = self.entry.get()
        self.root.destroy()
        root = Tk()
        app = FaceEmotionGUI(root, username, self.cap)
        root.mainloop()


class FaceEmotionGUI:
    def __init__(self, window, username, cap):
        self.window = window
        self.window.title("Face Emotion Detection")
        self.window.configure(bg="#FF0000")  # Set background color to red
        self.canvas = Canvas(window, width=800, height=600, bg="#FF0000")  # Set background color to red
        self.canvas.pack()

        self.is_detecting = False

        self.username = username

        self.start_button = Button(window, text="Start Detection", command=self.start_detection,bg="green")
        self.start_button.pack(side=LEFT)

        self.stop_button = Button(window, text="Stop Detection", command=self.stop_detection,bg="green")
        self.stop_button.pack(side=LEFT)

        self.quit_button = Button(window, text="Quit", command=self.quit_application,bg="green")
        self.quit_button.pack(side=LEFT)

        self.cap = cap

        self.update()

    def start_detection(self):
        self.is_detecting = True

    def stop_detection(self):
        self.is_detecting = False

    def quit_application(self):
        self.window.quit()

    def update(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.is_detecting:
                frame = self.detect_emotions(frame)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=NW)
        self.window.after(10, self.update)

    def detect_emotions(self, frame):
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Perform face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Check if any faces are detected
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]  # Extract the region of interest (ROI)
                roi_gray = cv2.resize(roi_gray, (48, 48))  # Resize the ROI to match the input size of the model
                roi_gray = np.expand_dims(roi_gray, axis=-1)  # Add a channel dimension
                roi_gray = np.expand_dims(roi_gray, axis=0)  # Add a batch dimension

                # Predict emotions using the model
                prediction = model.predict(roi_gray)
                maxindex = int(np.argmax(prediction))

                # Draw bounding box and label
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, emotion_dict[maxindex], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        return frame


# Define data generators
train_dir = 'data/train'
val_dir = 'data/test'

num_train = 28709
num_val = 7178
batch_size = 64
num_epoch = 50

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# Load the pre-trained weights
model.load_weights('model.h5')

# Dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Start the application by getting the username
cap = cv2.VideoCapture(0)
root = Tk()
app = UserNameWindow(root, cap)
root.mainloop()

# Release the capture object and destroy any OpenCV windows
cap.release()
cv2.destroyAllWindows()
