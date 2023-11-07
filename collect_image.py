import cv2
import os
import time

# Set the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Set the directory for the dataset
dataset_path = 'dataset/'

# Set the name of the person
person_name = input("Enter the name of the person: ")
person_path = os.path.join(dataset_path, person_name)

# Create the directory if it doesn't exist
if not os.path.exists(person_path):
    os.makedirs(person_path)

# Initialize the camera
cap = cv2.VideoCapture(0)

# Set the number of images to capture
num_images = 200
image_count = 0

while image_count < num_images:
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Save the captured face
        face_img = gray[y:y+h, x:x+w]

        # Preprocess the image
        resized_face = cv2.resize(face_img, (224, 224))  # Adjust the size based on your model's input size
        normalized_face = resized_face / 255.0

        # Save the preprocessed image
        img_name = f"img{image_count+1}.jpg"
        img_path = os.path.join(person_path, img_name)
        cv2.imwrite(img_path, (normalized_face * 255).astype("uint8"))

        image_count += 1

        # Display the captured image for a short time
        cv2.imshow('Captured Face', face_img)
        cv2.waitKey(500)  # Adjust the time in milliseconds (1000 = 1 seconds)

    # Display the frame
    cv2.imshow('Capture Face', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera
cap.release()

# Destroy all OpenCV windows
cv2.destroyAllWindows()