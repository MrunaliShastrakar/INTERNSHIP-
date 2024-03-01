import cv2
import numpy as np
from scipy.signal import find_peaks
import mediapipe as mp

def calculate_respiratory_rate_and_var(video_path, actual_respiratory_rate):
    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Variables for processing
    frame_count = 0
    prev_gray = None
    respiratory_rate = []

    # Initialize mediapipe face detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection()

    # Loop through the frames
    while True:
        # Read the current frame
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (15, 15), 0)

        # Convert the frame to RGB format for mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame using mediapipe
        results = face_detection.process(rgb_frame)

        # Check if any faces are detected
        if results.detections:
            for detection in results.detections:
                # Get the bounding box coordinates of the face
                x, y, w, h = int(detection.location_data.relative_bounding_box.xmin * frame.shape[1]), \
                             int(detection.location_data.relative_bounding_box.ymin * frame.shape[0]), \
                             int(detection.location_data.relative_bounding_box.width * frame.shape[1]), \
                             int(detection.location_data.relative_bounding_box.height * frame.shape[0])

                # Draw a rectangle around the face region
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Extract the region of interest (ROI)
                roi = gray[y:y + h, x:x + w]

                # Calculate the mean intensity of the ROI
                intensity = np.mean(roi)

                # Append the intensity value to the respiratory rate list
                respiratory_rate.append(intensity)

        # Display the frame with bounding box
        cv2.imshow('Respiratory Rate Detection', frame)

        # Wait for the 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Update the previous frame
        prev_gray = gray
        frame_count += 1

    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

    # Calculate the respiratory rate using the intensity peaks
    peaks, _ = find_peaks(respiratory_rate, distance=30)
    respiratory_rate = 1 / (np.mean(np.diff(peaks)) / frame_count)

    # Calculate the respiratory variability
    variability = np.std(np.diff(peaks))

    # Calculate the RMSE
    rmse = np.sqrt(np.mean((respiratory_rate - actual_respiratory_rate)**2))

    return respiratory_rate, variability, rmse

video_path = 'H56121B.mp4'
actual_respiratory_rate = 23 # Replace with the actual respiratory rate
respiratory_rate, variability, rmse = calculate_respiratory_rate_and_var(video_path, actual_respiratory_rate)

print('Respiratory Rate: {:.2f} breaths per minute'.format(respiratory_rate))
print('Respiratory Variability: {:.2f}'.format(variability))
print('RMSE: {:.2f}'.format(rmse))
