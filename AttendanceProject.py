import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import time

path = 'Images_Attendance'
images = []
classNames = []
myList = os.listdir(path)
print(f'Loading images from: {myList}')

# Load images and extract class names
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    if curImg is not None:
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    else:
        print(f'Warning: Could not load image {cl}')
print(f'Loaded {len(classNames)} images: {classNames}')

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)
        if len(encodings) > 0:
            encodeList.append(encodings[0])
        else:
            print('Warning: No face found in one of the images')
    return encodeList

# Dictionary to track last attendance time for each person (cooldown mechanism)
lastAttendanceTime = {}
ATTENDANCE_COOLDOWN = 3  # seconds - prevents duplicate entries within this time

def updateAttendanceCSV(name, time_str, date_str):
    """
    Real-time function to update Attendance.csv immediately.
    This ensures the CSV file is updated in real-time as faces are recognized.
    """
    try:
        # Check if file exists and has header
        file_exists = os.path.exists('Attendance.csv')
        file_empty = False
        
        if file_exists:
            file_empty = os.path.getsize('Attendance.csv') == 0
        else:
            file_empty = True
        
        # Open file in append mode and write immediately
        with open('Attendance.csv', 'a', newline='', encoding='utf-8') as f:
            # Add header if file is new or empty
            if file_empty:
                f.write('Name,Time,Date')
            
            # Write the attendance entry
            f.write(f'\n{name},{time_str},{date_str}')
            # Flush immediately to ensure real-time update
            f.flush()
            os.fsync(f.fileno())  # Force write to disk
        
        return True
    except Exception as e:
        print(f'Error updating CSV for {name}: {e}')
        return False

def markAttendance(name):
    """
    Mark attendance for a single person with cooldown check.
    Returns True if attendance was marked, False if skipped due to cooldown.
    """
    current_time = time.time()
    
    # Check if person was marked recently (cooldown to prevent rapid duplicates)
    if name in lastAttendanceTime:
        time_diff = current_time - lastAttendanceTime[name]
        if time_diff < ATTENDANCE_COOLDOWN:
            return False  # Too soon, skip marking
    
    # Get current time and date
    time_now = datetime.now()
    tString = time_now.strftime('%H:%M:%S')
    dString = time_now.strftime('%d/%m/%Y')
    
    # Update CSV in real-time
    if updateAttendanceCSV(name, tString, dString):
        lastAttendanceTime[name] = current_time
        print(f'✓ Attendance marked: {name} at {tString} on {dString}')
        return True
    return False

def markMultipleAttendance(recognized_faces):
    """
    Mark attendance for multiple faces simultaneously.
    This function processes all recognized faces in a batch and marks them all at once.
    
    Args:
        recognized_faces: List of recognized face names
    
    Returns:
        List of names that were successfully marked
    """
    if not recognized_faces:
        return []
    
    # Get current time once for all faces (same timestamp for simultaneous detection)
    time_now = datetime.now()
    tString = time_now.strftime('%H:%M:%S')
    dString = time_now.strftime('%d/%m/%Y')
    current_time = time.time()
    
    # Filter faces that can be marked (respect cooldown)
    faces_to_mark = []
    for name in recognized_faces:
        if name not in lastAttendanceTime:
            # Never marked before, can mark
            faces_to_mark.append(name)
        else:
            # Check cooldown
            time_diff = current_time - lastAttendanceTime[name]
            if time_diff >= ATTENDANCE_COOLDOWN:
                faces_to_mark.append(name)
    
    if not faces_to_mark:
        return []
    
    # Mark all faces simultaneously in CSV
    try:
        file_exists = os.path.exists('Attendance.csv')
        file_empty = False
        
        if file_exists:
            file_empty = os.path.getsize('Attendance.csv') == 0
        else:
            file_empty = True
        
        # Write all faces at once
        with open('Attendance.csv', 'a', newline='', encoding='utf-8') as f:
            if file_empty:
                f.write('Name,Time,Date')
            
            # Write all attendance entries
            for name in faces_to_mark:
                f.write(f'\n{name},{tString},{dString}')
                lastAttendanceTime[name] = current_time
                print(f'✓ Attendance marked: {name} at {tString} on {dString}')
            
            # Flush immediately for real-time update
            f.flush()
            os.fsync(f.fileno())
        
        return faces_to_mark
    except Exception as e:
        print(f'Error marking multiple attendance: {e}')
        return []

# Generate encodings for known faces
encodeListKnown = findEncodings(images)
print(f'Encoding Complete. {len(encodeListKnown)} faces encoded.')

if len(encodeListKnown) == 0:
    print('Error: No faces found in any images. Please check your Images_Attendance folder.')
    exit()

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('Error: Could not open camera.')
    exit()

print('Camera opened. Press ENTER to exit.')
print(f'Looking for {len(classNames)} known faces: {classNames}')

frame_count = 0

while True:
    success, img = cap.read()
    if not success:
        print('Failed to read from camera.')
        break
    
    # Resize image for faster processing
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Find all faces in current frame
    facesCurFrame = face_recognition.face_locations(imgS, model='hog')  # Using HOG model for faster processing
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    # Initialize variables for face processing
    recognized_faces = []  # List to collect all recognized faces in this frame
    recognized_count = 0
    face_info = []  # Store face info for drawing

    # Display face count on screen
    face_count_text = f'Faces Detected: {len(facesCurFrame)}'
    cv2.putText(img, face_count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, face_count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

    # Process all detected faces and collect recognized faces for batch marking
    
    # First pass: Identify all faces
    for i, (encodeFace, faceLoc) in enumerate(zip(encodesCurFrame, facesCurFrame)):
        # Compare with known faces
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace, tolerance=0.6)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        
        # Find best match
        matchIndex = np.argmin(faceDis)
        
        # Scale face location back to original size
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
        
        # Check if match is good (face distance < 0.6 is typically a good match)
        if matches[matchIndex] and faceDis[matchIndex] < 0.6:
            name = classNames[matchIndex].upper()
            recognized_count += 1
            recognized_faces.append(name)
            
            # Store face info for drawing
            confidence = round((1 - faceDis[matchIndex]) * 100, 1)
            face_info.append({
                'type': 'recognized',
                'name': name,
                'location': (x1, y1, x2, y2),
                'confidence': confidence
            })
        else:
            # Unknown face
            best_distance = round(faceDis[matchIndex], 2) if len(faceDis) > 0 else 0
            face_info.append({
                'type': 'unknown',
                'name': 'Unknown',
                'location': (x1, y1, x2, y2),
                'distance': best_distance
            })
    
    # Mark attendance for all recognized faces simultaneously
    if recognized_faces:
        marked_faces = markMultipleAttendance(recognized_faces)
        # Update face_info to show which ones were marked
        for info in face_info:
            if info['type'] == 'recognized' and info['name'] in marked_faces:
                info['marked'] = True
            elif info['type'] == 'recognized':
                info['marked'] = False  # On cooldown
    
    # Second pass: Draw all faces with visual feedback
    for info in face_info:
        x1, y1, x2, y2 = info['location']
        
        if info['type'] == 'recognized':
            # Draw rectangle around recognized face (green)
            color = (0, 255, 0)  # Green
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), color, cv2.FILLED)
            cv2.putText(img, info['name'], (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display confidence
            cv2.putText(img, f'{info["confidence"]}%', (x1+6, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Show "MARKED" indicator if attendance was just marked
            if info.get('marked', False):
                cv2.putText(img, 'MARKED!', (x1+6, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            elif info.get('marked') == False:
                cv2.putText(img, 'On Cooldown', (x1+6, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        else:
            # Unknown face (red)
            color = (0, 0, 255)  # Red
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), color, cv2.FILLED)
            cv2.putText(img, 'Unknown', (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, f'Dist: {info["distance"]}', (x1+6, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Display recognized count and status
    if len(facesCurFrame) > 0:
        recognized_text = f'Recognized: {recognized_count}/{len(facesCurFrame)}'
        cv2.putText(img, recognized_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, recognized_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        # Show CSV update status
        if recognized_faces:
            marked_count = len([f for f in face_info if f.get('marked', False)])
            if marked_count > 0:
                status_text = f'CSV Updated: {marked_count} marked'
                cv2.putText(img, status_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(img, status_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Display frame
    cv2.imshow('Face Recognition Attendance System - Press ENTER to exit', img)
    
    # Press ENTER (key code 13) to exit
    if cv2.waitKey(1) & 0xFF == 13:
        break
    
    frame_count += 1

# Cleanup
cap.release()
cv2.destroyAllWindows()
print('Camera released. Goodbye!')


