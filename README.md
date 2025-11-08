# Face-Recognition-Attendance-System

Face Recognition Attendance System

A real-time face recognition-based attendance system developed using Python, OpenCV, and face_recognition.
This project automatically detects and recognizes faces through a webcam and records attendance in a CSV file — eliminating manual entries and ensuring instant, reliable results.

📋 Overview

This system captures real-time video from your webcam, detects and identifies registered faces, and logs their name, time, and date into an Attendance.csv file.

It is ideal for automating attendance in:

Educational institutions

Offices

Workshops and seminars

Secure access systems

⚙️ Key Features

✅ Real-time Face Detection & Recognition – Detects faces instantly from webcam feed
✅ Automatic Attendance Marking – Saves name, time, and date in Attendance.csv
✅ Duplicate Prevention – Uses a cooldown timer to avoid repeated entries
✅ Multi-face Detection – Recognizes and marks multiple people simultaneously
✅ Instant File Updates – Attendance data is saved to disk in real time
✅ Visual Feedback – Displays names, confidence level, and marking status on screen

🗂️ Project Structure
Face-recognition-Attendance-System-Project-main/
│
├── main.py                  # Entry point to start the application
├── AttendanceProject.py     # Core logic (face recognition + CSV updates)
├── Attendance.csv           # Attendance record (Name, Time, Date)
├── Images_Attendance/       # Folder containing known face images
│   ├── person1.jpg
│   ├── person2.jpg
│   └── ...
└── README.md                # Project documentation

🧩 Technologies Used
Technology	Purpose
Python 3	Core programming language
OpenCV	Webcam access & image processing
face_recognition	Face encoding & comparison
NumPy	Numerical operations
CSV module	Attendance file management
🧱 Installation Guide
1️⃣ Clone or Download the Repository
git clone https://github.com/your-username/Face-Recognition-Attendance-System.git
cd Face-Recognition-Attendance-System

2️⃣ Install Dependencies

Make sure Python 3.8+ is installed, then run:

pip install opencv-python face_recognition numpy


If you face issues installing face_recognition, install these first:

pip install cmake dlib
pip install face_recognition

🖼️ Add Known Faces

Place clear, front-facing photos of individuals inside the folder:

Images_Attendance/


Example:

Images_Attendance/
├── mayank.jpg
├── elon.jpg
└── modi.jpg


The filename (without extension) becomes the person’s name in the attendance sheet.
Example: mayank.jpg → recognized as MAYANK

▶️ How to Run

Run the main script to start the system:

python main.py

Controls:

The webcam will open automatically.

Recognized faces appear in green boxes with names.

Unknown faces appear in red boxes.
