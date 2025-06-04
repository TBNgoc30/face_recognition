Face Recognition Attendance System
Overview
This project is a face recognition-based attendance system built using Python, OpenCV, PyTorch, and Tkinter. It allows users to:

Add new faces to the database by capturing images from a webcam or uploading images.
Perform real-time face recognition to mark attendance.
View and export attendance statistics in CSV and PDF formats.

The system uses the MTCNN model for face detection and InceptionResnetV1 (pretrained on VGGFace2) for generating face embeddings. Attendance data is stored in a CSV file, and a GUI provides an intuitive interface for managing the system.
Features

Add Member: Add a new person to the database by capturing a photo via webcam or uploading an image, with name and class information.
Attendance Tracking: Automatically detects and recognizes faces from a webcam feed, logging attendance with timestamps.
Statistics: View attendance records filtered by class or date, with options to export to PDF.
Responsive GUI: Built with Tkinter, featuring a welcome screen, add member interface, attendance tracking, and statistics views.
Face Recognition: Uses MTCNN for face detection and InceptionResnetV1 for face embedding and recognition.
CSV Storage: Attendance data is stored in Attendance.csv for persistence.
PDF Export: Generate PDF reports of attendance records, customizable by class and date.

Requirements

Python 3.8+
Libraries:
opencv-python
torch
numpy
pandas
Pillow
facenet-pytorch
beepy
fpdf


A webcam for real-time face detection and recognition.
A face_detection.jpg image for the welcome screen background (optional).

Install dependencies using:
pip install opencv-python torch numpy pandas Pillow facenet-pytorch beepy fpdf

Installation

Clone or download the repository:
git clone <repository-url>
cd <repository-directory>


Ensure a webcam is connected to your system.

Place a face_detection.jpg image in the project directory for the welcome screen (optional; if missing, an error message will be displayed).

Create an image directory in the project root to store face images (the system will create it automatically if it doesn't exist).


Usage

Run the main script:
python main.py


The GUI will open with the following options:

Welcome Screen: Displays the face_detection.jpg image (if available).
Add Member: Enter a name and class, then capture a face via webcam or upload an image.
Attendance: View the webcam feed with real-time face recognition and attendance logging.
Statistics: View attendance records, filter by class or date, and export to PDF.


Use the menu bar to navigate between features or exit the application.


Directory Structure

main.py: The main script containing the application logic.
image/: Directory to store face images, organized in subdirectories named <name>_<class> (e.g., BILL_GATE_DHCNTT/).
Attendance.csv: Stores attendance records with columns Name, Class, Time, and Date.
face_detection.jpg: Optional image for the welcome screen background.

How It Works

Face Database: Images are stored in the image/ directory, with subdirectories named <name>_<class>. Each subdirectory contains one or more .jpg images for a person. The system uses the first detected face embedding per person.
Face Recognition: MTCNN detects faces in the webcam feed or uploaded images. InceptionResnetV1 generates embeddings, which are compared to known embeddings using Euclidean distance (threshold: 0.8).
Attendance Logging: Recognized faces are logged in Attendance.csv with name, class, time, and date. Duplicate entries for the same person on the same day are prevented.
GUI: Built with Tkinter, featuring a responsive canvas for webcam display, a treeview for attendance records, and comboboxes for filtering statistics.
PDF Export: Uses fpdf to generate reports, with options to filter by class, date, or both.

Notes

The system requires a GPU for optimal performance but falls back to CPU if CUDA is unavailable.
Ensure the webcam is accessible; otherwise, the application will exit with an error.
Face images should be clear and well-lit for accurate detection and recognition.
The Attendance.csv file is created automatically if it doesn't exist or is empty.
The GUI is designed to be responsive, with canvases resizing to maintain aspect ratios (4:3 for Add Member, 16:9 for Attendance).

Troubleshooting

Webcam Issues: Ensure the webcam is connected and not in use by another application. Check the device index in cv2.VideoCapture(0) if necessary.
Face Detection Fails: Ensure good lighting and clear visibility of faces. Adjust the MTCNN margin or image size if needed.
Image Loading Errors: Verify that uploaded images are valid and in supported formats (.jpg, .jpeg, .png).
PDF Export Fails: Ensure you have write permissions in the selected directory.

License
This project is licensed under the MIT License. See the LICENSE file for details (if applicable).
Acknowledgments

facenet-pytorch for face detection and recognition models.
OpenCV for image processing and webcam capture.
Tkinter for the GUI framework.
FPDF for PDF generation.

