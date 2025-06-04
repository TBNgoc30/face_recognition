# Attendance System
## Overview
This project is a face recognition-based attendance system built using Python. It leverages the facenet-pytorch library for face detection and recognition, OpenCV for webcam capture and image processing, and Tkinter for the graphical user interface (GUI). The system allows users to add new faces, mark attendance via webcam, view attendance statistics, and export reports to PDF.
## Features

- Add New Face: Capture a face via webcam or load an image to register a person with a name and class.
- Attendance Marking: Detects and recognizes faces in real-time from the webcam, logs attendance to a CSV file, and displays it in a table.
- Statistics: View attendance records filtered by class and/or date.
- PDF Export: Generate PDF reports of attendance data based on selected filters.
- Responsive GUI: Built with Tkinter, featuring a welcome screen, navigation menu, and dynamic resizing of webcam feed.

## Requirements

- Python 3.6 or higher
- Required libraries (install via pip):
+ opencv-python (cv2)
+ torch
+ numpy
+ pandas
+ pillow (PIL)
+ facenet-pytorch
+ beepy
+ fpdf


- A webcam for real-time face detection
- An image file named face_detection.jpg for the welcome screen background (optional)

## Installation

- Clone the Repository:
```git clone <repository-url>```
```cd <repository-folder>```


- Install Dependencies:
```pip install opencv-python torch numpy pandas pillow facenet-pytorch beepy fpdf```


- Prepare the Image Directory:
-  Create an image folder in the project directory.
-  Add subfolders for each person in the format NAME_CLASS (e.g., BILL_GATE_DHCNTT).
-  Place .jpg, .jpeg, or .png images of the person in their respective subfolder.


- Optional: Place a face_detection.jpg file in the project directory for the welcome screen background.

## Usage

Run the Application:
```python main.py```


- GUI Navigation:
-  Welcome Screen: Displays a background image (face_detection.jpg) and navigation options.
-  Add New Face:
+ Enter a name and class.
+ Click "Capture Face" to use the webcam or "Load Image" to upload an image.
+ The face is saved in the image/NAME_CLASS folder.


- Attendance:
+ Real-time face recognition via webcam.
+ Recognized faces are logged to Attendance.csv with name, class, time, and date.
+ A beep sound confirms attendance.


- Statistics:
+ Filter attendance records by class and/or date using dropdown menus.
+ View results in a table.
+ Export filtered data to a PDF file.




- File Menu:
+ "Empty CSV": Clears the Attendance.csv file and the attendance table.
+ "Exit": Closes the application.



## How It Works

- Face Recognition:
+ Uses MTCNN from facenet-pytorch for face detection and InceptionResnetV1 for generating face embeddings.
+ Known faces are loaded from the image directory, where each subfolder (NAME_CLASS) contains images of a person.
+ During attendance, the system compares webcam face embeddings to known embeddings, identifying matches within a threshold + ).
+ Unknown faces are labeled and displayed with a red bounding box; recognized faces get a green box.


- Data Storage:
+ Attendance records are stored in Attendance.csv with columns: Name, Class, Time, Date.


- GUI:
+ Built with Tkinter, featuring a menu bar, canvas for webcam feed, and tables for attendance and statistics.
+ The webcam feed resizes dynamically to fit the canvas while maintaining aspect ratio.



## Notes

- Webcam: Ensure a webcam is connected and accessible. The program will exit with an error if it cannot open the webcam.
- Image Loading:
+ Only .jpg, .jpeg, and .png files are supported for adding new faces.
+ Subfolders in the image directory should be named in the format NAME_CLASS (e.g., BILL_GATE_DHCNTT).


- Error Handling:
+ Displays error messages for issues like missing faces, invalid images, or webcam failures.


- PDF Export:
+ Exports filtered attendance data to a PDF file with a customizable filename.
+ Includes headers, footers, and total record count.



## Limitations

- Requires a GPU for optimal performance (falls back to CPU if unavailable).
- Face recognition accuracy depends on image quality, lighting, and the threshold (0.8).
- Only one embedding per NAME_CLASS is stored; additional images in the same folder are ignored.
- No multi-face attendance marking in a single frame is prioritized (processes all detected faces but logs sequentially).

## Troubleshooting

- Webcam Issues: Check if the webcam is connected and not in use by another application.
- Image Loading Fails: Verify the file path and ensure the image is a valid .jpg, .jpeg, or .png file.
- No Face Detected: Ensure proper lighting and that the face is clearly visible.
- CSV Empty: Use the "Empty CSV" option to reset the Attendance.csv file if corrupted.

## License
This project is for educational and personal use. No specific license is provided.
