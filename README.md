# Workshop 2: Object Detection Using Web Camera

## Aim

To develop a Python program using the YOLO model and OpenCV to detect objects in real time using a webcam, display the detected objects with bounding boxes and labels, and show the output in a Jupyter Notebook.

## Software Used

- Python
- Jupyter Notebook
- OpenCV (`cv2`)
- Ultralytics YOLO
- Matplotlib
- IPython

## Algorithm

### Step 1: Import Libraries

Import the required libraries: YOLO from Ultralytics, OpenCV, Matplotlib, and `clear_output` from IPython.

### Step 2: Load the YOLO Model

Load the YOLOv8 Nano model using `YOLO("yolov8n.pt")`.

### Step 3: Open the Webcam

Access the webcam using `cv2.VideoCapture(0)`.

### Step 4: Check Webcam Availability

- Check whether the webcam is opened successfully using `cap.isOpened()`.
- Display an error message if the webcam cannot be opened.

### Step 5: Capture Video Frames

- Read frames continuously from the webcam using `cap.read()`.
- Check whether each frame is captured successfully.
- Stop the loop if a frame cannot be captured.

### Step 6: Perform Object Detection

- Pass each captured frame to the YOLO model.
- Set the confidence threshold to `0.60`.
- Disable verbose output.

### Step 7: Draw Bounding Boxes and Labels

Use `results[0].plot()` to draw bounding boxes and object labels on the detected frame.

### Step 8: Display the Detection Result

- Convert the annotated frame from BGR to RGB using `cv2.cvtColor()`.
- Clear the previous notebook output using `clear_output(wait=True)`.
- Display the annotated frame using Matplotlib.
- Set the title as "Real-Time Object Detection".

### Step 9: Release the Webcam

Release the webcam using `cap.release()` and close OpenCV windows using `cv2.destroyAllWindows()`.

## Program

**Developed By:** PANDEESWARAN N

**Register Number:** 212224230191

**Program File:** `Workshop 2.ipynb`

### Step 1: Import Libraries

```python
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from IPython.display import clear_output
```

### Step 2: Load the YOLO Model

```python
# Load a more accurate YOLO model
model = YOLO("yolov8n.pt")
```

### Step 3: Open the Webcam

```python
# Open webcam
cap = cv2.VideoCapture(0)
```

### Step 4: Check the Webcam and Perform Object Detection

```python
# Check if camera opened
if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame.")
            break

        # Perform object detection
        results = model(
            frame,
            conf=0.60,       
            verbose=False
        )

        # Draw bounding boxes and labels
        annotated_frame = results[0].plot()

        # Convert BGR to RGB for Matplotlib
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Display in Jupyter
        clear_output(wait=True)
        plt.figure(figsize=(10, 8))
        plt.imshow(annotated_frame)
        plt.title("Real-Time Object Detection")
        plt.axis("off")
        plt.show()

# Release webcam
cap.release()
cv2.destroyAllWindows()
```

## Output

<img width="1492" height="893" alt="image" src="https://github.com/user-attachments/assets/aa7f7919-50f2-4fe5-961c-8f839f6fa0e8" />


## Result

Thus, object detection using the YOLOv8 Nano model and OpenCV webcam input was implemented in Python. The program captures webcam frames, detects objects, draws bounding boxes and labels, and displays the annotated frames in the Jupyter Notebook.
