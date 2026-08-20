#  Lane Detection
##  Developed By

* **Name:** P.Bhavankumar
* **Register No:** 212225240026

##  Aim

To implement a basic lane detection pipeline using OpenCV by completing missing code segments at specified locations.

---

## Learning Objective

* Understand each stage of image processing
* Learn how to build a complete computer vision pipeline
* Practice writing code in guided sections

**Important Instruction:**
👉 Write code **ONLY in places marked as `# Your Code Here`**
👉 Do NOT modify any other part of the code

---

##  Software Used

* Anaconda – Python 3.7
* Jupyter Notebook / VS Code
* OpenCV (cv2)
* NumPy
* Matplotlib

---

##  Algorithm & Explanation

---

###  Step 1: Import Libraries

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```

---

###  Step 2: Read the Image

```
image = cv2.imread("road.png")

if image is None:
    raise FileNotFoundError("Could not load 'road.png'. Check the file path.")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB
```


###  Step 3: Convert to Grayscale

```
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

---

###  Step 4: Display Images

```
plt.figure(figsize=(10,5))


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

plt.tight_layout()
plt.show()

---

###  Step 5: Thresholding

```
threshold = 150
_, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

plt.figure(figsize=(6, 6))
plt.imshow(thresh, cmap="gray")
plt.title("Thresholded Image")
plt.axis("off")
plt.show()

---

###  Step 6: Region of Interest (ROI)

```
height, width = thresh.shape

roi_vertices = np.array([[
    (int(0.1 * width), height),
    (int(0.45 * width), int(0.6 * height)),
    (int(0.55 * width), int(0.6 * height)),
    (int(0.9 * width), height)
]], dtype=np.int32)

mask = np.zeros_like(thresh)
cv2.fillPoly(mask, roi_vertices, 255)
roi_masked = cv2.bitwise_and(thresh, mask)

plt.figure(figsize=(6, 6))
plt.imshow(roi_masked, cmap="gray")
plt.title("ROI Masked Image")
plt.axis("off")
plt.show()

---

### Step 7: Edge Detection (Canny)

```
edges = cv2.Canny(roi_masked, 50, 150)

plt.figure(figsize=(6, 6))
plt.imshow(edges, cmap="gray")
plt.title("Edge Detected Image")
plt.axis("off")
plt.show()

---

###  Step 8: Gaussian Blur

```
smoothed = cv2.GaussianBlur(edges, (5, 5), 0)

plt.figure(figsize=(6, 6))
plt.imshow(smoothed, cmap="gray")
plt.title("Smoothed (Blurred) Edge Image")
plt.axis("off")
plt.show()

---

###  Step 9: Hough Transform

```
# Detect lines using Hough Transform

lines = cv2.HoughLinesP(
    smoothed,
    rho=2,
    theta=np.pi / 180,
    threshold=50,
    minLineLength=40,
    maxLineGap=100
)

line_image = np.zeros_like(image)

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line.flatten()
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

line_image_rgb = cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(6, 6))
plt.imshow(line_image_rgb)
plt.title("Detected Lines")
plt.axis("off")
plt.show()

---

### Step 10: Lane Detection Logic

```

final_output = cv2.addWeighted(image, 0.8, line_image, 1.0, 0.0)
final_output_rgb = cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(6, 6))
plt.imshow(final_output_rgb)
plt.title("Final Lane Detection Output")
plt.axis("off")
plt.show()
---

##  Expected Output

* Original image

  <img width="373" height="585" alt="Screenshot 2026-08-20 090219" src="https://github.com/user-attachments/assets/33686308-e0f9-40d1-8b6b-002b5e61de13" />

* Grayscale image
   <img width="1005" height="651" alt="Screenshot 2026-08-20 090610" src="https://github.com/user-attachments/assets/df11dac9-8357-49d0-a8c3-12f731b4879b" />


* Thresholded image

  <img width="471" height="642" alt="637315438-396b4262-b839-4b6b-8b8a-81f407886a8e" src="https://github.com/user-attachments/assets/1e7e8553-c0f6-425b-9d8c-7a87b2c106b5" />

* ROI masked image

  <img width="474" height="654" alt="637315837-4bed56a0-2990-407d-b291-f8e9b16408db" src="https://github.com/user-attachments/assets/e55f99eb-3fb5-4887-a42a-1f433b2d3b70" />

* Edge detected image

  <img width="430" height="635" alt="637316086-b1f90593-ce9c-4abc-8390-ba27d2145ae9" src="https://github.com/user-attachments/assets/dd48f58e-e63e-4c01-860d-824df507abbe" />

* Smoothed image

  <img width="448" height="636" alt="637316315-70ef3fdf-bb1b-4735-8852-812b58f17420" src="https://github.com/user-attachments/assets/ab470ede-7f50-4fa5-a265-9c7267565d1f" />

* Detected lines

  <img width="407" height="419" alt="637316543-0835414e-4a9b-4c5c-92b6-7c9a9a94c1dd" src="https://github.com/user-attachments/assets/7e694f53-ea12-4eb7-a2f1-5b8fba885184" />

* Final lane detection output

  <img width="388" height="639" alt="637316729-512ddf5f-8dbe-479e-99f1-73720318b6f0" src="https://github.com/user-attachments/assets/cb0a6b3a-e674-42fb-866c-a46522d0ce1f" />



---

## Result

Thus, the lane detection pipeline is successfully implemented by completing the missing code sections. The system detects and highlights lane lines effectively.

---


