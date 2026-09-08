# License Plate Detection Using Haar Cascade

**Name:** Joel John Jobinse  
**Register Number:** 212223240062

---

## Aim

To develop a Python-based license plate detection system using OpenCV and a pre-trained Haar Cascade classifier, wiAth image preprocessing using CLAHE and Gaussian Blur to improve detection performance.

---

## Equipment Required

### Hardware
- Computer/Laptop
- Minimum 4 GB RAM recommended
- Webcam or a sample vehicle image containing a license plate

### Software
- Python 3.x
- Jupyter Notebook / JupyterLab
- OpenCV (`cv2`)
- NumPy
- Matplotlib
- `haarcascade_russian_plate_number.xml`
- Input image: `car_plate.jpg`

---

## Algorithm

1. Import the required Python libraries: OpenCV, NumPy, Matplotlib, and `Path`.
2. Locate the input image `car_plate.jpg` and the Haar Cascade XML file.
3. Read the input vehicle image using OpenCV.
4. Convert the image from BGR to grayscale.
5. Improve the grayscale image using **CLAHE (Contrast Limited Adaptive Histogram Equalization)**.
6. Apply a small **Gaussian Blur** to reduce high-frequency noise.
7. Load the pre-trained `haarcascade_russian_plate_number.xml` Haar Cascade classifier.
8. Detect license plates using `detectMultiScale()`.
9. Draw bounding boxes and labels around detected license plates.
10. Crop the detected license plate regions from the original image.
11. Compare basic grayscale detection with the improved preprocessing-based detection.
12. Select the largest detected plate region and save it as `outputs/detected_plate.png`.
13. Experiment with different `scaleFactor` and `minNeighbors` values to observe their effect on detection.
14. Display the detection and cropped license plate results.

### Detection Parameters

- **scaleFactor = 1.1:** Controls the scale reduction used during multi-scale detection.
- **minNeighbors = 5:** Controls how many neighboring detections are required before a region is accepted.
- **minSize = (30, 10):** Defines the minimum size of a detected license plate region.

---

## Program

### 1. Import Required Libraries

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print("OpenCV version:", cv2.__version__)
```

### 2. Locate the Input Image and Haar Cascade

```python
image_candidates = [
    Path("car_plate.jpg"),
    Path("/mnt/data/car_plate.jpg")
]

image_path = next((p for p in image_candidates if p.exists()), None)

if image_path is None:
    raise FileNotFoundError(
        "car_plate.jpg was not found. Place car_plate.jpg in the same folder as this notebook."
    )

cascade_candidates = [
    Path("haarcascade_russian_plate_number.xml"),
    Path(cv2.data.haarcascades) / "haarcascade_russian_plate_number.xml"
]

cascade_path = next((p for p in cascade_candidates if p.exists()), None)

if cascade_path is None:
    raise FileNotFoundError(
        "haarcascade_russian_plate_number.xml was not found. "
        "Place it in the notebook folder."
    )

print("Image:", image_path.resolve())
print("Cascade:", cascade_path.resolve())
```

### 3. Read and Display the Input Image

```python
img = cv2.imread(str(image_path))

if img is None:
    raise ValueError("The image could not be read. Check the image path.")

def display_image(image, title="", figsize=(12, 7)):
    plt.figure(figsize=figsize)
    if len(image.shape) == 2:
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()

print("Image shape:", img.shape)
display_image(img, "Original Input Image")
```

### 4. Convert the Image to Grayscale

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

display_image(gray, "Grayscale Image")
```

### 5. Image Preprocessing Using CLAHE and Gaussian Blur

```python
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

enhanced_gray = clahe.apply(gray)

preprocessed_gray = cv2.GaussianBlur(
    enhanced_gray,
    (5, 5),
    0
)
```

The preprocessing improves local contrast using CLAHE and reduces high-frequency noise using Gaussian Blur.

### 6. Load the Haar Cascade Classifier

```python
plate_cascade = cv2.CascadeClassifier(str(cascade_path))

if plate_cascade.empty():
    raise ValueError("The Haar Cascade XML could not be loaded correctly.")

print("Haar Cascade loaded successfully.")
```

### 7. License Plate Detection Function

```python
def detect_plates(
    image_bgr,
    detection_gray,
    scale_factor=1.1,
    min_neighbors=5
):
    rectangles = plate_cascade.detectMultiScale(
        detection_gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(30, 10)
    )

    annotated = image_bgr.copy()
    crops = []

    for i, (x, y, w, h) in enumerate(rectangles, start=1):

        cv2.rectangle(
            annotated,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated,
            f"Plate {i}",
            (x, max(y - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        crop = image_bgr[y:y+h, x:x+w].copy()
        crops.append(crop)

    return annotated, rectangles, crops
```

### 8. Basic Haar Cascade Detection

```python
basic_result, basic_rectangles, basic_crops = detect_plates(
    img,
    gray,
    scale_factor=1.1,
    min_neighbors=5
)

print("Basic detection count:", len(basic_rectangles))
print("Detected rectangles:", basic_rectangles)

display_image(
    basic_result,
    "Basic Haar Cascade Detection"
)
```

### 9. Improved Detection Using Preprocessing

```python
improved_result, improved_rectangles, improved_crops = detect_plates(
    img,
    preprocessed_gray,
    scale_factor=1.1,
    min_neighbors=5
)

print("Improved detection count:", len(improved_rectangles))
print("Detected rectangles:", improved_rectangles)

display_image(
    improved_result,
    "Improved Haar Cascade Detection"
)
```

### 10. Compare Basic and Improved Detection

```python
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(basic_result, cv2.COLOR_BGR2RGB))
plt.title(
    f"Basic Detection ({len(basic_rectangles)} plate(s))"
)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(improved_result, cv2.COLOR_BGR2RGB))
plt.title(
    f"Improved Detection ({len(improved_rectangles)} plate(s))"
)
plt.axis("off")

plt.tight_layout()
plt.show()
```

### 11. Crop and Save the Detected License Plate

```python
final_rectangles = (
    improved_rectangles
    if len(improved_rectangles) > 0
    else basic_rectangles
)

final_crops = (
    improved_crops
    if len(improved_crops) > 0
    else basic_crops
)

if len(final_crops) == 0:
    print(
        "No license plate was detected. "
        "Try lowering minNeighbors or adjusting scaleFactor."
    )
else:
    best_index = int(
        np.argmax([
            w * h
            for (x, y, w, h) in final_rectangles
        ])
    )

    detected_plate = final_crops[best_index]

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "detected_plate.png"
    cv2.imwrite(str(output_path), detected_plate)

    print(
        "Detected plate saved to:",
        output_path.resolve()
    )

    display_image(
        detected_plate,
        "Cropped License Plate",
        figsize=(8, 3)
    )
```

### 12. Parameter Experimentation

```python
parameter_sets = [
    (1.05, 3),
    (1.10, 5),
    (1.20, 5),
    (1.30, 5),
]

print("scaleFactor | minNeighbors | detections")
print("----------------------------------------")

for scale_factor, min_neighbors in parameter_sets:
    rects = plate_cascade.detectMultiScale(
        preprocessed_gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(30, 10)
    )

    print(
        f"{scale_factor:10.2f} | "
        f"{min_neighbors:12d} | "
        f"{len(rects)}"
    )
```

---

## Output

> **Note:** Replace each placeholder below with the corresponding screenshot from the executed Jupyter Notebook.

### 1. Original Input Image

<img width="1092" height="670" alt="image" src="https://github.com/user-attachments/assets/0aa7a1aa-39d0-46a1-bf1e-8b79408c0dfd" />


---

### 2. Grayscale Image
<img width="1147" height="662" alt="image" src="https://github.com/user-attachments/assets/60494062-6a59-4151-ba46-dc5043fef185" />


---

### 3. Preprocessing Result

This output shows the original grayscale image, CLAHE-enhanced image, and the final image after Gaussian Blur.

<img width="1258" height="285" alt="image" src="https://github.com/user-attachments/assets/5661ac0d-6aa0-4e74-a3c8-3f1786f59c2d" />

---

### 4. Basic Haar Cascade Detection
<img width="1143" height="652" alt="image" src="https://github.com/user-attachments/assets/4c914f1c-743d-46a7-b653-ba3bf39b3460" />


---

### 5. Improved Haar Cascade Detection

<img width="1102" height="698" alt="image" src="https://github.com/user-attachments/assets/2279b360-a0d7-47ec-a46f-a53335a90663" />

---

### 6. Basic vs Improved Detection

<img width="1270" height="407" alt="image" src="https://github.com/user-attachments/assets/5ad55bd4-bd4c-41f1-91db-9b96e920eac3" />

---

### 7. Cropped License Plate

<img width="1251" height="350" alt="image" src="https://github.com/user-attachments/assets/c871a7aa-81a0-4b02-8320-17477268979a" />

---

### 8. Parameter Experimentation
<img width="418" height="228" alt="image" src="https://github.com/user-attachments/assets/db10ecc7-5a93-4da6-8c81-e307ae704adc" />


---

## Result

The license plate detection system was successfully implemented using **OpenCV and a pre-trained Haar Cascade classifier**. The input vehicle image was converted to grayscale and enhanced using **CLAHE and Gaussian Blur** before detection. The system detected license plate regions, marked them with bounding boxes, cropped the detected plate, and saved the result as `outputs/detected_plate.png`. Basic and improved detection results were also compared, and different Haar Cascade parameters were tested to study their effect on detection.

---

## Conclusion

The notebook demonstrates a complete license plate detection pipeline using image preprocessing and Haar Cascade-based object detection. The preprocessing stage provides an improved input for the classifier, while parameter experimentation helps understand how detection sensitivity can be adjusted.
