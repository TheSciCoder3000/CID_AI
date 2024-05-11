
# Traffic Congestion Detection System - DSPML LAB


## Installation

**PIP Packages Installation**

navigate to the project directory and run the following code

```bash
  pip install -r requirements.txt
```
## How to Run

```bash
python main.py [--webcam-resolution <width> <height>]
```


## Warnings

The python script uses the 2nd Video Capture/webcam by default, just change the following code to select the only available video capture device available.
```python
# change the following code from this:
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# To this:
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
```

