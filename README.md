
# Traffic Congestion Detection System - DSPML LAB

## Objectives

The goal of this study is to propose an intelligent traffic light system using machine learning with object detection as well as algorithms to perform real-time strategic signal switching at intersections, potentially reducing waiting time for vehicles and improving the overall travel experience in the Philippines.
1. To adapt the YOLOv8 pre-trained deep learning model for object detection and assess its effectiveness in identifying cars, trucks, and motorcycles.
2. To develop an algorithm aimed at minimizing vehicle waiting times at intersections based on the level of road congestion.
3. To evaluate the performance of the entire program by testing it in an imitated traffic congestion scenario. 


## Installation

**Python Virtual Environment [Optional]**

You can optionally install the packages on a virtual environment using the command below.
```bash
  python -m venv venv
```

**PIP Packages Installation**

Navigate to the project directory and run the following code to install the packages used in this repo

```bash
  pip install -r requirements.txt
```

**GPU Support [Optional]**

For GPU support, install the CUDA enabled torch 2.1.0 and related libraries using the code below. 

```bash
  pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

This repo was built using NVIDIA GTX 1650 graphics card with CUDA Toolkit 11.8 installed and CuDNN 8.8.0. Please install torch with their respective CUDA version from the official [torch website](https://pytorch.org/get-started/previous-versions/)

## How to Run
Run the program using the command below. You can optionally set the width and height of the window as shown below.
```bash
python main.py [--webcam-resolution <width> <height>]
```


## Warnings

The python script uses the Video Capture/webcam by default, just change the following code to select the only available video capture device available.
```python
# change the following code from this:
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# To this:
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
```

