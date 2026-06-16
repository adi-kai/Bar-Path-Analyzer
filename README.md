# Bar-Path-Analyzer


# What is it?
A Python script that takes a weightlifting video as input and
outputs an annotated video with:


* Color-coded bar path trail (blue=slow → red=fast)
* Peak height annotation (dot + label box, floor-relative)
* Peak velocity annotation (dot + label box)
* MediaPipe Pose skeleton overlay on every frame
* Speed legend at bottom left
* Title bar at top

# Working enviroment:
OS: Windows (PowerShell)
Python: 3.11.9  (MUST use this, not 3.14)
MediaPipe: 0.10.9  (MUST use this specific version)
OpenCV: latest (opencv-python)
NumPy: latest

Why 3.11 not 3.14: 
MediaPipe does not support Python 3.14.

# How to run?
1st. Install deps (one time)
py -3.11 -m pip install opencv-python mediapipe==0.10.9 numpy
2nd. Run
py -3.11 bar_path_analyzer.py 
--input "yourfile.mp4" 
--output result.avi 
--bar_color (red | blue | black | green | yellow | auto)

Example: py -3.11 bar_path_analyzer.py --input "snatch.mp4" --output result.avi --bar_color green     


  
