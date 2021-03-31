# Posture monitor
Monitoring posture by OpenCV.

When system detects your poor posture, the monitor goes dark.

## requirements
OS: Windows, (Linux)

```
pip install opencv-python
pip install creen-brightness-control
```

# Preparations
import face detection model `haarcascade_frontalface_alt2.xml` from https://github.com/opencv/opencv/tree/master/data/haarcascades


# Run
```
python posture_monitor.py
```