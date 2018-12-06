# Traffic Counting app
An app that detects and counts vehicles from static camera videos. It's based on Comupter Vision and OpenCV.

We use background subtraction algorithm MOG2 to separate moving vehicles from the background and then some morphological transformations to improve accuracy of detection.

When camera is set on the middle of the road then just one line for couting is enough. Otherwise, there are needed 2 lines (set by default) - one for the left side cars and the another for the right side ones. Also, in this situation you should adjust these lines appropriate to the video. 
