# CarAlert
Israeli license plates recognition - does not require resource-consuming OCR to run!

This project aims at recongizing Israeli license plates but can be adjusted to recognize other number plates as well.
The use case is:
1) Point your smatrphone's camer at the car number plate (at "normal" distance).
2) Take a photo.
3) Invoke this code and you get a number plate recognized.
It is a good match for apps that require number plate recognition, e.g.:
- Send a message to the car owner
- Report a car that violated driving laws
etc.

The code depends on OpenCV v2 (attempted 2.4.9 and above), and can be easily compiled on both Windows and Linux (and, obviously, on Andriod).

License: GPL V2.

If you would like to embed this code to your application at different terms, please contact me at sergof@gmail.com


# Algorithm

## Basic pipeline
1) License plate detection.
  - Thresholding (attempting a simpler one first, if it yields no detection trying the more expensive one).
  - Connected components search.
  - Heuristics to determine whether a set of co-located connected components is an Israeli license plate (currently, only old-style 7-    digits license plates are supported).
2) Recognition of numbers in the license plate.
  - This is based on ConvNet trained on Street View House Numbers.
  - The code includes self-contained implementation of the required layers. While it might not be the best performing, it carries no external dependencies.
  
## Potential improvements
1) Make detection pipeline also ConvNet-based (style-YOLO or Faster-RCNN). In this particular case, it is probably an overkill.
2) Train more robust network for recognition (although the one I trained is pretty robust).
