# Drowsy man detector

A computer vision system that can automatically detect driver drowsiness in a real-time video stream and then warning if the driver appears to be drowsy.

Based on Eye Aspect Ratio and Mouth Aspect Ratio to detect when the driver’s eyes are closed too long or yawns too often.

Using pre-trained Haar cascade for localizing faces and dlib’s shape predictor to detect landmarks on the face.

Because of using Eye Aspect Ratio and Mouth Aspect Ratio thresholds, the system need to be adjusted that thresholds to fit with each person.

# Output examples
![alt text](https://github.com/daoducanhc/drowsy_man_detected_/blob/master/1.PNG) ![alt text](https://github.com/daoducanhc/drowsy_man_detected_/blob/master/2.PNG)

