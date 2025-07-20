import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import os
import sys

def main():
    model_path = os.path.join("Model", "keras_model.h5")
    labels_path = os.path.join("Model", "labels.txt")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please ensure you have the keras_model.h5 file in the Model folder")
        return
    
    if not os.path.exists(labels_path):
        print(f"Error: Labels file not found at {labels_path}")
        print("Please ensure you have the labels.txt file in the Model folder")
        return
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        print("Please check if your camera is connected and not being used by another application.")
        return
    
    detector = HandDetector(maxHands=1)
    
    try:
        classifier = Classifier(model_path, labels_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("This might be due to version compatibility issues.")
        cap.release()
        return
    
    offset = 20
    imgSize = 300
    labels = ["HELLO", "THANK YOU", "YES", "NO", "PEACE"]
    
    print("Press 'q' to quit")
    
    while True:
        success, img = cap.read()
        if not success:
            print("Error: Failed to read from webcam.")
            break
            
        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            y1, y2 = max(0, y-offset), min(img.shape[0], y + h + offset)
            x1, x2 = max(0, x-offset), min(img.shape[1], x + w + offset)
            imgCrop = img[y1:y2, x1:x2]
            
            if imgCrop.size == 0: 
                continue
                
            aspectRatio = h / w
            
            try:
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    if wCal > 0:
                        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                        wGap = math.ceil((imgSize-wCal)/2)
                        imgWhite[:, wGap: wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    if hCal > 0:
                        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                        hGap = math.ceil((imgSize - hCal) / 2)
                        imgWhite[hGap: hCal + hGap, :] = imgResize
                
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                cv2.rectangle(imgOutput, (x-offset, y-offset-70), 
                            (x-offset+300, y-offset+10), (255, 150, 4), cv2.FILLED)
                cv2.putText(imgOutput, labels[index], (x-offset+10, y-offset-30), 
                          cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
                cv2.rectangle(imgOutput, (x-offset, y-offset), 
                            (x + w + offset, y + h + offset), (255, 150, 4), 4)
                            
            except Exception as e:
                print(f"Error during prediction: {e}")
                continue
        else:
            cv2.putText(imgOutput, "No hand detected", (50, 50), 
                       cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('MarkSense - Hand Sign Recognition', imgOutput)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed successfully.")

if __name__ == "__main__":
    main()