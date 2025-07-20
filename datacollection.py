import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    detector = HandDetector(maxHands=1)
    offset = 20
    imgSize = 300
    counter = 0
    gesture_name = input("Enter gesture name (e.g., HELLO, PEACE, etc.): ").upper()
    folder = os.path.join(os.getcwd(), "train", gesture_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created folder: {folder}")
    
    print("Instructions:")
    print("- Position your hand in front of the camera")
    print("- Press 's' to save image")
    print("- Press 'q' to quit")
    print(f"Saving images to: {folder}")
    
    while True:
        success, img = cap.read()
        if not success:
            print("Error: Failed to read from webcam.")
            break
            
        hands, img = detector.findHands(img)
        
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            y1, y2 = max(0, y-offset), min(img.shape[0], y + h + offset)
            x1, x2 = max(0, x-offset), min(img.shape[1], x + w + offset)
            imgCrop = img[y1:y2, x1:x2]
            
            if imgCrop.size > 0: 
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
                    
                    cv2.imshow('ImageCrop', imgCrop)
                    cv2.imshow('ImageWhite', imgWhite)
                    
                except Exception as e:
                    print(f"Error processing image: {e}")
                    continue
        cv2.putText(img, f"Gesture: {gesture_name}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, f"Images saved: {counter}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, "Press 's' to save, 'q' to quit", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Data Collection', img)
        
        key = cv2.waitKey(1)
        if key == ord("s") and hands:
            counter += 1
            filename = f'{gesture_name}_{counter}_{int(time.time())}.jpg'
            cv2.imwrite(os.path.join(folder, filename), imgWhite)
            print(f"Saved: {filename} (Total: {counter})")
            
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Data collection completed. Saved {counter} images.")

if __name__ == "__main__":
    main()