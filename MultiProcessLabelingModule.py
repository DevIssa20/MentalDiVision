import numpy as np #For working with numpy arrays and functions
import cv2 #For working with images
import mediapipe as mp #For hand tracking
import multiprocessing #For the labeling script 
import time #For timing
import os #For file path_names
import csv #For writing csv files
import HandtrackingModule as htm

#Let's define some helper functions
def construct_vector(point1, point2): #Given two 3D points return their vector
    vector = point2 - point1
    return vector
def angle_between_vectors(vector1, vector2): #Given two vectors return their angle
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    cos_theta = dot_product / (magnitude1 * magnitude2)
    angle_radians = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees

# We'll also define the function of process 1
def captureAndLabel(label, framesLeft, tip):
    fileNames = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    fileName = "trainingData" + fileNames[tip//4 - 1] + ".csv"
    current_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_directory, fileName) 
    cap = cv2.VideoCapture(0)
    detector = htm.HandDetector()

    pTime = 0
    cTime = 0
    landmarksList = []
    frameNum = 0
    while True:  # Loop until terminate_event is set
        try:
            if(framesLeft.value > 0 and len(lmList) > 0):
                if(frameNum%2 == 0):
                    landmarksList.append([lmList[0][tip][1:4], lmList[0][tip-1][1:4], lmList[0][tip-2][1:4], lmList[0][tip-3][1:4], lmList[0][0][1:4]]) #We'll get data from only one hand at a time
                    framesLeft.value -= 1
                    frameNum = 0
                frameNum += 1
            elif(framesLeft.value == 0):
                with open(file_path, "a", newline="") as f:
                    csv_writer = csv.writer(f)
                    for entry in landmarksList:
                        angles = []
                        for point in range(1,4):
                            angles.append(angle_between_vectors(construct_vector(np.array(entry[point]), np.array(entry[point-1])),
                                                                construct_vector(np.array(entry[point]), np.array(entry[point+1]))))
                        angles.append(label.value)
                        csv_writer.writerow(angles)
                landmarksList = []
                            
            elif(framesLeft.value == -1):
                cap.release()
                cv2.destroyAllWindows()
                cap = None
                return
            success, img = cap.read()
            img = cv2.flip(img, 1)
            lmList, img = detector.findLandmarks(img)
                
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            
            cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3) #Display
            cv2.putText(img, str(framesLeft.value), (480, 70), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 255), 4) #Display frames left
            
            cv2.imshow("Image", img)
            cv2.waitKey(1)
        except KeyboardInterrupt:
            cap.release()
            cv2.destroyAllWindows()
            cap = None
            return
            
                
def startLabelingProcess(tip):
    multiprocessing.set_start_method("spawn", force=True)  # Required for Windows compatibility
    label = multiprocessing.Value("i", -1)
    framesLeft = multiprocessing.Value("i", 0)
    
    input_process = multiprocessing.Process(target=captureAndLabel, args=(label, framesLeft, tip))
    input_process.start()
    
    while True:
        if framesLeft.value == 0:
            label.value = int(input("Enter 0 or 1 for new labeling round or anything else to the stop program  : "))
            if(label.value not in [0,1]):
                framesLeft.value = -1
                break
            framesLeft.value = 250
                
if __name__ == "__main__":
    startLabelingProcess(20)