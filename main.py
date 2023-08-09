import cv2
import MultiProcessLabelingModule as mplm
import numpy as np
import HandtrackingModule as htm
import random
import pickle

with open("modelThumb.pkl", "rb") as f:
    modelThumb = pickle.load(f)

with open("modelIndex.pkl", "rb") as f:
    modelIndex = pickle.load(f)

def returnAngleArray(lmList, tip, hand):
    landmarksList = [lmList[hand][tip][1:4], lmList[hand][tip-1][1:4], lmList[hand][tip-2][1:4], lmList[hand][tip-3][1:4], lmList[hand][0][1:4]] #We'll get data from only one hand at a time
    angles = []
    for point in range(1,4):
        angles.append(mplm.angle_between_vectors(mplm.construct_vector(np.array(landmarksList[point]), np.array(landmarksList[point-1])),
                                                mplm.construct_vector(np.array(landmarksList[point]), np.array(landmarksList[point+1]))))
    return np.array(angles)

def advancedCountFingers(lmList): #Takes landmarks list and outputs the number of fingers raised
    tipIds = [4, 8, 12, 16, 20]
    fingersRaised = 0
    if(len(lmList) > 0):
        for hand in range(len(lmList)):
            for tip in tipIds:
                angles = returnAngleArray(lmList, tip, hand)
                angles = angles.reshape(1, -1)
                if(tip == 4): #Special logic for Thumb
                    prediction = modelThumb.predict(angles)
                    fingersRaised += 1 if prediction>=0.5 else 0
                else:
                    prediction = modelIndex.predict(angles)
                    fingersRaised += 1 if prediction >= 0.5 else 0
            
    return fingersRaised

def computerVisionMath(countFingersRaised, numQuestions):
    cap = cv2.VideoCapture(0)
    detector = htm.HandDetector()
    newRound = True
    rand1 = rand2 = res = 0
    framesRight = 0
    prevRand2 = 0
    round = 0
    
    try:
        while round <= numQuestions:
            success, img = cap.read()
            img = cv2.flip(img, 1)
            
            lmList, img = detector.findLandmarks(img)
            fingersRaised = countFingersRaised(lmList)
            
            if newRound:
                round += 1
                if(round > numQuestions):
                    continue
                rand1 = random.randint(1,10)
                while(rand2 == prevRand2):
                    rand2 = random.randint(1,10)
                prevRand2 = rand2
                res = rand1 * rand2
                print(f"Question {round}/{numQuestions} : {res} / {rand1} = ", end = "")
                newRound = False
            
            if(fingersRaised == rand2):
                framesRight += 1
                if(framesRight == 3): #You should get 3 frames right continously -> Avoid accidental answers
                    print(fingersRaised)
                    newRound = True
                    framesRight = 0
            else:
                framesRight = 0
            
            cv2.imshow("Computer Vision Math", img)
            cv2.waitKey(1)
        cap.release() #Release camera
        cv2.destroyAllWindows()  # Close all OpenCV windows
        cap = None
    except KeyboardInterrupt:
        cap.release() #Release camera
        cv2.destroyAllWindows()  # Close all OpenCV windows
        cap = None
        
if __name__ == "__main__":
    computerVisionMath(advancedCountFingers, 10)