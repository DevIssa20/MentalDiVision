import cv2
import time
import mediapipe as mp

class HandDetector(): #Defining the HandDetector class
    def __init__(self, mode = False, maxHands = 2, detectionCon = 0.5, trackCon = 0.5, model_complexity = 1):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.model_complexity = model_complexity

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode = self.mode,max_num_hands = self.maxHands,min_detection_confidence = self.detectionCon,min_tracking_confidence =  self.trackCon, model_complexity = self.model_complexity)
        self.mpDraw = mp.solutions.drawing_utils
        
    def findLandmarks(self, img, draw = True): #Takes an image and outputs a landmarks list and an image
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        
        lmList = []
        h,w,_ = img.shape
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                lmList.append([(lm[0], lm[1].x, lm[1].y, lm[1].z) for lm in enumerate(handLms.landmark)])
                if draw: #Draw the landmarks on the image if draw is set to true
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return lmList, img
        
def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0) #Create video object
    detector = HandDetector()
    frame = 0
    while True: #Start camera footage
        success, img = cap.read()
        img = cv2.flip(img, 1)
        lmList, img = detector.findLandmarks(img)
        if(frame == 60):
            frame = 0
        else:
            frame += 1
        
        #Calculate FPS
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        
        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3,(255,0,255),3) #Display FPS
        
        cv2.imshow("Image", img)
        cv2.waitKey(1)
    
if __name__ == "__main__":
    main()