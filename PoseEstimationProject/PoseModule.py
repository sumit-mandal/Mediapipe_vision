import cv2
import mediapipe as mp
import math
import time

class poseDetector():

    def __init__(self,mode = False,model_complexity=1, upBody= False, smooth = True,
    detectionCon= 0.5,trackCon=0.5):

        self.mode = mode
        self.model_complexity = model_complexity
        self.upBody = upBody # upper body
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode,self.model_complexity,self.upBody,self.smooth,self.detectionCon,self.trackCon)


    def findPose(self, img, draw = True):

        img = cv2.resize(img,(800,800))
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)


        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,
                self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self,img,draw=True):
        """Gives the list of all the points"""
        self.lmList = []
        if self.results.pose_landmarks:
            # Get landmark in respective positions
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c = img.shape
                # print(id,lm) # landmarks are basically ratio of the image

                # get the actuall pixel value and not just ratio.
                cx,cy = int(lm.x*w), int(lm.y*h)
                self.lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),4,(255,0,0),cv2.FILLED)
        return self.lmList

    def findAngle(self,img,p1,p2,p3,draw = True):
        '''Takes in an image and points p1 p2 and p3. Returns the angle'''

        # Get the landmarks
        x1,y1 = self.lmList[p1][1:]
        x2,y2 = self.lmList[p2][1:]
        x3,y3 = self.lmList[p3][1:]

        # Calculate the angle
        angle = math.degrees(math.atan2(y3-y2,x3-x2)- math.atan2(y1-y2,x1-x2))

        if angle < 0:
            angle += 360


        #Draw

        if draw:
            cv2.line(img,(x1,y1),(x2,y2),(255,0,255),3)
            cv2.line(img,(x3,y3),(x2,y2),(255,0,255),3)
            cv2.circle(img,(x1,y1),10,(255,0,0),cv2.FILLED)
            cv2.circle(img,(x1,y1),15,(0,0,255),2)
            cv2.circle(img,(x1,y2),10,(255,0,0),cv2.FILLED)
            cv2.circle(img,(x2,y2),15,(0,0,255),2)
            cv2.circle(img,(x3,y3),5,(255,0,0),cv2.FILLED)
            cv2.circle(img,(x3,y3),10,(0,0,255),2)
            cv2.putText(img,str(int(angle)),(x2-50,y2+50),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)

        return angle





def main():
    cap = cv2.VideoCapture('Video/4.mp4')
    pTime = 0
    detector = poseDetector()
    while (cap.isOpened()):
        success,img = cap.read()
        if success == True:
            img = detector.findPose(img)
            lmList = detector.findPosition(img,draw=False)
            if len(lmList) != 0:
                # print(lmList[14])
                cv2.circle(img,(lmList[14][1], lmList[14][2]), 5, (130,20,64),cv2.FILLED)

            cTime = time.time()
            fps = 1/(cTime-pTime)
            pTime = cTime
            cv2.putText(img,str(int(fps)),(70,50), cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),3)
            cv2.imshow("Image",img)

            if cv2.waitKey(1) == ord('q'):
                break
        else:
            break


    cap.release()

        # Closes all the frames
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()

# If we run  this file it will run the main() function
# If we are calling other function it will not run this part
