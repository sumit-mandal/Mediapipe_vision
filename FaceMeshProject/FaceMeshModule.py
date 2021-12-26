import cv2
import mediapipe as mp
import time

class FaceMeshDetector():
    def __init__(self,staticMode = False,maxFaces=2,refineLandmarks=False,minDetectionCon=0.5,minTrackingCon=0.5):

        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.refineLandmarks = refineLandmarks
        self.minDetectionCon = minDetectionCon
        self.minTrackingCon = minTrackingCon


        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode,self.maxFaces,
        self.refineLandmarks,self.minDetectionCon,self.minTrackingCon)
        self.drawSpec = self.mpDraw.DrawingSpec((240,0,240),thickness=1, circle_radius=2)

    def findFaceMesh(self,img,draw=True):

        img = cv2.resize(img,(800,800))
        self.imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,faceLms,self.mpFaceMesh.FACEMESH_CONTOURS,
                                         self.drawSpec,self.drawSpec)

                face = []
                for id,lm in enumerate(faceLms.landmark):
#                     print(lm)
                    #Convert landmark into pixels
                    ih,iw,ic = img.shape
                    x,y = int(lm.x*iw),int(lm.y*ih)
                    cv2.putText(img,str(id),(x,y),cv2.FONT_HERSHEY_PLAIN,0.3,(0,255,0),1)
                    # print(id,x,y)
                    face.append([x,y])
                faces.append(face)

        return img,faces





def main():

    cap = cv2.VideoCapture("Video/5.mp4")
    pTime = 0
    detector = FaceMeshDetector(maxFaces=2)

    while(cap.isOpened()):
        success,img = cap.read()
        if success == True:
            img,faces = detector.findFaceMesh(img)
            if len(faces)!=0:
                print(len(faces))
            # cTime = time.time()

            cTime = time.time()
            fps = 1/(cTime-pTime)
            pTime = cTime
            cv2.putText(img,f'FPS:{int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),3)
            cv2.imshow("Image",img)

            if cv2.waitKey(1) == ord('q'):
                break
        else:
            break


    cap.release()
    cv2.destroyAllWindows()



if __name__=='__main__':
    main()
