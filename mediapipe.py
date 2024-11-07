import cv2
import mediapipe as mp
import time
cap=cv2.VideoCapture("2.mp4")
pTime=0

while True:
		success, img=cap.read()
		
		cTime=time.time()
		fps=1/(cTime-pTime)
		pTime = cTime
		cv2.putText(img, f'FPS:{int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),2)
		cv2.imshow("Image",img)
		if cv2.waitKey(15) & 0xFF == ord('q'):
			break  # Press 'q' to exit