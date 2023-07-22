import mediapipe as mp 
import numpy as np 
import cv2

def main():
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        lmh, lmw, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        result = mp_face_mesh.process(rgb_frame)

        if result.multi_face_landmarks:  # Check if face landmarks were detected
            for facial_landmarks in result.multi_face_landmarks:
                for i in range(0, 468):
                    lm = facial_landmarks.landmark[i]
                    x = int(lm.x * lmw)
                    y = int(lm.y * lmh)
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        
        cv2.imshow('Show Face', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
