from ultralytics import YOLO 
import cv2 
import time

model = YOLO ("best.pt")

cam = cv2.VideoCapture('rtsp://admin:abcd1234@192.168.0.3:554/Streaming/Channels/1/') 
if not cam.isOpened():
    raise("No Camera")
while True:
    ret, frame = cam.read()
    video = cv2.resize(frame, (900, 600))
    if not ret: 
        break
    _time_mulai = time.time()
    result = model.predict(video, show=True)
    

    print("waktu", time.time()-_time_mulai) 
    #cv2.imshow("video_asli", video)
    _key = cv2.waitKey(1)
    if _key == ord('q'): 
        break

cam.release()
cv2.destroyAllWindows()