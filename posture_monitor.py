import cv2
import screen_brightness_control as sbc
import numpy as np
import time
import csv

def set_brightness(brightness):
    try:
        sbc.set_brightness(brightness)
    except sbc.ScreenBrightnessError as error:
        print(error)

# モデルを指定(haarcascade_frontalface_alt2.xmlを使用)
model_path = './haarcascade_frontalface_alt2.xml'

# 姿勢の記録先
log_path = './posture_log.csv'
f = open(log_path, 'a', newline='')
writer = csv.writer(f)
writer.writerow(['timestamp', 'x', 'y', 'y_th'])

# 設定パラメータ
monitor_interval = 100
th_posture = 30 # カメラ内で何ピクセル頭の位置が下がったら姿勢が悪いと判定するか(小さいほど自分に厳しい)
th_count = 10 # 何回姿勢が良い・悪いと判定されたら画面の明るさを変えるか
brightness_dark = 20
brightness_bright = 100

calib_flag = False
postures = []
good_posture = None
bad_posture_count = 0
good_posture_count = 0
bad_posture_flag = False

capture = cv2.VideoCapture(0) # VideoCapture オブジェクトを取得
capture_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
capture_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
cascade = cv2.CascadeClassifier(model_path) #学習データの取り込み

print('Push "s" to start calibration')
while(True):
    ret, frame = capture.read()
    frame = cv2.flip(frame, 1) # 反転
    image_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #グレイスケール化
    # 顔を抽出
    facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=1, minSize=(50,50))

    face_position=(0,0,0,0)
    
    if len(facerect) > 0:
        for rect in facerect:
            if(face_position[2]<rect[2]):
                face_position = rect
        (x, y, w, h) = face_position

        if calib_flag:
            postures.append((x, y, w, h))
        if good_posture is not None:
            line_y = int(good_posture[1]+good_posture[3]/2) + th_posture
            writer.writerow([time.time(), x+w//2, capture_height - (y+h//2), capture_height - line_y]) # 解釈しやすい座標系にして記録
            cv2.line(frame, (0, line_y), (capture_width, line_y), (0, 255, 0), thickness=2)
            # 姿勢が悪くなったか判定
            if (y+h//2) - (good_posture[1]+good_posture[3]//2) > th_posture:
                bad_posture_count += 1
            else:
                good_posture_count += 1

        cv2.circle(frame,(x+w//2, y+h//2), 3, (0, 255, 0),thickness = 5)
    cv2.imshow('posture monitor', frame)

    key = cv2.waitKey(monitor_interval)
    if key == ord('q'): # 停止
        break
    elif key == ord('s'):
        print('Calibration started')
        print('Push "e" to end calibration')
        calib_flag = True
        good_posture = None
    elif key == ord('e') and calib_flag:
        print('Calibration ended')
        print('Keep good posture!')
        good_posture = np.array(postures).mean(axis=0).astype(np.int64)
        calib_flag = False
        postures = []

    if bad_posture_count >= th_count:
        set_brightness(brightness_dark) # 明るさを20に
        bad_posture_count = 0
        good_posture_count = 0
        bad_posture_flag = True
    
    if bad_posture_flag and good_posture_count >= th_count:
        set_brightness(brightness_bright) # 明るさを100に
        bad_posture_count = 0
        good_posture_count = 0
        bad_posture_flag = False

f.close()
capture.release()
cv2.destroyAllWindows()
