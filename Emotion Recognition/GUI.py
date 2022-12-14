import tkinter as tk
import cv2  # opencv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.utils import img_to_array
from keras.models import load_model
import psycopg2
import time


def classify():
    path_ = path.get("1.0", "end-1c")
    if (path_ == 'camera'):
        path_ = 0
    cap = cv2.VideoCapture(path_)
    if (not cap.isOpened()):
        done_classify.configure(text='Wrong path')
        return
    i = 0
    j = 0
    st = time.time()
    ft = st+60
    while time.time() < ft:
        ret, test_img = cap.read()  # captures frame and returns boolean value and captured image
        if not ret:
            break
        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
        predicted_emotion = 'None'
        for (x, y, w, h) in faces_detected:
            cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)

            roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_pixels = img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255
            converted = tf.image.rgb_to_grayscale(img_pixels)
            predictions = model.predict(converted)
            # print(predictions, np.argmax(predictions))
            # find max indexed array
            max_index = np.argmax(predictions[0])

            emotions = ('happy', 'sad', 'neutral')
            predicted_emotion = emotions[max_index]

            cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        resized_img = cv2.resize(test_img, (1000, 700))
        cv2.imshow('Emotion Recognition ', resized_img)
        # cv2.waitKey(1)
        i = i+1
        if (i > j*24):
            data.append(predicted_emotion)
            j = j+1

        if cv2.waitKey(1) == ord('q'):  # wait until 'q' key is pressed
            break

    cap.release()
    cv2.destroyAllWindows()
    done_classify.configure(text='Done Classification')
    print(data[1:])


def log():
    password = passw.get("1.0", "end-1c")
    name = namet.get("1.0", "end-1c")

    try:
        conn = psycopg2.connect(dbname='logEmotions',
                                user='postgres',
                                password=password,
                                port="5432",
                                host="127.0.0.1")
        log_state.configure(text='')
    except:
        log_state.configure(text='Wrong password')
        return

    datastr = ''
    for i in range(1, len(data)):
        datastr = datastr+data[i]+', '
    cur = conn.cursor()

    sql = f"INSERT INTO logs(name,log) Values('{name}','{datastr}')"
    cur.execute(sql)
    conn.commit()
    cur.close()
    conn.close()
    log_state.configure(text='Done Logging')
    root.quit()


root = tk.Tk()
root.title('Facial Expression Recognition')
root.resizable(False, False)
tit = tk.Label(root, text='Facial Expression Recognition', padx=25, pady=6, font=("", 12)).pack()
# canvas = tk.Canvas(root, height=500, width=500, bg='grey')
# canvas.pack()
path_label = tk.Label(root, text='Enter the path here').pack()

path = tk.Text(root, padx=35, pady=10, height=2, width=52)
path.pack()
classify_video = tk.Button(root, text='Start Classification',
                           padx=35, pady=10,
                           fg="white", bg="grey", command=classify)
classify_video.pack()
# class_image.pack(side=tk.)

done_classify = tk.Label(root, text='')
done_classify.pack()

model = load_model('final.h5')
model.summary()

data = []

face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

path_labe = tk.Label(root, text='Enter the database password').pack()
passw = tk.Text(root, padx=35, pady=10, height=2, width=52)
passw.pack()

path_labe = tk.Label(root, text='Enter the person name').pack()
namet = tk.Text(root, padx=35, pady=10, height=2, width=52)
namet.pack()

log_data = tk.Button(root, text='Log the data',
                     padx=35, pady=10,
                     fg="white", bg="grey", command=log)
log_data.pack()

log_state = tk.Label(root, text='')
log_state.pack()

root.mainloop()
