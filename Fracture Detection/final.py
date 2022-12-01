import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import cv2  # openCV for images
import numpy as np
from keras.models import load_model
import psycopg2
from datetime import datetime


def load_img():
    global img, image_data
    for img_display in frame.winfo_children():
        img_display.destroy()

    image_data = filedialog.askopenfilename(initialdir="/", title="Choose an image",
                                            filetypes=(("all files", "*.*"), ("png files", "*.png")))
    basewidth = 150
    img = Image.open(image_data)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    file_name = image_data.split('/')
    panel = tk.Label(frame, text=str(file_name[len(file_name)-1]).upper()).pack()
    panel_image = tk.Label(frame, image=img).pack()


def classify():
    original = cv2.imread(image_data)/255
    print(original)
    print(original.shape)
    original = cv2.resize(original, (256, 256))
    # numpy_image = img_to_array(original)
    numpy_image = original
    image_batch = np.expand_dims(numpy_image, axis=0)
    print(image_batch.shape)
    predictions = model.predict(image_batch)
    text = ''
    print(predictions)
    if (predictions > 0.5):
        data = 'Positive'
        text = f'Positive: there is a fracture, reliability {100*predictions[0][0]:.2f}%'
    else:
        data = 'Negative'
        text = f'Negative: there is not a fracture, reliability {100-100*predictions[0][0]:.2f}%'

    prediction.configure(text=text)


def enter():
    password = passw.get("1.0", "end-1c")
    name = namet.get("1.0", "end-1c")

    try:
        conn = psycopg2.connect(dbname='fractures',
                                user='postgres',
                                password=password,
                                port="5432",
                                host="127.0.0.1")
        log_state.configure(text='')
    except:
        log_state.configure(text='You Entered the Wrong password, please retry')
        return

    cur = conn.cursor()

    sql = f"INSERT INTO history(name,fracture,date) Values('{name}','{data}','{str(datetime.now().date())}')"
    cur.execute(sql)
    conn.commit()
    cur.close()
    conn.close()
    log_state.configure(text='Done Logging')


root = tk.Tk()
root.title('Fracture Detection')
root.resizable(False, False)
tit = tk.Label(root, text='Fracture Detection', padx=25, pady=6, font=("", 12)).pack()

frame = tk.Frame(root, bg='white')
frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)
frame.pack()

chose_image = tk.Button(root, text='Choose Image',
                        padx=35, pady=10,
                        fg="white", bg="grey", command=load_img)
chose_image.pack()

classify_image = tk.Button(root, text='Start Classification',
                           padx=35, pady=10,
                           fg="white", bg="grey", command=classify)
classify_image.pack()


prediction = tk.Label(root, text='')
prediction.pack()

model = load_model('final.h5')
model.summary()

data = 'positive'

path_labe = tk.Label(root, text='Patient name').pack()
namet = tk.Text(root, padx=35, pady=10, height=2, width=52)
namet.pack()


path_labe = tk.Label(root, text='Database password').pack()
passw = tk.Text(root, padx=35, pady=10, height=2, width=52)
passw.pack()


enter_data = tk.Button(root, text='Log the data',
                       padx=35, pady=10,
                       fg="white", bg="grey", command=enter)
enter_data.pack()

log_state = tk.Label(root, text='')
log_state.pack()

root.mainloop()
