print("Initializing. Please wait...")
import os
from flask import Flask, request, render_template, send_file
from keras.preprocessing import image
from PIL import Image 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np

model = tf.keras.models.load_model('./models/cifar10cnn.h5')

def predict_image(model, x):
  x = x.astype('float32')
  x = x / 255.0

  x = np.expand_dims(x, axis=0)

  image_predict = model.predict(x, verbose=0)
  print("Predicted Label: ", np.argmax(image_predict))
  return np.argmax(image_predict)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

@app.route('/success', methods=['POST'])
def success():
    global filename
    if request.method == "POST":
        f = request.files['file']
        if (f.filename != ""):
            f.save(f.filename)
            os.replace(f.filename, "./static/uploads/" + f.filename)
            filename = "./static/uploads/" + str(f.filename)
            imgproc = Image.open(filename).convert('RGB')
            width, height = imgproc.size
            new_size = (32,32)
            resized = imgproc.resize(new_size)
            resized.save('./static/compressed/compressed.jpg')
            img = tf.keras.preprocessing.image.load_img('./static/compressed/compressed.jpg', target_size=(32,32), color_mode = "grayscale")
            imgarray = tf.keras.preprocessing.image.img_to_array(img)
            if imgarray.shape == (32, 32, 1):
                reshaped = np.concatenate([imgarray] * 3, axis=-1)
                print(reshaped.shape)
                answer = predict_image(model, reshaped)
                print(type(answer))
                if answer == 0:
                    answer = "airplane"
                elif answer == 1:
                    answer = "car"
                elif answer == 2:
                    answer = "bird"
                elif answer == 3:
                    answer = "cat"
                elif answer == 4:
                    answer = "deer"
                elif answer == 5:
                    answer = "dog"
                elif answer == 6:
                    answer = "frog"
                elif answer == 7:
                    answer = "horse"
                elif answer == 8:
                    answer = "boat"
                elif answer == 9:
                    answer = "truck"
            return render_template("correct.html", answer = answer)
        else:
            return render_template("incorrect.html")

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/models/', methods=['GET'])
def modeldownload():
    try:
        return send_file('./models/cifar10cnn.h5')
    except Exception as e:
        return str(e)

print("READY.")
app.run(host='0.0.0.0')
