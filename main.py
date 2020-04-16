import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
#import pickle
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
import os
import numpy as np
import tensorflow as tf
# load and evaluate a saved model
from numpy import loadtxt
from keras.models import load_model
import matplotlib.pyplot as plt






##############################################
app = Flask(__name__)
############## MODEL LOADING ################
# load model
CVDmodel = tf.keras.models.load_model('Covid_Vgg.h5')
print("Loaded model from disk")
# summarize model.
#CVDmodel.summary()
#############################


@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')
    #return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    image_path=request.form.values('image_file','')
    input_to_model = image_path.load_img(image_path, target_size=(150, 150))
    plt.imshow(input_to_model)
    input_to_model = np.expand_dims(input_to_model, axis=0)
    result=CVDmodel.predict_classes(input_to_model)
    if result[0]==0:
        output='Covid +Ve'
        print("Detected: Covid +Ve")
    else:
     output='No Covid Symptoms'   
     print("Detected: Non-Covid")
     print(result)
     #plt.show()
    
    return render_template('home.html', prediction_text="Detected: ".format(output))


if __name__ == '__main__':
    app.run(debug=True)
    #app.run(host='0.0.0.0', port=8000)