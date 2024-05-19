from flask import Flask, render_template, request
import pickle
import numpy as np
import os
from PIL import Image
from feature_extract import Texture_Extraction, Shape_Extraction, Color_Extraction

app = Flask(__name__)

model = pickle.load(open('static/model/model.pkl','rb'))


def Testing(im):
    tex = Texture_Extraction(im)
    sh = Shape_Extraction(im)
    col = Color_Extraction(im)
    fi = np.concatenate((tex,sh,col))
    fi=fi.reshape(1,-1)
    return fi

@app.route('/',methods=["GET", "POST"])
def index():
    if request.method == "GET":
        initial_file_image = 'images/white_bg.jpg'
        return render_template('index.html',full_filename = initial_file_image)
    
    if request.method == "POST":
        image_upload = request.files['image_upload']

        image = Image.open(image_upload)
        extracted_features = Testing(np.array(image))
        result = model.predict(extracted_features)[0]

        save_path = "static/uploads/temp_save.jpg"
        im1 = image.save(save_path)

        path = "dataset_full"
        label_list = os.listdir(path)
        
        return render_template('index.html', full_filename = "uploads/temp_save.jpg", pred = f"Selected Image is a {label_list[result]}")

if __name__ == "__main__":
    app.run(debug=True)