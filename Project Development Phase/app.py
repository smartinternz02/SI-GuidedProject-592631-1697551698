from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask,render_template,request
import os
import numpy as np

app = Flask(__name__)
# custom_objects = {'CustomLayerName': CustomLayerClass}
model = load_model("fakelogo.h5",compile = False)
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method=='POST':
        f = request.files['images']
        basepath=os.path.dirname(__file__)
        filepath = os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)

        
        img = image.load_img(filepath,target_size =(224,224))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis = 0)
        pred =np.argmax(model.predict(x),axis=1)
        index =['Bic', 'Samsung', 'Pepsi', 'Lays', 'Mars', 'MnM', 'Mtn dew', 'Oreo', 'Heinz', 'Marvel', 'PlayStation', 'Chevrolet', 'Burger King', 'Hp', 'Fila', 'Microsoft', 'Chrome', 'NASA', 'Reebok',
                'Oral b', 'Cowbell', 'Peak milk', 'Twitter', 'Google', 'Adidas', 'Android', 'Nutella', 'Puma', 'Pringles', 'Sprite' ,'Tesla', 'Netflix', 'Zara', 'Huawei', 'YouTube', 'Lipton', 'Nestle', 'Nescafe', 'Krisspy Kreme', 'Mtn'
                'Nestle', 'milo',  'Guinness' , 'Goya oil', 'Gillette', 'Starbucks', 'Nike',  'Dominos', 'McDonald',  'Coca Cola', 'Apple', 'Levis',  'Fila', 'Lg', 'Amazon', 'Ariel', 'Cadbury', 'Mercedes Benz', 'Rolex', 'Pepsi'
                 'Pay pal', 'Kfc', 'BMW', 'Honda', 'Instagram', 'Tiktok', 'Twitter']
        text="The LoGo is : " +str(index[pred[0]])
        return text    

if __name__=='__main__':
   app.run(host="0.0.0.0", port=8080,debug=True)
