import os
import numpy as np
import cv2
import tensorflow as tf
import tensorflow_hub as hub

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename


# Define a flask app
app = Flask(__name__)

STATIC_FOLDER = 'static'
# Path to the folder where we'll store the upload before prediction
UPLOAD_FOLDER = STATIC_FOLDER + '/uploads'

labels_list=['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane', 'bicycle', 'boat', 'bus', 'car', \
 'motorbike', 'train', 'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tv']
 
class_pred_prob=[] 
total_len=0
np.set_printoptions(suppress=True)

def load__model():
    print('[INFO] : Model loading ................')
    model = tf.keras.models.load_model('MobileNetV2_model.h5', custom_objects={"KerasLayer": hub.KerasLayer})
    #model = tf.keras.models.load_model('model.h5')

    return model


model = load__model()
print('[INFO] : Model loaded ................')


def preprocessing_image(path):
    img=cv2.imread(path)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, size=[224, 224])
    img = np.expand_dims(img, axis=0)

    return img


def predict(model, fullpath):
    image = preprocessing_image(fullpath)
    pred = model.predict(image)

    return pred

def build_ypred_label_dic(ypred,labels_list):
  labels_with_prob={}
  for i in range(len(ypred)):
    label_name=labels_list[i]
    labels_with_prob[label_name]=ypred[i]
  return labels_with_prob
  
  
def sort_result(ypred,labels_list):
  ###### Mapping proabilities with their labels#########
  result_label_with_prob=build_ypred_label_dic(ypred[0],labels_list)
  sorted_result=sorted(result_label_with_prob.items(), key=lambda x: x[1], reverse=True)
  print(sorted_result)
  ######## Printing those class labels who have probability greater than 0.5###########
  output_l_p=''

  c=1
  for label,prob in sorted_result:
    prob=round(prob,2)
    if prob<0.5:
      break
    else:
      #print("class label:",label,", probability:",prob)
      output_l_p+=str(c)+")class label:"+label+",probability:"+str(prob)+" "
      #class_pred_prob.append(str(c)+")class label:"+label+",probability:"+str(prob)+" ")
      c+=1
  #print(output_l_p)
  #if len(class_pred_prob)==0:
      #class_pred_prob.append(['No output'])
  #total_len=len(class_pred_prob)
  return output_l_p

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        file = request.files['file']
        fullname = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(fullname)

        # Make prediction
        pred = predict(model, fullname)
        result = sort_result(pred,labels_list)
        if result=='':
            result='No class output'
        os.remove(fullname)
        return result
        
    return None


def predict_caption_gen(model,fullname_image):
    return None

@app.route('/predict_caption', methods=['GET', 'POST'])
def upload_caption():
    if request.method == 'POST':
        # Get the file from post request
        file = request.files['file']
        fullname = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(fullname)

        # Make prediction
        pred = predict_caption_gen(model, fullname)
        if result=='':
            result='No caption output'
        os.remove(fullname)
        return result
        
    return None


if __name__ == '__main__':
    app.run(debug=True)
