import argparse
import base64
import cv2
import eventlet
import eventlet.wsgi
import json
import numpy as np
import socketio

from PIL import Image
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None


def resize_image(img):
    return cv2.resize(img[50:140, :, :], dsize = (200, 64), interpolation=cv2.INTER_AREA) 


@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    
    # The current throttle of the car
    throttle = data["throttle"]
    
    # The current speed of the car
    speed = data["speed"]
    
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    
    image_array = np.asarray(image)
    resized_image = resize_image(image_array)
    transformed_image_array = resized_image[None, :, :, :]
    # This model currently assumes that the features of the model are just the images. Feel free to change this.

    steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = 0.2
    if float(speed) < 10:
        throttle = 0.5
    
    print(steering_angle, throttle)
    send_control(steering_angle, throttle)
    

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
        'steering_angle': str(steering_angle),
        'throttle': str(throttle)
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        model = model_from_json(jfile.read())

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5') 
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
