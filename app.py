import os
from flask import Flask, jsonify, request, render_template
from PIL import Image
import numpy as np
from tensorflow import keras

app = Flask(__name__)

model = keras.models.load_model('model')

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    image_directory = 'images'
    os.makedirs(image_directory, exist_ok=True) 
    image_path = os.path.join(image_directory, 'receiver_image.jpg') 

    opened_image = Image.open(image_file)
    opened_image.save(image_path, format='JPEG')

    preprocessed_image = opened_image.resize((150, 150))
    preprocessed_image = np.array(preprocessed_image) / 255.0
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

    prediction = model.predict(preprocessed_image)
    class_index = np.argmax(prediction)

    class_labels = ['antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 'bison', 'boar', 'butterfly',
                    'cat', 'caterpillar', 'chimpanzee', 'cockroach', 'cow', 'coyote', 'crab', 'crow',
                    'deer', 'dog', 'dolphin', 'donkey', 'dragonfly', 'duck',
                    'eagle', 'elephant',
                    'flamingo', 'fly', 'fox',
                    'goat', 'goldfish', 'goose', 'gorilla', 'grasshopper',
                    'hamster', 'hare', 'hedgehog', 'hippopotamus', 'hornbill','horse','hummingbird', 'hyena',
                    'jellyfish',
                    'kangaroo', 'koala', 'ladybugs', 'leopard', 'lion', 'lizard', 'lobster',
                    'mosquito', 'moth', 'mouse', 'octopus', 'okapi', 'orangutan', 'otter', 'owl', 'ox', 'oyster',
                    'panda', 'parrot', 'pelecaniformes', 'penguin','pig','pigeon' 'porcupine', 'possum',
                    'raccoon', 'rat', 'reindeer', 'rhinoceros',
                    'sandpiper', 'seahorse', 'seal', 'shark', 'sheep', 'snake', 'sparrow', 'squid','squirrel', 'starfish', 'swan',
                    'tiger', 'turkey', 'turtle', 'whale', 'wolf', 'wombat', 'woodpecker', 'zebra']
    predicted_class = class_labels[class_index]

    opened_image.close()

    os.remove(image_path)

    return render_template('index.html', prediction=predicted_class)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
