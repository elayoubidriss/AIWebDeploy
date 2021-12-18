# Prevent ImportErrors w/ flask
import werkzeug
werkzeug.cached_property = werkzeug.utils.cached_property
from werkzeug.datastructures import FileStorage
# ML/Data processing
import tensorflow.keras as keras
# RESTful API packages
import flask.scaffold
flask.helpers._endpoint_from_view_func = flask.scaffold._endpoint_from_view_func
from flask_restplus import Api, Resource
from flask import Flask
from PIL import Image
import numpy as np

app = Flask(__name__)

api = Api(app, version="1.0", title="CIFAR10 API",
description="Identifying Animals And Means of Transport via Deep Learning")
ns = api.namespace("Classification", description="Classifies the image by the AI."
)

# Use Flask-RESTPlus argparser to process user-uploaded images
arg_parser = api.parser()
arg_parser.add_argument('image', location='files',
                        type=FileStorage, required=True)

# Model reconstruction
model = keras.models.load_model("cnn_model.h5")

#Utility functions
def get_image(arg_parser):
    '''Returns a Pillow Image given the uploaded image.'''
    args = arg_parser.parse_args()
    image_file = args.image  # reading args from file
    return Image.open(image_file)  # open the image

def predict(model, image):
    # predict on the image data - pull out the probabilities for each class
    prediction_probabilities = model(image, training=False)[0]
    # get the prediction label
    index_highest_proba = np.argmax(prediction_probabilities)
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    label = str(class_names[index_highest_proba])
    # get the prediction probability
    confidence = float(prediction_probabilities[index_highest_proba])
    # return the output as a JSON string
    output = {
        "label": label,
        "confidence": confidence
    }
    return output

# Add the route to run inference
@ns.route("/prediction")
class CNNPrediction(Resource):
    """Takes in the image, to pass to the CNN"""
    @api.doc(parser=arg_parser,
             description="Let the AI classify the image")
    def post(self):
        # A: get the image
        image = get_image(arg_parser)
        data = np.asarray(image)
        data = data / 255.0
        data = data[np.newaxis, ...]
        # B: make the prediction
        prediction = predict(model, data)
        # return the classification
        return prediction


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)