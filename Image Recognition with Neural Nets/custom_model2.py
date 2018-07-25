# part (d)
import NN_Implementation
import numpy as np
import NN_model
from PIL import Image


def model_evaluation(A):
    """

    :param pic: np array of dimensions 120 x 128 representing an image
    :return: String specifying emotion the subject is feeling
    """
    # train acc = 0.64 test acc = 0.37
    nn3 = NN_Implementation.NeuralNetMLP(n_output=4,
                                        n_features=960,
                                        n_hidden=70,
                                        l2=0.2,
                                        l1=0.0,
                                        epochs=1000,
                                        eta=0.01,
                                        alpha=0.000001,
                                        decrease_const=0.00001,
                                        minibatches=40,
                                        shuffle=True,
                                        random_state=1)
    nn3.w1 = np.load('w1_emotion_highest.npy')  # Loading maximum accuracy weights
    nn3.w2 = np.load('w2_emotion_highest.npy')
    A_flat = np.array([NN_model.compress_image(NN_model.compress_image(np.array(A))).flatten()])
    Prediction = nn3.predict(A_flat)
    example_output = ""
    if Prediction == 0:
        example_output = "angry"
    if Prediction == 1:
        example_output = "happy"
    if Prediction == 2:
        example_output = "neutral"
    if Prediction == 3:
        example_output = "sad"
    return example_output
