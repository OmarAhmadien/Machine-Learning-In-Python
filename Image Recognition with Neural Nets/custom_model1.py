# part (c)

import NN_model
import NN_Implementation
import numpy as np
def model_evaluation(A):
    """

    :param pic: np array of dimensions 120 x 128 representing an image
    :return: String specifying direction that the subject is facing
    """
    # Maximum Accuracy: train acc= 0.952380952381 test acc= 0.922222222222 >>  n_output=4,n_features=Training_Data.shape[1],n_hidden=40,l2=0.2,l1=0.0,epochs=2000,eta=0.01,alpha=0.0001,decrease_const=0.00001,minibatches=30,shuffle=True,random_state=1

    nn2 = NN_Implementation.NeuralNetMLP(n_output=4,
                                        n_features=960,
                                        n_hidden=40,
                                        l2=0.2,
                                        l1=0.0,
                                        epochs=2000,
                                        eta=0.01,
                                        alpha=0.0001,
                                        decrease_const=0.00001,
                                        minibatches=30,
                                        shuffle=True,
                                        random_state=1)
    nn2.w1 = np.load('w1_highest.npy') #Loading maximum accuracy weights
    nn2.w2 = np.load('w2_highest.npy')
    A_flat = np.array([NN_model.compress_image(NN_model.compress_image(np.array(A))).flatten()])
    Prediction = nn2.predict(A_flat)
    example_output = ""
    if Prediction == 0:
        example_output = "left"
    if Prediction == 1:
        example_output = "right"
    if Prediction == 2:
        example_output = "straight"
    if Prediction == 3:
        example_output = "up"
    return example_output

