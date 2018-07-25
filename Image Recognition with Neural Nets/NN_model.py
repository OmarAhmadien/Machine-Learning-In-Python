from __future__ import division
import numpy as np
import NN_Implementation
from PIL import Image
import glob
from sklearn.preprocessing import LabelEncoder

# part (a)
def compress_image(A):

    """

    :param pic: np array of dimensions M x N
    :return: np array of dimensions M/2 x N/2 using the formula specified in the assignment description
    """
    winSize = 2
    step = 2
    h = len(A)
    w = len(A[0])
    B = [[[] for o in range(w//2)] for o in range(h//2)]
    for i in range(0, h, step):
        for j in range(0, w, step):
            r = i // 2
            c = j // 2
            B[r][c].append(sum(sum(A[i:i + winSize,j:j + winSize])) / 4)
    return np.array(B)

"""""
labels_direction = ['up','straight','left','right']
labels_emotion = ['happy', 'sad', 'neutral', 'angry']
train_list = glob.glob('TrainingSet/*.jpg')

Training_Data = np.array([compress_image(compress_image(np.array(Image.open(fname)))).flatten() for fname in train_list])
y_train_direction = np.array(([direction for name in train_list for direction in name.split('_') if direction in labels_direction]))
y_train_emotion = np.array(([direction for name in train_list for direction in name.split('_') if direction in labels_emotion]))
test_list = glob.glob('TestSet/*.jpg')
Test_Data = np.array([compress_image(compress_image(np.array(Image.open(fname)))).flatten() for fname in test_list])
y_test_emotion = np.array(([direction for name in test_list for direction in name.split('_') if direction in labels_emotion]))
y_test_direction = np.array(([direction for name in test_list for direction in name.split('_') if direction in labels_direction]))

class_le = LabelEncoder()
train_Target_emotion = class_le.fit_transform(y_train_emotion)
train_Target_direction = class_le.fit_transform(y_train_direction)
test_Target_emotion = class_le.fit_transform(y_test_emotion)
test_Target_direction = class_le.fit_transform(y_test_direction)

nn = NN_Implementation.NeuralNetMLP(n_output=4,
                  n_features=Training_Data.shape[1],
                 n_hidden=70,
                  l2=0.2,
                  l1=0.0,
                  epochs=2000,
                  eta=0.01,
                  alpha=0.000001,
                  decrease_const=0.0001,
                  minibatches=70,
                  shuffle=True,
                  random_state=1)

nn.fit(Training_Data,train_Target_emotion, print_progress=True)
#Standard Accuracy for direction: train acc= 0.977777777778 test acc= 0.844444444444 >> standard as pdf
#Standard Accuracy for emotion: train acc = 0.21 test acc = 0.3 >> n_output=4,n_features=Training_Data.shape[1],n_hidden=70,l2=0.2,l1=0.0,epochs=2000,eta=0.01,alpha=0.000001,decrease_const=0.0001,minibatches=70,shuffle=True,random_state=1
train_Observed = nn.predict(Training_Data)
test_Observed = nn.predict(Test_Data)
trainAcc_direction=((np.sum(train_Target_direction == train_Observed, axis=0)).astype('float') / Training_Data.shape[0])
testAcc_direction = ((np.sum(test_Target_direction == test_Observed, axis=0)).astype('float') / Test_Data.shape[0])
trainAcc_emotion=((np.sum(train_Target_emotion == train_Observed, axis=0)).astype('float') / Training_Data.shape[0])
testAcc_emotion= ((np.sum(test_Target_emotion == test_Observed, axis=0)).astype('float') / Test_Data.shape[0])

np.save('w1_standar',nn.w1)

"""
# part (b)
def model_evaluation(A):
    nn = NN_Implementation.NeuralNetMLP(n_output=4,
                                        n_features=960,
                                        n_hidden=30,
                                        l2=0.1,
                                        l1=0.0,
                                        epochs=1000,
                                        eta=0.001,
                                        alpha=0.001,
                                        decrease_const=0.00001,
                                        minibatches=50,
                                        shuffle=True,
                                        random_state=1)
    nn.w1 = np.load('w1_Standar.npy')
    nn.w2 = np.load('w2_Standar.npy')
    # Standard Accuracy for direction: train acc= 0.977777777778 test acc= 0.844444444444 >> standard as pdf
    A_flat = np.array([compress_image(compress_image(np.array(A))).flatten()])
    Prediction = nn.predict(A_flat)
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
