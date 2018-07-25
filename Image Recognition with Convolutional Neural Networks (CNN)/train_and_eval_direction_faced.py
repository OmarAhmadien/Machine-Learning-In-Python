import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer,ReLU
import numpy as np
from PIL import Image
import glob
import theano
from sklearn.preprocessing import LabelEncoder
import theano.tensor as T
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
validation_list = glob.glob('ValidationSet/*.jpg')
Validation_Data = np.array([compress_image(compress_image(np.array(Image.open(fname)))).flatten() for fname in validation_list])
y_validation_emotion = np.array(([direction for name in validation_list for direction in name.split('_') if direction in labels_emotion]))
y_validation_direction = np.array(([direction for name in validation_list for direction in name.split('_') if direction in labels_direction]))

class_le = LabelEncoder()
y_train_emotion = class_le.fit_transform(y_train_emotion)
y_test_emotion = class_le.fit_transform(y_test_emotion)
y_validation_emotion = class_le.fit_transform(y_validation_emotion)

def shared(data):
    """Place the data into shared variables.  This allows Theano to copy
    the data to the GPU, if one is available.
    """
    shared_x = theano.shared(
        np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
    shared_y = theano.shared(
        np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
    return shared_x, T.cast(shared_y, "int32")

#train_data_direction = shared([Training_Data,y_train_direction])
#validation_data_direction = shared([Validation_Data,y_validation_direction])
#test_data_direction = shared([Test_Data,y_test_direction])
train_data_emotion = shared([Training_Data,y_train_emotion])
validation_data_emotion = shared([Validation_Data,y_validation_emotion])
test_data_emotion = shared([Test_Data,y_test_emotion])






mini_batch_size = 10

net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 30, 32),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2)),
        FullyConnectedLayer(n_in=20*13*14, n_out=100),
        SoftmaxLayer(n_in=100, n_out=4)], mini_batch_size)

net.SGD(train_data_emotion, 60, mini_batch_size, 0.01,
        validation_data_emotion, test_data_emotion)


