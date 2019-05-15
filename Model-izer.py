import keras
from keras import optimizers
from keras import losses
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense

from gensim.models import KeyedVectors
import re
import xlrd
import numpy

ROW_LEN = 500
COL_LEN = 300
NUM_SEN = 0

def stringList(line):
    return re.findall(r"\w+[']?\w+", line)

print("__CREATING WORD VECTOR MODEL..__")
filename = 'GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True, limit=500000)

#opening files for communist vs non-communist model inputs
ComFile = open('Taiwan_Democratic_Self.txt')
NonComFile = open('Control_Data.txt')

X_TRAIN_RAW = []
Y_TRAIN_RAW = []

print("__WRITING COMMUNIST FILES..__")
com_list = stringList(ComFile.read())
count = int(len(com_list)/5)

while count > 0:
    X_TRAIN_RAW.append(com_list[5*count:5*(count+1)])
    Y_TRAIN_RAW.append([1,0])
    count -= 1


print("__WRITING NON-COMMUNIST FILES..__")
non_com_list = stringList(NonComFile.read())
count = int(len(non_com_list)/5)

while count > 0:
    X_TRAIN_RAW.append(non_com_list[5*count:5*(count+1)])
    Y_TRAIN_RAW.append([0,1])
    count -= 1


print("__TRASNFORMING RAW DATA..__")
X_TRAIN = numpy.zeros((len(X_TRAIN_RAW),ROW_LEN,300))
Y_TRAIN = numpy.zeros((len(Y_TRAIN_RAW),2))

for i in range(len(X_TRAIN_RAW)):
    for j in range(ROW_LEN):
        try:
            X_TRAIN[i][j] = model[X_TRAIN_RAW[i][j]]
        except:
            pass

for i in range(len(Y_TRAIN_RAW)):
    Y_TRAIN[i] = Y_TRAIN_RAW[i]

print("__TRAINING MODEL..__")


kmodel = Sequential()
kmodel.add(Conv1D(64, 5, activation='relu'))
kmodel.add(Conv1D(64, 5, activation='relu'))
kmodel.add(MaxPooling1D(5))
kmodel.add(Dropout(.15))
kmodel.add(Flatten())
kmodel.add(Dense(64, activation='relu'))
kmodel.add(Dense(2, activation='softmax'))

kmodel.compile(loss=losses.binary_crossentropy,
                optimizer='RMSprop',
                metrics=['accuracy'])

kmodel.fit(X_TRAIN, Y_TRAIN,
          batch_size=64,
          epochs=8,
          verbose=2,
          validation_split = 0.2,
          shuffle = True)


kmodel.save('tai_model.h5')
