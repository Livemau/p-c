import keras
from keras import optimizers
from keras import losses
from keras.models import Sequential
from keras.models import load_model
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

ROW_LEN = 250
COL_LEN = 300
NUM_SEN = 0

def stringList(line):
    return re.findall(r"\w+[']?\w+", line)

filename = 'GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True, limit=500000)

print("Input text: ")
string = input()

X_TRAIN_RAW = stringList(string)         
X_TRAIN = numpy.zeros((1,500,300))

for i in range(len(X_TRAIN_RAW)):
    for j in range(ROW_LEN):
        try:
            X_TRAIN[i][j] = model[X_TRAIN_RAW[i][j]]
        except:
            pass


#[0,1] = True    
cmodel = load_model("com_model.h5")
rmodel = load_model("rev_model.h5")
tmodel = load_model("tai_model.h5")

prediction1 = cmodel.predict(X_TRAIN)
prediction2 = rmodel.predict(X_TRAIN)
prediction3 = tmodel.predict(X_TRAIN)

comTag = False
revTag = False
taiTag = False

if prediction1[0][1] > prediction1[0][0]:
    comTag = True

if prediction2[0][1] > prediction2[0][0]:
    revTag = True

if prediction3[0][1] > prediction3[0][0]:
    taiTag = True

print("com: ", prediction1)
print("rev: ", prediction2)
print("tai: ", prediction3)

print("com: ", comTag)
print("rev: ", revTag)
print("tai: ", taiTag)
