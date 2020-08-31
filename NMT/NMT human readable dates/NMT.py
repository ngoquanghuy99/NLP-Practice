from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np

from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
import matplotlib.pyplot as plt
%matplotlib inline

from utils import *
m = 10000
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)

dataset[:5]
print(human_vocab)
print('index of / is {}'.format(human_vocab['/']))
print(machine_vocab)
print('index of 9 is {}'.format(machine_vocab['9']))


Tx = 30
Ty = 10
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)
print(X.shape)
print(Y.shape)
print(Xoh.shape)
print(Yoh.shape)


print(X[0])
# example
index = 0
print("source date {}".format(dataset[index][0]))
print("target date {}".format(dataset[index][1]))

print("source afer processing {}".format(X[index]))
print("target after processing {}".format(Y[index]))

print("source after one hot encoding {}".format(Xoh[index]))
print("target after one hot encoding {}".format(Yoh[index]))


# implement one_step_attention
# just because of repeating this function Ty times so define these shared layers like globale variables
repeator = RepeatVector(Tx) # s<t-1> with Tx a<t'> with t' from 0 to Tx
concatenator = Concatenate(axis=-1)
densor1 = Dense(10, activation='tanh')
densor2 = Dense(1, activation='relu')
activator = Activation(softmax, name='attention_weghts')
dotor = Dot(axes=1)

def one_step_attention(a, s_prev):
    s_prev = repeator(s_prev)
    concat = concatenator([a, s_prev])
    e = densor1(concat)
    energies = densor2(e)
    alphas = activator(energies)
    context = dotor([alphas, a])
    return context


# implement model

n_a = 32
n_s = 64

post_activation_LSTM_cell = LSTM(n_s, return_sequences=True)
output_layer = Dense(len(machine_vocab), activation=softmax)

# GRADED FUNCTION: model

def model(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):

    # Define the inputs of the model with a shape (Tx,)
    # Define s0 (initial hidden state) and c0 (initial cell state)
    # for the decoder LSTM with shape (n_s,)
    X = Input(shape=(Tx, human_vocab_size))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0
    
    # Initialize empty list of outputs
    outputs = []
    
    # Define pre-attention Bi-LSTM
    a = Bidirectional(LSTM(n_a, return_sequences=True))(X)
    
    # Iterate for Ty steps
    for t in range(Ty):
    
        # Perform one step of the attention mechanism to get back the context vector at step t 
        context = one_step_attention(a, s)
        
        # Apply the post-attention LSTM cell to the "context" vector.
        # Don't forget to pass: initial_state = [hidden state, cell state] 
        s, _, c = post_activation_LSTM_cell(inputs = context, initial_state= [s, c])
        
        out = output_layer(s)
        
        outputs.append(out)
    
    # Create model instance taking three inputs and returning the list of outputs
    model = Model(inputs = [X, s0, c0], outputs = outputs)
    return model

# create model
model = model(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))
model.summary()

# compile model
opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# The list outputs[i][0], ..., outputs[i][Ty] represents the true labels (characters) corresponding to the  ithith  training example (X[i]).
# outputs[i][j] is the true label of the  jthjth  character in the  ithith  training example.

s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))
outputs = list(Yoh.swapaxes(0,1))

# fit the model
model.fit([Xoh, s0, c0], outputs, epochs=1, batch_size=100)

from keras.models import load_model
# test with a model trained before

model = load_model('models/model.h5')
# see the result on new examples
EXAMPLES = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018', 'March 3 2001', 'March 3rd 2001', '1 March 2001']
for example in EXAMPLES:
    
    source = string_to_int(example, Tx, human_vocab)
    source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source))).swapaxes(0,1)
    prediction = model.predict([source, s0, c0])
    prediction = np.argmax(prediction, axis = -1)
    output = [inv_machine_vocab[int(i)] for i in prediction]
    
    print("source:", example)
    print("output:", ''.join(output),"\n")
    
# attention map
attention_map = plot_attention_map(model, human_vocab, inv_machine_vocab, "Tuesday 09 Oct 1993", num = 7, n_s = 64)













    
    