import numpy as np

# Ignore some KERAS warnings
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


from keras import models
from keras import layers
from keras import callbacks

from generate_data import *
from training_procedure import build_model

# -------------------------------------------------------------------------------
# Generate the input data from file
complete_dat, N_input = generate_all_data()

folder = "simplex0/"
N_train = 2000
N_test = 250
N_val = 250


train_dat, test_dat, val_dat = select_input_normal(complete_dat, N_train, N_val, N_test)

# N_train, N_test, N_val = [0.8, 0.1, 0.1]
# train_dat, test_dat, val_dat = split(complete_dat, N_train, N_test, N_val)

Zarray = np.copy(complete_dat[:, 1])
Narray = np.copy(complete_dat[:, 0])

normalize_input(train_dat, test_dat, val_dat, complete_dat, N_input)


N_train = len(train_dat[:, 0])
N_val = len(val_dat[:, 0])
N_test = len(test_dat[:, 0])

# ----------------------------------------------------------
# Reinitialize the model
model = build_model(N_input, "adadelta")
# ... and load the weights corresponding to the best training point
model.load_weights(folder + "best.first_try.stage=3.weights.h5")
# and now ask for the model to make predictions for all nuclei!
y = model.predict(complete_dat[:, :N_input])

table = np.zeros((len(y), 5))
for k in range(len(y)):

    table[k, 0] = Zarray[k]
    table[k, 1] = Narray[k]
    table[k, 2] = complete_dat[k, -1]
    table[k, 3] = y[k, -1]
    table[k, 4] = complete_dat[k, -1] - y[k, -1]

table = table[np.lexsort((table[:, 1], table[:, 0]))]

np.savetxt(
    f"mass_tables/{folder[:-1]}.dat",
    table,
    fmt="%03d %03d %7.3f    %7.3f    %7.3f",
    header="Z  N   AME2020   Prediction   Difference",
)
