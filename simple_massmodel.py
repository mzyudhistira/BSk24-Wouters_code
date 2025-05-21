# -------------------------------------------------------------------------------
# import all necessary tools
# a) external libraries
import numpy as np
from keras import models
from keras import layers
from keras import callbacks

# b) our own routines
from training_procedure import train_model, build_model
from generate_data import generate_all_data, normalize_input, select_input_normal, split

# ------------------------------------------------------------------------------
# Define a location for the model to be stored. Make sure this folder exists!
folder = "simplex0/"
label = "first_try"  # extra string to label the model in the folder
# -------------------------------------------------------------------------------
# Generate the data from the AME20 file
complete_dat, N_input = generate_all_data()
# -------------------------------------------------------------------------------
# Creating training, testing and validation data
N_train = 2000
N_test = 250
N_val = 250
# Use a 'normal' selection to determine this data
train_dat, test_dat, val_dat = select_input_normal(complete_dat, N_train, N_test, N_val)

# N_train, N_test, N_val = [0.8, 0.1, 0.1]
# train_dat, test_dat, val_dat = split(complete_dat, N_train, N_test, N_val)

#  ...... normalize it
normalize_input(train_dat, test_dat, val_dat, complete_dat, N_input)
# -------------------------------------------------------------------------------
# Actual machine learning starts here
model = build_model(N_input, "rmsprop")
model.summary()
# Do all of the heavy-lifting with our three-step training procedure
history1, history2, history3 = train_model(
    model, folder, label, train_dat, val_dat, N_input
)
# -------------------------------------------------------------------------------
# The work is done, we save the loss as a function of iteration
for k, h in enumerate([history1, history2, history3]):
    np.savetxt(folder + "loss.%d.dat" % (k + 1), h.history["loss"])
    np.savetxt(folder + "val_loss.%d.dat" % (k + 1), h.history["val_loss"])
