import numpy as np

from keras import models
from keras import layers
from keras import callbacks
from generate_data import generate_all_data, normalize_input


def build_model(N_input, optimizer):
    """
    Build a model that expects data of dimension N_input , that should be
    optimized with specified algorithm.

    Input:
      N_input   : dimension of the input
      optimizer : algorithm for the training
    Output:
      model : Keras model object

    """

    # define the model as a sequence of layers
    model = models.Sequential()
    # first layer connected to the input
    model.add(layers.Dense(128, activation="relu", input_shape=(N_input,)))
    # hidden layer
    model.add(layers.Dense(64, activation="relu"))
    # hidden layer
    model.add(layers.Dense(32, activation="relu"))
    # final layer conected to the output
    model.add(layers.Dense(1))
    # mse = mean squared error
    # mae = mean average error
    # loss : what will be minimize
    # metrics : what will be shown
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


def train_stage(
    index,
    model,
    folder,
    name,
    train_dat,
    val_dat,
    N_input,
    epochs,
    batch_size,
    weightsfname="",
):
    """
    Perform one stage of the learning process.

    Input :
      index  : number of learning stage
      model  : Keras model to be optimized
      folder : folder name where results should be stored
      name   : extra string to include in filenames
      train_dat : training data
      val_dat   : validation data
      N_input   : dimension of the data
      epochs    : training time
      batch_size: size of the batches for the updates
      weightsfname :  filename where to find model weights to start from
                      IF this is an emptry string, take the weights as in the
                      model already.

    Output:
      history: Keras history object detailing the training step
      lastfname: filename of the model weights at the end of the training
      bestfname: filename of the model weights judged best during the training

    Side-effects:
      * model is now trained
      * two weight-files are saved
         bestfname : best model (as judged by val_mae) from this training step
         lastfname : model at the end of this training step

    """

    bestfname = folder + "best.%s.stage=%d.weights.h5" % (name, index)
    lastfname = folder + "last.%s.stage=%d.weights.h5" % (name, index)

    if weightsfname != "":
        model.load_weights(weightsfname)

    model_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=bestfname,
        save_weights_only=True,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
    )

    history = model.fit(
        train_dat[:, :N_input],
        # Input data, i.e. N_input columns
        train_dat[:, N_input],
        # Observable to train on, the binding energy
        epochs=epochs,
        batch_size=batch_size,
        # Parameters controlling the duration of training and how
        # many samples to take per epoch
        verbose=2,
        # Show us stuff
        validation_data=(val_dat[:, :N_input], val_dat[:, N_input]),
        # Validation data
        callbacks=[model_checkpoint_callback],
    )
    # Pass in a callback option

    model.save_weights(lastfname)

    return history, lastfname, bestfname


def train_model(model, folder, name, train_dat, val_dat, N_input):
    """
    Perform three training steps on our model.
      1. Rough training with prespecified optimizer in the model (rmsprop)
         and large (=32) batchsize.
      2. Less rough training with same optimizer, smaller batchsize (=16).
         This step starts from the last model obtained in the previous step.
      3. Fine-grained training with 'adagrad' optimizer for very small
         batchsize (=4). This step starts from the best model obtained in step
         2, NOT the last one.

      Input:
        model    : Keras model object
        folder   : foldername to save results
        train_dat : training data
        val_dat   : validation data
        N_input   : dimension of the data

      Output:
        history1/2/3: Keras History objects of all three steps
    """

    history_1, lastfname, bestfname = train_stage(
        1, model, folder, name, train_dat, val_dat, N_input, 1000, 32
    )

    history_2, lastfname, bestfname = train_stage(
        2,
        model,
        folder,
        name,
        train_dat,
        val_dat,
        N_input,
        500,
        16,
        weightsfname=lastfname,
    )

    model = build_model(N_input, "adagrad")

    history_3, lastfname, bestfname = train_stage(
        3,
        model,
        folder,
        name,
        train_dat,
        val_dat,
        N_input,
        500,
        4,
        weightsfname=bestfname,
    )

    return history_1, history_2, history_3
