# -------------------------------------------------------------------------------
# Import all data
import numpy as np
import pandas as pd
from keras import models
from keras import layers
from keras import callbacks


def read_df():
    df = pd.read_parquet("bsk24_variants_sample_mass_table.parquet")
    # df = pd.read_parquet("bsk24_variants_ext_sample_mass_table.parquet")
    selected_variant = 1

    df = df[df["varian_id"] == selected_variant]
    df = df[["Z", "N", "A", "m"]]
    df = df[~((df["Z"] == 81) & (df["N"] > 150))]

    return df.to_numpy()


def generate_all_data(NMN=[8, 20, 28, 50, 82, 126], PNM=[8, 20, 28, 50, 82, 126]):
    """
    Generate a complete set of nuclear data from the nuclear masses.
    Requires the file 'masses_ZN_shuf.dat' which contains in random order
        Z  N  BE(Z,N)
    This routine will generate a numpy array with 11 columns:

      0  1  2   3    4      5        6          7      8      9     10
      N  Z  PiN PiZ  A^2/3  Coul  (N-Z)**2/A  A^(-1/2)distN distZ  BE(Z,N)

    where
      Coul = Z * (Z-1)/(A**(1./3.))
      distN= distance to nearest neutron magic number, selected from input
             array NMN
      distZ= same but for proton magic numbers from PNM

    --------------------------------------------------------------------------

    Input :
      NMN, PNM: list of neutron and proton magic numbers
    Output:
      complete_dat : numpy array as described above
      N_input      : column dimension of complete_dat

    """

    # Size of the input, i.e. the number of parameters passed in to the MLNN for
    # any given input
    N_input = 10

    # Load the (randomized) data on nuclear masses
    # dat = np.loadtxt('masses_ZN_shuf.dat')
    dat = np.loadtxt("bsk24_mass_table.csv", delimiter=";", skiprows=1)
    # dat = read_df()
    # print(dat)

    # Creating a complete data set from the (N,Z,BE)-data the table
    # Note that this has N_input + 1 columns: we store the binding energy here too
    complete_dat = np.zeros((len(dat[:, 0]), N_input + 1))

    for i in range(len(dat[:, 0])):

        Z = dat[i, 0]
        N = dat[i, 1]
        A = N + Z

        # - - - - - - - - - - - - - - - - - -
        # Basic information
        complete_dat[i, 0] = N  # Neutron number N
        complete_dat[i, 1] = Z  # Proton number Z
        complete_dat[i, 2] = (-1) ** (N)  # Number parity of neutrons
        complete_dat[i, 3] = (-1) ** (Z)  #                  protons

        # - - - - - - - - - - - - - - - - - -
        # Liquid drop parameters
        complete_dat[i, 4] = (A) ** (2.0 / 3.0)  # A^2/3
        complete_dat[i, 5] = Z * (Z - 1) / (A ** (1.0 / 3.0))  # Coulomb term
        complete_dat[i, 6] = (N - Z) ** 2 / A  # Asymmetry
        complete_dat[i, 7] = A ** (-1.0 / 2.0)  # Pairing

        # - - - - - - - - - - - - - - - -
        # Distance to the next magic number for both protons and neutrons
        dist_N = 100000
        dist_Z = 100000
        for k in NMN:
            dist_Nb = abs(N - k)
            if dist_Nb < dist_N:
                dist_N = dist_Nb

        for k in PNM:
            dist_Zb = abs(Z - k)
            if dist_Zb < dist_Z:
                dist_Z = dist_Zb

        complete_dat[i, 8] = dist_N
        complete_dat[i, 9] = dist_Z
        # The binding energy
        complete_dat[i, 10] = dat[i, 3]

        # - - - - - - - - - - - - - - - -

    return complete_dat, N_input


def split(data, N_train, N_val, N_test):
    """
    Split a given dataset into training, validation, and test data

    Args:
        data (np array): Dataset to split
        N_train (float): Percentage of training data
        N_val (float): Percentage of validation data
        N_test (float): Percentage of test data

    Returns:
       data_train,  data_val, data_test (np array)
    """

    percentages = [N_train, N_val, N_test]
    total_length = len(data)

    indices = [
        int(total_length * sum(percentages[:i])) for i in range(1, len(percentages))
    ]
    split_arrays = np.split(data, indices)

    return split_arrays[0], split_arrays[1], split_arrays[2]


def select_input_normal(data, N_train, N_val, N_test):
    """
    Select from a complete set of data, a set of
      (1) training data
      (2) validation data
      (3) testing data

    If data is in random order, then all subsets of ddata will be too.

    Input:
      data   : array to chop up into parts
      N_train: number of training data
      N_val  : number of validation data
      N_test : number of testing data

    Output:
      train_dat : training data, first N_train rows in data
      test_dat  : testing data, last N_test rows in data
      val_dat   : validation data

    """

    val_dat = np.copy(data[-N_val - N_test : -N_test, :])
    test_dat = np.copy(data[-N_test:, :])
    train_dat = np.copy(data[:N_train, :])
    #
    return train_dat, test_dat, val_dat


def normalize_input(train_dat, test_dat, val_dat, complete_dat, N_input):
    """
    Normalize all input data, i.e. for all input columns do

      column = (column - mean)/std

    where mean is the mean over the column and std is the standard deviation.

    """

    for i in range(N_input):
        # Subtract mean
        mean = train_dat[:, i].mean(axis=0)
        train_dat[:, i] = train_dat[:, i] - mean
        test_dat[:, i] = test_dat[:, i] - mean
        val_dat[:, i] = val_dat[:, i] - mean
        complete_dat[:, i] = complete_dat[:, i] - mean

        # Divide by standard-deviation
        std = train_dat[:, i].std(axis=0)
        train_dat[:, i] = train_dat[:, i] / std
        test_dat[:, i] = test_dat[:, i] / std
        val_dat[:, i] = val_dat[:, i] / std
        complete_dat[:, i] = complete_dat[:, i] / std
