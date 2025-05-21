import matplotlib.pyplot as plt
import numpy as np


def get_model_loss(folder):

    loss = np.loadtxt(folder + "loss.%d.dat" % 1)
    val_loss = np.loadtxt(folder + "val_loss.%d.dat" % 1)

    for k in range(2, 4):

        dat = np.loadtxt(folder + "loss.%d.dat" % k)
        loss = np.append(loss, dat)

        dat = np.loadtxt(folder + "val_loss.%d.dat" % k)
        val_loss = np.append(val_loss, dat)

    return loss, val_loss


# ---------------------------------------------------------------------------------------------------------
# Plot the traning loss and validation loss of a particular set of models as a function of training epoch.
fig, ax = plt.subplots(2, figsize=(5, 8))
plt.subplots_adjust(hspace=0.01, wspace=0.01)

folder = "simplex2/"
for k, folder in enumerate([folder]):

    loss, val_loss = get_model_loss(folder)
    ax[0].plot(loss, label=str(k + 1))
    ax[1].plot(val_loss)

ax[0].legend(loc="upper right", title="Model")
ax[0].set_xticklabels([])
ax[1].set_xlabel("Training epochs")

ax[0].set_ylabel("Train. loss  (MeV$^2$)")
ax[1].set_ylabel("Val. loss  (MeV$^2$)")

ax[0].set_yscale("log")
ax[1].set_yscale("log")

plt.savefig(f"{folder}loss.pdf", bbox_inches="tight")
# --------------------------------------------
