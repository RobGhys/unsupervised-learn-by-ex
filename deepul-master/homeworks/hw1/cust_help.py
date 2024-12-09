from os.path import dirname, exists, join
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os 


def get_data_dir(hw_number: int):
    return join('deepul', 'homeworks', f'hw{hw_number}', 'data')

def load_pickled_data(fname: str, include_labels: bool = False):
    with open(fname, "rb") as f:
        data = pickle.load(f)

    train_data, test_data = data["train"], data["test"]
    if "mnist.pkl" in fname or "shapes.pkl" in fname:
        # Binarize MNIST and shapes dataset
        train_data = (train_data > 127.5).astype("uint8")
        test_data = (test_data > 127.5).astype("uint8")
    if "celeb.pkl" in fname:
        train_data = train_data[:, :, :, [2, 1, 0]]
        test_data = test_data[:, :, :, [2, 1, 0]]
    if include_labels:
        return train_data, test_data, data["train_labels"], data["test_labels"]
    return train_data, test_data

def savefig(fname: str, show_figure: bool = True) -> None:
    if not exists(dirname(fname)):
        os.makedirs(dirname(fname))
    plt.tight_layout()
    plt.savefig(fname)
    if show_figure:
        plt.show()


def show_samples(
    samples: np.ndarray, fname: str = None, nrow: int = 10, title: str = "Samples" # type: ignore
):
    import torch
    from torchvision.utils import make_grid

    samples = (torch.FloatTensor(samples) / 255).permute(0, 3, 1, 2) # type: ignore
    grid_img = make_grid(samples, nrow=nrow) # type: ignore
    plt.figure()
    plt.title(title)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis("off")

    if fname is not None:
        savefig(fname)
    else:
        plt.show()


def visualize_q2a_data_custom(dset_type):
    data_dir = '/home/rob/Documents/Github/unsupervised-learn-by-ex/deepul-master/homeworks/hw1/data'
    if dset_type == 1:
        train_data, test_data = load_pickled_data(join(data_dir, "shapes.pkl")) # type: ignore
        name = "Shape"
    elif dset_type == 2:
        train_data, test_data = load_pickled_data(join(data_dir, "mnist.pkl")) # type: ignore
        name = "MNIST"
    else:
        raise Exception("Invalid dset type:", dset_type)

    idxs = np.random.choice(len(train_data), replace=False, size=(100,))
    images = train_data[idxs].astype("float32") / 1 * 255
    show_samples(images, title=f"{name} Samples")


def q2a_save_results_cust(dset_type, q3_a):
    data_dir = '/home/rob/Documents/Github/unsupervised-learn-by-ex/deepul-master/homeworks/hw1/data'
    if dset_type == 1:
        train_data, test_data = load_pickled_data(join(data_dir, "shapes.pkl"))
        img_shape = (20, 20)
    elif dset_type == 2:
        train_data, test_data = load_pickled_data(join(data_dir, "mnist.pkl"))
        img_shape = (28, 28)
    else:
        raise Exception()

    train_losses, test_losses, samples = q3_a(
        train_data, test_data, img_shape, dset_type
    )
    samples = samples.astype("float32") * 255

    print(f"Final Test Loss: {test_losses[-1]:.4f}")
    save_training_plot(
        train_losses,
        test_losses,
        f"Q2(a) Dataset {dset_type} Train Plot",
        f"results/q2_a_dset{dset_type}_train_plot.png",
    )
    show_samples(samples, f"results/q2_a_dset{dset_type}_samples.png")

def save_training_plot(
    train_losses: np.ndarray, test_losses: np.ndarray, title: str, fname: str
) -> None:
    plt.figure()
    n_epochs = len(test_losses) - 1
    x_train = np.linspace(0, n_epochs, len(train_losses))
    x_test = np.arange(n_epochs + 1)

    plt.plot(x_train, train_losses, label="train loss")
    plt.plot(x_test, test_losses, label="test loss")
    plt.legend()
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("NLL")
    savefig(fname)


def save_timing_plot(
    time_1: np.ndarray,
    time_2: np.ndarray,
    title: str,
    fname: str,
    time1_label: str,
    time2_label: str,
) -> None:
    plt.figure()

    plt.plot(time_1, label=time1_label)
    plt.plot(time_2, label=time2_label)
    plt.legend()
    plt.title(title)
    plt.xlabel("sample step")
    plt.ylabel("seconds")
    savefig(fname)