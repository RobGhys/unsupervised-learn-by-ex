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
    samples: np.ndarray, fname: str = None, nrow: int = 10, title: str = "Samples"
):
    import torch
    from torchvision.utils import make_grid

    samples = (torch.FloatTensor(samples) / 255).permute(0, 3, 1, 2)
    grid_img = make_grid(samples, nrow=nrow)
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
        train_data, test_data = load_pickled_data(join(data_dir, "shapes.pkl"))
        name = "Shape"
    elif dset_type == 2:
        train_data, test_data = load_pickled_data(join(data_dir, "mnist.pkl"))
        name = "MNIST"
    else:
        raise Exception("Invalid dset type:", dset_type)

    idxs = np.random.choice(len(train_data), replace=False, size=(100,))
    images = train_data[idxs].astype("float32") / 1 * 255
    show_samples(images, title=f"{name} Samples")