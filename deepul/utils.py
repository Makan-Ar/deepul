import os
from os.path import join, dirname, exists, realpath
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
from torchvision.utils import make_grid
import git


def savefig(fname, show_figure=True):
    if not exists(dirname(fname)):
        os.makedirs(dirname(fname))
    plt.savefig(fname)
    if show_figure:
        plt.show()


def save_training_plot(train_losses, test_losses, title, fname):
    plt.figure()
    plt.plot(train_losses, label='train loss')
    plt.plot(test_losses, label='test loss')
    plt.legend()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('NLL')
    savefig(fname)


def save_distribution_1d(data, distribution, title, fname):
    d = len(distribution)

    plt.figure()
    plt.hist(data, bins=np.arange(d) - 0.5, label='train data', density=True)

    x = np.linspace(-0.5, d - 0.5, 1000)
    y = distribution.repeat(1000 // d)
    plt.plot(x, y, label='learned distribution')

    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('Probability')
    plt.legend()
    savefig(fname)


def save_distribution_2d(true_dist, learned_dist, fname):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    ax1.imshow(true_dist)
    ax1.set_title('True Distribution')
    ax1.axis('off')
    ax2.imshow(learned_dist)
    ax2.set_title('Learned Distribution')
    ax2.axis('off')
    savefig(fname)


def show_samples(samples, fname=None, nrow=10, title='Samples'):
    samples = (torch.FloatTensor(samples) / 255).permute(0, 3, 1, 2)
    grid_img = make_grid(samples, nrow=nrow)
    plt.figure()
    plt.title(title)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis('off')

    if fname is not None:
        savefig(fname)
    else:
        plt.show()


def load_pickled_data(fname):
    with open(join('data', fname), 'rb') as f:
        data = pickle.load(f)
    return data['train'], data['test']


def get_git_root_path():
    git_repo = git.Repo(realpath(__file__), search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return git_root


def get_data_dir(hw_number):
    git_root = get_git_root_path()
    return join(git_root, 'homeworks', f'hw{hw_number}', 'data')