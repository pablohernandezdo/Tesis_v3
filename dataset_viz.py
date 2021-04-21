import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng


class VisualizerNpy:
    def __init__(self, dataset_path):

        self.dataset_path = dataset_path
        self.dataset_name = self.dataset_path.split("/")[-1]
        self.dset_extension = self.dataset_path.split(".")[-1]
        assert self.dset_extension == "npy", "Dataset format must be npy!"

        self.dataset = np.load(dataset_path)

    def plot_single_trace(self, idx_tr):

        assert(idx_tr < len(self.dataset)), "Invalid trace number!"

        savepath = f"Figures/{self.dataset_name}/Traces"

        if not os.path.exists(savepath):
            os.makedirs(savepath, exist_ok=True)

        plt.figure(figsize=(12, 9))
        plt.plot(self.dataset[idx_tr])
        plt.title(f'Dataset {self.dataset_name}, trace number {idx_tr}')
        plt.xlabel('Samples')
        plt.ylabel('[-]')
        plt.grid(True)
        plt.savefig(f"{savepath}/Trace_{idx_tr}.png")

    def plot_random_traces(self, n_tr):

        rng = default_rng()
        tr_ids = rng.choice(len(self.dataset), n_tr, replace=False)

        for i in tr_ids:
            self.plot_single_trace(i)

    def plot_single_boxplot(self, idx_tr):

        assert (idx_tr < len(self.dataset)), "Invalid trace number!"

        savepath = f"Figures/{self.dataset_name}/Boxplots"

        if not os.path.exists(savepath):
            os.makedirs(savepath, exist_ok=True)

        trace = self.dataset[idx_tr].reshape(10, -1)

        plt.figure(figsize=(12, 9))
        plt.boxplot(trace-T)
        plt.title(f'Dataset {self.dataset_name}, trace number {idx_tr} boxplot')
        plt.savefig(f"{savepath}/Boxplot_{idx_tr}.png")

    def plot_random_boxplot(self, n_tr):
        rng = default_rng()
        tr_ids = rng.choice(len(self.dataset), n_tr, replace=False)

        for i in tr_ids:
            self.plot_single_boxplot(i)


class VisualizerHDF5:
    def __init__(self, dataset_path):
        pass