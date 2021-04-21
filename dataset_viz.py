import os
import h5py
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
        plt.close()

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
        plt.boxplot(trace.T)
        plt.title(f'Dataset {self.dataset_name}, trace number {idx_tr} boxplot')
        plt.savefig(f"{savepath}/Boxplot_{idx_tr}.png")
        plt.close()

    def plot_random_boxplot(self, n_tr):
        rng = default_rng()
        tr_ids = rng.choice(len(self.dataset), n_tr, replace=False)

        for i in tr_ids:
            self.plot_single_boxplot(i)


class VisualizerHDF5:
    def __init__(self, dataset_path):

        self.dataset_path = dataset_path
        self.dataset_name = self.dataset_path.split("/")[-1]
        self.dset_extension = self.dataset_path.split(".")[-1]

        assert self.dset_extension in ["hdf5", "HDF5"],\
            "Dataset format must be HDF5!"

        self.dataset = h5py.File(self.dataset_path, "r")
        self.seis_grp = self.dataset["earthquake/local"]
        self.nonseis_grp = self.dataset["non_earthquake/noise"]

        self.len_seis = len(self.seis_grp)
        self.len_nonseis = len(self.nonseis_grp)

    def plot_single_trace(self, tr_type, idx_tr):

        assert tr_type in ["seismic", "non_seismic"], "Invalid trace type!"
        if tr_type == "seismic":
            assert idx_tr < self.len_seis, "Invalid trace number!"
            savepath = f"Figures/{self.dataset_name}/Traces/Seismic"
        else:
            assert idx_tr < self.len_nonseis, "Invalid trace number!"
            savepath = f"Figures/{self.dataset_name}/Traces/NonSeismic"

        if not os.path.exists(savepath):
            os.makedirs(savepath, exist_ok=True)

        if tr_type == "seismic":
            for i, tr in enumerate(self.seis_grp):
                if i == idx_tr:
                    plt.figure(figsize=(12, 9))
                    plt.plot(tr[:, 0])
                    plt.title(f'Dataset {self.dataset_name},'
                              f' trace number {idx_tr}')
                    plt.xlabel('Samples')
                    plt.ylabel('[-]')
                    plt.grid(True)
                    plt.savefig(f"{savepath}/Trace_{idx_tr}.png")
                    plt.close()

        else:
            for i, tr in enumerate(self.nonseis_grp):
                if i == idx_tr:
                    plt.figure(figsize=(12, 9))
                    plt.plot(tr[:])
                    plt.title(f'Dataset {self.dataset_name},'
                              f' trace number {idx_tr}')
                    plt.xlabel('Samples')
                    plt.ylabel('[-]')
                    plt.grid(True)
                    plt.savefig(f"{savepath}/Trace_{idx_tr}.png")
                    plt.close()

    def plot_random_traces(self, tr_type, n_tr):

        assert tr_type in ["seismic", "non_seismic"], "Invalid trace type!"

        rng = default_rng()
        if tr_type == "seismic":
            savepath = f"Figures/{self.dataset_name}/Traces/Seismic"
            tr_ids = rng.choice(self.len_seis, n_tr, replace=False)
        else:
            savepath = f"Figures/{self.dataset_name}/Traces/NonSeismic"
            tr_ids = rng.choice(self.len_nonseis, n_tr, replace=False)

        if not os.path.exists(savepath):
            os.makedirs(savepath, exist_ok=True)

        if tr_type == "seismic":
            for i, tr in enumerate(self.seis_grp):
                if i in tr_ids:
                    plt.figure(figsize=(12, 9))
                    plt.plot(tr[:, 0])
                    plt.title(f'Dataset {self.dataset_name},'
                              f' trace number {i}')
                    plt.xlabel('Samples')
                    plt.ylabel('[-]')
                    plt.grid(True)
                    plt.savefig(f"{savepath}/Trace_{i}.png")
                    plt.close()

        else:
            for i, tr in enumerate(self.nonseis_grp):
                if i in tr_ids:
                    plt.figure(figsize=(12, 9))
                    plt.plot(tr[:])
                    plt.title(f'Dataset {self.dataset_name},'
                              f' trace number {i}')
                    plt.xlabel('Samples')
                    plt.ylabel('[-]')
                    plt.grid(True)
                    plt.savefig(f"{savepath}/Trace_{i}.png")
                    plt.close()

    def plot_single_boxplot(self, tr_type, idx_tr):
        assert tr_type in ["seismic", "non_seismic"], "Invalid trace type!"
        if tr_type == "seismic":
            assert idx_tr < self.len_seis, "Invalid trace number!"
            savepath = f"Figures/{self.dataset_name}/Boxplot/Seismic"
        else:
            assert idx_tr < self.len_nonseis, "Invalid trace number!"
            savepath = f"Figures/{self.dataset_name}/Boxplot/NonSeismic"

        if not os.path.exists(savepath):
            os.makedirs(savepath, exist_ok=True)

        if tr_type == "seismic":
            for i, tr in enumerate(self.seis_grp):
                if i == idx_tr:
                    tr = tr[:, 0].reshape(10, -1)
                    plt.figure(figsize=(12, 9))
                    plt.plot(tr.T)
                    plt.title(f'Dataset {self.dataset_name},'
                              f' boxplot number {idx_tr}')
                    plt.savefig(f"{savepath}/Boxplot_{idx_tr}.png")
                    plt.close()

        else:
            for i, tr in enumerate(self.nonseis_grp):
                if i == idx_tr:
                    tr = tr[:].reshape(10, -1)
                    plt.figure(figsize=(12, 9))
                    plt.plot(tr.T)
                    plt.title(f'Dataset {self.dataset_name},'
                              f' boxplot number {idx_tr}')
                    plt.savefig(f"{savepath}/Boxplot_{idx_tr}.png")
                    plt.close()

    def plot_random_boxplot(self, tr_type, n_tr):

        assert tr_type in ["seismic", "non_seismic"], "Invalid trace type!"

        rng = default_rng()
        if tr_type == "seismic":
            savepath = f"Figures/{self.dataset_name}/Boxplot/Seismic"
            tr_ids = rng.choice(self.len_seis, n_tr, replace=False)
        else:
            savepath = f"Figures/{self.dataset_name}/Boxplot/NonSeismic"
            tr_ids = rng.choice(self.len_nonseis, n_tr, replace=False)

        if not os.path.exists(savepath):
            os.makedirs(savepath, exist_ok=True)

        if tr_type == "seismic":
            for i, tr in enumerate(self.seis_grp):
                if i in tr_ids:
                    tr = tr[:, 0].reshape(10, -1)
                    plt.figure(figsize=(12, 9))
                    plt.plot(tr.T)
                    plt.title(f'Dataset {self.dataset_name},'
                              f' boxplot number {i}')
                    plt.savefig(f"{savepath}/Boxplot_{i}.png")
                    plt.close()

        else:
            for i, tr in enumerate(self.nonseis_grp):
                if i in tr_ids:
                    tr = tr[:].reshape(10, -1)
                    plt.figure(figsize=(12, 9))
                    plt.plot(tr.T)
                    plt.title(f'Dataset {self.dataset_name},'
                              f' boxplot number {i}')
                    plt.xlabel('Samples')
                    plt.ylabel('[-]')
                    plt.grid(True)
                    plt.savefig(f"{savepath}/Boxplot_{i}.png")
                    plt.close()
