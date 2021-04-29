import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.random import default_rng


class VisualizerNpy:
    def __init__(self, dataset_path):

        self.dataset_path = dataset_path
        self.dataset_name = self.dataset_path.split("/")[-1]
        self.dset_extension = self.dataset_path.split(".")[-1]
        assert self.dset_extension == "npy", "Dataset format must be npy!"

        self.dataset = np.load(dataset_path)

    def plot_single_trace(self, idx_tr):

        assert (idx_tr < len(self.dataset)), "Invalid trace number!"

        savepath = f"Figures/Datasets/{self.dataset_name}/Traces"

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
        assert n_tr < len(self.dataset), "Too many traces!"
        tr_ids = rng.choice(len(self.dataset), n_tr, replace=False)

        for i in tr_ids:
            self.plot_single_trace(i)

    def plot_single_boxplot(self, idx_tr):

        assert (idx_tr < len(self.dataset)), "Invalid trace number!"

        savepath = f"Figures/Datasets/{self.dataset_name}/Boxplots"

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
        assert n_tr < len(self.dataset), "Too many boxplots!"
        tr_ids = rng.choice(len(self.dataset), n_tr, replace=False)

        for i in tr_ids:
            self.plot_single_boxplot(i)


class VisualizerHDF5:
    def __init__(self, dataset_path):

        self.dataset_path = dataset_path
        self.dataset_name = self.dataset_path.split("/")[-1]
        self.dset_extension = self.dataset_path.split(".")[-1]

        assert self.dset_extension in ["hdf5", "HDF5"], \
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
            savepath = f"Figures/Datasets/{self.dataset_name}/Traces/Seismic"
        else:
            assert idx_tr < self.len_nonseis, "Invalid trace number!"
            savepath = f"Figures/Datasets/{self.dataset_name}/Traces/NonSeismic"

        if not os.path.exists(savepath):
            os.makedirs(savepath, exist_ok=True)

        if tr_type == "seismic":
            for i, tr in enumerate(self.seis_grp):
                if i == idx_tr:
                    plt.figure(figsize=(12, 9))
                    plt.plot(self.seis_grp[tr][:])
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
                    plt.plot(self.nonseis_grp[tr][:])
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
            savepath = f"Figures/Datasets/{self.dataset_name}/Traces/Seismic"
            assert n_tr < self.len_seis, "Too many traces!"
            tr_ids = rng.choice(self.len_seis, n_tr, replace=False)
        else:
            savepath = f"Figures/Datasets/{self.dataset_name}/Traces/NonSeismic"
            assert n_tr < self.len_nonseis, "Too many boxplots!"
            tr_ids = rng.choice(self.len_nonseis, n_tr, replace=False)

        if not os.path.exists(savepath):
            os.makedirs(savepath, exist_ok=True)

        if tr_type == "seismic":
            for i, tr in enumerate(self.seis_grp):
                if i in tr_ids:
                    plt.figure(figsize=(12, 9))
                    plt.plot(self.seis_grp[tr][:])
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
                    plt.plot(self.nonseis_grp[tr][:])
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
            savepath = f"Figures/Datasets/{self.dataset_name}/Boxplot/Seismic"
        else:
            assert idx_tr < self.len_nonseis, "Invalid trace number!"
            savepath = f"Figures/Datasets/{self.dataset_name}/Boxplot/NonSeismic"

        if not os.path.exists(savepath):
            os.makedirs(savepath, exist_ok=True)

        if tr_type == "seismic":
            for i, tr in enumerate(self.seis_grp):
                if i == idx_tr:
                    tr = self.seis_grp[tr][:].reshape(10, -1)
                    plt.figure(figsize=(12, 9))
                    plt.boxplot(tr.T)
                    plt.title(f'Dataset {self.dataset_name},'
                              f' boxplot number {idx_tr}')
                    plt.savefig(f"{savepath}/Boxplot_{idx_tr}.png")
                    plt.close()

        else:
            for i, tr in enumerate(self.nonseis_grp):
                if i == idx_tr:
                    tr = self.nonseis_grp[tr][:].reshape(10, -1)
                    plt.figure(figsize=(12, 9))
                    plt.boxplot(tr.T)
                    plt.title(f'Dataset {self.dataset_name},'
                              f' boxplot number {idx_tr}')
                    plt.savefig(f"{savepath}/Boxplot_{idx_tr}.png")
                    plt.close()

    def plot_random_boxplot(self, tr_type, n_tr):

        assert tr_type in ["seismic", "non_seismic"], "Invalid trace type!"

        rng = default_rng()
        if tr_type == "seismic":
            savepath = f"Figures/Datasets/{self.dataset_name}/Boxplot/Seismic"
            assert n_tr < self.len_seis, "Too many boxplots!"
            tr_ids = rng.choice(self.len_seis, n_tr, replace=False)
        else:
            savepath = f"Figures/Datasets/{self.dataset_name}/Boxplot/NonSeismic"
            assert n_tr < self.len_nonseis, "Too many boxplots!"
            tr_ids = rng.choice(self.len_nonseis, n_tr, replace=False)

        if not os.path.exists(savepath):
            os.makedirs(savepath, exist_ok=True)

        if tr_type == "seismic":
            for i, tr in enumerate(self.seis_grp):
                if i in tr_ids:
                    tr = self.seis_grp[tr][:].reshape(10, -1)
                    plt.figure(figsize=(12, 9))
                    plt.boxplot(tr.T)
                    plt.title(f'Dataset {self.dataset_name},'
                              f' boxplot number {i}')
                    plt.savefig(f"{savepath}/Boxplot_{i}.png")
                    plt.close()

        else:
            for i, tr in enumerate(self.nonseis_grp):
                if i in tr_ids:
                    tr = self.nonseis_grp[tr][:].reshape(10, -1)
                    plt.figure(figsize=(12, 9))
                    plt.boxplot(tr.T)
                    plt.title(f'Dataset {self.dataset_name},'
                              f' boxplot number {i}')
                    plt.xlabel('Samples')
                    plt.ylabel('[-]')
                    plt.grid(True)
                    plt.savefig(f"{savepath}/Boxplot_{i}.png")
                    plt.close()


class VizUnprocessed:
    def __init__(self, dataset_path, clip=None):

        self.dataset_path = dataset_path
        self.dataset_name = self.dataset_path.split("/")[-1].split(".")[0]
        self.clip = clip

    def plot_heatmap(self):

        unproc = np.load(self.dataset_path)
        clip_traces = np.zeros(unproc.shape)

        if self.clip:
            clip_traces = np.clip(unproc, -self.clip, self.clip)

        savepath = f"Figures/Classification/Heatmaps/" \
                   f"{self.dataset_name}"
        os.makedirs(savepath, exist_ok=True)

        plt.figure(figsize=(15, 12))
        plt.imshow(clip_traces.T, cmap=plt.cm.seismic)
        plt.savefig(f"{savepath}/"
                    f"{self.dataset_name}_clip_{self.clip}.png")
        plt.close()

        # plt.imsave(f"{savepath}/"
        #            f"{self.dataset_name}_full_clip_{self.clip}.png",
        #            clip_traces.T, cmap=plt.cm.seismic)

    def plot_class_result(self, csv, model):

        savepath = f"Figures/Classification/Outputs/{model}"
        if not os.path.exists(savepath):
            os.makedirs(savepath, exist_ok=True)

        df = pd.read_csv(csv)

        # Create matrix of output values as columns
        extend = np.ones((len(df["out"]), 6000))
        output_matrix = df["out"].to_numpy().reshape(-1, 1) * extend

        plt.figure(figsize=(15, 12))
        plt.imshow(output_matrix.T, cmap=plt.cm.Greys)
        plt.colorbar()
        plt.title(f"Resultados clasificación dataset {self.dataset_name}")
        plt.xlabel('Canales')
        plt.ylabel('Valor de salida clasificación')
        plt.savefig(f"{savepath}/"
                    f"{self.dataset_name}"
                    f"_{model}_classification.png")
        plt.close()

        plt.imsave(f"{savepath}/{self.dataset_name}"
                   f"_{model}_full_classification.png",
                   output_matrix.T, cmap=plt.cm.Greys)

    def plot_class_heatmap(self, csv, model):

        # Clip traces
        unproc = np.load(self.dataset_path)
        clip_traces = np.zeros(unproc.shape)

        if self.clip:
            clip_traces = np.clip(unproc, -self.clip, self.clip)
            clip_traces = (clip_traces / self.clip).astype('uint8') * 255

        # Output_matrix
        df = pd.read_csv(csv)

        # Create matrix of output values as columns
        extend = np.ones((len(df["out"]), 6000))
        output_matrix = df["out"].to_numpy().reshape(-1, 1) * extend

        savepath = f"Figures/Classification/Heatmaps+Outputs/" \
                   f"{self.dataset_name}/{model}"
        os.makedirs(savepath, exist_ok=True)

        both = np.hstack([clip_traces, output_matrix * 255])

    def plot_class_heatmap_superimposed(self, csv, model):

        # Clip traces
        unproc = np.load(self.dataset_path)
        clip_traces = np.zeros(unproc.shape)

        if self.clip:
            clip_traces = np.clip(unproc, -self.clip, self.clip)
            clip_traces = (clip_traces / np.amax(np.abs(clip_traces))).astype('uint8') * 255

        # Output_matrix
        df = pd.read_csv(csv)

        # Create matrix of output values as columns
        extend = np.ones((len(df["out"]), len(clip_traces[0])))
        output_matrix = df["out"].to_numpy().reshape(-1, 1) * extend

        savepath = f"Figures/Classification/Superimposed/" \
                   f"{self.dataset_name}/{model}"
        os.makedirs(savepath, exist_ok=True)

        plt.figure(figsize=(15, 12))
        plt.imshow(clip_traces.T, cmap=plt.cm.hot)
        plt.colorbar()
        plt.imshow(output_matrix.T * 255, cmap=plt.cm.Greys, alpha=0.4)
        plt.title(f"Dataset {self.dataset_name},"
                  f" resultados modelo {model}")
        plt.xlabel("Canales")
        plt.ylabel("Muestras de cada señal")
        plt.savefig(f"{savepath}/{self.dataset_name}_{model}_"
                    f"superimposed.png", facecolor='white')
        plt.close()
