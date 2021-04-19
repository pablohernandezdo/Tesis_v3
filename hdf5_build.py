import os
import h5py
import json
import numpy as np


class H5Builder:
    """
    Tomar datasets en formato npy y construir un dataset HDF5, guardarlo
    """
    def __init__(self, config_file):

        # Load dataset config
        with open(config_file, "r") as config:
            self.cfg = json.load(config)

        if not os.path.exists(os.path.dirname(self.cfg["OutputFile"])):
            os.makedirs(os.path.dirname(self.cfg["OutputFile"]), exist_ok=True)

        # Create hdf5 dataset file
        self.h5 = h5py.File(self.cfg["OutputFile"], "w")

        # Create seismic and non-seismic groups
        g_earthquake = self.h5.create_group("earthquake/local")
        g_non_earthquake = self.h5.create_group("non_earthquake/noise")

        # For every dataset in config file
        for dset in self.cfg["Datasets"]:

            print(f"Loading {dset} dataset")
            dset_params = self.cfg["Datasets"][dset]

            # read npy datasets
            if dset_params["format"] == "npy":

                traces = np.load(dset_params["path"])
                if dset_params["type"] == "seismic":
                    for i, tr in enumerate(traces):
                        tr = np.expand_dims(tr, 1)
                        tr = np.hstack([tr] * 3).astype("float32")
                        g_earthquake.create_dataset(f"{dset}-{i}", data=tr)

                elif dset_params["type"] == "nonseismic":
                    for i, tr in enumerate(traces):
                        tr = np.expand_dims(tr, 1)
                        tr = np.hstack([tr] * 3).astype("float32")
                        g_non_earthquake.create_dataset(f"{dset}-{i}", data=tr)

                else:
                    print(f"Bad dataset type! Dataset: {dset}")

            # read h5py datasets
            elif dset_params["format"] == "hdf5":
                with h5py.File(dset_params["path"], "r") as h5_dset:

                    g_source_earthquake = h5_dset["earthquake/local"]
                    g_source_non_earthquake = h5_dset["non_earthquake/noise"]

                    # Copy seismic traces
                    for arr in g_source_earthquake:
                        data = g_source_earthquake[arr]
                        g_earthquake.copy(data, arr)

                    # Copy non seismic traces
                    for arr in g_source_non_earthquake:
                        data = g_source_non_earthquake[arr]
                        g_non_earthquake.copy(data, arr)

            else:
                print(f"Bad dataset format! Dataset: {dset}")

        self.h5.close()

    def copy_dataset(self, source, destiny):
        pass

    def create_dataset(self, source, destiny):
        pass

