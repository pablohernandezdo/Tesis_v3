import os
import h5py
import json
import numpy as np
from numpy.random import default_rng


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


class H5Splitter:
    """
    Tomar un dataset hdf5 y dividirlo en conjuntos de entrenamiento, validacion
    y prueba según una proporción especificada

    Deberia hacer un shuffle de los datos primero, despues armar los datasets
    que haga falta
    """

    def __init__(self, dataset_path, ratios):

        self.dataset_path = dataset_path
        self.dataset_name = self.dataset_path.split("/")[-1]

        assert len(ratios) == 3, "Ratios must have 3 values!"

        self.train_ratio = ratios[0]
        self.val_ratio = ratios[1]
        self.test_ratio = ratios[2]

        # Cargar dataset de origen
        self.source_h5 = h5py.File(self.dataset_path, "r")
        self.source_seismic_group = self.source_h5["earthquake/local"]
        self.source_nonseismic_group = self.source_h5["non_earthquake/noise"]

        # SHUFFLE

        # Numero de trazas sismicas y no sismicas
        self.source_seismic_len = len(self.source_seismic_group)
        self.source_nonseismic_len = len(self.source_nonseismic_group)

        # Calcular el numero de trazas para cada uno
        self.train, self.val, self.test = self.get_traces_division()

        train_name = f"{self.dataset_name}_{self.train_ratio}_train.hdf5"
        val_name = f"{self.dataset_name}_{self.val_ratio}_val.hdf5"
        test_name = f"{self.dataset_name}_{self.test_ratio}_test.hdf5"

        with h5py.File(train_name, "w") as train_h5, \
                h5py.File(val_name, "w") as val_h5, \
                h5py.File(test_name, "w") as test_h5:

            # Create groups
            grp_train_seismic = train_h5.create_group("earthquake/local")
            grp_train_nonseismic = train_h5.create_group("non_earthquake/noise")

            grp_val_seismic = val_h5.create_group("earthquake/local")
            grp_val_nonseismic = val_h5.create_group("non_earthquake/noise")

            grp_test_seismic = test_h5.create_group("earthquake/local")
            grp_test_nonseismic = test_h5.create_group("non_earthquake/noise")

            # Recorrer trazas sismicas del dataset
            for i, dset in self.source_seismic_group:

                if i in self.train["seismic"]:
                    grp_train_seismic.copy(self.source_seismic_group[dset],
                                           dset)

                elif i in self.val["seismic"]:
                    grp_val_seismic.copy(self.source_seismic_group[dset], dset)

                elif i in self.test["seismic"]:
                    grp_test_seismic.copy(self.source_seismic_group[dset], dset)

                else:
                    print("Index not found!")

            # Recorrer trazas no sismicas del dataet
            for i, dset in self.source_nonseismic_group:

                if i in self.train["nonseismic"]:
                    grp_train_nonseismic.copy(
                        self.source_nonseismic_group[dset],
                        dset)

                elif i in self.val["nonseismic"]:
                    grp_val_nonseismic.copy(self.source_nonseismic_group[dset],
                                            dset)

                elif i in self.test["nonseismic"]:
                    grp_test_nonseismic.copy(self.source_nonseismic_group[dset],
                                             dset)
                else:
                    print("Index not found!")

    def get_traces_division(self):

        # Seismic division
        seis_divs = self.source_seismic_len * np.array([self.train_ratio,
                                                        self.val_ratio,
                                                        self.test_ratio])

        seis_divs = np.floor(seis_divs).astype(np.int32)

        # Trazas que quedan sin grupo se suman al de train
        seis_diff = self.source_seismic_len - np.sum(seis_divs)

        # Non-seismic division
        nonseis_divs = self.source_nonseismic_len * np.array([self.train_ratio,
                                                              self.val_ratio,
                                                              self.test_ratio])

        nonseis_divs = np.floor(nonseis_divs).astype(np.int32)

        # Numero de trazas que quedan sin grupo
        nonseis_diff = self.source_nonseismic_len - np.sum(nonseis_divs)

        # Elegir los indices de las trazas sismicas a copiar
        rng = default_rng()

        # Se crean una lista con todos los indices, se eligen los de train
        # y se quitan de a lista
        source_ids = list(range(self.source_seismic_len))
        seismic_train_ids = rng.choice(source_ids, seis_divs[0], replace=False)
        source_ids = list(set(source_ids)-set(seismic_train_ids))

        # Se eligen los ids de val y se quitan de la lista
        seismic_val_ids = rng.choice(source_ids, seis_divs[1], replace=False)
        source_ids = list(set(source_ids) - set(seismic_val_ids))

        # Los ids que quedan son para test
        seismic_test_ids = source_ids

        assert_msg = f"seismic_train_ids: {len(seismic_train_ids)}\n" \
                     f"seismic_val_ids: {len(seismic_val_ids)}\n" \
                     f"seismic_test_ids: {len(seismic_test_ids)}\n" \
                     f"source_ids: {self.source_seismic_len}"

        assert len(seismic_train_ids) + len(seismic_val_ids) + len(
            seismic_test_ids) == self.source_seismic_len, assert_msg

        # Se crean una lista con todos los indices, se eligen los de train
        # y se quitan de a lista
        source_ids = list(range(self.source_nonseismic_len))
        nonseismic_train_ids = rng.choice(source_ids, nonseis_divs[0])
        source_ids = list(set(source_ids) - set(nonseismic_train_ids))

        # Se eligen los ids de val y se quitan de la lista
        nonseismic_val_ids = rng.choice(source_ids, nonseis_divs[1])
        source_ids = list(set(source_ids) - set(nonseismic_val_ids))

        # Los ids que quedan son para test
        nonseismic_test_ids = source_ids

        assert_msg = f"nonseismic_train_ids: {len(nonseismic_train_ids)}\n" \
                     f"nonseismic_val_ids: {len(nonseismic_val_ids)}\n" \
                     f"nonseismic_test_ids: {len(nonseismic_test_ids)}\n" \
                     f"source_ids: {self.source_nonseismic_len}"

        assert len(nonseismic_train_ids) + len(nonseismic_val_ids) + len(
            nonseismic_test_ids) == self.source_nonseismic_len, assert_msg

        train_ids = [seismic_train_ids, nonseismic_train_ids]
        val_ids = [seismic_val_ids, nonseismic_val_ids]
        test_ids = [seismic_test_ids, nonseismic_test_ids]

        return train_ids, val_ids, test_ids

