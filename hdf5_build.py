import os
import h5py
import json


class H5Builder:
    """
    Tomar datasets en formato npy y construir un dataset HDF5, guardarlo
    """
    def __init__(self, config_file):

        # Load dataset config
        with open(config_file, 'r') as config:
            self.cfg = json.load(config)

        if not os.path.exists(os.path.dirname(self.cfg['OutputFile'])):
            os.makedirs(os.path.dirname(self.cfg['OutputFile']), exist_ok=True)

        # Create hdf5 dataset file
        self.h5 = h5py.File(self.cfg['OutputFile'], 'w')

        # Create seismic and non-seismic groups
        g_earthquake = self.h5.create_group('earthquake/local')
        g_non_earthquake = self.h5.create_group('non_earthquake/noise')

        # For ebery dataset in config file
        # if dset type is hdf5 -> copy
        # if dset type is npy  -> create

        self.h5.close()

    def copy_dataset(self, source, destiny):
        pass

    def create_dataset(self, source, destiny):
        pass

