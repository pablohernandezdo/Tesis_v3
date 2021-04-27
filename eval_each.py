import os
import time
import argparse

import numpy as np
import tqdm
import torch
import pandas as pd
from torch.utils.data import DataLoader
from humanfriendly import format_timespan

from model_cnn import *


def main():
    # Measure exec time
    start_time = time.time()

    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name",
                        default='STEAD-STEAD',
                        help="Name of dataset to evaluate on")
    parser.add_argument("--model_name", default='1h6k_test_model',
                        help="Name of model to eval")
    parser.add_argument("--model_folder", default='models',
                        help="Model to eval folder")
    parser.add_argument("--classifier", default='1h6k',
                        help="Choose classifier architecture")
    parser.add_argument("--test_path", default='Francia.npy',
                        help="HDF5 test Dataset path")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Size of the training batches")
    args = parser.parse_args()

    # Select training device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load specified Classifier
    net = get_classifier(args.classifier)
    net.to(device)

    # Count number of parameters
    params = count_parameters(net)

    # Load npy data
    dset_npy = np.load(args.test_path)

    # Load from trained model
    net.load_state_dict(torch.load(f'{args.model_folder}/'
                                   f'{args.model_name}.pth'))
    net.eval()

    # Evaluate model on test set
    evaluate_dataset(dset_npy, args.dataset_name + '/Test',
                     device, net, args.model_name,
                     args.model_folder,
                     'Net_outputs')

    eval_end = time.time()
    total_time = eval_end - start_time

    print(f'Total evaluation time: {format_timespan(total_time)}\n\n'
          f'Number of network parameters: {params}')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_dataset(dset_npy, dataset_name, device, net,
                     model_name, model_folder, csv_folder):

    # List of outputs and labels used to create pd dataframe
    dataframe_rows_list = []

    with tqdm.tqdm(total=len(dset_npy),
                   desc=f'{dataset_name} dataset evaluation') as data_bar:

        with torch.no_grad():
            for data in dset_npy:

                trace = data[:6000].astype(np.float32)
                label = np.array([data[-1]]).astype(np.float32)

                trace = torch.from_numpy(trace).to(device)
                label = torch.from_numpy(label).to(device)

                outputs = net(trace)

                for out, lab in zip(outputs, label):
                    new_row = {'out': out.item(),
                               'label': int(lab.item())}
                    dataframe_rows_list.append(new_row)

                data_bar.update(1)

    test_outputs = pd.DataFrame(dataframe_rows_list)

    if len(model_folder.split("/")) > 1:
        model_dirname = model_folder.split("/")[-1]

    else:
        model_dirname = ""

    # Create csv folder if necessary
    save_folder = f'{csv_folder}/{dataset_name}/{model_dirname}'

    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)

    # Save outputs and labels to csv file
    test_outputs.to_csv(f'{save_folder}/{model_name}.csv', index=False)


def get_classifier(x):
    if x == 'Cnn1_6k':
        return Cnn1_6k()
    if x == 'Cnn1_5k':
        return Cnn1_5k()
    if x == 'Cnn1_4k':
        return Cnn1_4k()
    if x == 'Cnn1_3k':
        return Cnn1_3k()
    if x == 'Cnn1_2k':
        return Cnn1_2k()
    if x == 'Cnn1_1k':
        return Cnn1_1k()
    if x == 'Cnn1_5h':
        return Cnn1_5h()
    if x == 'Cnn1_2h':
        return Cnn1_2h()
    if x == 'Cnn1_5h':
        return Cnn1_1h()
    if x == 'Cnn1_10':
        return Cnn1_10()
    if x == 'Cnn1_6k_6k':
        return Cnn1_6k_6k()
    if x == 'Cnn1_6k_5k':
        return Cnn1_6k_5k()
    if x == 'Cnn1_6k_4k':
        return Cnn1_6k_4k()
    if x == 'Cnn1_6k_3k':
        return Cnn1_6k_3k()
    if x == 'Cnn1_6k_2k':
        return Cnn1_6k_2k()
    if x == 'Cnn1_6k_1k':
        return Cnn1_6k_1k()
    if x == 'Cnn1_6k_5h':
        return Cnn1_6k_5h()
    if x == 'Cnn1_6k_2h':
        return Cnn1_6k_2h()
    if x == 'Cnn1_6k_1h':
        return Cnn1_6k_1h()
    if x == 'Cnn1_6k_10':
        return Cnn1_6k_10()
    if x == 'Cnn1_5k_6k':
        return Cnn1_5k_6k()
    if x == 'Cnn1_5k_5k':
        return Cnn1_5k_5k()
    if x == 'Cnn1_5k_4k':
        return Cnn1_5k_4k()
    if x == 'Cnn1_5k_3k':
        return Cnn1_5k_3k()
    if x == 'Cnn1_5k_2k':
        return Cnn1_5k_2k()
    if x == 'Cnn1_5k_1k':
        return Cnn1_5k_1k()
    if x == 'Cnn1_5k_5h':
        return Cnn1_5k_5h()
    if x == 'Cnn1_5k_2h':
        return Cnn1_5k_2h()
    if x == 'Cnn1_5k_1h':
        return Cnn1_5k_1h()
    if x == 'Cnn1_5k_10':
        return Cnn1_5k_10()
    if x == 'Cnn1_4k_6k':
        return Cnn1_4k_6k()
    if x == 'Cnn1_4k_5k':
        return Cnn1_4k_5k()
    if x == 'Cnn1_4k_4k':
        return Cnn1_4k_4k()
    if x == 'Cnn1_4k_3k':
        return Cnn1_4k_3k()
    if x == 'Cnn1_4k_2k':
        return Cnn1_4k_2k()
    if x == 'Cnn1_4k_1k':
        return Cnn1_4k_1k()
    if x == 'Cnn1_4k_5h':
        return Cnn1_4k_5h()
    if x == 'Cnn1_4k_2h':
        return Cnn1_4k_2h()
    if x == 'Cnn1_4k_1h':
        return Cnn1_4k_1h()
    if x == 'Cnn1_4k_10':
        return Cnn1_4k_10()
    if x == 'Cnn1_3k_6k':
        return Cnn1_3k_6k()
    if x == 'Cnn1_3k_5k':
        return Cnn1_3k_5k()
    if x == 'Cnn1_3k_4k':
        return Cnn1_3k_4k()
    if x == 'Cnn1_3k_3k':
        return Cnn1_3k_3k()
    if x == 'Cnn1_3k_2k':
        return Cnn1_3k_2k()
    if x == 'Cnn1_3k_1k':
        return Cnn1_3k_1k()
    if x == 'Cnn1_3k_5h':
        return Cnn1_3k_5h()
    if x == 'Cnn1_3k_2h':
        return Cnn1_3k_2h()
    if x == 'Cnn1_3k_1h':
        return Cnn1_3k_1h()
    if x == 'Cnn1_3k_10':
        return Cnn1_3k_10()
    if x == 'Cnn1_2k_6k':
        return Cnn1_2k_6k()
    if x == 'Cnn1_2k_5k':
        return Cnn1_2k_5k()
    if x == 'Cnn1_2k_4k':
        return Cnn1_2k_4k()
    if x == 'Cnn1_2k_3k':
        return Cnn1_2k_3k()
    if x == 'Cnn1_2k_2k':
        return Cnn1_2k_2k()
    if x == 'Cnn1_2k_1k':
        return Cnn1_2k_1k()
    if x == 'Cnn1_2k_5h':
        return Cnn1_2k_5h()
    if x == 'Cnn1_2k_2h':
        return Cnn1_2k_2h()
    if x == 'Cnn1_2k_1h':
        return Cnn1_2k_1h()
    if x == 'Cnn1_2k_10':
        return Cnn1_2k_10()
    if x == 'Cnn1_1k_6k':
        return Cnn1_1k_6k()
    if x == 'Cnn1_1k_5k':
        return Cnn1_1k_5k()
    if x == 'Cnn1_1k_4k':
        return Cnn1_1k_4k()
    if x == 'Cnn1_1k_3k':
        return Cnn1_1k_3k()
    if x == 'Cnn1_1k_2k':
        return Cnn1_1k_2k()
    if x == 'Cnn1_1k_1k':
        return Cnn1_1k_1k()
    if x == 'Cnn1_1k_5h':
        return Cnn1_1k_5h()
    if x == 'Cnn1_1k_2h':
        return Cnn1_1k_2h()
    if x == 'Cnn1_1k_1h':
        return Cnn1_1k_1h()
    if x == 'Cnn1_1k_10':
        return Cnn1_1k_10()
    if x == 'Cnn1_5h_6k':
        return Cnn1_5h_6k()
    if x == 'Cnn1_5h_5k':
        return Cnn1_5h_5k()
    if x == 'Cnn1_5h_4k':
        return Cnn1_5h_4k()
    if x == 'Cnn1_5h_3k':
        return Cnn1_5h_3k()
    if x == 'Cnn1_5h_2k':
        return Cnn1_5h_2k()
    if x == 'Cnn1_5h_1k':
        return Cnn1_5h_1k()
    if x == 'Cnn1_5h_5h':
        return Cnn1_5h_5h()
    if x == 'Cnn1_5h_2h':
        return Cnn1_5h_2h()
    if x == 'Cnn1_5h_1h':
        return Cnn1_5h_1h()
    if x == 'Cnn1_5h_10':
        return Cnn1_5h_10()
    if x == 'Cnn1_2h_6k':
        return Cnn1_2h_6k()
    if x == 'Cnn1_2h_5k':
        return Cnn1_2h_5k()
    if x == 'Cnn1_2h_4k':
        return Cnn1_2h_4k()
    if x == 'Cnn1_2h_3k':
        return Cnn1_2h_3k()
    if x == 'Cnn1_2h_2k':
        return Cnn1_2h_2k()
    if x == 'Cnn1_2h_1k':
        return Cnn1_2h_1k()
    if x == 'Cnn1_2h_5h':
        return Cnn1_2h_5h()
    if x == 'Cnn1_2h_2h':
        return Cnn1_2h_2h()
    if x == 'Cnn1_2h_1h':
        return Cnn1_2h_1h()
    if x == 'Cnn1_2h_10':
        return Cnn1_2h_10()
    if x == 'Cnn1_1h_6k':
        return Cnn1_1h_6k()
    if x == 'Cnn1_1h_5k':
        return Cnn1_1h_5k()
    if x == 'Cnn1_1h_4k':
        return Cnn1_1h_4k()
    if x == 'Cnn1_1h_3k':
        return Cnn1_1h_3k()
    if x == 'Cnn1_1h_2k':
        return Cnn1_1h_2k()
    if x == 'Cnn1_1h_1k':
        return Cnn1_1h_1k()
    if x == 'Cnn1_1h_5h':
        return Cnn1_1h_5h()
    if x == 'Cnn1_1h_2h':
        return Cnn1_1h_2h()
    if x == 'Cnn1_1h_1h':
        return Cnn1_1h_1h()
    if x == 'Cnn1_1h_10':
        return Cnn1_1h_10()
    if x == 'Cnn1_10_6k':
        return Cnn1_10_6k()
    if x == 'Cnn1_10_5k':
        return Cnn1_10_5k()
    if x == 'Cnn1_10_4k':
        return Cnn1_10_4k()
    if x == 'Cnn1_10_3k':
        return Cnn1_10_3k()
    if x == 'Cnn1_10_2k':
        return Cnn1_10_2k()
    if x == 'Cnn1_10_1k':
        return Cnn1_10_1k()
    if x == 'Cnn1_10_5h':
        return Cnn1_10_5h()
    if x == 'Cnn1_10_2h':
        return Cnn1_10_2h()
    if x == 'Cnn1_10_1h':
        return Cnn1_10_1h()
    if x == 'Cnn1_10_10':
        return Cnn1_10_10()
    if x == 'Cnn2_6k':
        return Cnn2_6k()
    if x == 'Cnn2_5k':
        return Cnn2_5k()
    if x == 'Cnn2_4k':
        return Cnn2_4k()
    if x == 'Cnn2_3k':
        return Cnn2_3k()
    if x == 'Cnn2_2k':
        return Cnn2_2k()
    if x == 'Cnn2_1k':
        return Cnn2_1k()
    if x == 'Cnn2_5h':
        return Cnn2_5h()
    if x == 'Cnn2_2h':
        return Cnn2_2h()
    if x == 'Cnn2_5h':
        return Cnn2_1h()
    if x == 'Cnn2_10':
        return Cnn2_10()
    if x == 'Cnn2_6k_6k':
        return Cnn2_6k_6k()
    if x == 'Cnn2_6k_5k':
        return Cnn2_6k_5k()
    if x == 'Cnn2_6k_4k':
        return Cnn2_6k_4k()
    if x == 'Cnn2_6k_3k':
        return Cnn2_6k_3k()
    if x == 'Cnn2_6k_2k':
        return Cnn2_6k_2k()
    if x == 'Cnn2_6k_1k':
        return Cnn2_6k_1k()
    if x == 'Cnn2_6k_5h':
        return Cnn2_6k_5h()
    if x == 'Cnn2_6k_2h':
        return Cnn2_6k_2h()
    if x == 'Cnn2_6k_1h':
        return Cnn2_6k_1h()
    if x == 'Cnn2_6k_10':
        return Cnn2_6k_10()
    if x == 'Cnn2_5k_6k':
        return Cnn2_5k_6k()
    if x == 'Cnn2_5k_5k':
        return Cnn2_5k_5k()
    if x == 'Cnn2_5k_4k':
        return Cnn2_5k_4k()
    if x == 'Cnn2_5k_3k':
        return Cnn2_5k_3k()
    if x == 'Cnn2_5k_2k':
        return Cnn2_5k_2k()
    if x == 'Cnn2_5k_1k':
        return Cnn2_5k_1k()
    if x == 'Cnn2_5k_5h':
        return Cnn2_5k_5h()
    if x == 'Cnn2_5k_2h':
        return Cnn2_5k_2h()
    if x == 'Cnn2_5k_1h':
        return Cnn2_5k_1h()
    if x == 'Cnn2_5k_10':
        return Cnn2_5k_10()
    if x == 'Cnn2_4k_6k':
        return Cnn2_4k_6k()
    if x == 'Cnn2_4k_5k':
        return Cnn2_4k_5k()
    if x == 'Cnn2_4k_4k':
        return Cnn2_4k_4k()
    if x == 'Cnn2_4k_3k':
        return Cnn2_4k_3k()
    if x == 'Cnn2_4k_2k':
        return Cnn2_4k_2k()
    if x == 'Cnn2_4k_1k':
        return Cnn2_4k_1k()
    if x == 'Cnn2_4k_5h':
        return Cnn2_4k_5h()
    if x == 'Cnn2_4k_2h':
        return Cnn2_4k_2h()
    if x == 'Cnn2_4k_1h':
        return Cnn2_4k_1h()
    if x == 'Cnn2_4k_10':
        return Cnn2_4k_10()
    if x == 'Cnn2_3k_6k':
        return Cnn2_3k_6k()
    if x == 'Cnn2_3k_5k':
        return Cnn2_3k_5k()
    if x == 'Cnn2_3k_4k':
        return Cnn2_3k_4k()
    if x == 'Cnn2_3k_3k':
        return Cnn2_3k_3k()
    if x == 'Cnn2_3k_2k':
        return Cnn2_3k_2k()
    if x == 'Cnn2_3k_1k':
        return Cnn2_3k_1k()
    if x == 'Cnn2_3k_5h':
        return Cnn2_3k_5h()
    if x == 'Cnn2_3k_2h':
        return Cnn2_3k_2h()
    if x == 'Cnn2_3k_1h':
        return Cnn2_3k_1h()
    if x == 'Cnn2_3k_10':
        return Cnn2_3k_10()
    if x == 'Cnn2_2k_6k':
        return Cnn2_2k_6k()
    if x == 'Cnn2_2k_5k':
        return Cnn2_2k_5k()
    if x == 'Cnn2_2k_4k':
        return Cnn2_2k_4k()
    if x == 'Cnn2_2k_3k':
        return Cnn2_2k_3k()
    if x == 'Cnn2_2k_2k':
        return Cnn2_2k_2k()
    if x == 'Cnn2_2k_1k':
        return Cnn2_2k_1k()
    if x == 'Cnn2_2k_5h':
        return Cnn2_2k_5h()
    if x == 'Cnn2_2k_2h':
        return Cnn2_2k_2h()
    if x == 'Cnn2_2k_1h':
        return Cnn2_2k_1h()
    if x == 'Cnn2_2k_10':
        return Cnn2_2k_10()
    if x == 'Cnn2_1k_6k':
        return Cnn2_1k_6k()
    if x == 'Cnn2_1k_5k':
        return Cnn2_1k_5k()
    if x == 'Cnn2_1k_4k':
        return Cnn2_1k_4k()
    if x == 'Cnn2_1k_3k':
        return Cnn2_1k_3k()
    if x == 'Cnn2_1k_2k':
        return Cnn2_1k_2k()
    if x == 'Cnn2_1k_1k':
        return Cnn2_1k_1k()
    if x == 'Cnn2_1k_5h':
        return Cnn2_1k_5h()
    if x == 'Cnn2_1k_2h':
        return Cnn2_1k_2h()
    if x == 'Cnn2_1k_1h':
        return Cnn2_1k_1h()
    if x == 'Cnn2_1k_10':
        return Cnn2_1k_10()
    if x == 'Cnn2_5h_6k':
        return Cnn2_5h_6k()
    if x == 'Cnn2_5h_5k':
        return Cnn2_5h_5k()
    if x == 'Cnn2_5h_4k':
        return Cnn2_5h_4k()
    if x == 'Cnn2_5h_3k':
        return Cnn2_5h_3k()
    if x == 'Cnn2_5h_2k':
        return Cnn2_5h_2k()
    if x == 'Cnn2_5h_1k':
        return Cnn2_5h_1k()
    if x == 'Cnn2_5h_5h':
        return Cnn2_5h_5h()
    if x == 'Cnn2_5h_2h':
        return Cnn2_5h_2h()
    if x == 'Cnn2_5h_1h':
        return Cnn2_5h_1h()
    if x == 'Cnn2_5h_10':
        return Cnn2_5h_10()
    if x == 'Cnn2_2h_6k':
        return Cnn2_2h_6k()
    if x == 'Cnn2_2h_5k':
        return Cnn2_2h_5k()
    if x == 'Cnn2_2h_4k':
        return Cnn2_2h_4k()
    if x == 'Cnn2_2h_3k':
        return Cnn2_2h_3k()
    if x == 'Cnn2_2h_2k':
        return Cnn2_2h_2k()
    if x == 'Cnn2_2h_1k':
        return Cnn2_2h_1k()
    if x == 'Cnn2_2h_5h':
        return Cnn2_2h_5h()
    if x == 'Cnn2_2h_2h':
        return Cnn2_2h_2h()
    if x == 'Cnn2_2h_1h':
        return Cnn2_2h_1h()
    if x == 'Cnn2_2h_10':
        return Cnn2_2h_10()
    if x == 'Cnn2_1h_6k':
        return Cnn2_1h_6k()
    if x == 'Cnn2_1h_5k':
        return Cnn2_1h_5k()
    if x == 'Cnn2_1h_4k':
        return Cnn2_1h_4k()
    if x == 'Cnn2_1h_3k':
        return Cnn2_1h_3k()
    if x == 'Cnn2_1h_2k':
        return Cnn2_1h_2k()
    if x == 'Cnn2_1h_1k':
        return Cnn2_1h_1k()
    if x == 'Cnn2_1h_5h':
        return Cnn2_1h_5h()
    if x == 'Cnn2_1h_2h':
        return Cnn2_1h_2h()
    if x == 'Cnn2_1h_1h':
        return Cnn2_1h_1h()
    if x == 'Cnn2_1h_10':
        return Cnn2_1h_10()
    if x == 'Cnn2_10_6k':
        return Cnn2_10_6k()
    if x == 'Cnn2_10_5k':
        return Cnn2_10_5k()
    if x == 'Cnn2_10_4k':
        return Cnn2_10_4k()
    if x == 'Cnn2_10_3k':
        return Cnn2_10_3k()
    if x == 'Cnn2_10_2k':
        return Cnn2_10_2k()
    if x == 'Cnn2_10_1k':
        return Cnn2_10_1k()
    if x == 'Cnn2_10_5h':
        return Cnn2_10_5h()
    if x == 'Cnn2_10_2h':
        return Cnn2_10_2h()
    if x == 'Cnn2_10_1h':
        return Cnn2_10_1h()
    if x == 'Cnn2_10_10':
        return Cnn2_10_10()
    else:
        return Cnn2_10_10()


if __name__ == "__main__":
    main()
