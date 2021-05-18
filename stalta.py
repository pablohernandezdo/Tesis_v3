import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from obspy.signal.trigger import classic_sta_lta


class Trigger:
    def __init__(self, dataset_path, thresh):

        self.dataset_path = dataset_path
        self.dataset_name = self.dataset_path.split("/")[-1].split(".")[0]
        self.thresh = thgresh

        # load dataset
        self.dataset = np.load(self.dataset_path)

        # run sta lta en every trace
        self.fs = 100

        self.cfts = self.save_triggers()

    def save_triggers(self):

        # ARREGLAR PARA QUE SE GUARDE EN UNA CARPETA CUYO
        # NOMBRE TENGA EL VALOR DEL TRIGGER

        trig_traces = []
        cfts = []
        df = pd.DataFrame(columns=["Trace number", "Trigger"])

        for i, tr in enumerate(self.dataset):
            trig = 0
            cft = classic_sta_lta(tr, int(5 * self.fs), int(10 * self.fs))
            cfts.append(cft)
            print(i)

            if np.max(cft) > self.thresh:
                trig_traces.append(tr)
                self.plot_trace(tr, i)
                trig = 1

            df.loc[i] = [i, trig]

        os.makedirs("STA-LTA-Triggers", exist_ok=True)
        df.to_csv(f"STA-LTA-Triggers/{self.dataset_name}_{self.thresh}.csv",
                  index=False)

        trig_traces = np.asarray(trig_traces)

        os.makedirs("Data/Trigger", exist_ok=True)
        np.save(f"Data/Trigger/{self.dataset_name}_{self.thresh}.npy",
                trig_traces)

        return np.asarray(cfts)

    def plot_trace(self, trace, idx):

        savepath = f"STA-LTA-Triggers/{self.dataset_name}"

        os.makedirs(savepath, exist_ok=True)

        plt.figure(figsize=(20, 20))
        plt.plot(trace)
        plt.title(f'Dataset {self.dataset_name}, trace number {idx}')
        plt.xlabel('Samples')
        plt.ylabel('[-]')
        plt.grid(True)
        plt.savefig(f"{savepath}/Trace_{idx}.png")
        plt.close()
