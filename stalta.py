import os
import numpy as np
import pandas as pd

from obspy.signal.trigger import classic_sta_lta


class Trigger:
    def __init__(self, dataset_path, thresh):

        self.dataset_path = dataset_path
        self.dataset_name = self.dataset_path.split("/")[-1].split(".")[0]
        self.thresh = thresh

        # load dataset
        self.dataset = np.load(self.dataset_path)

        # run sta lta en every trace
        self.fs = 100

        self.save_triggers_csv()

    def save_triggers_csv(self):

        df = pd.DataFrame(columns=["Trace number", "Trigger"])

        for i, tr in enumerate(self.dataset):
            trig = 0
            cft = classic_sta_lta(tr, int(5 * self.fs), int(10 * self.fs))

            if np.max(cft) > self.thresh:
                trig = 1

            df.loc[i] = [i, trig]

        os.makedirs("STA-LTA-Triggers", exist_ok=True)
        df.to_csv(f"STA-LTA-Triggers/{self.dataset_name}.csv", index=False)
