import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Results:
    def __init__(self, csv_path, dset, beta=2, n_thresholds=100):

        self.csv_path = csv_path
        self.dset = dset
        self.beta = beta

        self.n_thresholds = n_thresholds
        self.thresholds = np.linspace(0, 1, self.n_thresholds + 1)[:-1]

        self.df = pd.read_csv(self.csv_path)
        self.model_name = self.csv_path.split('/')[-1].split('.')[0]

        # Preallocate variables
        acc = np.zeros(len(self.thresholds))
        prec = np.zeros(len(self.thresholds))
        rec = np.zeros(len(self.thresholds))
        fpr = np.zeros(len(self.thresholds))
        fscore = np.zeros(len(self.thresholds))

        for i, thr in enumerate(self.thresholds):
            predicted = (self.df['out'] > thr)
            tp = sum(predicted & self.df['label'])
            fp = sum(predicted & ~self.df['label'])
            fn = sum(~predicted & self.df['label'])
            tn = sum(~predicted & ~self.df['label'])

            # Evaluation metrics
            acc[i], prec[i], rec[i], fpr[i], fscore[i] = self.get_metrics(tp,
                                                                          fp,
                                                                          tn,
                                                                          fn,
                                                                          self.beta)

        self.acc = acc
        self.prec = prec
        self.rec = rec
        self.fpr = fpr
        self.fscore = fscore

        self.pr_auc, self.roc_auc = self.get_aucs(self.prec, self.rec, self.fpr)
        self.best_threshold = self.thresholds[np.argmax(self.fscore)]
        self.best_fscore = np.amax(self.fscore)

        self.save_histogram()
        self.save_fsc()
        self.save_pr()
        self.save_roc()

    @staticmethod
    def get_aucs(precision, recall, fpr):
        pr_auc = np.trapz(precision[::-1], x=recall[::-1])
        roc_auc = np.trapz(recall[::-1], x=fpr[::-1])

        return pr_auc, roc_auc

    @staticmethod
    def get_metrics(tp, fp, tn, fn, beta):
        acc = (tp + tn) / (tp + fp + tn + fn)

        # Evaluation metrics
        if (not tp) and (not fp):
            precision = 0
        else:
            precision = tp / (tp + fp)

        if (not tp) and (not fn):
            recall = 1
        else:
            recall = tp / (tp + fn)

        if (not fp) and (not tn):
            fpr = 0
        else:
            fpr = fp / (fp + tn)

        if (not precision) and (not recall):
            fscore = 0
        else:
            fscore = (1 + beta ** 2) * (precision * recall) / \
                     ((beta ** 2) * precision + recall)

        return acc, recall, precision, fpr, fscore

    def save_histogram(self):
        if not os.path.exists(f"Figures/Histogram/{self.dset}"):
            os.makedirs(f"Figures/Histogram/{self.dset}", exist_ok=True)

        plt.figure(figsize=(12, 9))
        plt.hist(self.df[self.df['label'] == 1]['out'], 100)
        plt.hist(self.df[self.df['label'] == 0]['out'], 100)
        plt.title(f'Output values histogram')
        plt.xlabel('Output values')
        plt.ylabel('Counts')
        plt.legend(['positive', 'negative'], loc='upper left')
        plt.grid(True)
        plt.savefig(f"Figures/Histogram/{self.dset}/{self.model_name}.png")
        plt.close()

    def save_fsc(self):
        if not os.path.exists(f"Figures/Fscore/{self.dset}"):
            os.makedirs(f"Figures/Fscore/{self.dset}/", exist_ok=True)

        plt.figure(figsize=(12, 9))
        plt.plot(self.thresholds, self.fscore, '--o')
        plt.xlabel('Thresholds')
        plt.ylabel('F-score')
        plt.title('Fscores vs Thresholds')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(True)
        plt.savefig(f"Figures/Fscore/{self.dset}/{self.model_name}.png")
        plt.close()

    def save_pr(self):
        if not os.path.exists(f"Figures/PR/{self.dset}/"):
            os.makedirs(f"Figures/PR/{self.dset}", exist_ok=True)

        plt.figure(figsize=(12, 9))
        plt.plot(self.rec, self.prec, '--o')
        plt.hlines(0.5, 0, 1, 'b', '--')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision vs Recall (PR curve)')
        plt.grid(True)
        plt.savefig(f"Figures/PR/{self.dset}/{self.model_name}.png")
        plt.close()

    def save_roc(self):
        if not os.path.exists(f"Figures/ROC/{self.dset}/"):
            os.makedirs(f"Figures/ROC/{self.dset}/", exist_ok=True)

        plt.figure(figsize=(12, 9))
        plt.plot(self.fpr, self.rec, '--o')
        plt.plot([0, 1], [0, 1], 'b--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('Recall')
        plt.title('Recall vs FPR (ROC curve)')
        plt.grid(True)
        plt.savefig(f"Figures/ROC/{self.dset}/{self.model_name}.png")
        plt.close()

