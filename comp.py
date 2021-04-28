import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Comp:
    def __init__(self, csv1_path, csv2_path, dset, beta=2, n_thresholds=100):

        self.csv1_path = csv1_path
        self.csv2_path = csv2_path
        self.dset = dset
        self.beta = beta

        self.n_thresholds = n_thresholds
        self.thresholds = np.linspace(0, 1, self.n_thresholds + 1)[:-1]

        self.df1 = pd.read_csv(self.csv1_path)
        self.df2 = pd.read_csv(self.csv2_path)

        self.model1_name = self.csv1_path.split('/')[-1].split('.')[0]
        self.model2_name = self.csv2_path.split('/')[-1].split('.')[0]

        self.acc1, self.prec1, \
            self.rec1, self.fpr1, self.fscore1 = self.get_full_metrics(self.df1)

        self.acc2, self.prec2, \
            self.rec2, self.fpr2, self.fscore2 = self.get_full_metrics(self.df2)

        self.pr_auc1, self.roc_auc1 = self.get_aucs(self.prec1,
                                                    self.rec1, self.fpr1)

        self.pr_auc2, self.roc_auc2 = self.get_aucs(self.prec2,
                                                    self.rec2, self.fpr2)

        self.best_threshold1 = self.thresholds[np.argmax(self.fscore1)]
        self.best_threshold2 = self.thresholds[np.argmax(self.fscore2)]
        self.best_fscore1 = np.amax(self.fscore1)
        self.best_fscore2 = np.amax(self.fscore2)

        self.save_histogram()
        self.save_fsc()
        self.save_pr()
        self.save_roc()

    def get_full_metrics(self, df):
        # Preallocate variables
        acc = np.zeros(len(self.thresholds))
        prec = np.zeros(len(self.thresholds))
        rec = np.zeros(len(self.thresholds))
        fpr = np.zeros(len(self.thresholds))
        fscore = np.zeros(len(self.thresholds))

        for i, thr in enumerate(self.thresholds):
            predicted = (df['out'] > thr)
            tp = sum(predicted & df['label'])
            fp = sum(predicted & ~df['label'])
            fn = sum(~predicted & df['label'])
            tn = sum(~predicted & ~df['label'])

            # Evaluation metrics
            acc[i], prec[i], rec[i], fpr[i], fscore[i] = self.get_metrics(tp,
                                                                          fp,
                                                                          tn,
                                                                          fn,
                                                                          self.beta)

        return acc, prec, rec, fpr, fscore

    @staticmethod
    def get_metrics(tp, fp, tn, fn, beta):
        acc = (tp + tn) / (tp + fp + tn + fn)

        # Evaluation metrics
        if (not tp) and (not fp):
            precision = 1
        else:
            precision = tp / (tp + fp)

        if (not tp) and (not fn):
            recall = 0
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

    @staticmethod
    def get_aucs(precision, recall, fpr):
        pr_auc = np.trapz(precision[::-1], x=recall[::-1])
        roc_auc = np.trapz(recall[::-1], x=fpr[::-1])

        return pr_auc, roc_auc

    def save_histogram(self):
        if not os.path.exists(f"Comp/Histogram/{self.dset}"):
            os.makedirs(f"Comp/Histogram/{self.dset}", exist_ok=True)

        plt.figure(figsize=(12, 9))
        plt.subplot(2, 1, 1)
        plt.hist(self.df1[self.df1['label'] == 1]['out'], 100)
        # plt.hist(self.df1[self.df1['label'] == 0]['out'], 100)
        plt.title(f'Output values histogram {self.model1_name}')
        plt.xlabel('Output values')
        plt.ylabel('Counts')
        plt.legend(['positive', 'negative'], loc='upper left')
        plt.grid(True)
        plt.subplot(2, 1, 2)
        plt.hist(self.df2[self.df2['label'] == 1]['out'], 100)
        # plt.hist(self.df2[self.df2['label'] == 0]['out'], 100)
        plt.title(f'Output values histogram {self.model2_name}')
        plt.xlabel('Output values')
        plt.ylabel('Counts')
        plt.legend(['positive', 'negative'], loc='upper left')
        plt.grid(True)
        plt.savefig(f"Comp/Histogram/{self.dset}/"
                    f"Comp_{self.model1_name}_{self.model2_name}.png")
        plt.close()

    def save_fsc(self):
        if not os.path.exists(f"Comp/Fscore/{self.dset}"):
            os.makedirs(f"Comp/Fscore/{self.dset}/", exist_ok=True)

        plt.figure(figsize=(12, 9))
        plt.plot(self.thresholds, self.fscore1, '--o', label=self.model1_name)
        plt.plot(self.thresholds, self.fscore2, '--o', label=self.model2_name)
        plt.xlabel('Thresholds')
        plt.ylabel('F-score')
        plt.title('Fscores vs Thresholds')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(True)
        plt.legend(loc='best')
        plt.savefig(f"Comp/Fscore/{self.dset}/"
                    f"Comp_{self.model1_name}_{self.model2_name}.png")
        plt.close()

    def save_pr(self):
        if not os.path.exists(f"Comp/PR/{self.dset}/"):
            os.makedirs(f"Comp/PR/{self.dset}", exist_ok=True)

        plt.figure(figsize=(12, 9))
        plt.plot(self.rec1, self.prec1, '--o', label=self.model1_name)
        plt.plot(self.rec2, self.prec2, '--o', label=self.model2_name)
        plt.hlines(0.5, 0, 1, 'b', '--')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision vs Recall (PR curve)')
        plt.grid(True)
        plt.legend(loc='best')
        plt.savefig(f"Comp/PR/{self.dset}/"
                    f"Comp_{self.model1_name}_{self.model2_name}.png")
        plt.close()

    def save_roc(self):
        if not os.path.exists(f"Comp/ROC/{self.dset}/"):
            os.makedirs(f"Comp/ROC/{self.dset}/", exist_ok=True)

        plt.figure(figsize=(12, 9))
        plt.plot(self.fpr1, self.rec1, '--o', label=self.model1_name)
        plt.plot(self.fpr2, self.rec2, '--o', label=self.model2_name)
        plt.plot([0, 1], [0, 1], 'b--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('Recall')
        plt.title('Recall vs FPR (ROC curve)')
        plt.grid(True)
        plt.legend(loc='best')
        plt.savefig(f"Comp/ROC/{self.dset}/"
                    f"Comp_{self.model1_name}_{self.model2_name}.png")
        plt.close()
