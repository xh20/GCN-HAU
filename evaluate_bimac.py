import json

import numpy
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

import argparse

label_list = ['idle', 'approach', 'retreat', 'list', 'place', 'hold', 'pour', 'cut', 'hammer', 'saw', ' stir', 'screw',
              'drink', 'wipe']

def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')
    parser.add_argument(
        '--filename',
        default=' ',
        help='the saved filename')

    return parser


class seg_cache():
    def __init__(self):
        self.label = 0
        self.start = 0
        self.end = 0
    
    def toseg(self):
        return([self.label, self.start, self.end])


def framewise_eval(preds, truth):
    sample_name = preds.keys()
    tp = np.zeros(32)
    fp = np.zeros(32)
    fn = np.zeros(32)
    right = 0
    wrong = 0

    '''
    for n in sample_name:
        p = preds[n]
        t = truth[n]
        for i in range(len(p)):
           if t[i] != -1:
                if p[i] == t[i]:
                    tp[p[i]] += 1
                    right += 1
                else:
                    fp[p[i]] += 1
                    fn[t[i]] += 1
                    wrong += 1

    f1 = tp / (tp + 0.5 * (fp + fn))
    f1_macro = f1.mean()
    acc = right / (right + wrong)
    print('f1 score macro: {0:.3%}'.format(f1_macro))
    print('f1 score micro: {0:.3%}'.format(acc))
    '''
    
    pr = []
    tr = []
    for n in sample_name:
        p = preds[n]
        t = truth[n]

        for i in range(len(t)):
            if t[i] != -1:
                pr.append(p[i])
                tr.append(t[i])
                if p[i] == t[i]:
                    right += 1
                else:
                    wrong += 1

    plt.rcParams['font.size'] = '17'
    cnf_matrix = confusion_matrix(tr, pr, normalize="true")
    # cnf_matrix = cnf_matrix/np.sum(cnf_matrix, axis=1)
    # cnf_text = numpy.array2string(cnf_matrix, formatter={'float_kind': lambda cnf_matrix: "%.2f" % cnf_matrix})
    disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix, display_labels=label_list)
    disp.plot(cmap='Blues', xticks_rotation=45, values_format='.2f', fontsize=17)
    # disp.plot(cmap='Blues', xticks_rotation='vertical', values_format='.2f', fontsize=17)
    # plt.show()
    fig = plt.gcf()
    fig.set_size_inches(15, 12)
    plt.savefig('confusion_matrix.png', dpi=300)
    # print(cnf_matrix)

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    f1 = TP / (TP + 0.5 * (FP + FN))
    f1_macro = f1.mean()
    f1_macro_weighted = f1_score(tr, pr, average='weighted')
    acc = right / (right + wrong)

    print('f1 score macro: {0:.3%}'.format(f1_macro))
    print('f1 score weighted: {0:.3%}'.format(f1_macro_weighted))
    print('f1 score micro: {0:.3%}'.format(acc))


def framewiselabel2seglabel(frame_label):
    seg_label = {}
    for key in frame_label.keys():
        seg = []
        cache = seg_cache()
        for i in range(len(frame_label[key])):
            if i == 0:
                cache.label = frame_label[key][0]
            elif i == len(frame_label[key]) - 1:
                cache.end = i
                seg.append(cache.toseg())
            elif frame_label[key][i] != cache.label:
                seg.append(cache.toseg())
                cache.label = frame_label[key][i]
                cache.start = i
                cache.end = i
            else:
                cache.end = i
        seg_label[key] = seg
    return seg_label


def f1_overlap(ratio, n_class, gt_seg, pr_seg):

    TP = np.zeros(n_class, np.float32)
    FP = np.zeros(n_class, np.float32)
    FN = np.zeros(n_class, np.float32)

    precision = np.zeros(n_class, np.float32)
    recall = np.zeros(n_class, np.float32)
    f1 = np.zeros(n_class, np.float32)

    for a in range(n_class):
        n_true = 0
        for key in gt_seg.keys():
            gt = gt_seg[key]
            pr = pr_seg[key]
            for seg in gt:
                if seg[0] != a:
                    continue
                else:
                    n_true += 1
                    gt_start = seg[1]
                    gt_end = seg[2]
                    tp_flag = 0
                    for segp in pr:
                        pr_label = segp[0]
                        pr_start = segp[1]
                        pr_end = segp[2]
                        if pr_end < gt_start:
                            continue
                        if  pr_start > gt_end:
                            break
                        if pr_label == a:
                            space = [gt_start, gt_end, pr_start, pr_end]
                            space.sort()
                            iou = (space[2]-space[1]+1)/(space[3]-space[0]+1)
                            if iou >= ratio and tp_flag == 0:
                                TP[a] += 1
                                tp_flag = 1
                            else:
                                FP[a] += 1
        FN[a] = n_true - TP[a]

        # precision[a] = TP[a] / (TP[a] + FP[a])
        # recall[a] = TP[a] / (TP[a] + FN[a])
        # if TP[a] != 0:
        #     f1[a] = 2 * (precision[a] * recall[a]) / (precision[a] + recall[a])
        # else:
        #     f1[a] = 0

    # f1_avg = np.mean(f1)
    # n_true = TP + FN
    # N = np.sum(n_true)
    # weight = n_true / N
    # f1_weight = np.sum(weight * f1)
    # f1_micro = np.mean(TP / (TP + FP))
    # print('f1@{}: {}'.format(ratio, f1_avg))
    # print('f1@{}: {}'.format(ratio, f1_weight))
    # print('f1@{}: {}'.format(ratio, f1_micro))

    TP_sum = np.sum(TP).item()
    FP_sum = np.sum(FP).item()
    FN_sum = np.sum(FN).item()

    p = TP_sum / (TP_sum + FP_sum)
    r = TP_sum / (TP_sum + FN_sum)

    f = 2*p*r/(p+r)
    print('f1@{0:.2f}: {1:.3%}'.format(ratio, f))


if __name__=='__main__':
    parser = get_parser()
    arg = parser.parse_args()
    # filename = './spa_psp_sub1_acc.json'
    # filename = 'agcn_psp_sub1_test.json'
    # filename = './cat_fcn_seg_res.json'
    # filename = './spa_psp_sub1_best.json'
    # filename = './spa_psp_sub3_loss.json'
    res = json.load(open(arg.filename, 'r'))
    preds = {}
    truth = {}
    for key, value in res.items():
        preds[key] = value['predict']
        truth[key] = value['ground_truth']
    framewise_eval(preds, truth)
    preds_seg = framewiselabel2seglabel(preds)
    truth_seg = framewiselabel2seglabel(truth)
    f1_overlap(0.1, 14, truth_seg, preds_seg)
    f1_overlap(0.25, 14, truth_seg, preds_seg)
    f1_overlap(0.5, 14, truth_seg, preds_seg)
    # f1_overlap(0.1, 32, truth_seg, preds_seg)
    # f1_overlap(0.25, 32, truth_seg, preds_seg)
    # f1_overlap(0.5, 32, truth_seg, preds_seg)
