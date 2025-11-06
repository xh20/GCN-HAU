import json
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import os

import argparse

label_list_bimacs = ['idle', 'approach', 'retreat', 'lift', 'place', 'hold', 'pour', 'cut', 'hammer', 'saw', 'stir',
                     'screw',
                     'drink', 'wipe']

label_list_ikea = ["pick up leg", "pick up table top", "pick up shelf", "pick up side panel", "pick up front panel",
                   "pick up back panel", "pick up bottom panel", "pick up pin", "lay down leg",
                   "lay down table top", "lay down shelf", "lay down side panel", "lay down front panel",
                   "lay down back panel", "lay down bottom panel", "push table top", "push table",
                   "align leg screw with table thread", "spin leg", "tighten leg", "rotate table",
                   "flip table top", "flip table", "flip shelf", "attach shelf to table",
                   "align side panel holes with front panel dowels", "attach drawer side panel",
                   "attach drawer back panel", "slide bottom of drawer", "insert drawer pin",
                   "position the drawer right side up",
                   "other"]
datasets = ['bimacs', 'ikea']


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')
    parser.add_argument(
        '--filename',
        default=' ',
        help='the saved filename')
    parser.add_argument(
        '--dataset',
        default='ikea',
        help='the dataset')

    return parser


class seg_cache():
    def __init__(self):
        self.label = 0
        self.start = 0
        self.end = 0

    def toseg(self):
        return ([self.label, self.start, self.end])


def framewise_eval(preds, truth, label_list):
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
            # gt with -1
            if t[i] >= 0:
                pr.append(p[i])
                tr.append(t[i])
                if p[i] == t[i]:
                    right += 1
                else:
                    wrong += 1

    cnf_matrix = confusion_matrix(tr, pr)
    # print(cnf_matrix)

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    cnf_matrix_n = confusion_matrix(tr, pr, normalize="true")
    plt.rcParams['font.size'] = '17'
    disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix_n, display_labels=label_list)
    disp.plot(cmap='Blues', xticks_rotation=45, values_format='.2f')
    # disp.plot(cmap='Blues', xticks_rotation=45, values_format='.2f', fontsize=17)
    # disp.plot(cmap='Blues', xticks_rotation='vertical', values_format='.2f', fontsize=17)
    # plt.show()
    fig = plt.gcf()
    fig.set_size_inches(15, 12)
    plt.savefig('confusion_matrix.png', dpi=200)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    f1 = TP / (TP + 0.5 * (FP + FN))
    f1_macro = f1.mean()
    True_class = TP + FN
    True_class[True_class == 0.0] = 1e-5
    recall = TP / True_class
    macro_recall = recall.mean()
    f1_macro_weighted = f1_score(tr, pr, average='weighted')
    acc = right / (right + wrong)
    top1 = TP.sum() / len(tr)

    print('macro-recall: {0:.3%}'.format(macro_recall))
    print('top1: {0:.3%}'.format(top1))
    print('f1 score macro: {0:.3%}'.format(f1_macro))
    print('f1 score weighted: {0:.3%}'.format(f1_macro_weighted))
    print('f1 score micro (accuracy): {0:.3%}'.format(acc))

    num_class = cnf_matrix.shape[0]
    return num_class


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

    seg_numbers = []
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
                        if pr_start > gt_end:
                            break
                        if pr_label == a:
                            space = [gt_start, gt_end, pr_start, pr_end]
                            space.sort()
                            iou = (space[2] - space[1] + 1) / (space[3] - space[0] + 1)
                            if iou >= ratio and tp_flag == 0:
                                TP[a] += 1
                                tp_flag = 1
                            else:
                                FP[a] += 1
        seg_numbers.append(n_true)
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

    N_segs = np.sum(seg_numbers).item()
    TP_sum = np.sum(TP).item()
    FP_sum = np.sum(FP).item()
    FN_sum = np.sum(FN).item()

    p = TP_sum / (TP_sum + FP_sum)
    r = TP_sum / (TP_sum + FN_sum)

    f = 2 * p * r / (p + r)
    print('f1@{0:.2f}: {1:.3%}'.format(ratio, f))


# new version
def get_label_start_end_time(frame_wise_labels, bg_class=[-1]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
    return labels, starts, ends


def f_score(predictions, ground_truth, overlap, bg_class=[-1]):
    gt_array = np.array(ground_truth).squeeze()
    pr = np.array(predictions).squeeze()[gt_array >= 0].tolist()
    gt = gt_array[gt_array >= 0].tolist()

    if len(gt) == 0:
        return 0.0, 0.0, 0.0

    p_label, p_start, p_end = get_label_start_end_time(pr, bg_class)
    y_label, y_start, y_end = get_label_start_end_time(gt, bg_class)

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0 * intersection / union) * ([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)


def get_f1_score(predictions, labels, overlap=0.1):
    TP = []
    FP = []
    FN = []
    for key in predictions.keys():
        tp, fp, fn = f_score(predictions[key], labels[key], overlap)
        TP.append(tp)
        FP.append(fp)
        FN.append(fn)
    TP = np.array(TP).sum()
    FP = np.array(FP).sum()
    FN = np.array(FN).sum()

    p = TP / (TP + FP)
    r = TP / (TP + FN)

    f = 2 * p * r / (p + r)
    print('f1@{0:.2f}: {1:.3%}'.format(overlap, f))


if __name__ == '__main__':
    parser = get_parser()
    arg = parser.parse_args()
    results_dir = './results/ikea/bpgcn_SN_max3_res_sn_max3_ds1/'
    # filename = './results/ikea/ctr_psp_test_ikea.json'
    # filename = './results/ikea/spa_multi_scale_test_ikea.json'
    # filename = 'test_ikea_46.json'
    # filename = 'test_ikea_bpgcn_SN_max3_res_no_sn_8.json' # best for sn_no_res IKEA
    filename = 'test_ikea_bpgcn_SN_max3_res_sn_max3_ds1_7.json'
    # filename = 'agcn_psp_sub1_test.json'
    # filename = './cat_fcn_seg_res.json'
    # filename = './spa_psp_sub1_best.json'
    # filename = './spa_psp_sub3_loss.json'
    if arg.dataset not in datasets:
        print('None dataset is found, please use bimacs or ikea dataset')
        exit(-1)
    if arg.dataset == 'bimacs':
        label_list = label_list_bimacs
    else:
        label_list = label_list_ikea

    print('reading log file: ', filename)
    res = json.load(open(os.path.join(results_dir, filename), 'r'))
    preds = {}
    predsGP = {}
    truth = {}
    gp = False
    for key, value in res.items():
        try:
            predsGP[key] = value['predictGP']
            preds[key] = value['predict']
            gp = True
        except:
            preds[key] = value['predict']
        truth[key] = value['ground_truth']
    num_class = framewise_eval(preds, truth, label_list)
    preds_seg = framewiselabel2seglabel(preds)
    truth_seg = framewiselabel2seglabel(truth)
    f1_overlap(0.1, num_class, truth_seg, preds_seg)
    f1_overlap(0.25, num_class, truth_seg, preds_seg)
    f1_overlap(0.5, num_class, truth_seg, preds_seg)
    if gp:
        print("GP: ")
        num_class = framewise_eval(predsGP, truth, label_list)
        preds_segGP = framewiselabel2seglabel(predsGP)
        f1_overlap(0.1, num_class, truth_seg, preds_segGP)
        f1_overlap(0.25, num_class, truth_seg, preds_segGP)
        f1_overlap(0.5, num_class, truth_seg, preds_segGP)

    print('Done!')
