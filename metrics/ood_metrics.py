# Utility functions to get OOD detection ROC curves and AUROC scores
# Ideally should be agnostic of model architectures

import torch
import torch.nn.functional as F
from sklearn import metrics
from scipy.linalg import sqrtm, det

from utils.ensemble_utils import ensemble_forward_pass
from metrics.classification_metrics import get_logits_labels
from metrics.uncertainty_confidence import entropy, logsumexp, confidence
import numpy as np


def get_roc_auc(net, test_loader, ood_test_loader, uncertainty, device, confidence=False):
    logits, _ = get_logits_labels(net, test_loader, device)
    ood_logits, _ = get_logits_labels(net, ood_test_loader, device)

    return get_roc_auc_logits(logits, ood_logits, uncertainty, device, confidence=confidence)


def get_roc_auc_logits(logits, logitsB, uncertainty, device, confidence=False):
    # logits: T x 14
    # uncertainties: T
    if isinstance(logits, list):
        logits = torch.cat(logits, dim=0)
    if isinstance(logitsB, list):
        logitsB = torch.cat(logitsB, dim=0)
    uncertainties = uncertainty(logits)
    uncertaintiesB = uncertainty(logitsB)

    # In-distribution
    bin_labels = torch.zeros(uncertainties.shape[0]).to(device)
    in_scores = uncertainties
    # 2 dataset
    bin_labels = torch.cat((bin_labels, torch.ones(uncertaintiesB.shape[0]).to(device)))

    if confidence:
        bin_labels = 1 - bin_labels
    scoresB = uncertaintiesB  # entropy(logitsB)
    scores = torch.cat((in_scores, scoresB))

    fpr, tpr, thresholds = metrics.roc_curve(bin_labels.cpu().numpy(), scores.cpu().numpy())
    precision, recall, prc_thresholds = metrics.precision_recall_curve(bin_labels.cpu().numpy(), scores.cpu().numpy())
    auroc = metrics.roc_auc_score(bin_labels.cpu().numpy(), scores.cpu().numpy())
    auprc = metrics.average_precision_score(bin_labels.cpu().numpy(), scores.cpu().numpy())

    return (fpr, tpr, thresholds), (precision, recall, prc_thresholds), auroc, auprc


def get_roc_auc_ensemble(model_ensemble, test_loader, ood_test_loader, uncertainty, device):
    bin_labels_uncertainties = None
    uncertainties = None

    for model in model_ensemble:
        model.eval()

    bin_labels_uncertainties = []
    uncertainties = []
    with torch.no_grad():
        # Getting uncertainties for in-distribution data
        for data, label in test_loader:
            data = data.to(device)
            label = label.to(device)

            bin_label_uncertainty = torch.zeros(label.shape).to(device)
            if uncertainty == "mutual_information":
                net_output, _, unc = ensemble_forward_pass(model_ensemble, data)
            else:
                net_output, unc, _ = ensemble_forward_pass(model_ensemble, data)

            bin_labels_uncertainties.append(bin_label_uncertainty)
            uncertainties.append(unc)

        # Getting entropies for OOD data
        for data, label in ood_test_loader:
            data = data.to(device)
            label = label.to(device)

            bin_label_uncertainty = torch.ones(label.shape).to(device)
            if uncertainty == "mutual_information":
                net_output, _, unc = ensemble_forward_pass(model_ensemble, data)
            else:
                net_output, unc, _ = ensemble_forward_pass(model_ensemble, data)

            bin_labels_uncertainties.append(bin_label_uncertainty)
            uncertainties.append(unc)

        bin_labels_uncertainties = torch.cat(bin_labels_uncertainties)
        uncertainties = torch.cat(uncertainties)

    fpr, tpr, roc_thresholds = metrics.roc_curve(bin_labels_uncertainties.cpu().numpy(), uncertainties.cpu().numpy())
    precision, recall, prc_thresholds = metrics.precision_recall_curve(
        bin_labels_uncertainties.cpu().numpy(), uncertainties.cpu().numpy()
    )
    auroc = metrics.roc_auc_score(bin_labels_uncertainties.cpu().numpy(), uncertainties.cpu().numpy())
    auprc = metrics.average_precision_score(bin_labels_uncertainties.cpu().numpy(), uncertainties.cpu().numpy())

    return (fpr, tpr, roc_thresholds), (precision, recall, prc_thresholds), auroc, auprc


def frechet_distance(mean1, mean2, cov1, cov2): # mean 14 x 512, cov 14 x 512 x 512
    mean_norm = np.linalg.norm(mean1 - mean2, ord=2, axis=-1) ** 2
    mat_cov = np.matmul(cov1, cov2)
    try: m = sqrtm(mat_cov)
    except:
        m = []
        for i in range(len(mat_cov)):
            m.append(sqrtm(mat_cov[i]))
        m = np.stack(m)
    cov = cov1 + cov2 - 2.0 * m
    trace = np.trace(cov, offset=0, axis1=-2, axis2=-1)
    return np.real(mean_norm + trace).tolist()

def KL_divergence(mean1, mean2, cov1, cov2):
    # manual implemented KL_divergence for 2 MVN according to arxiv 2102.05485, KL(N1||N2)
    try:
        det1 = det(cov1*10)
        det2 = det(cov2*10)
        distance = np.sum(np.matmul((mean1 - mean2) ** 2, np.linalg.pinv(cov2)), axis=-1)
    except:
        det1, det2, distance = [], [], []
        for i in range(len(cov1)):
            det1.append(det(cov1[i]*10))
            det2.append(det(cov2[i]*10))
            distance.append(np.sum(np.matmul((mean1[i] - mean2[i]) ** 2, np.linalg.pinv(cov2[i])), axis=-1))
        det1, det2 = np.stack(det1), np.stack(det2)
        distance = np.stack(distance)
    trace = np.trace(np.matmul(np.linalg.pinv(cov2), cov1), offset=0, axis1=-2, axis2=-1)
    d = cov1.shape[-1]
    kl = 0.5 * (np.log(det2) - np.log(det1) - d + trace + distance)
    return kl.tolist()