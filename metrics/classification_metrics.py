"""
Metrics to measure classification performance
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from utils.ensemble_utils import ensemble_forward_pass

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from fvcore.nn import FlopCountAnalysis


def get_logits_labels(model, data_loader, device):
    """
    Utility function to get logits and labels.
    """
    model.eval()
    logits = []
    labels = []

    with torch.no_grad():
        for _, (data, label, _) in enumerate(data_loader):
            with torch.no_grad():
                data = Variable(data.float().cuda(device), requires_grad=False)
                label = Variable(torch.tensor(label, dtype=int).unsqueeze(0).cuda(device), requires_grad=False)
                if label.shape[0] != 1:
                    label = label.unsqueeze(0)
                logit = model(data)
                # dtype=float).cuda(output_device)
                # flops = FlopCountAnalysis(model, data)
                # print(flops.total())
                if type(logit) is tuple:
                    # duq[0]: prediction 1x14xT, duq[1]: output from the original model: 1x14xT
                    logit = logit[0]
                if -1 in label:
                    logit = logit[:, :, label.squeeze() >= 0]
                    label = label[:, label.squeeze() >= 0]
            logits.append(logit)
            labels.append(label)
    logits = torch.cat(logits, dim=0)
    labels = torch.cat(labels, dim=0)
    return logits, labels


def test_classification_net_softmax(softmax_prob, labels):
    """
    This function reports classification accuracy and confusion matrix given softmax vectors and
    labels from a model.
    """
    labels_list = []
    predictions_list = []
    confidence_vals_list = []

    confidence_vals, predictions = torch.max(softmax_prob, dim=1)
    labels_list.extend(np.squeeze(labels.cpu().numpy()))
    predictions_list.extend(np.squeeze(predictions.cpu().numpy()))
    confidence_vals_list.extend(np.squeeze(confidence_vals.cpu().numpy()))
    accuracy = accuracy_score(labels_list, predictions_list)
    return (confusion_matrix(labels_list, predictions_list), accuracy, labels_list, predictions_list,
            confidence_vals_list)


def test_classification_net_logits(logits, labels):
    """
    This function reports classification accuracy and confusion matrix given logits and labels
    from a model.
    """
    softmax_prob = F.softmax(logits, dim=1)
    return test_classification_net_softmax(softmax_prob, labels)


def test_classification_net(model, data_loader, device):
    """
    This function reports classification accuracy and confusion matrix over a dataset.
    """
    logits, labels = get_logits_labels(model, data_loader, device)
    return test_classification_net_logits(logits, labels)


def test_classification_net_ensemble(model_ensemble, data_loader, device):
    """
    This function reports classification accuracy and confusion matrix over a dataset
    for a deep ensemble.
    """
    for model in model_ensemble:
        model.eval()
    softmax_prob = []
    labels = []
    with torch.no_grad():
        for data, label, _ in data_loader:
            data = data.to(device)
            label = label.to(device)
            softmax, _, _ = ensemble_forward_pass(model_ensemble, data)
            softmax_prob.append(softmax)
            labels.append(label)
    softmax_prob = torch.cat(softmax_prob, dim=0)
    labels = torch.cat(labels, dim=0)

    return test_classification_net_softmax(softmax_prob, labels)
