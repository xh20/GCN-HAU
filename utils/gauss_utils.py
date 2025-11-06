import torch
from torch import nn
from tqdm import tqdm
import numpy as np
from torch.autograd import Variable
# from torch.distributions.multivariate_normal import MultivariateNormal

DOUBLE_INFO = torch.finfo(torch.double)
JITTERS = [0, DOUBLE_INFO.tiny] + [10 ** exp for exp in range(-308, 0, 1)]
JITTERS_2 = [exp/10.0 for exp in range(1, 10, 1)]


def centered_cov_torch(x):
    n = x.shape[0]
    res = 1 / (n - 1) * np.matmul(x.transpose(1, 0), x)
    # res_diag = np.diagonal(res)
    # res = np.diag(res_diag)
    return res


def save_features_distribution(features, labels, num_classes):
    import matplotlib.pyplot as plt
    import seaborn as sb
    import os
    range_config = dict(bins=50, element="step", fill=True, alpha=0.7)
    for c in range(num_classes):
        classwise_feature = features[labels == c]
        folder_name = '/media/dataset/Human_Object_Interaction_Segmentation/segmentation/features_distribution/max3/'\
                      + str(c)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        for j in range(classwise_feature.shape[1]):
            plt.figure(figsize=(2.5, 2.5 / 1.6))
            plt.tight_layout()
            sb.histplot(data=classwise_feature[:, j], color=sb.color_palette("tab10")[0],
                        stat='probability', kde=False, **range_config, label="dummy",
                        legend=False)
            figure_name = os.path.join(folder_name, 'feature_' + str(j) + '.png')
            plt.savefig(figure_name, bbox_inches='tight', dpi=300)
            plt.close()


def get_embeddings(
    net, loader: torch.utils.data.DataLoader, dtype, device, storage_device,
):
    num_samples = len(loader.dataset)
    # embeddings = torch.empty((num_samples, num_dim[0], num_dim[1]), dtype=dtype, device=storage_device)
    # labels = torch.empty((num_samples, num_dim[1]), dtype=torch.int, device=storage_device)
    embeddings = []
    labels = []
    net.eval()
    with torch.no_grad():
        start = 0
        for data, label, _ in tqdm(loader):
            data = data.to(device)
            if isinstance(label, list):
                label = torch.stack(label).cuda(device=device)
                label = label.transpose(1, 0)
            # label = torch.tensor(label, device=storage_device)
            if isinstance(net, nn.DataParallel):
                _ = net.module(data)
                out = net.module.feature
            else:
                _ = net(data)
                out = net.feature
            # if isinstance(net, nn.DataParallel):
            #     out = net.module(data)
            #     _ = net.module.feature
            # else:
            #     out = net(data)
            #     _ = net.feature

            # ignoring -1 for ikea
            # label_ = label.clone().squeeze()
            # out = out[:, :, label_ >= 0, :]
            # label = label[:, label_ >= 0]

            # end = start + len(data)
            # embeddings[start:end].copy_(out, non_blocking=True)
            # labels[start:end].copy_(label, non_blocking=True)
            # start = end
            embeddings.append(out.cpu().data.numpy())
            labels.append(label.cpu().data.numpy())

    return embeddings, labels


def gmm_forward(net, gmm, data_B_X):
    net.eval()
    log_probs = []
    log_probs_weighted = []
    with torch.no_grad():
        if isinstance(net, nn.DataParallel):
            logits_softmax, features_B_Z = net.module(data_B_X)
            # features_B_Z = net.module.feature
            # features_B_Z = logits_softmax
        else:
            # logits_softmax, features_B_Z = net(data_B_X)
            logits_softmax = net(data_B_X)
            features_B_Z = net.feature
            # features_B_Z = logits_softmax
        # features: 1 x C x T x 1
        features_B_Z = features_B_Z.squeeze().permute(1, 0)
        logits_softmax = logits_softmax.squeeze().permute(1, 0)
        for gaussians_model in gmm:
            log_probs_B_Y = gaussians_model.log_prob(features_B_Z[:, None, :])
            # log_probs_B_Y_weighted = gaussians_model.log_prob_weighted(features_B_Z[:, None, :])
            log_probs.append(log_probs_B_Y)
            # log_probs_weighted.append(log_probs_B_Y_weighted)
        # log_probs_B_Y: 14 x T
    return torch.cat(log_probs, dim=1), logits_softmax, features_B_Z


def gmm_evaluate(net, gaussians_model, loader, device, num_classes, storage_device):

    # num_samples = len(loader.dataset)
    # logits_N_C = torch.empty((num_samples, num_classes), dtype=torch.float, device=storage_device)
    # labels_N = torch.empty(num_samples, dtype=torch.int, device=storage_device)
    logits_gauss_probability = []
    features_gauss = []
    labels_N = []
    logits_softmax = []
    net.eval()
    with torch.no_grad():
        # start = 0
        for data, label, _ in tqdm(loader):
            data = Variable(data.float().cuda(device), requires_grad=False)
            if torch.is_tensor(label):
                label = label.clone().detach().to(dtype=int).requires_grad_(False)
            else:
                label = Variable(torch.tensor(label, dtype=int).unsqueeze(0).cuda(device), requires_grad=False)
            if label.shape[0] != 1:
                label = label.unsqueeze(0)
            if isinstance(label, list):
                label = torch.stack(label).cuda(device=device)
                label = label.transpose(1, 0)
            logit_features_probability, logit_soft, features = gmm_forward(net, gaussians_model, data)
            if -1 in label:
                features = features[label.squeeze() >= 0, :]
                logit_soft = logit_soft[label.squeeze() >= 0, :]
                logit_features_probability = logit_features_probability[label.squeeze() >= 0, :]
                label = label[:, label.squeeze() >= 0]
            # end = start + len(data)
            # logits_N_C[start:end].copy_(logit_B_C, non_blocking=True)
            # labels_N[start:end].copy_(label, non_blocking=True)
            # start = end
            logits_gauss_probability.append(logit_features_probability)
            features_gauss.append(features)
            logits_softmax.append(logit_soft)
            labels_N.append(label)
        logits_gauss_probability = torch.cat(logits_gauss_probability, dim=1)
        features_gauss = torch.cat(features_gauss, dim=1)
        labels_N = torch.cat(labels_N, dim=1)
        logits_softmax = torch.cat(logits_softmax, dim=1)
    return logits_gauss_probability, labels_N, logits_softmax, features_gauss


def gmm_get_logits(gmm, embeddings):
    log_probs_B_Y = gmm.log_prob(embeddings[:, None, :])
    return log_probs_B_Y


def get_gmm(classwise_mean_features, classwise_cov_features, num_classes, jitter_eps=0.1, device=None):
    # mean: num_classes x C
    classwise_mean_features = torch.tensor(classwise_mean_features, device=device)
    # cov: num_classes x C x C
    classwise_cov_features = torch.tensor(classwise_cov_features, device=device)
    gmm = []
    jitter_eps_list = []
    with torch.no_grad():
        for c in range(num_classes):
            # tumhoi dataset
            # try:
            #     diagonal_array = jitter_eps * torch.abs(classwise_cov_features[c].diagonal())
            #     # print(diagonal_array)
            #     # diagonal_array[diagonal_array > 0.05] = 0.05
            #     # print(diagonal_array)
            #     jitter = torch.eye(classwise_cov_features.shape[1], device=device)
            #     jitter.diagonal().copy_(diagonal_array)
            #     gmm_c = torch.distributions.MultivariateNormal(
            #         loc=classwise_mean_features[c].squeeze(),
            #         covariance_matrix=(classwise_cov_features[c].squeeze() + jitter))
            # except RuntimeError as e:
            #     continue
            # except ValueError as e:
            #     if "The parameter covariance_matrix has invalid values" in str(e):
            #         continue
            #     break
            # jitter_eps = 0.07
            # ikea 0.001, bimacs 0.1
            jitter_eps = 0.1
            try:
                jitter = jitter_eps * torch.eye(classwise_cov_features.shape[1], device=device).unsqueeze(0)
                gmm_c = torch.distributions.MultivariateNormal(
                    loc=classwise_mean_features[c], covariance_matrix=(classwise_cov_features[c] + jitter))
            except RuntimeError as e:
                return -1
            except ValueError as e:
                if "The parameter covariance_matrix has invalid values" in str(e):
                    return -1
            gmm.append(gmm_c)
            jitter_eps_list.append(jitter_eps)
    return gmm, jitter_eps_list


def gmm_fit(embeddings, labels, num_classes, device=None):
    with torch.no_grad():
        if type(embeddings) is list:
            # BS x T x C
            features = np.concatenate(embeddings, axis=0)
            if len(features.shape) == 4:
                features = features.transpose(0, 2, 1, 3).squeeze()
            elif len(features.shape) == 3:
                features = features.transpose(0, 2, 1).squeeze()
            else:
                return -1
        if type(labels) is list:
            labels = np.concatenate(labels, axis=0)

        # save_features_distribution(features, labels, num_classes)

        classwise_mean_features = np.stack([np.mean(features[labels == c], axis=0) for c in range(num_classes)])
        classwise_cov_features = np.stack([centered_cov_torch(features[labels == c] - classwise_mean_features[c])
                                           for c in range(num_classes)])

    # return get_gmm(classwise_mean_features, classwise_cov_features, num_classes, device)
    return classwise_mean_features, classwise_cov_features


def centered_cov(x):
    n = x.shape[0]
    res = 1 / (n - 1) * torch.matmul(x.permute(1, 0), x)
    return res


def pos_definit_cov(cov_matrix, device=None):
    with torch.no_grad():
        for jitter_eps in JITTERS:
            try:
                jitter = jitter_eps * torch.eye(cov_matrix.shape[1], device=device)
                mean = torch.ones(cov_matrix.shape[1], device=device)
                if len(cov_matrix.shape) != 2: jitter = jitter.unsqueeze(0)
                gmm = torch.distributions.MultivariateNormal(
                    loc=mean, covariance_matrix=(cov_matrix + jitter))
                res = cov_matrix + jitter
            except RuntimeError as e:
                if "cholesky" in str(e):
                    continue
            except ValueError as e:
                if "The parameter covariance_matrix has invalid values" in str(e):
                    continue
            # break
    gmm = torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=(res))
    return res


def test_feature_gmm(num_class, labels, features, device=None):
    with torch.no_grad():
        labels = torch.cat(labels, dim=0).squeeze()
        features = torch.cat(features, dim=0)  # T x 512
        mean_features = torch.mean(features, dim=0)  # 512
        cov_features = centered_cov(features - mean_features) # 512 x 512
        classwise_mean_features = torch.stack([torch.mean(features[labels == c], dim=0) for c in range(num_class)])
        classwise_cov_features = torch.stack([centered_cov(features[labels == c] - classwise_mean_features[c])
                                           for c in range(num_class)])
        cov_features = pos_definit_cov(cov_features, device=device)
        classwise_cov_features = pos_definit_cov(classwise_cov_features, device=device)

    # 512, 512 x 512, 14 x 512, 14 x 512 x 512
    return mean_features, cov_features, classwise_mean_features, classwise_cov_features

