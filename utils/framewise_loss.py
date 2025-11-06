import torch
import torch.nn as nn
from torch.nn.modules.loss import BCELoss
from torch.nn import functional as F


class TaskGraphLoss(nn.Module):
    def __init__(self, exclude_self_transitions=True, num_class=10):
        super(TaskGraphLoss, self).__init__()
        self.exclude_self_transitions = exclude_self_transitions
        self.num_classes = num_class
        self.mse_loss = nn.MSELoss()


    def compute_adjacency_matrix_batch(self, actions_onehot, onehot=True):
        """
        Compute the adjacency matrix from a batch of one-hot encoded action labels, optionally excluding self-transitions.

        Parameters
        ----------
        actions_onehot : torch.Tensor
            Tensor of shape (B, C, T), where B is the batch size, C is the number of unique actions,
            and T is the sequence length (time steps).
        exclude_self_transitions : bool
            Whether to exclude self-transitions (i.e., transitions where the current action is the same as the next action).

        Returns
        -------
        torch.Tensor
            Adjacency matrix of shape (C, C) where A[i, j] represents the number of times
            action i transitions to action j across all sequences in the batch.
        """
        # Step 1: Convert one-hot encoded tensor to action indices (integer labels) for each sequence
        # actions_onehot: (B, C, T) -> action_indices: (B, T)
        if onehot:
            action_indices = torch.argmax(actions_onehot, dim=1)  # shape (B, T)
        else:
            action_indices = actions_onehot

        # Step 2: Initialize an empty adjacency matrix
        B, T = action_indices.shape
        C = self.num_classes  # The number of unique actions
        adjacency_matrix = torch.zeros((B, C, C), dtype=torch.int32, device=action_indices.device)

        # Step 3: Count transitions between consecutive actions for each sequence
        for b in range(B):  # Loop through each sequence in the batch
            for t in range(T - 1):  # Loop through each timestep (except the last)
                current_action = int(action_indices[b, t])
                next_action = int(action_indices[b, t + 1])

                # Optionally exclude self-transitions
                if self.exclude_self_transitions and current_action == next_action:
                    continue  # Skip self-transitions

                adjacency_matrix[b, current_action, next_action] += 1

        return adjacency_matrix

    def forward(self, predictions, actions_label, onehot=True):
        adjacent_matrix = self.compute_adjacency_matrix_batch(actions_label, onehot)
        true_adj = adjacent_matrix.float()  # Create a copy of true_adj to modify
        true_adj /= (adjacent_matrix.sum(dim=-1, keepdim=True) + 1e-8)
        # Step 1: Apply softmax to the predicted adjacency matrix along the rows

        value, predicted_action = torch.max(predictions, 1)
        prediction_adj = self.compute_adjacency_matrix_batch(predicted_action, onehot=False)
        prediction_adj /= (prediction_adj.sum(dim=-1, keepdim=True) + 1e-8)
        prediction_adj.requires_grad_(True)
        # prediction_adj = predictions
        # loss = -torch.mean(true_adj * torch.log(prediction_adj + 1e-8))
        # loss_mse = self.mse_loss(prediction_adj, true_adj)
        # loss_l1 = F.l1_loss(prediction_adj, true_adj)
        # loss_l1_smooth = F.smooth_l1_loss(prediction_adj, true_adj)

        # Create a binary ground truth for zero/non-zero classification
        non_zero_mask = true_adj != 0
        binary_ground_truth = (non_zero_mask).float()  # 1 for non-zero, 0 for zero
        # Predict probability of non-zero with sigmoid
        # dense_pred_prob = torch.sigmoid(prediction_adj)
        dense_pred_prob = torch.tanh(prediction_adj)
        softmax_pred = torch.softmax(prediction_adj, dim=-1)
        # BCE loss for zero/non-zero classification
        bce_loss = F.binary_cross_entropy(dense_pred_prob, binary_ground_truth)

        # Optionally, combine with MSE for non-zero values
        if non_zero_mask.sum() > 0:
            loss_non_zero = F.mse_loss(prediction_adj[non_zero_mask], true_adj[non_zero_mask])
        else:
            loss_non_zero = 0
        # Combined loss
        combined_loss = bce_loss + loss_non_zero
        # combined_loss = loss_non_zero
        return combined_loss


class TaskGraphDistanceLoss(nn.Module):
    def __init__(self, exclude_self_transitions=True, num_class=10):
        super(TaskGraphDistanceLoss, self).__init__()
        self.exclude_self_transitions = exclude_self_transitions
        self.num_classes = num_class
        self.loss_fn = nn.MSELoss()

    def compute_adjacency_matrix_batch(self, labels, onehot=True):
        # actions_onehot: (B, C, T) -> action_indices: (B, T)
        if labels.ndim == 2:
            onehot = False
        elif labels.ndim == 3:
            onehot = True
        # if labels.dtype != torch.int64:
        #     labels = labels.type(torch.int64)
        if onehot:
            label_onehot = labels  # shape (B, T)
        else:
            label_onehot = F.one_hot(labels.to(torch.long), num_classes=self.num_classes).permute(0, 2, 1).float()
        label_changes = label_onehot[:, :, 1:] - label_onehot[:, :, :-1]
        gt_adj = (label_changes.unsqueeze(2) - label_changes.unsqueeze(1)).sum(-1)
        return gt_adj

    def forward(self, predictions, actions_label, onehot=True):
        gt_adj = self.compute_adjacency_matrix_batch(actions_label, onehot)
        # value, pred_adj = torch.max(predictions, 1)
        loss = self.loss_fn(predictions, gt_adj)
        return loss


class TaskLikelihoodLoss(nn.Module):
    def __init__(self, exclude_self_transitions=True, num_class=10):
        super(TaskLikelihoodLoss, self).__init__()
        self.exclude_self_transitions = exclude_self_transitions
        self.num_classes = num_class

    def task_graph_rate(self, sequence, A, all_nodes, beta):
        """
        Description
        -----------
        Compute the rate of a sequence of nodes in a task graph.

        Parameters
        ----------
        - sequence : torch.Tensor (B, T)
            The sequence of action indices per batch.
        - A : torch.Tensor (B, C, C)
            The adjacency matrix of the task graph (B, C, C).
        - all_nodes : torch.Tensor (C,)
            The list of all nodes (action labels).
        - beta : float
            The beta parameter.

        Returns
        -------
        - **float**: The rate of the sequence.
        """

        B, T = sequence.shape
        C = A.shape[1]

        # Compute the task graph rate in a batch-wise way
        rates = torch.zeros(B, device=A.device)  # Initialize rates for all batches

        for i in range(1, T - 1):  # Iterate through timesteps
            current_actions = sequence[:, i]  # Shape (B,)
            prev_actions = sequence[:, i - 1]  # Shape (B,)

            # Ignore transitions where the current action is the same as the previous one (self-connection)
            non_self_transitions = current_actions != prev_actions  # Shape (B,)

            if non_self_transitions.any():  # Only process if there are valid transitions
                valid_batches = non_self_transitions.nonzero(as_tuple=True)[0]  # Get indices of valid transitions

                current_actions = current_actions[valid_batches]  # Filter valid batches for current actions
                J = sequence[valid_batches, :i]  # Previous actions in valid batches
                # mask = torch.ones(len(valid_batches), C, dtype=bool, device=A.device)
                # mask.scatter_(1, J, False)
                # num = A[valid_batches, current_actions].gather(2, J.unsqueeze(-1)).sum(dim=1)
                # notJ_mask = mask.unsqueeze(-1)  # Shape (valid_batches, C, 1)
                # J_expanded = J.unsqueeze(1).expand(-1, C, -1)  # Expand J to match notJ dimension
                # den = torch.masked_select(A[valid_batches], notJ_mask).view(len(valid_batches), C, -1)
                # den = den.gather(2, J_expanded).sum(dim=(1, 2))

                # Numerator: Sum A[current_actions, J] over valid batches
                num = torch.zeros(len(valid_batches), device=A.device)
                for batch_idx, batch in enumerate(valid_batches):
                    current_action = current_actions[batch_idx]  # Scalar
                    prev_actions_in_batch = J[batch_idx]  # Shape (i,)
                    num[batch_idx] = A[batch, current_action, prev_actions_in_batch].sum()

                # Denominator: Sum A[notJ, J] over valid batches
                den = torch.zeros(len(valid_batches), device=A.device)
                for batch_idx, batch in enumerate(valid_batches):
                    prev_actions_in_batch = J[batch_idx]  # Shape (i,)
                    notJ = all_nodes[~torch.isin(all_nodes, prev_actions_in_batch)]  # Shape (C - i,)
                    den[batch_idx] = A[batch, notJ][:, prev_actions_in_batch].sum()

                # Calculate rates for valid batches only
                den_ = torch.max(den, num)
                num_ = torch.min(num, den)
                rates[valid_batches] += beta * torch.log(den_ + 1e-8) - torch.log(num_ + 1e-8)

        return rates

    def task_graph_maximum_likelihood_loss(self, y, A, beta):
        """
        Description
        -----------
        Compute the maximum likelihood loss of a sequence of nodes in a task graph.

        Parameters
        ----------
        - y : torch.Tensor (B, C, T)
            The ground truth action sequence (one-hot encoded).
        - A : torch.Tensor (B, C, C)
            The predicted adjacency matrix.
        - beta : float
        The beta parameter.

        Returns
        -------
        - **torch.Tensor**: The loss of the sequence.
        """

        all_nodes = torch.arange(self.num_classes, device=A.device)  # All possible action labels

        # Convert ground truth one-hot action labels to indices (B, T)
        sequences = y  # Shape (B, T)

        # Compute the task graph rate for all sequences in the batch
        rates = self.task_graph_rate(sequences, A, all_nodes, beta)

        # Sum over all batches to get the total loss
        total_loss = rates.sum()
        return total_loss

    def forward(self, pred_adjacency_matrix, actions_label, onehot=True):
        if onehot:
            gt = torch.argmax(actions_label, dim=1)
        else:
            gt = actions_label
        loss = self.task_graph_maximum_likelihood_loss(gt, pred_adjacency_matrix, beta=1.0)
        return loss.mean()


class TemporalMultiClassDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(TemporalMultiClassDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        """
        preds: [B, C, T] -> raw logits (before softmax)
        targets: [B, C, T] -> ground truth class labels (0 to C-1)
        """
        # Flatten preds and targets across batch and time (B * C * T)
        preds_flat = preds.view(-1)  # Shape: [B * C * T]
        targets_flat = targets.view(-1)  # Shape: [B * C * T]

        # Calculate intersection and union
        intersection = (preds_flat * targets_flat).sum()
        dice_score = (2. * intersection + self.smooth) / (preds_flat.sum() + targets_flat.sum() + self.smooth)

        return 1 - dice_score  # Dice Loss is 1 - Dice Coefficient


class SegmentationLoss(nn.CrossEntropyLoss):
    def __init__(self, num_class=14, reduction='mean'):
        super(SegmentationLoss, self).__init__()
        self.num_class = num_class
        self.bceloss = BCELoss(reduction=reduction)

    def forward(self, res, tar):
        # print(res.size(), tar.size())
        return super(SegmentationLoss, self).forward(res, tar)


class SmoothLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SmoothLoss, self).__init__()

    def forward(self, predictions):
        loss = torch.mean((predictions[:, :, 1:] - predictions[:, :, :-1]) ** 2)
        return loss

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1e-4):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, predictions, labels):
        B, C, T = predictions.shape

