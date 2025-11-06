import torch
import numpy as np  

def create_combination_mapping(original_classes, new_class_id):
    """
    Create a mapping for all possible combinations of original classes to new class labels.
    
    Args:
    - original_classes: The total number of original classes (default 11).
    
    Returns:
    - combination_map: A dictionary that maps each pair of class indices (c1, c2)
                        to a new class label.
    """
    combination_map = {}
    for i in range(original_classes):
        for j in range(i, original_classes):
            combination_map[(i, j)] = new_class_id
            new_class_id += 1
    return combination_map

def mixup_new(x, labels, alpha=1.0, combination_map=None):
    """
    Mix two batches and map the combination of temporal class labels to a new class label.
    
    Args:
    - x: tensor containing the input data (shape: [batch_size, length, features]).
    - labels: tensor containing the original class labels (shape: [batch_size, length]).
    - batch_size: The number of samples in the batch.
    - length: The temporal length of the sequences.
    - lam: The mixing ratio between two batches (default 0.5).
    - combination_map: A dictionary that maps each pair of class indices (c1, c2) to a new class label.
    
    Returns:
    - mixed_x: tensor containing the mixed batches (shape: [batch_size, length, features]).
    - new_labels: tensor containing the new class labels after the combination (shape: [batch_size, length]).
    """
    # Generate random permutation of indices to mix batches
    batch_size, length = labels.shape[0], labels.shape[1]
    index = torch.randperm(batch_size)
    # lam = np.random.beta(alpha, alpha)
    lam = 0.5
    
    # Mix the data across the temporal dimension
    mixed_x = lam * x + (1 - lam) * x[index, :, :]
    
    # Generate new labels based on combination for each time step
    new_labels = torch.zeros((batch_size, length))
    for i in range(batch_size):
        for t in range(length):
            original_label = labels[i, t].item()
            mixed_label = labels[index[i], t].item()
            
            if original_label == -1 or mixed_label == -1:
                new_labels[i, t] = -1
            else:
                # Determine the new label based on the combination
                # Ensure the lower class comes first in the tuple to match the combination_map
                if original_label <= mixed_label:
                    new_labels[i, t] = combination_map[(original_label, mixed_label)]
                else:
                    new_labels[i, t] = combination_map[(mixed_label, original_label)]

    return mixed_x, new_labels

def mixup_data(data, labels, alpha=1.0, only_objects=False, only_skeleton=False, index_start=12):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    x = data["motion"]
    batch_size = labels.shape[0]
    index = torch.randperm(batch_size)
    if only_objects and not only_skeleton:
        mixed_x = x
        objects = x[:, :, :, index_start:, :]
        noise = torch.randn(3, 1, 1)*alpha + 1.0
        objects = objects * noise
        mixed_x[:, :, :, index_start:, :] = objects
        lam = 1
    elif only_skeleton and not only_objects:
        mixed_x = x
        sk = x[:, :, :, :index_start, :]
        mixed_x[:, :, :, :index_start, :] = (lam * sk + (1 - lam) * sk[index, :, :, :, :])
    else:
        mixed_x = lam * x + (1 - lam) * x[index, :, :]
    y_a, y_b = labels, labels[index]
    data["motion"] = mixed_x
    return data, y_a, y_b, lam