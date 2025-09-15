from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import pytorch3d.ops

from scipy.optimize import linear_sum_assignment

from src.utils.rendering import get_render_settings, render_gaussian_model_features
from src.utils.utils import numpy_to_torch, torch_to_numpy

def compute_mean_segmentation_mask_features(feature_map: torch.Tensor, instance_segmentation: torch.Tensor, normalize: bool = False):
    """
    Estimate the mean features from the feature map for each instance segmentation mask.

    Args:
        feature_map: The feature map of shape (H, W, C).
        instance_segmentation: The instance segmentation of shape (H, W).
    Returns:
        torch.Tensor: The cluster centers of shape (num_instances, C).
    """
    # # normalize features
    feature_map = F.normalize(feature_map, p=2, dim=2)
    
    # determine unique instances, obtain their masks and compute respective mean features
    instance_ids = torch.unique(instance_segmentation)
    instance_masks = torch.stack([instance_segmentation == id for id in instance_ids])
    mean_features = torch.stack([feature_map[mask].mean(dim=0) for mask in instance_masks])
    if normalize:
        mean_features = F.normalize(mean_features, p=2, dim=1)

    return mean_features

def perform_entity_association(
    entities: Optional[np.ndarray],
    entity_labels: Optional[list],
    candidate_entities: np.ndarray, 
    candidate_entity_labels: list, 
    assignment_threshold: float = 0.8
) -> None:
    if entities is not None:
        assert entities.shape[0] == len(entity_labels)
    assert candidate_entities.shape[0] == len(candidate_entity_labels)

    if entities is None:
        # if no entities have been segmented yet, add all candidates as entities
        entities = candidate_entities
        entity_labels = candidate_entity_labels
    else:
        
        new_entities = []
        new_entity_labels = []

        for label in list(set(candidate_entity_labels)):
            if label == "unknown":
                continue
            if label in entity_labels:
                existing_idces = np.where(np.array(entity_labels) == label)[0]
                candidate_idces = np.where(np.array(candidate_entity_labels) == label)[0]

                # evaluate alignment between candidates and existing entities
                correlation_matrix = entities[existing_idces] @ candidate_entities[candidate_idces].T
                if correlation_matrix.shape == ():
                    if correlation_matrix > assignment_threshold:
                        # merge entities; requires just combining the vectors
                        # TODO: apply some sort of smoothing
                        entities[existing_idces] = (entities[existing_idces] + candidate_entities[candidate_idces]) / 2
                    else:
                        new_entities.append(candidate_entities[candidate_idces])
                        new_entity_labels.append(label)
                else:
                    # associate candidate entities with existing entities
                    correspondence_existing, correspondence_candidate = linear_sum_assignment(-correlation_matrix)

                    # check associations
                    for i, j in zip(correspondence_existing, correspondence_candidate):
                        if correlation_matrix[i, j] > assignment_threshold:
                            # merge entities; requires just combining the vectors
                            entities[existing_idces[i]] = (entities[existing_idces[i]] + candidate_entities[candidate_idces[j]]) / 2
                        else:
                            new_entities.append(candidate_entities[candidate_idces[j]])
                            new_entity_labels.append(label)

                    # add unassigned candidates as new entities #TODO: merge here also
                    for j in range(len(candidate_idces)):
                        if j not in correspondence_candidate:
                            new_entities.append(candidate_entities[candidate_idces[j]])
                            new_entity_labels.append(label)
            else:
                new_entities.append(candidate_entities[candidate_entity_labels.index(label)])
                new_entity_labels.append(label)
    
        # update segmented entities and annotations
        if len(new_entities) > 0:
            entities = np.vstack((entities, np.vstack(new_entities)))
            entity_labels += new_entity_labels

        assert entities.shape[0] == len(entity_labels)

    return entities, entity_labels

def perform_entity_discovery(feature_maps, segmentation_images, detection_labels, assignment_threshold=0.8):
    # initialize storage for entities and labels
    entities = None
    entity_labels = None

    for i, (feature_map, segmentation_image, labels) in enumerate(zip(feature_maps, segmentation_images, detection_labels)):
        # compute mean features for current frame
        mean_features = compute_mean_segmentation_mask_features(feature_map, segmentation_image, normalize=True) # it is important to normalize this

        candidate_entities = torch_to_numpy(mean_features)
        candidate_entity_labels = labels
        entities, entity_labels = perform_entity_association(
            entities, 
            entity_labels, 
            candidate_entities, 
            candidate_entity_labels, 
            assignment_threshold
        )

    return entities, entity_labels

# def merge_duplicate_entities(entities: np.ndarray, entity_labels: list, merge_threshold: float = 0.9):
#     autocorrelation_matrix = entities @ entities.T
#     entity_types = list(set(entity_labels))

#     for entity_type in entity_types:
#         indices = np.where(np.array(entity_labels) == entity_type)[0]

#         if len(indices) < 2:
#             continue
        
#         while True:
#             # Extract the submatrix corresponding to these indices
#             submatrix = autocorrelation_matrix[np.ix_(indices, indices)]

#             # check if merges need to be performed
#             submatrix += np.diag(-np.inf * np.ones(submatrix.shape[0]))
#             max_value = np.max(np.triu(submatrix))
#             if max_value < merge_threshold:
#                 break
#             max_value_index = np.unravel_index(np.argmax(submatrix), submatrix.shape)

#             # merge
#             merge_entity_1 = entities[indices[max_value_index[0]]]
#             merge_entity_2 = entities[indices[max_value_index[1]]]
#             entities[indices[max_value_index[0]]] = (merge_entity_1 + merge_entity_2) / 2
#             entities = np.delete(entities, indices[max_value_index[1]], axis=0)
#             entity_labels.pop(indices[max_value_index[1]])

#             # update submatrix
#             indices = np.where(np.array(entity_labels) == entity_type)[0]
#             submatrix = autocorrelation_matrix[np.ix_(indices, indices)]

#     return entities, entity_labels

def merge_duplicate_entities(entities: np.ndarray, entity_labels: list, merge_threshold: float = 0.9):
    autocorrelation_matrix = entities @ entities.T
        
    while True:
        # Extract the submatrix corresponding to these indices
        submatrix = autocorrelation_matrix.copy()
        # check if merges need to be performed
        submatrix += np.diag(-np.inf * np.ones(autocorrelation_matrix.shape[0]))
        max_value = np.max(np.triu(submatrix))
        if max_value < merge_threshold:
            break
        max_value_index = np.unravel_index(np.argmax(submatrix), submatrix.shape)

        # merge
        merge_entity_1 = entities[max_value_index[0]]
        merge_entity_2 = entities[max_value_index[1]]
        entities[max_value_index[0]] = (merge_entity_1 + merge_entity_2) / 2
        entities = np.delete(entities, max_value_index[1], axis=0)
        entity_labels.pop(max_value_index[1])

        # update submatrix, remove column from autocorrelation matrix
        autocorrelation_matrix = np.delete(autocorrelation_matrix, max_value_index[1], axis=0)
        autocorrelation_matrix = np.delete(autocorrelation_matrix, max_value_index[1], axis=1)

    return entities, entity_labels

def segment_point_cloud(point_features: torch.Tensor, entity_feature_prototypes: torch.Tensor):
    # compute correlation matrix
    correlation_matrix = point_features @ entity_feature_prototypes.T

    # assign points to entities
    out = torch.max(correlation_matrix, axis=1)
    point_classes, scores = out.indices, out.values

    return point_classes, scores

def knn_point_cloud_smoothing(point_xyzs: torch.Tensor, point_classes: torch.Tensor, k: int, features: Optional[torch.Tensor] = None, method: str = "majority"):
    knn = pytorch3d.ops.knn_points(
        point_xyzs.unsqueeze(0), 
        point_xyzs.unsqueeze(0), 
        K=k
    )
    
    nearest_k_idx = knn.idx.squeeze(0)
    nearest_k_distances = knn.dists.squeeze(0)[:, 1:]
    nearest_k_neighbor_classes = point_classes[nearest_k_idx[:, 1:]]

    if method == "majority":
        num_classes = len(point_classes.unique())
        one_hot = F.one_hot(nearest_k_neighbor_classes, num_classes=num_classes)
        class_counts = one_hot.sum(dim=1)
        max_count, max_class = class_counts.max(dim=1)

        neighbor_majority_class = max_class

        # return neighbor_majority_class
        return class_counts

    elif method == "supermajority":
        num_classes = len(point_classes.unique())
        one_hot = F.one_hot(nearest_k_neighbor_classes, num_classes=num_classes)
        class_counts = one_hot.sum(dim=1)
        max_count, max_class = class_counts.max(dim=1)

        mask = max_count > int((k - 1) * 0.5)
        point_classes[mask] = max_class[mask]
        return point_classes
    elif method == "weighted":
        num_classes = len(point_classes.unique())
        one_hot = F.one_hot(nearest_k_neighbor_classes, num_classes=num_classes)

        weights = 1 / nearest_k_distances
        weights = weights / weights.sum(dim=1, keepdim=True)
        weighted_votes = one_hot * weights.unsqueeze(-1)
        weighted_vote_counts = weighted_votes.sum(dim=1)
        
        smoothed_classes = weighted_vote_counts.argmax(dim=1)

        return smoothed_classes
    elif method == "features":
        num_classes = len(point_classes.unique())
        one_hot = F.one_hot(nearest_k_neighbor_classes, num_classes=num_classes)

        # Compute dot product between normal vectors of points and their nearest neighbors
        point_normals = features
        nearest_k_neighbor_normals = features[nearest_k_idx[:, 1:]]
        dot_products = (point_normals.unsqueeze(1) * nearest_k_neighbor_normals).sum(dim=-1)

        # Use dot products as weights
        weights = dot_products
        weights = weights / weights.sum(dim=1, keepdim=True)
        weighted_votes = one_hot * weights.unsqueeze(-1)
        weighted_vote_counts = weighted_votes.sum(dim=1)
        
        smoothed_classes = weighted_vote_counts.argmax(dim=1)

        # prevent conversion to class 4 if point has z greater than 0.015
        # if point has z greater than 0.015 and is classified as class 4, find second most voted class that isn't class 4
        change_mask = smoothed_classes != point_classes
        mask = (smoothed_classes == 4) & (point_xyzs[:, 2] > 0) & change_mask

        # get second highest count
        smoothed_classes[mask] = torch.argsort(weighted_vote_counts[mask], descending=True)[:, 1]

        return smoothed_classes



    
    

    












