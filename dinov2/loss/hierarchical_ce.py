import numpy as np
import torch
from typing import List
from copy import deepcopy
from nltk.tree import Tree
import lzma
import pickle
from math import exp, fsum
import os

def get_label(node):
    if isinstance(node, Tree):
        return node.label()
    else:
        return node

def get_exponential_weighting(hierarchy: Tree, value, normalize=True):
    """
    Construct exponentially decreasing weighting, where each edge is weighted
    according to its distance from the root as exp(-value*dist).

    Args:
        hierarchy: The hierarchy to use to generate the weights.
        value: The decay value.
        normalize: If True ensures that the sum of all weights sums
            to one.

    Returns:
        Weights as a nltk.Tree whose labels are the weights associated with the
        parent edge.
    """
    weights = deepcopy(hierarchy)
    all_weights = []
    for p in weights.treepositions():
        node = weights[p]
        weight = exp(-value * len(p))
        all_weights.append(weight)
        if isinstance(node, Tree):
            node.set_label(weight)
        else:
            weights[p] = weight
    total = fsum(all_weights)  # stable sum
    if normalize:
        for p in weights.treepositions():
            node = weights[p]
            if isinstance(node, Tree):
                node.set_label(node.label() / total)
            else:
                weights[p] /= total
    return weights



def get_weighting(hierarchy: Tree, weighting="uniform", **kwargs):
    """
    Get different weightings of edges in a tree.

    Args:
        hierarchy: The tree to generate the weighting for.
        weighting: The type of weighting, one of 'uniform', 'exponential'.
        **kwards: Keyword arguments passed to the weighting function.
    """
    if weighting == "uniform":
        return get_uniform_weighting(hierarchy, **kwargs)
    elif weighting == "exponential":
        return get_exponential_weighting(hierarchy, **kwargs)
    else:
        raise NotImplementedError("Weighting {} is not implemented".format(weighting))

def load_hierarchy():
    hierarchy = Tree("root", [
        Tree("blast", [
            Tree("blast_normality", [
                Tree("blast_maturity", [
                    "lymphoblast",
                    "myeloblast",
                    "lymphocyte_immature"
                    ])
                ])
        ]),

        Tree("lymphopoiesis", [
            Tree("normal", [
                Tree("lymphocyte_mature", [
                        "lymphocyte_reactive",
                        "lymphocyte_typical",
                        "plasma_cell"
                        ])
                ]),
            Tree("lymphocyte_atypical", [
                Tree('undefined_maturity', [
                    "hairy_cell",
                    "lymphocyte_large_granular",
                    "lymphocyte_neoplastic"
                ])
            ])
        ]),
        Tree("granulopoiesis", [
            Tree("normal", [
                Tree("immature", [
                    "metamyelocyte",
                    "myelocyte",
                    "promyelocyte"
                    ]),
                Tree("mature", [
                    "basophil",
                    "eosinophil",
                    Tree("neutrophil", [
                        "neutrophil_band",
                        "neutrophil_segmented"
                        ])
                    ])
            ]),
            Tree("abnormal", [
                Tree("immature", [
                "promyelocyte_bilobed"
                ]),
                Tree("mature", [
                    "eosinophil_abnormal",
                ])
            ])
        ]),
        Tree("monocytopoiesis", [
            Tree("normal", [
                Tree("immature", [
                    "monoblast"
                ]),
                Tree("mature", [
                    "monocyte"
                ]),
            ]),
            Tree("abnormal", [
                Tree("immature", [
                    "fagott_cell"
                ])
            ])
        ]),
        Tree("thrombopoiesis", [
            "platelet"
        ]),
        Tree("erythropoiesis", [
            Tree('normal', [
                Tree("immature", [
                    "proeryhtroblast",
                    "erythroblast"
                ]),
            ]),
        ]),

        Tree("artefacts", [
            "smudge_cell"
        ])
    ])
    return hierarchy

class HierarchicalLLLoss(torch.nn.Module):
    """
    Hierachical log likelihood loss.

    The weights must be implemented as a nltk.tree object and each node must
    be a float which corresponds to the weight associated with the edge going
    from that node to its parent. The value at the origin is not used and the
    shapre of the weight tree must be the same as the associated hierarchy.

    The input is a flat probability vector in which each entry corresponds to
    a leaf node of the tree. We use alphabetical ordering on the leaf nodes
    labels, which corresponds to the 'normal' imagenet ordering.

    Args:
        hierarchy: The hierarchy used to define the loss.
        leaf_classes: A list of leaf_classes defining the order of the leaf nodes.
        intern_classes: List of intern nodes that can be classes.
        weights: The weights as a tree of similar shape as hierarchy.
    """

    def __init__(self, hierarchy: Tree, leaf_classes: List[str], intern_classes: List[str], weights: Tree):
        super(HierarchicalLLLoss, self).__init__()

        assert hierarchy.treepositions() == weights.treepositions()

        # the tree positions of all the leaves
        positions_nodes_dic = {get_label(hierarchy[p]): p for p in hierarchy.treepositions()}

        num_leaf_classes = len(leaf_classes)
        num_intern_classes = len(intern_classes)
        # we use classes in the given order
        positions_leaves = [positions_nodes_dic[c] for c in leaf_classes]
        positions_intern_nodes = [positions_nodes_dic[c] for c in intern_classes]

        # the tree positions of all the edges (we use the bottom node position)
        positions_edges = hierarchy.treepositions()[1:]  # the first one is the origin

        # map from position tuples to leaf/edge indices
        index_map_leaves = {positions_leaves[i]: i for i in range(len(positions_leaves))}
        index_map_edges = {positions_edges[i]: i for i in range(len(positions_edges))}

        # edge indices corresponding to the path from each index to the root
        edges_from_leaf = [[index_map_edges[position[:i]] for i in range(len(position), 0, -1)] for position in positions_leaves]

        # get max size for the number of edges to the root
        num_edges = max([len(p) for p in edges_from_leaf])

        # helper that returns all leaf positions from another position wrt to the original position
        def get_leaf_positions(position):
            node = hierarchy[position]
            if isinstance(node, Tree):
                return node.treepositions("leaves")
            else:
                return [()]

        # indices of all leaf nodes for each edge index
        leaf_indices = [[index_map_leaves[position + leaf] for leaf in get_leaf_positions(position)] for position in positions_edges]

        # save all relevant information as pytorch tensors for computing the loss on the gpu
        self.onehot_den = torch.nn.Parameter(torch.zeros([num_leaf_classes + num_intern_classes, num_leaf_classes, num_edges]), requires_grad=False)
        self.onehot_num = torch.nn.Parameter(torch.zeros([num_leaf_classes + num_intern_classes, num_leaf_classes, num_edges]), requires_grad=False)
        self.weights = torch.nn.Parameter(torch.zeros([num_leaf_classes + num_intern_classes, num_edges]), requires_grad=False)

        # one hot encoding of the numerators and denominators and store weights
        for i in range(num_leaf_classes):
            for j, k in enumerate(edges_from_leaf[i]):
                self.onehot_num[i, leaf_indices[k], j] = 1.0
                self.weights[i, j] = get_label(weights[positions_edges[k]])
            for j, k in enumerate(edges_from_leaf[i][1:]):
                self.onehot_den[i, leaf_indices[k], j] = 1.0
            self.onehot_den[i, :, j + 1] = 1.0  # the last denominator is the sum of all leaves
        
        for i in range(num_intern_classes): # we add columns for internal nodes
            position_node = positions_intern_nodes[i]
            leaves_from_node = [index_map_leaves[position_node + leaf] for leaf in get_leaf_positions(position_node)]

            self.onehot_den[num_leaf_classes + i,:,:] = torch.clamp(torch.sum(self.onehot_den[leaves_from_node,:,:], dim = 0), max = 1)
            self.onehot_num[num_leaf_classes + i,:,:] = torch.clamp(torch.sum(self.onehot_num[leaves_from_node,:,:], dim = 0), max = 1)
            self.weights[num_leaf_classes + i, :] = torch.mean(self.weights[leaves_from_node, :], dim = 0)


    def forward(self, inputs, target):
        """
        Foward pass, computing the loss.

        Args:
            inputs: Class _probabilities_ ordered as the input hierarchy.
            target: The index of the ground truth class.
        """
        # add a sweet dimension to inputs
        inputs = torch.unsqueeze(inputs, 1)
        # sum of probabilities for numerators
        num = torch.squeeze(torch.bmm(inputs.float(), self.onehot_num[target]))
        # sum of probabilities for denominators
        den = torch.squeeze(torch.bmm(inputs.float(), self.onehot_den[target]))
        # compute the neg logs for non zero numerators and store in there
        idx = num != 0
        num[idx] = -torch.log(num[idx] / den[idx])
        # weighted sum of all logs for each path (we flip because it is numerically more stable)
        num = torch.sum(torch.flip(self.weights[target] * num, dims=[1]), dim=1)
        # return sum of losses / batch size
        return torch.mean(num)


class HierarchicalCrossEntropyLoss(HierarchicalLLLoss):
    """
    Combines softmax with HierachicalNLLLoss. Note that the softmax is flat.
    """

    def __init__(self, hierarchy: Tree, leaf_classes: List[str], intern_classes: List[str], weights: Tree):
        super(HierarchicalCrossEntropyLoss, self).__init__(hierarchy, leaf_classes, intern_classes, weights)

    def forward(self, inputs, index):
        return super(HierarchicalCrossEntropyLoss, self).forward(torch.nn.functional.softmax(inputs, 1), index)
