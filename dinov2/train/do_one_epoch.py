import os
import torch
from pytorch_metric_learning import losses
import logging
from torch import nn
from dinov2.loss import HierarchicalCrossEntropyLoss, get_weighting, load_hierarchy
logger = logging.getLogger("cls")

def do_one_epoch(cfg, batch, model):

    images, labels, filepath, domain_label = batch
    logger.info("OPTIONS -- DINO")
    
    if cfg.n_levels == 1:
        loss_function = nn.CrossEntropyLoss(ignore_index=-1)
    elif cfg.n_levels > 1:
        hier_ce = True
        hierarchy = load_hierarchy(cfg.n_levels)
        classes = sorted([l for l in {','.join([str(k) for k in j  if k!= -1]) for (i,j) in cfg.label_dict.items()}])
        leaves_nodes = [i for i in classes if i in hierarchy.leaves()]
        intern_nodes = [i for i in classes if i not in hierarchy.leaves()]
        classes_to_int = {class_label:i for (i, class_label) in enumerate(leaves_nodes + intern_nodes)}
        alpha = 0.1
        weights = get_weighting(hierarchy, "exponential", value= alpha)
        loss_function = HierarchicalCrossEntropyLoss(hierarchy, leaves_nodes, intern_nodes, weights).cuda()

    loss_dict = {}
    labels = labels.to('cuda')
    images = images.to('cuda')

    if cfg.n_levels == 1:
        mask = labels != -1
    elif cfg.n_levels > 1:
        #norm_labels = torch.Tensor([cfg.revert_label[','.join([str(k.int().item()) for k in j])] for j in labels.squeeze(0)]).type(torch.LongTensor)
        norm_labels = [','.join([str(k.int().item()) for k in j]) for j in labels.squeeze(0)]
        mask = [i != '-1,-1,-1,-1' for i in norm_labels]

    masked_labels = labels[mask]
    masked_images = images[mask]
    out = model(masked_images)
    if cfg.n_levels > 1: 
        str_labels = [','.join([str(k.int().item()) for k in j if k!= -1]) for j in masked_labels]
        masked_labels = [classes_to_int[k] for k in str_labels]


    current_loss = loss_function(out, masked_labels)
    current_loss.backward()
    loss_dict['{}CE'.format('Hier' if cfg.n_levels > 1 else '')] = current_loss
    return loss_dict

    