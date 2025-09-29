import torch.nn as nn
from pytorch_metric_learning.losses import SupConLoss
import torch

def unique(x, dim=None):
    unique, inverse = torch.unique(
        x, sorted=True, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype,
                        device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    return unique, inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)

class HMLC(nn.Module):
    def __init__(self, temperature=0.07,
                 base_temperature=0.07, layer_penalty=None, loss_type='hmce', pad_value=-1):
        super(HMLC, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.layer_penalty = layer_penalty if layer_penalty else self.pow_2
        self.sup_con_loss = SupConLoss(temperature)
        self.loss_type = loss_type
        self.pad_value = pad_value

    def pow_2(self, value):
        return torch.pow(2, value)

    def forward(self, features, labels):
        device = features.device
        cumulative_loss = torch.tensor(0.0, device=device)
        max_loss_lower_layer = torch.tensor(float('-inf'), device=device)

        for l in range(1, labels.shape[1]):
            # Skip level if it's all padding
            if torch.all(labels[:, l] == self.pad_value):
                continue

            # Mask valid entries (exclude padding)
            valid = labels[:, l] != self.pad_value
            if valid.sum() <= 1:
                continue  # Not enough samples for contrastive learning

            layer_labels = labels[valid]
            layer_features = features[valid]

            layer_mask = torch.ones(layer_labels.shape, device=device)
            layer_mask[:, l:] = 0
            masked_labels = layer_labels * layer_mask
            
            dic_labels = {}
            count = 0
            int_labels = []
            for i in masked_labels:
                str_label = ','.join([str(k) for k in i.long().cpu().detach().numpy()])
                if str_label not in dic_labels:
                    dic_labels[str_label] = count
                    count+=1
                int_labels.append(dic_labels[str_label])
            int_labels = torch.Tensor([int_labels]).to('cuda')

            layer_loss = 1
            #layer_loss = self.sup_con_loss(layer_features, mask = mask_labels)

            layer_loss = self.sup_con_loss(layer_features, int_labels.squeeze())
            if self.loss_type == 'hmc':
                cumulative_loss += self.layer_penalty(torch.tensor(1 / l, device=device)) * layer_loss
            elif self.loss_type == 'hce':
                layer_loss = torch.max(max_loss_lower_layer, layer_loss)
                cumulative_loss += layer_loss
            elif self.loss_type == 'hmce':
                layer_loss = torch.max(max_loss_lower_layer, layer_loss)
                cumulative_loss += self.layer_penalty(torch.tensor(1 / l, device=device)) * layer_loss
            else:
                raise NotImplementedError('Unknown loss type')

            max_loss_lower_layer = torch.max(max_loss_lower_layer, layer_loss)

            _, unique_indices = unique(layer_labels, dim=0)
            labels = layer_labels[unique_indices]
            features = layer_features[unique_indices]

        return cumulative_loss / labels.shape[1]