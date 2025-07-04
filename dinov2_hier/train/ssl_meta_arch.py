# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from functools import partial
import logging

import torch
from torch import nn
import os

from dinov2_hier.loss import DINOLoss, iBOTPatchLoss, KoLeoLoss, HierarchicalCrossEntropyLoss, get_weighting, load_hierarchy
from pytorch_metric_learning import losses
from dinov2_hier.models import build_model_from_cfg
from dinov2_hier.layers import DINOHead
from dinov2_hier.utils.utils import has_batchnorms
from dinov2_hier.utils.param_groups import get_params_groups_with_decay, fuse_params_groups
from dinov2_hier.fsdp import get_fsdp_wrapper, ShardedGradScaler, get_fsdp_modules, reshard_fsdp_model
import math
from dinov2_hier.models.vision_transformer import BlockChunk
import requests
try:
    from xformers.ops import fmha
except ImportError:
    raise AssertionError("xFormers is required for training")


logger = logging.getLogger("dinov2")
print('ssl meta arch orig')
class MLP(nn.Module):
    """
    (from pytorch metric learning package)
    layer_sizes[0] is the dimension of the input
    layer_sizes[-1] is the dimension of the output
    """
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()

        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=False))
            layer_list.append(nn.Linear(input_size, curr_size))

        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        return self.net(x)



def interpolate_pos_encoding(x, w, h):

    offset=1
    N = x.shape[1] - offset
    dim = x.shape[-1]
    w0 = w / int(math.sqrt(N))
    h0 = h / int(math.sqrt(N))

    # Interpolate the position embeddings without changing the first row (class token)
    patch_pos_embed = nn.functional.interpolate(
        x[:, offset:].reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
        scale_factor=(w0, h0),
        mode="bicubic",
    )

    # assert int(w0) == patch_pos_embed.shape[-2]
    # assert int(h0) == patch_pos_embed.shape[-1]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

    # Concatenate the class token with the interpolated position embeddings
    return torch.cat((x[:, :1], patch_pos_embed), dim=1)


def get_downloaded_dino_vit_interpolated(modelname="dinov2_vits14", image_size=224):
    n_patches = int(image_size / 14)
    model = torch.hub.load("facebookresearch/dinov2", modelname).cuda()  #
    input_tensor = model.pos_embed
    tensor_corr_shape = interpolate_pos_encoding(input_tensor, n_patches, n_patches)
    pos_embed = nn.Parameter(torch.zeros(1, n_patches))
    pos_embed.data = tensor_corr_shape
    model.pos_embed = pos_embed

    return model

def download_file(url, dest):
    if not os.path.exists(dest):
        response = requests.get(url)
        with open(dest, 'wb') as f:
            f.write(response.content)
    return dest
    
def get_dinobloom(modelname):

        model_urls = {
            "dinov2_vits14": "https://zenodo.org/records/10908163/files/DinoBloom-S.pth?download=1",
            "dinov2_vitb14": "https://zenodo.org/records/10908163/files/DinoBloom-B.pth?download=1",
            "dinov2_vitl14": "https://zenodo.org/records/10908163/files/DinoBloom-L.pth?download=1",
            "dinov2_vitg14": "https://zenodo.org/records/10908163/files/DinoBloom-G.pth?download=1",
        }
        
        if modelname not in model_urls:
            raise ValueError(f"Unsupported modelname: {modelname}")

        model_url = model_urls[modelname]
        cache_dir = os.path.expanduser("~/.cache/dinov2_models")
        os.makedirs(cache_dir, exist_ok=True)
        model_path = os.path.join(cache_dir, f"{modelname.split('_')[1]}.pth")
        
        download_file(model_url, model_path)
        state_dict = torch.load(model_path)
        return state_dict

class SSLMetaArch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fp16_scaler = ShardedGradScaler() if cfg.compute_precision.grad_scaler else None

        student_model_dict = dict()
        teacher_model_dict = dict()
        if cfg.student.arch in ["dinov2_vits14","dinov2_vitb14","dinov2_vitl14","dinov2_vitg14","dinov2_vits14_reg","dinov2_vitb14_reg","dinov2_vitl14_reg","dinov2_vitg14_reg"]:
            student_backbone = get_downloaded_dino_vit_interpolated(cfg.student.arch,image_size=cfg.crops.global_crops_size)
            teacher_backbone = get_downloaded_dino_vit_interpolated(cfg.student.arch,image_size=cfg.crops.global_crops_size)
            embed_dict={"dinov2_vits14":384,"dinov2_vitb14":768,"dinov2_vitl14":1024,"dinov2_vitg14":1536}
            embed_dim = embed_dict[cfg.student.arch.replace("_reg","")]
        else:
            student_backbone, teacher_backbone, embed_dim = build_model_from_cfg(cfg)
        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone
        logger.info(f"OPTIONS -- architecture : embed_dim: {embed_dim}")

        if cfg.student.pretrained_weights:
            chkpt = torch.load(cfg.student.pretrained_weights)
            logger.info(f"OPTIONS -- pretrained weights: loading from {cfg.student.pretrained_weights}")
            student_backbone.load_state_dict(chkpt["model"], strict=False)

        self.embed_dim = embed_dim
        self.dino_out_dim = cfg.dino.head_n_prototypes

        self.do_dino = cfg.dino.loss_weight > 0
        self.do_koleo = cfg.dino.koleo_loss_weight > 0
        self.do_ibot = cfg.ibot.loss_weight > 0
        self.do_supervised_loss = cfg.supervised.loss_weight > 0
        self.do_smooth_rank_loss=cfg.dino.smooth_rank_loss_weight>0

        self.ibot_separate_head = cfg.ibot.separate_head

        logger.info("OPTIONS -- DINO")
        if self.do_dino:
            logger.info(f"OPTIONS -- DINO -- loss_weight: {cfg.dino.loss_weight}")
            logger.info(f"OPTIONS -- DINO -- head_n_prototypes: {cfg.dino.head_n_prototypes}")
            logger.info(f"OPTIONS -- DINO -- head_bottleneck_dim: {cfg.dino.head_bottleneck_dim}")
            logger.info(f"OPTIONS -- DINO -- head_hidden_dim: {cfg.dino.head_hidden_dim}")
            self.dino_loss_weight = cfg.dino.loss_weight
            dino_head = partial(
                DINOHead,
                in_dim=embed_dim,
                out_dim=cfg.dino.head_n_prototypes,
                hidden_dim=cfg.dino.head_hidden_dim,
                bottleneck_dim=cfg.dino.head_bottleneck_dim,
                nlayers=cfg.dino.head_nlayers,
            )
            self.dino_loss = DINOLoss(self.dino_out_dim)
            if self.do_koleo:
                logger.info("OPTIONS -- DINO -- applying KOLEO regularization")
                self.koleo_loss = KoLeoLoss()

        else:
            logger.info("OPTIONS -- DINO -- not using DINO")

        if self.do_dino or self.do_ibot:
            student_model_dict["dino_head"] = dino_head()
            teacher_model_dict["dino_head"] = dino_head()

        logger.info("OPTIONS -- IBOT")
        logger.info(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
        logger.info(f"OPTIONS -- IBOT masking -- ibot_mask_ratio_tuple: {cfg.ibot.mask_ratio_min_max}")
        logger.info(f"OPTIONS -- IBOT masking -- ibot_mask_sample_probability: {cfg.ibot.mask_sample_probability}")
        if self.do_ibot:
            self.ibot_loss_weight = cfg.ibot.loss_weight
            assert max(cfg.ibot.mask_ratio_min_max) > 0, "please provide a positive mask ratio tuple for ibot"
            assert cfg.ibot.mask_sample_probability > 0, "please provide a positive mask probability for ibot"
            self.ibot_out_dim = cfg.ibot.head_n_prototypes if self.ibot_separate_head else cfg.dino.head_n_prototypes
            self.ibot_patch_loss = iBOTPatchLoss(self.ibot_out_dim)
            if self.ibot_separate_head:
                logger.info(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
                logger.info(f"OPTIONS -- IBOT -- head_n_prototypes: {cfg.ibot.head_n_prototypes}")
                logger.info(f"OPTIONS -- IBOT -- head_bottleneck_dim: {cfg.ibot.head_bottleneck_dim}")
                logger.info(f"OPTIONS -- IBOT -- head_hidden_dim: {cfg.ibot.head_hidden_dim}")
                ibot_head = partial(
                    DINOHead,
                    in_dim=embed_dim,
                    out_dim=cfg.ibot.head_n_prototypes,
                    hidden_dim=cfg.ibot.head_hidden_dim,
                    bottleneck_dim=cfg.ibot.head_bottleneck_dim,
                    nlayers=cfg.ibot.head_nlayers,
                )
                student_model_dict["ibot_head"] = ibot_head()
                teacher_model_dict["ibot_head"] = ibot_head()
            else:
                logger.info("OPTIONS -- IBOT -- head shared with DINO")
        if self.do_supervised_loss: 
            supervised_head = partial(MLP, layer_sizes=[embed_dim, cfg.supervised.n_classes])
            student_model_dict["supervised_head"] =supervised_head()
            if self.cfg.hier:
                hierarchy = load_hierarchy(version= self.cfg.version) #load the tree
                classes = sorted(self.cfg.classes_to_int.keys()) #load all possible classes for cells
                leaves_nodes = [i for i in classes if i in hierarchy.leaves()]
                intern_nodes = [i for i in classes if i not in hierarchy.leaves()]
                alpha = self.cfg.alpha # it controls the difference between low level and high levels in the hierarchy
                weights = get_weighting(hierarchy, "exponential", value= alpha) #weight each edge of the hierarchy
                self.supervised_losses = HierarchicalCrossEntropyLoss(hierarchy, leaves_nodes, intern_nodes, weights).cuda()
            else:
                self.supervised_losses = nn.CrossEntropyLoss(ignore_index=-1)
            self.supervised_loss_weight = cfg.supervised.loss_weight
   
        self.need_to_synchronize_fsdp_streams = True

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)
        state_dict=get_dinobloom(cfg.student.arch)
        if state_dict["teacher"]["backbone.pos_embed"].shape != self.student.backbone.pos_embed.shape:
            n_patches = int((self.student.backbone.pos_embed.shape[1]-1)**0.5)
            input_tensor = state_dict["teacher"]["backbone.pos_embed"]
            tensor_corr_shape = interpolate_pos_encoding(input_tensor, n_patches, n_patches)
            pos_embed = nn.Parameter(torch.zeros(1, n_patches))
            pos_embed.data = tensor_corr_shape
            state_dict["teacher"]["backbone.pos_embed"] = pos_embed

        self.student.load_state_dict(state_dict["teacher"],strict=False)
        # just for logging purposes we see which keys could get matched
        original_teacher_keys = set(self.teacher.state_dict().keys())
        # Load the state dictionary with strict=False
        missing_keys, unexpected_keys = self.teacher.load_state_dict(state_dict["teacher"], strict=False)
        
        # Determine the loaded keys
        loaded_keys = set(state_dict["teacher"].keys()) - set(missing_keys)
        successfully_loaded_keys = loaded_keys & original_teacher_keys
        
        # Print the results
        print("Successfully loaded keys:", successfully_loaded_keys)
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)

        # there is no backpropagation through the teacher, so no need for gradients
        for p in self.teacher.parameters():
            p.requires_grad = False
        logger.info(f"Student and Teacher are built: they are both {cfg.student.arch} network.")

    def forward(self, inputs):
        raise NotImplementedError

    def backprop_loss(self, loss):
        if self.fp16_scaler is not None:
            self.fp16_scaler.scale(loss).backward()
        else:
            loss.backward()

    def get_supervised_loss(self, images_sup):
        
        image_sup, labels = images_sup["collated_global_crops"], images_sup["labels"]
        image_sup, labels = image_sup.to('cuda'), labels.to('cuda')
        student_global_backbone_output_dict_sup = self.student.backbone(
            image_sup, masks=None, is_training=True
        )
        student_global_cls_tokens_sup = student_global_backbone_output_dict_sup["x_norm_clstoken"]
        #student_global_cls_tokens_sup = student_global_backbone_output_dict_sup
        cls_output = self.student.module.supervised_head(student_global_cls_tokens_sup) if hasattr(self.student, 'module') else self.student.supervised_head(student_global_cls_tokens_sup)
        mask = labels != -1
        masked_labels = labels[mask]
        cls_output = cls_output[mask]       

        supervised_loss = self.supervised_loss_weight * self.supervised_losses(cls_output, masked_labels)
        loss_value = supervised_loss/2
        loss_name = 'HierCrossEntropy' if self.cfg.hier else 'CrossEntropy' 

        del cls_output
        del student_global_cls_tokens_sup
        torch.cuda.empty_cache()
        
        return loss_name, loss_value       

    def forward_backward(self, images, images_sup, teacher_temp):
        n_global_crops = 2
        assert n_global_crops == 2
        n_local_crops = self.cfg.crops.local_crops_number

        global_crops = images["collated_global_crops"].cuda(non_blocking=True)
        local_crops = images["collated_local_crops"].cuda(non_blocking=True)

        masks = images["collated_masks"].cuda(non_blocking=True)
        mask_indices_list = images["mask_indices_list"].cuda(non_blocking=True)
        n_masked_patches_tensor = images["n_masked_patches"].cuda(non_blocking=True)
        n_masked_patches = mask_indices_list.shape[0]
        upperbound = images["upperbound"]
        masks_weight = images["masks_weight"].cuda(non_blocking=True)

        n_local_crops_loss_terms = max(n_local_crops * n_global_crops, 1)
        n_global_crops_loss_terms = (n_global_crops - 1) * n_global_crops

        do_dino = self.do_dino
        do_ibot = self.do_ibot

        # loss scales
        ibot_loss_scale = 1.0 / n_global_crops

        # teacher output
        @torch.no_grad()
        def get_teacher_output():
            x, n_global_crops_teacher = global_crops, n_global_crops
            teacher_backbone_output_dict = self.teacher.backbone(x, is_training=True)

            teacher_cls_tokens = teacher_backbone_output_dict["x_norm_clstoken"]
            teacher_cls_tokens = teacher_cls_tokens.chunk(n_global_crops_teacher)
            # watch out: these are chunked and cat'd in reverse so A is matched to B in the global crops dino loss
            teacher_cls_tokens = torch.cat((teacher_cls_tokens[1], teacher_cls_tokens[0]))
            ibot_teacher_patch_tokens = teacher_backbone_output_dict["x_norm_patchtokens"]
            _dim = ibot_teacher_patch_tokens.shape[-1]
            n_cls_tokens = teacher_cls_tokens.shape[0]

            if do_ibot and not self.ibot_separate_head:
                buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound + n_cls_tokens, _dim)
                buffer_tensor_teacher[:n_cls_tokens].copy_(teacher_cls_tokens)
                torch.index_select(
                    ibot_teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor_teacher[n_cls_tokens : n_cls_tokens + n_masked_patches],
                )
                tokens_after_head = self.teacher.dino_head(buffer_tensor_teacher)
                teacher_cls_tokens_after_head = tokens_after_head[:n_cls_tokens]
                masked_teacher_patch_tokens_after_head = tokens_after_head[
                    n_cls_tokens : n_cls_tokens + n_masked_patches
                ]
            elif do_ibot and self.ibot_separate_head:
                buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound, _dim)
                torch.index_select(
                    ibot_teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor_teacher[:n_masked_patches],
                )
                teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)
                masked_teacher_patch_tokens_after_head = self.teacher.ibot_head(buffer_tensor_teacher)[
                    :n_masked_patches
                ]
            else:
                teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)
                masked_teacher_ibot_softmaxed_centered = None

            if self.cfg.train.centering == "centering":
                teacher_dino_softmaxed_centered_list = self.dino_loss.softmax_center_teacher(
                    teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                ).view(n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:])
                self.dino_loss.update_center(teacher_cls_tokens_after_head)
                if do_ibot:
                    masked_teacher_patch_tokens_after_head = masked_teacher_patch_tokens_after_head.unsqueeze(0)
                    masked_teacher_ibot_softmaxed_centered = self.ibot_patch_loss.softmax_center_teacher(
                        masked_teacher_patch_tokens_after_head[:, :n_masked_patches], teacher_temp=teacher_temp
                    )
                    masked_teacher_ibot_softmaxed_centered = masked_teacher_ibot_softmaxed_centered.squeeze(0)
                    self.ibot_patch_loss.update_center(masked_teacher_patch_tokens_after_head[:n_masked_patches])

            elif self.cfg.train.centering == "sinkhorn_knopp":
                teacher_dino_softmaxed_centered_list = self.dino_loss.sinkhorn_knopp_teacher(
                    teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                ).view(n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:])

                if do_ibot:
                    masked_teacher_ibot_softmaxed_centered = self.ibot_patch_loss.sinkhorn_knopp_teacher(
                        masked_teacher_patch_tokens_after_head,
                        teacher_temp=teacher_temp,
                        n_masked_patches_tensor=n_masked_patches_tensor,
                    )

            else:
                raise NotImplementedError

            return teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered

        teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered = get_teacher_output()
        reshard_fsdp_model(self.teacher)

        loss_dict = {}

        loss_accumulator = 0  # for backprop
        student_global_backbone_output_dict, student_local_backbone_output_dict = self.student.backbone(
            [global_crops, local_crops], masks=[masks, None], is_training=True
        )

        inputs_for_student_head_list = []

        # 1a: local crops cls tokens
        student_local_cls_tokens = student_local_backbone_output_dict["x_norm_clstoken"]
        inputs_for_student_head_list.append(student_local_cls_tokens.unsqueeze(0))

        # 1b: global crops cls tokens
        student_global_cls_tokens = student_global_backbone_output_dict["x_norm_clstoken"]
        inputs_for_student_head_list.append(student_global_cls_tokens.unsqueeze(0))

        # 1c: global crops patch tokens
        if do_ibot:
            _dim = student_global_backbone_output_dict["x_norm_clstoken"].shape[-1]
            ibot_student_patch_tokens = student_global_backbone_output_dict["x_norm_patchtokens"]
            buffer_tensor_patch_tokens = ibot_student_patch_tokens.new_zeros(upperbound, _dim)
            buffer_tensor_patch_tokens[:n_masked_patches].copy_(
                torch.index_select(ibot_student_patch_tokens.flatten(0, 1), dim=0, index=mask_indices_list)
            )
            if not self.ibot_separate_head:
                inputs_for_student_head_list.append(buffer_tensor_patch_tokens.unsqueeze(0))
            else:
                student_global_masked_patch_tokens_after_head = self.student.ibot_head(buffer_tensor_patch_tokens)[
                    :n_masked_patches
                ]

        # 2: run
        _attn_bias, cat_inputs = fmha.BlockDiagonalMask.from_tensor_list(inputs_for_student_head_list)
        outputs_list = _attn_bias.split(self.student.dino_head(cat_inputs))

        # 3a: local crops cls tokens
        student_local_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)

        # 3b: global crops cls tokens
        student_global_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)

        # 3c: global crops patch tokens
        if do_ibot and not self.ibot_separate_head:
            student_global_masked_patch_tokens_after_head = outputs_list.pop(0).squeeze(0)[:n_masked_patches]

        if n_local_crops > 1:
            dino_local_crops_loss = self.dino_loss(
                student_output_list=student_local_cls_tokens_after_head.chunk(n_local_crops),
                teacher_out_softmaxed_centered_list=teacher_dino_softmaxed_centered_list,
            ) / (n_global_crops_loss_terms + n_local_crops_loss_terms)

            # store for display
            loss_dict["dino_local_crops_loss"] = dino_local_crops_loss

            # accumulate loss
            loss_accumulator += self.dino_loss_weight * dino_local_crops_loss

        # process global crops
        loss_scales = 2  # this is here since we process global crops together

        if do_dino:
            # compute loss
            dino_global_crops_loss = (
                self.dino_loss(
                    student_output_list=[student_global_cls_tokens_after_head],
                    teacher_out_softmaxed_centered_list=[
                        teacher_dino_softmaxed_centered_list.flatten(0, 1)
                    ],  # these were chunked and stacked in reverse so A is matched to B
                )
                * loss_scales
                / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            )

            loss_dict["dino_global_crops_loss"] = dino_global_crops_loss

            # accumulate loss
            loss_accumulator += self.dino_loss_weight * dino_global_crops_loss

            student_cls_tokens = student_global_cls_tokens

            if self.do_koleo:
                koleo_loss = self.cfg.dino.koleo_loss_weight * sum(
                    self.koleo_loss(p) for p in student_cls_tokens.chunk(2)
                )  # we don't apply koleo loss between cls tokens of a same image
                loss_accumulator += koleo_loss
                loss_dict["koleo_loss"] = (
                    koleo_loss / loss_scales
                )  # this is to display the same losses as before but we can remove eventually
        
        
        if self.do_supervised_loss:
            k, v = self.get_supervised_loss(images_sup)
            loss_dict[k] = v
            loss_accumulator += v

        if do_ibot:
            # compute loss
            ibot_patch_loss = (
                self.ibot_patch_loss.forward_masked(
                    student_global_masked_patch_tokens_after_head,
                    masked_teacher_ibot_softmaxed_centered,
                    student_masks_flat=masks,
                    n_masked_patches=n_masked_patches,
                    masks_weight=masks_weight,
                )
                * loss_scales
                * ibot_loss_scale
            )

            # store for display
            loss_dict["ibot_loss"] = ibot_patch_loss / 2

            # accumulate loss
            loss_accumulator += self.ibot_loss_weight * ibot_patch_loss

        self.backprop_loss(loss_accumulator)

        self.fsdp_synchronize_streams()

        return loss_dict

    def fsdp_synchronize_streams(self):
        if self.need_to_synchronize_fsdp_streams:
            torch.cuda.synchronize()
            for attr in {"_unshard_stream", "_post_backward_stream", "_pre_unshard_stream", "_all_reduce_stream", "_default_stream"}:
                stream = getattr(self.teacher.backbone, attr)
                setattr(self.student.dino_head, attr, stream)
                setattr(self.teacher.dino_head, attr, stream)
                setattr(self.student.backbone, attr, stream)
            self.need_to_synchronize_fsdp_streams = False

    def update_teacher(self, m):
        student_param_list = []
        teacher_param_list = []
        with torch.no_grad():
            for k in self.student.keys():
                if k != 'supervised_head':
                    for ms, mt in zip(get_fsdp_modules(self.student[k]), get_fsdp_modules(self.teacher[k])):
                        student_param_list += ms.params
                        teacher_param_list += mt.params
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

    def train(self):
        super().train()
        self.teacher.eval()

    def get_maybe_fused_params_for_submodel(self, m):
        params_groups = get_params_groups_with_decay(
            model=m,
            lr_decay_rate=self.cfg.optim.layerwise_decay,
            patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
        )
        fused_params_groups = fuse_params_groups(params_groups)
        logger.info("fusing param groups")

        for g in fused_params_groups:
            g["foreach"] = True
        return fused_params_groups

    def get_params_groups(self):
        all_params_groups = []
        for m in self.student.values():
            all_params_groups += self.get_maybe_fused_params_for_submodel(m)
        return all_params_groups

    def prepare_for_distributed_training(self):
        logger.info("DISTRIBUTED FSDP -- preparing model for distributed training")
        if has_batchnorms(self.student):
            raise NotImplementedError
        # below will synchronize all student subnetworks across gpus:
        for k, v in self.student.items():
            if "supervised_head" in k:  # removed supervised head from teacher bc of fsdp error
                student_model_cfg = self.cfg.compute_precision.student["supervised_head" ]
                self.student[k] = get_fsdp_wrapper(student_model_cfg, modules_to_wrap={BlockChunk})(self.student[k])
            elif "domain_head" in k:  # removed supervised head from teacher bc of fsdp error
                student_model_cfg = self.cfg.compute_precision.student["domain_head" ]
                self.student[k] = get_fsdp_wrapper(student_model_cfg, modules_to_wrap={BlockChunk})(self.student[k])
            else:
                self.teacher[k].load_state_dict(self.student[k].state_dict())
                student_model_cfg = self.cfg.compute_precision.student[k]
                self.student[k] = get_fsdp_wrapper(student_model_cfg, modules_to_wrap={BlockChunk})(self.student[k])
                teacher_model_cfg = self.cfg.compute_precision.teacher[k]
                self.teacher[k] = get_fsdp_wrapper(teacher_model_cfg, modules_to_wrap={BlockChunk})(self.teacher[k])
