# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch, warnings
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Union, Literal

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone, Joiner
from .matcher import build_clip_matcher, build_matcher, ClipHungarianMatcher
from .segmentation import dice_loss, sigmoid_focal_loss
from .transformer import build_st_transformer, SpatialTemporalTransformer


class AssociaTR(nn.Module):
    """ This is the AssociaTR module that performs stqtic sequential object detections """
    def __init__(self, num_frames: int, backbone: Joiner, transformer: SpatialTemporalTransformer, num_classes: int, num_queries: int, aux_loss: bool = False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_frames = num_frames
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.objness_embed = nn.Linear(hidden_dim, num_frames)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4 * num_frames, 3)

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_objness": objectness logits
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose() # [BxT, C, H, W], [BxT, H, W]
        src = self.input_proj(src) # [BxT, D, H, W]
        assert mask is not None
        hs = self.transformer(src, mask, self.query_embed.weight, pos[-1])[0] # [B, N, D]

        outputs_class = self.class_embed(hs) # [B, N, N_cls + 1]
        outputs_objness = self.objness_embed(hs) # [B, N, T]
        outputs_coord = self.bbox_embed(hs).sigmoid() # [B, N, 4T]
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'pred_objness': outputs_objness[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_objness)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class: Tensor, outputs_coord: Tensor, outputs_objness: Tensor):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_objness': c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_objness[:-1])]
    

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SetCriterion(nn.Module):
    """ This class computes the loss for AssociaTR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes: int, matcher: ClipHungarianMatcher, weight_dict: dict[str, float], eos_coef: float, losses: list[str]):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs: dict[str, Tensor], targets: list[dict[str, Tensor]], indices: list[tuple[Tensor, Tensor]], norm_term: Tensor, log: bool = True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'] # [B, N, N_cls + 1]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([mat[J, 1] for mat, (_, J) in zip(self.matcher.target_mats, indices)]).long()
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device) # [B, N]
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs: dict[str, Tensor], targets: list[dict[str, Tensor]], indices: list[tuple[Tensor, Tensor]], norm_term: Tensor):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(mat) for mat in self.matcher.target_mats], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs: dict[str, Tensor], targets: list[dict[str, Tensor]], indices: list[tuple[Tensor, Tensor]], norm_term: Tensor):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs

        num_frames = self.matcher.num_frames
        idx = self._get_src_permutation_idx(indices)
        box_keep = torch.cat(
            [mat[i, -num_frames:] for mat, (_, i) in zip(self.matcher.target_mats, indices)],
            dim=0
        ).bool() # [N_obj, N_frames]
        src_boxes = outputs['pred_boxes'][idx].view(-1, num_frames, 4)[box_keep] # [N_obj_tp, 4]
        target_boxes = torch.cat([mat[i, 2: 2 + 4 * num_frames] for mat, (_, i) in zip(self.matcher.target_mats, indices)], dim=0) # [N_obj, 4 * N_frames]
        target_boxes = target_boxes.view(-1, num_frames, 4)[box_keep] # [N_obj_tp, 4]

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum(-1).mean()
        losses['loss_giou'] = loss_giou.mean()

        return losses
    
    def loss_objness(self, outputs: dict[str, Tensor], targets: list[dict[str, Tensor]], indices: list[tuple[Tensor, Tensor]], norm_term: Tensor):
        """
            Compute objectness of each instance for every frame in a video clip. The loss function is Binary Cross Entropy.
            Note that unmatched queries (treated as background) are not supervised.
        """
        num_frames = self.matcher.num_frames
        idx = self._get_src_permutation_idx(indices)
        src_objness = outputs["pred_objness"][idx] # [N_obj, N_frames]
        target_objness = torch.cat([mat[i, -num_frames:] for mat, (_, i) in zip(self.matcher.target_mats, indices)], dim=0) # [N_obj, N_frames]

        loss_objness = F.binary_cross_entropy_with_logits(src_objness, target_objness, reduction="sum") # [N_obj, N_frames]

        return {"loss_objness": loss_objness / norm_term} # sum over frames, then normalized by number of instances

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices: list[tuple[Tensor, Tensor]]):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices: list[tuple[Tensor, Tensor]]):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'objness': self.loss_objness,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs: dict[str, Union[Tensor, list]], targets: dict[str, Tensor]):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_instances = sum(mat.size(0) for mat in self.matcher.target_mats)
        device = next(iter(outputs.values())).device
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        num_instances = torch.as_tensor([num_instances], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
            torch.distributed.all_reduce(num_instances)

        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        num_instances = torch.clamp(num_instances / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            norm_term = num_instances if loss == "objness" else num_boxes
            losses.update(self.get_loss(loss, outputs, targets, indices, norm_term))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses
    

class _postprocess(nn.Module):
    def __init__(self, instance_conf: float = 0.8, objness_conf: float = 0.6) -> None:
        super().__init__()
        self.instance_conf = instance_conf
        self.objness_conf = objness_conf

    @torch.no_grad()
    def forward(self):...
    @torch.no_grad()
    def _decode_output_boxes(self, boxes: Tensor):
        """
        decode / de-normalize cross-frame boxes of instances that are normalized by frame number.

        Args:
            -- boxes: shape: [B, N, Nf x 4], output by model, format must be cxcywh
        """
        assert boxes.ndim == 3, boxes.size()
        assert boxes.size(2) % 4 == 0, boxes.size(2)

        B, N, _ = boxes.size()
        Nf = boxes.size(2) // 4
        adder = torch.arange(Nf, dtype=boxes.dtype, device=boxes.device)
        adder = adder.view(1, 1, Nf, 1).repeat(B, N, 1, 2).flatten(2) # [B, N, Nf * 2]
        boxes[..., 0::2] *= Nf
        boxes[..., 0::2] -= adder

        return boxes


class PostProcessForCOCODet(_postprocess):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs: dict[str, Union[Tensor, list]], target_sizes: Tensor, exclude_negatives: bool = False):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x num_frames x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        # [B, N, N_cls + 1], [B, N, Nf * 4], [B, N, Nf]
        out_logits, out_bbox, out_objness = [outputs[key] for key in ["pred_logits", "pred_boxes", "pred_objness"]]

        num_frames = out_objness.size(-1)
        exclude_negatives = exclude_negatives and num_frames > 1
        B, N, Cp1 = out_logits.shape
        out_bbox = self._decode_output_boxes(out_bbox).view(B, N, num_frames, 4).permute(0, 2, 1, 3) # [B, Nf, N, 4]

        assert len(out_logits) * num_frames == len(target_sizes), "{} vs. {}".format(out_logits.size(), target_sizes)
        assert target_sizes.shape[1] == 2

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.view(B, num_frames, 2).unbind(2) # [B, Nf]
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=2) # [B, Nf, 4]
        boxes = boxes * scale_fct.unsqueeze(2) # [B, Nf, N, 4]

        prob = F.softmax(out_logits, -1) # [B, N, N_cls + 1]
        if exclude_negatives:
            out_objness = out_objness.permute(0, 2, 1) > self.objness_conf # [B, Nf, N]
            scores, labels = prob.unsqueeze(1).repeat(1, num_frames, 1, 1).max(-1) # [B, Nf, N]
            tp_keep = torch.ne(labels, Cp1 - 1) # [B, Nf, N]
            total_keep = out_objness & tp_keep # [B, Nf, N]
            results = [
                {'scores': s[keep], 'labels': l[keep], 'boxes': b[keep]}
                for s, l, b, keep in zip(scores.flatten(0, 1), labels.flatten(0, 1), boxes.flatten(0, 1), total_keep.flatten(0, 1))
            ]
        else:
            scores, labels = prob.unsqueeze(1).repeat(1, num_frames, 1, 1)[..., :-1].max(-1) # [B, Nf, N]
            results = [
                {'scores': s, 'labels': l, 'boxes': b}
                for s, l, b in zip(scores.flatten(0, 1), labels.flatten(0, 1), boxes.flatten(0, 1))
            ]

        return results
    

class PostProcessForMOT(_postprocess):
    """ This module converts the model's output into the format expected by pymotmetrics API (CLEAR MOT)"""
    @torch.no_grad()
    def forward(self, outputs: dict[str, Union[Tensor, list]], targets: list[dict[Literal["labels", "boxes", "track_ids", "orig_size"], Tensor]]):
        # [B, N, N_cls + 1], [B, N, Nf * 4], [B, N, Nf]
        out_logits, out_bbox, out_objness = [outputs[key] for key in ["pred_logits", "pred_boxes", "pred_objness"]]

        num_frames = out_objness.size(-1)
        B, N = out_logits.shape[:2]
        assert B * num_frames == len(targets), "BxNf: {} vs. num_targets: {}".format(B * num_frames, len(targets))

        out_bbox = self._decode_output_boxes(out_bbox).view(B, N, num_frames, 4)
        prob = F.softmax(out_logits, -1)
        scores, _ = prob[..., :-1].max(-1) # [B, N]
        
        videos: list[list[dict[str, Union[Tensor, list]]]] = [] # batch_list -> frame_list -> info_dict
        for batch_index in range(B): # traverse number of videos
            video_targets = targets[batch_index * num_frames: (batch_index + 1) * num_frames] # len == Nf
            indices = torch.arange(N, dtype=torch.long, device=out_logits.device) # this is also used as track id
            _cond1 = scores[batch_index] > self.instance_conf
            per_frame_objness_keep = out_objness[batch_index] > self.objness_conf # [N, Nf]
            _cond2 = per_frame_objness_keep.float().sum(-1) > 0 # [N]
            per_video_keep = _cond1 & _cond2 # [N]

            instance_indices = indices[per_video_keep] # [N_ins]
            instance_boxes = out_bbox[batch_index][per_video_keep] # [N_ins, Nf, 4]
            per_frame_objness_keep = per_frame_objness_keep[per_video_keep] # [N_ins, Nf]

            frame_info: list[dict[str, Union[Tensor, list]]] = []
            for frame_index in range(num_frames):
                target = video_targets[frame_index]
                oids = target["track_ids"].tolist()
                oboxes = target["boxes"] # xyxy format

                frame_keep = per_frame_objness_keep[:, frame_index] # [N_ins]
                hids = instance_indices[frame_keep].tolist() # [N_ins_obj]
                hboxes = instance_boxes[:, frame_index][frame_keep] # [N_ins_obj, 4] cxcywh format
                hboxes = box_ops.box_cxcywh_to_xyxy(hboxes)
                img_h, img_w = target["orig_size"] # [2]
                hboxes = hboxes * torch.stack([img_w, img_h, img_w, img_h]).view(1, 4) # scale boxes to original sizes
                frame_info.append(
                    dict(hids=hids, oids=oids, dists=box_ops.box_iou(oboxes, hboxes)[0].cpu()) # note that cost mat are transposed as [N_oboxes, N_hboxes] for motmetrics api
                )

            videos.append(frame_info)

        return videos


def init_from_pretrained_detr(model: AssociaTR, detr_state_dict: Union[str, dict[str, Tensor]] = None, 
                                load_backbone_state: bool = False, skip_mismatch: bool = False, verbose: bool = False):
    
    if detr_state_dict is None:
        detr_state_dict: str = "https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth"

    if isinstance(detr_state_dict, str):
        if detr_state_dict.startswith("https:"):
            detr_state_dict = torch.hub.load_state_dict_from_url(
                url=detr_state_dict, map_location="cpu", check_hash=True
            )
        else:
            detr_state_dict: dict[str, Tensor] = torch.load(detr_state_dict, weights_only=True, map_location="cpu")

    if "model" in detr_state_dict: # checkpoint format
        detr_state_dict = detr_state_dict["model"]

    model_state_dict = model.state_dict()
    initialized_param_names: list[str] = []
    shape_unmatched_param_names: list[str] = []
    unloaded_param_names: list[str] = []

    def _validate_and_cover(target_key: str, src_key: str, record: bool = True):
        target_state, detr_state = model_state_dict[target_key], detr_state_dict[src_key]

        if not target_state.shape == detr_state.shape:
            msg = "shape mismatch: {} vs. {} at layer `{}` - `{}`".format(
                target_state.shape, detr_state.shape, target_key, src_key
            )
            if not skip_mismatch:
                raise ValueError(msg)
            # warnings.warn(msg)
            shape_unmatched_param_names.append(target_key)
        else:
            model_state_dict[target_key] = detr_state

        if record:
            initialized_param_names.append(target_key)
        
        if verbose:
            print("loaded detr state: `{}` to model state: `{}`".format(src_key, target_key))

    for _param_name, _ in model_state_dict.items():
        if _param_name not in detr_state_dict:
            unloaded_param_names.append(_param_name)
            continue

        _validate_and_cover(_param_name, _param_name)

    model.load_state_dict(model_state_dict)

    print("Unmatched modules: ", shape_unmatched_param_names)
    print("unloaded modules: ", unloaded_param_names)

    return model


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    cls_map = dict(coco=91, coco_panoptic=250, mot17=1, joint=1, bddmot20=11)
    num_classes = 20 if args.dataset_file not in cls_map else cls_map[args.dataset_file]
    device = torch.device(args.device)

    backbone = build_backbone(args)
    transformer = build_st_transformer(args)

    model = AssociaTR(
        args.num_frames,
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )

    matcher = build_clip_matcher(args) if args.num_frames > 1 else build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef, 'loss_objness': args.objness_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.num_frames > 1:
        losses.insert(2, "objness")

    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcessForCOCODet()}
    if args.num_frames > 1:
        postprocessors.update({'mot': PostProcessForMOT()})

    return model, criterion, postprocessors
