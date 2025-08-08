# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch

from typing import Literal, Union
from scipy.optimize import linear_sum_assignment
from torch import nn, Tensor

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

        self.target_mats = None

    def _construct_target_matrices(self, targets: list[dict[Literal["labels", "boxes", "track_ids"], Tensor]]):
        mats: list[Tensor] = []
        for batch_index in range(self.batch_size):
            target_clip = targets[batch_index * self.num_frames: (batch_index + 1) * self.num_frames]
            instances: Tensor = torch.cat([target["track_ids"] for target in target_clip]).unique()

            mat = torch.zeros([len(instances), 1 + 1 + 4 * self.num_frames + self.num_frames], device=instances.device).float() # dim2: ids + cls + T * coords + T_objness
            mat[:, 0] = instances.float()

            for frame_index, target in enumerate(target_clip):
                for target_index, track_id in enumerate(target["track_ids"]):
                    mat_index = torch.where(mat[:, 0] == track_id)
                    mat[mat_index, 1] = target["labels"][target_index].float()
                    # note that boxes of a instance across frames are normalized (cx and w)
                    # so that those cross-frame boxes are considered in one frame, which is benefit for prediction
                    _box = target["boxes"][target_index].clone()
                    _box[0::2] += frame_index
                    _box[0::2] /= self.num_frames
                    mat[mat_index, 2 + frame_index * 4: 2 + (frame_index + 1) * 4] = _box
                    mat[mat_index, 2 + self.num_frames * 4 + frame_index] = 1.

            mats.append(mat)

        self.target_mats = mats

        return mats
    
    def get_bs_numframes_(self, outputs: dict[Literal["pred_logits", "pred_boxes", "pred_objness"], Tensor]):
        _shape = outputs["pred_objness"].size()
        self.num_frames = _shape[-1]
        self.batch_size = _shape[0]

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        self.get_bs_numframes_(outputs)
        self._construct_target_matrices(targets)
        bs, num_queries = outputs["pred_logits"].shape[:2] # [B, N, C]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids] # range [-1, 0] shape [BN, BM]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1) # [BN, BM]

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        # for i, c in enumerate(C.split(sizes, -1)):
        #     print(f"batch {i + 1}\n{c}")
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
    

class ClipHungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, cost_objness: float = None) -> None:
        super().__init__()

        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_objness = self.cost_class if not cost_objness else cost_objness

        self.num_frames: int = None
        self.batch_size: int = None

        self.target_mats: list[Tensor] = None

        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def _construct_target_matrices(self, targets: list[dict[Literal["labels", "boxes", "track_ids"], Tensor]]):
        mats: list[Tensor] = []
        for batch_index in range(self.batch_size):
            target_clip = targets[batch_index * self.num_frames: (batch_index + 1) * self.num_frames]
            instances: Tensor = torch.cat([target["track_ids"] for target in target_clip]).unique()

            mat = torch.zeros([len(instances), 1 + 1 + 4 * self.num_frames + self.num_frames], device=instances.device).float() # dim2: ids + cls + T * coords + T_objness
            mat[:, 0] = instances.float()

            for frame_index, target in enumerate(target_clip):
                for target_index, track_id in enumerate(target["track_ids"]):
                    mat_index = torch.where(mat[:, 0] == track_id)
                    mat[mat_index, 1] = target["labels"][target_index].float()
                    # note that boxes of a instance across frames are normalized (cx and w)
                    # so that those cross-frame boxes are considered in one frame, which is benefit for prediction
                    _box = target["boxes"][target_index].clone()
                    _box[0::2] += frame_index
                    _box[0::2] /= self.num_frames
                    mat[mat_index, 2 + frame_index * 4: 2 + (frame_index + 1) * 4] = _box
                    mat[mat_index, 2 + self.num_frames * 4 + frame_index] = 1.

            mats.append(mat)

        self.target_mats = mats

        return mats

    def get_bs_numframes_(self, outputs: dict[Literal["pred_logits", "pred_boxes", "pred_objness"], Tensor]):
        _shape = outputs["pred_objness"].size()
        self.num_frames = _shape[-1]
        self.batch_size = _shape[0]

    @torch.no_grad()
    def forward(self, 
                outputs: dict[Literal["pred_logits", "pred_boxes", "pred_objness"], Tensor], 
                targets: list[dict[Literal["labels", "boxes", "track_ids"], Tensor]]):
        """ Performs the matching

        Args:
        ---
            - outputs: This is a dict that contains at least these entries:
                - "pred_logits": Tensor of dim [B, Nq, C] with the classification logits
                - "pred_boxes": Tensor of dim [B, Nq, Tx4] with the predicted box coordinates \\
                    where dimension `T`: number of frames in a video 'clip'
                - "pred_objness": [B, Nq, T] denotes whether a object instance exists in each frame.

            - targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                - "labels": Tensor of dim [M] (where M is the number of ground-truth
                        objects in the target) containing the class labels
                - "boxes": Tensor of dim [M, T, 4] containing the target box coordinates. 
                    See `outputs` for details of dimension `T`. 
                - "track_ids": ids marking inter-frame instances, a list of strings.

        Returns:
        ---
            - A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order) \\
                For each batch element, it holds:
                    len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        self.get_bs_numframes_(outputs)
        mats = self._construct_target_matrices(targets) # [B_i, N_instances_i, 2 + 5 * N_frames]
        assert len(mats) == self.batch_size

        # construct Hugnarian cost matrices batch-wise
        indices = []
        for batch_index, mat in enumerate(mats):
            objness = mat[:, -self.num_frames:] # [N_obj, N_frames]
            # compute class cost
            probs = outputs["pred_logits"][batch_index].softmax(-1)
            cost_cls = -probs[:, mat[:, 1].long()] # [N_q, N_obj]

            # compute boxes cost
            pred_boxes = outputs["pred_boxes"][batch_index] # [N_q, T * 4]
            target_boxes = mat[:, 2: 2 + self.num_frames * 4] # [N_obj, T * 4]

            ## compute cdist with p = 1, [N_q, N_obj, N_frames, 4] -> [N_q, N_obj, N_frames]
            cdists = (pred_boxes[:, None] - target_boxes[None]).abs().unflatten(-1, [-1, 4]).sum(-1)
            mask = objness.unsqueeze(0).repeat(cdists.size(0), 1, 1).bool()
            cdists[~mask] = 0. # distance where no obj set to 0.
            cost_box_dist = torch.div(cdists.sum(-1), objness.sum(-1)[None]) # normalize cost by num of objness of each instance, [N_q, N_obj]

            ## compute giou
            gious = []
            for pred_boxes_t, target_boxes_t in zip(
                pred_boxes.split([4] * self.num_frames, -1),
                target_boxes.split([4] * self.num_frames, -1),
            ):
                giou = generalized_box_iou(
                        box_cxcywh_to_xyxy(pred_boxes_t),
                        box_cxcywh_to_xyxy(target_boxes_t)
                    )[..., None] # [N_q, N_obj, 1]
                gious.append(giou)

            gious = -torch.cat(gious, dim=-1) # [N_q, N_obj, N_frames]
            gious[~mask] = 0.
            cost_box_giou = torch.div(gious.sum(-1), objness.sum(-1)[None]) # [N_q, N_obj]

            # compute objectness cost
            cost_objness = 0.
            if self.num_frames > 1:
                pred_objness = outputs["pred_objness"][batch_index].sigmoid() # [N_q, N_frames]
                cost_objness = -pred_objness[:, None] * objness[None] # [N_q, N_obj, N_frames]
                cost_objness -= (1 - pred_objness)[:, None] * (1 - objness)[None]
                cost_objness = cost_objness.sum(-1) # [N_q, N_obj]

            batch_cost = cost_cls * self.cost_class + cost_box_dist * self.cost_bbox + cost_box_giou * self.cost_giou + cost_objness * self.cost_objness # [N_q, N_obj]
            
            indices.append(linear_sum_assignment(batch_cost.cpu()))

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)

def build_clip_matcher(args):
    return ClipHungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)
