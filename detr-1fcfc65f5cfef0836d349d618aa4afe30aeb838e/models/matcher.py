# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

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
        # 权重保存
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels,so it should be [batch_size,num_target_boxes] 
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates  ,Tensor of dim likes
                          [batch_size,num_target_boxes,4]

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
            A list of size is [batch_size,min(num_queries,num_target_boxes)]
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])  #[batch_size*obj_num]            # obj_num is not constant
        tgt_bbox = torch.cat([v["boxes"] for v in targets])  #[batch_size*obj_num,4]

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        # cost_class 每一行代表这预测一个bbox类别，与target类别之间的对应得分
        # Each row of cost_class represents the corresponding core 
        # between the predicted bbox category and the target category
        cost_class = -out_prob[:, tgt_ids]  #[batch_size * num_queries,batch_size*obj_num]  

        # Compute the L1 cost between boxes
        # Each row of cost_bbox represents the corresponding core of L1 regularization 
        # between the predicted out_bbox coordinate and the tgt_box_coordinate 
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)  #[batch_size * num_queries,batch_size*obj_num] 

        # Compute the giou cost betwen boxes
        # [batch_size * num_queries,batch_size*obj_num]
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        # C size [bs,num_queries,bs*num_obj]
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        # c[i]  [num_queries,num_obj]  num_queries,num_obj, 每个图 预测框与真实框对关系  nu_queries就是预测狂 ，nb就是True框
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        # 返回二部图，点对匹配 size(bs,2,num_obj) 0 is pred and 1 is true
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    # cost_class:类别权重
    # cost_bbox:类别权重
    # cost_args.set_cost_giou:并交比权重
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)
