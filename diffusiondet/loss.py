"""
DiffusionDet model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
from fvcore.nn import sigmoid_focal_loss_jit
import torchvision.ops as ops
from .util import box_ops
from .util.misc import get_world_size, is_dist_avail_and_initialized
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, complete_box_iou, distance_box_iou, generalized_box_iou
import math

def angle2matrix(bz_r_pose):
    # Convert angles to radians
    bz_r_pose_rad = bz_r_pose * (math.pi / 180.0)

    cos_x = torch.cos(bz_r_pose_rad[:, 0])
    sin_x = torch.sin(bz_r_pose_rad[:, 0])
    cos_y = torch.cos(bz_r_pose_rad[:, 1])
    sin_y = torch.sin(bz_r_pose_rad[:, 1])
    cos_z = torch.cos(bz_r_pose_rad[:, 2])
    sin_z = torch.sin(bz_r_pose_rad[:, 2])

    rotation_matrixes = torch.zeros(bz_r_pose.size(0), 3, 3, dtype=torch.float32, device=bz_r_pose.device)
    rotation_matrixes[:, 0, 0] = cos_y * cos_z
    rotation_matrixes[:, 0, 1] = - cos_x * sin_z + sin_x * sin_y * cos_z
    rotation_matrixes[:, 0, 2] = sin_x * sin_z + cos_x * sin_y * cos_z
    rotation_matrixes[:, 1, 0] = cos_y * sin_z
    rotation_matrixes[:, 1, 1] = cos_x * cos_z + sin_x * sin_y * sin_z
    rotation_matrixes[:, 1, 2] = - sin_x * cos_z + cos_x * sin_y * sin_z
    rotation_matrixes[:, 2, 0] = - sin_y
    rotation_matrixes[:, 2, 1] = sin_x * cos_y
    rotation_matrixes[:, 2, 2] = cos_x * cos_y

    return rotation_matrixes

def matrix2angle(bz_rotation_matrix):
    
    factor = (180.0 / math.pi)
    sy = torch.sqrt(bz_rotation_matrix[:, 0, 0] * bz_rotation_matrix[:, 0, 0] +  bz_rotation_matrix[:, 1, 0] * bz_rotation_matrix[:, 1, 0])
    singular = sy < 1e-6
    
    rx = torch.atan2(bz_rotation_matrix[:, 2, 1] , bz_rotation_matrix[:, 2, 2])  # 
    ry = torch.atan2(-bz_rotation_matrix[:, 2, 0], sy)
    rz = torch.atan2(bz_rotation_matrix[:, 1, 0], bz_rotation_matrix[:, 0, 0])

    rx[singular] = torch.atan2(-bz_rotation_matrix[singular, 1, 2], bz_rotation_matrix[singular, 1, 1])
    ry[singular] = torch.atan2(-bz_rotation_matrix[singular, 2, 0], sy[singular])
    rz[singular] = 0

    # Convert radians to Euler angles
    angles = torch.stack((rx, ry, rz), dim=1) * factor # [numgts,3]
        
    return angles 

def box3d(src_pose6dof, target_3dboxes_lwh):
    """ 
    Args:
        src_pose6dof (tensor[num_dts,6]):
        target_3dboxes_lwh (tensor[num_dts,3]):
    Returns:
        pre_3dboxes (): 
    """
    src_box_center = src_pose6dof[:, :3] 
    src_rotation_matrix = angle2matrix(src_pose6dof[:, 3:])
    
    l, w, h = target_3dboxes_lwh[:,0],target_3dboxes_lwh[:,1],target_3dboxes_lwh[:,2] 
    x_corners = torch.stack([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2], dim=1).to(src_rotation_matrix.device) 
    zeros=torch.zeros(l.shape,device=src_rotation_matrix.device) 
    y_corners = torch.stack([zeros,zeros,zeros,zeros,-h,-h,-h,-h], dim=1).to(src_rotation_matrix.device) 
    z_corners = torch.stack([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2], dim=1).to(src_rotation_matrix.device)
    corners = torch.stack([x_corners, y_corners, z_corners], dim=2) 
    pre_3dboxes = torch.matmul(corners, src_rotation_matrix.transpose(1, 2)) + src_box_center.unsqueeze(1).expand(-1, 8, -1) 
    
    return pre_3dboxes 

def xyzs_to_xys(bz_tgt_P2, bz_src_3dboxes):
    """_summary_

    Args:
        bz_tgt_P2 (tensor[nums,3,4]): _description_
        bz_src_3dboxes (tensor[nums,8,3]): 
    Return
        3dboxes_xy(tensor[nums,8,2]): 
    """
    ones_column = torch.ones((bz_src_3dboxes.shape[0],bz_src_3dboxes.shape[1],1), device=bz_src_3dboxes.device, dtype=torch.float32)
    boxes_xyz_homo = torch.cat((bz_src_3dboxes, ones_column), dim=2)  
    boxes_xy = torch.matmul(boxes_xyz_homo, bz_tgt_P2.permute(0, 2, 1)) 
    boxes_xy = boxes_xy[:, :, :2] / boxes_xy[:, :, 2:]
    
    return boxes_xy[:, :, :2]

def projection(bz_tgt_P2, bz_src_3dboxes):
    """
    Args:
        bz_tgt_P2 (tensor[13,3,4]): 
        bz_src_3dboxes (tensor[13,8,3]): 
    return:
        min_max_coords (tensor[13,4,2]):
    """
    ones_column = torch.ones((bz_src_3dboxes.shape[0],bz_src_3dboxes.shape[1],1), device=bz_src_3dboxes.device, dtype=torch.float32) 
    boxes_xyz_homo = torch.cat((bz_src_3dboxes, ones_column), dim=2) 
    boxes_xy = torch.matmul(boxes_xyz_homo, bz_tgt_P2.permute(0, 2, 1))  
    boxes_xy = boxes_xy[:, :, :2] / boxes_xy[:, :, 2:]
    x = boxes_xy[:, :, 0]
    y = boxes_xy[:, :, 1]
    min_x, _ = torch.min(x, dim=1, keepdim=True)
    max_x, _ = torch.max(x, dim=1, keepdim=True)
    min_y, _ = torch.min(y, dim=1, keepdim=True)
    max_y, _ = torch.max(y, dim=1, keepdim=True)
    min_max_coords = torch.cat((min_x, min_y, max_x, max_y), dim=1)  
    return min_max_coords

class SetCriterionDynamicK(nn.Module):
    """ This class computes the loss for DiffusionDet.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, cfg, num_classes, matcher, weight_dict, eos_coef, losses, use_focal):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.use_focal = use_focal
        self.use_fed_loss = cfg.MODEL.DiffusionDet.USE_FED_LOSS
        if self.use_fed_loss:
            self.fed_loss_num_classes = 50
            from detectron2.data.detection_utils import get_fed_loss_cls_weights
            cls_weight_fun = lambda: get_fed_loss_cls_weights(dataset_names=cfg.DATASETS.TRAIN, freq_weight_power=cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT_POWER)  
            fed_loss_cls_weights = cls_weight_fun()
            assert (
                    len(fed_loss_cls_weights) == self.num_classes
            ), "Please check the provided fed_loss_cls_weights. Their size should match num_classes"
            self.register_buffer("fed_loss_cls_weights", fed_loss_cls_weights)

        if self.use_focal:
            self.focal_loss_alpha = cfg.MODEL.DiffusionDet.ALPHA
            self.focal_loss_gamma = cfg.MODEL.DiffusionDet.GAMMA
        else:
            empty_weight = torch.ones(self.num_classes + 1)
            empty_weight[-1] = self.eos_coef
            self.register_buffer('empty_weight', empty_weight)

    # copy-paste from https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/roi_heads/fast_rcnn.py#L356
    def get_fed_loss_classes(self, gt_classes, num_fed_loss_classes, num_classes, weight):
        """
        Args:
            gt_classes: a long tensor of shape R that contains the gt class label of each proposal.
            num_fed_loss_classes: minimum number of classes to keep when calculating federated loss.
            Will sample negative classes if number of unique gt_classes is smaller than this value.
            num_classes: number of foreground classes
            weight: probabilities used to sample negative classes
        Returns:
            Tensor:
                classes to keep when calculating the federated loss, including both unique gt
                classes and sampled negative classes.
        """
        unique_gt_classes = torch.unique(gt_classes)
        prob = unique_gt_classes.new_ones(num_classes + 1).float()
        prob[-1] = 0
        if len(unique_gt_classes) < num_fed_loss_classes:
            prob[:num_classes] = weight.float().clone()
            prob[unique_gt_classes] = 0
            sampled_negative_classes = torch.multinomial(
                prob, num_fed_loss_classes - len(unique_gt_classes), replacement=False
            )
            fed_loss_classes = torch.cat([unique_gt_classes, sampled_negative_classes])
        else:
            fed_loss_classes = unique_gt_classes
        return fed_loss_classes
    
    def loss_labels(self, outputs, targets, indices, num_boxes, log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        batch_size = len(targets)

        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        src_logits_list = []
        target_classes_o_list = []
        for batch_idx in range(batch_size):
            valid_query = indices[batch_idx][0] 
            gt_multi_idx = indices[batch_idx][1]
            if len(gt_multi_idx) == 0:
                continue
            bz_src_logits = src_logits[batch_idx]
            target_classes_o = targets[batch_idx]["labels"] 
            target_classes[batch_idx, valid_query] = target_classes_o[gt_multi_idx] 

            src_logits_list.append(bz_src_logits[valid_query])
            target_classes_o_list.append(target_classes_o[gt_multi_idx])

        if self.use_focal or self.use_fed_loss:
            num_boxes = torch.cat(target_classes_o_list).shape[0] if len(target_classes_o_list) != 0 else 1 

            target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], self.num_classes + 1],
                                                dtype=src_logits.dtype, layout=src_logits.layout,
                                                device=src_logits.device) 
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1) 

            gt_classes = torch.argmax(target_classes_onehot, dim=-1)
            target_classes_onehot = target_classes_onehot[:, :, :-1]

            src_logits = src_logits.flatten(0, 1)
            target_classes_onehot = target_classes_onehot.flatten(0, 1)
            if self.use_focal:
                cls_loss = sigmoid_focal_loss_jit(src_logits, target_classes_onehot, alpha=self.focal_loss_alpha, gamma=self.focal_loss_gamma, reduction="none")
            else:
                cls_loss = F.binary_cross_entropy_with_logits(src_logits, target_classes_onehot, reduction="none")
            if self.use_fed_loss: 
                K = self.num_classes
                N = src_logits.shape[0]
                fed_loss_classes = self.get_fed_loss_classes(
                    gt_classes,
                    num_fed_loss_classes=self.fed_loss_num_classes,
                    num_classes=K,
                    weight=self.fed_loss_cls_weights,
                )
                fed_loss_classes_mask = fed_loss_classes.new_zeros(K + 1)
                fed_loss_classes_mask[fed_loss_classes] = 1
                fed_loss_classes_mask = fed_loss_classes_mask[:K]
                weight = fed_loss_classes_mask.view(1, K).expand(N, K).float()

                loss_ce = torch.sum(cls_loss * weight) / num_boxes
            else:
                loss_ce = torch.sum(cls_loss) / num_boxes

            losses = {'loss_ce': loss_ce}
        else:
            raise NotImplementedError

        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        src_boxes = outputs['pred_boxes']

        batch_size = len(targets)
        pred_box_list = []
        pred_norm_box_list = []
        tgt_box_list = []
        tgt_box_xyxy_list = []
        for batch_idx in range(batch_size):
            valid_query = indices[batch_idx][0]
            gt_multi_idx = indices[batch_idx][1]
            if len(gt_multi_idx) == 0:
                continue
            bz_image_whwh = targets[batch_idx]['image_size_xyxy']
            bz_src_boxes = src_boxes[batch_idx] 
            bz_target_boxes = targets[batch_idx]["boxes"]  
            bz_target_boxes_xyxy = targets[batch_idx]["boxes_xyxy"]  
            pred_box_list.append(bz_src_boxes[valid_query])
            pred_norm_box_list.append(bz_src_boxes[valid_query] / bz_image_whwh)  
            tgt_box_list.append(bz_target_boxes[gt_multi_idx])
            tgt_box_xyxy_list.append(bz_target_boxes_xyxy[gt_multi_idx])

        if len(pred_box_list) != 0:
            src_boxes = torch.cat(pred_box_list)
            src_boxes_norm = torch.cat(pred_norm_box_list)  
            target_boxes = torch.cat(tgt_box_list)
            target_boxes_abs_xyxy = torch.cat(tgt_box_xyxy_list)
            num_boxes = src_boxes.shape[0] 

            losses = {}
            loss_bbox = F.l1_loss(src_boxes_norm, box_cxcywh_to_xyxy(target_boxes), reduction='none')
            losses['loss_bbox'] = loss_bbox.sum() / num_boxes

            loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(src_boxes, target_boxes_abs_xyxy))
            losses['loss_giou'] = loss_giou.sum() / num_boxes
        else:
            losses = {'loss_bbox': outputs['pred_boxes'].sum() * 0,
                      'loss_giou': outputs['pred_boxes'].sum() * 0}

        return losses
    
    def loss_pose6dof(self, outputs, targets, indices, num_boxes):
        """6dofloss, consisting of l1 loss of rotation and translation.
        targets dicts must contain the key "translation_matrix" containing a tensor of dim [nb_target_boxes, 3]
        targets dicts must contain the key "rotation_matrix" containing a tensor of dim [nb_target_boxes, 3, 3]
        """
        assert 'pred_poses' in outputs
        src_poses = outputs['pred_poses'] 
        batch_size = len(targets)
        
        pred_translation_list = []
        pred_rotation_list = []
        tgt_translation_list = []
        tgt_rotation_list = []
        tgt_area_list = []
        for batch_idx in range(batch_size):
            valid_query = indices[batch_idx][0] 
            gt_multi_idx = indices[batch_idx][1]
            if len(gt_multi_idx) == 0:
                continue
            bz_target_translation_vector = targets[batch_idx]['translation_matrix'][gt_multi_idx] 
            bz_target_rotation_matrix = targets[batch_idx]['rotation_matrix'][gt_multi_idx]
            bz_src_translation_vector = src_poses[batch_idx][:, :3][valid_query] 
            bz_src_rotation_angle = src_poses[batch_idx][:, 3:][valid_query]  
            bz_target_area = targets[batch_idx]["area"][gt_multi_idx] 
            
            pred_translation_list.append(bz_src_translation_vector) 
            pred_rotation_list.append(bz_src_rotation_angle) 
            tgt_translation_list.append(bz_target_translation_vector) 
            tgt_rotation_list.append(bz_target_rotation_matrix) 
            tgt_area_list.append(bz_target_area)

        if len(pred_translation_list) != 0 or len(pred_rotation_list) != 0:
            src_translation = torch.cat(pred_translation_list) 
            src_rotation = torch.cat(pred_rotation_list) 
            target_translation = torch.cat(tgt_translation_list)
            target_rotation = torch.cat(tgt_rotation_list) 
            num_boxes = src_translation.shape[0] 

            losses = {}
            loss_translation = F.l1_loss(src_translation, target_translation, reduction='none') 
            losses['loss_translation'] = loss_translation.sum() / num_boxes / 3 
            loss_rotation = F.l1_loss(src_rotation, target_rotation, reduction='none') 
            losses['loss_rotation'] = loss_rotation.sum() / num_boxes / 3 
        else:
            losses = {'loss_translation': outputs['pred_boxes'].sum() * 0,
                        'loss_rotation': outputs['pred_boxes'].sum() * 0}

        return losses
    
    def loss_union3d2d(self, outputs, targets, indices, num_boxes):
        """ 
        Project 3dbox into 2d space and then obtain its maximum circumscribed rectangle, and design an l1loss with the result using gt2dbox
        """
        assert 'pred_poses' in outputs
        src_poses = outputs['pred_poses'] 
        src_boxes = outputs['pred_boxes'] 
        batch_size = len(targets)
        
        src_3dboxes_list = []
        tgt_3dboxes_list = []
        src_boxes_list = []
        target_boxes_list = []
        src_2dboxes_list = []
        for batch_idx in range(batch_size):
            valid_query = indices[batch_idx][0]
            gt_multi_idx = indices[batch_idx][1]
            if len(gt_multi_idx) == 0:
                continue
            bz_image_whwh = targets[batch_idx]['image_size_xyxy'] 
       
            bz_target_3dboxes = targets[batch_idx]['3dboxes'][gt_multi_idx] 
            tgt_3dboxes_list.append(bz_target_3dboxes)
            bz_target_3dboxes_lwh = targets[batch_idx]['lwhs'][gt_multi_idx] 
            bz_src_pose6dof = src_poses[batch_idx][valid_query]
            bz_src_3dboxes = box3d(bz_src_pose6dof, bz_target_3dboxes_lwh) 
            src_3dboxes_list.append(bz_src_3dboxes)
            
            bz_tgt_P2 = targets[batch_idx]['P2s'][gt_multi_idx] 
            bz_src_2dboxes = projection(bz_tgt_P2, bz_src_3dboxes) 
            src_2dboxes_list.append(bz_src_2dboxes/bz_image_whwh) 
            bz_src_boxes = src_boxes[batch_idx][valid_query] 
            src_boxes_list.append(bz_src_boxes/bz_image_whwh) 
            bz_target_boxes = targets[batch_idx]["boxes"][gt_multi_idx] 
            target_boxes_list.append(box_cxcywh_to_xyxy(bz_target_boxes))
        
        if len(target_boxes_list) != 0:
            src_3dboxes = torch.cat(src_3dboxes_list)
            tgt_3dboxes = torch.cat(tgt_3dboxes_list) 
            src_2dboxes = torch.cat(src_2dboxes_list) 
            src_o_boxes = torch.cat(src_boxes_list) 
            tgt_boxes = torch.cat(target_boxes_list)
            
            num_boxes = tgt_boxes.shape[0]
            
            losses = {}
            loss_3dbox = F.l1_loss(src_3dboxes, tgt_3dboxes, reduction='none')
            losses['loss_3dbox'] = loss_3dbox.sum() / num_boxes / 24
            loss_2dbox2tgtbox = F.l1_loss(src_2dboxes, tgt_boxes, reduction='none')
            losses['loss_2dbox2tgtbox'] = loss_2dbox2tgtbox.sum() / num_boxes
            loss_2dbox2srcbox = F.l1_loss(src_2dboxes, src_o_boxes, reduction='none')
            losses['loss_2dbox2srcbox'] = loss_2dbox2srcbox.sum() / num_boxes 
        else:
            losses = {  'loss_3dbox': outputs['pred_poses'].sum() * 0,
                        'loss_2dbox2tgtbox': outputs['pred_poses'].sum() * 0,
                        'loss_2dbox2srcbox': outputs['pred_poses'].sum() * 0,
            } 
            
        return losses
    
    def loss_3dgiou(self, outputs, targets, indices, num_boxes):
        """ 
        Project 3dbox into 2d space and then obtain its maximum circumscribed rectangle, and design an l1loss with the result using gt2dbox
        """
        assert 'pred_poses' in outputs
        src_poses = outputs['pred_poses']  
        src_boxes = outputs['pred_boxes'] 
        batch_size = len(targets)
        
        src_3dboxes_list = []
        tgt_3dboxes_list = []
        src_boxes_list = []
        target_boxes_list = []
        src_2dboxes_abs_xyxy_list = []
        src_2dboxes_list = []
        tgt_2dboxes_abs_xyxy_list = []
        for batch_idx in range(batch_size):
            valid_query = indices[batch_idx][0]
            gt_multi_idx = indices[batch_idx][1]
            if len(gt_multi_idx) == 0:
                continue
            bz_image_whwh = targets[batch_idx]['image_size_xyxy']
            bz_target_3dboxes = targets[batch_idx]['3dboxes'][gt_multi_idx] 
            tgt_3dboxes_list.append(bz_target_3dboxes)
            bz_target_3dboxes_lwh = targets[batch_idx]['lwhs'][gt_multi_idx]
            trans_factor = torch.tensor([[31,21,100]],device=bz_image_whwh.device)
            rotate_factor = torch.tensor([[90,40,90]],device=bz_image_whwh.device)  
            bz_src_pose6dof = src_poses[batch_idx][valid_query]*torch.cat((trans_factor,rotate_factor),dim=-1) 
            bz_src_3dboxes = box3d(bz_src_pose6dof, bz_target_3dboxes_lwh) 
            src_3dboxes_list.append(bz_src_3dboxes)
            
            bz_tgt_P2 = targets[batch_idx]['P2s'][gt_multi_idx] 
            bz_src_2dboxes = projection(bz_tgt_P2, bz_src_3dboxes) 
            src_2dboxes_abs_xyxy_list.append(bz_src_2dboxes)
            src_2dboxes_list.append(bz_src_2dboxes/bz_image_whwh) 
            bz_src_boxes = src_boxes[batch_idx][valid_query] 
            src_boxes_list.append(bz_src_boxes/bz_image_whwh)
            bz_target_boxes = targets[batch_idx]["boxes"][gt_multi_idx] 
            target_boxes_list.append(box_cxcywh_to_xyxy(bz_target_boxes))
            tgt_2dboxes_abs_xyxy_list.append(targets[batch_idx]["boxes_xyxy"][gt_multi_idx]) 
        
        if len(tgt_3dboxes_list) != 0:
            src_3dboxes = torch.cat(src_3dboxes_list) 
            tgt_3dboxes = torch.cat(tgt_3dboxes_list) 
            src_2dboxes_abs_xyxy = torch.cat(src_2dboxes_abs_xyxy_list)  
            src_2dboxes = torch.cat(src_2dboxes_list)  
            src_o_boxes = torch.cat(src_boxes_list) 
            tgt_boxes = torch.cat(target_boxes_list)
            tgt_2dboxes_abs_xyxy = torch.cat(tgt_2dboxes_abs_xyxy_list)
            
            num_boxes = tgt_3dboxes.shape[0]
            
            losses = {}
            loss_3dbox = F.l1_loss(src_2dboxes, tgt_boxes, reduction='none') 
            losses['loss_3dbbox'] = loss_3dbox.sum() / num_boxes
            loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(src_2dboxes_abs_xyxy, tgt_2dboxes_abs_xyxy)) 
            losses['loss_3dgiou'] = loss_giou.sum() / num_boxes
        else:
            losses = {  'loss_3dbbox': outputs['pred_poses'].sum() * 0,
                        'loss_3dgiou': outputs['pred_poses'].sum() * 0,
            } 
            
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'pose6dof': self.loss_pose6dof,
            'union3d2d': self.loss_union3d2d,
            '3dgiou': self.loss_3dgiou,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices, _ = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices, _ = self.matcher(aux_outputs, targets)
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


class HungarianMatcherDynamicK(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-k (dynamic) matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    def __init__(self, cfg, cost_class: float = 1, 
                 cost_bbox: float = 1, 
                 cost_giou: float = 1,
                 cost_t: float = 1, 
                 cost_r: float = 1, 
                 cost_3dgiou: float = 1,
                 cost_3dbbox: float = 1,
                 cost_mask: float = 1, 
                 use_focal: bool = False):
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
        self.cost_t = cost_t
        self.cost_r = cost_r
        self.cost_3dgiou = cost_3dgiou
        self.cost_3dbbox = cost_3dbbox
        self.use_focal = use_focal
        self.use_fed_loss = cfg.MODEL.DiffusionDet.USE_FED_LOSS
        self.ota_k = cfg.MODEL.DiffusionDet.OTA_K
        if self.use_focal:
            self.focal_loss_alpha = cfg.MODEL.DiffusionDet.ALPHA
            self.focal_loss_gamma = cfg.MODEL.DiffusionDet.GAMMA
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0,  "all costs cant be 0"
    
    def forward(self, outputs, targets):
        """ simOTA for detr"""
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]
            # We flatten to compute the cost matrices in a batch
            if self.use_focal or self.use_fed_loss:
                out_prob = outputs["pred_logits"].sigmoid() 
                out_bbox = outputs["pred_boxes"] 
                out_pose = outputs["pred_poses"] 
            else:
                out_prob = outputs["pred_logits"].softmax(-1) 
                out_bbox = outputs["pred_boxes"]  
                out_pose = outputs["pred_poses"]

            indices = []
            matched_ids = []
            assert bs == len(targets)
            for batch_idx in range(bs):
                bz_boxes = out_bbox[batch_idx] 
                bz_out_prob = out_prob[batch_idx] 
                bz_pose = out_pose[batch_idx] 
                bz_tgt_ids = targets[batch_idx]["labels"] 
                num_insts = len(bz_tgt_ids)
                if num_insts == 0: 
                    non_valid = torch.zeros(bz_out_prob.shape[0]).to(bz_out_prob) > 0
                    indices_batchi = (non_valid, torch.arange(0, 0).to(bz_out_prob))
                    matched_qidx = torch.arange(0, 0).to(bz_out_prob)
                    indices.append(indices_batchi)
                    matched_ids.append(matched_qidx)
                    continue
                
                bz_gtboxs = targets[batch_idx]['boxes']  
                bz_gtboxs_abs_xyxy = targets[batch_idx]['boxes_xyxy']  
                
                pre_trans = bz_pose[:, :3] 
                ones_column = torch.ones((pre_trans.shape[0],1), device=pre_trans.device, dtype=torch.float32)
                pre_trans_column = torch.cat((pre_trans, ones_column), dim=1) 
                pre_trans_xy = torch.matmul(pre_trans_column, targets[batch_idx]["P2s"][0].permute(1,0))  
                pre_trans_center = pre_trans_xy[:, :2] / pre_trans_xy[:, 2:] 
           
                pair_wise_ious = ops.box_iou(bz_boxes, bz_gtboxs_abs_xyxy)
                
                
                # Compute the classification cost.
                if self.use_focal:
                    alpha = self.focal_loss_alpha
                    gamma = self.focal_loss_gamma
                    neg_cost_class = (1 - alpha) * (bz_out_prob ** gamma) * (-(1 - bz_out_prob + 1e-8).log()) 
                    pos_cost_class = alpha * ((1 - bz_out_prob) ** gamma) * (-(bz_out_prob + 1e-8).log()) 
                    cost_class = pos_cost_class[:, bz_tgt_ids] - neg_cost_class[:, bz_tgt_ids] 
                elif self.use_fed_loss:
                    # focal loss degenerates to naive one
                    neg_cost_class = (-(1 - bz_out_prob + 1e-8).log())
                    pos_cost_class = (-(bz_out_prob + 1e-8).log())
                    cost_class = pos_cost_class[:, bz_tgt_ids] - neg_cost_class[:, bz_tgt_ids]
                else:
                    cost_class = -bz_out_prob[:, bz_tgt_ids]

                # Compute the L1 cost between boxes
                # image_size_out = torch.cat([v["image_size_xyxy"].unsqueeze(0) for v in targets])
                # image_size_out = image_size_out.unsqueeze(1).repeat(1, num_queries, 1).flatten(0, 1)
                # image_size_tgt = torch.cat([v["image_size_xyxy_tgt"] for v in targets])

                bz_image_size_out = targets[batch_idx]['image_size_xyxy']
                bz_image_size_tgt = targets[batch_idx]['image_size_xyxy_tgt'] 

                bz_out_bbox_ = bz_boxes / bz_image_size_out  
                bz_tgt_bbox_ = bz_gtboxs_abs_xyxy / bz_image_size_tgt  
                cost_bbox = torch.cdist(bz_out_bbox_, bz_tgt_bbox_, p=1) 

                cost_giou = -generalized_box_iou(bz_boxes, bz_gtboxs_abs_xyxy) 

                bz_pose_translation_vector = bz_pose[:, :3] 
                bz_pose_rotation_angle = bz_pose[:, 3:]
                bz_gtpose_translation_vector = targets[batch_idx]["translation_matrix"] 
                bz_gtpose_rotation_angle = targets[batch_idx]["rotation_matrix"]
                cost_t_pose = torch.cdist(bz_pose_translation_vector, bz_gtpose_translation_vector, p=1)
                cost_r_pose = torch.cdist(bz_pose_rotation_angle, bz_gtpose_rotation_angle, p=1)
                
                gious_pose = generalized_box_iou(bz_boxes, bz_gtboxs_abs_xyxy) 
                _,index = torch.max(gious_pose,dim=-1)
                bz_prelwhs = targets[batch_idx]['lwhs'][index]      
                
                pair_wise_ious = ops.box_iou(bz_boxes, bz_gtboxs_abs_xyxy)
                cost_pose = self.cost_t * cost_t_pose + self.cost_r * cost_r_pose 
                
                fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
                    box_xyxy_to_cxcywh(bz_boxes),  
                    box_xyxy_to_cxcywh(bz_gtboxs_abs_xyxy),  
                    expanded_strides=32
                )
                
                cost_detect = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou + 100.0 * (~is_in_boxes_and_center)
                cost = cost_detect + cost_pose 
                cost[~fg_mask] = cost[~fg_mask] + 10000.0 

                indices_batchi, matched_qidx = self.dynamic_k_matching(cost, pair_wise_ious, bz_gtboxs.shape[0]) 
                indices.append(indices_batchi)
                matched_ids.append(matched_qidx)
                
        return indices, matched_ids

    def get_in_boxes_info(self, boxes, target_gts, expanded_strides): 
        xy_target_gts = box_cxcywh_to_xyxy(target_gts)  

        anchor_center_x = boxes[:, 0].unsqueeze(1)
        anchor_center_y = boxes[:, 1].unsqueeze(1)
        

        b_l = anchor_center_x > xy_target_gts[:, 0].unsqueeze(0) 
        b_r = anchor_center_x < xy_target_gts[:, 2].unsqueeze(0)
        b_t = anchor_center_y > xy_target_gts[:, 1].unsqueeze(0)
        b_b = anchor_center_y < xy_target_gts[:, 3].unsqueeze(0)
        
        is_in_boxes = ((b_l.long() + b_r.long() + b_t.long() + b_b.long() 
                        ) == 4)  
        is_in_boxes_all = is_in_boxes.sum(1) > 0  
        # in fixed center
        center_radius = 2.5
        # Modified to self-adapted sampling --- the center size depends on the size of the gt boxes
        # target_gts[:, 0/1]是真值框的中心点xy坐标；(xy_target_gts[:, 2/3] - xy_target_gts[:, 0/1])得到的是真值框的wh
        # https://github.com/dulucas/UVO_Challenge/blob/main/Track1/detection/mmdet/core/bbox/assigners/rpn_sim_ota_assigner.py#L212
        b_l = anchor_center_x > (target_gts[:, 0] - (center_radius * (xy_target_gts[:, 2] - xy_target_gts[:, 0]))).unsqueeze(0)
        b_r = anchor_center_x < (target_gts[:, 0] + (center_radius * (xy_target_gts[:, 2] - xy_target_gts[:, 0]))).unsqueeze(0)
        b_t = anchor_center_y > (target_gts[:, 1] - (center_radius * (xy_target_gts[:, 3] - xy_target_gts[:, 1]))).unsqueeze(0)
        b_b = anchor_center_y < (target_gts[:, 1] + (center_radius * (xy_target_gts[:, 3] - xy_target_gts[:, 1]))).unsqueeze(0)
        # c_l = proj_center_x > (target_gts[:, 0] - (center_radius * (xy_target_gts[:, 2] - xy_target_gts[:, 0]))).unsqueeze(0)
        # c_r = proj_center_x < (target_gts[:, 0] + (center_radius * (xy_target_gts[:, 2] - xy_target_gts[:, 0]))).unsqueeze(0)
        # c_t = proj_center_y > (target_gts[:, 1] - (center_radius * (xy_target_gts[:, 3] - xy_target_gts[:, 1]))).unsqueeze(0)
        # c_b = proj_center_y < (target_gts[:, 1] + (center_radius * (xy_target_gts[:, 3] - xy_target_gts[:, 1]))).unsqueeze(0)

        is_in_centers = ((b_l.long() + b_r.long() + b_t.long() + b_b.long()
                        # + c_l.long() + c_r.long() + c_t.long() + c_b.long()
                          ) == 4)
        is_in_centers_all = is_in_centers.sum(1) > 0

        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all 
        is_in_boxes_and_center = (is_in_boxes & is_in_centers) 

        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, num_gt): 
        matching_matrix = torch.zeros_like(cost)  
        ious_in_boxes_matrix = pair_wise_ious 
        n_candidate_k = 50

        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=0)  
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)

        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)  
            matching_matrix[:, gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(1) 

        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[anchor_matching_gt > 1], dim=1)
            matching_matrix[anchor_matching_gt > 1] *= 0
            matching_matrix[anchor_matching_gt > 1, cost_argmin,] = 1

        while (matching_matrix.sum(0) == 0).any():
            num_zero_gt = (matching_matrix.sum(0) == 0).sum()
            matched_query_id = matching_matrix.sum(1) > 0
            cost[matched_query_id] += 100000.0
            unmatch_id = torch.nonzero(matching_matrix.sum(0) == 0, as_tuple=False).squeeze(1)
            for gt_idx in unmatch_id:
                pos_idx = torch.argmin(cost[:, gt_idx])
                matching_matrix[:, gt_idx][pos_idx] = 1.0
            if (matching_matrix.sum(1) > 1).sum() > 0:  # If a query matches more than one gt
                _, cost_argmin = torch.min(cost[anchor_matching_gt > 1],
                                           dim=1)  # find gt for these queries with minimal cost
                matching_matrix[anchor_matching_gt > 1] *= 0  # reset mapping relationship
                matching_matrix[anchor_matching_gt > 1, cost_argmin,] = 1  # keep gt with minimal cost

        assert not (matching_matrix.sum(0) == 0).any()
        selected_query = matching_matrix.sum(1) > 0
        gt_indices = matching_matrix[selected_query].max(1)[1]
        assert selected_query.sum() == len(gt_indices)

        cost[matching_matrix == 0] = cost[matching_matrix == 0] + float('inf')
        matched_query_id = torch.min(cost, dim=0)[1]
        return (selected_query, gt_indices), matched_query_id
