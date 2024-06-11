#!/usr/bin/env python
# coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        """
        Initialize the IOUloss module.

        Args:
            reduction (str): The reduction method to apply to the output. Can be "none", "mean", or "sum".
            loss_type (str): The type of IOU loss to use. Can be "iou" or "giou".
        """

        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        """
        Compute the IOU or GIOU loss between predicted and target bounding boxes.

        Args:
            pred (torch.Tensor): The predicted bounding boxes with shape (N, 4).
            target (torch.Tensor): The target bounding boxes with shape (N, 4).

        Returns:
            torch.Tensor: The computed loss value.
        """

        assert pred.shape[0] == target.shape[0], "Predictions and targets must have the same batch size."

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)

        # Compute the coordinates of the top-left and bottom-right corners of the intersection
        tl = torch.max((pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2))
        br = torch.min((pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2))

        # Compute the areas of the predicted and target bounding boxes
        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        # Compute the intersection area
        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en

        # Compute the union area
        area_u = area_p + area_g - area_i

        # Compute the Intersection over Union (IOU)
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            # Compute the IOU loss
            loss = 1 - iou**2
        elif self.loss_type == "giou":
            # Compute the coordinates of the top-left and bottom-right corners of the smallest enclosing box
            c_tl = torch.min((pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2))
            c_br = torch.max((pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2))

            # Compute the area of the smallest enclosing box
            area_c = torch.prod(c_br - c_tl, 1)

            # Compute the Generalized Intersection over Union (GIOU)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)

            # Compute the GIOU loss
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class HumanDetectionLoss(nn.Module):
    def __init__(self, num_classes, strides=[8, 16, 32]):
        """
        Initialize HumanDetectionLoss module.

        Args:
            num_classes (int): Number of classes.
            strides (list): List of strides for different feature map levels.
        """

        super(HumanDetectionLoss, self).__init__()
        self.num_classes = num_classes
        self.strides = strides

        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")

    def forward(self, inputs, labels):
        """
        Forward pass of the HumanDetectionLoss module.

        Args:
            inputs (list): List of predicted outputs at different feature map levels.
                           Each element has shape [batch_size, 4 + 1 + num_classes, height, width].
            labels (list): List of ground truth labels for each batch.
                           Each element has shape [num_bboxes, 4 + 1 + num_classes].

        Returns:
            torch.Tensor: Computed loss value.
        """

        outputs = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for stride, output in zip(self.strides, inputs):
            output, grid = self.get_output_and_grid(output, stride)
            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
            expanded_strides.append(torch.ones_like(grid[:, :, 0]) * stride)
            outputs.append(output)
        concat_outputs = torch.cat(outputs, dim=1)
        loss = self.get_losses(x_shifts, y_shifts, expanded_strides, labels, concat_outputs)

        return loss

    def get_output_and_grid(self, output, stride):
        """
        Convert predicted output to actual ground truth values and generate grid.

        Args:
            output (torch.Tensor): Predicted output tensor.
            stride (int): Stride value for the current feature map level.

        Returns:
            tuple: Tuple containing the converted output and grid tensor.
        """

        height, width = output.shape[-2:]
        grid_y, grid_x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        grid = torch.stack((grid_x, grid_y), 2).view(1, height, width, 2).view(1, -1, 2)
        output = output.flatten(start_dim=2).permute(0, 2, 1)
        output[:, :, :2] = (output[:, :, :2] + grid.type_as(output)) * stride
        output[:, :, 2:4] = torch.exp(output[:, :, 2:4]) * stride

        return output, grid

    def get_losses(self, x_shifts, y_shifts, expanded_strides, labels, outputs):
        """
        Compute the losses for the predicted outputs.

        Args:
            x_shifts (list): List of x-coordinate shifts for each feature map level.
            y_shifts (list): List of y-coordinate shifts for each feature map level.
            expanded_strides (list): List of expanded strides for each feature map level.
            labels (list): List of ground truth labels for each batch.
            outputs (torch.Tensor): Concatenated predicted outputs from all feature map levels.

        Returns:
            torch.Tensor: Computed loss value.
        """

        bbox_preds = outputs[:, :, :4]
        obj_preds = outputs[:, :, 4:5]
        cls_preds = outputs[:, :, 5:]
        total_num_anchors = outputs.shape[1]

        x_shifts = torch.cat(x_shifts, dim=1).type_as(outputs)
        y_shifts = torch.cat(y_shifts, dim=1).type_as(outputs)
        expanded_strides = torch.cat(expanded_strides, dim=1).type_as(outputs)

        cls_targets = []
        reg_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        for batch_idx in range(outputs.shape[0]):
            num_gt = len(labels[batch_idx])

            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx][:, :4].type_as(outputs)
                gt_classes = labels[batch_idx][:, 4].type_as(outputs)
                bboxes_preds_per_image = bbox_preds[batch_idx]
                cls_preds_per_image = cls_preds[batch_idx]
                obj_preds_per_image = obj_preds[batch_idx]

                (
                    gt_matched_classes,
                    fg_mask,
                    pred_ious_this_matching,
                    matched_gt_inds,
                    num_fg_img,
                ) = self.get_assignments(
                    num_gt,
                    total_num_anchors,
                    gt_bboxes_per_image,
                    gt_classes,
                    bboxes_preds_per_image,
                    cls_preds_per_image,
                    obj_preds_per_image,
                    expanded_strides,
                    x_shifts,
                    y_shifts,
                )
                torch.cuda.empty_cache()
                num_fg += num_fg_img
                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ).float() * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.type(cls_target.type()))
            fg_masks.append(fg_mask)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)

        num_fg = max(num_fg, 1)
        loss_iou = (self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum()
        loss_obj = (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)).sum()
        loss_cls = (self.bcewithlog_loss(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets)).sum()
        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls

        return loss / num_fg

    @torch.no_grad()
    def get_assignments(
        self,
        num_gt,
        total_num_anchors,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        cls_preds_per_image,
        obj_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
    ):
        """
        Assign ground truth bounding boxes to anchors.

        Args:
            num_gt (int): Number of ground truth bounding boxes.
            total_num_anchors (int): Total number of anchors.
            gt_bboxes_per_image (torch.Tensor): Ground truth bounding boxes for the image.
            gt_classes (torch.Tensor): Ground truth classes for the bounding boxes.
            bboxes_preds_per_image (torch.Tensor): Predicted bounding boxes for the image.
            cls_preds_per_image (torch.Tensor): Predicted class probabilities for the image.
            obj_preds_per_image (torch.Tensor): Predicted objectness scores for the image.
            expanded_strides (torch.Tensor): Expanded strides for each anchor.
            x_shifts (torch.Tensor): X-coordinate shifts for each anchor.
            y_shifts (torch.Tensor): Y-coordinate shifts for each anchor.

        Returns:
            tuple: Tuple containing the assigned ground truth classes, fg mask, predicted IOUs,
                   matched ground truth indices, and number of foreground anchors.
        """

        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts, total_num_anchors, num_gt
        )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds = cls_preds_per_image[fg_mask]
        obj_preds = obj_preds_per_image[fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        pair_wise_ious = self.calc_bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image)
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        cls_preds = (
            cls_preds.float().unsqueeze(dim=0).repeat(num_gt, 1, 1).sigmoid()
            * obj_preds.unsqueeze(dim=0).repeat(num_gt, 1, 1).sigmoid()
        )
        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
            .unsqueeze(1)
            .repeat(1, num_in_boxes_anchor, 1)
        )
        pair_wise_cls_loss = F.binary_cross_entropy(cls_preds.sqrt(), gt_cls_per_image, reduction="none").sum(-1)
        del cls_preds

        cost = pair_wise_cls_loss + 3.0 * pair_wise_ious_loss + 100000.0 * (~is_in_boxes_and_center).float()

        num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds = self.dynamic_k_matching(
            cost, pair_wise_ious, gt_classes, num_gt, fg_mask
        )

        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        return gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg

    def calc_bboxes_iou(self, bboxes_a, bboxes_b):
        """
        Calculate the IoU between two sets of bounding boxes.

        Args:
            bboxes_a (torch.Tensor): First set of bounding boxes with shape [num_bboxes_a, 4].
            bboxes_b (torch.Tensor): Second set of bounding boxes with shape [num_bboxes_b, 4].

        Returns:
            torch.Tensor: IoU matrix with shape [num_bboxes_a, num_bboxes_b].
        """

        lt_bboxes_a = bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2
        lt_bboxes_b = bboxes_b[:, :2] - bboxes_b[:, 2:] / 2
        rb_bboxes_a = bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2
        rb_bboxes_b = bboxes_b[:, :2] + bboxes_b[:, 2:] / 2

        lt = torch.max(lt_bboxes_a, lt_bboxes_b)
        rb = torch.min(rb_bboxes_a, rb_bboxes_b)

        area_a = torch.prod(bboxes_a[:, 2:], dim=1)
        area_b = torch.prod(bboxes_b[:, 2:], dim=1)

        en = (lt < rb).type(lt.type()).prod(dim=2)
        area_intersection = torch.prod(rb - lt, dim=2) * en

        iou = area_intersection / (area_a[:, None] + area_b - area_intersection)

        return iou

    def get_in_boxes_info(
        self, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts, total_num_anchors, num_gt, center_radius=2.5
    ):
        """
        Get information about which anchors are inside the ground truth bounding boxes.

        Args:
            gt_bboxes_per_image (torch.Tensor): Ground truth bounding boxes for the image.
            expanded_strides (torch.Tensor): Expanded strides for each anchor.
            x_shifts (torch.Tensor): X-coordinate shifts for each anchor.
            y_shifts (torch.Tensor): Y-coordinate shifts for each anchor.
            total_num_anchors (int): Total number of anchors.
            num_gt (int): Number of ground truth bounding boxes.
            center_radius (float): Radius for determining if an anchor is near the center of a ground truth box.

        Returns:
            tuple: Tuple containing the fg mask and is_in_boxes_and_center mask.
        """

        expanded_strides_per_image = expanded_strides[0]

        x_centers_per_image = ((x_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(dim=0).repeat(num_gt, 1)
        y_centers_per_image = ((y_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(dim=0).repeat(num_gt, 1)

        gt_bboxes_per_image_ltx = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(dim=1).repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_rbx = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(dim=1).repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_lty = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(dim=1).repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_rby = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(dim=1).repeat(1, total_num_anchors)
        )

        b_ltx = x_centers_per_image - gt_bboxes_per_image_ltx
        b_rbx = gt_bboxes_per_image_rbx - x_centers_per_image
        b_lty = y_centers_per_image - gt_bboxes_per_image_lty
        b_rby = gt_bboxes_per_image_rby - y_centers_per_image
        bbox_deltas = torch.stack([b_ltx, b_lty, b_rbx, b_rby], dim=2)

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0

        gt_bboxes_per_image_ltx = (gt_bboxes_per_image[:, 0]).unsqueeze(dim=1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(dim=0)
        gt_bboxes_per_image_rbx = (gt_bboxes_per_image[:, 0]).unsqueeze(dim=1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(dim=0)
        gt_bboxes_per_image_lty = (gt_bboxes_per_image[:, 1]).unsqueeze(dim=1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(dim=0)
        gt_bboxes_per_image_rby = (gt_bboxes_per_image[:, 1]).unsqueeze(dim=1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(dim=0)

        c_ltx = x_centers_per_image - gt_bboxes_per_image_ltx
        c_rbx = gt_bboxes_per_image_rbx - x_centers_per_image
        c_lty = y_centers_per_image - gt_bboxes_per_image_lty
        c_rby = gt_bboxes_per_image_rby - y_centers_per_image
        center_deltas = torch.stack([c_ltx, c_lty, c_rbx, c_rby], dim=2)

        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all
        is_in_boxes_and_center = is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]

        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        """
        Perform dynamic k-matching between predicted bounding boxes and ground truth bounding boxes.

        Args:
            cost (torch.Tensor): Cost matrix for matching.
            pair_wise_ious (torch.Tensor): IoU matrix between predicted and ground truth bounding boxes.
            gt_classes (torch.Tensor): Ground truth classes for the bounding boxes.
            num_gt (int): Number of ground truth bounding boxes.
            fg_mask (torch.Tensor): Foreground mask indicating which anchors are inside the ground truth boxes.

        Returns:
            tuple: Tuple containing the number of foreground anchors, matched ground truth classes,
                   predicted IoUs for the matching, and matched ground truth indices.
        """

        matching_matrix = torch.zeros_like(cost)

        if num_gt == 0:
            fg_mask_inboxes = torch.zeros_like(fg_mask, dtype=torch.bool)
            num_fg = 0
            gt_matched_classes = torch.tensor([], dtype=gt_classes.dtype, device=gt_classes.device)
            matched_gt_inds = torch.tensor([], dtype=torch.long, device=matching_matrix.device)
            pred_ious_this_matching = torch.tensor([], dtype=torch.float, device=matching_matrix.device)
        else:
            n_candidate_k = min(10, pair_wise_ious.size(1))

            if n_candidate_k != 0:
                topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
                dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)

                for gt_idx in range(num_gt):
                    _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
                    matching_matrix[gt_idx][pos_idx] = 1.0

                del topk_ious, dynamic_ks, pos_idx
            else:
                pos_idx = None

                for gt_idx in range(num_gt):
                    if cost[gt_idx].numel() > 0:
                        _, pos_idx = torch.topk(cost[gt_idx], k=1, largest=False)
                        matching_matrix[gt_idx][pos_idx] = 1.0

                if pos_idx is not None:
                    del pos_idx

            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0

            fg_mask_inboxes = matching_matrix.sum(0) > 0.0
            num_fg = fg_mask_inboxes.sum().item()

            fg_mask[fg_mask.clone()] = fg_mask_inboxes

            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
            gt_matched_classes = gt_classes[matched_gt_inds]

            pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]

        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
