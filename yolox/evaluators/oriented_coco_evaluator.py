#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import itertools
import time
from collections import ChainMap, defaultdict

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from yolox.utils import (
    gather,
    is_main_process,
    oriented_postprocess,
    rotated_iou,
    synchronize,
    time_synchronized,
)


def _annotation_box(annotation):
    """Read oriented geometry prepared by the strict dataset loader."""
    if "_oriented_bbox" in annotation:
        return np.asarray(annotation["_oriented_bbox"][:5], dtype=np.float64)
    if "rbbox" in annotation:
        return np.asarray(annotation["rbbox"][:5], dtype=np.float64)
    raise ValueError("ground truth has no rbbox or prepared oriented geometry")


def _periodic_angle_error(predicted, target):
    return abs((float(predicted) - float(target) + 90.0) % 180.0 - 90.0)


def _angle_error_degrees(predicted_box, target_box):
    """Angle error accounting for equivalent width/height-swapped boxes."""
    predicted_box = np.asarray(predicted_box, dtype=np.float64)
    target_box = np.asarray(target_box, dtype=np.float64)
    same_axes = abs(np.log(predicted_box[2] / target_box[2])) + abs(
        np.log(predicted_box[3] / target_box[3])
    )
    swapped_axes = abs(np.log(predicted_box[2] / target_box[3])) + abs(
        np.log(predicted_box[3] / target_box[2])
    )
    target_angle = target_box[4] + (90.0 if swapped_axes < same_axes else 0.0)
    return _periodic_angle_error(predicted_box[4], target_angle)


def _interpolated_ap(recall, precision):
    recall_points = np.linspace(0.0, 1.0, 101)
    return float(np.mean([
        np.max(precision[recall >= point]) if np.any(recall >= point) else 0.0
        for point in recall_points
    ]))


class OrientedCOCOEvaluator:
    """COCO-style AP evaluator using exact OpenCV rotated-box IoU."""

    iou_thresholds = np.arange(0.50, 0.96, 0.05)

    def __init__(
        self,
        dataloader,
        img_size,
        confthre,
        nmsthre,
        num_classes,
        testdev=False,
        per_class_AP=True,
        class_agnostic=False,
    ):
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.testdev = testdev
        self.per_class_AP = per_class_AP
        self.class_agnostic = class_agnostic

    def evaluate(
        self, model, distributed=False, half=False, trt_file=None,
        decoder=None, test_size=None, return_outputs=False
    ):
        model = model.eval()
        if half:
            model = model.half()
        data_list = []
        output_data = defaultdict(dict)
        progress_bar = tqdm if is_main_process() else iter
        inference_time = 0.0
        nms_time = 0.0
        n_samples = max(len(self.dataloader) - 1, 1)

        if trt_file is not None:
            from torch2trt import TRTModule
            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))
            size = test_size or self.img_size
            warmup = torch.ones(1, 3, size[0], size[1]).cuda()
            model(warmup)
            model = model_trt

        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device("cpu")
        image_dtype = torch.float16 if half else torch.float32

        for current, (images, _, info_images, image_ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                images = images.to(device=model_device, dtype=image_dtype)
                record_time = current < len(self.dataloader) - 1
                if record_time:
                    start = time.time()
                outputs = model(images)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())
                if record_time:
                    inference_end = time_synchronized()
                    inference_time += inference_end - start
                outputs = oriented_postprocess(
                    outputs, self.num_classes, self.confthre, self.nmsthre,
                    class_agnostic=self.class_agnostic,
                )
                if record_time:
                    nms_end = time_synchronized()
                    nms_time += nms_end - inference_end

            detections, image_data = self.convert_to_coco_format(
                outputs, info_images, image_ids, return_outputs=True
            )
            data_list.extend(detections)
            output_data.update(image_data)

        statistics = torch.tensor(
            [inference_time, nms_time, n_samples],
            dtype=torch.float64,
            device=model_device,
        )
        if distributed:
            synchronize()
            data_list = list(itertools.chain(*gather(data_list, dst=0)))
            output_data = dict(ChainMap(*gather(output_data, dst=0)))
            torch.distributed.reduce(statistics, dst=0)

        results = self.evaluate_prediction(data_list, statistics.cpu())
        synchronize()
        return (results, output_data) if return_outputs else results

    def convert_to_coco_format(self, outputs, info_images, image_ids,
                               return_outputs=False):
        data_list = []
        image_wise_data = defaultdict(dict)
        for output, image_h, image_w, image_id in zip(
            outputs, info_images[0], info_images[1], image_ids
        ):
            if output is None:
                continue
            output = output.detach().cpu()
            scale = min(
                self.img_size[0] / float(image_h),
                self.img_size[1] / float(image_w),
            )
            boxes = output[:, :5].clone()
            boxes[:, :4] /= scale
            classes = output[:, 7].to(torch.int64)
            scores = output[:, 5] * output[:, 6]
            image_id = int(image_id)
            category_ids = [
                self.dataloader.dataset.class_ids[int(class_id)]
                for class_id in classes
            ]
            image_wise_data[image_id] = {
                "bboxes": boxes.tolist(),
                "scores": scores.tolist(),
                "categories": category_ids,
            }
            for box, score, category_id in zip(boxes, scores, category_ids):
                data_list.append({
                    "image_id": image_id,
                    "category_id": int(category_id),
                    "bbox": box.tolist(),
                    "score": float(score),
                })
        return (data_list, image_wise_data) if return_outputs else data_list

    def _ground_truth(self):
        coco = self.dataloader.dataset.coco
        ground_truth = defaultdict(list)
        for annotation in coco.dataset.get("annotations", []):
            if annotation.get("iscrowd", 0):
                continue
            ground_truth[
                (int(annotation["image_id"]), int(annotation["category_id"]))
            ].append(_annotation_box(annotation))
        return ground_truth

    @staticmethod
    def _evaluate_class(detections, ground_truth, category_id, threshold):
        class_detections = sorted(
            (d for d in detections if d["category_id"] == category_id),
            key=lambda d: (-d["score"], d["image_id"]),
        )
        class_gt = {
            key: boxes for key, boxes in ground_truth.items()
            if key[1] == category_id
        }
        gt_count = sum(len(boxes) for boxes in class_gt.values())
        if gt_count == 0:
            return None, []

        matched = {key: np.zeros(len(boxes), dtype=bool)
                   for key, boxes in class_gt.items()}
        true_positive = np.zeros(len(class_detections), dtype=np.float64)
        angle_errors = []
        for detection_index, detection in enumerate(class_detections):
            key = (detection["image_id"], category_id)
            candidates = class_gt.get(key, [])
            available = [
                (rotated_iou(detection["bbox"], box), index)
                for index, box in enumerate(candidates)
                if not matched[key][index]
            ]
            if not available:
                continue
            best_iou, best_index = max(available, key=lambda item: (item[0], -item[1]))
            if best_iou >= threshold:
                matched[key][best_index] = True
                true_positive[detection_index] = 1.0
                angle_errors.append(
                    _angle_error_degrees(detection["bbox"], candidates[best_index])
                )

        false_positive = 1.0 - true_positive
        true_positive = np.cumsum(true_positive)
        false_positive = np.cumsum(false_positive)
        recall = true_positive / gt_count
        precision = true_positive / np.maximum(
            true_positive + false_positive, np.finfo(np.float64).eps
        )
        precision = np.maximum.accumulate(precision[::-1])[::-1]
        return _interpolated_ap(recall, precision), angle_errors

    def evaluate_prediction(self, detections, statistics):
        if not is_main_process():
            return 0.0, 0.0, None

        inference_time, nms_time, samples = [float(value) for value in statistics]
        denominator = max(samples * self.dataloader.batch_size, 1.0)
        forward_ms = 1000.0 * inference_time / denominator
        nms_ms = 1000.0 * nms_time / denominator
        info = (
            "Average forward time: {:.2f} ms, Average NMS time: {:.2f} ms, "
            "Average inference time: {:.2f} ms\n"
        ).format(forward_ms, nms_ms, forward_ms + nms_ms)

        ground_truth = self._ground_truth()
        coco = self.dataloader.dataset.coco
        category_ids = sorted(self.dataloader.dataset.class_ids)
        names = {
            int(category["id"]): category["name"]
            for category in coco.dataset.get("categories", [])
        }
        class_aps = {}
        angle_errors = []
        ap_by_threshold = []
        for threshold in self.iou_thresholds:
            threshold_aps = []
            for category_id in category_ids:
                ap, errors = self._evaluate_class(
                    detections, ground_truth, category_id, float(threshold)
                )
                if ap is not None:
                    threshold_aps.append(ap)
                    if np.isclose(threshold, 0.5):
                        angle_errors.extend(errors)
            ap_by_threshold.append(
                float(np.mean(threshold_aps)) if threshold_aps else 0.0
            )

        for category_id in category_ids:
            values = []
            for threshold in self.iou_thresholds:
                ap, _ = self._evaluate_class(
                    detections, ground_truth, category_id, float(threshold)
                )
                if ap is not None:
                    values.append(ap)
            if values:
                class_aps[names.get(category_id, str(category_id))] = float(
                    np.mean(values)
                )

        ap50 = ap_by_threshold[0]
        ap50_95 = float(np.mean(ap_by_threshold))
        angle_mae = float(np.mean(angle_errors)) if angle_errors else float("nan")
        info += "Rotated AP50: {:.4f}, AP50:95: {:.4f}, matched angle MAE: {}\n".format(
            ap50, ap50_95,
            "{:.3f} deg".format(angle_mae) if np.isfinite(angle_mae) else "n/a",
        )
        if self.per_class_AP:
            info += "Per-class rotated AP50:95: " + ", ".join(
                "{}={:.4f}".format(name, ap)
                for name, ap in class_aps.items()
            ) + "\n"
        logger.info(info.rstrip())
        return ap50_95, ap50, info
