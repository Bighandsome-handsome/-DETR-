# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path
from PIL import Image

import random
import numpy as np
import torch
import torch.utils.data
from pycocotools import mask as coco_mask

from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms as T


class CocoDetection(TvCocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetection, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target
    
def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks

class CocoDetectionWithCutMix(TvCocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetectionWithCutMix, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        image, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        image, target = self.prepare(image, target)

        cutmix_prob = 0.2
        copypaste_prob = 0.2
        r = random.random()

        if r < cutmix_prob:
            return self._cutmix(image, target)
        elif r < cutmix_prob + copypaste_prob:
            return self._copypaste(image, target)
        else:
            if self._transforms is not None:
                image, target = self._transforms(image, target)
            return image, target

    def _cutmix(self, image, target):
        mix_idx = random.randint(0, len(self.ids) - 1)
        mix_image, mix_target = super().__getitem__(mix_idx)
        mix_image_id = self.ids[mix_idx]
        mix_target = {'image_id': mix_image_id, 'annotations': mix_target}
        mix_image, mix_target = self.prepare(mix_image, mix_target)

        image_np = np.array(image)
        mix_image_np = np.array(mix_image)
        h, w, _ = image_np.shape
        cut_x = random.randint(int(w * 0.25), int(w * 0.75))
        cut_y = random.randint(int(h * 0.25), int(h * 0.75))
        image_np[cut_y:, cut_x:, :] = mix_image_np[cut_y:, cut_x:, :]
        image = Image.fromarray(image_np)

        def filter_boxes(boxes, area):
            x0, y0, x1, y1 = area
            keep = []
            for box in boxes:
                bx0, by0, bx1, by1 = box
                if bx1 > x0 and bx0 < x1 and by1 > y0 and by0 < y1:
                    keep.append(box)
            return keep

        boxes1 = target["boxes"]
        boxes2 = mix_target["boxes"]
        labels1 = target["labels"]
        labels2 = mix_target["labels"]
        cut_area = [cut_x, cut_y, w, h]
        boxes2_cut = filter_boxes(boxes2, cut_area)

        if len(boxes2_cut) > 0:
            # 确保每个 box 是 list 或 tuple，而不是 Tensor
            boxes2_array = np.array([box.tolist() if isinstance(box, torch.Tensor) else box for box in boxes2_cut], dtype=np.float32)
            boxes2_tensor = torch.from_numpy(boxes2_array)
            labels2_tensor = torch.zeros((boxes2_tensor.shape[0],), dtype=torch.int64)
        else:
            boxes2_tensor = torch.empty((0, 4), dtype=torch.float32)
            labels2_tensor = torch.empty((0,), dtype=torch.int64)

        target["boxes"] = torch.cat([boxes1, boxes2_tensor], dim=0)
        target["labels"] = torch.cat([labels1, labels2_tensor], dim=0)

        for field in ["area", "iscrowd"]:
            if field in target and field in mix_target:
                f1 = target[field]
                f2 = mix_target[field]
                if isinstance(f1, torch.Tensor) and isinstance(f2, torch.Tensor):
                    f2_cut = f2[:len(boxes2_tensor)]
                    target[field] = torch.cat([f1, f2_cut], dim=0)

        if self._transforms is not None:
            image, target = self._transforms(image, target)
        return image, target

    def _copypaste(self, image, target):
        mix_idx = random.randint(0, len(self.ids) - 1)
        mix_image, mix_target = super().__getitem__(mix_idx)
        mix_image_id = self.ids[mix_idx]
        mix_target = {'image_id': mix_image_id, 'annotations': mix_target}
        mix_image, mix_target = self.prepare(mix_image, mix_target)

        image_np = np.array(image)
        mix_np = np.array(mix_image)
        H, W, _ = image_np.shape

        new_boxes = []
        new_labels = []

        for box, label in zip(mix_target["boxes"], mix_target["labels"]):
            x0, y0, x1, y1 = map(int, box.tolist())
            w, h = x1 - x0, y1 - y0
            if w * h > 32 * 32 or w < 4 or h < 4:
                continue
            obj_crop = mix_np[y0:y1, x0:x1, :]
            new_x = random.randint(0, W - w)
            new_y = random.randint(0, H - h)
            image_np[new_y:new_y + h, new_x:new_x + w, :] = obj_crop
            new_boxes.append([new_x, new_y, new_x + w, new_y + h])
            new_labels.append(label)

        if new_boxes:
            new_boxes_tensor = torch.tensor(new_boxes, dtype=torch.float32)
            new_labels_tensor = torch.tensor(new_labels, dtype=torch.int64)
            target["boxes"] = torch.cat([target["boxes"], new_boxes_tensor], dim=0)
            target["labels"] = torch.cat([target["labels"], new_labels_tensor], dim=0)
            # ✅ 粘贴后同步其他字段（area, iscrowd, masks）
            for field in ["area", "iscrowd"]:
                if field in target and field in mix_target:
                    f1 = target[field]
                    f2 = mix_target[field]
                    if isinstance(f1, torch.Tensor) and isinstance(f2, torch.Tensor):
                        f2_cut = f2[:len(new_boxes)]
                        target[field] = torch.cat([f1, f2_cut], dim=0)

            if "masks" in target and "masks" in mix_target:
                f1 = target["masks"]
                f2 = mix_target["masks"]
                if isinstance(f1, torch.Tensor) and isinstance(f2, torch.Tensor):
                    f2_cut = f2[:len(new_boxes)]
                    target["masks"] = torch.cat([f1, f2_cut], dim=0)
        
        image = Image.fromarray(image_np)
        if self._transforms is not None:
            image, target = self._transforms(image, target)
        return image, target

    '''
    def __init__(self, img_folder, ann_file, transforms, return_masks, cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetectionWithCutMix, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
    '''    
    '''下面这个函数训练62轮，后面打算用新的数据处理方法
    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target
    '''
    '''
    def __getitem__(self, idx):
        image, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        image, target = self.prepare(image, target)
        
        cutmix_prob = 0.2  # ✅ 控制 CutMix 触发概率为 20%

        if random.random() > 0.5:
            # 不使用 CutMix，正常流程
            if self._transforms is not None:
                image, target = self._transforms(image, target)
            return image, target

        # 否则执行 CutMix
        mix_idx = random.randint(0, len(self.ids) - 1)
        mix_image, mix_target = super().__getitem__(mix_idx)
        mix_image_id = self.ids[mix_idx]
        mix_target = {'image_id': mix_image_id, 'annotations': mix_target}
        mix_image, mix_target = self.prepare(mix_image, mix_target)

        # 转为 numpy
        image_np = np.array(image)
        mix_image_np = np.array(mix_image)

        h, w, _ = image_np.shape
        cut_x = random.randint(int(w * 0.25), int(w * 0.75))
        cut_y = random.randint(int(h * 0.25), int(h * 0.75))

        image_np[cut_y:, cut_x:, :] = mix_image_np[cut_y:, cut_x:, :]
        image = Image.fromarray(image_np)

        # 合并 boxes
        def filter_boxes(boxes, area):
            x0, y0, x1, y1 = area
            keep = []
            for box in boxes:
                bx0, by0, bx1, by1 = box
                if bx1 > x0 and bx0 < x1 and by1 > y0 and by0 < y1:
                    keep.append(box)
            return keep

        boxes1 = target["boxes"]
        boxes2 = mix_target["boxes"]
        labels1 = target["labels"]
        labels2 = mix_target["labels"]

        cut_area = [cut_x, cut_y, w, h]
        boxes2_cut = filter_boxes(boxes2, cut_area)

        if len(boxes2_cut) > 0:
            # 先用 numpy 统一格式，再转为 Tensor
            boxes2_array = np.array(boxes2_cut, dtype=np.float32).reshape(-1, 4)
            boxes2_tensor = torch.from_numpy(boxes2_array)
            labels2_tensor = torch.zeros((boxes2_tensor.shape[0],), dtype=torch.int64)
        else:
            boxes2_tensor = torch.empty((0, 4), dtype=torch.float32)
            labels2_tensor = torch.empty((0,), dtype=torch.int64)

        boxes = torch.cat([boxes1, boxes2_tensor], dim=0)
        labels = torch.cat([labels1, labels2_tensor], dim=0)

        target["boxes"] = boxes
        target["labels"] = labels

        # ✅ 在这里插入字段同步代码
        for field in ["area", "iscrowd"]:
            if field in target and field in mix_target:
                f1 = target[field]
                f2 = mix_target[field]
                if isinstance(f1, torch.Tensor) and isinstance(f2, torch.Tensor):
                    f2_cut = f2[:len(boxes2_tensor)]  # 注意：这里用 boxes2_tensor 的长度更安全
                    target[field] = torch.cat([f1, f2_cut], dim=0)
                    
        if self._transforms is not None:
            image, target = self._transforms(image, target)

        return image, target
    '''


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)
        # print("标签分布：", torch.unique(torch.tensor(classes)))
        # test successfully
        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    scales = [608, 640, 672, 704, 736, 768, 800, 864, 896]

    '''80轮之后采用这段代码
    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])
    '''
    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([640, 704, 768]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])
        
    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    
    '''
    dataset = CocoDetection(
        img_folder, ann_file, 
        transforms=make_coco_transforms(image_set), 
        return_masks=args.masks,
        cache_mode=args.cache_mode, 
        local_rank=get_local_rank(), 
        local_size=get_local_size()
    )
    '''

    '''
    dataset = CocoDetection(
        img_folder, ann_file, 
        transforms=make_coco_transforms(image_set), 
        return_masks=args.masks,
        cache_mode=args.cache_mode, 
        local_rank=0,        # ✅ 强制单卡
        local_size=1         # ✅ 使用完整数据集
    )

    
    return dataset
    '''
    if image_set == "train":
        dataset = CocoDetectionWithCutMix(
            img_folder, ann_file,
            transforms=make_coco_transforms(image_set),
            return_masks=args.masks,
            cache_mode=args.cache_mode,
            local_rank=get_local_rank(),
            local_size=get_local_size()
        )
    else:
        dataset = CocoDetection(
            img_folder, ann_file,
            transforms=make_coco_transforms(image_set),
            return_masks=args.masks,
            cache_mode=args.cache_mode,
            local_rank=get_local_rank(),
            local_size=get_local_size()
        )
    return dataset
   
