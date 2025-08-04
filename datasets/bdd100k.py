import os, json, shutil

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torchvision.datasets import CocoDetection
from typing import Callable, Any, Literal

from .coco_builder import COCOBuilder
from .coco import ConvertCocoPolysToMask
from util.misc import nested_tensor_from_tensor_list

import datasets.transforms as T


class BDD100kMOT20SeqDataset(CocoDetection):
    _bdd_mot20_classes = ['train', 'pedestrian', 'car', 'other vehicle', 'bus', 'trailer', 'truck', 'motorcycle', 'other person', 'bicycle', 'rider']
    def __init__(self, img_folder: str, anno_file: str, num_frames: int = 4, transforms: Callable = None) -> None:
        super().__init__(img_folder, anno_file)
        self.num_frames = num_frames
        self.video_to_img_ids, self.valid_ids = self._register_videos()
        self.prepare = ConvertCocoPolysToMask(False)

        self._trans = transforms

    def __len__(self) -> int:
        assert len(self.valid_ids) % self.num_frames == 0
        return len(self.valid_ids) // self.num_frames
    
    def __getitem__(self, index: int):
        frame_ids = self.valid_ids[index * self.num_frames: (index + 1) * self.num_frames]

        imgs, targets = [], []
        for frame_id in frame_ids:
            img, target = self._load_single_frame(frame_id)
            imgs.append(img)
            targets.append(target)

        if self._trans:
            imgs, targets = self._trans(imgs, targets)

        return imgs, targets

    def _register_videos(self):
        _valid_ids: list[int] = []
        _videos: dict[str, list[int]] = {}

        for img in self.coco.dataset['images']:
            video_name = img["video_name"]

            if video_name not in _videos:
                _videos[video_name] = []
            _videos[video_name].append(img["id"])

        for video, ids in _videos.items():
            residual = len(ids) % self.num_frames
            if residual > 0:
                ids = ids[:-residual]

            _videos[video] = ids
            _valid_ids.extend(ids)

        return _videos, _valid_ids
    
    def _load_image(self, id: int) -> Image.Image: # override
        img_info = self.coco.loadImgs(id)[0]
        file_name = img_info["file_name"]
        video_name = img_info["video_name"]
        path = os.path.join(self.root, video_name, file_name)

        return Image.open(path)
    
    def _load_single_frame(self, frame_id: int) -> tuple[Image.Image, dict[str, Any]]:
        image = self._load_image(frame_id)
        # import numpy as np
        # image = Image.fromarray(np.random.randint(0, 256, size=[600, 600, 3], dtype=np.uint8))

        target = self._load_target(frame_id)
        target = {'image_id': frame_id, 'annotations': target}

        return self.prepare(image, target)
    

class BDD100kMOT20COCOBuilder(COCOBuilder):
    _bdd_mot20_classes = ['train', 'pedestrian', 'car', 'other vehicle', 'bus', 'trailer', 'truck', 'motorcycle', 'other person', 'bicycle', 'rider']
    def __init__(self, seq_root: str, anno_root: str, sample_interval: int = 1, max_num_videos: int = None) -> None:
        super().__init__(None, anno_root, sample_interval, 0.)
        self.seq_root = seq_root

        # based on video clips rather than annotations
        self.video_names = [name for name in os.listdir(self.seq_root) if not name == ".DS_Store"]
        if max_num_videos:
            self.video_names = self.video_names[:max_num_videos]

    def _build_data_as_coco_format(self):
        coco = self._get_coco_skeleton("BDD100k MOT20 COCO Format")
        image_id = 0
        obj_id = 0

        coco["categories"] = [
            {"id": index, "name": value} for index, value in enumerate(self._bdd_mot20_classes)
        ]

        for video in tqdm(self.video_names, ncols=90):
            anno_path = os.path.join(self.anno_root, video + ".json")
            with open(anno_path, "r") as f:
                annos = json.load(f)

            for anno in annos[::self.sample_interval]: # here start index is fixed to 0
                if not anno["labels"]:
                    continue

                img_path = os.path.join(self.seq_root, video, anno["name"])
                img = Image.open(img_path)
                width, height = img.size

                # FIXME: copy or move images
                # if self.rebuild_coco_path:
                #     dst_path = os.path.join(
                #         self.rebuild_coco_path, 
                #         f"{self.mode}2017", 
                #         anno["name"]
                #     )
                #     shutil.copy2(img_path, dst_path)

                coco["images"].append({
                    "id": image_id,
                    "file_name": anno["name"],
                    "video_name": video,
                    "height": height,
                    "width": width
                })

                for obj in anno["labels"]:
                    x1, x2, y1, y2 = list(obj["box2d"].values())
                    w, h = x2 - x1, y2 - y1
                    if w <= 0 or h <= 0:
                        continue

                    coco["annotations"].append({
                        "id": obj_id,
                        "image_id": image_id,
                        "category_id": self._bdd_mot20_classes.index(obj["category"]),
                        "bbox": [float(x1), float(y1), float(w), float(h)],
                        "area": float(w) * float(h),
                        "iscrowd": int(obj["attributes"]["crowd"]),
                        "track_id": obj["id"],
                    })

                    obj_id += 1
                image_id += 1

        return coco


def make_bdd100k_mot20_transform(mode: Literal["train", "val"]):
    normalize = T.MotCompose([
        T.MotToTensor(),
        T.MotNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    # scales = [480 - 32]

    if mode == 'train':
        color_transforms = []
        scale_transforms = [
            T.MotRandomHorizontalFlip(),
            T.MotRandomResize(scales, max_size=1333),
            normalize,
        ]

        return T.MotCompose(color_transforms + scale_transforms)

    if mode == 'val':
        return T.MotCompose([
            T.MotRandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {mode}')


def bdd100k_mot20_collate_fn(batch: list[tuple[list, list[dict]]]):
    batch_images, batch_targets = [], []
    for imgs, targets in batch:
        batch_images.extend(imgs)
        batch_targets.extend(targets)

    batch_images = nested_tensor_from_tensor_list(batch_images)

    return batch_images, batch_targets


def build(image_set: Literal["train", "val"], args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'

    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    if args.img_folder:
        img_folder = os.path.join(args.img_folder, image_set)
    dataset = BDD100kMOT20SeqDataset(img_folder, ann_file, args.num_frames, transforms=make_bdd100k_mot20_transform(image_set))

    return dataset, bdd100k_mot20_collate_fn

