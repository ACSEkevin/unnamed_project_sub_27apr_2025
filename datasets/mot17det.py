import os, json, shutil, configparser
import numpy as np

from pathlib import Path
from PIL import Image
from typing import Callable, Literal, Any
from torchvision.datasets import CocoDetection

from .coco_builder import COCOBuilder
from .coco import ConvertCocoPolysToMask
from util.misc import nested_tensor_from_tensor_list

import datasets.transforms as T



class MOT17SeqDataset(CocoDetection):
    _mot17_classes = ["person"]
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
        file_name = self.coco.loadImgs(id)[0]["file_name"]
        path = os.path.join(self.root, file_name)

        return Image.open(path)
    
    def _load_single_frame(self, frame_id: int) -> tuple[Image.Image, dict[str, Any]]:
        image = self._load_image(frame_id)

        target = self._load_target(frame_id)
        target = {'image_id': frame_id, 'annotations': target}

        return self.prepare(image, target)


class MOT17DetectionCOCOBuilder(COCOBuilder):
    def __init__(self, root: str, sample_interval: int = 1, val_ratio: float = 0.2) -> None:
        super().__init__(root, None, sample_interval, val_ratio)

    @staticmethod
    def _parse_mot_seq_ini_info(ini_path: str):
        config = configparser.ConfigParser()
        config.read(ini_path, encoding="utf-8-sig")

        try:
            seq_info = {
                'imDir': os.path.join(ini_path, config.get('Sequence', 'imDir')),
                'frameRate': config.getfloat('Sequence', 'frameRate'),
                'seqLength': config.getint('Sequence', 'seqLength'),
                'imWidth': config.getint('Sequence', 'imWidth'),
                'imHeight': config.getint('Sequence', 'imHeight'),
                'imExt': config.get('Sequence', 'imExt')
            }
        except Exception as exc:
            print("error occured at file `{}`:".format(ini_path))
            raise exc
        
        return config.get('Sequence', 'name'), seq_info
    
    def _build_data_as_coco_format(self):
        data_path = os.path.join(self.img_root, "train")
        video_list = sorted([_n for _n in os.listdir(data_path) if _n.endswith("DPM")])
        coco_labels = self._get_coco_skeleton("MOT17 COCO Format")

        image_paths: list[str] = []
        seq_length = 0
        obj_count = 1
        for v_name in video_list: # traverse videos
            v_path = os.path.join(data_path, v_name) # listdir: img1, gt, det, seqinfo.ini
            _, seq_info = self._parse_mot_seq_ini_info(os.path.join(v_path, "seqinfo.ini"))

            img_folder = os.path.join(v_path, "img1")
            anno_path = os.path.join(v_path, "gt/gt.txt")
            annos = np.loadtxt(anno_path, delimiter=",").astype(np.float32)
            # annos = annos[annos[:, 0].argsort()] # sort by frame FIXME: not necessary
            annos = annos[annos[:, 6] == 1] # drop inactives # [N, 9]

            # fixed interval sampling
            start_index = np.random.choice([i for i in range(self.sample_interval)])
            indices = [i for i in range(seq_info["seqLength"])][start_index::self.sample_interval]
            num_train_imgs = int(len(indices) * (1 - self.val_ratio))
            indices = indices[:num_train_imgs] if self.mode == "train" else indices[num_train_imgs:]
            
            # fill in images info
            for index in indices:
                # img_path = os.path.join(img_folder, img_name)
                frame_id = index + 1
                img_path = f"{img_folder}/{frame_id:06d}{seq_info['imExt']}"
                dst_path = ""
                if self.rebuild_coco_path:
                    dst_path = os.path.join(
                            self.rebuild_coco_path, 
                            f"{self.mode}2017", 
                            f"{seq_length + frame_id:06d}{seq_info['imExt']}"
                        )
                    shutil.copy2(img_path, dst_path)
                
                coco_labels["images"].append({
                    "id": seq_length + frame_id,
                    "file_name": dst_path if dst_path else img_path,
                    "video_name": v_name,
                    "height": seq_info["imHeight"],
                    "width": seq_info["imWidth"]
                })
                image_paths.append(dst_path)

                # fill in annotations
                frame_annos = annos[annos[:, 0] == frame_id]
                if frame_annos.shape[0] == 0:
                    raise Exception("err at video: {}, frame: {}".format(v_name, frame_id))
                
                for obj_idx, obj in enumerate(frame_annos):
                    frame_id, obj_id, x_min, y_min, w, h, _, _, vis = obj.tolist()
                    coco_labels["annotations"].append({
                        "id": obj_count,
                        "image_id": int(seq_length + frame_id),
                        "category_id": 0,  # only 1 class
                        # FIXME: might normalize box and area
                        "bbox": [float(x_min), float(y_min), float(w), float(h)],
                        "area": float(w) * float(h),
                        "iscrowd": 0,
                        "track_id": int(obj_id),
                    })
                    obj_count += 1

            seq_length += seq_info["seqLength"]

            # categories exist in skeleton, no need to load.
    
        return coco_labels
    

def make_mot17_transform(mode: Literal["train", "val"]):
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


def mot17_collate_fn(batch: list[tuple[list, list[dict]]]):
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
    dataset = MOT17SeqDataset(img_folder, ann_file, args.num_frames, transforms=make_mot17_transform(image_set))

    return dataset, mot17_collate_fn

