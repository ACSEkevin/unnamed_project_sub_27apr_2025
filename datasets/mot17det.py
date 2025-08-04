import os, json, shutil, configparser
import numpy as np

from typing import Literal

from .coco_builder import COCOBuilder


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

