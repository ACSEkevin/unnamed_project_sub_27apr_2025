import os, json

from typing import Literal


class COCOBuilder:
    def __init__(self, img_root: str, anno_root: str, sample_interval: int = 1, val_ratio: float = 0.2) -> None:
        super().__init__()

        self.img_root = img_root
        self.anno_root = anno_root
        self.sample_interval = sample_interval
        self.mode = "train"
        self.rebuild_coco_path: str = None
        self.val_ratio = val_ratio
        self.label = None

    def __len__(self) -> int:
        return len(self.label["images"])
    
    def build(self, rebuild_coco_path: str = None, mode: Literal["train", "val"] = "train"):
        self.mode = mode
        self.rebuild_coco_path = rebuild_coco_path

        if rebuild_coco_path:
            self._build_coco_data_folder(rebuild_coco_path)
        
        self.label = self._build_data_as_coco_format()

        if rebuild_coco_path:
            _label_save_path = os.path.join(
                rebuild_coco_path, "annotations", f"instances_{self.mode}2017.json"
            )
            with open(_label_save_path, "w") as f:
                json.dump(self.label, f, indent=4)

        return self
        
    @staticmethod
    def _build_coco_data_folder(path: str):
        content_dirs = ["train2017", "val2017", "annotations"]
        if not os.path.exists(path):
            os.mkdir(path)

        for _dir in content_dirs:
            if not os.path.exists(os.path.join(path, _dir)):
                os.mkdir(os.path.join(path, _dir))

    @staticmethod
    def _get_coco_skeleton(desc: str = None):
        desc = "COCO Format" if not desc else desc
        return {
            "info": {"description": desc},
            "licenses": [{"name": "MIT"}],
            "images": [],
            "annotations": [],
            "categories": [{"id": 0, "name": "person", "supercategory": "human"}]
        }
    
    def _build_data_as_coco_format(self):
        return self._get_coco_skeleton()
    