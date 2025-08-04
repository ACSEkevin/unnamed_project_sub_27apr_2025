import os, json, shutil

from PIL import Image
from tqdm import tqdm

from .mot17det import MOT17DetectionCOCOBuilder


class JointCOCOBuilder(MOT17DetectionCOCOBuilder):
    """
    Joint dataset of MOT17 and CrowdHuman
    """
    def __init__(self, mot_root: str, crowdhuman_anno: str, crowdhuman_img_dir: str, 
                 mot_sample_interval: int = 5, mot_val_ratio: float = 0.2, crowdhuman_val_ratio: float = 0.2) -> None:
        super().__init__(mot_root, mot_sample_interval, mot_val_ratio)

        self.crowdhuman_anno = crowdhuman_anno
        self.crowdhuman_img_dir = crowdhuman_img_dir
        self.crowdhuman_val_ratio = crowdhuman_val_ratio

    def _build_data_as_coco_format(self):
        coco = super()._build_data_as_coco_format()

        image_id = coco["images"][-1]["id"]
        annotation_id = coco["annotations"][-1]["id"]

        print("end image_id: ", image_id)
        print("end anno_id: ", annotation_id)

        # 读取.odgt文件
        with open(self.crowdhuman_anno, 'r') as f:
            lines = f.readlines()

        train_index = int(len(lines) * (1 - self.crowdhuman_val_ratio))
        lines = lines[:train_index] if self.mode == "train" else lines[train_index:]

        print(len(json.loads(lines[0].strip())['gtboxes']))
        for line in tqdm(lines, ncols=90):
            data = json.loads(line.strip())
            file_name = f"{data['ID']}.jpg"

            # 获取图像尺寸
            # FIXME: load and copy to train / val path
            image_path = os.path.join(self.crowdhuman_img_dir, file_name)
            dst_path = ""
            # img = Image.open(image_path)
            # width, height = img.size
            width, height = [1920, 1080]

            # FIXME: copy images
            if self.rebuild_coco_path:
                dst_path = os.path.join(
                    self.rebuild_coco_path, 
                    f"{self.mode}2017", 
                    file_name
                )
                shutil.copy2(image_path, dst_path)

            # 添加图像信息
            coco['images'].append({
                "id": image_id,
                "file_name": dst_path if dst_path else image_path,
                "height": height,
                "width": width
            })

            # 处理每个标注实例
            for box in data['gtboxes']:
                if box['tag'] == 'person':
                    # 使用Full BBox（全身边界框）
                    x1, y1, w, h = box['fbox']
                    area = w * h
                    
                    iscrowd = int(box['extra'].get('occ', 0) >= 0.9)

                    # 添加标注
                    coco['annotations'].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 0,
                        "bbox": [x1, y1, w, h],
                        "area": area,
                        "iscrowd": iscrowd
                    })
                    annotation_id += 1

            image_id += 1

        return coco


def convert_crowdhuman_to_coco(odgt_path: str, image_dir: str, output_json: str):
    # 初始化COCO结构
    coco = {
        "info": {"description": "CrowdHuman COCO Format"},
        "licenses": [{"name": "MIT"}],
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "person", "supercategory": "human"}]
    }

    annotation_id = 1

    # 读取.odgt文件
    with open(odgt_path, 'r') as f:
        lines = f.readlines()

    for line in tqdm(lines):
        data = json.loads(line.strip())
        image_id = int(data['ID'].split(',')[0])  # 提取图像ID
        file_name = f"{data['ID']}.jpg"

        # 获取图像尺寸
        # FIXME: load and copy to train / val path
        # img = Image.open(image_path)
        image_path = os.path.join(image_dir, file_name)
        # width, height = img.size
        width, height = [1920, 1080]

        # 添加图像信息
        coco['images'].append({
            "id": image_id,
            "file_name": file_name,
            "height": height,
            "width": width
        })

        # 处理每个标注实例
        for box in data['gtboxes']:
            if box['tag'] == 'person':
                # 使用Full BBox（全身边界框）
                x1, y1, x2, y2 = box['fbox']
                w, h = x2 - x1, y2 - y1
                area = w * h
                
                iscrowd = int(box['extra'].get('occ', 0) >= 0.9)

                # 添加标注
                coco['annotations'].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 0,
                    "bbox": [x1, y1, w, h],
                    "area": area,
                    "iscrowd": iscrowd
                })
                annotation_id += 1

    # 保存为JSON文件
    with open(output_json, 'w') as f:
        json.dump(coco, f, indent=2)
