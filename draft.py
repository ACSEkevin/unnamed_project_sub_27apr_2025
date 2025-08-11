from PIL import Image
from torch import nn
from torchvision.models import resnet50
from IPython.display import display, clear_output


import torch, requests, math, os
import matplotlib.pyplot as plt
import torch.nn.functional as F
import ipywidgets as widgets
import torchvision.transforms as T

# %config InlineBackend.figure_format = 'retina'

from datasets.coco import CLASSES
from models.backbone import build_backbone
from models.transformer import build_transformer
from models.detr import DETR
from main import get_args_parser


# torch.set_grad_enabled(False)


# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


class DETRGradCAM:
    def __init__(self, model: DETR, target_layer: nn.Module = None):
        self.model = model
        self.target_layer = target_layer if target_layer else model.transformer.decoder.layers[-1]
        self.gradients: torch.Tensor = None
        self.features: torch.Tensor = None
        
        # 注册钩子
        self.target_layer.register_forward_hook(self.save_features)
        self.target_layer.register_full_backward_hook(self.save_gradients)
    
    def save_features(self, module, input, output):
        self.features = output
    
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def __call__(self, input_image: torch.Tensor, obj_index: int = 0, target_class: int = None, filter_negatives: bool = True):
        # 前向传播
        outputs: dict[str, torch.Tensor] = self.model(input_image)

        logits = outputs['pred_logits']
        probas = logits.softmax(-1).squeeze()[..., :-1] # [B, N_instances, N_cls]
        keep = probas.max(-1).values > 0.85 # [N]
        # probas_kept = probas[keep]
        boxes_kept = outputs['pred_boxes'][0, keep]
        logits_kept = logits[0, keep] # [N, N_cls]
        print(f"fetched {logits_kept.size(0)} objects")

        if target_class:
            logits_kept = logits_kept[obj_index, target_class] # [1]
        else:
            logits_kept = logits_kept[obj_index].max() # [1]

        boxes_kept = boxes_kept[obj_index] # [4]

        ys = torch.concat([boxes_kept, logits_kept.view(1)]) # [5]
        contributes: dict[str, torch.Tensor] = dict(dxdq=None, dydq=None, dwdq=None, dhdq=None, dcdq=None)

        for _y, _key in zip(ys, contributes.keys()):
            self.model.zero_grad()
            _y.backward(retain_graph=True)

            grad = self.gradients.squeeze()[keep] #[N, 1, 256] -> [N, 256] -> [N_obj, 256]
            weights = grad[obj_index]
            if filter_negatives:
                F.relu(weights, True)
            contributes[_key] = weights
        
        return contributes


def build_detr_model(pretrained: bool = True) -> nn.Module:
    args = get_args_parser().parse_args()
    num_classes = 91

    backbone = build_backbone(args)
    transformer = build_transformer(args)
    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )

    if pretrained:
        param_to_resize = ["class_embed.weight", "class_embed.bias"]
        weight_path = "weights/detr-r50-joint-epoch260.pth" if num_classes == 1 else "./weights/detr-r50-e632da11.pth"
        state_dict = torch.load(weight_path, map_location="cpu")["model"]

        # if num_classes < len(CLASSES):
        #     for key in param_to_resize:
        #         state_dict[key] = state_dict[key][1: num_classes + 2] # [N_cls + 1, D]

        model.load_state_dict(
            state_dict
        )

    return model


def get_transforms():
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return transform


def box_cxcywh_to_xyxy(x: torch.Tensor):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox: torch.Tensor, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def plot_results(pil_img, prob, boxes, cls_list: list = None, cls: str = None, output_path: str = None):
    plt.figure(figsize=(12, 8))
    plt.imshow(pil_img)
    ax = plt.gca()

    if isinstance(prob, (float, int)):
        prob = torch.ones([len(boxes), 1])

    if cls is None:
        cls = ["N/A"] * len(boxes)

    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), color, c in zip(prob, boxes.tolist(), colors, cls):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=color, linewidth=2))
        cl = p.argmax()
        if cls_list is None:
            cls_list = CLASSES
        _c = cls_list[-1] if c == "N/A" else cls_list[c]

        text = f'{_c}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=10,
                bbox=dict(facecolor='yellow', alpha=0.3))
    plt.axis('off')
    plt.show()

    if output_path:
        plt.savefig(output_path, dpi=200)


def filter_predictions(outputs: dict[str, torch.Tensor], conf: float = .85):
    probas = outputs['pred_logits'].softmax(-1)[:, :, :-1] # [B, N_instances, N_cls + 1]
    keep = probas.max(-1).values > conf #

    return keep


def inference_single_image(model: nn.Module, im: Image.Image = None, conf: float = 0.85, device: str = "cpu"):
    if im is None:
        im = Image.open("./test_images/test_coco.jpg")

    print("image size: ", im.size)

    trans = get_transforms()
    img: torch.Tensor = trans(im).unsqueeze(0)
    img.to(device)
    model.to(device)

    outputs: dict[str, torch.Tensor] = model(img)
    probas = outputs['pred_logits'].softmax(-1).squeeze()[..., :-1] # [B, N_instances, N_cls]
    keep = probas.max(-1).values > conf # [N]
    probas_kept = probas[keep]
    boxes_kept = outputs['pred_boxes'][0, keep]

    print(probas.shape, probas_kept.shape)

    boxes_scaled = rescale_bboxes(boxes_kept, im.size)

    plot_results(im, probas_kept, boxes_scaled)


def vis_out_embed_contributions(model: DETR, im: Image = None, device: str = "cpu"):
    if im is None:
        im = Image.open("./test_images/test_coco.jpg")

    print("image size: ", im.size)

    trans = get_transforms()
    img: torch.Tensor = trans(im).unsqueeze(0)
    img = img.to(device)
    model.to(device)

    cam = DETRGradCAM(model)

    res = cam(img)

    fig, ax = plt.subplots(3, 2, figsize=(12, 6))  # 移除 polar 投影
    for index, (name, con) in enumerate(res.items()):
        row, col = index // 2, index % 2  # 修正索引计算
        # if row >= 2:  # 防止数据过多导致越界
        #     break
        
        # 显示热力图（推荐使用pcolormesh或seaborn热图）
        im = ax[row, col].pcolormesh(con.unsqueeze(0),
                        cmap="rainbow",
                        shading='auto')
        ax[row, col].set_title(name)
        ax[row, col].axis("off")

        fig.colorbar(im, ax=ax[row, col], shrink=0.6)

    plt.tight_layout()
    plt.axis("off")
    plt.show()


class AttentionVisualizer:
    def __init__(self, model: nn.Module, transform: T.Compose):
        self.model = model
        self.transform = transform

        self.url = ""
        self.cur_url = None
        self.pil_img = None
        self.tensor_img = None

        self.conv_features = None
        self.enc_attn_weights = None
        self.dec_self_attn_weights = None
        self.dec_attn_weights = None

        self.setup_widgets()

    def setup_widgets(self):
        self.sliders = [
            widgets.Text(
                value="./test_images/MOT17-09-DPM_000105.jpg",
                # value='http://images.cocodataset.org/val2017/000000039769.jpg',
                placeholder='Type something',
                description='URL (ENTER):',
                disabled=False,
                continuous_update=False,
                layout=widgets.Layout(width='100%')
            ),
            widgets.FloatSlider(min=0, max=0.99,
                        step=0.02, description='X coordinate', value=0.85,
                        continuous_update=False,
                        layout=widgets.Layout(width='50%')
                        ),
            widgets.FloatSlider(min=0, max=0.99,
                        step=0.02, description='Y coordinate', value=0.4,
                        continuous_update=False,
                        layout=widgets.Layout(width='50%')),
            widgets.Checkbox(
              value=False,
              description='Direction of self attention',
              disabled=False,
              indent=False,
              layout=widgets.Layout(width='50%'),
          ),
            widgets.Checkbox(
              value=True,
              description='Show red dot in attention',
              disabled=False,
              indent=False,
              layout=widgets.Layout(width='50%'),
          )
        ]
        self.o = widgets.Output()

    def compute_features(self, img):
        model = self.model
        # use lists to store the outputs via up-values
        conv_features, enc_attn_weights, dec_self_attn_weights, dec_attn_weights = [], [], [], []

        hooks = [
            model.backbone[-2].register_forward_hook(
                lambda self, input, output: conv_features.append(output)
            ),
            model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
                lambda self, input, output: enc_attn_weights.append(output[1])
            ),
            model.transformer.decoder.layers[-1].self_attn.register_forward_hook(
                lambda self, input, output: dec_self_attn_weights.append(output[1])
            ),
            model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
                lambda self, input, output: dec_attn_weights.append(output[1])
            ),
        ]
        # propagate through the model
        outputs = model(img)

        for hook in hooks:
            hook.remove()

        # don't need the list anymore
        self.conv_features = conv_features[0]
        self.dec_self_attn_weights = dec_self_attn_weights[0] # [1, N, N]
        self.dec_attn_weights = dec_attn_weights[0] # [1, N, HW]
        # get the HxW shape of the feature maps of the CNN
        shape = self.conv_features['0'].tensors.shape[-2:] # [H, W]
        # and reshape the self-attention to a more interpretable shape
        self.enc_attn_weights = enc_attn_weights[0].reshape(shape + shape) # [1, HW, HW] -> [H, W, H, W]

    def compute_on_image(self, url):
        if url != self.url:
            self.url = url
            self.pil_img = Image.open(url) #FIXME
            # mean-std normalize the input image (batch-size: 1)
            self.tensor_img = self.transform(self.pil_img).unsqueeze(0)
            self.compute_features(self.tensor_img)
    
    def update_chart(self, change):
        with self.o:
            clear_output()

            # j and i are the x and y coordinates of where to look at
            # sattn_dir is which direction to consider in the self-attention matrix
            # sattn_dot displays a red dot or not in the self-attention map
            url, j, i, sattn_dir, sattn_dot = [s.value for s in self.sliders]

            fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(9, 4))
            self.compute_on_image(url)

            # convert reference point to absolute coordinates
            j = int(j * self.tensor_img.shape[-1])
            i = int(i * self.tensor_img.shape[-2])

            # how much was the original image upsampled before feeding it to the model
            scale = self.pil_img.height / self.tensor_img.shape[-2]

            # compute the downsampling factor for the model
            # it should be 32 for standard DETR and 16 for DC5
            sattn = self.enc_attn_weights # [HW, HW] -> [H, W, H, W]
            fact = 2 ** round(math.log2(self.tensor_img.shape[-1] / sattn.shape[-1]))

            # round the position at the downsampling factor
            x = ((j // fact) + 0.5) * fact
            y = ((i // fact) + 0.5) * fact

            axs[0].imshow(self.pil_img)
            axs[0].axis('off')
            axs[0].add_patch(plt.Circle((x * scale, y * scale), fact // 4, color='r'))

            idx = (i // fact, j // fact)
            
            if sattn_dir:
                sattn_map = sattn[idx[0], idx[1], ...]
            else:
                sattn_map = sattn[..., idx[0], idx[1]]

            sattn_map = self._interpolate_attn_map(sattn_map, scale_factor=1)
            
            axs[1].imshow(sattn_map, cmap='cividis', interpolation='nearest')
            # if sattn_dot:
            #     axs[1].add_patch(plt.Circle((idx[1],idx[0]), 0.5, color='r'))
            axs[1].axis('off')
            axs[1].set_title(f'self-attention{(i, j)}')

            print("decoder self attn shape: ", self.dec_self_attn_weights.shape)
            axs[2].imshow(self.dec_self_attn_weights[0], cmap='cividis', interpolation='nearest')
            axs[2].axis('off')
            axs[2].set_title('query self-attn')

            plt.tight_layout()
            plt.show()

    @staticmethod
    def _interpolate_attn_map(attn_map: torch.Tensor, img_size: tuple[int, int] = None, scale_factor: float = None):
        attn_map = attn_map[None, None] # [1, 1, fH, fW]

        size = img_size[::-1] if img_size else None

        attn_map = F.interpolate(attn_map, size=size, scale_factor=scale_factor, mode="bilinear")

        return attn_map[0, 0]
    
    @torch.no_grad()
    def run(self):
      for s in self.sliders:
          s.observe(self.update_chart, 'value')
      self.update_chart(None)
      url, x, y, d, sattn_d = self.sliders
      res = widgets.VBox(
      [
          url,
          widgets.HBox([x, y]),
          widgets.HBox([d, sattn_d]),
          self.o
      ])
      return res



def test_build_coco_mot17():
    from datasets.mot17det import MOT17DetectionCOCOBuilder

    for mode in ["train", "val"]:
        path = "/Users/kevin/datasets/MOT17Labels/"
        to = "./coco"
        builder = MOT17DetectionCOCOBuilder(path).build(to, "val")
        print(len(builder))


def test_load_coco_mot17():
    from datasets.mot17det import MOT17SeqDataset

    folder = "./coco"
    mode = "train"
    data = MOT17SeqDataset(os.path.join(folder, f"{mode}2017"), 
                           os.path.join(folder, "annotations", f"instances_{mode}2017.json"),
                           num_frames=4)
    
    print(len(data))


def test_build_coco_crowdhuman():
    from datasets.crowdhuman import JointCOCOBuilder

    mode = "val"
    mot_path = "/Users/kevin/datasets/MOT17Labels/"
    ch_path = f"/Users/kevin/datasets/CrowdHuman/annotation_{mode}.odgt"
    img_folder = ""
    to = "./coco"
    # to = f"./crowdhuman_{mode}.json"

    builder = JointCOCOBuilder(mot_path, ch_path, img_folder)

    builder.build(to, "train")
    print(len(builder))
    builder.build(to, "val")
    print(len(builder))


def test_load_annos():
    from datasets.coco import CocoDetection, make_coco_transforms
    # from pycocotools.coco import COCO
    # from pycocotools.cocoeval import COCOeval

    import json

    img_folder = "coco/train2017"
    anno_path = "coco/annotations/instances_train2017.json"

    with open(anno_path, "r") as f:
        res = json.load(f)["annotations"]

    coco = CocoDetection(img_folder, anno_path, None, False)

    print(coco.ids[-15:], len(coco.ids))
    annos = coco._load_target(8657)

    print(len(annos), coco._load_target(8657))


def test_build_bdd100k():
    from datasets.bdd100k import BDD100kMOT20COCOBuilder

    mode = "val"
    path = f"/Users/kevin/datasets/BDD100kMOT20/labels/{mode}"
    partial_images = f"/Users/kevin/datasets/BDD100kMOT20/images/{mode}"

    builder = BDD100kMOT20COCOBuilder(partial_images, path, sample_interval=2, max_num_videos=5)

    builder.build("./coco", mode=mode)

    print(len(builder), len(builder.video_names))


def test_load_bdd100k():
    from datasets.bdd100k import BDD100kMOT20SeqDataset, bdd100k_mot20_collate_fn, make_bdd100k_mot20_transform
    from datasets.coco_eval import CocoEvaluator
    from torch.utils.data import DataLoader
    from models.associatr import build, init_from_pretrained_detr
    from torch.optim import AdamW
    from util.misc import MOTMetrics

    from models.matcher import build_matcher, build_clip_matcher

    args = get_args_parser().parse_args()
    num_frames = 1
    args.num_frames = num_frames

    model, crit, post = build(args)
    model = init_from_pretrained_detr(model, "./weights/detr-r50-e632da11.pth", skip_mismatch=True)

    matcher = build_matcher(args)
    clip_matcher = build_clip_matcher(args)

    optim = AdamW(model.parameters(), lr=5e-5)

    mode = "train"
    partial_images = f"/Users/kevin/datasets/BDD100kMOT20/images/{mode}"
    path = f"coco/annotations/instances_{mode}2017.json"

    data = BDD100kMOT20SeqDataset(partial_images, path, num_frames=num_frames, transforms=make_bdd100k_mot20_transform(mode))
    evaluator = CocoEvaluator(data.coco, ["bbox"])
    dataloader = DataLoader(data, batch_size=2, shuffle=True, num_workers=1, collate_fn=bdd100k_mot20_collate_fn)

    print("dataloader len: ", len(dataloader))
    for imgs, targets in dataloader:
        print(imgs.tensors.shape, len(targets))
        
        outputs = model(imgs)

        for key, val in outputs.items():
            if key == "aux_outputs":
                continue

            print(key, val.shape)

        matcher_res = matcher.forward(outputs, targets)
        clip_matcher_res = clip_matcher.forward(outputs, targets)

        print(matcher_res)
        print(clip_matcher_res)

        for _mres, _cmres in zip(matcher_res, clip_matcher_res):
            print(torch.eq(_mres[0], _cmres[0]), torch.eq(_mres[1], _cmres[1]))

        raise

        optim.zero_grad()
        loss_dict = crit(outputs, targets)
        weight_dict = crit.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        print(loss_dict)
        losses.backward()

        optim.step()

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = post["bbox"].forward(outputs, orig_target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        evaluator.update(res)

        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        evaluator.summarize()

        break

    # index = 166
    # print(len(data), len(data.valid_ids))
    # print(data[index][1])
    # imgs, targets = data[index] # [N], [N]

    # for index in range(len(imgs)):
    #     plot_results(
    #         imgs[index], 1., targets[index]["boxes"], cls_list=data._bdd_mot20_classes, cls=targets[index]["labels"]
    #     )


def build_associatr():

    a = torch.Tensor([4, 4, 6, 7, 7, 9, 2, 4, 3, 3, 6, 1])
    print(a.unique())

    print(torch.sort(a))



if __name__ == "__main__":
    # model = build_detr_model()
    # vis_out_embed_contributions(model)

    test_build_coco_mot17()
    test_load_coco_mot17()

    # test_build_bdd100k()
    # test_load_bdd100k()
    # build_associatr()