from numpy.ma.core import nomask
import pandas as pd
import torch
from pathlib import Path
import os
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import torchvision
# from datasets.transforms import resize
from diffusiondet.util.utils import get_camera_intrinsics, RMatrix_2_RQuaternion, RQuaternion_2_RMatrix, RMatrix_2_REuler, \
    REuler_2_RMatrix, get_all_path
from PIL import Image
import datasets_labelimg3d.transforms as T
from pytorch3d.io import load_obj, save_obj
from torchvision import transforms
from datasets_labelimg3d.configs import cfg, config

# Set the device
if torch.cuda.is_available():
    device = torch.device(config["device"])
else:
    device = torch.device("cpu")
    print("WARNING: CPU only when reading models, this will be slow!")


CLASSES = config["model"]["classes"]

all_class_num = {0:0, 1:0, 2:0, 3:0, 4:0} 

class SingleAnnotationParser:
    def __init__(self, img_path, annotation_path):
        annotation_data = json.loads(pd.read_json(annotation_path, orient="records").to_json())
        # with open(annotation_path, 'r+') as load_f:
        #     annotation_data = json.load(load_f)

        self.img_path = img_path
        self.orig_size = Image.open(img_path).size
        self.camera_matrix = get_camera_intrinsics(annotation_data["camera"]["fov"], self.orig_size)
        self.model_ids, self.class_names, self.bboxes_2d, self.bboxes_3d = [], [], [], []

        self.bboxes_3d_w = []

        self.R_matrix_c2o, self.R_quaternion_c2o, self.R_euler_c2o = [], [], []
        self.T_matrix_c2o = []
        self.class_ids = []
        self.model_num = annotation_data["model"]["num"]

        for i in range(int(annotation_data["model"]["num"])):
            all_class_num[annotation_data["model"][str(i)]["class"] - 1] += 1

            self.model_ids.append(annotation_data["model"][str(i)]["class"] - 1)
            self.class_ids.append(0)
            self.class_names.append(annotation_data["model"][str(i)]["class_name"])
            self.bboxes_2d.append(annotation_data["model"][str(i)]["2d_bbox"])
            self.bboxes_3d.append(annotation_data["model"][str(i)]["3d_bbox"])
            self.bboxes_3d_w.append(annotation_data["model"][str(i)]["3d_bbox_w"])
            self.T_matrix_c2o.append(annotation_data["model"][str(i)]["T_matrix_c2o"])

            self.R_matrix_c2o.append(annotation_data["model"][str(i)]["R_matrix_c2o"])
            self.R_quaternion_c2o.append(RMatrix_2_RQuaternion(
                np.array(annotation_data["model"][str(i)]["R_matrix_c2o"]).reshape(3, 3)).tolist())
            self.R_euler_c2o.append(
                RMatrix_2_REuler(np.array(annotation_data["model"][str(i)]["R_matrix_c2o"]).reshape(3, 3)).tolist())

    def __getitem__(self):
        img = Image.open(self.img_path)
        boxes = torch.tensor(self.bboxes_2d)
        if len(boxes) == 0:
            print(self.img_path)

        targets = {
            'bboxes_2d': torch.cat([(boxes[:, :2] + boxes[:, 2:]) / 2, boxes[:, 2:] - boxes[:, :2]], dim=1) if len(
                boxes) != 0 else torch.empty(0, 4),  # convert to cxcywh
            "bboxes_3d": torch.tensor(self.bboxes_3d) if len(boxes) != 0 else torch.empty(0, 8, 2),
            "bboxes_3d_w": torch.tensor(self.bboxes_3d_w) if len(boxes) != 0 else torch.empty(0, 8, 3),
            "model_ids": torch.tensor(self.model_ids).long() if len(boxes) != 0 else torch.empty(0).long(),
            "labels": torch.tensor(self.class_ids).long() if len(boxes) != 0 else torch.empty(0).long(),
            "T_matrix_c2o": torch.tensor(self.T_matrix_c2o) if len(boxes) != 0 else torch.empty(0, 3),
            "R_quaternion_c2o": torch.tensor(self.R_quaternion_c2o) if len(boxes) != 0 else torch.empty(0, 4),
            "R_euler_c2o": torch.tensor(self.R_euler_c2o) if len(boxes) != 0 else torch.empty(0, 3),
            'orig_size': torch.tensor(self.orig_size),
            "img_path": torch.tensor(int(self.img_path.split("/")[-1].split(".")[0]))
        }
        return img, targets


class ModelParser:
    def __init__(self, model_folder, model_name=config["model"]["model_name"]):
        self.model_path = []
        for model in model_name:
            self.model_path.append(os.path.join(model_folder, model))

    def __getitem__(self):
        model_target = []
        for p in self.model_path:
            verts, _, _ = load_obj(p)
            model_target.append(verts)

        return model_target


class Li3dDataset(Dataset):
    """
    the datasets Annotated using LabelImg3d
    """

    def __init__(self, img_folder, anno_folder, transform=None):
        self.img_folder = img_folder
        self.anno_folder = anno_folder
        self.transform = transform

        self.data = []
        self.lens = 0

        self.load_data()

    def load_data(self):
        # analysis files in DETRAC-Train-Annotations-MOT
        assert self.img_folder and self.anno_folder is not None
        all_img_path = get_all_path(self.img_folder)
        pbar = tqdm(all_img_path)
        for path in pbar:
            pbar.set_description(f'reading: {path}')
            an_path = os.path.join(self.anno_folder, path.split("/")[-1].split(".")[0] + ".json")
            self.data = self.data + [SingleAnnotationParser(path, an_path).__getitem__()]
            self.lens = self.lens + 1

    def __len__(self):
        return self.lens

    def __getitem__(self, item):
        assert item < self.__len__(), f'the item of dataset must less than length {self.__len__()}, but get {item}'
        img, targets = self.data[item]
        if self.transform is not None:
            img, targets = self.transform(img, targets)

        return img, targets

    def get_orig_data(self, item):
        assert item < self.__len__(), f'the item of dataset must less than length {self.__len__()}, but get {item}'
        img, targets = self.data[item]
        return img, targets


def make_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize(config["pixel_mean"], config["pixel_std"])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            # T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    # T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            # T.RandomResize([config["image_height"]]),
            T.RandomResize(scales, max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.dataset_path) / image_set
    assert root.exists(), f'provided labelImg3d scene folder path {root} does not exist'
    PATHS = {
        "train": (root / "images", root / "annotations"),
        "val": (root / "images", root / "annotations")
    }

    img_folder, ann_folder = PATHS[image_set]
    dataset = Li3dDataset(img_folder=img_folder,
                          anno_folder=ann_folder,
                          transform=make_transforms(image_set)
                          )
    return dataset


if __name__ == '__main__':
    
    anno_path = "./datasets/KITTI3D/val/annotations/000002.json"
    img_path = "./datasets/KITTI3D/val/images/000002.png"
    model_folder = "./datasets/KITTI3D/val/model"
    img_folder = "./datasets/KITTI3D/val/annotations"
    anno_folder = "./datasets/KITTI3D/val/images"
    # ann = SingleAnnotationParser(img_path, anno_path)
    # ann.__getitem__()
    # m = ModelParser(model_folder, config["model"]["model_name"])
    # m.__getitem__()
    config["dataset_path"] = "./datasets/KITTI3D"
    dataset = build("train", config)
    dataset.__getitem__(0)
    pass
