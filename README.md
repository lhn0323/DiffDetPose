## Getting Started

### Installation

The codebases are built on top of [Detectron2](https://github.com/facebookresearch/detectron2), [Sparse R-CNN](https://github.com/PeizeSun/SparseR-CNN), and [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch).
Thanks very much.

#### Requirements
- Python ≥ 3.9.0
- PyTorch ≥ 2.3.0 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  You can install them together at [pytorch.org](https://pytorch.org) to make sure of this
- OpenCV is optional and needed by demo and visualization
- mmcv-full ≥ 1.7.2

#### Steps
1. Install Detectron2 following https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md#installation.

2. Prepare datasets
please download the source data sets from https://thudair.baai.ac.cn/rope, and run read_dataset/read_rope3d_dataset.py, save as a .pkl forms of true value. Create a new data directory and save .pkl to it.

3. Prepare pretrain models
Create a new models directory and place the pre-training weights in it.

4. Train DiffusionDet
```
python train_net.py --num-gpus 8 \
    --config-file configs/diffdet.coco.res50.yaml
```

5. Evaluate DiffusionDet
```
python train_net.py --num-gpus 8 \
    --config-file configs/diffdetpose.rope3d.res50.yaml \
    --eval-only MODEL.WEIGHTS path/to/model.pth
```

### Inference Demo with Pre-trained Models
We provide a command line tool to run a simple demo following [Detectron2](https://github.com/facebookresearch/detectron2/tree/main/demo#detectron2-demo).

```bash
python demo.py --config-file configs/diffdetpose.rope3d.res50.yaml \
    --input image.jpg --opts MODEL.WEIGHTS diffdetpose_rope3d_res50.pth
```
