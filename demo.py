# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import mmcv
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data.build import build_detection_test_loader, build_detection_train_loader
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from diffusiondet.predictor import VisualizationDemo
from diffusiondet import DiffusionDetDatasetMapper, add_diffusiondet_config, DiffusionDetWithTTA
from diffusiondet.util.model_ema import add_model_ema_configs, may_build_model_ema, may_get_ema_checkpointer, EMAHook, \
    apply_model_ema_and_restore, EMADetectionCheckpointer
from read_dataset.read_rope3d_dataset import generate_info_rope3d

# constants
WINDOW_NAME = "Rope3D Pose Estimation"

'''
detectron2/data/datasets/builtin.py 包含所有数据集结构
detectron2/data/datasets/builtin_meta.py 包含每个数据集元数据
detectron2/data/transforms/transform_gen.py 包含基本的数据增强
detectron2/config/defaults.py 包含所有模型参数
'''
# args传参 -- cfg获取 -- VisualizationDemo模型建立 (元数据获取 -- \
#   DefaultPredictor预测模型 (元结构注册 -- 指定要评估 -- 元数据注册 -- 加载模型权重 -- 图像resize和BGR)) -- \
#   预测数据读入 -- VisualizationDemo预测结果及可视化 (DefaultPredictor调用模型和BGR图像预测 -- visualizer画图)
 
# MetadataCatalog 含有数据集的元数据, 比如COCO的类别；全局变量,禁止滥用.
# DatasetCatalog 保留了用于获取数据集的方法.
 
# 基于demo.py 写模型运行main函数
# 基于config/defaults.py 改所新模型包含的参数
# 基于.yaml 改defaults所包含参数的参数值
# 包含所有用到的函数的子函数 builtins.py
 
# 向默认.yaml里读入参数
def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()#获取包含大量参数的默认.yaml，cfg是一个CfgNode对象，为节点包含很多子节点和参数 
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    add_diffusiondet_config(cfg)#向上一步生成的默认.yaml cfg对象读入参数
    add_model_ema_configs(cfg)#向默认.yaml里读入参数
    cfg.merge_from_file(args.config_file)#覆盖默认.yaml里对应参数值 从给定的配置文件中加载内容，并将其合并到自身
    cfg.merge_from_list(args.opts)#读入相关参数设定到默认yaml里
    # Set score_threshold for builtin models设置模型负样本判定阈值
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()#冻结默认.yaml里的参数
    return cfg


def get_parser():# 用于解析命令行参数
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    #传入。yaml文件 ，对应各种参数配置
    parser.add_argument(
        "--config-file",
        default="configs/diffdet.coco.res50.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")#是否使用摄像头
    parser.add_argument("--video-input", help="Path to video file.")    #视频输入
    parser.add_argument(    #图片输入
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'"
    )
    parser.add_argument(    #输出的路径
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(    #置信度阈值
        "--confidence-threshold",
        type=float,
        default=0.5, # 0.5
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(# 其余没指定的参数
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser
# 参数列表过长：ulimit -s 65536

def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False

if __name__ == "__main__":
    
    DatasetCatalog.register("rope3d_train", lambda : mmcv.load("data/rope3d/rope3d_all_train.pkl"))
    MetadataCatalog.get("rope3d_train" )
    DatasetCatalog.register("rope3d_val", lambda : mmcv.load("data/rope3d/rope3d_all_val.pkl"))
    MetadataCatalog.get("rope3d_val" )
    
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    # args = Namespace(confidence_threshold=0.5, config_file='configs/diffdet.coco.res50.yaml', input=['./datasets/coco/train2017/*.jpg'], opts=['MODEL.WEIGHTS', './models/diffdet_coco_res50_300boxes.pth'], output=None, video_input=None, webcam=False)
    setup_logger(name="fvcore")# 日志记录
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)# 向默认.yaml里读入参数，获取参数配置

    demo = VisualizationDemo(cfg)
    
    mapper = DiffusionDetDatasetMapper(cfg, is_train=True)# 接收Detectron2数据集格式的数据集dict， 并将其映射为DiffusionDet使用的格式
    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST, mapper=mapper)
    # data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST, mapper=mapper, dataset=datapkl)
    # data_loader = build_detection_train_loader(cfg, mapper=mapper)
    # data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST, mapper=mapper)

    if args.input: 
        for per_data in tqdm.tqdm(data_loader.dataset, disable= args.output):
            
            img_path = ["validation-image_2", "training-image_2a", "training-image_2b", "training-image_2c", "training-image_2d"]
            for a in img_path:
                if os.path.exists(os.path.join("F:\Rope3D", a, per_data[0] + ".jpg")):
                    img_file = os.path.join("F:\Rope3D", a, per_data[0] + ".jpg")
                    break 
            # img_file = os.path.join("datasets/carla/training/image_2",  per_data[0] + ".png")    
            # if per_data[1] not in [25]: #(5/50) 8,16,25,43,38;
            #     continue
               
            img = read_image(img_file,  format="BGR") # 读取图片 img: img.shape = (480, 640, 3) img是numpy数组 
            start_time = time.time()# time.time()返回当前时间的时间戳
            predictions, visualized_output = demo.run_on_image(per_data, img) #demo.run_on_image(img)返回预测结果和可视化结果 demo.run_on_image(img)调用DefaultPredictor调用模型和BGR图像预测 -- visualizer画图
            logger.info(
                "{}: {} in {:.2f}s".format(# in {：.2f}s 表示小数点后保留两位
                    per_data[1],# 图片路径
                    "detected {} instances".format(len(predictions["instances"]))# len(predications) = 4 instances是预测结果的key，len(predictions["instances"])是预测结果的数量
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )# [04/06 19:27:42 detectron2]: ./datasets/coco/train2017/000000000009.jpg: detected 4 instances in 108785.20s

            if args.output:# 如果有输出路径
                if os.path.isdir(args.output):# 如果输出路径是一个文件夹
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))# args.output为输出路径 path为图片路径os.path.basename()返回文件名  
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)# 保存可视化结果
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)# 创建一个窗口namewindow cv2.WINDOW_NORMAL表示窗口大小可调整
                cv2.resizeWindow(WINDOW_NAME, 1920, 1088)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])# 显示图片 get_image()返回可视化结果的图片[:, :, ::-1]表示将BGR转为RGB
                # cv2.waitKey(0)
                if cv2.waitKey(0) == 27:# 等待按键，27为esc键
                    break  # esc to quit
    elif args.webcam:# 如果是摄像头
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:  # 如果是视频
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)
        codec, file_ext = (
            ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
        )
        if codec == ".mp4v":
            warnings.warn("x264 codec not available, switching to mp4v")
        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + file_ext
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input) 
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
