import os
import csv
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
import sys
sys.path.append("./") 
from read_dataset.read_rope3d_dataset import alpha2roty, box_3d_origin, load_denorm, matrix2angle 

from PIL import Image

# 功能：查看图片能否打开，有没有这个图片、是不是完好、没有损坏的图片
# 参数：图片路径
# 返回：True
def check_pic(path_pic):
    try:
        img = Image.open(path_pic)    # 如果图片不存在，报错FileNotFoundError
        img.load()    # 如果图片不完整，报错OSError: image file is truncated
        return True
    except (FileNotFoundError, OSError):
        # print('文件损坏')
        return False
    
name2nuscenceclass = {
    "car": "vehicle.car",
    "van": "vehicle.car",
    "truck": "vehicle.truck",
    "bus": "vehicle.bus.rigid",
    "cyclist": "vehicle.bicycle",
    "tricyclist": "vehicle.trailer",
    "motorcyclist": "vehicle.motorcycle",
    "pedestrian": "human.pedestrian.adult",
    "trafficcone": "movable_object.trafficcone",
}

def filter_damaged_label(rope3d_root, split='train'):
    # 对label中的box[xmin,ymin,xmax,ymax]真值筛选掉xmin>=xmax，ymin>=ymax的部分，然后删除label中该行
    if split == 'train':
        src_dir = os.path.join(rope3d_root, "training")
        img_path = ["training-image_2a", "training-image_2b", "training-image_2c", "training-image_2d"]
    else:
        src_dir = os.path.join(rope3d_root, "validation")
        img_path = ["validation-image_2"]
    label_path = os.path.join(src_dir, "label_2")
    denorm_path = os.path.join(src_dir, "denorm")
    split_txt = os.path.join(src_dir, "train.txt" if split=='train' else 'val.txt')  # 训练/测试 集的数据列表
    idx_list = [x.strip() for x in open(split_txt).readlines()]
    idx_list_valid = []
    for index in idx_list:
        for sub_img_path in img_path:
            img_file = os.path.join(rope3d_root, sub_img_path, index + ".jpg")
            if os.path.exists(img_file):
                idx_list_valid.append((sub_img_path, index))
                break
    lx_max = 0
    ly_max = 0
    lz_max = 0
    gt_nums=0 
    trans2cameras = []
    # gt_ratios = np.array()    
    # for idx in tqdm(range(len(idx_list_valid))):
    
    infos = list()
    ann_infos = list() 
    box2d_wrong_nums = 0
    lwh_wrong_nums = 0
    
    for idx in tqdm(range(len(idx_list_valid))):
        sub_img_path, index = idx_list_valid[idx]
        img_file = os.path.join(rope3d_root,sub_img_path, index + ".jpg")
        label_file = os.path.join(label_path, index + ".txt")
        denorm_file = os.path.join(denorm_path, index + ".txt")
        
        denorm = load_denorm(denorm_file)# 地面方程
        
        if check_pic(img_file) == False:
            print("This jpg is truncated" + img_file)
            continue
        
        fieldnames = ['type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'dh', 'dw',
                        'dl', 'lx', 'ly', 'lz', 'ry']
        
        with open(label_file, 'r') as csv_file:
            
            reader = csv.DictReader(csv_file, delimiter=' ', fieldnames=fieldnames)
            for line, row in enumerate(reader):
                if row["type"] in name2nuscenceclass.keys():
                    ann_info = dict()
                    name = name2nuscenceclass[row["type"]]
                    box2d = [float(row['xmin']), float(row['ymin']), float(row['xmax']), float(row['ymax'])]# 二维边界框
                    dim = [float(row['dl']), float(row['dw']), float(row['dh'])]# 三维物体的尺寸（长、宽、高）
                    if box2d[0] >= box2d[2] or box2d[1] >= box2d[3]:
                        box2d_wrong_nums += 1
                        break
                    if dim[0]==0 or dim[0]==0 or dim[0]==0:
                        lwh_wrong_nums += 1
                        break
                    # # 观察旋转角度的变化范围
                    # ry = float(row["ry"])
                    # alpha = float(row["alpha"])
                    # pos = np.array((float(row['lx']), float(row['ly']), float(row['lz'])), dtype=np.float32)# 原相机坐标系下三维物体的位置
                    # if alpha > np.pi:
                    #     alpha -= 2 * np.pi
                    #     ry = alpha2roty(alpha, pos)
                    # rotation_y = ry # 与可视化做相同处理
                    # location = [float(row['lx']), float(row['ly']), float(row['lz'])]
                    # ann_info["location"] = np.array(location)
                    # # lx: -59 ~ 62    trans： 31,21,100
                    # # ly: -42 ~ 6
                    # # lz: 7 ~ 198
                    # _, rotation_matrix = box_3d_origin(dim,location,rotation_y,denorm)
                    # ann_info["rotation"] = matrix2angle(rotation_matrix)
                    # # rx: 10 ~ 171   rotate: 90,40,90
                    # # ry: -80 ~ 80
                    # # rz: -180 ~ 180
                    
                    # ann_info["box2d"] = box2d
                    # ann_info["lwh"] = dim
                    
                    # # 求面积的max
                    # # area = (np.array(box2d)[2]-np.array(box2d)[0])*(np.array(box2d)[3]-np.array(box2d)[1]) # [4,]
                    # # ann_info["box2d_area"] = area
                    # # a = np.array([b["box2d_area"] for b in ann_infos])  # max:600000;min:28.5;mean=6942;
                    # # import matplotlib.pyplot as plt
                    # # b=a[a<=100000]
                    # # 求数据分布：a[a<=20000].shape[0]/a.shape[0]
                    # # plt.scatter(range(len(b)), b)
                    # # plt.show()
                    
                    
                    # ann_infos.append(ann_info)
                    # infos.append(ann_infos)
                    
                    
                    # 求x、y、z的max
                    # lx,ly,lz = abs(float(row['lx'])),abs(float(row['ly'])),abs(float(row['lz']))
                    # lx_max = max(lx_max,lx) # 61.23
                    # ly_max = max(ly_max,ly) # 42.0 
                    # lz_max = max(lz_max,lz) # 197.7
                    # trans2camera = np.sqrt(lx**2+ly**2+lz**2)
                    # trans2cameras.append(trans2camera) # max:200;min:9.2;mean=66.8;ratio:a[a>70].shape[0]/a.shape[0]
                    # gt_nums+=1
            # if len(infos)==reader.line_num:
            #     gt_nums+=1 
            #     gt_ratios.append(len(infos) / reader.line_num)
            # else:
            #     gt_ratios.append(len(infos) / reader.line_num)

    print(" split: ", split)
    print("box2d_wrong_nums：", box2d_wrong_nums)
    print("lwh_wrong_nums：", lwh_wrong_nums)
    
            
    # 观察面积的经变换后的曲线
    # area_view = np.array([b["box2d_area"] for b in ann_infos])
    # y=np.exp(1 - area_view/30000)
    # import matplotlib.pyplot as plt
    # plt.scatter(area_view, y, label='e^(1-x/2000)')
    # plt.show()
            
    # 观察目标物体到相机的距离的分布情况
    # a=np.array([trans2cameras])[0]
    # a_sin=np.sin(a/400* np.pi)+1
    # import matplotlib.pyplot as plt
    # plt.scatter(a, a_sin, label='sin(x)', color='red', linestyle='-', linewidth=2)
    # plt.legend()
    # plt.title('Sine and Cosine Functions')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.show()
    # print("gt_ratios:", np.mean(gt_ratios)) 
    
                    
                    
if __name__ == "__main__":
    
    rope3d_root = "F:\Rope3D"
    filter_damaged_label(rope3d_root, split='train')
    filter_damaged_label(rope3d_root, split='val')
    

    
    