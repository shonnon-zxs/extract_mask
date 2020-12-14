import glob
import json
import os
import sys
from argparse import ArgumentParser

import cv2
import numpy as np
from scipy import ndimage
from tqdm import tqdm

# 提取特征
def do_extract(path):
    # path是所有jpg图片名
    annotation = annotations[os.path.basename(path)]
    bbox = annotation['bbox']
    # print(bbox)

    x, y, w, h = [int(x) for x in bbox]  # 获取bbox的x，y，w，h数据信息
    img = cv2.imread(path)  # 读取保存该路径的图片信息
    origin_height, origin_width = img.shape[:2]  # 获取图片的宽高

    # 将盒子box的边缘扩大5个像素
    box_pad = 5
    crop_x1 = x - box_pad
    crop_y1 = y - box_pad
    crop_x2 = x + w + box_pad
    crop_y2 = y + h + box_pad

    # 重新确定原始顶点坐标x, y
    x = x - crop_x1
    y = y - crop_y1

    # 得到对应区域的原始图像（得到矩形的小盒子，里面只包含一个检测物品的bbox）
    origin_img = img = img[crop_y1:crop_y2, crop_x1:crop_x2]

    # 在边检清新的情况下去除噪音，较其他方法，这种方法比较慢
    # 3是邻域直径，两个75分别是空间高斯函数标准差，灰度值相似性高斯函数标准差
    img = cv2.bilateralFilter(img, 3, 75, 75)

    # -------------------------
    # edge detect  边缘检测（对应第二张图）
    # -------------------------
    edges = detector.detectEdges(np.float32(img) / 255)
    # print(edges)

    # -------------------------
    # edge process  边缘处理  第二张图到第三张图的操作
    # -------------------------
    object_box_mask = np.zeros_like(edges, dtype=np.uint8)  # 产生一个维度和edges一样的全0数组
    object_box_mask[y:y + h, x:x + w] = 1  # 把关键mask部分设为1，
    edges[(1 - object_box_mask) == 1] = 0  # 把去掉mask的边缘部分都是0
    edges[(edges < (edges.mean() * 0.5)) & (edges < 0.1)] = 0  # 边界均值较小的都设置为0

    # -------------------------
    # erode and dilate  填补空洞，腐蚀与扩张 第三张图到第四张图的操作
    # -------------------------
    filled = ndimage.binary_fill_holes(edges).astype(np.uint8)
    filled = cv2.erode(filled, np.ones((32, 32), np.uint8))   # ones()里面的值越大腐蚀的越厉害，效果越平滑，越少毛刺
    filled = cv2.dilate(filled, np.ones((32, 32), np.uint8))  # 膨胀后的操作，使图像更清晰
    filled = cv2.erode(filled, np.ones((8, 8), np.uint8))
    filled = cv2.medianBlur(filled, 17) # 将17*17个数据从小到大排列，取中间值作为当前值
    # filled里面的值只有0和1

    # save_image里面似乎跟流程图无关------------------------------------------------------------------
    save_image = np.zeros((origin_height, origin_width), np.uint8)  # 定义一个大小为长*宽的全0矩阵（黑色图片）
    save_image[crop_y1:crop_y2, crop_x1:crop_x2] = np.array(filled * 255, dtype=np.uint8)  # save_image里面的只有0和255
    # 保存图像，将save_image保存到output_dir中
    cv2.imwrite(os.path.join(output_dir, os.path.basename(path).split('.')[0] + '.png'), save_image)
    # ----------------------------------------------------------------------------------------------

    # origin_img里面是包含一个检测物体的原始图像，filled中间全1，周围边缘全0，相乘就得到了想要的关键图像。
    # 如果要想得到相反图片，就全1变全0，全0边全1
    masked_img = origin_img * filled[:, :, None]
    # compare_img是把图像放正
    compare_img = np.concatenate([origin_img, masked_img], axis=1)  # 在1维度连接，1维度大小可以不同，但是其他维度必修相同
    # masked_img使提取的mask部分
    # 保存图片
    cv2.imwrite(os.path.join(compare_dir, os.path.basename(path)), compare_img)


def extract(paths):
    for path in tqdm(paths):  # 进度条
        do_extract(path)


if __name__ == '__main__':
    parser = ArgumentParser(description="Extract masks")  # 提取被遮住的部分
    parser.add_argument('--ann_file', type=str, default='instances_train2019.json')  # 标注文件
    parser.add_argument('--images_dir', type=str, default='train2019')  # 图片文件
    parser.add_argument('--model_file', type=str, default='model.yml.gz')  # 模型文件
    args = parser.parse_args()

    with open(args.ann_file) as fid:
        data = json.load(fid)    # 加载标注文件
    images = {}
    for x in data['images']:  # data['images']指的是标注文件的图像id
        images[x['id']] = x   # image{}里面x[id]是图像id
    annotations = {}
    for x in data['annotations']:  # data['annotations']指的是标注的id
        annotations[images[x['image_id']]['file_name']] = x

    output_dir = 'extracted_masks/masks'  # 将提取的隐藏部分放入这个文件路径
    compare_dir = 'extracted_masks/masked_images'   # 将处理后的图片放到这个文件路径
    if not os.path.exists(output_dir):  # 如果这个路径不存在，就新建一个
        os.makedirs(output_dir)

    if not os.path.exists(compare_dir):  # 如果这个路径不存在，就新建一个
        os.makedirs(compare_dir)

    categories = [i + 1 for i in range(200)]  # categories: [1, 2, 3, , , 199, 200]
    paths = glob.glob(os.path.join(args.images_dir, '*.jpg'))  # 查找当前路径（image_dir）下的所有jpg文件，返回的是所有文件名而不是文件
    detector = cv2.ximgproc.createStructuredEdgeDetection(args.model_file) # 从第一张图到第二章图
    extract(paths)
