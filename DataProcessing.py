import nibabel as nib
import numpy as np
from glob import glob
import os
import imageio
import warnings
from tqdm import tqdm

# warnings.filterwarnings('ignore')


# 读取文件名
data_dir = "Dataset/volumes/"
images_1 = sorted(glob(os.path.join(data_dir, "coronacases_org_*.nii.gz")))
images_2 = sorted(glob(os.path.join(data_dir, "radiopaedia*.nii.gz")))

t_num = 0

for i in tqdm(images_1):
    img = nib.load(i)  #读取nii
    img_fdata = img.get_fdata()
    i = i.split('/')[-1]  #去掉路径，只保留文件名
    i = i.replace('.nii.gz', '') #去掉nii的后缀名
    i = "Data-1_" + i
    img_f_path = os.path.join('Dataset/TrainingSet/LungInfection-Train/Pseudo-label/Imgs', i)

    #开始转换图像
    (x,y,z) = img.shape
    num=0
    for j in range(z):   #是z的图象序列
        if j>= int(z * 0.8):
            break
        slice = img_fdata[:, :, j] 
        # 裁剪
        slice = slice[100:400, 200:430]
        # 筛选出肺部清晰的图片
        if np.mean(slice) > -500 and np.mean(slice) < -250:
            # 将 slice 的数据范围映射到 [0, 255]
            slice = np.interp(slice, (slice.min(), slice.max()), (0, 255))
            # 转换为 uint8 类型
            slice = slice.astype(np.uint8)
            imageio.imwrite(img_f_path+'_{}.jpg'.format(num + 1), slice)
            num += 1
            t_num += 1
    num = 0

for i in tqdm(images_2):
    img = nib.load(i)  #读取nii
    img_fdata = img.get_fdata()
    i = i.split('/')[-1]  #去掉路径，只保留文件名
    i = i.replace('.nii.gz', '') #去掉nii的后缀名
    i = "Data-1_" + i
    img_f_path = os.path.join('Dataset/TrainingSet/LungInfection-Train/Pseudo-label/Imgs', i)

    #开始转换图像
    (x,y,z) = img.shape
    num=0
    for j in range(z):   #是z的图象序列
        if t_num >= 1600:
            break
        if j>= int(z * 0.8):
            break
        slice = img_fdata[:, :, j] 
        # 裁剪
        slice = slice[50:-50, 150:500]
        # 筛选出肺部清晰的图片
        if np.mean(slice) > -500 and np.mean(slice) < 200:
            # 将 slice 的数据范围映射到 [0, 255]
            slice = np.interp(slice, (slice.min(), slice.max()), (0, 255))
            # 转换为 uint8 类型
            slice = slice.astype(np.uint8)
            imageio.imwrite(img_f_path+'_{}.jpg'.format(num + 1), slice)
            num += 1
            t_num += 1
    if t_num >= 1600:
        break
    num = 0

print("Num of Unabled Imgs:{}".format(t_num))