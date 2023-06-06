import nibabel as nib
import numpy as np
from glob import glob
import os
import imageio
from tqdm import tqdm
from Code.utils.format_conversion import binary2edge



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



img_dir = "Dataset/tr_im.nii.gz"
mask_dir = "Dataset/tr_mask.nii.gz"
img = nib.load(img_dir)
mask = nib.load(mask_dir)
img_fdata = img.get_fdata()
mask_fdata = mask.get_fdata()

# 生成包含0-100的随机index tensor
index = np.random.randint(0, 100, size=(1, 100)).squeeze()
index_train = index[0:67]
index_test = index[67:100]
train_mask = mask_fdata[:, :, index_train]
train_img = img_fdata[:, :, index_train]
test_mask = mask_fdata[:, :, index_test]
test_img = img_fdata[:, :, index_test]

(_,_,z) = train_img.shape
for j in range(z):   #是z的图象序列
    slice = img_fdata[:, :, j] 
    # 裁剪
    # slice = slice[100:400, 200:430]
    # 筛选出肺部清晰的图片
    # if np.mean(slice) > -500 and np.mean(slice) < -250:
    # 将 slice 的数据范围映射到 [0, 255]
    slice = np.interp(slice, (slice.min(), slice.max()), (0, 255))
    # 转换为 uint8 类型
    slice = slice.astype(np.uint8)
    imageio.imwrite("Dataset/TrainingSet/LungInfection-Train/Doctor-label/Imgs/"+'{}.jpg'.format(index_train[j]), slice)

(_,_,z) = test_img.shape
for j in range(z):   #是z的图象序列
    slice = img_fdata[:, :, j] 
    slice = np.interp(slice, (slice.min(), slice.max()), (0, 255))
    slice = slice.astype(np.uint8)
    imageio.imwrite("Dataset/TestingSet/LungInfection-Test/Imgs/"+'{}.jpg'.format(index_test[j]), slice)

(_,_,z) = train_mask.shape
for j in range(z):   #是z的图象序列
    slice = mask_fdata[:, :, j] 
    slice = np.interp(slice, (slice.min(), slice.max()), (0, 255))
    # mask统一为0和255
    slice[slice > 20] = 255
    slice = slice.astype(np.uint8)
    imageio.imwrite("Dataset/TrainingSet/LungInfection-Train/Doctor-label/GT/"+'{}.png'.format(index_train[j]), slice)

(_,_,z) = test_mask.shape
for j in range(z):   #是z的图象序列
    slice = mask_fdata[:, :, j]
    slice = np.interp(slice, (slice.min(), slice.max()), (0, 255))
    slice[slice > 20] = 255
    slice = slice.astype(np.uint8)
    imageio.imwrite("Dataset/TestingSet/LungInfection-Test/GT/"+'{}_mask.png'.format(index_test[j]), slice)

for i in glob("Dataset/TrainingSet/LungInfection-Train/Doctor-label/GT/*.png"):
    edge = binary2edge(i)
    imageio.imwrite("Dataset/TrainingSet/LungInfection-Train/Doctor-label/Edge/" + i.split("/")[-1], edge)

# for i in glob("Dataset/TestingSet/LungInfection-Test/GT/*.png"):
#     edge = binary2edge(i)
#     imageio.imwrite("Dataset/TestingSet/LungInfection-Test/Edge/" + i.split("/")[-1], edge)