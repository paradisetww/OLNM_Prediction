# preprocessing
import monai
from monai.transforms import Compose, LoadImaged, Orientationd, Spacingd, ScaleIntensityRanged, SpatialPadd, RandCropByPosNegLabeld, SaveImaged
from monai.data import load_decathlon_datalist
from tqdm import tqdm
import os
import torch
import numpy as np

space_x = 1.0
space_y = 1.0
space_z = 1.0
a_min = -1600.0
a_max = 400.0
b_min = 0.0
b_max = 1.0
roi_x = 96
roi_y = 96
roi_z = 96
data_dir = "./datasets/image_mask_old/"
json_dir = "./datasets/"
json_list = "training_validation.json"
hd_json_list = "test_1.json"
zs_json_list = "test_2.json"

# Fixed random seed
torch.manual_seed(5)
torch.cuda.manual_seed(5)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
np.random.seed(5)

train_transforms = Compose(
    [
        LoadImaged(keys=["image", "mask"], ensure_channel_first=True),
        Orientationd(keys=["image", "mask"], axcodes="RAS"),
        Spacingd(keys=["image", "mask"], pixdim=(space_x, space_y, space_z), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, clip=True),
        SpatialPadd(keys=["image", "mask"], spatial_size=(roi_x, roi_y, roi_z), method='symmetric', mode='constant'),
        SaveImaged(keys=["image", "mask"], meta_keys=["image_meta_dict", "mask_meta_dict"], output_dir="./datasets/image_mask_new", output_postfix="", resample=False, data_root_dir="./datasets/image_mask_old", separate_folder=False, print_log=False),
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=["image", "mask"], ensure_channel_first=True),
        Orientationd(keys=["image", "mask"], axcodes="RAS"),
        Spacingd(keys=["image", "mask"], pixdim=(space_x, space_y, space_z), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, clip=True),
        SpatialPadd(keys=["image", "mask"], spatial_size=(roi_x, roi_y, roi_z), method='symmetric', mode='constant'),
        RandCropByPosNegLabeld(
            keys=["image", "mask"],
            label_key="mask",
            spatial_size=(roi_x, roi_y, roi_z),
            pos=1,
            neg=0,
            num_samples=1,
            image_key="image",
            image_threshold=0,
        ),
        SaveImaged(keys=["image", "mask"], meta_keys=["image_meta_dict", "mask_meta_dict"], output_dir="./datasets/image_mask_new", output_postfix="", resample=False, data_root_dir="./datasets/image_mask_old", separate_folder=False, print_log=False),
    ]
)

test_transforms = Compose(
    [
        LoadImaged(keys=["image", "mask"], ensure_channel_first=True),
        Orientationd(keys=["image", "mask"], axcodes="RAS"),
        Spacingd(keys=["image", "mask"], pixdim=(space_x, space_y, space_z), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, clip=True),
        SpatialPadd(keys=["image", "mask"], spatial_size=(roi_x, roi_y, roi_z), method='symmetric', mode='constant'),
        RandCropByPosNegLabeld(
            keys=["image", "mask"],
            label_key="mask",
            spatial_size=(roi_x, roi_y, roi_z),
            pos=1,
            neg=0,
            num_samples=1,
            image_key="image",
            image_threshold=0,
        ),
        SaveImaged(keys=["image", "mask"], meta_keys=["image_meta_dict", "mask_meta_dict"], output_dir="./datasets/image_mask_new", output_postfix="", resample=False, data_root_dir="./datasets/image_mask_old", separate_folder=False, print_log=False),
    ]
)

datalist_json = os.path.join(json_dir, json_list)
hd_datalist_json = os.path.join(json_dir, hd_json_list)
zs_datalist_json = os.path.join(json_dir, zs_json_list)

train_files = load_decathlon_datalist(datalist_json, False, "train", base_dir=data_dir)
train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=3)
print(f'There are {len(train_ds)} training samples!')  # 470
for sample in tqdm(train_ds):
    pass

val_files = load_decathlon_datalist(datalist_json, False, "validation", base_dir=data_dir)
val_ds = monai.data.CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=3)
print(f'There are {len(val_ds)} validation samples!')  # 202
for sample in tqdm(val_ds):
    pass

hd_test_files = load_decathlon_datalist(hd_datalist_json, False, "test", base_dir=data_dir)
hd_test_ds = monai.data.CacheDataset(data=hd_test_files, transform=test_transforms, cache_rate=1.0, num_workers=3)
print(f'There are {len(hd_test_ds)} test samples in HD!')  # 227
for sample in tqdm(hd_test_ds):
    pass

zs_test_files = load_decathlon_datalist(zs_datalist_json, False, "test", base_dir=data_dir)
zs_test_ds = monai.data.CacheDataset(data=zs_test_files, transform=test_transforms, cache_rate=1.0, num_workers=3)
print(f'There are {len(zs_test_ds)} test samples in ZS!')  # 426
for sample in tqdm(zs_test_ds):
    pass