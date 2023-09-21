# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import glob
import numpy as np
import torch

import monai
from monai.data import CSVSaver, DataLoader
from monai.transforms import Activations, AsDiscrete, Compose, LoadImaged, RandRotate90d, Resized, ScaleIntensityd, \
    EnsureChannelFirstd, ScaleIntensityRanged, CropForegroundd, Orientationd, EnsureTyped, RandCropByPosNegLabeld, \
    ConcatItemsd, RandFlipd, RandShiftIntensityd


def densenet_inference(ckpt_path, nii_path):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    images_pt = sorted(glob.glob(os.path.join(nii_path, "SUV*")))
    images_ct = sorted(glob.glob(os.path.join(nii_path, "CTres*")))
    data_dicts = [
        {"image_pt": image_name_pt, "image_ct": image_name_ct}
        for image_name_pt, image_name_ct in zip(images_pt, images_ct)
    ]
    val_files = data_dicts
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    keys = ["image_pt", "image_ct"]
    # Define transforms for image
    val_transforms = Compose(
        [
            LoadImaged(keys=keys, image_only=True),
            EnsureChannelFirstd(keys=keys),

            ScaleIntensityRanged(
                keys=["image_ct"], a_min=-100, a_max=250,
                b_min=0.0, b_max=1.0, clip=False,
            ),
            ScaleIntensityRanged(
                keys=["image_pt"], a_min=0, a_max=15,
                b_min=0.0, b_max=1.0, clip=False,
            ),
            Resized(keys=keys, spatial_size=(128, 128, 128)),
            Orientationd(keys=keys, axcodes="LAS"),
            ConcatItemsd(keys=["image_pt", "image_ct"], name="image_petct", dim=0),

            EnsureTyped(keys=["image_petct"], device=device, track_meta=True),
        ]
    )

    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=2, num_workers=4, pin_memory=torch.cuda.is_available())

    # Create DenseNet121
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=2, out_channels=2).to(device)

    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    with torch.no_grad():
        num_correct = 0.0
        metric_count = 0
        saver = CSVSaver(output_dir="./output")
        for val_data in val_loader:
            val_images = val_data["image_petct"].to(device)
            val_outputs = model(val_images).argmax(dim=1)
            print(val_outputs)
            # value = torch.eq(val_outputs)
            # metric_count += len(value)
            # num_correct += value.sum().item()
            # saver.save_batch(val_outputs, val_data["img"].meta)
        # metric = num_correct / metric_count
        # print("evaluation metric:", metric)
        # saver.finalize()


if __name__ == "__main__":
    densenet_inference(ckpt_path='/opt/algorithm/best_metric_model_classification3d_dict.pth', nii_path='opt/algorithm/')
