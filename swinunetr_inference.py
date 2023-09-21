from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    ConcatItemsd, Invertd,
    ToTensord, EnsureChannelFirstd, AsDiscreted, Activationsd
)
from monai.networks.nets import SwinUNETR
# from monai.metrics import DiceMetric, compute_meandice
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import list_data_collate, TestTimeAugmentation, DataLoader, Dataset
import torch
import pytorch_lightning

import os
import glob
import numpy as np

import nibabel as nib

device = torch.device("cuda:0")


class Net(pytorch_lightning.LightningModule):
    def __init__(self):
        super().__init__()
        self._model = SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=2,
            out_channels=2,
            feature_size=48,
            use_checkpoint=True,
            use_v2=True
        ).to(device)
        self.loss_function = DiceLoss(to_onehot_y=True, softmax=True, include_background=False, batch=True)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=2)
        self.post_label = AsDiscrete(to_onehot=2)
        self.best_val_dice = 0
        self.best_val_epoch = 0

    def forward(self, x):
        return self._model(x)


    def prepare_data(self, data_dir):
        # # set up the correct data path
        images_pt = sorted(glob.glob(os.path.join(data_dir, "SUV*")))
        images_ct = sorted(glob.glob(os.path.join(data_dir, "CTres*")))
        data_dicts = [
            {"image_pt": image_name_pt, "image_ct": image_name_ct}
            for image_name_pt, image_name_ct in zip(images_pt, images_ct)
        ]
        self.val_files = data_dicts

        self.val_transforms = Compose(
            [
                LoadImaged(keys=["image_pt", "image_ct"], image_only=True),
                EnsureChannelFirstd(keys=["image_pt", "image_ct"]),

                ScaleIntensityRanged(
                    keys=["image_ct"], a_min=-100, a_max=250,
                    b_min=0.0, b_max=1.0, clip=False,
                ),
                ScaleIntensityRanged(
                    keys=["image_pt"], a_min=0, a_max=15,
                    b_min=0.0, b_max=1.0, clip=False,
                ),
                # CropForegroundd(keys=["image_pt", "image_ct"], source_key="image_ct"),
                Orientationd(keys=["image_pt", "image_ct"], axcodes="LAS"),

                # concatenate pet and ct channels
                ConcatItemsd(keys=["image_pt", "image_ct"], name="image_petct", dim=0),
                ToTensord(keys=["image_petct"]),
            ]
        )
        self.val_ds = Dataset(data=self.val_files, transform=self.val_transforms)


    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_ds, batch_size=1, num_workers=0, collate_fn=list_data_collate)
        return val_loader


def segment_PETCT_swin(ckpt_path, data_dir):
    print("starting")
    model = Net()
    # Load the saved weights
    checkpoint = torch.load(ckpt_path)
    # model.load_state_dict(checkpoint)
    # Adjust the keys
    new_state_dict = {}
    for key, value in checkpoint.items():
        new_key = "_model." + key  # prepend "_model." to the key
        new_state_dict[new_key] = value

    # Load the adjusted state_dict into the model
    model.load_state_dict(new_state_dict)
    model.eval()
    model.to(device)
    model.prepare_data(data_dir)

    # Test Time Augmentation

    tt_aug = TestTimeAugmentation(
        model.val_transforms, batch_size=1, num_workers=0, inferrer_fn=lambda x: torch.softmax(sliding_window_inference(x, (96, 96, 96), 4, model), dim=1), device=device, image_key="image_petct"
    )

    # Get images
    with torch.no_grad():
        for file in model.val_files:
            mode_tta, mean_tta, std_tta, vvc_tta = tt_aug(file, num_examples=5)
            return mean_tta

    # with torch.no_grad():
        # for i, val_data in enumerate(model.val_dataloader()):
        #     roi_size = (96, 96, 96)
        #     sw_batch_size = 4

            # val_data["PRED_swin"] = sliding_window_inference(val_data["image_petct"].to(device), roi_size, sw_batch_size,
            #                                             model)
            # val_data = [post_transforms(i) for i in decollate_batch(val_data)]
            #
            # mask_out = torch.argmax(val_data[0]["PRED_swin"], dim=0).detach()
            # mask_out = mask_out.astype(np.uint8)
            # print("done inference")
            # CT = nib.load(
            #     os.path.join(data_dir, "CTres.nii.gz"))  # needs to be loaded to recover nifti header and export mask
            # ct_affine = CT.affine
            # mask_export = nib.Nifti1Image(mask_out, ct_affine)
            # print(os.path.join(data_dir, "PRED_swin.nii.gz"))
            #
            # nib.save(mask_export, os.path.join(data_dir, "PRED_swin.nii.gz"))
            # print("done writing")
            #
            # return val_data["PRED_swin"], mean_tta


def swin_run_inference(ckpt_path='/opt/algorithm/best_swin_metric_model.pth', data_dir='/opt/algorithm/'):
    return segment_PETCT_swin(ckpt_path, data_dir)


if __name__ == '__main__':
    swin_run_inference()
