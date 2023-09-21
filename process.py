import SimpleITK
import torch
import swinunetr_inference
import dynunet_inference
import os
import numpy as np
import nibabel as nib
import gc

# from uNet_baseline.densenet_inference import densenet_inference


class Unet_baseline():  # SegmentationAlgorithm is not inherited in this class anymore
    def __init__(self):
        """
        Write your own input validators here
        Initialize your model etc.
        """
        self.input_path = '/input/'  # according to the specified grand-challenge interfaces
        self.output_path = '/output/images/automated-petct-lesion-segmentation/'  # according to the specified grand-challenge interfaces
        self.nii_path = '/opt/algorithm/'  # where to store the nii files
        self.ckpt_dyn_path = '/opt/algorithm/best_dyn_metric_model.pth'
        self.ckpt_swin_path = '/opt/algorithm/best_swin_metric_model.pth'
        self.ckpt_class_path = '/opt/algorithm/best_metric_model_classification3d_dict.pth'
        self.ckpt_attention_path = '/opt/algorithm/best_attention_metric_model.pth'
        # self.export_dir = '/output/'

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        pass

    def convert_mha_to_nii(self, mha_input_path, nii_out_path):
        img = SimpleITK.ReadImage(mha_input_path)
        SimpleITK.WriteImage(img, nii_out_path, True)

    def convert_nii_to_mha(self, nii_input_path, mha_out_path):
        img = SimpleITK.ReadImage(nii_input_path)
        SimpleITK.WriteImage(img, mha_out_path, True)

    def check_gpu(self):
        """
        Check if GPU is available
        """
        print('Checking GPU availability')
        is_available = torch.cuda.is_available()
        print('Available: ' + str(is_available))
        print(f'Device count: {torch.cuda.device_count()}')
        if is_available:
            print(f'Current device: {torch.cuda.current_device()}')
            print('Device name: ' + torch.cuda.get_device_name(0))
            print('Device memory: ' + str(torch.cuda.get_device_properties(0).total_memory))

    def load_inputs(self):
        """
        Read from /input/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        ct_mha = os.listdir(os.path.join(self.input_path, 'images/ct/'))[0]
        pet_mha = os.listdir(os.path.join(self.input_path, 'images/pet/'))[0]
        uuid = os.path.splitext(ct_mha)[0]

        self.convert_mha_to_nii(os.path.join(self.input_path, 'images/pet/', pet_mha),
                                os.path.join(self.nii_path, 'SUV.nii.gz'))
        self.convert_mha_to_nii(os.path.join(self.input_path, 'images/ct/', ct_mha),
                                os.path.join(self.nii_path, 'CTres.nii.gz'))
        return uuid

    def write_outputs(self, uuid):
        """
        Write to /output/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.convert_nii_to_mha(os.path.join(self.output_path, "PRED.nii.gz"),
                                os.path.join(self.output_path, uuid + ".mha"))
        print('Output written to: ' + os.path.join(self.output_path, uuid + ".mha"))

    def predict(self, inputs):
        """
        Your algorithm goes here
        """
        pass
        # return outputs

    def avg_mask(self, output_path, mask_swin, mask_dyn, ct_affine):
        # Calculate the element-wise average of the argmax masks
        average_mask = (0.4 * mask_swin + 0.6 * mask_dyn) / 1
        argmax_mask = np.argmax(average_mask.cpu().numpy(), axis=0)

        # Threshold the average mask to get a binary mask
        threshold = 0.5
        mask_out = (argmax_mask > threshold).astype(np.uint8)
        mask_export = nib.Nifti1Image(mask_out, ct_affine)
        print(os.path.join(output_path, "PRED.nii.gz"))
        nib.save(mask_export, os.path.join(output_path, "PRED.nii.gz"))

    def process(self):
        """
        Read inputs from /input, process with your algorithm and write to /output
        """
        self.check_gpu()
        print('Start processing')
        uuid = self.load_inputs()
        print('Start prediction')
        CT = nib.load(os.path.join(self.nii_path, "CTres.nii.gz"))
        ct_affine = CT.affine
        # is_tumor = densenet_inference(self.ckpt_class_path, self.nii_path)
        # print("Tumor probability:", is_tumor)
        # if is_tumor == 0:
        #     mask = np.zeros(CT.shape, dtype=np.uint8)
        #     mask_export = nib.Nifti1Image(mask, ct_affine)
        #     print(os.path.join(self.output_path, "PRED.nii.gz"))
        #     nib.save(mask_export, os.path.join(self.output_path, "PRED.nii.gz"))
        # else:
        swin_pred = swinunetr_inference.swin_run_inference(self.ckpt_swin_path, self.nii_path)
        dyn_pred = dynunet_inference.dyn_run_inference(self.ckpt_dyn_path, self.nii_path)
        self.avg_mask(self.output_path, swin_pred, dyn_pred, ct_affine)
        # attention_inference.attention_run_inference(self.ckpt_attention_path, self.nii_path)
        print('Start output writing')
        self.write_outputs(uuid)


if __name__ == "__main__":
    Unet_baseline().process()
