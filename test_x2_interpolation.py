# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import os
import time
from typing import Any
from datetime import datetime

import cv2
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

import model
from dataset import CUDAPrefetcher, PairedImageDataset, CPUPrefetcher
from imgproc import tensor_to_image
from utils import build_iqa_model, load_pretrained_state_dict, make_directory, AverageMeter, ProgressMeter, Summary


def load_dataset(config: Any, device: torch.device) -> CUDAPrefetcher:
    test_datasets = PairedImageDataset(config["TEST"]["DATASET"]["PAIRED_TEST_GT_IMAGES_DIR"],
                                       config["TEST"]["DATASET"]["PAIRED_TEST_LR_IMAGES_DIR"])
    test_dataloader = DataLoader(test_datasets,
                                 batch_size=config["TEST"]["HYP"]["IMGS_PER_BATCH"],
                                 shuffle=config["TEST"]["HYP"]["SHUFFLE"],
                                 num_workers=config["TEST"]["HYP"]["NUM_WORKERS"],
                                 pin_memory=config["TEST"]["HYP"]["PIN_MEMORY"],
                                 drop_last=False,
                                 persistent_workers=config["TEST"]["HYP"]["PERSISTENT_WORKERS"])
    test_test_data_prefetcher = CPUPrefetcher(test_dataloader)

    return test_test_data_prefetcher

def upsample_patch(image_patch):
    """
    Upsamples a given image patch from 128x128 to 256x256 using bicubic interpolation.

    Parameters:
    image_patch (numpy.ndarray): Input image patch of size 128x128.

    Returns:
    numpy.ndarray: Upsampled image patch of size 256x256.
    """
    # Ensure the input patch is of size 128x128
    # if image_patch.shape[0] != 128 or image_patch.shape[1] != 128:
    #     print(image_patch)
    #     print(image_patch.shape[0])
    #     raise ValueError("Input image patch must be of size 128x128")

    # Upsample the image patch to 256x256 using bicubic interpolation
    upsampled_patch = cv2.resize(image_patch, (256, 256), interpolation=cv2.INTER_CUBIC)

    return upsampled_patch

def test(test_data_prefetcher: CPUPrefetcher,
        psnr_model: nn.Module,
        ssim_model: nn.Module,
        device: torch.device,
        config: Any,

) -> [float, float]:
    save_image = False
    save_image_dir = ""

    if config["TEST"]["SAVE_IMAGE_DIR"]:
        save_image = True
        save_image_dir = os.path.join(config["TEST"]["SAVE_IMAGE_DIR"], config["EXP_NAME"] + "_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        make_directory(save_image_dir)

    # Calculate the number of iterations per epoch
    batches = len(test_data_prefetcher)
    # Interval printing
    if batches > 50:
        print_freq = 50
    else:
        print_freq = batches
    # The information printed by the progress bar
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    psnres = AverageMeter("PSNR", ":4.2f", Summary.AVERAGE)
    psnres_sd = AverageMeter("PSNR", ":4.2f", Summary.SD)
    ssimes = AverageMeter("SSIM", ":4.4f", Summary.AVERAGE)
    ssimes_sd = AverageMeter("SSIM", ":4.4f", Summary.SD)
    progress = ProgressMeter(len(test_data_prefetcher),
                             [batch_time, psnres, psnres_sd, ssimes, ssimes_sd],
                             prefix=f"Test: ")

    # Initialize data batches
    batch_index = 0

    # Set the data set iterator pointer to 0 and load the first batch of data
    test_data_prefetcher.reset()
    batch_data = test_data_prefetcher.next()

    # Record the start time of verifying a batch
    end = time.time()

    while batch_data is not None:
        # Load batches of data
        gt = batch_data["gt"].to(device, non_blocking=True)
        lr = batch_data["lr"].to(device, non_blocking=True)

        # Reasoning
        sr = upsample_patch(lr)

        # Calculate the image sharpness evaluation index
        psnr = psnr_model(sr, gt)
        ssim = ssim_model(sr, gt)

        # record current metrics
        psnres.update(psnr.item(), sr.size(0))
        ssimes.update(ssim.item(), ssim.size(0))
        psnres_sd.update(psnr.item(), sr.size(0))
        ssimes_sd.update(ssim.item(), ssim.size(0))

        # Record the total time to verify a batch
        batch_time.update(time.time() - end)
        end = time.time()

        # Output a verification log information
        if batch_index % print_freq == 0:
            progress.display(batch_index)

            # Save the processed image after super-resolution
        if batch_data["image_name"] == "":
            raise ValueError("The image_name is None, please check the dataset.")
        if save_image:
            image_name = os.path.basename(batch_data["image_name"][0])
            sr_image = tensor_to_image(sr, False, False)
            sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_image_dir, image_name), sr_image)

            # Preload the next batch of data
        batch_data = test_data_prefetcher.next()

        # Add 1 to the number of data batches
        batch_index += 1

    # Print the performance index of the model at the current Epoch
    progress.display_summary()

    return psnres.avg, ssimes.avg


def main() -> None:
    # Read parameters from configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path",
                        type=str,
                        default="./configs/test/SRGAN_x4-SRGAN_ImageNet-Set5.yaml",
                        required=True,
                        help="Path to   config file.")
    parser.add_argument("--device",
                    type=str,
                    default="cpu",
                    required=False,
                    help="device (cpu/cuda/mps)")
    
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.full_load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # if args.device == "mps" and torch.backends.mps.is_available():
    #     device = torch.device("mps")
    # elif device == "cuda":
    #     device = torch.device("cuda", config["DEVICE_ID"])
    # else:
    #     device = torch.device("cpu")
    
    print("device:", device)
    
    test_data_prefetcher = load_dataset(config, device)

    psnr_model, ssim_model = build_iqa_model(
        config["SCALE"],
        config["TEST"]["ONLY_TEST_Y_CHANNEL"],
        device,
    )

    test(test_data_prefetcher,
         psnr_model,
         ssim_model,
         device,
         config)


if __name__ == "__main__":
    main()
