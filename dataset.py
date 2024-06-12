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
import os
import queue
import threading

import cv2
import numpy as np
import torch
from natsort import natsorted
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from imgproc import image_to_tensor, image_resize

__all__ = [
    "BaseImageDataset", "PairedImageDataset",
    "PrefetchGenerator", "PrefetchDataLoader", "CPUPrefetcher", "CUDAPrefetcher",
]


class BaseImageDataset(Dataset):
    """Define training dataset loading methods."""

    def __init__(
            self,
            gt_images_dir: str,
            lr_images_dir: str,
            upscale_factor: int = 4,
    ) -> None:
        """

        Args:
            gt_images_dir (str): Ground-truth image address.
            lr_images_dir (str, optional): Low resolution image address. Default: ``None``
            upscale_factor (int, optional): Image up scale factor. Default: 4
        """

        self.skipped_images = 0

        super(BaseImageDataset, self).__init__()
        # check if the ground truth images folder is empty
        if os.listdir(gt_images_dir) == 0:
            raise RuntimeError("GT image folder is empty.")
        # check if the image magnification meets the model requirements
        if upscale_factor not in [2, 4, 8]:
            raise RuntimeError("Upscale factor must be 2, 4, or 8.")

        # Read a batch of low-resolution images
        if lr_images_dir is None:
            image_file_names = natsorted(os.listdir(gt_images_dir))
            valid_extensions = ('.png', '.jpg', '.jpeg')
            image_file_names = [file for file in image_file_names if file.lower().endswith(valid_extensions)]
            self.lr_image_file_names = None
            self.gt_image_file_names = [os.path.join(gt_images_dir, image_file_name) for image_file_name in image_file_names]
        else:
            if os.listdir(lr_images_dir) == 0:
                raise RuntimeError("LR image folder is empty.")
            
            image_file_names = natsorted(os.listdir(gt_images_dir))
            valid_extensions = ('.png', '.jpg', '.jpeg')
            image_file_names = [file for file in image_file_names if file.lower().endswith(valid_extensions)]
            self.lr_image_file_names = [os.path.join(lr_images_dir, image_file_name) for image_file_name in image_file_names]
            self.gt_image_file_names = [os.path.join(gt_images_dir, image_file_name) for image_file_name in image_file_names]

            print('len lr image names:', len(self.lr_image_file_names))
            print('len gt image names:', len(self.gt_image_file_names))
            
            n1 = self.check_files_exists(self.lr_image_file_names)
            n2 = self.check_files_exists(self.gt_image_file_names)

            if n1 + n2 > 0:
                raise RuntimeError("Names images does not match")

        self.upscale_factor = upscale_factor

    def check_files_exists(self, file_names):
        non_existent_files = []
        for file_name in file_names:
            if not os.path.exists(file_name):
                non_existent_files.append(file_name)
        print('non existent:', non_existent_files)
        return len(non_existent_files)

    def print_differences(self, list1, list2):
        differences = 0
        max_length = max(len(list1), len(list2))
        
        for i in range(max_length):
            if i >= len(list1):
                print(f"Index {i}: {None} != {list2[i]}")
                differences += 1
            elif i >= len(list2):
                print(f"Index {i}: {list1[i]} != {None}")
                differences += 1
            elif list1[i] != list2[i]:
                print(f"Index {i}: {list1[i]} != {list2[i]}")
                differences += 1

        print('number of differences', differences)
        return differences
    
    # ./data/Flickr2k_LRWRGT/dataset_GT_patches/000446_0007.png
    # ./data/Flickr2k_LRWRGT/dataset_LRW_patches/000446_0007.png

    def __getitem__(
            self,
            batch_index: int
    ) -> [Tensor, Tensor]:
        # Read a batch of ground truth images



        gt_image = cv2.imread(self.gt_image_file_names[batch_index])
        # gt_image = cv2.imread('./data/Flickr2k_LRWRGT/dataset_GT_patches/000001_0001.png') 
        # gt_image = cv2.imread('./data/Flickr2k_LRWRGT/dataset_GT_patches/000223_0125.png')

        if gt_image is None or (isinstance(gt_image, np.ndarray) and not gt_image.any()):
            print('gt_image', batch_index, len(self.gt_image_file_names))
            print('file name', self.gt_image_file_names[batch_index])
            path = Path(self.gt_image_file_names[batch_index])
            print('path exists', path.exists())

            # Check if the file is readable
            if not os.access(self.gt_image_file_names[batch_index], os.R_OK):
                print("File is not readable")
            
            # Check for unsupported file format
            if path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                print(f"Unsupported file format: {path.suffix}")

            # Check if the file is corrupted or OpenCV is not functioning correctly
            if gt_image is None:
                print('gt_image is None', self.skipped_images)
                # Fallback to the first image
                gt_image = cv2.imread(self.gt_image_file_names[0])
                self.skipped_images += 1

            
            
        gt_image = gt_image.astype(np.float32) / 255.
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        gt_tensor = image_to_tensor(gt_image, False, False)

        # Read a batch of low-resolution images
        if self.lr_image_file_names is not None:
            lr_image = cv2.imread(self.lr_image_file_names[batch_index]).astype(np.float32) / 255.
            lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
            lr_tensor = image_to_tensor(lr_image, False, False)
        else:
            lr_tensor = image_resize(gt_tensor, 1 / self.upscale_factor)
            print("resizing GT")

        return {"gt": gt_tensor,
                "lr": lr_tensor}

    def __len__(self) -> int:
        return len(self.gt_image_file_names)
    
    def get_skipped(self):
        return self.skipped_images
    



class PairedImageDataset(Dataset):
    """Define Test dataset loading methods."""

    def __init__(
            self,
            paired_gt_images_dir: str,
            paired_lr_images_dir: str,
    ) -> None:
        """

        Args:
            paired_gt_images_dir: The address of the ground-truth image after registration
            paired_lr_images_dir: The address of the low-resolution image after registration
        """

        super(PairedImageDataset, self).__init__()
        if not os.path.exists(paired_lr_images_dir):
            raise FileNotFoundError(f"Registered low-resolution image address does not exist: {paired_lr_images_dir}")
        if not os.path.exists(paired_gt_images_dir):
            raise FileNotFoundError(f"Registered high-resolution image address does not exist: {paired_gt_images_dir}")

        # Get a list of all image filenames
        image_files = natsorted(os.listdir(paired_lr_images_dir))
        valid_extensions = ('.png', '.jpg', '.jpeg')

        

        self.paired_gt_image_file_names = [os.path.join(paired_gt_images_dir, x) for x in image_files if x.lower().endswith(valid_extensions)]
        self.paired_lr_image_file_names = [os.path.join(paired_lr_images_dir, x) for x in image_files if x.lower().endswith(valid_extensions)]

    def __getitem__(self, batch_index: int) -> [Tensor, Tensor, str]:
        # Read a batch of image data
        gt_image = cv2.imread(self.paired_gt_image_file_names[batch_index]).astype(np.float32) / 255.
        lr_image = cv2.imread(self.paired_lr_image_file_names[batch_index]).astype(np.float32) / 255.

        # BGR convert RGB
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        gt_tensor = image_to_tensor(gt_image, False, False)
        lr_tensor = image_to_tensor(lr_image, False, False)

        return {"gt": gt_tensor,
                "lr": lr_tensor,
                "image_name": self.paired_lr_image_file_names[batch_index]}

    def __len__(self) -> int:
        return len(self.paired_lr_image_file_names)


class PrefetchGenerator(threading.Thread):
    """A fast data prefetch generator.

    Args:
        generator: Data generator.
        num_data_prefetch_queue (int): How many early data load queues.
    """

    def __init__(self, generator, num_data_prefetch_queue: int) -> None:
        threading.Thread.__init__(self)
        self.queue = queue.Queue(num_data_prefetch_queue)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self) -> None:
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


class PrefetchDataLoader(DataLoader):
    """A fast data prefetch dataloader.

    Args:
        num_data_prefetch_queue (int): How many early data load queues.
        kwargs (dict): Other extended parameters.
    """

    def __init__(self, num_data_prefetch_queue: int, **kwargs) -> None:
        self.num_data_prefetch_queue = num_data_prefetch_queue
        super(PrefetchDataLoader, self).__init__(**kwargs)

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_data_prefetch_queue)


class CPUPrefetcher:
    """Use the CPU side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
    """

    def __init__(self, dataloader: DataLoader) -> None:
        self.original_dataloader = dataloader
        self.data = iter(dataloader)

    def next(self):
        try:
            return next(self.data)
        except StopIteration:
            return None

    def reset(self):
        self.data = iter(self.original_dataloader)

    def __len__(self) -> int:
        return len(self.original_dataloader)


class CUDAPrefetcher:
    """Use the CUDA side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader: DataLoader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)
