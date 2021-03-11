# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
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
import logging
import os
import random

import torch
import torch.backends.cudnn as cudnn

import srgan_pytorch.models as models

__all__ = [
    "create_folder", "configure", "init_torch_seeds", "save_checkpoint", "weights_init",
    "AverageMeter", "ProgressMeter"
]

logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)


def create_folder(folder):
    try:
        os.makedirs(folder)
        logger.info(f"Create `{os.path.join(os.getcwd(), folder)}` directory successful.")
    except OSError:
        logger.warning(f"Directory `{os.path.join(os.getcwd(), folder)}` already exists!")
        pass


def configure(args):
    """Global profile.

    Args:
        args (argparse.ArgumentParser.parse_args): Use argparse library parse command.
    """

    # Create model
    if args.pretrained:
        logger.info(f"Using pre-trained model `{args.arch}`")
        model = models.__dict__[args.arch](pretrained=True)
    else:
        logger.info(f"Creating model `{args.arch}`")
        model = models.__dict__[args.arch]()
        if args.model_path:
            logger.info(f"You loaded the specified weight. Load weights from `{args.model_path}`")
            model.load_state_dict(torch.load(args.model_path, map_location=torch.device("cpu")), state_dict=False)

    return model


def init_torch_seeds(seed: int = None):
    r""" Sets the seed for generating random numbers.

    Args:
        seed (int): The desired seed.
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        cudnn.deterministic = True
        logger.warning("You have chosen to seed training. "
                       "This will turn on the CUDNN deterministic setting, which can slow down your "
                       "training considerably! You may see unexpected behavior when restarting from checkpoints.")


def save_checkpoint(state, is_best: bool, source_filename: str, target_filename: str):
    torch.save(state, source_filename)
    if is_best:
        torch.save(state["state_dict"], target_filename)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + '/' + fmt.format(num_batches) + "]"
