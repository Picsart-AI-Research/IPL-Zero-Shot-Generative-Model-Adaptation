import os
import shutil

import torch
from torchvision import utils


def get_dir_img_list(dir_path, valid_exts=[".png", ".jpg", ".jpeg"]):
    file_list = [os.path.join(dir_path, file_name) for file_name in os.listdir(dir_path) 
                 if os.path.splitext(file_name)[1].lower() in valid_exts]

    return file_list

def copytree(src, dst, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                shutil.copy2(s, d)

def save_images(images: torch.Tensor, output_dir: str, file_prefix: str, nrows: int, iteration: int) -> None:
    utils.save_image(
        images,
        os.path.join(output_dir, f"{file_prefix}_{str(iteration).zfill(6)}.jpg"),
        nrow=nrows,
        normalize=True,
        range=(-1, 1),
    )

def resize_img(img: torch.Tensor, size: int) -> torch.Tensor:
    return torch.nn.functional.interpolate(img.unsqueeze(0), (size, size))[0]
