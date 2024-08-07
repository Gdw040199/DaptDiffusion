# *************************************************************************
# Copyright (2023) Bytedance Inc.
#
# Copyright (2023) DragDiffusion Authors 
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 
#
#     http://www.apache.org/licenses/LICENSE-2.0 
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
# *************************************************************************

# evaluate similarity between images before and after dragging
import argparse
import os
import shutil
import tempfile

import PIL
import clip
import lpips
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from einops import rearrange
from pytorch_fid import fid_score


def preprocess_image(image, device):
    image = torch.from_numpy(image).float() / 127.5 - 1  # Normalize to [-1, 1]
    image = rearrange(image, "h w c -> 1 c h w")
    image = image.to(device)
    return image


def calculate_fid(source_folder, target_folder, device):
    fid = fid_score.calculate_fid_given_paths([source_folder, target_folder], 50, device, 2048)
    return fid


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="setting arguments")
    parser.add_argument('--eval_root', action='append', help='root of dragging results for evaluation', required=True)
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device, jit=False)

    all_category = [
        'art_work',
        'land_scape',
        'building_city_view',
        'building_countryside_view',
        'animals',
        'human_head',
        'human_upper_body',
        'human_full_body',
        'interior_design',
        'other_objects',
    ]

    original_img_root = 'drag_bench_data/'

    for target_root in args.eval_root:
        all_lpips = []
        all_clip_sim = []
        all_fids = []

        # Temporary directories to store images for FID calculation
        temp_dir_source = tempfile.mkdtemp()
        temp_dir_target = tempfile.mkdtemp()

        for cat in all_category:
            category_source_path = os.path.join(original_img_root, cat)
            category_target_path = os.path.join(target_root, cat)
            subfolders = set(os.listdir(category_source_path)) & set(os.listdir(category_target_path))

            for subfolder in subfolders:
                source_image_path = os.path.join(category_source_path, subfolder, 'original_image.png')
                dragged_image_path = os.path.join(category_target_path, subfolder, 'dragged_image.png')

                if not os.path.exists(dragged_image_path):
                    continue

                # Copy images to temporary directories for FID calculation
                shutil.copy(source_image_path, temp_dir_source)
                shutil.copy(dragged_image_path, temp_dir_target)

                source_image_PIL = Image.open(source_image_path)
                dragged_image_PIL = Image.open(dragged_image_path)
                dragged_image_PIL = dragged_image_PIL.resize(source_image_PIL.size, PIL.Image.BILINEAR)

                source_image = preprocess_image(np.array(source_image_PIL), device)
                dragged_image = preprocess_image(np.array(dragged_image_PIL), device)

                with torch.no_grad():
                    source_image_224x224 = F.interpolate(source_image, (224, 224), mode='bilinear')
                    dragged_image_224x224 = F.interpolate(dragged_image, (224, 224), mode='bilinear')
                    cur_lpips = loss_fn_alex(source_image_224x224, dragged_image_224x224)
                    all_lpips.append(cur_lpips.item())

                source_image_clip = clip_preprocess(source_image_PIL).unsqueeze(0).to(device)
                dragged_image_clip = clip_preprocess(dragged_image_PIL).unsqueeze(0).to(device)

                with torch.no_grad():
                    source_feature = clip_model.encode_image(source_image_clip)
                    dragged_feature = clip_model.encode_image(dragged_image_clip)
                    source_feature /= source_feature.norm(dim=-1, keepdim=True)
                    dragged_feature /= dragged_feature.norm(dim=-1, keepdim=True)
                    cur_clip_sim = (source_feature * dragged_feature).sum()
                    all_clip_sim.append(cur_clip_sim.cpu().numpy())

        # Calculate FID for the entire category
        fid_value = calculate_fid(temp_dir_source, temp_dir_target, device)
        all_fids.append(fid_value)

        # Cleanup temporary directories
        shutil.rmtree(temp_dir_source)
        shutil.rmtree(temp_dir_target)

        print(target_root)
        print('Average LPIPS: ', np.mean(all_lpips))
        print('Average CLIP similarity: ', np.mean(all_clip_sim))
        print('FID:', np.mean(all_fids) if all_fids else "No FID calculated due to missing images")
