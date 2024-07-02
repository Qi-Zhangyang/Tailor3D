# Copyright (c) 2023-2024, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from typing import Union
import random
import numpy as np
import torch
from megfile import smart_path_join, smart_open

from .cam_utils import build_camera_standard, build_camera_principle, camera_normalization_objaverse
from ..utils.proxy import no_proxy
from .objaverse import ObjaverseDataset
from .back_transform.back_transform import transform_back_image

from PIL import Image
from torchvision import transforms

__all__ = ['GobjaverseDataset']

def opposite_view(i):
   if 0 <= i <= 24:
       return (i + 12) % 24
   elif 27 <= i <= 39:
       return ((i - 27) + 6) % 12 + 27
   else:
       raise ValueError("Input number must be between 0-24 or 27-39.")

def get_random_views(rgba_dir, num_views=4):
    all_files = [f for f in os.listdir(rgba_dir) if f.endswith('.png')]
    view_numbers = [int(os.path.splitext(f)[0]) for f in all_files]
    selected_views = random.sample(view_numbers, num_views)
    return np.array(selected_views)

class GobjaverseDataset(ObjaverseDataset):

    def __init__(self, root_dirs: list[str], meta_path: str,
                 sample_side_views: int,
                 render_image_res_low: int, render_image_res_high: int, render_region_size: int,
                 source_image_res: int, normalize_camera: bool,
                 normed_dist_to_center: Union[float, str] = None, num_all_views: int = 32):
        super().__init__(
            root_dirs, meta_path,
            sample_side_views,
            render_image_res_low,
            render_image_res_high,
            render_region_size,
            source_image_res,
            normalize_camera,
            normed_dist_to_center,
            num_all_views,
            )

        self.back_transforms = transform_back_image()

    # This is for gobjaverse and objaverse_mengchen
    @staticmethod
    def _load_pose_txt(file_path):  # load .txt         #!!!
        with open(file_path, 'r') as file:
            lines = file.readlines()
        pose_data = np.array([list(map(float, line.split())) for line in lines], dtype=np.float32)
        pose = torch.from_numpy(pose_data).reshape(4, 4)     # [1. 16] -> [4, 4] -> [3, 4]
        opengl2opencv = np.array([
                    [1,  0,  0, 0], 
                    [0, -1,  0, 0], 
                    [0,  0, -1, 0], 
                    [0,  0,  0, 1]
                ], dtype=np.float32)
        # This is the camera pose in OpenCV format.
        pose = np.matmul(pose, opengl2opencv)
        return pose[:3, :] # [4, 4] -> [3, 4]

    @staticmethod
    def _load_rgba_image_transform(file_path, bg_color: float = 1.0, extra_transforms=None): #!!!
        ''' Load and blend RGBA image to RGB with certain background, 0-1 scaled '''
        rgba = np.array(Image.open(smart_open(file_path, 'rb')) )                            # (512, 512, 4)
        rgba = torch.from_numpy(rgba).float() / 255.0
        rgba = rgba.permute(2, 0, 1).unsqueeze(0)
        rgb = rgba[:, :3, :, :] * rgba[:, 3:4, :, :] + bg_color * (1 - rgba[:, 3:, :, :])
        if extra_transforms is not None:
            rgb = extra_transforms(
                transforms.ToPILImage()(rgb.squeeze())
            ).unsqueeze(0)
        return rgb                                                                          # [1, 3, 512, 512]

    @no_proxy
    def inner_get_item(self, idx):
        """
        Loaded contents:
            rgbs: [M, 3, H, W]
            poses: [M, 3, 4], [R|t]
            intrinsics: [3, 2], [[fx, fy], [cx, cy], [weight, height]]
        """
        uid = self.uids[idx]
        root_dir = self._locate_datadir(self.root_dirs, uid, locator="pose")
        
        pose_dir = os.path.join(root_dir, uid, 'pose')
        rgba_dir = os.path.join(root_dir, uid, 'rgb')

        # only one intrinsics
        intrinsics = torch.tensor([[384, 384], [256, 256], [512, 512]], dtype=torch.float) 

        # sample views (incl. source view and side views)
        sample_views = get_random_views(rgba_dir, num_views=self.sample_side_views) 
        source_image_view_back = opposite_view(sample_views[0])
        sample_views = np.insert(sample_views, 1, source_image_view_back)

        poses, rgbs, bg_colors = [], [], []
        source_image = None
        for view in sample_views:
            pose_path = smart_path_join(pose_dir, f'{view:03d}.txt')
            rgba_path = smart_path_join(rgba_dir, f'{view:03d}.png')
            pose = self._load_pose_txt(pose_path) #!!!
            bg_color = random.choice([0.0, 0.5, 1.0])
            rgb = self._load_rgba_image(rgba_path, bg_color=bg_color)
            poses.append(pose)
            rgbs.append(rgb)
            bg_colors.append(bg_color)
            if source_image is None:
                source_image = self._load_rgba_image(rgba_path, bg_color=1.0)
        assert source_image is not None, "Really bad luck!"
        poses = torch.stack(poses, dim=0)
        rgbs = torch.cat(rgbs, dim=0)

        #!!! lora for the backview
        source_image_back = self._load_rgba_image_transform(smart_path_join(rgba_dir, f'{sample_views[1]:03d}.png'), bg_color=bg_color)

        if self.normalize_camera:
            poses = camera_normalization_objaverse(self.normed_dist_to_center, poses)

        # build source and target camera features
        source_camera = build_camera_principle(poses[:1], intrinsics.unsqueeze(0)).squeeze(0)
        render_camera = build_camera_standard(poses, intrinsics.repeat(poses.shape[0], 1, 1))

        # adjust source image resolution
        source_image = torch.nn.functional.interpolate(
            source_image, size=(self.source_image_res, self.source_image_res), mode='bicubic', align_corners=True).squeeze(0)
        source_image = torch.clamp(source_image, 0, 1)

        #!!! adjust source_image_back resolution
        source_image_back = torch.nn.functional.interpolate(
            source_image_back, size=(self.source_image_res, self.source_image_res), mode='bicubic', align_corners=True).squeeze(0)
        source_image_back = torch.clamp(source_image_back, 0, 1)

        # adjust render image resolution and sample intended rendering region
        render_image_res = np.random.randint(self.render_image_res_low, self.render_image_res_high + 1)
        render_image = torch.nn.functional.interpolate(
            rgbs, size=(render_image_res, render_image_res), mode='bicubic', align_corners=True)
        render_image = torch.clamp(render_image, 0, 1)
        anchors = torch.randint(
            0, render_image_res - self.render_region_size + 1, size=(self.sample_side_views + 1, 2))
        crop_indices = torch.arange(0, self.render_region_size, device=render_image.device)
        index_i = (anchors[:, 0].unsqueeze(1) + crop_indices).view(-1, self.render_region_size, 1)
        index_j = (anchors[:, 1].unsqueeze(1) + crop_indices).view(-1, 1, self.render_region_size)
        batch_indices = torch.arange(self.sample_side_views + 1, device=render_image.device).view(-1, 1, 1)
        cropped_render_image = render_image[batch_indices, :, index_i, index_j].permute(0, 3, 1, 2)

        return {
            'uid': uid,
            'source_camera': source_camera,
            'render_camera': render_camera,
            'source_image': source_image,
            'render_image': cropped_render_image,
            'source_image_back': source_image_back, #!!!
            'render_anchors': anchors,
            'render_full_resolutions': torch.tensor([[render_image_res]], dtype=torch.float32).repeat(self.sample_side_views + 1, 1),
            'render_bg_colors': torch.tensor(bg_colors, dtype=torch.float32).unsqueeze(-1),
        }
