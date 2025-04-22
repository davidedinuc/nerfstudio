from uco3d import UCO3DDataset, UCO3DFrameDataBuilder, opencv_cameras_projection_from_uco3d
from uco3d.dataset_utils.utils import get_dataset_root
import os
import numpy as np
from PIL import Image
import torch

def load_uco_data():
    dataset_root = get_dataset_root(assert_exists=True)
    # Get the "small" subset list containing a small subset
    # of the uCO3D categories. For loading the whole dataset
    # use "set_lists_all-categories.sqlite".
    subset_lists_file = os.path.join(
        dataset_root,
        "set_lists", 
        "set_lists_test.sqlite",
    )
    
    uco_dataset = UCO3DDataset(
    subset_lists_file=subset_lists_file,
    subsets=["train"],
    #n_frames_per_sequence=10,
    frame_data_builder=UCO3DFrameDataBuilder(
        apply_alignment=True,
        load_images=True,
        load_depths=True,
        load_masks=True,
        load_depth_masks=False,
        load_gaussian_splats=True,
        gaussian_splats_truncate_background=False,
        load_point_clouds=True,
        load_segmented_point_clouds=False,
        load_sparse_point_clouds=True,
        box_crop=True,
        box_crop_context=0.4,
        load_frames_from_videos=True,
        image_height=512,
        image_width=512,
        undistort_loaded_blobs=True,
        use_cache=False,
        )
    )

    return uco_dataset

def create_transforms_from_uco():
    uco_data = load_uco_data()
    
    return True
