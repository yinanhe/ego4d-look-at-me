# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import random
import math
import numpy as np

def topk(matrix, K, axis=1):
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        topk_index = np.argpartition(-matrix, K, axis=axis)[0:K, :]
        topk_data = matrix[topk_index, row_index]
        topk_index_sort = np.argsort(-topk_data,axis=axis)
        topk_data_sort = topk_data[topk_index_sort,row_index]
        topk_index_sort = topk_index[0:K,:][topk_index_sort,row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        topk_index = np.argpartition(-matrix, K, axis=axis)[:, 0:K]
        topk_data = matrix[column_index, topk_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[column_index, topk_index_sort]
        topk_index_sort = topk_index[:,0:K][column_index,topk_index_sort]
    return (topk_data_sort, topk_index_sort)

class RandomMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 3

        self.frames, self.height, self.width = input_size

        self.num_patches = self.frames *  self.height * self.width # 8x14x14
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self):
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_mask),
            np.ones(self.num_mask),
        ])
        np.random.shuffle(mask)
        return mask # [196*8]

class TemporalConsistencyMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame =  self.height * self.width # 14x14
        self.total_patches = self.frames * self.num_patches_per_frame 
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        mask_per_frame = np.hstack([
            np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
            np.ones(self.num_masks_per_frame),
        ])
        np.random.shuffle(mask_per_frame)
        mask = np.tile(mask_per_frame, (self.frames,1)).flatten()
        return mask # [196*8]


class TemporalProgressiveMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame =  self.height * self.width # 14x14
        self.total_patches = self.frames * self.num_patches_per_frame  # 8x14x14
        max_keep_patch = int((1 - mask_ratio) * self.num_patches_per_frame) # 1 - 0.75 = 0.25
        min_keep_patch = int(0.05 * self.num_patches_per_frame)
        self.keep_patches_list = np.linspace(max_keep_patch, min_keep_patch, self.frames).astype(int)
        self.total_masks = self.total_patches - self.keep_patches_list.sum()
    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        
        rand = np.random.randn(1, self.num_patches_per_frame)
        mask = np.zeros((self.frames, self.num_patches_per_frame), dtype=np.bool)
        for i in range(self.frames):
            top_k, _ = topk(rand, self.keep_patches_list[i])
            the_topk = top_k[0][-1]
            mask[i] = rand<=the_topk
        mask = mask.flatten().astype(int)
        return mask # [196*8]


class TemporalCenteringProgressiveMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        self.num_frames, self.height, self.width = input_size
        self.num_patches_per_frame =  self.height * self.width # 14x14
        self.total_patches = self.num_frames * self.num_patches_per_frame  # 8x14x14
        min_mask_ratio = mask_ratio # 0.9 -> keep 19 token
        # 0.979 -> keep 4 token  0.95 -> keep 9 token
        max_mask_ratio = 0.95  
        max_keep_patch = int((1 - min_mask_ratio) * self.num_patches_per_frame) # 1 - 0.9 = 0.1
        min_keep_patch = int((1 - max_mask_ratio) * self.num_patches_per_frame) # 1 - 0.95 = 0.05
        patches_list = np.linspace(max_keep_patch, min_keep_patch, self.num_frames//2 ).astype(int).tolist()
        self.keep_patches_list = patches_list.copy()
        patches_list.reverse()
        self.keep_patches_list = patches_list + self.keep_patches_list
        self.total_masks = self.total_patches - sum(self.keep_patches_list)
    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        
        rand = np.random.randn(1, self.num_patches_per_frame)
        mask = np.zeros((self.num_frames, self.num_patches_per_frame), dtype=np.bool)
        for i in range(self.num_frames):
            top_k, _ = topk(rand, self.keep_patches_list[i])
            the_topk = top_k[0][-1]
            mask[i] = rand<=the_topk
        mask = mask.flatten().astype(int)
        return mask # [196*8]
