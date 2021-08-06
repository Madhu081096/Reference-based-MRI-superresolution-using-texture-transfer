from collections import OrderedDict

import torch
import torch.nn.functional as F
import numpy as np

class Swapper:
    """
    Class for feature swapping.

    Parameters
    ---
    patch_size : int
        default patch size. increased depending on map size when applying.
    stride : int
        default stride. increased depending on map size when applying.
    """

    def __init__(self, patch_size: int = 3, stride: int = 1):
        super(Swapper, self).__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.device = torch.device('cpu')
        
    def __call__(self, map_in: OrderedDict,map_ref: OrderedDict,map_ref_blur: OrderedDict,is_weight: bool = True): 
        """

        Parameters
        ---
        map_in : OrderedDict
        map_ref : OrderedDict
        map_ref_blur : OrderedDict
        is_weight : bool, optional
            whethere weights is output.

        Returns
        ---
        maps : dict of np.array
            swapped feature maps for each layer.
        weights : np.array
            weight maps for each layer if `is_weight`'s True, otherwise `None`.
        max_idx : np.array
            index maps of the most similar patch for each position and layer.
        """
        N=len(map_ref.keys())
        max_idx=OrderedDict()
        weights=OrderedDict()
        wgts,ids,Slice,identity=None,None,None,None

        for n in range(N):
            max_idx, max_val, weights = self.match(map_in, map_ref_blur[str(n+1)], is_weight)
            if wgts is None:
                wgts=weights
                ids=max_idx
                Slice = torch.ones(wgts.shape)
                identity = torch.ones(wgts.shape)
            else:
                index=weights>wgts
                # print('type ',type(index))
                wgts[index]=weights[index]
                ids[index]=max_idx[index]
                Slice[index] =  (n+1)*identity[index]
        maps = self.swap(map_in, map_ref, ids,Slice)

        if is_weight:
            weights = weights.to('cpu').numpy()

        return maps, weights, max_idx.to('cpu').numpy()

    def match(self,map_in: OrderedDict,map_ref_blur: OrderedDict,is_weight: bool = True):
        """
        Patch matching between content and condition images.

        Parameters
        ---
        content : torch.Tensor
            The VGG feature map of the content image, shape: (C, H, W)
        patch_condition : torch.Tensor
            The decomposed patches of the condition image,
            shape: (C, patch_size, patch_size, n_patches)

        Returns
        ---
        max_idx : torch.Tensor
            The indices of the most similar patches
        max_val : torch.Tensor
            The pixel value within max_idx.
        """

        content = map_in.squeeze(0)
        patch_content = self.sample_patches(content) # patch decomposition
        patch_content /= patch_content.norm(p=2, dim=(0, 1, 2)) + 1e-5 # normalize content and condition
        
        
        condition = map_ref_blur.squeeze(0)
        patch_condition = self.sample_patches(condition) # patch decomposition
        patch_condition /= patch_condition.norm(p=2, dim=(0, 1, 2)) + 1e-5 # normalize content and condition

        _, H, W = content.shape
        batch_size = int(1024. ** 2 * 512 / (H * W)) #Not necessary, we can specify any big nummber
        n_patches = patch_condition.shape[-1]

        max_idx, max_val = None, None
        for idx in range(0, n_patches, batch_size):
            batch = patch_condition[..., idx:idx+batch_size]
            corr = F.conv2d(content.unsqueeze(0),
                            batch.permute(3, 0, 1, 2),
                            stride=self.stride)

            max_val_tmp, max_idx_tmp = corr.squeeze(0).max(dim=0)

            if max_idx is None:
                max_idx, max_val = max_idx_tmp, max_val_tmp
            else:
                indices = max_val_tmp > max_val
                max_val[indices] = max_val_tmp[indices]
                max_idx[indices] = max_idx_tmp[indices] + idx
                

        if is_weight:  # weight calculation
            weight = self.compute_weights(
                patch_content, patch_condition).reshape(max_idx.shape)
        else:
            weight = None

        return max_idx, max_val, weight
    
    def sample_patches(self,inputs: torch.Tensor,patch_size: int = None,stride: int = None):
        """
        Patch sampler for feature maps.

        Parameters
        ---
        inputs : torch.Tensor
            the input feature maps, shape: (c, h, w).
        patch_size : int, optional
            the spatial size of sampled patches
        stride : int, optional
            the stride of sampling.

        Returns
        ---
        patches : torch.Tensor
            extracted patches, shape: (c, patch_size, patch_size, n_patches).
        """

        if patch_size is None:
            patch_size = self.patch_size
        if stride is None:
            stride = self.stride
        # print(inputs.shape)
        c, h, w = inputs.shape
        patches = inputs.unfold(1, patch_size, stride)\
                        .unfold(2, patch_size, stride)\
                        .reshape(c, -1, patch_size, patch_size)\
                        .permute(0, 2, 3, 1)
        return patches
    
    def compute_weights(self,
                    patch_content: torch.Tensor,
                    patch_condition: torch.Tensor):
        """
        Compute weights

        Parameters
        ---
        patch_content : torch.Tensor
            The decomposed patches of the content image,
            shape: (C, patch_size, patch_size, n_patches)
        patch_condition : torch.Tensor
            The decomposed patches of the condition image,
            shape: (C, patch_size, patch_size, n_patches)
        """

        # reshape patches to (C * patch_size ** 2, n_patches)
        content_vec = patch_content.reshape(-1, patch_content.shape[-1])
        style_vec = patch_condition.reshape(-1, patch_condition.shape[-1])

        # compute matmul between content and condition,
        # output shape is (n_patches_content, n_patches_condition)
        # print(content_vec.transpose(0, 1).shape, style_vec.shape)
        N=int(np.sqrt(content_vec.transpose(0, 1).shape[0]))
        # if N<40:
        corr = torch.matmul(content_vec.transpose(0, 1), style_vec)

        # the best match over condition patches
        weights, _ = torch.max(corr, dim=-1)
        return weights
        # else:
        #     weights1 = torch.zeros([N*N,1]).to('cpu').squeeze()
        #     for n in range(N):
        #         corr=torch.matmul(content_vec.transpose(0, 1)[N*n:N*(n+1),:], style_vec)
        #         weights,_=torch.max(corr,dim=-1)
        #         weights1[N*n:N*(n+1)]=weights.to('cpu')+torch.cat(weights.shape[0]*[torch.Tensor([N*n])]).to('cpu')
        #     return weights1

    def swap(self,
             map_in: OrderedDict,
             map_ref: OrderedDict,
             max_idx: torch.Tensor, Slice) -> dict:
        """
        Feature swapping

        Parameter
        ---
        map_in : namedtuple
        map_ref : namedtuple
        max_idx : namedtuple
        """

        swapped_maps = {}
#         for idx, layer in enumerate(['relu3_1', 'relu2_1', 'relu1_1']):
        _patch_size = self.patch_size
        _stride = self.stride

        content = map_in.squeeze(0)
        patches_style=OrderedDict()
        for n in range(len(map_ref.keys())):
            style = map_ref[str(n+1)].squeeze()
            patches_style[str(n+1)] = self.sample_patches(style, _patch_size, _stride)

        target_map = torch.zeros_like(content).to(self.device)
        count_map = torch.zeros(target_map.shape[1:]).to(self.device)
        for i in range(max_idx.shape[0]):
            for j in range(max_idx.shape[1]):
                _i, _j = i , j 
                selected_patch=patches_style[str(int(Slice[_i,_j]))]
                target_map[:, _i:_i+_patch_size, _j:_j+_patch_size]\
                    += selected_patch[..., max_idx[i, j]]
                count_map[_i:_i+_patch_size, _j:_j+_patch_size] += 1
        target_map /= count_map

        assert not torch.isnan(target_map).any()

        swapped_maps.update({'tex1': target_map.cpu().numpy()})

        return swapped_maps
        
    def to(self, device):
        self.device = device
        return self