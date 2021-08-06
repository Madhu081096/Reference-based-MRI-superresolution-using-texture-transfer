# Reference based MRI Super resolution using texture transfer
Official pytorch implementation of the paper [Reference-Based Texture Transfer For Single Image Super-Resolution Of Magnetic Resonance Images](https://ieeexplore.ieee.org/document/9433961) accepted in International Symposium on Biomedical Imaging (ISBI) 2021

In this paper, we propose a reference-based, unpaired multi-contrast texture-transfer strategy for deep learning based in-plane and across-plane MRI super-resolution. 
We use the scattering transform to relate the texture features of image patches to unpaired reference image patches, and additionally a loss term for multicontrast texture.
We apply our scheme in different superresolution architectures, observing improvement in PSNR and SSIM for 4x super-resolution in most of the cases.


## Method

Morlet wavelet transform is used for computing the texture features of the image.

Feature swapping is the key procedure in the proposed method and has two steps - Dense matching and texture transfer.

Dense matching – takes the wavelet transform of LR input and blurred reference, and obtains the matched correspondences by taking small patches. The indices of the reference texture patches that give the maximum correspondence scores are stored

Texture transfer – the stored indices are used to locate the patches of corresponding texture feature patches of HR reference and these patches are placed at the appropriate positions replacing the LR image features. 

The swapped features are then concatenated to the initial Convolutional layer of the super-resolution network.
![image](https://user-images.githubusercontent.com/37436778/128494282-9031ff78-608b-4ff7-996f-928171134c4c.png)

