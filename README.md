# Reference based MRI Super resolution using texture transfer
Official pytorch implementation of the paper [Reference-Based Texture Transfer For Single Image Super-Resolution Of Magnetic Resonance Images](https://ieeexplore.ieee.org/document/9433961) accepted in International Symposium on Biomedical Imaging (ISBI) 2021

In this paper, we propose a reference-based, unpaired multi-contrast texture-transfer strategy for deep learning based in-plane and across-plane MRI super-resolution. 
We use the scattering transform to relate the texture features of image patches to unpaired reference image patches, and additionally a loss term for multicontrast texture.
We apply our scheme in different superresolution architectures, observing improvement in PSNR and SSIM for 4x super-resolution in most of the cases.


## Method

1. Morlet wavelet transform is used for computing the texture features of the image.

2. Feature swapping is the key procedure in the proposed method and has two steps - Dense matching and texture transfer.

3. Dense matching – takes the wavelet transform of LR input and blurred reference, and obtains the matched correspondences by taking small patches. The indices of the reference texture patches that give the maximum correspondence scores are stored

4. Texture transfer – the stored indices are used to locate the patches of corresponding texture feature patches of HR reference and these patches are placed at the appropriate positions replacing the LR image features. 

5. The swapped features are then concatenated to the initial Convolutional layer of the super-resolution network.

![alt text](https://github.com/Madhu081096/Reference-based-MRI-superresolution-using-texture-transfer/blob/main/Architecture.png)


## Dataset

1. [Zenodo dataset](https://zenodo.org/record/22304)
2. [Spineweb - Dataset 1](http://spineweb.digitalimaginggroup.ca/Index.php?n=Main.Datasets#Dataset_1.3A_Cross_Modality_Spinal_Images_for_Spine_Workshop)

## Results

The Table gives the quantitative comparison of SRCNN, RCAN and RFDN without and with texture transfer (metrics improvements shown in bold).
Sagittal view data - Zenodo images, Axial view data - SpineWeb images. Sagittal and axial texture transfer models end with ’-Sag’ and ’-Ax’

![alt text](https://github.com/Madhu081096/Reference-based-MRI-superresolution-using-texture-transfer/blob/main/table.png)

Qualitative result

![alt text](https://github.com/Madhu081096/Reference-based-MRI-superresolution-using-texture-transfer/blob/main/first_v3.png)

## Citation

@INPROCEEDINGS{9433961,  
  author={Madhu Mithra, K K and Ramanarayanan, Sriprabha and Ram, Keerthi and Sivaprakasam, Mohanasankar},  
  booktitle={2021 IEEE 18th International Symposium on Biomedical Imaging (ISBI)},  
  title={Reference-Based Texture Transfer For Single Image Super-Resolution Of Magnetic Resonance Images},   
  year={2021},  
  volume={},  
  number={},  
  pages={579-583},  
  doi={10.1109/ISBI48211.2021.9433961}}  
  
## Contact

For further queries, contact ee19s019@smail.iitm.ac.in





