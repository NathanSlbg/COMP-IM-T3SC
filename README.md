# A Trainable Spectral-Spatial Sparse Coding Model for Hyperspectral Image Restoration
## Explorating noise complexity on the dataset dcmall

This repository contains codes, slide presentation of our final project in the Computational Imaging course at IMT Atlantique (Brest 2025).
Members:
- Nathan SALBEGO
- Xueyun WENG

This project is inspired by [this paper](https://proceedings.neurips.cc/paper/2021/file/2b515e2bdd63b7f034269ad747c93a42-Paper.pdf). The original repository is available [here](https://github.com/inria-thoth/T3SC).



## 1. Introduction
Trainable spectral-spatial sparse coding model(T3SC)is a powerful hybrid approach that combines Deep Learning and Sparse Coding to effectively denoise hyperspectral images. Its is a 2-layer architecture:
- 1) The first layer decomposes the spectrum measured at each pixel as a sparse linear combination of a few elements from a learned dictionary, thus performing a form of linear spectral unmixing per pixel
- 2) The second layer builds upon the output of the first one, which is represented as a two-dimensional feature map, and sparsely encodes patches on a dictionary in order to take into account spatial relationships between pixels within small receptive fields.

![](figs/architecture.png)

Starting with the code implementation from the initial repository, we've tested [some pre-trained model](https://pascal.inrialpes.fr/data2/tbodrito/t3sc/). Moreover, as part of this project, we have focused mainly in explorying the performance of the model with complex noise on the input dataset dcmall. 

## 2. Investigation on the impact of noise and occlusion
### 2.1. Noise
We implement some colors of noise to evaluate the model in dealing with complex noise. Some bands are affected by these noises (0.33 as default) and the rest have Gaussian white noise. 

There are 3 types supported currently: Pink, Brownian, and Blue. Noises and their impacts on Lena's image are shown below.

![](figs/noise.png)
![](figs/noise_power.png)
![](figs/noisy_lena.png)


### 2.2. Occlusion
An area in the 2D image is zeroed. Similar to the previous part, we choose some bands to mask and add Gaussian white noise to every band of the image.

As described in the article, during training phase, each image will be divided into patches of size 64x64x31 pixels(ICVL dataset). In each patch, 10 arbitrary channels contain this masked area whose position is random and cross-spectrally independent. We train 2 models with mask size as follows:
- 20x20 pixels on a patch of 64x64 pixels
- 32x32 pixels on patch of 64x64 pixels

![](figs/training_mechanism.png)

An example
![](figs/occlusion.png)

## 3. Experiments and Results
### 3.1. Experiments
Firstly, we run tests for pink noise and occlusion using the pretrained models provided by authors. Then, we train our own models and run inference once again to evaluate the improvement. It is worthy to remark that we do not use Noise Adaptive Sparse Coding mechanism in these experiments, since it is strongly biased to the type of sensor, according to the authors.

In all tests, we use $\sigma=25$ for noise.

**For noise**:
- Inference with pretrained model.
- Inference with our trained model.

**NOTE:** We can modify the ratio between color noise and white noise  noise scale.

**NOTE 2:** Currently, we only show the result for pink noise.

**For occlusion**:
- Tests with pretrained models.
- Test with its own trained model (same configured parameters).
- Test with its own trained model (different configured parameters).

**NOTE:** Occlusion is configured by number of affected bands and size of masked area.

### 3.2. Results
**Noise**  
Firstly, we test the pink-noisy image (as described previously) at different percentages of affected bands with Gaussian pretrained model. Later, we train a model of all bands being pink-noisy then evaluate the performance at 33% and 100% of bands being affected.

*100% bands affected*
Original |Noisy| Gaussian pretrained | Our weights
:---:|:---:|:---:|:---:
<img src="figs/original.png" height="200"/>|<img src="figs\pink_noise\constant_25\100\in.png" height="200"/>|<img src="figs\pink_noise\constant_25\100\out.png" height="200"/>| <img src="figs\pink_noise\pink_trained_100\100\out.png" height="200"/>

*33% bands affected*
Original |Noisy|Gaussian pretrained | Our weights
:---:|:---:|:---:|:---:
<img src="figs/original.png" height="200"/>|<img src="figs\pink_noise\constant_25\33\in.png" height="200"/>|<img src="figs\pink_noise\constant_25\33\out.png" height="200"/>| <img src="figs\pink_noise\pink_trained_100\33\out.png" height="200"/>

The performance metrics are as follows:
Percentage| Weights | MPSNR in | MPSNR out | MSSIM in | MSSIM out 
:---:|:---:|:---:|:---:|:---:|:---:
100% |Gaussian pretrained|33.717|38.457|0.889|0.967
100% |Ours|33.717|39.182|0.889|0.980
33% |Gaussian pretrained|24.998|41.467|0.395|0.972
33% |Ours|24.998|31.712|0.395|0.665


**Occlusion**  
We train 2 models, then perform tests with some pretraied models and our weights on the ICVL dataset. We set the number of occluded bands to 10 and 5 respectively.

*Results of 10 bands*

Original |Gaussian pretrained  σ=25 | Stripes pretrained
:---:|:---:|:---:
<img src="./figs/original.png" height="200"/>|<img src="./figs/occlusion/occlusion_100_100_10_bands_with_25_gaussian_noise/out_25_gaussian_pretrained.png" height="200"/>|<img src="./figs/occlusion/occlusion_100_100_10_bands_with_25_gaussian_noise/out_stripe_pretrained.png" height="200"/>

Input |Our weights (20/64) | Our weights (32/64)
:---:|:---:|:---:
<img src="./figs/occlusion/occlusion_100_100_10_bands_with_25_gaussian_noise/input.png" height="200"/>|<img src="./figs/occlusion/occlusion_100_100_10_bands_with_25_gaussian_noise/our_weights_33_percent_10_bands.png" height="200"/>|<img src="./figs/occlusion/occlusion_100_100_10_bands_with_25_gaussian_noise/our_weights_50_percent_10_bands.png" height="200"/>

*Results of 5 bands*

Original |Gaussian pretrained σ=25 | Stripes pretrained
:---:|:---:|:---:
<img src="./figs/original.png" height="200"/>|<img src="./figs/occlusion/occlusion_100_100_5_bands/out_25_gaussian_pretrained.png" height="200"/>|<img src="./figs/occlusion/occlusion_100_100_5_bands/out_stripe_pretrained.png" height="200"/>

Input |Our weights (20/64) | Our weights (32/64)
:---:|:---:|:---:
<img src="./figs/occlusion/occlusion_100_100_5_bands/input.png" height="200"/>|<img src="./figs/occlusion/occlusion_100_100_5_bands/our_weights_33_percent_10_bands.png" height="200"/>|<img src="./figs/occlusion/occlusion_100_100_5_bands/our_weights_50_percent_10_bands.png" height="200"/>

The performance metrics of 10 affected bands:  
**MPSNR in** = 20.818, **MSSIM in** = 0.159, **MSE in** = 0.0083

Test | MPSNR out | MSSIM out | MSE out
:---:|:---:|:---:|:---:
Gaussian pretrained|39.234|0.971|0.0002
Stripes pretrained|29.219|0.693| - 
Our weights (20/64) (10 bands)|40.970|0.970|0.000103
Our weights (32/64) (10 bands)|40.982|0.968|0.000096

The performance metrics of 5 affected bands:  
**MPSNR in** = 20.863, **MSSIM in** = 0.159

Test | MPSNR out | MSSIM out 
:---:|:---:|:---:
Gaussian pretrained|41.098|0.972
Stripes pretrained|29.639|0.693
Our weights (20/64) (10 bands)|41.887|0.970
Our weights (32/64) (10 bands)|40.712|0.968

### 3.3. Conclusion
- The model proves the robustness not only for Gaussian noise but also its variants for certain percentages of affected bands.
- In case occlusion, the model not only maintain good capability of noise denoising but also well reconstruct the missing area. The performance can be slightly improved if we use a model trained with higher occluded band ratio to recover an image with lower ratio.

## 4. How to run
### 4.1. Requirements
The model is developed with Python 3.8.8. You can run the command below, or follow the instruction in **run_compi** notebook.
```
pip install -r requirements.txt
```

### 4.2. Training and Testing
Please visit the original repository for more details. Currently, there are three available colors of noise: Pink, Brownian and Blue.

### Some examples
**Train**

ICVL dataset with pink noise
```
$ python main.py data=icvl noise=pink
```

Washington DC Mall dataset with occlusion
```
$ python main.py data=dcmall noise=occlusion
```

**Test**

```
$ python main.py mode=test data=icvl noise=occlusion noise params.sigma=25 model.ckpt=path/to/
icvl_occlusion_20_20_mask_10_bands.ckpt
```

Some pretrained models can be found [here](https://drive.google.com/drive/folders/1Ak6JDUvH6bFK0h5FnE3cmvagIolMzyjB). Feel free to modify the noise parameters to see the differences.
