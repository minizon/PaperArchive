---
typora-root-url: assets
---

# Paper Archive

[TOC]

### Segmentation

- Panoptic Feature Pyramid Networks

在mask rcnn的基础上并行一路semantic segmentaion，实现panoptic segmentation。

针对重叠的情况，制定了三个规则：不同instance的重叠部分利用confidence score来区分，instance与semantic的重叠部分有限选择instance，semantic 低于某个阈值的需要剔除

![Semantic_Panoptic](/Semantic_Panoptic.png)



- Attention U-Net: Learning Where to Look for the Pancreas

在skip connection中引入上一尺度的信息，形成attention gate

处理精细分割的两种方式：multi-stage， attention

![Semantic_AttentionUnet](/Semantic_AttentionUnet.png)





### Detection



### Caption



### Image Synthesis

- High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs 2018.8

  <https://github.com/NVIDIA/pix2pixHD>

  尺度1024x2048

  采用了coarse-to-fine的策略，现在512x1024上生成。Loss方面不仅用GAN loss，还在discriminator的中间层进行feature matching。针对instance引入edge map使得边界上有较强的先验信息。

  ![GAN_pix2pixHD](/GAN_pix2pixHD.png)






### OCR

- Attention-based Extraction of Structured Information from Street View Imagery

84.2% on FSNS (four views)

提到了attention机制的permutation invariant，为了使attention和位置相关，本文增加了空间位置的编码信息；

本文的saliency map是基于的反向梯度，attention map则是基于的前向计算