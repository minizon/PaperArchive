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



- Semantic Human Matting 2018.9

  文中提到了语义分割与matting所要实现的细节分割是不一样的

  the matting module is highly nonlinear and trained to focus on structural patterns of details, thus the semantic information from input hardly retains.

  本文也是设计了两个网络来分别完整大致的语义分割和精细的matting，需要通过pretrain来预热网络，之后再进行端到端的训练。Loss方面除了alpha prediction loss和合成loss(compositional loss)外，还增加了decomposition loss

  ![Segmentation_HumanMatting](/Segmentation_HumanMatting.png)




### Detection

- Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks 2015.6

  利用RPN生成class-agnostic proposals，在利用anchor回归proposal的坐标(x,y,h,w)时，采用的是和R-CNN相同的方式：

  $t_{x} = (x- x_{a})/w_{a}, t_{y}=(y-y_{a})/h_{a}, t_{w}=\log(w/w_{a}), t_{y}=\log(h/h_{a})$

  $t_{x}^{\ast}=(x^{\ast} -x_{a})/w_{a}, t_{y}^{\ast}=(y^{\ast} -y_{a})/h_{a}, t_{w}^{\ast}=\log(w^{\ast}/w_{a}), t_{y}^{\ast}=\log(h^{\ast}/h_{a})$

  即拟合的是gt相对anchor的偏移量，可以把anchor看成是一种中间过渡状态

  L1 smooth loss:

  $smooth_{L1}(x)=\left\{ \begin{array}{ll} 0.5x^{2} & if \|x\|<1 \\ \|x\|-0.5 & otherwise \end{array} \right.$

  在训练RPN时，正负样本比例为$1:1$

  RPN是否就是一种conditional的机制

- SSD: Single Shot MultiBox Detector 2015.12

  SSD在backbone后接入降尺度的多个卷积层用于生成多尺度的anchor，相对R-CNN系列其利用anchor直接回归目标的bbox

  ![Detection_SSD](/Detection_SSD.png)

  在判断正负样本时，其对大于0.5IoU的achors都认为是正样本

  binary classification for each class

- YOLOv3: An Incremental Improvement 2018.4

  At $320 \times 320$ YOLOv3 runs in 22 ms at 28.2 mAP, as accurate as SSD but three times faster.

  回归函数略有不同：

  $\sigma(t_{x}) = x-x_{c},  \sigma(t_{y})=y-y_{c}, t_{w}=\log(w/w_{a}), t_{y}=\log(h/h_{a})$

  这里x, y, w, h已经相对图像(w,h)作了归一化

  指出了在实验中Focal loss失效了

  YOLOv3 is a good detector.  It’s fast, it’s accurate.  It’s not as great on the COCO average AP between .5 and .95 IOU metric. But it’s very good on the old detection metric of .5 IOU.



- Region Proposal by Guided Anchoring 2019.1

  本文提出利用网络特征来预测anchor，替代原有检测框架中设置稠密anchor的先验过程。anchor的预测采用“中心点+长宽”的方式。（能否在此基础上对方向进行估计？）

  ![Detection_GARPN](/Detection_GARPN.png)

  对于长宽的预测，本文预先进行了坐标变换：$w=\sigma \cdot s \cdot e^{dw}; h=\sigma \cdot s \cdot e^{dh}$, 使得变量的取值范围相对集中；同时在回归的过程中，为了拟合最佳的$w,h$，采用了采样的方式，用了9对预设参数（退化？）

  对于正负样本的设置，在合适的scale的feature map上设置center region作为正样本采样区域；设置ignore region作为正负样本的隔离区，隔离区在每个scale上都有；除正样本区域和隔离区外，均设置为outside region作为负样本区域。

  Feature adaption采样1x1conv估计offsef field应用于deformable conv（调整的应该是局部偏差量，不是整体尺寸的缩放）

- ThunderNet: Towards Real-time Generic Object Detection 2019.3

  整体感觉工程化，从网络结构的一些参数上可能参考了Light Head R-CNN，Context Enhancement Module对三个尺度的特征进行求和，Spatial Attention Module在motivation上更多是想让较少的特征通道表达更多有效的信息，利用RPN中的特征层（具备对前后景的感知）作为condition

  ![Detection_ThunderNet](/Detection_ThunderNet.png)




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