---
typora-root-url: assets
---

# Paper Archive

[TOC]

哪些是可落地的技术，哪些是可构建系统的技术，哪些是需关注了解的技术

### Segmentation

> ICNet for Real-Time Semantic Segmentation on High-Resolution Images 2017.04

image cascade net, 这里的实时还是指计算机上显卡

思想还是通过coarse-to-fine来获得更准确的分割， $1024\times 2048$上的速度为30fps

![Segmentation_ICNet](/Segmentation_ICNet.png)



>UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation 2019.12

主要思路是UNet的嵌套，再加上Deeply supervised learning。它提到了和Deep layer aggregation思想类似。

<img src="/Seg_UNetplusplus.png" style="zoom:75%;" />

这类方法其实与densenet类似，存在的问题就是推理速度，尤其是是否适合在移动端部署？



> Learning Fully Dense Neural Networks for Image Semantic Segmentation 2019.05

特征复用和Focal loss在边界上的应用

<img src="/Segmentation_FDNet.png" style="zoom: 50%;" />

<img src="/Segmentation_FDNet2.png" style="zoom:75%;" />



> Gated-SCNN: Gated Shape CNNs for Semantic Segmentation 2019.07

本文从边界的视角对网络进行了处理，加入了shape stream用于显示的边界生成，再将其作为特征用于后续的网络分割

GSL的设计思想是通过high level的语义来滤除边界上的毛刺（noise，或者非语义的边缘），shape stream利用edge bce loss来引导其完成这个边界预测的任务（**当然要去边界比较准**）。通过ASPP对图像梯度和边界的引入来生成最终的分割结果。这样缓解了regular支路对边缘的学习（或者是low level对边缘学习的过分侧重）。

Dual task是对边缘部分做的精细化loss，对分割结果的边界进行二次loss计算，同时对shape stream上预测的强边界做正则（这部分比较像trick）

![](/Segmentation_GSCNN.png)



> Panoptic Feature Pyramid Networks

在mask rcnn的基础上并行一路semantic segmentaion，实现panoptic segmentation。

针对重叠的情况，制定了三个规则：不同instance的重叠部分利用confidence score来区分，instance与semantic的重叠部分优先选择instance，semantic 低于某个阈值的需要剔除

![Semantic_Panoptic](/Semantic_Panoptic.png)



> Auto-DeepLab: Hierarchical Neural Architecture Search for Semantic Image Segmentation

自动搜索分割网络的结构及其内部的block

![Segmentation_AutoDeepLab](/Segmentation_AutoDeepLab.png)

![Segmentation_AutoDeepLab2](/Segmentation_AutoDeepLab2.png)

训练时的两个原则：1. 后层输入的spatial resolution只能是2倍，1倍，0.5倍；2. 最小的spatial resolution为降采样1/32。训练过程会构造两个不同的训练集分别用于网络权重和网络结构的训练，防止过拟合

启示：U-Net中再结合U-Net



> Attention U-Net: Learning Where to Look for the Pancreas

在skip connection中引入上一尺度的信息，形成attention gate

处理精细分割的两种方式：multi-stage， attention

![Semantic_AttentionUnet](/Semantic_AttentionUnet.png)

![Semantic_AttentionGate](/Semantic_AttentionGate.png)



#### Portrait

> Automatic Portrait Segmentation for Image Stylization 2016

PortraitFCN+, **虽然是工程向的文章，但是细节说得较少，对于特定条件下构建提升分割性能的手段值得深思**

引入了相对人脸中心归一化的x，y坐标通道

**Position Channels** The objective of these channels is to encode the pixel positions relative to the face. Intuitively, this procedure expresses the position of each pixel in a coordinate system centered on the face and scaled according to the face size.

**Shape Channel** By including a channel in which a subject-shaped region is aligned with the actual portrait subject, we are explicitly providing a feature to the network which should be a reasonable initial estimate of the final solution.



> Fast Deep Matting for Portrait Animation on Mobile Phone 2017.07

双阶段的方式，处理的分辨率是$128\times 128$

![](/Segmentation_PortraitMatting.png)



>  Semantic Human Matting 2018.9

文中提到了语义分割与matting所要实现的细节分割是不一样的

the matting module is highly nonlinear and trained to focus on structural patterns of details, thus the semantic information from input hardly retains.

本文也是设计了两个网络来分别完整大致的语义分割和精细的matting，需要通过pretrain来预热网络，之后再进行端到端的训练。Loss方面除了alpha prediction loss和合成loss(compositional loss)外，还增加了decomposition loss

![Segmentation_HumanMatting](/Segmentation_HumanMatting.png)



> AUTOMATIC SKIN AND HAIR MASKING USING FULLY CONVOLUTIONAL NETWORKS

依然是coarse to fine的思路



>  Disentangled Image Matting 2019.09
>
> 旷视

同样是trimap和alpha估计是不同任务的出发点，在构造网络时利用了双解码器各自完成任务，并使用了ConvLSTM完成propagation的机制。

提出trimap是偏向语义结构，因此偏重用高阶的特征；alpha估计是偏向photometric（光学），需偏重低阶的特征。文中用了很多技术组合：large kernel增大视野，LSTM完成多次传导，uncertainty的多任务学习

![](/Segmentation_AdaMatting.png)

除了炒冷饭，其实我们还是需要去思考下，现在这些网络结构是否对任务分支有新的理解，对特征复用有新的尝试（跳跃连接，视场，上下文，效率）。例如这里在得到trimap和alpha粗略估计后，使用resblock进行propagation，这与CANet中迭代采用residual结构也很类似。



> Context-Aware Image Matting for Simultaneous Foreground and Alpha Estimation 2019.10

该方法设计了两个encoder支路，分别完成context information和local features的提取，之后用两个解码支路分别完成alpha通道和前景的预测。对于context 支路采用了4次降尺度，而local支路采用了2次降尺度，同时local支路还会采用skip connection与解码网络进行拼接。

在对alpha通道和前景通道预测结果进行监督训练时，采用了两种主要的损失函数，一种是alpha通道采用Laplacian pyramid，进行多尺度的损失计算。一种是特征损失，在gt 前景的基础上，通过gt alpha和预测alpha的合成图在网络层中的特征差异进行比对。

$\mathcal{L}_{F}^{\alpha} = \sum_{layer} \| \phi_{layer} (\hat{\alpha} \ast \hat{F} )- \phi_{layer}(\alpha \ast \hat{F})\|$

网络采用的是VGG16中的conv1_2, conv2_2, conv3_3和conv4_3层

对于前景通道，一个是采用特征损失，在gt alpha的基础上，通过gt foreground和预测前景的合成图在网络中进行特征差异的比对。第二个是L1 loss，对于gt alpha大于0的区域进行前景损失的计算

<img src="/Seg_ContextAwareMatting.png" style="zoom:75%;" />

该方法对于样本的生成进行了较为详细的描述，特别是选取trimap包含unknown区域的处理，同时对样本增强后在边缘处的合成效果进行了细致地观察，提出用re-jpeg和gaussian smoothing的方式，避免网络对光滑的背景产生错误的bias。



> FIGARO, HAIR DETECTION AND SEGMENTATION IN THE WILD

介绍了如何去构造一个头发分割的数据集，用的是人工特征加超像素的分割方法



#### Weakly

> FickleNet: Weakly and Semi-supervised Semantic Image Segmentation using Stochastic Inference 2019.03

弱监督分割

FickleNet can be regarded as a generalization of dilated convolution.

利用擦除局部区域的方式来判断区域是否属于某一类；利用多尺度的dilated convolution来生成CAM；利用判别性区域的像素点来训练一个像素级别的语义相似性估计网络AffinityNet

![Segmentation_FickleNet](/Segmentation_FickleNet.png)

We do not drop the center of the kernel of each sliding window position, so that relationships between kernel center and other locations in each stride can be found. 中心点类似于参考点，保持其不变应该是为了稳定训练过程。

弱监督能否去学习语义边界？



> CANet: Class-Agnostic Segmentation Networks with Iterative Refinement and Attentive Few-Shot Learning 2019.03

小样本分割采用双支路（a support branch and a query branch）

support和query之间采用metric learning来比对，对于support采用其ground truth区域的global pooling来表征，然后和query的feature map做比对，比对函数通过网络学习

对于k-shot的情况，设计了一个带attention的网络，用于筛选合适的物体

<img src="/Seg_CANet.png" style="zoom:75%;" />

对于Dense Comparison Module，主要是加了比对函数的网络，而不是直接对特征进行相似度计算

对于Iterative Iptimization Module，主要是设计了残差网络来调整前次分割结果可能对输入造成的分布影响

attention模块感觉是增量性的工作



#### Instance

> Pixel-level Encoding and Depth Layering for Instance-level Semantic Labeling 2016.07

该方法在语义分割的基础上利用depth 估计和instance（各点相对中心点）的方向估计来区分出各个instance。

<img src="/Seg_InstanceDepthandDirection.png" style="zoom:75%;" />

特别针对方向部分的处理，其采用了量化的方式，把每个量化角度作为一个类在同一个输出通道上进行，这与一些分通道输出的方式又有所不同。



> Hybrid Task Cascade for Instance Segmentation 2019.01

![](/Segmentation_HTC.png)

主要思想还是通过设计反向传播时的flow，以使信息可以方便地回传到特征学习的backbone上



#### Video

> Video Object Segmentation and Tracking: A Survey 2019

比较普通的总结，列举了一些数据集



> End-to-End Learning of Motion Representation for Video Understanding 2018.04

TVNet，对原本的TV-L1求解过程展开成网络形式



#### Interactive

> Efficient Interactive Annotation of Segmentation Datasets with Polygon-RNN++ 2018.03

堆砌了RNN，RL，GNN，attention多种技术，实现交互式的物体轮廓生成

<img src="/Segmentation_PolyRNNplusplus.png" style="zoom:75%;" />



> Fast Interactive Object Annotation with Curve-GCN 2019.03

图卷积的形式 https://github.com/fidler-lab/curve-gcn

![](/Segmentation_CurveGCN.png)




### Detection

> Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks 2015.6

利用RPN生成class-agnostic proposals，在利用anchor回归proposal的坐标(x,y,h,w)时，采用的是和R-CNN相同的方式：

$t_{x} = (x- x_{a})/w_{a}, t_{y}=(y-y_{a})/h_{a}, t_{w}=\log(w/w_{a}), t_{y}=\log(h/h_{a})$

$t_{x}^{\ast}=(x^{\ast} -x_{a})/w_{a}, t_{y}^{\ast}=(y^{\ast} -y_{a})/h_{a}, t_{w}^{\ast}=\log(w^{\ast}/w_{a}), t_{y}^{\ast}=\log(h^{\ast}/h_{a})$

即拟合的是gt相对anchor的偏移量，可以把anchor看成是一种中间过渡状态

L1 smooth loss:

$smooth_{L1}(x)=\left\{ \begin{array}{ll} 0.5x^{2} & if \|x\|<1 \\ \|x\|-0.5 & otherwise \end{array} \right.$

在训练RPN时，正负样本比例为$1:1$

RPN是否就是一种conditional的机制



> SSD: Single Shot MultiBox Detector 2015.12

SSD在backbone后接入降尺度的多个卷积层用于生成多尺度的anchor，相对R-CNN系列其利用anchor直接回归目标的bbox

![Detection_SSD](/Detection_SSD.png)

在判断正负样本时，其对大于0.5IoU的achors都认为是正样本

binary classification for each class



> YOLOv3: An Incremental Improvement 2018.4

At $320 \times 320$ YOLOv3 runs in 22 ms at 28.2 mAP, as accurate as SSD but three times faster.

回归函数略有不同：

$\sigma(t_{x}) = x-x_{c},  \sigma(t_{y})=y-y_{c}, t_{w}=\log(w/w_{a}), t_{y}=\log(h/h_{a})$

这里x, y, w, h已经相对图像(w,h)作了归一化

指出了在实验中Focal loss失效了

YOLOv3 is a good detector.  It’s fast, it’s accurate.  It’s not as great on the COCO average AP between .5 and .95 IOU metric. But it’s very good on the old detection metric of .5 IOU.



> Region Proposal by Guided Anchoring 2019.1

本文提出利用网络特征来预测anchor，替代原有检测框架中设置稠密anchor的先验过程。anchor的预测采用“中心点+长宽”的方式。（能否在此基础上对方向进行估计？）

![Detection_GARPN](/Detection_GARPN.png)

对于长宽的预测，本文预先进行了坐标变换：$w=\sigma \cdot s \cdot e^{dw}; h=\sigma \cdot s \cdot e^{dh}$, 使得变量的取值范围相对集中；同时在回归的过程中，为了拟合最佳的$w,h$，采用了采样的方式，用了9对预设参数（退化？）

对于正负样本的设置，在合适的scale的feature map上设置center region作为正样本采样区域；设置ignore region作为正负样本的隔离区，隔离区在每个scale上都有；除正样本区域和隔离区外，均设置为outside region作为负样本区域。

Feature adaption采样1x1conv估计offsef field应用于deformable conv（调整的应该是局部偏差量，不是整体尺寸的缩放）



> ThunderNet: Towards Real-time Generic Object Detection 2019.3

整体感觉工程化，从网络结构的一些参数上可能参考了Light Head R-CNN，Context Enhancement Module对三个尺度的特征进行求和，Spatial Attention Module在motivation上更多是想让较少的特征通道表达更多有效的信息，利用RPN中的特征层（具备对前后景的感知）作为condition

![Detection_ThunderNet](/Detection_ThunderNet.png)



> A survey of Object Classification and Detection based on 2D/3D data   2019.05

![Detection_survey](/Detection_survey.png)

![Detection_survey2](/Detection_survey2.png)

![Detection_survey3](/Detection_survey3.png)

![Detection_survey4](/Detection_survey4.png)

![Detection_survey5](/Detection_survey5.png)



#### Video

> TACNet: Transition-Aware Context Network for Spatio-Temporal Action Detection 2019.05

本文主要是检测边界的调优，对ACT的改进，加入Conv-LSTM

![Detection_TACNet](/Detection_TACNet.png)

The standard SSD performs action detection from multiple feature maps with different scales in the spatial level, but it does not consider temporal context information. To extract temporal context, we embed Bi-ConvLSTM unit into SSD framework to design a recurrent detector.

有个巧妙的处理是将两个有耦合（inter-coupled）的任务通过加减处理来解耦合

16帧输入



> DeepVS: A Deep Learning Based Video Saliency Prediction Approach

![](/Detection_DeepVS.png)

同时利用物体特征和运动特征，单整体网络结构较重，特别还利用LSTM实现帧间的显著性平滑

**在显著性训练时使用KL散度来计算loss**



### NLP

> Attention Is All You Need 2017.12

去掉RNN，只利用attention和fc完成ecode-decode过程

<img src="/NLP_Transformer.png" alt="NLP_Transformer" style="zoom:50%;" />

<img src="/NLP_Transformer2.png" alt="NLP_Transformer2" style="zoom:50%;" />

注意力公式$Attention (Q,K,V) = softmax(\frac{QK^{T}}{\sqrt{d_{k}}})V$

其中的scaling factor $\frac{1}{\sqrt{d_{k}}}$是为了对抗维度增加带来的负面影响（We suspect that for large values of $d_{k}$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients.）如何理解？

假设几个输出$a_{1}, a_{2}, a_{3}$，其中$a_{2}$最大，其中维度的作用也起着影响，softmax函数会突出相对特别大的那个数值，所以当$a_{2}$有维度加成后，会变得更大，由softmax的导数可知：

$softmax^{'} = \left\{\begin{array}{\\} S_{i}(1-S_{j}), & i=j \\ -S_{j}S_{i}, & i\neq j\end{array} \right.$

此时softmax在此处的导数接近0，其他处也因为$S_{j}$过小而为0

Multi-head的关系式$d_{k}=d_{v}=d_{model}/h=64, \, h=8$

对序列的positional encoding





#### Caption

> From Captions to Visual Concepts and Back cvpr2015

利用多示例学习MIL生成word detectors，再依据这些words来生成最合理的句子

用的是Noisy-OR MIL，用一个global threshold得到图片可用的word

<img src="/Caption_VisualConcept.png" alt="Caption_VisualConcept" style="zoom:50%;" />



> Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering 2017.07

预先提取图像中有意义的区域，相较于单纯soft attention驱动的特征选择，这种方式优势在于：1.属于hard attention，2.引入中间过程的任务学习，相对来说信息维度更加的丰富

整体是从图像侧改进

<img src="/Caption_TopDown.png" alt="Caption_TopDown" style="zoom:50%;" />

下层完成注意力的特征选择，上层是单层的语言模型



> Stack-Captioning: Coarse-to-Fine Learning for Image Captioning 2017.9

多级LSTM对生成的caption进行finetune，问题是前后语句存在对齐的问题，虽然是想解决单层LSTM caption能力弱的问题，但是对齐是否也会限制其性能提升？

用到了多层的training loss来解决梯度消失的问题

训练的细节：先用adam $4\times 10^{-4}$, momumten 0.9，CE loss; 再用adam $5\times 10^{-5}$, RL微调

![Caption_StackCaption](/Caption_StackCaption.png)



> Discriminability objective for training descriptive captions 2018

对文本描述提出了descriptive的要求，可用于text retrieval或者image retrevial；discriminative体现在生成的语句应该返回去能够检索到相应的图片，从而让caption的过程去关注图片上相对有差异的部分

利用 “Vse++: Improved visual-semantic embeddings”实现对image&caption embedding的学习，再利用SCST结合Cider来训练；图像的特征基于Visual Genome来完成



> Unsupervised Image Captioning 2019.04
>
> Tecent

无监督的文本标注，该方法是通过对抗生成的方式使得从图像特征到句子符合词库的陈述，同时通过一个visual concept detector来对齐图像和生成句子中的关键词，在利用生成器和判别器的互换（cyclegan的形式）使得图像和句子能在common latent space中对齐

![](/NLP_unsupervised.png)

![](/NLP_unsupervised2.png)



> Semantic Compositional Networks for Visual Captioning 2017.03

顾名思义，利用visual concepts补全成句子

<img src="/NLP_SCN.png" style="zoom:75%;" />



> Video Storytelling 2018.07

视频讲故事需要处理两个子问题，一个是从中提取出important clips，另一个则是将这些clips表述成连贯的段落。

作者在clip到sentence问题上使用embedding的方式，对相关的句子使用ranking loss

在选取clip时，提出Narrator Network，decide both when to sample a clip and the length of the sampled clip.

针对when to sample a clip，利用Candidate Gate来判断当前位置与上个sample位置是否有足够的差异，若差异足够则将其视为一个candidate；再利用Clip Indicator来判断当前位置是否对一个story的叙述足够重要；Clip Length则是完成对clip长度的估计，以当前帧为中心位置。

对于NLP的训练方法，依然采用基于Cider的增强学习方法，不过其baseline reward并不是基于推理时一次固定的结果，而是通过随机抽取视频帧K times做的平均。**很有可能的情况是提取clip的方式并不是非常有效，这在video summarization中有提及。**



#### VQA

> Visual Question Reasoning on General Dependency Tree 2018

Very recently, a few pioneering works take advantage of sturcture inherently contained in text and image, which parses the question-image input into a tree or graph layout and assembles local features of nodes to predict the answer.

总结了两种关系，clausal predicate relation和modifier relation，第一种用residual composition module解决，第二种使用adversarial attention module解决

![VQA_ACMN](/VQA_ACMN.png)



> From Recognition to Cognition: Visual Commonsense Reasoning 2019

这里的reason是做的选择题，不是开放式的

Given a question along with four answer choices, a model must first select the right answer. If its answer was correct, then it is provided four rational choices (that could purportedly justify its correct answer), and it must select the correct rationale.

![VQA_R2C](/VQA_R2C.png)

We use attention mechanisms to contextualize these sentences with repect to each other and the image context. 用注意力机制来实现对上下文的提取



### Classification

#### Image

> Zero-Shot Visual Recognition using Semantics-Preserving Adversarial Embedding Networks CVPR2018

对于zero-shot来说其希望能保留尽可能多的语义信息以应对未知的情况，但是对于一般的分类网络而言，其通常会把不具有判别性质的语义给丢掉，这样就很难应对zero-shot的问题

本文的思想就是想通过重建的方式保留尽可能多的语义用于后续的判别

due to the semantic discrepancy between seen and unseen classes, these attributes would be discriminative at test time, resulting in a lossy semantic space that is problematic for unseen class recognition.

One main stream of ZSL is the attribute-based visual recognition where the attributes serve as an intermediate feature space that transfer semantics across classes, supporting zero-shot recognition of unseen classes.

To scale up ZSL, embedding based methods are prevailing.

利用中间的属性语义可以借助组合来提升zero-shot的能力，但这种方式是离散的，易受标签集的限制；利用embedding的连续空间可以进一步提升zs的能力，当然网络本质上还是内插，辨别能力依然受网络和数据分布的影响

Hubness states the phenomenon that the mapped semantic embeddings from images would be collapsed to hubs, which are near many other points without being similar to the class label in any meaningful way.

![Classification_SemanticPreserving](/Classification_SemanticPreserving.png)

Our goal is to combine the rich semantics preserved in $F(x)$ from multiple $E(x^{‘})$ across a variety of classes. However, it is hard to hand-engineer a plausible combination rule for the dynamic $F(x)$ and $E(x)$ during training. To this end, we apply the adversarial objective to encourage $E(x)$ to favor solutions that reside on the manifold of $F(x)$ that preserves semantics,  by "fooling" a discrimintor network $D$ that outputs the probabilities that $E(x^{'})$ is as "real" as $F(x)$

学习相似的分布

To prevent mode collapse problem, we followed the strategy of WGAN.



#### Video

> Action Recognition with Improved Trajectories 2013

主要的想法是消除相机运动的影响，具体是去除人体部分，对背景的运动进行估计以此抵消相机运动，使用RANSAC方法估计环境的一致性



> What Actions are Needed for Understanding Human Actions in Videos? 2017.08

探讨分析动作分类的一些问题：

长尾效应，小样本的类别对整体性能提升

定位人对分类性能有帮助



> Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset  2017.05

探讨了预训练对视频分类的作用，通过将2D网络结构沿时间维拼接成3D网络结构(inflate)

![Classification_I3D](/Classification_I3D.png)

问题：视频分类的目标类型影响（单纯的物体运动，人物交互，细粒度动作区别）是否需要有针对性的建模？

有没有可视化的方法来做这件事？



> Temporal 3D ConvNets: New Architecture and Transfer Learning for Video Classification 2017.11

DenseNet+Temporal Transition Layer+match loss for pretraining

![Classification_T3D](/Classification_T3D.png)

通过匹配来预训练有点蒸馏或者GAN的意思

Temporal Transition Layer对不同时间长度的特征进行提取，视频识别时间维的多尺度？

问题：如果视频分类确实存在某些关键帧，而且间距有长有短怎么办？softmax单点突出的attention对这种多元的情形应该不合适吧？



> Attentional Pooling for Action Recognition

These methods assume that focusing on the human or its parts is always usefull for discriminating actions. This might not necessarily be true for all actions; some actions might be easier to distinguish using the background and context, like a 'basketball shoot' vs a "throw"

作者用类似secord-order pooling的方式来实现所谓的both predict and apply an attention map，并且利用low-rank approximation来避免显式计算二阶特征

![Classification_AttentionPooling](/Classification_AttentionPooling.png)



> Video Representation Learning Using Discriminative Pooling

利用MIL 和SVM做pooling



> Compressed Video Action Recognition 2018

1. videos have a very low information density
2. with only RGB images, learning temporal structure is difficult

利用视频压缩后的格式数据完成分类，只处理I-frame和P-frame的情况，对于P-frame还需要对偏差进行累积，同时对格式数据中的residual部分进行类似的处理；网络结构用TSN

![Classification_Compressed](/Classification_Compressed.png)



> MiCT: Mixed 3D/2D Convolutional Tube for Human Action Recognition CVPR2018

11-layer 3D CNN 的参数数152-layer ResNet 的1.5倍

为了增加3D CNN的深度，在网络中引入2D卷积

之前的2D网络常常需要借助optical flow之类的特征来获得时间维度的特征；但是直接堆叠3D卷积的方式不利于网络的训练

![Classification_MiCT](/Classification_MiCT.png)

最终利用TSN结构在UCF101上从88.9%提升到94.7%是不是有点打脸，说明网络学习时序特征还是不及optical flow



> WHERE  AND  WHEN  TO  LOOK?   SPATIO-TEMPORAL ATTENTION FOR ACTION RECOGNITION IN VIDEOS 2018.10

将spatial temporal attention显式地分开，并且temporal attention采用了uni-model 分布进行正则，而spatial attention则增加了total variation正则

![Classification_Spatio-temporalAttention](/Classification_Spatio-temporalAttention.png)

Spatial Attention: $\tilde{\mathbf{X}}_{i} = \mathbf{X}_{i} \bigodot \mathbf{M}_{i}$

Temporal Attention: $\mathbf{Y}_{t} = \frac{1}{n} \sum_{i=1}^{n} w_{ti} \tilde{\mathbf{X}}_{i}$



> AdaFrame: Adaptive Frame Selection for Fast Video Recognition 2018.11

uniform sampling assumes information is evenly distributed over time, which could therefore incorporate noisy background frames that are not relevant to the class of interest

如何去预测哪些帧是有用的 supervision informing which frames are important

select frames with reinforcement learning

利用global memory表征视频的上下文信息，此处使用轻量级的网络和降采样的数据来减少计算量，在global memory中也引入动态注意力

在预测分类时，加入reward：observing one more frame is expected to produce more accurate predictions

$r_{t} = \max \{ 0, m_{t} - \max_{t^{'} \in [0, t-1] m_{t^{'}}}\}$

$m_{t} = s_{t}^{gt} - max\{ s_{t}^{c}|c\neq gt \}$ is the margin between the probabiliyt of the ground-truth class and the largest proabilities from other classes.

进而引入selection network用于最大化未来的reward，而utility network则是用来根据当前状态估计未来的reward，相当于价值网络

在Inference阶段，没有用一个单一阈值来评价价值网络的输出，而是选取预测过程中最大的价值输出，设置一个margin和一定数量$p$步，决定是否结束推断

![Classification_AdaFrame](/Classification_AdaFrame.png)



> SlowFast Networks for Video Recognition 2018.12

提到物体在空间的各向同性，而运动在时空的各向异性（该表述是否合理）

If all spatiotemporal orientations are not equally likely, then there is no reason for us to treat space and time symmetrically, as is implicit in approaches to video recognition based on spatiotemporal convolutions.

We might instead "factor" the architecture to treat spatial structures and temporal events separately.

![Classification_SlowFast](/Classification_SlowFast.png)

![Classification_SlowFast2](/Classification_SlowFast2.png)

文中提到了单向的lateral connection和双向的效果差不多

每个视频用64帧，Slow用8帧，Fast用32帧，Fast的通道数是Slow的$1/\alpha=1/4$（有趣的是TSM用的双向shift也是1/4）

作者提到Slow在靠前的卷积层用时间卷积会降低分类性能；但是Fast则在每层都有时间卷积

lateral对Slow的提升有3个点，相对无lateral的slowfast有2.1个点；ResNet101的backbone比ResNet50有1个多点的提升



> Attention Clusters: Purely Attention Based Local Feature Integration for Video Classification

本文提出的一个观点是，时间信息对于常见的裁剪过的视频分类数据集来说并不是必须的（包括一些长时的动作分类）当然这个对于something数据集来说这样的假设是不成立的，但对于Kinetics数据集还算可以接受

该方法可参考Transformer模型，每个视频帧都利用fc层的输出进行表征，然后利用多个attention heads进行特征抽取再拼接

![Classification_AttentionClusters](/Classification_AttentionClusters.png)

单个attention无法表征时间的先后顺序，多头学习是否是对时间信息的一种补充？



> Lightweight Network Architecture for Real-Time Action Recognition 2019.05

总结了对视频序列分类的方法

1. 双流法
2. 3D
3. RNN
4. Transformer

In multi-head self-attention block, a temporal interrelationship between frames is modeled by informing each frame representation by representation of other frames using the attention mechanism

作者还尝试用RGB和RGB diff两种信息蒸馏出一个基于RGB的网络用于动作分类

We train and validate our models on 16-frame input sequences that are formed by sampling every second frame from the original video. 处理视频的帧率为2fps

transformer的特征维度为512，head 为8，提到了最佳的对应公式$d_{q}=d_{k}=d_{v} = \frac{512}{M}$



> Video Action Transformer Network 2019.05

动作识别需关注人与环境的交互

首先提取出人，基于I3D网络，利用RPN针对中间帧提取人物

We start by extracting a T-frame (typically 64) clip from the original video, encoding about 3 seconds of context around a given keyframe. 所以视频帧率大致为20fps

Our problem setup has a natural choice for the query (Q), key (K) and value (V) tensors: the person being classified is the query, and the clip around the person is the memory, projected into key and values.

![Classification_ActionTransformer](/Classification_ActionTransformer.png)

这里person的特征经过RoIPooling后还经过了一个query preprocessor（QPr），它是为了保留person的位置信息，转换为一维特征，之后就可以用Transformer结构对context特征进行attetion，再加回原来的person特征，作为信息的补充；transformer用到的特征维度为128

作者还对location information做了embedding，加入到context特征中，但这部分没有详细展开说明

在实验部分，作者说明了action tranformer适合动作识别，而原来的I3D head适合对人物进行box拟合



> Grouped Spatial-Temporal Aggregation for Efficient Action Recognition 2019.09

该文提出将通道维度进行spatial和temporal groups的分离，主要是针对依赖时间序列推理的动作类别

<img src="/Classification_GST.png" style="zoom:75%;" />

spatial支路各帧用2D卷积，temporal支路各帧用3D卷积

比较神奇是通过实验该文也建议用1/4的支路做temporal modelling，网络只在ImageNet上pretrain，Sth数据集性能超过TSM？

思考：双流法在单个网络结构下的实现



> Spatio-Temporal FAST 3D Convolutions for Human Action Recognition 2019.10

该文的思路是对3x3x3卷积在HWT三维平面在做有序分解，没有提及CoST的方法

![](/Classification_FAST.png)

分析部分对水平运动和竖直运动解释较清晰，但效果没有特别好



#### Semi-supervised

>  Tri-Training: Exploiting Unlabeled Data Using Three Classifiers 2005
>
>  周志华

co-training方法

新增一个分类器用于无标签样本的训练

In each round of tri-training, an unlabeled example is labeled for a classifier if the other two classifiers agree on the labeling, under certain conditions.

原本的co-training是从不同特征上去分类的： A prominent achievement in this area is  the co-training paradigm proposed by Blum and Mitchell, which trains two classifiers separately on two different views, i.e. two independent sets of attributes, and uses the predictions of each classifier on unlabeled examples to augment the training set of the other.

tri-training 则通过三个分类器之间的相互提升（1 vs 2）实现对无标记样本的纳入。



> Learning from Web Data with Memory Module 2019.06

处理两种噪声：label noise 和background noise

总体还是用MIL，对待label noise采用多图片，对待background noise采用proposals（利用edge box提取）

![](/Classification_webdata.png)

该方法采用了key memory的这种形式，但它做了一些调整

首先，它的memory module采用了聚类的方式获得，在计算各个instance与聚类中心的相似度时分成了两部分，一部分叫discriminative，另一部分叫representative。聚类采用SOM算法得到。训练时还采用了curriculum learning来稳定训练过程，先易后难。



#### Zero Shot

>  From Red Wine to Red Tomato: Composition with Context 2017

总体上还属于探索性的方向，方法比较生硬

![](/Classification_Compositionality.png)





#### Adaptive Inference

> BlockDrop: Dynamic Inference Paths in Residual Networks 

结合强化学习完成instance-specific inference；BlockDrop speeds up a ResNet-101 model on ImageNet by 20% while maintaining the same 76.4% top-1 accuracy.

涉及四个表述：layer dropping in resnet, model compression, conditional computation，early prediction

![Classification_BlockDrop](/Classification_BlockDrop.png)

提前预测是否还是鸡-蛋问题？

提到了课件学习



> SkipNet: Learning Dynamic Routing in Convolutional Networks

提到逐步训练的方式来确保算法work

类似量化学习的方式做预训练

To provide an effective supervised initialization procedure we introduce a form of supervised pre-training that combines hard-gating during the forward pass with soft-gating during backpropagation. We round the output gating probability of the skipping modules in the forward pass. During backpropagation we use the softmax approximation and compute the gradients with respect to softmax outputs.



> Convolutional Networks with Adaptive Inference Graphs

在网络的残差模块之前插入一个gate，we model the gates conditional on the output of the previous layer. 提到和注意力机制有关。

![Classification_CNNAIG](/Classification_CNNAIG.png)

关于Gumbel-Softmax的介绍<https://www.cnblogs.com/initial-h/p/9468974.html>



#### Self-supervised Learning

> A Simple Framework for Contrastive Learning of Visual Representations 2020.02

无监督学习目前进化到了通过对比的方式来学习表征（思考重建相对于对比是有什么劣势吗？）

本文提到了几个重要的发现：1.数据增强的方式很重要，一个几何变换和apperance变换的组合对特征学习很关键，两个近似的组合会是的模型在无监督上过拟合，而不同的变换组合虽然让对比学习的任务变难，但有助于提升表征的质量 2. 在表征和对比学习之间加入一个非线性变换对特征学习很重要，对比学习和下游分类任务是不相同的，这样会导致对比loss前的那一层特征变换到有利于对比学习的任务上，缺少了泛化能力，通过引入一层非线性变换，可以隔开与任务相关的特征提升表征的泛化性 3. 大batch和更长的训练，更大更深的网络都有助于对比学习

文中提到了Global BN对SimCLR的重要性，分布式训练时独立的BN会在策略上泄露正样本对导致潜在的过拟合，对特征学习有害

文中设计了多种对比学习的loss，经测试NT-Xent效果最好

Linear evaluation是指冻结特征层，对最后fc层进行训练

<img src="/Classification_SimCLR1.png" style="zoom: 67%;" />

<img src="/Classification_SimCLR2.png" alt="Classification_SimCLR2" style="zoom:67%;" />



### Recommendation

> From Zero-Shot Learning to Cold-Start Recommendation 2019.06

cold-start recommendation can be defined as a problem to generate recommendations for a fresh user where we have nothing about the user in the behavior space but some side information about the user in the attribute space.

作者应用zero-shot learning的思想，利用low-rank linear autoencoder，实现从attribute space到behavior space的重建

<img src="/Recommend_ZSL.png" style="zoom:80%;" />

关于推荐多样性的思考，如果单纯是重建的话，其实解还是唯一的。那么如果需要做一对多的映射，一种是否是加白噪声扰动（或者辅助属性，类似多样的风格化）；一种是bayesian网络。并且相对数据集上的评价，实际场景往往会是一个开放性问题。

<img src="/Recommend_ZSL2.png" style="zoom:75%;" />



### Arch

> Deep Layer Aggregation CVPR2018
>
>  Trevor Darrell

UNet++也是跟这个思路类似，即在长距离（低尺度）的skip connection之间加入升降尺度，甚至与Auto-DeepLab搜索出的结构有一样的启发

<img src="/Arch_DLA.png" style="zoom:75%;" />



#### Heterogeneous

> TernaryNet: faster deep model inference without GPUs for medical 3D segmentation using sparse and binary convolutions

利用只有$\{-1,0,1\}$三个取值的神经元构建网络，利用Hamming distance来进行卷积运算



#### Block

> Deformable Convolutional Networks 2017.06

DeformConv是在原有卷积的基础上增加了对位置偏移量的估计，这种方式和shift有些类似，对应的运动参考系不同。

<img src="/Block_DeformConv.png" style="zoom:75%;" />

As illustrated in Figure 2, the offsets are obtained by applying a convolutional layer over the same input feature map. The convolution kernel is of the sample spatial resolution and dilation as those of the current convolutional layer. 所以DeformConv是在网络层上并联了一个分支用于修改原本卷积的固定位置。

思考：类似SENet是更改了通道的权重，那对通道而言有没有更精细的方式呢？或者是否需要有什么样的约束可以理想实验上比较合理（比如不同位置的通道作用程度是不一样的，但是又要保持一点的spatial continuous）

特别要提下的是该文其实还提出了Deformable ROI pooling

<img src="/Block_DeformPooling.png" style="zoom:75%;" />

针对ROI offsets的预测稍微不太一样的是，它通过原本ROI pooling的特征图加上FC后预测得到normalized offsets，这个normalized offsets是相对ROI的高宽做过约束的。

另外值得一提的是，该文详细讨论了几种类似的思想，例如：

1. Spatial Transform Network 整个feature map的仿射变换
2. Active Convolution 各个位置共享偏移量，这就跟shift操作很像了
3. Effective Receptive Field DeformConv确实改变了网络的视场，**传统网络有效视场随深度是平方根增长，并非计算公式的线性增长**
4. Atrous Convolution 可以看出是DeformConv的特例
5. Dynamic Filter 卷积核权重随位置变换，并非采样位置本身；这里是否又可以引出图的概念，如何去定义affinity

接下来的问题便是，deformconv放哪里，提升了多少

**实验测试一般放在res5上，点数提升上对DeepLab和RPN帮助较大，对二阶的Faster RCNN不太明显**



> Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution 2019.04

考虑在卷积操作中区分低频和高频信息

related work 包括multigrid convolution, Xception， MobileNet， NAS， PNAS, AmoebaNet,也包括了ThiNet (prunes convolutional filters based on statistics computed from its next layer.) HetNet (replaces the vanilla convolution filters with heterogeneous convolution filters that are in different sizes.)

也和多尺度的一些网络结构有联系

在处理高频和低频表征时，是对通道进行了区分，利用$\alpha \in [0,1]$控制高低频的比例

<img src="/Block_OctConv.png" style="zoom:75%;" />

相对于factorize 卷积核，OctConv拆分了特征图，一部分是$h \times w$全尺寸对应高频，一部分是$h/2 \times w/2$半尺寸对应低频

**需要注意的是：除了高频对高频、低频对低频的独立输出外，卷积也对高低频之间的交互作了建模**，在高频到低频的过程中还对降尺度的方式进行了讨论（不是简单的用stride来处理， strided  convolution leads to misalignment）$\alpha$在实验中取值0.125



> Bilinear CNNs for Fine-grained Visual Recognition 2017.05

capture localized feature interactions in a translationally invariant manner, belong to the class of orderless texture representations

the Gram matrix is identical to a pooled bilinear representation when the two features are the same.

![Block_Bilinear](/Block_Bilinear.png)



> Factorized Bilinear Models for Image Recognition 2017.09

对原本的bilinear pooling进行矩阵分解，此时的bilinear pooling用在最后的全连接层之前，所指的pooling是对所有位置进行的sum pooling，以消除位置的信息

$\mathbf{z} = \sum_{i\in \mathbb{S}} \mathbf{x}_{i}\mathbf{x}_{i}^{T}$ where $\{\mathbf{x}_{i}| \mathbf{x}_{i} \in \mathbb{R}^{n}, i\in \mathbb{S}\}$ is the input feature map, $\mathbb{S}$ is the set of spatial locations in the feature map, $n$ is the dimension of each feature vector, and $\mathbf{z} \in \mathbb{R}^{n\times n}$ is the **global feature descriptor**. 这个描述在GCNet中也出现过，与位置无关的特征就可以表述为global feature?

Then a fully connected layer is appended as the final classification layer: $\mathbf{y} = \mathbf{b} + \mathbf{W} vec(\mathbf{z})$ where $vec(\cdot)$ is the vectorization operator which converts a matrix to a vector, $\mathbf{W} \in \mathbb{R}^{c\times n^{2}}$ and $\mathbf{b} \in \mathbb{R}^{c}$ are the weight and bias of the fully connected layer, respectively.  



> Temporal Bilinear Networks for Video Action Recognition 2018.11

Temporal Bilinear不同于Bilinear Pooling之处是没有sum pooling的操作，每个点处每个通道上进行一次内积运算，但如果是这样，是否和orderless texture representation的概念不符；如果是构建帧间关系，这样的二次项是否有必要（位置变换导致运算得到的特征图变化，平移性好像无法得到保证）？

对比Non Local，Non Local每个位置相当于一个样本，各个位置间计算相似性，运算维度为特征C；而Bilinear Pooling则是对Space进行运算压缩，得到平方维的特征

![Block_TemporalBilinear](/Block_TemporalBilinear.png)





> Collaborative Spatiotemporal Feature Learning for Video Action Recognition 2019.03

将视频看成是volume数据，在不同切面上进行2D卷积，CoST应用的是残差部分

![Classification_CoST1](/Classification_CoST1.png)

![Classification_CoST2](/Classification_CoST2.png)

![Classification_CoST3](/Classification_CoST3.png)

提到随着深度的增加，和T相关的attention部分权重在增加

we first sample 64 continuous frames from a video and then sub-sample one frame for every 8 frames.

测试时采用了10 clips



> All You Need is a Few Shifts: Designing Efficient Convolutional Neural Networks for Image Classification 2019.03

为了使shift操作可导，对偏移量进行双线性插值

The prerequisite of SSL is to ensure shift operation learnable. A common solution is to relax the displacement from integer to real-value and relax shift operation to bilinear interpolation so as to make it differentiable.

借用量化网络的思想，设计量化的shift，前向用整数化的displacement，反向用实数化的displacement

![Block_SparseShift](/Block_SparseShift.png)

利用loss中的正则控制shift的稀疏度

整体网络还使用了Fully-Exploited结构，将通道分组，部分通道直接进入下个stage

需要可以复用内存的深度学习框架，才能提高真实运行速度？



> GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond 2019.04

通过可视化方法看到了Non-local block的问题，从计算性能等角度出发进行了优化，并将SE block纳入进体系内

![Block_NLandSimplifiedNL](/Block_NLandSimplifiedNL.png)

![Block_GC](/Block_GC.png)

实验对比中得出add比multiplication的效果要好，两层的$1\times 1$卷积训练较困难

non-local中的双线性部分与精细分类中的应用关联度

GC block在Kinetics上较Non-local block优势基本在于减少计算量，是否在时序上值得进一步探讨问题所在？空间上各向同性，时间上各向异性？



> BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs 2019.07

将depthwise的kernel从$3 \times 3$扩大到$5\times 5$，将重叠的bbox进行加权平均

作者观察到pointwise卷积比depthwise要耗时，因此扩大depthwise的卷积计算耗时对整体影响不大

提到Pooling Pyramid Network中暗示一定分辨率之后的特征作用不大，把$8\times 8$之后的anchor都加到$8\times 8$这一层



> Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning 2016

Inception V1就引入了并行网络实现多尺度的概念 1x1,3x3,5x5，也有dimension reduction的概念

Inception V2 主要是降低计算量，采用更多的因式分解5x5变两个3x3，3x3变成1x3，3x1

Inception V3是发现辅助分类器再训练快结束时发挥的作用较大，同时使用了更大的卷积核并采用分解的方式降低参数量

Inception V4再stem中也加入了并联网络，对每个尺度设计了专门的降尺度模块，并且尝试了res结构



### Understanding

> MovieGraphs: Towards Understanding Human-Centric Situations from Videos

We use graphs to decribe this behavior because graphs are more structured than natural language, and allow us to easily ground information in videos

用到人脸检测，对话文本，利用RNN完成交互、推理



> Video Relationship Reasoning using Gated Spatio-Temporal Energy Graph 2019.03

利用CRF完成多标签推理的过程？引入时序的关系

#### Forecasting

> Peeking into the Future: Predicting Future Person Activities and Locations in Videos 2019.02

整合多种工作以完成更高层次的视觉任务

![Forecasting_ActivityAndPath](/Forecasting_ActivityAndPath.png)

![Forecasting_PersonBehavior](/Forecasting_PersonBehavior.png)

![Forecasting_PersonInteraction](/Forecasting_PersonInteraction.png)

数据集url<https://actev.nist.gov/>（需翻墙）



### Fashion

> DeepFashion2: A Versatile Benchmark for Detection, Pose Estimation, Segmentation and Re-Identification of Clothing Images

The annotations of DeepFasion2 are much larger than its counterparts such as $8\times$ of FashionAI Global Challenge.

DeepFashion2  contains 491K images of 13 popular clothing categories.

![Fashion_Deep2](/Fashion_Deep2.png)

![Fashion_MatchRCNN](/Fashion_MatchRCNN.png)



### Face

> ArcFace: Additive Angular Margin Loss for Deep Face Recognition 2019.02

softmax loss:

$L_{1} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{e^{W_{y_{i}}^{T} x_{i} + b_{y_{i}}}}{ \sum_{j=1}^{n} e^{W_{j}^{T} x_{i} + b_{j}}}$

把$W_{j}^{T} x_{i}$看出向量内积$\|W_{j}\| \|x_{i}\| \cos \theta_{j}$，然后$\|W_{j} \|$固定为1， $\| x_{i} \|$则在L2正则化后重新rescale到$s$

ArcFace在此基础上在角度方向加了一个margin，可以理解为正确的类别加了一个偏置量，依然要求其角度与类别中心最接近，以此达到既要类内紧凑、又要类间差异的要求

$L_{3} = -\frac{1}{N} \sum_{i=1}{N} \log\frac{e^{s(\cos(\theta_{y_{i}}+m))}}{e^{s(\cos(\theta_{y_{i}}+m))} + \sum_{j=1,j\neq y_{i}}^{n} e^{s\cos\theta_{j}}}$

![Face_ArcFace](/Face_ArcFace.png)



> Look at Boundary: A Boundary-Aware Face Alignment Algorithm 2018.05

该文主要针对遮挡、形变等情况利用边缘检测的方法来提高人脸关键点landmark定位准确性。

*It is easier to identify facial boundaries comparing to facial landmarks under large pose and occlusion.* **如何可以断定这样的结论？多变量的联合分布比独立（或者也并非独立只是不相关）分布更容易拟合？**

文中利用stacked hourglass来预测边界的热力图，通过GAN来判断边界预测是否合理，需要注意的是其通过关键点构造边界的方式，以及其中介绍的message passing实现（Structured feature learning for pose estimation）

![](/Face_Boundary.png)





> Large Scale 3D Morphable Models 2018

标准脸加现有脸的线性组合

![](/Face_3DMM.png)



> A Morphable Model For The Synthesis Of 3D Faces

New faces and expressions can be modeled by forming linear combinations of the prototypes



> Large Pose 3D Face Reconstruction from a Single Image via Direct Volumetric CNN Regression

VRN



### Pose

> Photo Wake-Up: 3D Character Animation from a Single Photo 2018.12

将2D人体变成3D人体：包含了分割、姿态估计、3D模型拟合的技术

人体检测、分割：Mask RCNN

pose estimation： Convolutional Pose Machines

人体精细分割：Dense CRF

纹理填充：PatchMatch

人体3D拟合： SMPL: a skinned multi-person linear model, Keep it SMPL

动作序列：CMU Graphics Lab Motion Capture Database

![](/Pose_Wakeup.png)



> Everybody Dance Now 2019.08

通过人体关键点和人脸精细仿真实现视频级舞蹈动作迁移

Our work is made possible by recent rapid advances along two separate directions: robust  pose estimation, and realistic image-to-image translation.

姿态估计： OpenPose

图像生成：pix2pixHD， CoGAN， CycleGAN, DiscoGan, Cascaded Refinement Network, unsupervised image-to-image translation

针对视频级别的动作生成，采用了连续两帧的生成用于平滑

![](/Pose_Dance.png)

这里有个问题是每个目标任务都需要训练一个网络，同时其需要目标有足够的动作可供训练

类似技术 MoCoGAN（分离动作和表象）

对人脸进行了细化

![](/Pose_Dance2.png)






### Image Synthesis

> High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs 2018.8

<https://github.com/NVIDIA/pix2pixHD>

尺度1024x2048

采用了coarse-to-fine的策略，现在512x1024上生成。Loss方面不仅用GAN loss，还在discriminator的中间层进行feature matching。针对instance引入edge map使得边界上有较强的先验信息。

![GAN_pix2pixHD](/GAN_pix2pixHD.png)



### Image Processing

> Replacing Mobile Camera ISP with a Single Deep Learning Model 2020.02

该文利用华为P20拍摄的raw data和佳能拍摄的图片进行配对，通过神经网络实现ISP功能

该文工作可分为两个部分：1. 训练配对数据构建，由于手机拍摄和相机拍摄存在位置的偏差（很可能还有时差），因此其提出用匹配的方式构建训练对数据。匹配的工作也是该文作者之前的构造。主要通过SIFT关键点和RANSAC算法实现。

2. 作者提出了一个金字塔结构的模型，在每个层次上施加不同目标的损失，最终使得图像增强后的效果接近佳能相机

<img src="/ImageProcessing_ISP.png" style="zoom:75%;" />

在高尺度level4-5上施加global colorand brightness/gamma correction，在中尺度level2-3上施加VGG perceptual和MSE loss，在level 1上施加VGG perceptual、SSIM、MSE三种loss




### OCR

> Attention-based Extraction of Structured Information from Street View Imagery

84.2% on FSNS (four views)

提到了attention机制的permutation invariant，为了使attention和位置相关，本文增加了空间位置的编码信息；

本文的saliency map是基于的反向梯度，attention map则是基于的前向计算



> ASTER: An Attentional Scene Text Recognizer with Flexible Rectification 2018

OCR的识别方法

包含校正网络和识别网络两个部分，能够处理旋转或者弯曲文字

<img src="/OCR_ASTER.png" style="zoom:75%;" />

特点是校正网络的加入，包括对control points的预测以及利用Thin-Plate-Spline对图像进行变换

<img src="/OCR_ASTER_rectification.png" style="zoom:75%;" />

类似Spatial Transformer Network，通过双线性插值的方式来得到校正后各点的插值

识别网络用LSTM+Attention的方式替代CTC方法



> Tightness-aware Evaluation Protocol for Scene Text Detection 2019.04

提出了基于紧凑性的文字检测指标，对原本IoU做二值的recall [0,1]的进一步修正。在对ICDAR2015做评价时提到了先根据word-level annotation生成text-line annotation，以便处理one-to-many或者many-to-one的情况

总结处提到用于训练或者半监督学习（目前看做训练有点扯）



> Self-organized Text Detection with Minimal Post-processing via Border Learning 2017

行文本检测，提出将文字边界作为一类，利用FCN做三类的分割

自制数据集的方法：利用PPT生成行文本训练数据集

对文本长宽占比等指标进行分析



> Graph Convolution for Multimodal Information Extraction from Visually Rich Documents 2019.03

视觉富文本使用结构推理的前提是单纯的文本识别无法准确判别实体，需要视觉和布局信息来确定实体。

模板解决不了的原因是，模板本身一旦数量级上去了，会导致系统无法扩展，另一个是图像扭曲、模糊或者被干扰，造成匹配错误





### Visualization/Interpretability

> Understanding Intra-Class Knowledge Inside CNN 2015.07

本文尝试可视化FC层，提及parametric visualizaton model的问题是低层视觉重建容易产生多个全局最优解而导致色彩分布与真实自然色彩分布不一致，提出用建一个自然图像的patch库来帮助重建的图像进行色彩拟合



> Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization 2019.12

基于梯度求取CAM，Grad-CAM is a strict generalization of CAM

原本CAM需要FC层的权重来作为响应图的权重，Grad-CAM则是通过反向求导的梯度来替代，并且用CAM的特例来进行了证明

$\alpha_{k}^{c}=\frac{1}{Z} \sum_{i}\sum_{j} \frac{\partial y^{c}}{\partial A_{ij}^{k}}$

$L_{Grad-CAM}^{c} = ReLU(\sum_{k} \alpha_{k}^{c} A^{k})$

**思考：该文展示了FC层的权重对应着反向梯度传播时各神经元接受的权重（好像本来就是这个逻辑），是否意味着反向传播其实也可以通过预测的方式来进行**



> Network Dissection: Quantifying Interpretability of Deep Visual Representations 2017.04

通过响应的样本统计来确定各个neuron的concept，这里比较有趣的点是interpretability与网络的discriminativity无关，也就是判断能力不能代表可解释性；另一个是网络的concept是axis-aligned，



### Attention

#### Hard

> Recurrent Models of Visual Attention

模拟人眼运动 One important property of human perception is that one does not tend to process a whole scene in its entirety at once.

利用glimpse sensor获取多个尺度的信息，从而引入context以便location network对下一次位置的移动作出估计

训练过程是减少loss的期望（强化学习的方式）

![Attention_Recurrent](/Attention_Recurrent.png)



#### Soft

> Pyramid Attention Network for Semantic Segmentation 2018.05

our FCN baseline lacks ability to make prediction on small parts

造成小物体分割不理想的原因：1. 是否是一种样本不均衡的问题？即small parts在数据集中的占比偏少，导致网络无法给予足够的关注度（与loss的设计也有关系）； 2. 网络结构本身的原因，由于网络单一的前向过程，对于不同尺度的part而言并不是合理的

In ASPP module dilated convolution is a kind of sparse calculation which may cause grid artifacts.

Some kind of U-shape networks, such as SegNet, RefineNet, Tiramisu and Large Kernel Matters perform complicate decoder module wich use low-level information to help high-level features recover images detail. However, they as time comsuming.

尽管低层、高层特征融合成为分割一个关键构成，但如何有效的利用高低特征依然有探索的价值？

However, most methods attempt to combine the features of adjacent stages to enhance low-level features, without consideration of their diverse representation and the glocal context information.

在作者看来，使用高低特征时依然需要对它们进行重校正（类似SENet）；问题：高低特征拼接时是否会破坏流形或者不利于生成判别性的流形

作者认为多尺度的特征融合依然缺乏global context prior attention，而引入的SENet的channel-wise attention 又会导致缺乏pixel-wise information.

We consider that the main character of decoder module is to repair category pixel localization. Furthermore, high-level features with abundant category information can be used to weight low-level information to select precise resolution details.

![Attention_GlobalAttentionUpsample](/Attention_GlobalAttentionUpsample.png)

Global Attention Upsample 与Attention Unet的Attention Gate具有明显的不同，Attention Gate对空间位置上的特征响应进行了调整，而Global Attention Upsample则是对各个特征（concept）的权重进行了调整；channel attention与spatial attention的区别？

参考文章<https://blog.csdn.net/u010142666/article/details/80994439>

对于Feature Pyramid Attention，**类似于PSPNet、DeepLab采用空间金字塔pooling实现不同的尺度以及多孔金字塔池化ASPP结构，问题一：pooling容易丢失掉局部信息，问题二：ASPP因为是一种稀疏的操作会造成棋盘伪影效应，问题三：只是简单地多个scale concat缺乏上下文的信息，没有关注上下文信息情况下效果不佳（下图作图为现有的方法），该部分处理主要是用在处理高层特征上的操作**

**在提取到高层特征之后不再进行pooling的操作，而是通过三个继续的卷积实现更高层的语义，我们知道更高层的语义会更加接近ground truth的情况，会关注一些物体信息，所以用更高层的语义来作为一种Attention的指导，与高层特征做完1×1卷积不变化大小的情况下进行相乘，也就是加强了具有物体信息的部位带有的权值，得到了带有Attention的输出，同时因为金字塔卷积的结构采用不同大小的卷积核，代表着不同的感受野，所以也解决不同物体不同scale的问题。**



对于Global Attention Upsample，**采用了解码器decoder也就是反卷积之类再加上底层的特征，一层层地往上累加以便恢复图像细节，论文中讲到了这种虽然是可以实现底层和高层的结合以及图像重构，但是computation burden**，这里指的是cancat之后的卷积操作

**抛弃了decoder的结构，原始形式是直接用底层特征加FPA得到的高层特征，但在skip底层特征的时候论文采用了高层特征作为指导设置了相应的权重，使得底层与高层的权重保持一致性，高层特征采用了Global Pooling得到权重，底层特征经过一个卷积层实现与高层特征相同数量的map，然后相乘后再高底层相加。这样减少了decoder复杂的计算同时也是一种新的高底层融合的形式**



> LEARN TO PAY ATTENTION 2018.04

提到了attention可以分为post hoc的方式和learnable的方式，基于梯度的和周博磊的CAM属于post hoc。并且CAM improved localisation performance comes at the cost of classification accuracy.

Learnable的方式又分为hard和soft，而Spatial Transformer Networks则介于两者之间。It uses a prameterised transform to estimate hard attention on the input image deterministically, where the parameters of the image transformation are estimated using differentiable functions.(这里的技巧是将输出响应与位置的偏移量联系起来，这样在求导时map的坐标位置也成了输入量求偏导)



> Exploring Self-attention for Image Recognition CVPR2020

<img src="/Attention_selfattention.png" style="zoom:75%;" />

除了常见的pairwise attention，作者提出了patchwise attention的概念。相对于pairwise的加权组合，patchwise引入了patch中的竞争，更像是high order的建模



### Style

> Style Transfer for Headshot Portraits

处理过程：1. 对齐人脸； 2. 多尺度的方式迁移局部特征；3. 对眼睛和背景处理

检测人脸landmark，对example进行调整（morph the example to the input using the segments on the face template），再用SIFT Flow进行微调

所用的多尺度技术为DoG



### Domain Adaption

> Taking A Closer Look at Domain Shift: Category-level Adversaries for Semantics Consistent Domain Adaptation 2019.04

对分割情况下的domain adaption，提出应关注类别层面的适配。文中采用了co-training的策略，同时用两个分类器来实现对数据分布界面的标定，使得adaption过程对**高阶结构性**有所关注（前提是存在多个等价的（次优）分类界面，通过平均后仍可得到一个最优界面）。

利用不同视角下的不一致性，驱动局部的类别层面做相应的调整。这里采用的是cosine夹角作为度量

<img src="/Adaption_CLAN2.png" style="zoom:75%;" />

![](/Adaption_CLAN.png)

思考：这种二分类器的形式能否用在其他地方，作为两个参考面的话。



### Distillation

> Structured Knowledge Distillation for Semantic Segmentation 2019.03

在pixel-wise distillation的基础上，加入了pair-wise和holistic distillation

其中pair-wise用的是特征图的gram matrix，holistic distillation用的是GAN的判别器

整体上loss还是很多的，可操行有点复杂



### NAS

> Auto-Keras: An Efficient Neural Architecture Search System

NAS的问题是将原本网络结构设计时的一些超参进行简化，同时在优化算法上应用某些规则来简化求解的难度

由于大量的时间消耗在训练过程，即是在其定义的三步（update、generation、observation）的observation中，这就需要利用之前相似的已训练的网络层参数来作初始化（这是前提假设）

作者通过编辑距离和高斯过程来对网络的搜索过程进行参数化，文中提到了树状结构搜索中的$A^{\star}$算法

文中定义了四种操作，加层deep，加滤波器数量wide，加求和操作skip，加拼接操作concatenate



> NEURAL ARCHITECTURE SEARCH WITH REINFORCEMENT LEARNING 2017.02

该文使用controller的概念制造每层网络参数用于构成网络，利用RL的方法将准确率作为reward来学习controller的参数。controller是一个RNN网络，每次按序生成各个层的参数

![](/NAS_RL.png)

针对单纯的卷积网络提出了skip connection的预测方案，在原本预测序列中插入一个动作点，预测与之前N-1层是否需要连接

训练采用分布式，这里采用的是一般的采样估计

![](/NAS_RL2.png)

文中还介绍了用NAS的方式去生成RNN的cell结构，以区别于LSTM，GRU

**该方法针对单一网络层结构内的参数选择和网络连接点选择**，可以**思考如何去模仿作者利用现有技术构造理想实验到构建具体的实践**



> Learning Transferable Architectures for Scalable Image Recognition 2018.04

该文在NAS的基础上，放弃了对整体网络结构的搜索，转而去构建新的Cell，通过复用新的block去实现新任务网络的搭建

针对尺度不变和尺度下降两种Cell进行搜索，依然采用NAS的方法

![](/NAS_NASNet.png)

![](/NAS_NASNet2.png)

需要注意的是，**每个Cell中一个block需重复B次，在B次的生成过程中，上一次输出状态需要加入进选择域**

作者提到RL比random search略优，random search时不以RNN输出的概率结果进行采样，而是进行均匀采样

在训练过程中还对cell中的连接进行ScheduledDropPath



> COMPUTATION REALLOCATION FOR OBJECT DETECTION 2019.12

该文是对object detection的backbone上各个stage里的卷积层进行重新的分配，包括引入dilated conv从而改善effective receptive field

<img src="/NAS_CR.png" style="zoom:75%;" />

<img src="/NAS_CR2.png" style="zoom:75%;" />



### Bayesian

>BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning 2019.06

In AL, the formativeness of new points is assessed by an acquisition function. There are a number of intuitive choices, such as model uncertainty and mutual information.

BALD is based on mutual information and scores points based on how well their label would inform us about the true model parameter distribution.

要实现对模型参数的预测，就需要引入bayesian的方法，从后验概率来处理

In deep learning models, we generally treat the parameters as point estimates instead of distributions. However, Bayesian neural networks have become a powerful alternative to traditional neural networks and do provide a distribution over their parameters.

贝叶斯方法把传统的权值预测变为分布预测

Naively finding the best batch to acquire requires enumerating all possible subsets within the available data, which is intractable as the number of potential subsets grows exponentially with the acquisition size b and the size of available points to choose from. Instead, we develop a greedy algorithm that selects a batch in linear time, and show that it is at worst a $1-1/e$ approximation to the optimal choice for our acquisition function.

文章对bayesian active learning的数学定义值得参考

从数学上定义清楚问题：

BALD uses an acquisition function that estimates the mutual information between the model predictions and the model parameters. 

$I(y;w|x,D_{train}) = H(y|x, D_{train}) - E_{p(w|D_{train})}[H(y|x,w,D_{train})]$

in deep learning, retraining takes a substantial amount of time.

期望的求取就涉及到Monte-Carlo方法

局限性：

Unbalanced datasets. 如果测试集是不均匀的，该方法表现欠佳。这个是本身分布就不同的问题

Unlabelled data. BatchBALD does not take into account any information from the unlabelled dataset. Semi-supervised learning could improve these estimates by  providing more information about the underlying structure of the feature space. 也就是diversity的问题还是存在的

Noisy estimator. 蒙特卡罗的变分近似会引入噪声



### Misc

> Varifocal-Net: A Chromosome Classification Approach Using Deep Convolutional Networks

利用Global net完成一次染色体的核型分类和极性分类，输出染色体的大致位置，染色体的位置由三元参数表示，用于确定左上角，右下角和相对外框的长宽比例

在截取局部图像用于Local net时，设计了可导的boxcar，由于染色体真实的位置是未知的，所以先用粗略的估计方式训练Gnet的定位分支，再利用预训练的定位分支结合Lnet完成端到端的训练

![Misc_VarifocalNet](/Misc_VarifocalNet.png)

