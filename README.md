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



> Panoptic Feature Pyramid Networks

在mask rcnn的基础上并行一路semantic segmentaion，实现panoptic segmentation。

针对重叠的情况，制定了三个规则：不同instance的重叠部分利用confidence score来区分，instance与semantic的重叠部分有限选择instance，semantic 低于某个阈值的需要剔除

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



- Semantic Human Matting 2018.9

  文中提到了语义分割与matting所要实现的细节分割是不一样的

  the matting module is highly nonlinear and trained to focus on structural patterns of details, thus the semantic information from input hardly retains.

  本文也是设计了两个网络来分别完整大致的语义分割和精细的matting，需要通过pretrain来预热网络，之后再进行端到端的训练。Loss方面除了alpha prediction loss和合成loss(compositional loss)外，还增加了decomposition loss

  ![Segmentation_HumanMatting](/Segmentation_HumanMatting.png)



> FickleNet: Weakly and Semi-supervised Semantic Image Segmentation using Stochastic Inference 2019.03

弱监督分割

FickleNet can be regarded as a generalization of dilated convolution.

利用擦除局部区域的方式来判断区域是否属于某一类；利用多尺度的dilated convolution来生成CAM；利用判别性区域的像素点来训练一个像素级别的语义相似性估计网络AffinityNet

![Segmentation_FickleNet](/Segmentation_FickleNet.png)

We do not drop the center of the kernel of each sliding window position, so that relationships between kernel center and other locations in each stride can be found. 中心点类似于参考点，保持其不变应该是为了稳定训练过程。

弱监督能否去学习语义边界？

#### Video

> Video Object Segmentation and Tracking: A Survey 2019

比较普通的总结，列举了一些数据集




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



####Video

> TACNet: Transition-Aware Context Network for Spatio-Temporal Action Detection 2019.05

本文主要是检测边界的调优，对ACT的改进，加入Conv-LSTM

![Detection_TACNet](/Detection_TACNet.png)

The standard SSD performs action detection from multiple feature maps with different scales in the spatial level, but it does not consider temporal context information. To extract temporal context, we embed Bi-ConvLSTM unit into SSD framework to design a recurrent detector.

有个巧妙的处理是将两个有耦合（inter-coupled）的任务通过加减处理来解耦合

16帧输入



###NLP

> Attention Is All You Need 2017.12

去掉RNN，只利用attention和fc完成ecode-decode过程

![NLP_Transformer](/NLP_Transformer.png)

![NLP_Transformer2](/NLP_Transformer2.png)

注意力公式$Attention (Q,K,V) = softmax(\frac{QK^{T}}{\sqrt{d_{k}}})V$

其中的scaling factor $\frac{1}{\sqrt{d_{k}}}$是为了对抗维度增加带来的负面影响（We suspect that for large values of $d_{k}$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients.）如何理解？

假设几个输出$a_{1}, a_{2}, a_{3}$，其中$a_{2}$最大，其中维度的作用也起着影响，softmax函数会突出相对特别大的那个数值，所以当$a_{2}$有维度加成后，会变得更大，由softmax的导数可知：

$softmax^{'} = \left\{\begin{array}{\\} S_{i}(1-S_{j}), & i=j \\ -S_{j}S_{i}, & i\neq j\end{array} \right.$

此时softmax在此处的导数接近0，其他处也因为$S_{j}$过小而为0

Multi-head的关系式$d_{k}=d_{v}=d_{model}/h=64, \, h=8$

对序列的positional encoding





####Caption

> From Captions to Visual Concepts and Back cvpr2015

利用多示例学习MIL生成word detectors，再依据这些words来生成最合理的句子

用的是Noisy-OR MIL，用一个global threshold得到图片可用的word

![Caption_VisualConcept](/Caption_VisualConcept.png)



> Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering 2017.07

预先提取图像中有意义的区域，相较于单纯soft attention驱动的特征选择，这种方式优势在于：1.属于hard attention，2.引入中间过程的任务学习，相对来说信息维度更加的丰富

整体是从图像侧改进

![Caption_TopDown](/Caption_TopDown.png)

下层完成注意力的特征选择，上层是单层的语言模型



> Stack-Captioning: Coarse-to-Fine Learning for Image Captioning 2017.9

多级LSTM对生成的caption进行finetune，问题是前后语句存在对齐的问题，虽然是想解决单层LSTM caption能力弱的问题，但是对齐是否也会限制其性能提升？

用到了多层的training loss来解决梯度消失的问题

训练的细节：先用adam $4\times 10^{-4}$, momumten 0.9，CE loss; 再用adam $5\times 10^{-5}$, RL微调

![Caption_StackCaption](/Caption_StackCaption.png)



> Discriminability objective for training descriptive captions 2018

对文本描述提出了descriptive的要求，可用于text retrieval或者image retrevial；discriminative体现在生成的语句应该返回去能够检索到相应的图片，从而让caption的过程去关注图片上相对有差异的部分

利用 “Vse++: Improved visual-semantic embeddings”实现对image&caption embedding的学习，再利用SCST结合Cider来训练；图像的特征基于Visual Genome来完成



####VQA

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



####Video

> What Actions are Needed for Understanding Human Actions in Videos? 2017.08

探讨分析动作分类的一些问题：

长尾效应，小样本的类别对整体性能提升

定位人对分类性能有帮助



> Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset  2017.5

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

每个视频用64帧，Slow用8帧，Fast用32帧，Fast的通道数是Slow的$1/\alpha=1/4​$（有趣的是TSM用的双向shift也是1/4）

作者提到Slow在靠前的卷积层用时间卷积会降低分类性能；但是Fast则在每层都有时间卷积

lateral对Slow的提升有3个点，相对无lateral的slowfast有2.1个点；ResNet101的backbone比ResNet50有1个多点的提升



> Attention Clusters: Purely Attention Based Local Feature Integration for Video Classification

本文提出的一个观点是，时间信息对于常见的裁剪过的视频分类数据集来说并不是必须的（包括一些长时的动作分类）当然这个对于something数据集来说这样的假设是不成立的，但对于Kinetics数据集还算可以接受

该方法可参考Transformer模型，每个视频帧都利用fc层的输出进行表征，然后利用多个attention heads进行特征抽取再拼接

![Classification_AttentionClusters](/Classification_AttentionClusters.png)

单个attention无法表征时间的先后顺序，多头学习是否是对时间信息的一种补充？



>Lightweight Network Architecture for Real-Time Action Recognition 2019.05

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





### Block

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

需要可以复用内存的深度学习框架？



> GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond 2019.04

通过可视化方法看到了Non-local block的问题，从计算性能等角度出发进行了优化，并将SE block纳入进体系内

![Block_NLandSimplifiedNL](/Block_NLandSimplifiedNL.png)

![Block_GC](/Block_GC.png)

实验对比中得出add比multiplication的效果要好，两层的$1\times 1$卷积训练较困难

non-local中的双线性部分与精细分类中的应用关联度

GC block在Kinetics上较Non-local block优势基本在于减少计算量，是否在时序上值得进一步探讨问题所在？空间上各向同性，时间上各向异性？



###Understanding

> MovieGraphs: Towards Understanding Human-Centric Situations from Videos

We use graphs to decribe this behavior because graphs are more structured than natural language, and allow us to easily ground information in videos

用到人脸检测，对话文本，利用RNN完成交互、推理

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





### Visualization

> Understanding Intra-Class Knowledge Inside CNN 2015.07

本文尝试可视化FC层，提及parametric visualizaton model的问题是低层视觉重建容易产生多个全局最优解而导致色彩分布与真实自然色彩分布不一致，提出用建一个自然图像的patch库来帮助重建的图像进行色彩拟合





### Attention

####Hard

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

We consider that the main character of decoder module is to repair category pixel localization. Furthermore, high-level features with abundant category information can be used to wieght low-level information to select precise resolution details.

![Attention_GlobalAttentionUpsample](/Attention_GlobalAttentionUpsample.png)

Global Attention Upsample 与Attention Unet的Attention Gate具有明显的不同，Attention Gate对空间位置上的特征响应进行了调整，而Global Attention Upsample则是对各个特征（concept）的权重进行了调整；channel attention与spatial attention的区别？



> LEARN TO PAY ATTENTION 2018.04

提到了attention可以分为post hoc的方式和learnable的方式，基于梯度的和周博磊的CAM属于post hoc。并且CAM improved localisation performance comes at the cost of classification accuracy.

Learnable的方式又分为hard和soft，而Spatial Transformer Networks则介于两者之间。It uses a prameterised transform to estimate hard attention on the input image deterministically, where the parameters of the image transformation are estimated using differentiable functions.(这里的技巧是将输出响应与位置的偏移量联系起来，这样在求导时map的坐标位置也成了输入量求偏导)



### Style

> Style Transfer for Headshot Portraits

处理过程：1. 对齐人脸； 2. 多尺度的方式迁移局部特征；3. 对眼睛和背景处理

检测人脸landmark，对example进行调整（morph the example to the input using the segments on the face template），再用SIFT Flow进行微调

所用的多尺度技术为DoG



### Heterogeneous Network

> TernaryNet: faster deep model inference without GPUs for medical 3D segmentation using sparse and binary convolutions

利用只有$\{-1,0,1\}$三个取值的神经元构建网络，利用Hamming distance来进行卷积运算



### Misc

> Varifocal-Net: A Chromosome Classification Approach Using Deep Convolutional Networks

利用Global net完成一次染色体的核型分类和极性分类，输出染色体的大致位置，染色体的位置由三元参数表示，用于确定左上角，右下角和相对外框的长宽比例

在截取局部图像用于Local net时，设计了可导的boxcar，由于染色体真实的位置是未知的，所以先用粗略的估计方式训练Gnet的定位分支，再利用预训练的定位分支结合Lnet完成端到端的训练

![Misc_VarifocalNet](/Misc_VarifocalNet.png)

