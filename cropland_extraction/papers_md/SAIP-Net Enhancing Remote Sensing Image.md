SAIP-Net: Enhancing Remote Sensing Image
Segmentation via Spectral Adaptive Information
Propagation

Zhongtao Wang, Xizhe Cao, Yisong Chen∗, Guoping Wang

School of Computer Science, Peking University, No.5 Yiheyuan Road, Haidian
District, Beijing, 100871, P.R. China

Abstract

Semantic segmentation of remote sensing imagery demands precise spatial boundaries

and robust intra-class consistency, challenging conventional hierarchical models. To
address limitations arising from spatial domain feature fusion and insufficient receptive

fields, this paper introduces SAIP-Net, a novel frequency-aware segmentation frame-

work that leverages Spectral Adaptive Information Propagation. SAIP-Net employs
adaptive frequency filtering and multi-scale receptive field enhancement to effectively

suppress intra-class feature inconsistencies and sharpen boundary lines. Comprehen-

sive experiments demonstrate significant performance improvements over state-of-the-
art methods, highlighting the effectiveness of spectral-adaptive strategies combined

with expanded receptive fields for remote sensing image segmentation. Our code is
available at https://github.com/ZhongtaoWang/SAIP-Net.

Keywords: Remote Sensing, Semantic Segmentation

1. Introduction

Semantic segmentation of remote sensing images is a fundamental yet challeng-

ing task with profound implications in numerous critical applications, such as land-use

∗Corresponding author.
Email addresses: wangzhongtao@stu.pku.edu.cn (Zhongtao Wang), chenyisong@pku.edu.cn

(Yisong Chen)

Figure 1: Overview of the challenges and motivations behind SAIP-Net. Remote sensing images often ex-
hibit large intra-class variance, small inter-class differences, and irregular class layouts. To address these

challenges, SAIP-Net uses the combination of four modules to improve intra-class consistency and enhance

boundary accuracy, significantly improving the segmentation of complex structures in remote sensing im-

ages.

classification, urban planning, agriculture monitoring, environmental assessment, and

disaster management. Despite significant advances achieved through deep learning

techniques, remote sensing imagery continues to pose distinct challenges that conven-

tional hierarchical deep neural networks struggle to address adequately. Such chal-

lenges include diverse and complex textural features, significant variations in object

scales, intricate boundaries, and the coexistence of highly structured urban areas along-

side irregular natural landscapes.

Existing segmentation approaches typically rely on hierarchical feature fusion meth-

ods operating primarily within the spatial domain. These conventional fusion methods

often directly combine multi-scale feature representations obtained from various hier-

archical levels through simple addition or concatenation. However, this spatial-domain

fusion frequently leads to significant intra-class inconsistencies, primarily due to dis-

turbances caused by unregulated high-frequency information. Moreover, such methods

are typically hindered by boundary ambiguities and inaccuracies arising from blurred
details and insufficient contextual information captured by limited receptive fields.

To overcome these limitations, this paper introduces SAIP-Net, an innovative frequency-

aware segmentation framework explicitly designed for remote sensing image analysis.

At the core of our method is the novel concept of Spectral Adaptive Information Prop-

2

RemoteSensing ImagesOurs SAIP-NetBaselineMulti-ScaleTransformerEncoderSAIP-Netc•Smallinter-classdifference•Irregularclasslayouts•Large intra-class variance•…Learnable High-Pass Filter StemComposite Dilated Convolution LayerSpectral Adaptive Feature FusionModuleagation, a mechanism that dynamically manipulates frequency-domain characteristics

of image features. As shown in Figure 1, SAIP-Net systematically employs adaptive

spectral filtering techniques to mitigate disruptive high-frequency noise within seman-

tic object regions, thereby significantly enhancing intra-class consistency. Simulta-

neously, our method incorporates frequency-aware processes to recover and sharpen

high-frequency boundary details typically diminished during conventional downsam-

pling and upsampling operations. Beyond frequency-domain enhancements, SAIP-Net

also introduces strategies to substantially enlarge the receptive fields of feature extrac-

tion modules. Enlarged receptive fields facilitate more comprehensive contextual un-

derstanding by capturing extensive spatial interactions and dependencies, crucial for

correctly identifying semantic classes at varying scales. By integrating frequency-

domain processing and spatial contextual awareness, SAIP-Net significantly improves

both feature representation consistency and boundary delineation accuracy.

In conclusion, our contributions can be summarized as:

• We introduced SAIP-Net, a frequency-aware segmentation framework that in-

corporates the novel Spectral Adaptive Information Propagation mechanism to

address the inherent challenges of remote sensing image analysis.

• By adaptively propagating spectral information and enlarging the receptive fields,
our approach effectively eliminates disruptive high-frequency information within

intra-class regions while preserving essential frequency-domain features along

class boundaries, which results in substantially improved intra-class consistency

and enhanced boundary accuracy.

• Experimental results demonstrate that our proposed method significantly outper-

forms the baseline in semantic segmentation task on remote sensing images.

2. Related Work

Semantic segmentation assigns semantic labels to each pixel and is widely applied

in both natural and remote sensing images. While the two domains share foundational

3

techniques, their data characteristics and task constraints differ significantly. Natu-

ral image segmentation focuses on perspective-view scenes with diverse objects and

consistent contexts [41], while remote sensing imagery features large-scale, top-down

views with high intra-class variability, inter-class similarity, which requires domain-

specific solutions [42].

2.1. Semantic Segmentation in Natural Images

Early methods, such as thresholding and region growing, were limited by image

quality and computational constraints. The advent of deep learning, particularly Con-

volutional Neural Networks (CNNs) [3] and Fully Convolutional Networks (FCNs)

[41], enabled end-to-end pixel-level prediction, forming the basis for encoder-decoder

architectures like U-Net [35] and SegNet [36]. These models extract high-level seman-

tic features while preserving spatial resolution.

Subsequent research addressed challenges such as class imbalance, small object de-

tection, and complex backgrounds. Dilated convolutions [8] expanded receptive fields,

while PSPNet [9] introduced pyramid pooling to aggregate multi-scale contextual in-

formation. Attention mechanisms, such as Coordinate Attention [7], further refine

segmentation, especially for small or complex objects. Additionally, unsupervised and

semi-supervised approaches [4, 5] have been explored to reduce reliance on large an-

notated datasets.

2.2. Semantic Segmentation in Remote Sensing Images

Remote sensing segmentation faces unique challenges, including large-scale scenes,

spectral diversity, and subtle intra-class variation. While FCN-based methods [11, 12,

13, 14, 15, 16] remain prevalent, additional modules have been developed to enhance

spatial and contextual feature extraction. Techniques like channel attention [17] and

region-aware methods, such as S-RA-FCN [18] and HMANet [19], address feature

redundancy and spatial dependencies. Edge-aware strategies, like Edge-FCN [6], im-

prove boundary precision by incorporating edge detection.

Recently, Transformer-based models [20, 21, 22, 23, 24, 25] have gained attention

for their ability to model long-range dependencies via self-attention. For instance,

4

RSSFormer [21] introduces adaptive feature fusion to reduce background noise and

enhance object saliency. However, Transformers still face challenges in capturing fine

local details and accurately delineating object boundaries.

2.3. Frequency-domain Methods for Semantic Segmentation

Frequency-domain feature processing has emerged as a promising direction in se-

mantic segmentation. Shan et al. decomposed images via the Fourier transform into

high and low frequency components [26], which capture edge and body information

respectively, ensuring object consistency and edge supervision through deep fusion.

Zhang et al. proposed FsaNet [27], leveraging a frequency self-attention mechanism on

low-frequency components to drastically reduce computational costs. Chen et al. intro-

duced Frequency-Adaptive Dilated Convolution (FADC) [28], dynamically adjusting
dilation rates by local frequency patterns and enhancing feature bandwidth/receptive

fields via adaptive kernels and frequency selection. Their FreqFusion [1] further im-

proves boundary sharpness and feature coherence for dense prediction tasks. Ma et

al. developed AFANet [29], optimizing semantic structures with a CLIP-guided spatial

adapter to boost weakly-supervised few-shot segmentation. These advances underscore

the transformative potential of frequency-aware strategies in semantic segmentation.

Building on these works, our work proposes a novel approach named SAIP-Net.

By adaptively propagating spectral information and enlarging the receptive fields, our

approach substantially improved intra-class consistency and enhanced boundary accu-

racy, thereby significantly improving the segmentation of complex structures in remote

sensing images.

3. Methods

Figure 2 illustrates the overall architecture of our proposed SAIP-Net, which in-

tegrates several novel components to address the challenges of remote sensing image

segmentation. Our method combines a transformer encoder, a frequency-aware de-

coder equipped with Spectral Adaptive Feature Fusion (SAFF) module and Composite

Dilated Convolution (CDC) Layers, and a Learnable High-Pass Filter Stem (LhpfStem)

5

Figure 2: Overview of the proposed SAIP-Net architecture. The network combines: a Learnable High-

Pass Filter Stem to enhance edge details, a Transformer Encoder that extracts global context via multi-stage

learnable pooling, and a Frequency-Aware Decoder that fuses high- and low-level features using SAFF mod-

ules, while integrated Composite Dilated Convolution Layers expand the receptive field. These modules

ultimately lead to result in improved intra-class consistency and enhanced boundary accuracy, thereby sig-

nificantly improving the segmentation of complex structures in remote sensing images.

into a unified framework. The encoder efficiently captures global context through

multi-stage feature extraction and learnable pooling. Then the decoder fuses high-

and low-level features via the Spectral Adaptive Feature Fusion module to enhance

intra-class consistency and boundary precision. The CDC layers in decoder leverage

parallel dilated convolutions with varying receptive fields to extract multi-scale spa-

tial details. The LhpfStem suppresses background interference by emphasizing high-

frequency edge information at the very beginning of the network. Together, these com-
ponents enable SAIP-Net to effectively integrate spatial and frequency domain infor-

mation, resulting in improved segmentation performance under the complex conditions

typical of remote sensing imagery.

3.1. Transformer Encoder

Our transformer encoder is designed to balance global context modeling and com-
putational efficiency via pooling attention and hierarchical feature extraction. The en-

coder consists of 4 stages. Following [2], each stage of our encoder includes multi-

ple Transformer blocks, and spatial downsampling between stages is achieved through

6

LearnableHigh-PassFilterStemInput ImagesSegmentationOutputTransformerEncoderFrequency-AwareDecoderCDCSAFFCDCSAFFCDCSAFFSAFFOutput LayerTTTTTTransformer Encoder BlocksCDCSpectralAdaptiveFeatureFusionModuleSAFFComposite Dilated Convolution Layerlearnable pooling operations.

Patch Embedding. The encoder starts by applying a strided convolution (patch em-

shape H

bedding layer) on the input image of size H × W × 3, generating token embeddings of
p × C. This effectively partitions the image into non-overlapping patches
and projects each to a C-dimensional embedding. A smaller patch size p preserves

p × W

local details crucial for dense prediction.

Transformer Encoder Blocks. Each block follows the standard structure with multi-

head self-attention and MLP, each preceded by LayerNorm and followed by residual

connections. We use pooling attention to reduce spatial resolution within attention
computation. Specifically, input Xin ∈ RL×D is projected and pooled as:

Q = PQ(XinWQ), K = PK(XinWK), V = PV (XinWV ),

(1)

where P· denotes a learnable pooling operator, reducing the length from L to ˜L.

We also incorporate relative positional embeddings to maintain spatial awareness

and employ a residual connection on the query branch. The output at the l-th stage Xl

can be calculated as:

Xl = Attn(Q, K, V) + Q = Softmax

(cid:33)

(cid:32)

QK⊤ + E(rel)
√
D

V + Q

(2)

where E(rel)i j = Qi · Rp(i), p( j). Here, Qi represents the query vector at token posi-

tion i, and Rp(i),p( j) denotes the learnable relative positional embedding between tokens

i and j, where p(i) and p( j) map token indices to their spatial locations. The inner

product Qi · Rp(i),p( j) injects relative position bias into the attention logits. To reduce the

number of learnable parameters, we follow [2] and decompose Rp(i),p( j) along spatial

dimensions as:

Rp(i),p( j) = Rh

h(i),h( j)

+ Rw

w(i),w( j),

(3)

where Rh and Rw are independent learnable embeddings along the height and width

axes, and h(i), w(i) refer to the row and column indices of token i. This decomposition
reduces the embedding complexity from O(HW) to O(H + W).

7

Figure 3: (a).
Illustration of Spectral Adaptive Feature Fusion (SAFF) module. The module integrates
high-level and low-level features using content guided low-pass and high-pass filters alongside spatial offset

estimation. The structure of F HP is enlarged at the bottom while F LP shares a similar architecture. (b).

Our module enhances intra-class consistency and boundary accuracy at the feature level, leading to better

segmentation results.

Hierarchical Structure. Like most multi-scale image encoders, our encoder employs

hierarchical pooling across stages. The spatial size is reduced, and the channel and
4 × W

16 × 4C and

8 × 2C, H

4 × C, H

16 × W

8 × W

dimension at each stage are increased at H
32 × W

H

32 × 8C.
This design enables the encoder to capture multi-scale semantics with reduced com-

plexity, making it well-suited for high-resolution remote sensing image segmentation.

3.2. Spectral Adaptive Feature Fusion Module

To fully utilize the multi-scale features extracted by the encoder, we propose a

Spectral Adaptive Feature Fusion Module (SAFF Module). This module is guided

by high- and low-level features respectively, leverages frequency-domain priors to im-

prove feature alignment and semantic consistency during upsampling, and address the

challenges of intra-class inconsistency and boundary degradation.

8

LP+HPLPHPR+ResampleGuidanceReshape&Norm.SpatiallyVaryingKernelsGuidanceInputOutputConv×ConvInputSeg.FeatureBaselineSeg.BaselineFeature(a)(b)Inspired by [1, 30, 31], incorporating a guidance signal (e.g., low-level features)

provides valuable local structural priors. These priors help direct the filtering process

such that the spatially varying kernels can adapt based on local content, ensuring that

essential details like edges and textures are accurately preserved while undesired noise

is suppressed. Additionally, guided filters that vary at each position allow fine-grained

feature reassembly and precise spatial alignment, ensuring that the fused features ac-

curately reflect both the global context and the local details. Such flexibility directly

contributes to improved overall segmentation quality and enhanced boundary delin-

eation.

As illustrated in Figure 3 (a), our SAFF module begins with an initial fusion step

that integrates complementary high-level and low-level features. First, the low-level

feature Xl is enhanced by a content guided high-pass filter F HP guided by Xl itself

that emphasizes edge details, with a residual connection preserving the original infor-
mation. In parallel, the high-level feature Yl+1 is processed through a content guided

low-pass filter F LP guided by Xl to suppress noisy high-frequency components and up-

sampled to match the resolution of the low-level feature. Then these processed features

are fused via element-wise addition to produce the initial fused feature map Zl:

˜Yl+1 = F LP(Xl, Yl+1),

˜Xl = F HP(Xl, Xl) + Xl, Zl = ˜Yl+1 + ˜Xl.

(4)

After this, a second fusion is performed by refining the alignment between high-

In particular, we rely on Zl to perform an additional
level and low-level features.
content guided low-pass filtering on Yl+1 and content guided high-pass filtering on Xl:

ˆXl = F HP(Zl, Xl) + Xl,

ˆYl+1 = F LP(Zl, Yl+1).

(5)

Specifically, an offset generator predicts spatial offsets (cid:0)u(i, j), v(i, j)(cid:1) based on local
feature similarities. These offsets are used to resample the upsampled, low-pass filtered
high-level feature ˆYl+1, aligning it with the high-pass enhanced low-level feature ˆXl.

The secondly fused feature is then obtained by element-wise addition:

u(i, j), v(i, j) = OG(Zl)i j, Yl
i, j

= ˆYl+1

i+u(i, j), j+v(i, j)

+ ˆXl

i, j.

(6)

9

As shown in Figure 3 (b), this process ensures that the feature map benefits from both

the smooth global context and the detailed local structure, leading to improved segmen-
tation quality. It also effectively combines the smooth, context-rich high-level features

with the detail-preserving low-level features, laying a solid foundation for improved

segmentation performance.

The key components of this module are two adaptive filter: Content Guided Low-
Pass/High-Pass Filter F LP/F HP and an offset generator OG. We elaborate on each

component as follows:

Content Guided Low-Pass/High-Pass Filter. Inspired by [30, 31], we take Z as the

guidance feature to predict the spatially variant filter weights filtering feature map X.

The filter consists of several steps:

1. Filter Prediction. A 3 × 3 convolution predicts spatially variant filters ¯Vl from Z

and normalizes them using softmax. The softmax operation imposes a low-pass

behavior on the convolutional kernel by concentrating weights on nearby po-

sitions and suppressing distant responses, thus favoring smooth, low-frequency

features. For the high-pass filter, it is then subtracted from an identity kernel E

to obtain high-pass behavior:



(cid:0)Z(cid:1),

¯V = Conv3×3

exp( ¯Vp,q
i, j )
p,q∈Ω exp( ¯Vp,q
i, j )
exp( ˆVp,q
i, j )
p,q∈Ω exp( ˆVp,q
i, j )
where Ω is the K × K convolution window (e.g., K = 3).

¯Wp,q
i, j




E −

=

(cid:80)

(cid:80)

,

(low-pass)

(7)

,

(high-pass)

2. Spatially Variant Convolution. The high-level feature Yl+1 is convolved with

the predicted weights:

Yg
i, j

=

(cid:88)

p,q∈Ω

¯Wg,p,q

i, j Xi+p, j+q,

(8)

where channels are grouped (g = 1, . . . , 4) following sub-pixel convolution pat-

terns.

3. Upsampling of High-Level Features.

If the input is high-level features, the
four groups Yg are rearranged using Pixel Shuffle[32] to obtain a smooth 2×
upsampled feature map Y ∈ RC×2H×2W .

10

Offset Generator.. For further refinement, especially near complex boundaries, the off-

set generator OG(Z) estimates spatial displacements for further resampling step under

the guidance of input feature map Z :

1. Local Similarity Computation. Cosine similarity is computed between a pixel

and its neighbors:

(cid:80)
c(Zc,i, j)2 (cid:112)(cid:80)
2. Offset Prediction. Concatenate Z and S, and apply two convolutions to obtain

c Zc,i, j Zc,i+p, j+q

c(Zc,i+p, j+q)2

Sp,q
i, j

(cid:112)(cid:80)

(9)

=

,

directional offset D and magnitude A:

D = Conv3×3([Z, S]), A = σ(cid:0)Conv3×3([Z, S])(cid:1),

(10)

Where [...] denotes the concat operation and σ denotes the sigmoid function.
Finally, the offset is given by O = D · A.

3. Spatial Resampling. As in Equation 6, high-level features ˜Y are resampled
using the offsets O to correct class-wise inconsistencies and refine boundaries.

3.3. Composite Dilated Convolution Layer

To expand the receptive field at low cost, we design a Composite Dilated Convolu-

tion Layer that processes features in a multi-scale manner. As shown in Figure 4, this

framework is built upon a cascade of operations that first integrates channel informa-

tion, then diversifies spatial context via parallel dilated convolutions, and finally refines

the combined feature representation.

Given an input feature map Y l, which fuses high-level semantic information from

deeper layers, we begin with a lightweight channel integration. This is achieved using

a 1 × 1 convolution that acts as a channel-mixer, blending the channel information

without altering the spatial dimensions. This preliminary mixing primes the feature

map for subsequent multi-scale processing.

The integrated feature x is then partitioned equally into D sub-tensors, denoted as

x1 . . . xD. Each sub-tensor is independently processed by a dilated convolution opera-

tion with a specific dilation rate di. The receptive field size of each branch is calculated

11

Figure 4: Composite Dilated Convolution Framework: After a channel-mixing operation, we split the feature
map into multiple parts processed with different dilation rates, and the resulting multi-scale features are

concatenated, refined, and upsampled to reconstruct high-resolution segmentation maps.

as ri = di(k − 1) + 1, where larger k for larger receptive fields and smaller k for smaller

ones. The individual outputs are given by:

yi = Convri(xi),

i ∈ {1 . . . D}.

(11)

These outputs are then concatenated to form a multi-scale feature representation. Then

the concatenated features are further refined through a merging module. This module

employs a series of convolutions—typically a combination of 1 × 1 and 3 × 3 ker-

nels—each followed by Batch Normalization and ReLU activation. This post-mixing

step not only fuses the diverse scale information but also introduces non-linearity to en-

hance the discriminative capability of the feature representation. The entire Composite

Dilated Convolution operation can be summarized as:

FCDC(Y l) = Conv

(cid:16)(cid:104)

y1 . . . yD

(cid:105)(cid:17)

.

(12)

To restore the original spatial resolution, the output of the Composite Dilated Con-

volution is passed to an upsampling block. Through extensive tuning and experiments,

12

Dilated Conv Kernels…………1×1 ConvSplittoDpartsMerge1×1ConvConvr1Convr2ConvrDH×W×CH×W×CH×W×H×W×H×W×CD partsD partsH×W×CFigure 5: (a). Overview of the proposed LhpfStem. The module is composed of a stack of Lhpf layers. In

a Lhpf layer, the high-pass output is computed by subtracting the low-pass response features from the input,

thereby enhancing edge details and fine structures. (b). Our module extracts key high-frequency features,

sharpens feature boundaries, and improves segmentation performance.

we found that setting D = 3 provides an optimal balance between effectively capturing

diverse contextual information and maintaining a reasonable model complexity.

3.4. Learnable High-Pass Filter Stem

Remote sensing images feature intricate high-frequency details (e.g., boundaries)

alongside low-frequency noise like clouds and shadows, requiring frequency-domain

guidance during feature extraction. The SAFF module (Section 3.2) depends on well-

formed feature maps, but the raw 3-channel RGB input at the network’s top resides

in a shallow, low-dimensional space which lacks the rich, abstract features present in

deeper latent representations. So we need to extract useful information for reliably

generating dynamic convolution kernels.

To address this, we propose the Learnable High-Pass Filter Stem (LhpfStem),

a dedicated module positioned at the residual connection of the network’s top layer.

It employs fixed, learnable convolution kernels that are optimized during training to

perform high-pass filtering directly on the raw input at low cost.

13

Hamming Window××FeatureMapH×𝑊×3ImageLhpfLayerLhpfLayerConv LayerLhpfLayerConv LayerLearnable High-Pass FilterStemLearnable WeightSoftmaxLearnable High-Pass FilterLayerG.T.InputSeg.FeatureBaselineSeg.BaselineFeature(a)(b)As shown in Figure 5 (a), our LhpfStem is designed as a stack of Learnable High-

Pass Filter Layers (Lhpf Layers) and interspersed with convolutional layers. In each
Lhpf layer, given the input image or feature map X ∈ RC×H×W , the module uses a

learnable weight Wl of shape (C, 1, K2) to perform adaptive high-pass filtering. The

weights are firstly normalized using a softmax over the kernel dimension. A hamming
window H ∈ RK×K is then applied to modulate these weights, and the result is re-

normalized:

˜W(p, q) = Softmax(Wl)(p, q) · H(p, q)
p′,q′∈Ω ¯W(p′, q′) · H(p′, q′)

(cid:80)

,

(13)

where Ω denotes the set of kernel indices. The modulated kernel ˜W is reshaped to

a convolution kernel of size (C, 1, K, K) and used in a depthwise convolution on x to

obtain the low-pass filtered output:

˜X(i, j) =

(cid:88)

p,q∈Ω

X(i + p, j + q) ˜W(p, q).

(14)

Finally, the high-pass filtered output is computed by subtracting the low-pass response

from the original input:

HighPass(x)(i, j) = X(i, j) − ˜X(i, j).

(15)

As shown in Figure 5 (b), this design not only enhances edge details and preserves

fine structures but also provides robust frequency guidance to subsequent layers. Con-
sequently, LhpfStem effectively overcomes the limitations of using raw images for

adaptive filtering, ensuring that critical high-frequency information is retained and en-

hancing the overall segmentation performance in complex remote sensing scenarios.

The LhpfStem offers a significant low-cost advantage compared to transformer-

based models or deeper convolutional networks. By relying on fixed, learnable convo-

lution kernels and depthwise convolutions for high-pass filtering, the LhpfStem avoids

the extensive computational overhead and large memory footprint typically associated

with those alternative architectures.

To generate the final segmentation prediction, we fuse Xhp = LhpfStem(x) with the

top-level features from the decoder Y 1 into Yout via a SAFF module. Finally, an output

layer is applied to Yout to produce the segmentation logits, and a softmax activation

converts these logits into pixel-wise class probabilities.

14

3.5. Loss Functions

To train SAIP-Net, we adopt a hybrid loss that combines Cross-Entropy (CE) and

Dice[33] losses, balancing pixel-wise classification and region-level overlap. The total

loss is defined as:

Ltotal = λ


−



1
N

N(cid:88)

C(cid:88)

i=1

c=1



yi,c log(ˆyi,c)

+ (1 − λ)

(cid:32)

1 −

(cid:80)

i,c yi,c ˆyi,c

2 (cid:80)
i,c yi,c + (cid:80)

i,c ˆyi,c

(cid:33)

,

(16)

where N is the number of pixels, C the number of classes, yi,c the ground truth, and ˆyi,c

the predicted probability. This formulation promotes both accurate classification and

spatial consistency.

4. Experiments

4.1. Implementation Details

Dataset Selection.. To evaluate our model, we conduct experiments on two widely

used remote sensing datasets: Potsdam1 and LoveDA[34]. All results are reported on
the test/validation set of LoveDA and the test set of Potsdam, which are strictly held

out during training and model selection to avoid data leakage.

Settings.. Our model is trained with an initial learning rate of 0.00006, utilizing a

warm-up and polynomial decay strategy to adjust the learning rate during training. The
loss weight balancing factor is set to λ = 0.5. A batch size of 16 is employed to

balance memory consumption and training speed. The Adam optimizer is used with

momentum parameters set to 0.9 and 0.999. All experiments are performed on a server

equipped with an NVIDIA RTX 4090 GPU.

4.2. Evaluation Metrics.

We compare our method against several baseline approaches with consistent or
larger parameter magnitudes, including UNet [35], SegNet [36], DeepLabV3+ [11],

DANet [37], LANet [38], FFPNet [39], TransUNet [24], UperNet RSP-Swin-T [40],

1https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx

15

Table 1: Comparison of model efficiency and computational cost. SAIP-Net achieves a favorable balance

between performance and complexity, with reduced computational cost, particularly in GFLOPs (1024×1024

input).

Method

Params(MB)↓ GFLOPs↓

TransUnet

DeepLabV3+

AerialFormer-T

Our SAIP-Net

90.7

12.47

42.7

37.21

233.7

216.85

192.76

176.8

RSSFormer [21], FactSeg [15], UNetFormer [23], [10], LSKNet[43] , DecoupleNet
[44] and LOGCAN++ [45]. As shown in Table 1, our model achieves comparable or

even lower parameter counts and GFLOPs (1024 × 1024 input) compared to baseline
methods by adopting a more efficient backbone, a lightweight upsampling strategy, an
effective receptive field expansion mechanism, and a compact stem with fewer param-

eters.

For clarity and focus, Table 1 reports comparisons with a subset of representative

methods, including those with strong overall performance and diverse architectural de-

signs. Notably, AerialFormer-T has demonstrated state-of-the-art performance across

multiple benchmarks, making it a strong reference for evaluating comprehensive ef-

fectiveness. Thus, we include AerialFormer and several other key baselines in the
parameter and GFLOPs comparison to highlight the efficiency of our design without

loss of generality.

We assess model performance using standard metrics: mean Intersection over Union

(mIoU), Overall Accuracy (OA), and mean F1 score (mF1).

4.3. Test Results

Figure 6 presents segmentation results on the LoveDA dataset, which includes high-

resolution images of both urban and rural scenes with complex semantics. Our method

is evaluated on both the validation and test sets of LoveDA, with detailed quantitative

results provided in Table 2 and Table 3, respectively. The results demonstrate that our
model consistently achieves competitive performance across different classes. Visual

comparisons further highlight the robustness of our approach in handling diverse object

16

Figure 6: Visual comparison of segmentation results on LoveDA dataset. The regions enclosed by pink el-

lipses indicate areas with improved intra-class consistency, and the areas marked by orange ellipses highlight

enhanced boundary accuracy.

appearances and fine-grained details inherent to remote sensing imagery.

The Potsdam dataset features diverse urban scenes captured under various condi-

tions. As summarized in Table 4, our method achieves the highest mIoU, OA, and mF1

scores, demonstrating its capability to handle the complexities of urban remote sensing

imagery. Notably, as illustrated in Figure 7, our approach shows superior performance

in capturing fine segmentation details and accurately delineating class boundaries.

Extensive experimental evaluations, including comprehensive qualitative and quan-

17

InputImageGroundTruthDeeplabV3+AerialFormerOursFigure 7: Visual comparison of segmentation results on Potsdam dataset. The regions enclosed by pink el-

lipses indicate areas with improved intra-class consistency, and the areas marked by orange ellipses highlight

enhanced boundary accuracy.

titative analyses conducted on diverse remote sensing datasets, validate the superiority

of SAIP-Net over baseline methods. Our method outperforms baseline approaches
with comparable model complexity, demonstrating superior efficiency and effective-

ness. The results clearly demonstrate the profound benefits and potential of integrating

spectral-adaptive information propagation strategies.

18

InputImageGroundTruthDeeplabV3+AerialFormerOursTable 2: Quantitative results on the official validation set of LoveDA. Bold numbers indicate the best perfor-

mance in each column.

Method

mIoU ↑

Bkg.

Building Road Water Barren Forest Agri.

UNet

TransUNet
DeepLabV3+

SegNet

47.33

53.95

50.56

53.18

50.15

51.75

48.34

53.02

AerialFormer-T

52.57

53.80

Our SAIP-Net

54.46

55.36

56.13

62.37

59.92

54.53

64.02

64.53

51.26

69.15

19.75

29.00

52.10

52.27

69.06

27.20

34.61

55.19

53.76

68.35

26.60

39.14

51.53

55.42

69.08

55.61

66.90

16.37

32.83

35.67

54.31

45.37

49.46

57.66

71.77

28.90

43.53

59.44

Table 3: Quantitative results on the official test set of LoveDA. Bold numbers indicate the best performance

in each column.

Method

mIoU ↑

Bkg.

Building Road Water Barren Forest Agri.

UNet

TransUNet
DeepLabV3+

SegNet

FactSeg

UNetFormer

RSSFormer-B

AerialFormer-T

LSKNet-T

LSKNet-S

DecoupleNet-D2
LOGCAN++

47.84

43.06

48.90

43.00

49.30

44.20

46.70

41.20

48.9

52.4

52.4

52.0

53.2

54.0

53.1

52.0

42.6

44.7

52.4

46.4

46.7

45.3

47.37

Our SAIP-Net

53.30

46.32

4.4. Analysis of Failure Cases

52.74

56.10

55.00

51.20

53.6

58.8

60.7

52.78

73.08

10.33

43.05

59.87

53.70

78.00

9.30

44.90

56.90

54.20

77.10

10.50

45.30

61.00

50.70

72.00

52.8

54.9

76.9

79.6

9.80

16.2

20.1

42.10

58.30

42.9

46.0

57.5

62.5

55.21

76.29

18.73

45.39

58.33

45.21

57.84

56.46

79.63

19.20

46.12

59.53

59.5

59.9

59.5

58.38

58.64

57.1

58.3

56.3

79.9

80.2

80.6

21.8

24.6

20.9

46.6

46.4

46.2

61.4

61.8

63.1

56.46

80.05

18.44

47.91

64.80

59.65

81.42

15.38

47.33

64.33

Although SAIP-Net achieves consistent gains on most categories, we observe a

noticeable drop for Barren on LoveDA dataset.

Data and appearance factors.. As shown in Figure 8, compared with other categories,
Barren in LoveDA is intrinsically ambiguous under RGB due to low texture/contrast

19

Table 4: Quantitative results on the Potsdam dataset. Bold numbers indicate the best performance in each

column.

Method

mIoU ↑ OA ↑ mF1 ↑

F1 Score per Class ↑

ImpSurf. Build. LowVeg.

Tree

Car

DeepLabV3+

DANet

LANet

FFPNet

81.69

89.60

89.79

87.28

89.72

89.14

86.20

90.84

91.95

86.50

91.10

92.44

UNetFormer

86.80

91.31

92.89

UperNet RSP-Swin-T

73.50

90.78

90.03

AerialFormer-T
LOGCAN++

88.51

93.72

93.55

78.58

85.34

86.62

Our SAIP-Net

88.88

93.79

93.97

92.27

91.61

93.05

93.61

93.62

92.65

94.60

87.51

94.84

95.52

96.44

97.19

96.70

97.24

96.35

97.78

93.76

97.66

85.71

86.11

87.30

87.31

87.73

86.02

90.87

77.20

90.85

86.04

89.42

88.04

83.54

88.04

94.19

88.11

96.46

88.91

96.52

85.39

89.75

88.40

96.94

79.78

93.13

89.49

96.99

Figure 8: Analysis of failure cases on LoveDA Validation Set. (a). Class-wise normalized confusion matrix,

showing that the largest confusion involves the Barren class, which is frequently predicted as agricultural or

background. (b). The Barren class (purple) illustrates typical errors such as boundary confusion with nearby

man-made edges and “interior erosion” in low-texture regions.

and appearance proximity to dry agriculture or background compounded by boundary-

level annotation uncertainty. These conditions amplify class ambiguity and increase

annotation noise near transitions. Together with the relatively smaller frequency of

20

(a)(b)Input ImageGround TruthOurPredictionBarren ClassBarren pixels, the class becomes under-represented and more susceptible to bias during

optimization.

Frequency bias induced by our design.. SAIP-Net explicitly enhances high-frequency

structures via LhpfStem and HP-guided paths in SAFF. While this helps boundaries,

Barren regions are often low-texture, low-contrast. The HP-centric pathway may down-

weight interior evidence in such regions, causing “interior erosion”. In addition, the

LP branch in SAFF, normalized by softmax, favors smooth responses but is guided

by features dominated by man-made edges in mixed scenes; this can pull Barren to-
ward adjacent classes during the second-stage offset-based alignment. Finally, CDC

enlarges the receptive field with dilations; when texture is scarce, multi-scale context

may inadvertently absorb surrounding semantics, increasing contextual leakage into

Barren.

Our design prioritizes boundary accuracy and intra-class consistency at controlled

complexity, thus we deliberately avoid class-specific heavy treatments (e.g., large reweight-

ing or post-processing), which leads to a moderate drop on this particular class. We
also note a plausible side effect of our spectral priors: high-pass–oriented pathways

can slightly erode interiors of textureless regions, and large receptive fields may absorb

surrounding context. We view handling such low-texture, look-alike regions as a com-

plementary future direction (e.g., class-aware calibration and mid-band constraints or
multimodal cues), aiming to improve Barren without compromising efficiency.

4.5. Ablation Study

To thoroughly investigate the effectiveness of each critical component in our pro-

posed SAIP-Net, we conduct a detailed ablation study on Potsdam dataset and LoveDA

validation set. For consistency, we use the same training configurations and evaluation

metrics described in Section 4.1 and Section 4.2. Each ablation configuration is trained

and tested independently.

As shown in Table 5, progressively adding these modules into baselines improves

segmentation performance across all metrics. Qualitative comparison of ablation is

shown in Figure 9. We can see that the combination of the SAFF module, CDC lay-

21

Figure 9: Qualitative comparison of ablation variants. The regions enclosed by pink ellipses indicate areas

with improved intra-class consistency, and the areas marked by orange ellipses highlight enhanced boundary

accuracy.

ers, and LhpfStem collaboratively enhances boundary quality, class consistency, multi-

scale feature fusion, and edge information representation.

We also ablate the proposed SAFF module to isolate its contribution to robust-
ness. Specifically, we compare the full SAIP-Net against (i) an ablated variant w/o

SAFF that removes the module while keeping all other components, training protocol,

and hyperparameters unchanged, and (ii) the baseline AerialFormer. To probe robust-
ness under common corruptions, we add synthetic Gaussian noise (prob=0.5, σ=10),
salt-and-pepper noise (prob=0.5, ratio= 0.01), and speckle noise (prob=0.5, σ=0.1)

at validation time. We report the drop in mIoU relative to the clean setting (lower is

better).

As summarized in Table 6, SAIP-Net consistently exhibits the smallest perfor-

mance degradation across all noise types. Relative to the baseline, SAIP-Net reduces

22

w/o SAFFw/o CDC Layerw/oLhpfStemInput ImageGround TruthFull ModelTable 5: Ablation Study Results on the Potsdam dataset and LoveDA validation set. We use DeepLabV3+

as the baseline model, progressively replacing its components with our proposed modules. The symbol
“✓” indicates the inclusion of the corresponding component, while “–” denotes its absence. T.E indicates

transformer encoder.

Components

mIoU↑

T.E.

SAFF CDC LhpfStem LoveDA Potsdam

–

✓

✓

✓

✓

–

–

✓

✓

✓

–

–

–

✓

✓

–

–

–

–

✓

50.15

52.85

53.16

54.07

54.46

81.69

87.76

88.07

88.34

88.88

the mIoU drop by 1.87 (Gaussian), 1.50 (Speckle), and 2.32 (S&P). Compared to the
w/o SAFF ablation, SAIP-Net further narrows the drop by 0.28, 0.33, and 0.74 mIoU

under Gaussian, Speckle, and S&P noise, respectively. These gains align with SAFF’s

design goal of suppressing disruptive high-frequency components while preserving

task-relevant structures.

Table 6: Performance drop in mIoU (↓) under synthetic corruptions at validation time. Values denote the
decrease relative to the clean setting (lower is better). We compare SAIP-Net (Ours), its ablation w/o SAFF,

and the AerialFormer baseline. Bold numbers indicate the best performance in each row.

Noise

Ours w/o SAFF Baseline

Gaussian

Speckle

S&P

0.56

0.11

2.47

0.84

0.44

3.21

2.43

1.61

4.79

Additionally, we conduct ablation experiments on the hyperparameter D of the

CDC Layer using the LoveDA validation set. As shown in Table 7, the experimental
results indicate that D = 3 achieves the best performance. Therefore, we adopt D = 3

in our final model.

Our summarized ablation results clearly show the incremental performance im-

23

Table 7: Ablation Study on the Effect of D in CDC Layer on Model Performance. We conducted experiments
on the LoveDA validation set and found that setting D = 3 yields the best performance.
The Value of D D = 2 D = 3 D = 4 D = 5

mIoU↑

54.04

54.46

54.14

53.58

provement as each component is introduced. Furthermore, qualitative visualizations of

segmentation maps illustrate specific improvements in boundary sharpness and intra-

class consistency brought about by the combination of proposed modules. Through

these experiments, we reinforce the rationale behind the design of SAIP-Net.

5. Conclusion

In this paper, we introduced SAIP-Net: Enhancing Remote Sensing Image Seg-

mentation via Spectral Adaptive Information Propagation. By adaptively propagating
spectral information and enlarging the receptive fields, our approach effectively elimi-

nates disruptive high-frequency information within intra-class regions while preserving

essential frequency-domain features along class boundaries. Experimental results con-

firm that our design significantly improves intra-class consistency and enhances bound-

ary accuracy, thereby substantially improving the segmentation of complex structures

in remote sensing images. A remaining limitation is reduced performance on inherently

ambiguous, low-texture categories under RGB-only inputs (e.g., Barren in LoveDA

dataset). As future work, we will explore methods to better disambiguate such regions

without increasing complexity.

References

[1] Chen, Linwei and Fu, Ying and Gu, Lin and Yan, Chenggang and Harada, Tatsuya

and Huang, Gao, Frequency-aware Feature Fusion for Dense Image Prediction,

IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 1, no. 1,

pp. 1–18, 2024.

24

[2] Li, Yanghao and Wu, Chao-Yuan and Fan, Haoqi and Mangalam, Karttikeya and

Xiong, Bo and Malik, Jitendra and Feichtenhofer, Christoph, MViTv2: Improved

multiscale vision transformers for classification and detection, Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.

[3] LeCun, Yann and Bottou, Léon and Bengio, Yoshua and Haffner, Patrick,

Gradient-based learning applied to document recognition, Proceedings of the

IEEE, vol. 86, no. 11, pp. 2278–2324, 1998.

[4] Ahn, Jiwoon and Cho, Sunghyun and Kwak, Suha, Weakly supervised learning
of instance segmentation with inter-pixel relations, Proceedings of the IEEE/CVF

Conference on Computer Vision and Pattern Recognition, pp. 2209–2218, 2019.

[5] Lin, Ci-Siang and Wang, Chien-Yi and Wang, Yu-Chiang Frank and Chen, Min-

Hung, SemPLeS: Semantic Prompt Learning for Weakly-Supervised Semantic

Segmentation, arXiv preprint arXiv:2401.11791, 2024.

[6] He, Chu and Li, Shenglin and Xiong, Dehui and Fang, Peizhang and Liao, Ming-

sheng, Remote sensing image semantic segmentation based on edge information

guidance, Remote Sensing, vol. 12, no. 9, p. 1501, 2020.

[7] Hou, Qibin and Zhou, Daquan and Feng, Jiashi, Coordinate attention for efficient
mobile network design, Proceedings of the IEEE/CVF Conference on Computer

Vision and Pattern Recognition, pp. 13713–13722, 2021.

[8] Li, Yuhong and Zhang, Xiaofan and Chen, Deming, Csrnet: Dilated convolu-

tional neural networks for understanding the highly congested scenes, Proceed-

ings of the IEEE Conference on Computer Vision and Pattern Recognition, pp.

1091–1100, 2018.

[9] Zhou, Jingchun and Hao, Mingliang and Zhang, Dehuan and Zou, Peiyu and

Zhang, Weishi, Fusion PSPnet image segmentation based method for multi-focus

image fusion, IEEE Photonics Journal, vol. 11, no. 6, pp. 1–12, 2019.

[10] Yamazaki, Kashu and Hanyu, Taisei and Tran, Minh and Garcia, Adrian and

Tran, Anh and McCann, Roy and Liao, Haitao and Rainwater, Chase and Ad-

25

kins, Meredith and Molthan, Andrew and others, AerialFormer: Multi-resolution

Transformer for Aerial Image Segmentation, arXiv preprint arXiv:2306.06842,

2023.

[11] Chen, Liang-Chieh and Zhu, Yukun and Papandreou, George and Schroff, Flo-

rian and Adam, Hartwig, Encoder-decoder with atrous separable convolution for

semantic image segmentation, Proceedings of the European Conference on Com-

puter Vision (ECCV), pp. 801–818, 2018.

[12] Sun, Ke and Xiao, Bin and Liu, Dong and Wang, Jingdong, Deep high-

resolution representation learning for human pose estimation, Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5693–

5703, 2019.

[13] Li, Xiangtai and He, Hao and Li, Xia and Li, Duo and Cheng, Guangliang and

Shi, Jianping and Weng, Lubin and Tong, Yunhai and Lin, Zhouchen, Pointflow:

Flowing semantics through points for aerial image segmentation, Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp.

4217–4226, 2021.

[14] Xue, Gunagkuo and Liu, Yikun and Huang, Yuwen and Li, Mingsong and Yang,

Gongping, AANet: an attention-based alignment semantic segmentation network

for high spatial resolution remote sensing images, International Journal of Re-

mote Sensing, vol. 43, no. 13, pp. 4836–4852, 2022.

[15] Ma, Ailong and Wang, Junjue and Zhong, Yanfei and Zheng, Zhuo, FactSeg:

Foreground Activation-Driven Small Object Semantic Segmentation in Large-

Scale Remote Sensing Imagery, IEEE Transactions on Geoscience and Remote

Sensing, vol. 60, pp. 1-16, 2022.

[16] Hou, Jianlong and Guo, Zhi and Wu, Youming and Diao, Wenhui and Xu, Tao,

Bsnet: Dynamic hybrid gradient convolution based boundary-sensitive network

for remote sensing image segmentation, IEEE Transactions on Geoscience and

Remote Sensing, vol. 60, pp. 1–22, 2022.

26

[17] Hu, Jie and Shen, Li and Sun, Gang, Squeeze-and-excitation networks, CVPR,

pp. 7132–7141, 2018.

[18] Mou, Lichao and Hua, Yuansheng and Zhu, Xiao Xiang, Relation matters: Rela-

tional context-aware fully convolutional network for semantic segmentation of

high-resolution aerial images, IEEE Transactions on Geoscience and Remote

Sensing, vol. 58, no. 11, pp. 7557–7569, 2020.

[19] Niu, Ruigang and Sun, Xian and Tian, Yu and Diao, Wenhui and Chen, Kaiqiang

and Fu, Kun, Hybrid multiple attention network for semantic segmentation in

aerial images, IEEE Transactions on Geoscience and Remote Sensing, vol. 60,

pp. 1–18, 2021.

[20] Wang, Libo and Li, Rui and Duan, Chenxi and Zhang, Ce and Meng, Xiaoliang

and Fang, Shenghui, A novel transformer based semantic segmentation scheme

for fine-resolution remote sensing images, IEEE Geoscience and Remote Sensing

Letters, vol. 19, pp. 1–5, 2022.

[21] Xu, Rongtao and Wang, Changwei and Zhang, Jiguang and Xu, Shibiao and

Meng, Weiliang and Zhang, Xiaopeng, Rssformer: Foreground saliency enhance-

ment for remote sensing land-cover segmentation, IEEE Transactions on Image

Processing, vol. 32, pp. 1052–1064, 2023.

[22] Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Al-
varez, Jose M and Luo, Ping, SegFormer: Simple and efficient design for seman-

tic segmentation with transformers, Advances in Neural Information Processing

Systems, vol. 34, pp. 12077–12090, 2021.

[23] Wang, Libo and Li, Rui and Zhang, Ce and Fang, Shenghui and Duan, Chenxi and

Meng, Xiaoliang and Atkinson, Peter M, UNetFormer: A UNet-like transformer
for efficient semantic segmentation of remote sensing urban scene imagery, IS-

PRS Journal of Photogrammetry and Remote Sensing, vol. 190, pp. 196–214,

2022.

27

[24] Chen, Jieneng and Lu, Yongyi and Yu, Qihang and Luo, Xiangde and Adeli,

Ehsan and Wang, Yan and Lu, Le and Yuille, Alan L and Zhou, Yuyin, Tran-

sunet: Transformers make strong encoders for medical image segmentation,

arXiv preprint arXiv:2102.04306, 2021.

[25] Sun, Xian and Wang, Peijin and Lu, Wanxuan and Zhu, Zicong and Lu, Xiao-

nan and He, Qibin and Li, Junxi and Rong, Xuee and Yang, Zhujun and Chang,

Hao and others, Ringmo: A remote sensing foundation model with masked image

modeling, IEEE Transactions on Geoscience and Remote Sensing, 2022.

[26] Shan, Lianlei and Li, Xiaobin and Wang, Weiqiang, Decouple the high-frequency

and low-frequency information of images for semantic segmentation, ICASSP

2021-2021 IEEE International Conference on Acoustics, Speech and Signal Pro-

cessing (ICASSP), pp. 1805–1809, 2021.

[27] Zhang, Fengyu and Panahi, Ashkan and Gao, Guangjun, FsaNet: Frequency self-

attention for semantic segmentation, IEEE Transactions on Image Processing,

vol. 32, pp. 4757–4772, 2023.

[28] Chen, Linwei and Gu, Lin and Zheng, Dezhi and Fu, Ying, Frequency-adaptive
dilated convolution for semantic segmentation, Proceedings of the IEEE/CVF

Conference on Computer Vision and Pattern Recognition, pp. 3414–3425, 2024.

[29] Ma, Jiaqi and Xie, Guo-Sen and Zhao, Fang and Li, Zechao, AFANet: Adaptive

Frequency-Aware Network for Weakly-Supervised Few-Shot Semantic Segmenta-

tion, IEEE Transactions on Multimedia, 2025.

[30] He, Kaiming and Sun, Jian and Tang, Xiaoou, Guided image filtering, IEEE

Transactions on Pattern Analysis and Machine Intelligence, vol. 35, no. 6, pp.

1397–1409, 2012.

[31] Ma, Ningning and Zhang, Xiangyu and Zheng, Haitao and Sun, Jian, CARAFE:
Content-Aware ReAssembly of FEatures, Proceedings of the IEEE/CVF Interna-

tional Conference on Computer Vision (ICCV), pp. 3007–3016, 2019.

28

[32] Shi, Wenzhe and Caballero, Jose and Huszár, Ferenc and Totz, Johannes and

Aitken, Andrew P and Bishop, Rob and Rueckert, Daniel and Wang, Zehan,
Real-time single image and video super-resolution using an efficient sub-pixel

convolutional neural network, CVPR, pp. 1874–1883, 2016.

[33] Milletari, Fausto and Navab, Nassir and Ahmadi, Seyed-Ahmad, V-Net: Fully

Convolutional Neural Networks for Volumetric Medical Image Segmentation,

2016 Fourth International Conference on 3D Vision (3DV), pp. 565–571, 2016.

[34] Wang, Junjue and Zheng, Zhuo and Ma, Ailong and Lu, Xiaoyan and Zhong,

Yanfei, LoveDA: A Remote Sensing Land-Cover Dataset for Domain Adaptive

Semantic Segmentation, Proceedings of the Neural Information Processing Sys-

tems Track on Datasets and Benchmarks, 2021.

[35] Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas, U-net: Convolutional

networks for biomedical image segmentation, Medical Image Computing and

Computer-Assisted Intervention–MICCAI 2015: 18th International Conference,

Munich, Germany, October 5-9, 2015, Proceedings, Part III 18, pp. 234–241,

2015.

[36] Badrinarayanan, Vijay and Kendall, Alex and Cipolla, Roberto, Segnet: A deep

convolutional encoder-decoder architecture for image segmentation, TPAMI, vol.

39, no. 12, pp. 2481–2495, 2017.

[37] Fu, Jun and Liu, Jing and Tian, Haijie and Li, Yong and Bao, Yongjun and Fang,

Zhiwei and Lu, Hanqing, Dual attention network for scene segmentation, Pro-
ceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recog-

nition, pp. 3146–3154, 2019.

[38] Ding, Lei and Tang, Hao and Bruzzone, Lorenzo, LANet: Local attention em-

bedding to improve the semantic segmentation of remote sensing images, IEEE

Transactions on Geoscience and Remote Sensing, vol. 59, no. 1, pp. 426–435,

2020.

29

[39] Xu, Qingsong and Yuan, Xin and Ouyang, Chaojun and Zeng, Yue, Spatial–

spectral FFPNet: Attention-Based Pyramid Network for Segmentation and Clas-

sification of Remote Sensing Images, arXiv preprint arXiv:2008.08775, 2020.

[40] Wang, Di and Zhang, Jing and Du, Bo and Xia, Gui-Song and Tao, Dacheng, An

empirical study of remote sensing pretraining, IEEE Transactions on Geoscience

and Remote Sensing, 2022.

[41] Long, Jonathan and Shelhamer, Evan and Darrell, Trevor, Fully convolutional

networks for semantic segmentation, CVPR, pp. 3431–3440, 2015.

[42] Zhu, Xiao Xiang and Tuia, Devis and Mou, Lichao and Xia, Gui-Song and Zhang,

Liangpei and Xu, Feng and Fraundorfer, Friedrich, Deep learning in remote sens-

ing: A comprehensive review and list of resources, IEEE Geoscience and Remote

Sensing Magazine, vol. 5, no. 4, pp. 8–36, 2017.

[43] Y. Li, X. Li, Y. Dai, Q. Hou, L. Liu, Y. Liu, M.-M. Cheng, and J. Yang, “LSKNet:

A Foundation Lightweight Backbone for Remote Sensing,” International Journal

of Computer Vision, vol. 133, no. 3, pp. 1410–1431, 2025. Springer.

[44] W. Lu, S.-B. Chen, Q.-L. Shu, J. Tang, and B. Luo, “Decouplenet: A lightweight
backbone network with efficient feature decoupling for remote sensing visual

tasks,” IEEE Transactions on Geoscience and Remote Sensing, 2024. IEEE.

[45] X. Ma, R. Lian, Z. Wu, H. Guo, F. Yang, M. Ma, S. Wu, Z. Du, W. Zhang, and
S. Song, “Logcan++: Adaptive local-global class-aware network for semantic

segmentation of remote sensing images,” IEEE Transactions on Geoscience and

Remote Sensing, 2025. IEEE.

30

