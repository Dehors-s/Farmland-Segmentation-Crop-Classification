fabSAM: A Farmland Boundary Delineation Method
Based on the Segment Anything Model

Yufeng Xiea,1, Hanzhi Wub,1, Hongxiang Tongc, Lei Xiaod, Wenwen Zhoue,

Ling Lib,∗, Thomas Cherico Wangerf,g,∗

a  Department of Mathematics, Hong Kong University, Hong Kong, China
b. Key Laboratory of Coastal Environment and Resources of Zhejiang Province, School of Engineering,

Westlake University, Hangzhou, Zhejiang, China

c. Alibaba Cloud, Hangzhou, Zhejiang, China
d. Alibaba Group, Hangzhou, Zhejiang, China
e. Sungrow iCarbon, Hefei, Anhui, China
f.  Sustainable  Agricultural  Systems  and  Engineering  Laboratory,  School  of  Engineering,  Westlake

University, Hangzhou, Zhejiang, China

g. Production Technology and Cropping Systems Group, Department of Plant Production, AgroScope,

Nyon, Switzerland

∗  Corresponding authors: liling@westlake.edu.cn (Ling Li), tomcwanger@gmail.com (Thomas

Cherico Wanger)

1  Contributed equally to this work.

Abstract

Delineating farmland boundaries is essential for agricultural management such

as crop monitoring and agricultural census. Traditional methods using remote

sensing imagery have been efficient but limited in generalization. The Segment

Anything Model (SAM), known for its impressive zero-shot performance, has

been  adapted  for  remote  sensing  tasks  through  prompt  learning  and  fine-

tuning.  Here,  we  propose  a  SAM-based  farmland  boundary  delineation

framework (fabSAM) that combines a Deeplabv3+-based Prompter and SAM.

Also, a fine-tuning strategy was introduced to enable SAM’s decoder to improve

the use of prompt information. Experimental results on the AI4Boundaries and

AI4SmallFarms  datasets  have  shown  that

fabSAM  has  a  significant

improvement  in  farmland  region  identification  and  boundary  delineation.

Compared  to  zero-shot  SAM,  fabSAM  surpassed  it  by  23.47%  and  15.10%  in

mIOU  on  the  AI4Boundaries  and  AI4SmallFarms  datasets,  respectively.  For

Deeplabv3+,  fabSAM  outperformed  it  by  4.87%  and  12.50%  in  mIOU,

respectively.  These  results  highlight  the  effectiveness  of  fabSAM,  which  also

means that we can more easily obtain the global farmland region and boundary

maps from open-source satellite image datasets like Sentinel-2.

Keywords: fabSAM, Segment Anything Model, Farmland boundary delineation,

Semantic segmentation, Prompt engineering,

1.  Introduction

One  of  the  fundamental  and  challenging  tasks  of  smart  and  precision

agriculture  is  obtaining  the  spatial  distribution  of  agricultural  parcels  and

extracting  their  accurate  boundaries,  which  also  are

indispensable

prerequisites for downstream tasks in agricultural management, such as crop

monitoring  and  agricultural  census.  Compared  with  conventional  methods

such as field surveys, methods that utilize Remote Sensing (RS) imagery can

be  much  more  efficient  and  cost-effective,  as  there  are  a  large  amount  of

publicly available RS datasets with various spatial and temporal resolutions,

and the imagery interpretation technique is rapidly developing [1, 2].

Depending on the delineation methods used, existing studies extracting

farmland  boundaries  from  RS  imagery  can  be  roughly  divided  into  two

2

categories: traditional computer vision-based [3, 4] and deep learning-based

methods [5, 6]. The traditional ones can be further sub-categorized into edge-

based  [7]  and  region-based  techniques  [4,  8].  As  deep  learning-based

methods  have  shown  greater  accuracy  and  stability  than  traditional  ones,

models based on Deep Convolutional Networks (DCN) such as FCN [9], UNet

[10], Deeplab [11] and their variants [5, 12] and Vision Transformers (ViTs)

[13] have become popular recently.

However, the lack of training data in some countries and regions limits the

generalization of these data-driven models [13]. To overcome this limitation,

Visual  Fundamental  Models  (VFMs)  have  been  extended  to  RS  imagery

interpretation [14, 15, 16, 6], especially the Segment Anything Model (SAM)

[17]. SAM has been trained and used on a dataset with over one billion masks

and eleven million images for segmentation and therefore performs well on

different  kinds  of  tasks,  including  edge  detection  [17].  Also,  SAM  can  be

prompted  by  multiple  forms  of prompts,  including points,  bounding  boxes,

texts  and  masks,  then  allows  it  to  be  conveniently  applied  to  specific

downstream tasks [18].

Recently,  a  hybrid  architecture  has  gained  prominence  in  the  RS

community [14]. It contains a DCN model designed for object detection tasks

to output bounding boxes that pinpoint the region of interest (ROI) in the RS

images.  Then,  these  boxes  can  serve  as  prompts  for  the  SAM’s  prompt

encoder. Combining the Prompter, a SAM-based workflow that automatically

processes RS images has been developed.

Considering  the  variety  of  spatial  resolution  of  RS  datasets  and  the

expected outputs of farmland boundary delineation tasks, we identified two

challenges  in  constructing  this  hybrid  model  based  on  SAM:  (1)  How  to

3

generate the most effective prompts for SAM? (2) How can the SAM be fine-

tuned to get more accurate outputs?

To  address  the  aforementioned  challenges,  we  released  a  SAM-based

farmland  boundary  delineation  hybrid  framework  (fabSAM).  This  framework

contains  a  Deeplabv3+-based  Prompter  that  generates  low-resolution  masks

and points as prompts, and two SAM-based parts that can separately identify the

regions and boundaries of farmland. We then introduced a fine-tuning strategy

for  different  delineation  tasks.  At  the  end,  we  evaluated  whether  fabSAM  can

improve the performance of the original Prompter on the AI4Boundaries (AI4B)

and AI4SmallFarms (AI4S) datasets [19, 20]. To the best of our knowledge, this

work  is  the  first  to  introduce  a  hybrid  architecture  including  a  mask-prompt

generator and a SAM-based block for farmland boundary

delineation.

The rest of this article is organized as follows: Section 2 offers a review of

SAM’s applications in RS tasks, especially the farmland boundary delineation.

Section 3 introduces the proposed fabSAM in detail. Section  4 presents the

experimental results of fabSAM on the AI4B and AI4S datasets.

Finally, the conclusion is given in section 5.

2.  Related works

Currently, there are mainly three kinds of methods that have been adopted

to  enhance  SAM’s  performance  on  RS  tasks:  direct  use  of  SAM,  prompt

learning skills, and fine-tuning techniques. In a case study from Bihar, India,

SAM  was  employed  directly  for  farmland  boundary  delineation  in  an

unsupervised manner and successfully identified approximately 58% of the

4

boundaries [13]. Also, SAM was directly applied to improve the Cropland Data

Layer of the United States Department of Agriculture by incorporating a pixel-

based classifier [21]. Besides, serving as a plug-in-and-play module, SAM was

included  in  a  CocoaNet  architecture  (i.e.  a  consistency-constrained  multi-

class attention model), which was used to obtain image-level class labels for

RS images [22].

Prompt learning skills usually determine the performance of SAM through

the  quality  of  prompts,  and  how  to  automatically  generate  suitable

taskoriented  prompts  has  been  systematically  studied[16,  23].  A  Python

package for segmenting geospatial data with SAM was released, which allows

users  to  prompt  SAM  with  points,  boxes  and  texts  [24,  6].  Point  and  box

prompts are the most widely adopted due to their flexibility, clarity and ease

of acquisition. In a case study from Heilongjiang, China, a point prompter that

focused on the ROI extracted by a simple machine learning-based classifier

was  added  to  SAM  for  crop  field  boundary  delineation  [18].  GeoSAM  also

proposed a CNN-based sparse prompter to generate point prompts for SAM

to  automatically  segment  mobility  infrastructure  [25].  Besides  point

prompts, a SAM-based model prompted by boxes that were generated from

pseudo-labels  was  integrated  into  a  CS-WSCDNet  for  the  change  detection

task [26]. In addition, to take both point and box prompts into consideration,

a  SAM-based  model  for  the  local  RS  segmentation  task  was designed to  be

prompted by either center points of the ROI or bounding boxes that contain

the  ground-truth  masks  [15,  27].  As  for  text  prompts,  Text2Seg  developed

three basic methods to guide SAM for RS imagery segmentation:

”Grounding DINO + SAM”, ”CLIP Surgery + SAM” and ”SAM + CLIP”

[28].

5

There are few studies focusing on mask prompts, however, they still have

great potential in efficiently guiding SAM for RS segmentation compared to

sparse prompts (i.e., points and boxes). First, mask prompts can provide the

most exact spatial information for objects of interest. For example, GeoSAM

contained a CNN-based mask prompter that can provide dense prompts for

SAM[25]. Second, SAM supports self-guided iterative predictions. To be more

specific, SAM can take the masks it predicted as input and iteratively predict

masks to extract maximal information, then get stable and exact predictions.

Take  the  Few-shot  Self-guided  Large  Vision  Model  (Few-shot  SLVM)  as  an

example, this framework introduced an automatic prompt learning technique

using  the  SAM  for  rendering  coarse  pixel-wise  prompts  instead  of  heavy

reliance on manual guidance [29].

Recently,  fine-tuning  techniques  have  been  proven  to  be  necessary  for

extending SAM to downstream tasks [6, 30]. For example, several works have

utilized  a  low-rank  adaptation  (LoRA)  approach  for  crop-specific  feature

extraction and image encoder’s fine-tuning [30, 31]. Also, there are some text-

based  one-shot  learning  approaches  have  been  developed  to  retrain  SAM

[28].  Furthermore,  a  fine-tuning  strategy  of  SAM’s  mask  decoder  was

introduced  to  improve  the  performance  of  GeoSAM  [25].  Despite  all  the

efforts  of  integration  and  improvements  of  SAM  in  farmland  boundary

delineation, an integrated and fine-tuned approach of a mask Prompter and

SAM-based block is underutilized.

3.  Methodology

The  overall  framework  of  fabSAM  is  illustrated  in  Figure.  1.  fabSAM

involves two main blocks, including a Prompter and a SAM-based block. The

RS images are first utilized by the Prompter to generate both mask and point

6

prompts for SAM to localize farmland region. Following this, in the SAMbased

block, the frozen SAM-based image encoder is utilized to extract highquality

semantic  features  from  original  RS  images,  and  the  prompt  encoder  is

employed to obtain explicit positional knowledge from prompts generated by

the Prompter. Then, two fine-tunable decoders are introduced for farmland

identification and boundary delineation tasks. Finally, the region mask  and

boundary  mask  are  generated  by  the  corresponding  decoder  and

postprocessed  to  generate  farmland  parcels  with  exact  and  closed

boundaries. This section first gives an introduction to the Prompter and the

SAM-based block, followed by a discussion of our training strategies.

Figure 1: The overview of fabSAM framework consists of a Prompter and a SAM-based block

(core  block).  The  Prompter  based  on  deeplabv3+  focuses  on  generating  mask  and  point

prompts for SAM (large yellow dashed box and arrows). In the SAM-based block (blue dashed

7

box  and  arrows),  the  input  image  and  prompts  are  processed  by  the  image  encoder  and

prompt encoder. Corresponding embeddings are then utilized by two fine-tunable decoders

to produce the masks for farmland identification and boundary de-lineation, respectively.

3.1.  Prompter

The  Prompter  is  designed  to  determine  whether  the  pixels  in  the  RS

images belong to the farmland or background and generate both masks and

points as prompts for the SAM-based block. As the range of farmland area is

large  and  the  boundaries  are  irregular,  we  use  Deeplabv3+  [32]  here  as

Prompter to better capture the multi-scale semantic features for farmland

identification.

The  Prompter  follows  the  standard  design  of  Deeplabv3+,  which  is  an

encoder-decoder  structure.  The  encoder  consists  of  two  blocks:  a

ResNet50based  DCNN  block  [33]  and  an  Atrous  Spatial  Pyramid  Pooling

(ASPP)based block. In the prompt decoder, the low-level features drawn from

the ResNet50-based DCNN block and the multi-level features extracted from

the ASPP-based block are both aggregated and fed into a 3 × 3 convolution

layer. In the end, an upsampling layer is used to restore the input shape. It is

worth mentioning that the depthwise separable convolutions are used in the

ASPP-based block and the prompt decoder to reduce the computation

complexity [32].

Following  this,  we  directly  take  the  logits  containing  abundant  location

information  output  by  the  prompt  decoder  as  the  mask  prompt,  which

follows the original design of SAM. Let mp ∈ RH×W denote the mask prompts,

where  H  and  W  represent  the  resolution  of  the  input  RS  image.  Then,  we

introduce a point prompt generator (Gen) to obtain point prompts. In Gen, a

farmland  probability  map  (P  ∈  RH×W)  is  first  produced  by  applying  the

8

sigmoid  function  on  mp.  Several  foreground  and  background  points  are

randomly chosen as prompts according to the following rules: (1) probability

of belonging to farmland Pi of the foreground pixel i must be larger than 0.7

while Pj of background pixel  j must be less than 0.3 to ensure the accuracy

(these two thresholds are selected based on the experiment results); (2) the

larger  the  Pi (the  lower  the  Pj),  the  more  likely  it  is  to  be  selected  as  point

prompts. The whole process of the point prompt generation can be defined

as:

P = Sigmoid(mp),

pp = Gen(P)

(1)

where pp ∈ RN×2 denote the point prompts, and N denotes the number

of chosen points.

3.2.  SAM-based block

The  SAM-based  block  comprises  four  components  that  are  duplicated

from  SAM:  an  image  encoder  (EncI),  a  prompt  encoder  (EncP)  and  two

identical  mask  decoders  (DecM and  DecB)  that  are  fine-tuned separately  for

region identification and boundary delineation. The EncI based on MAE pre-

trained ViT [17] is used to extract high-level semantic features from the input

images. Meanwhile, EncP is utilized to encode the masks and points that are

automatically generated by the Prompter as prompt embeddings. Then, the

two light-weight decoders DecB and DecM based on self-attention and cross-

attention  are  utilized  to  interact  between  image  embedding  and  prompt

embeddings  and  predict  the  boundary  and  region  masks.  To  provide  more

insight, the whole process of the SAM-based block is defined as:

9

FI = EncI(I),

Fmp = EncP(mp),

Fpp = EncP(pp),

B = DecB(FI + Fmp,Fpp),

(2)

M = DecM(FI + Fmp,Fpp)
Where  FI  ∈  Rh×w×c,  Fmp  ∈  Rh×w×c,  and  Fpp  ∈  RN×c  represent  the  image

embeddings,  mask  representation,  and  point  prompt  tokens  respectively,  h

and  w  represent  the  resolution  of  the  image  features,  and  c  denotes  the

feature dimension. Furthermore,  B and M denote the predicted boundaries

and regions.

A post-processor is then introduced to generate the farmland parcels with

closed and exact boundaries. First, a symmetrical difference is made between

the  two  outputs,  B  and  M,  to  more  clearly  distinguish  the  boundaries,  and

then the images are stitched into the original shape to get the final outputs.

3.3.  Training strategies

Note  that  most  of  the  classical  segment  models  can  be  re-molded  as

autoprompt generators, and we can, therefore, treat the SAM-based model as

a post-processing block that can improve the performance of these segment

models. To make our framework more convenient to use and transform, we

fine-tune the Prompter and the SAM-based block, separately.

3.3.1.  Training Prompter

In this training phase, the deeplabv3+-based Prompter calculates the loss

by comparing the output pre-predicted mask to the extent mask of farmland.

It  is  important  to  highlight  that  we  add  an  auxiliary  head  after  the  image

10

encoder  to  calculate  the  auxiliary  loss,  which  is  used  as  an  assist  in  the

training of a ResNet50-based DCNN block to optimize shallow layers [34, 35].

To be more detailed, the auxiliary head consists of a convolution layer with

kernel size 3, a batch normalization layer and a convolution layer with kernel

size 1 in series.

For the loss function to train the Prompter, LP, we choose Cross-Entropy

Loss (LC) for both main loss (Lm) and auxiliary loss (La), so that they can be

expressed using the same formula:

!

(3)

Where N denotes the number of pixels in the input image, ysi denotes the

ground truth of farmland region, wm and wa are the weight of main loss and

auxiliary  loss,  and  yi  represents  the  predicted  probability  of  pixel  i.  yi  is

generated by a Softmax function operating on logit, which is output by either

a prompt decoder or an auxiliary decoder.

3.3.2.  Fine-tuning SAM-based block

During  the  fine-tuning  phase,  we  mainly  update  the  parameters  of  the

lightweight  mask  decoder  and  boundary  decoder  separately  while  the

heavyweight image encoder is kept frozen. Meanwhile, we also fine-tune the

parameters of the prompt encoder to better map the mask and point prompts

to embeddings. For the loss function used to fine-tune the SAM-based block,

we adopt the combination of Dice Loss and Focal Loss [25], which are both

designed for situations where there is a strong imbalance between positive

and  negative  samples.  Dice  Loss  based  on  the  dice  coefficient  is  closely

11

associated with the F1 score which is a widely used evaluation metric. The

Dice Loss (LD) can be articulated as:

Where ybi denotes the ground truth of farmland boundary or region,  yiO

represents the output of the mask decoder, which is processed by a Sigmoid

(4)

function.

The Focal Loss is an enhancement of the Cross-Entropy Loss in terms of

the imbalance of samples. Two parameters  α and γ are introduced to make

the model focus on fewer examples and difficult-to-distinguish examples. The

definition of Focal Loss (LF) is shown as:

In conclusion, the loss function for fine-tuning the SAM-based block (LFT)

(5)

can be expressed as:

LFT = wDLD + wFLF

(6)

Where wD and wF denote the weight of Dice Loss (LD) and Focal Loss

(LF) respectively.

4.  Experiments and discussion

4.1.  Datasets and pre-processing

Two  open  AI-ready  field  boundary  datasets  named  AI4B  [19]  and  AI4S

[20] were  utilized  to  fine-tune  and  evaluate  fabSAM.  For  the  AI4B  dataset,

12

these  images  were  captured  in  2019  over  Austria,  Catalonia,  France,

Luxembourg,  the  Netherlands,  Slovenia  and  Sweden.  We  employed  the  10-

meter  cloud-free  Sentinel-2  subset,  which  consisted  of  7831  monthly

composite  images  captured  in  May,  and  their  corresponding  ground  truth

boundary labels contained an extent mask and a boundary mask. For the AI4S

dataset,  62  cloud-free  Sentinel-2  images  with  a  spatial  resolution  of  10

meters were provided along with extent labels in vector format and boundary

labels  in  raster  format.  The  areas  in  these  images  were  distributed  over

Vietnam and Cambodia.

Before taking these images as input for fabSAM, we pre-processed them

as  follows.  First,  we  converted  the  data  from  both  two  datasets  to  Geotiff

format, and performed band rendering and contrast enhancement - in which

minimum and maximum band values were set to 0 and 3000 respectively - to

make sure the enhanced images can be shown clearly. Then, we exported the

rendered images (band range 0-255) and cropped them to a size of 256 × 256

pixels. As for the label data, the extent masks of farmland (transformed into

raster format) were used for the training of Prompter and the fine-tuning of

the mask decoder. The boundary masks were used for the fine-tuning of the

boundary decoder. For both datasets, we took 70% of the data for training,

15% for testing, and the remaining 15% for validation.

4.2.  Implementation details

Our proposed model was developed using the Pytorch framework. MM-

Segmentation and Lightning packages were used to train and fine-tune the

Prompter  and  the  SAM-based  block  separately.  All  model  training  and

experiments  were  conducted  using  an  NVIDIA  A10  GPU  with  24GB  RAM.

13

Detailed  parameters  setting  for  the  training  of  Prompter  and  SAM-based

block are listed in Table. 1.

Table 1: Parameters setting for model training and fine-tuning

Configuration

Prompter

SAM-based block

Optimizer

Batch size

SGD

8

Adam

4

Total epochs (iterations)  80000 iterations

20 epochs

Initial learning rate

LR policy

0.004

Poly

0.0003

Poly

Decay factor

0.0001

0.0001

4.3.  Evaluation metrics

To  clearly  and  quantitatively  evaluate  the  performance  of  fabSAM  on

farmland  identification  and  boundary  delineation,  we used  four  evaluation

indicators:  Intersection  over  Union  (IoU),  F1-score,  Accuracy,  and  a

composite indicator mean Intersection Over Union (mIOU). The calculation

formulas for the first three metrics are as follows:

IoU =
F1 =(7)

Accuracy
=

Where TP, FP, TN, and FN represent the numbers of pixels that belong

14

to true positive, false positive, true negative, and false negative, respectively.

In order to measure the comprehensive ability of our fabSAM in the tasks

of  both  region  recognition  and  boundary  delineation,  we  introduced  a

composite indicator mIOU, which is defined as follows:

mIOU =

)

(8)

Where  IoUr  and  IoUb  denote  the  IoU  of  the  farmland  region  class  and

boundary class, respectively. The extent of farmland fragmentation and the

resolution of RS images can influence the mIOU value. In our experiment, a

model with a mIOU value exceeding 35% was deemed to exhibit satisfactory

performance, while a value surpassing 50% indicated superior performance.

4.4.  Results and discussion

4.4.1.  Performance comparison

In  order  to  demonstrate  the  performance  enhancement  achieved  by

fabSAM compared to the original prompter and SAM, we initially conducted

a comparative analysis of fabSAM against zero-shot SAM [17] and Deeplabv3+

[32].  Additionally,  we  selected  two  state-of-the-art  semantic  segmentation

models - UNet+PSPNet (based on convolution) [36]) and MaskFormer (based

on  transformer  architecture)  [37])  -  for  further  comparison  to  verify  the

effectiveness of fabSAM. We then quantitatively compared the performance

of fabSAM with these models based on IoU, F1-score, Accuracy and mIOU

evaluation metrics.

From Table. 2, our fabSAM exhibited significant advancements over other

models in the composite indicator mIOU on both AI4B and AI4S datasets. For

example,  compared  to  zero-shot  SAM,  fabSAM  surpassed  it by  23.47%  and

15

15.10%  in  mIOU  on  AI4B  and  AI4S  datasets,  respectively.  Also,  fabSAM

surpassed Deeplabv3+ by 4.87% and 12.50% in mIOU, respectively.

While showing only slight improvements in some metrics, for example, fab-

SAM  outperformed  MaskFormer  by  only  0.31%  in  Accuracy  on  farmland

region identification on the AI4B dataset. However, fabSAM truly shined in

the boundary delineation task, outperforming all other models on the AI4B

dataset in IoU, F1 and Accuracy. Overall, fabSAM has superior comprehensive

performance, particularly excelling in farmland boundary delineation.

Table 2: Quantitative comparison on the performance of fabSAM and selected models:

Zero-shot SAM, Deeplabv3+, UNet+PSPNet, MaskFormer
Farmland Boundary

Farmland Region

Composite

Dataset

Method

IoU

F1-score  Accuracy

IoU

F1-score  Accuracy

mIOU

AI4B

AI4S

fabSAM
SAM (zero-shot)
Deeplabv3+

60.64
44.23
59.30

71.11
57.22
69.85

85.66
74.69
85.86

37.40
6.86
28.99

51.01
12.37
40.87

84.63
75.16
83.66

49.02
25.55
44.15

Unet+PSPNet

57.98

68.54

84.80

29.39

41.41

84.15

43.69

Maskformer

57.93

68.75

85.35

31.97

44.73

82.94

44.95

fabSAM
SAM (zero-shot)
Deeplabv3+

84.93
73.95
84.30

91.52
84.70
91.12

88.47
79.76
87.92

27.62
8.40
3.25

42.84
15.36
5.89

72.51
69.84
74.28

56.28
41.18
43.78

Unet+PSPNet

83.07

90.34

86.94

8.13

13.61

70.76

45.60

Maskformer

83.30

90.48

87.17

27.23

42.36

31.36

55.27

The  visual  performance  of  fabSAM  on  several  test  images  is  shown  in

Figure.  2.  These  figures  demonstrate  that  the  performance  of  fabSAM  in

16

recognizing  small-scale  farmland  regions  and  irregular  boundaries  was

better  than  zero-shot  SAM,  Deeplabv3+  and  other  models.  In  order  to  see

more  intuitively  how  fabSAM  can  optimize  the  outputs  of  the  Deeplabv3+

model,  which  also  serves  as  the  primary  component  of  the  Prompter  in

fabSAM, the logit maps generated by deeplabv3+, fabSAM’s mask decoder and

boundary  decoder  are  presented  in  Figure.  3.  These  maps  represent  the

probability that corresponding pixels belong to the foreground. Focusing on

the areas delineated by the red boxes in Figure. 3, we can find that fabSAM

outperforms  Deeplabv3+  in terms  of  differentiating  farmland  regions  from

complex backgrounds, including water bodies, roads, and trees. Based on the

multilevel features extracted by SAM-based Image Encoder, fabSAM has much

better prediction stability by increasing the difference between foreground

and background in logit maps.

Figure 2: Qualitative comparison on farmland boundary delineation: fabSAM, Zero-shot

SAM, Deeplabv3+, UNet+PSPNet and MaskFormer.

17

Figure  3:  Improvement  in  prediction  accuracy  of  logits:  (a)  original  image,  (b)  logits

predicted by Deeplabv3+, (c) logits predicted by fabSAM’s Mask Decoder, (d) logits predicted

by fabSAM’s Boundary Decoder.

As  for  fabSAM’s  performance  on  the  farmland  boundary  delineation  task,

Figure. 4 shows the comparisons between fabSAM and the best-performing three

models  on  the  AI4S  dataset:  Deeplabv3+,  MaskFormer  and  UNet+PSPNet.  The

boundaries generated by fabSAM are more accurate and have better connectivity.

Though the clarity of these images in the dataset was affected by the resolution

and  acquisition  time  of  the  satellite  imagery,  fabSAM  still  performs  well  in

difficult-to-distinguish regions.

18

Figure  4:  Qualitative  comparison  on  boundary  delineation:  fabSAM,  MaskFormer,

UNet+PSPNet and Deeplabv3+.

4.4.2.  Ablation experiments

In  this  part,  we  removed  a  certain  part  of  fabSAM  to  verify  the

effectiveness  of  each  component,  and  the  quantitative  results  of  different

settings  are  shown  in  Table.  3.  The  first  row  corresponds  to  the  original

fabSAM, which has the  highest scores. Notably, not fine-tuning the decoder

would lead to a substantial decline in model performance, with a 16.32% IoU

decrease  for  region  identification  and  a  38.65%  F1  decrease  for  boundary

delineation. The effects of different forms of prompts are shown in the last

two  rows.  Considering  the  both  region  identification  and  boundary

delineation tasks, removing the Mask Prompt resulted in an approximately

1%  decrease  in  IoU  and  F1  while  removing  the  Point  Prompt  caused  an

approximately 2% decrease.

19

Table 3: Ablation experimental results. FTD = Fine-tuning Decoder, FTPE =
Fine-tuning Prompt Encoder, MP = Mask Prompt, PP = Point Prompt

Techniques

Region

Boundary

SAM ENC

FTD

FTPE  MP  PP

IOU

F1

IOU

F1

✓

✓

✓

✓

✓

✓

×

✓

✓

✓

✓ ✓ ✓

✓ ✓ ✓

× ✓ ✓

✓ × ✓

✓ ✓ ×

60.55  71.04  37.4  51.01

44.23  57.22  6.85  12.36

60.44  70.82  36.85  50.52

59.58  69.96  37.26  50.57

58.11  69.04  35.85  49.11

Although  fabSAM  performs  well  on  the  AI4B  and  AI4S  datasets,  this

framework has a notable limitation that its accuracy depends mainly on the

accuracy of the Prompter. In addition, as SAM itself has the disadvantage of

missing  fine  structures,  it  may  lead  fabSAM  to  ignore  small  areas  of

fragmented  farmland.  This  is  the  reason  why  the  improvement  in  the

performance of fabSAM on region identification is not significant.

5.  Conclusion

In  this  paper,  we  proposed  a  fabSAM  framework  for  farmland  region

identification and boundary delineation tasks, which can generate farmland

region  parcels  with  valid  perimeters  and  boundary  maps  with  good

20

connectivity  from  satellite  images  without  human  intervention.  We  also

introduced a new architecture to generate prompts for the SAM from masks

and  points  that  an  arbitrary  semantic  segmentation  model  predicts.

Furthermore,  a  SAM-based  block  decoder  was  developed  for  two  tasks:

farmland  region  identification  (instance  segmentation)  and  boundary

delineation  (edge  detection).  Experimental  results  on  the  AI4B  and  AI4S

datasets highlight the effectiveness and great generalization of fabSAM, which

also  means  that  we  can  more  easily  obtain  the  global  farmland  region  and

boundary  maps  from  open-source  satellite  image  datasets  like  Sentinel-2.

However, according to the results of ablation experiments, it is challenging to

improve segmentation performance by adding a SAM-based block and fine-

tuning a lightweight decoder. Therefore, to promote the future development

of  smart  and  precision  farmland  planning  and  management,  future  work

could focus on how to generate more stable and exact prompts, and how to

construct new modules that refine boundary details.

References

[1]  X. Wang, L. Shu, R. Han, F. Yang, T. Gordon, X. Wang, H. Xu, A survey of

farmland  boundary  extraction  technology  based  on  remote  sensing

images, Electronics 12 (5) (2023). doi:10.3390/electronics12051156.

URL https://www.mdpi.com/2079-9292/12/5/1156

[2]  M. Li, J. Long, A. Stein, X. Wang, Using a semantic edge-aware

multi-task neural network to delineate agricultural parcels from remote

sensing images, ISPRS Journal of Photogrammetry and Remote Sensing

21

200

(2023)

24–40.

doi:https://doi.org/10.1016/j.isprsjprs.2023.04.019.

URL https://www.sciencedirect.com/science/article/pii/ S0924271623001077

[3]  S.  R.  Debats,  D.  Luo,  L.  D.  Estes,  T.  J.  Fuchs,  K.  K.  Caylor,  A  generalized

computer  vision  approach  to  mapping  crop  fields  in  heterogeneous

agricultural  landscapes,  Remote  Sensing  of  Environment  179  (2016)

210–221. doi:https://doi.org/10.1016/j.rse.2016.03.010.

URL https://www.sciencedirect.com/science/article/pii/ S0034425716301031

[4]  B. Watkins, A. van Niekerk, A comparison of object-based image analysis

approaches  for  field  boundary  delineation  using  multi-temporal

sentinel-2  imagery,  Computers  and  Electronics  in  Agriculture  158

(2019) 294–302. doi:https://doi.org/10.1016/j.compag.2019.02.009.

URL https://www.sciencedirect.com/science/article/pii/ S0168169918317733

[5]  F.  Waldner,  F.  I.  Diakogiannis,  Deep  learning  on  edge:  Extracting  field

boundaries from satellite images with a convolutional neural network,

Remote Sensing of Environment 245 (2020) 111741.

doi:https://doi.org/10.1016/j.rse.2020.111741.

URL

https://www.sciencedirect.com/science/article/pii/

S0034425720301115

[6]  L. P. Osco, Q. Wu, E. L. de Lemos, W. N. Gon¸calves, A. P. M. Ramos, J. Li, J.

Marcato,  The  segment  anything  model  (sam)  for  remote  sensing

applications:  From  zero  to  one  shot,  International  Journal  of  Applied

Earth Observation and Geoinformation 124 (2023) 103540.

doi:https://doi.org/10.1016/j.jag.2023.103540.

22

URL

https://www.sciencedirect.com/science/article/pii/

S1569843223003643

[7]  J.  Graesser,  N.  Ramankutty,  Detection  of  cropland  field  parcels  from

landsat imagery, Remote Sensing of Environment 201 (2017) 165–180.

doi:https://doi.org/10.1016/j.rse.2017.08.027.

URL https://www.sciencedirect.com/science/article/pii/ S0034425717303930

[8]  L. Yan, D. Roy, Automated crop field extraction from multi-temporal web

enabled landsat data, Remote Sensing of Environment 144 (2014) 42–

64. doi:https://doi.org/10.1016/j.rse.2014.01.006.

URL https://www.sciencedirect.com/science/article/pii/ S0034425714000194

[9]  K. M. Masoud, C. Persello, V. A. Tolpekin, Delineation of agricultural field

boundaries  from  sentinel-2  images  using  a  novel  super-resolution

contour detector based on fully convolutional networks, Remote Sensing

12 (1) (2020). doi:10.3390/rs12010059.

URL https://www.mdpi.com/2072-4292/12/1/59

[10] H.  L.

Aung,  B.

Uzkent,

M.

Burke, D.

Lobell, S.

Ermon,

Farm parcel delineation using spatio-temporal convolutional networks,

in:  2020  IEEE/CVF  Conference  on  Computer  Vision  and  Pattern

Recognition Workshops (CVPRW), 2020, pp. 340–349.

doi:10.1109/CVPRW50498.2020.00046.

[11] W. Shunying, Z. Ya’nan, Y. Xianzeng, F. Li, W. Tianjun, L. Jiancheng, Bsnet:

Boundary-semantic-fusion  network  for  farmland  parcel  mapping  in

high-resolution satellite images,

23

Computers and  Electronics

(2023)
107683. doi:https://doi.org/10.1016/j.compag.2023.107683.

Agriculture  206

in

URL https://www.sciencedirect.com/science/article/pii/ S0168169923000716

[12] Y. Wang, L. Gu, T. Jiang, F. Gao, Mde-unet: A multitask deformable unet

combined enhancement network for farmland boundary segmentation,

IEEE  Geoscience  and  Remote  Sensing  Letters  20  (2023)  1–5.

doi:10.1109/LGRS.2023.3252048.

[13] P. Tripathy, K. Baylis, K. Wu, J. Watson, R. Jiang, Investigating the segment

anything  foundation  model  for  mapping  smallholder  agriculture  field

boundaries without training labels (2024). arXiv:2407.01846.

[14] S. Ren, F. Luzi, S. Lahrichi, K. Kassaw, L. M. Collins, K. Bradbury, J. M. Malof,

Segment anything, from space?, in: 2024 IEEE/CVF Winter Conference

on Applications of Computer Vision (WACV), 2024, pp.

8340–8350. doi:10.1109/WACV57701.2024.00817.

[15] D. Wang, J. Zhang, B. Du, M. Xu, L. Liu, D. Tao, L. Zhang, Samrs: Scaling-up

remote sensing segmentation dataset with segment anything model, in:

A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, S. Levine (Eds.),

Advances  in  Neural  Information  Processing  Systems,  Vol.  36,  Curran

Associates, Inc., 2023, pp. 8815–8827.

URL https://proceedings.neurips.cc/paper_files/paper/2023/

file/1be3843e534ee06d3a70c7f62b983b31-Paper-Datasets_and_

Benchmarks.pdf

[16] K. Chen, C. Liu, H. Chen, H. Zhang, W. Li, Z. Zou, Z. Shi, Rsprompter:

24

Learning to prompt for remote sensing instance segmentation based on

visual foundation model (2023). arXiv:2306.16269.

URL https://arxiv.org/abs/2306.16269

[17] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson,

T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Doll´ar, R. Girshick,

Segment anything (2023). arXiv:2304.02643.

URL https://arxiv.org/abs/2304.02643

[18] X.  Liu,  A  sam-based  method  for  large-scale  crop  field  boundary

delineation,  in:  2023  20th  Annual  IEEE  International  Conference  on

Sensing,  Communication,  and  Networking  (SECON),  2023,  pp.  1–6.

doi:10.1109/SECON58729.2023.10287502.

[19] R. d’Andrimont, M. Claverie, P. Kempeneers, D. Muraro, M. Yordanov, D.

Peressutti, M. Batiˇc, F. Waldner, Ai4boundaries: an open ai-ready dataset

to map field boundaries with sentinel-2 and aerial photography,

Earth  System  Science  Data  15  (1)  (2023)  317–329.  doi:10.5194/essd-
15-

317-2023.

URL https://essd.copernicus.org/articles/15/317/2023/

[20] C.  Persello,  J.  Grift,  X.  Fan,  C.  Paris,  R.  Ha¨nsch,  M.  Koeva,  A.  Nelson,

Ai4smallfarms:  A  dataset  for  crop  field  delineation  in  southeast  asian

smallholder  farms,  IEEE  Geoscience  and  Remote  Sensing  Letters  20

(2023) 1–5. doi:10.1109/LGRS.2023.3323095.

[21] C. Zhang, P. Marfatia, H. Farhan, L. Di, L. Lin, H. Zhao, H. Li,

M. D. Islam, Z. Yang, Enhancing usda nass cropland data layer

25

with segment anything model, in:

2023 11th International Conference

on Agro-Geoinformatics (Agro-Geoinformatics), 2023, pp. 1–5.

doi:10.1109/Agro-Geoinformatics59224.2023.10233404.

[22] J. Zhang, Q. Zhang, Y. Gong, J. Zhang, L. Chen, D. Zeng, Weakly supervised

semantic  segmentation  with  consistency-constrained  multiclass

attention  for  remote  sensing  scenes,  IEEE  Transactions  on  Geoscience

and Remote Sensing 62 (2024) 1–18. doi:10.1109/TGRS.2024.3392737.

[23] J. Huang, K. Jiang, J. Zhang, H. Qiu, L. Lu, S. Lu, E. Xing, Learning to prompt

segment anything models (2024). arXiv:2401.04651.

URL https://arxiv.org/abs/2401.04651

[24] Q. Wu, L. P. Osco, samgeo: A python package for segmenting geospatial

data with the segment anything model (sam), J. Open Source Softw. 8

(2023) 5663.

URL https://api.semanticscholar.org/CorpusID:261797762

[25] R. I. Sultan, C. Li, H. Zhu, P. Khanduri, M. Brocanelli, D. Zhu, Geosam: Fine-

tuning  sam  with  sparse  and  dense  visual  prompting  for  automated

segmentation of mobility infrastructure (2024). arXiv:2311.11319.

URL https://arxiv.org/abs/2311.11319

[26] L.  Wang,  M.  Zhang,  W.  Shi,  Cs-wscdnet:  Class  activation  mapping  and

segment  anything  model-based  framework  for  weakly  supervised

change detection, IEEE Transactions on Geoscience and Remote Sensing

61

(2023) 1–12. doi:10.1109/TGRS.2023.3330479.

26

[27] S.  Julka,  M.  Granitzer,  Knowledge  Distillation  with  Segment  Anything

(SAM)  Model  for  Planetary  Geological  Mapping,  2024,  pp.  68–77.

doi:10.1007/978-3-031-53969-56.

[28] J. Zhang, Z. Zhou, G. Mai, M. Hu, Z. Guan, S. Li, L. Mu,  Text2seg: Remote

sensing image semantic segmentation via text-guided visual foundation

models (2024). arXiv:2304.10597.

URL https://arxiv.org/abs/2304.10597

[29] X.  Qi,  Y. Wu,  Y.  Mao,  W.  Zhang,  Y.  Zhang,  Self-guided  few-shot  semantic

segmentation for remote sensing imagery based on large vision models

(2023). arXiv:2311.13200.

URL https://arxiv.org/abs/2311.13200

[30] B. Xue,

H. Cheng,

Q. Yang,

Y. Wang,

X. He,  Adapting

segment

anything model to aerial land cover classification with low-rank adaptation,

IEEE Geoscience and Remote Sensing Letters 21 (2024) 1–5.

doi:10.1109/LGRS.2024.3357777.

[31] B. Song, H. Yang, Y. Wu, P. Zhang, B. Wang, G. Han, A multispectral remote

sensing  crop segmentation  method  based  on  segment  anything  model

using  multistage  adaptation

fine-tuning,

IEEE  Transactions  on

Geoscience

and

Remote

Sensing

62

(2024)

1–18.

doi:10.1109/TGRS.2024.3411398.

[32] L.-C. Chen, Y. Zhu, G. Papandreou, F. Schroff, H. Adam, Encoder-decoder

with  atrous  separable  convolution  for  semantic  image  segmentation

(2018). arXiv:1802.02611.

27

URL https://arxiv.org/abs/1802.02611

[33] K.  He,  X.  Zhang,  S.  Ren,  J.  Sun,  Deep  residual  learning  for  image

recognition

(2015). arXiv:1512.03385.

URL https://arxiv.org/abs/1512.03385

[34] C.-Y. Lee, S. Xie, P. Gallagher, Z. Zhang, Z. Tu, Deeply-supervised nets

(2014). arXiv:1409.5185.

URL https://arxiv.org/abs/1409.5185

[35] L.  Wang,  C.-Y.  Lee,  Z.  Tu,  S.  Lazebnik,  Training  deeper  convolutional

networks with deep supervision (2015). arXiv:1505.02496.

URL https://arxiv.org/abs/1505.02496

[36] O.  Ronneberger,  P.  Fischer,  T.  Brox,  U-net:  Convolutional  networks  for

biomedical image segmentation (2015). arXiv:1505.04597.

URL https://arxiv.org/abs/1505.04597

[37] B. Cheng, A. G. Schwing, A. Kirillov, Per-pixel classification is not all you

need for semantic segmentation (2021). arXiv:2107.06278.

URL https://arxiv.org/abs/2107.06278

28

