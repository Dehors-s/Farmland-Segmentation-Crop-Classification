## PRUE: A Practical Recipe for Field Boundary Segmentation at Scale

### Gedeon Muhawenayo^1 ∗#, Caleb Robinson^2 ∗, Subash Khanal^3 ∗, Zhanpei Fang^4 ∗,

### Isaac Corley^5 , Alexander Wollam^3 , Tianyi Gao^3 , Leonard Strnad^5 , Ryan Avery^5 ,

### Lyndon Estes^6 , Ana M. Tárano^1 , Nathan Jacobs^3 , Hannah Kerner^1

(^1) Arizona State University (^2) Microsoft AI for Good (^3) Washington University in St. Louis
(^4) Oregon State University (^5) Wherobots (^6) Clark University
Figure 1. Example results from the PRUE model in Japan. PRUE specifies a practical recipe for country-scale field boundary segmentation
that is robust to reflectance variation and noise in Sentinel-2 L2A imagery, improving standard metrics and large-scale map quality.

### Abstract

```
Large-scale maps of field boundaries are essential for
agricultural monitoring tasks. Existing deep learning ap-
proaches for satellite-based field mapping are sensitive to
illumination, spatial scale, and changes in geographic loca-
tion. We conduct the first systematic evaluation of segmen-
tation and geospatial foundation models (GFMs) for global
field boundary delineation using the Fields of The World
(FTW) benchmark. We evaluate 18 models under unified
experimental settings, showing that a U-Net semantic seg-
mentation model outperforms instance-based and GFM al-
ternatives on a suite of performance and deployment metrics.
We propose a new segmentation approach that combines
a U-Net backbone, composite loss functions, and targeted
data augmentations to enhance performance and robustness
under real-world conditions. Our model achieves a 76%
IoU and 47% object-F1 on FTW, an increase of 6% and
9% over the previous baseline. Our approach provides a
practical framework for reliable, scalable, and reproducible
field boundary delineation across model design, training,
and inference. We release all models and model-derived
field boundary datasets for five countries.
```
### 1. Introduction

```
Agricultural field boundary maps (digitized polygons that
define individual farm plots) are foundational for agricul-
tural monitoring and decision-making, enabling downstream
applications such as crop type mapping, yield estimation,
pest and disease surveillance, and tracking of conservation
and climate programs [ 40 ]. Field boundary maps enhance
national agricultural statistics, providing spatially consistent
units for analysis across regions and seasons [42].
Accurate, up-to-date field boundary maps would trans-
form agriculture and food security applications, but are in-
feasible with existing methods. Manually delineating field
boundaries–typically through image interpretation–is slow,
labor-intensive, and must be regularly repeated as boundaries
shift due to land use and management changes [ 64 ], making
this approach impractical for large-scale monitoring [ 19 ]. Al-
ternatively, machine learning and computer vision methods
for satellite imagery enable fast, automated, and repeatable
extraction of field boundaries across large regions and over
time [ 13 , 34 ]. Satellite imagery offers wide spatial cover-
```
```
#Corresponding Author: gmuhawen@asu.edu
*Equal Contribution
```
# arXiv:2603.27101v1 [cs.CV] 28 Mar 2026


Figure 2. Example visualization of predictions over Illinois, USA
(top, MGRS tile 16TDL) and Mato Grosso, Brazil (bottom, MGRS
tile 20LRQ) with the FTW Baseline, Terramind and PRUE. The
FTW Baseline and Terramind models show strong sensitivity to
scene characteristics, producing discontinuous and noisy field
boundaries. The PRUE model achieves stable boundaries across
regions and imaging conditions.

age, frequent revisit intervals, and cost-effective (or free)
access to decades of historical observations, making it an
ideal data source for scalable agricultural field boundary
mapping [ 13 , 64 ]. These advantages have motivated the cre-
ation of several open, machine learning-ready field boundary
delineation datasets to accelerate progress and benchmarking
[13, 21, 26, 34, 45].
However, reliably delineating field boundaries in satellite
imagery remains a difficult computer vision problem. Unlike
typical instance segmentation tasks, field boundaries are of-
ten narrow and poorly defined, varying with crop phenology,
management practices, and imaging conditions [ 34 ]. Field
edges may appear discontinuous due to mixed pixels, cloud
shadows, or low contrast, especially in smallholder systems
where fields are heterogeneous and irregularly shaped. These
factors make field delineation a challenging and imperfectly
supervised problem, since even expert annotators may dis-
agree on true boundary locations [21].
Benchmarks such as Fields of The World (FTW) [ 34 ],
PASTIS [ 26 ], and AI4Boundaries [ 13 ] help advance research
progress on field boundary segmentation, but do not capture
errors that arise when models are deployed for large-scale
map-making. Applying the best FTW model from [ 34 ]
to map large, out-of-domain regions shows it is sensitive
to changes in brightness, temporal ordering, pixel resolu-
tion/scale, and receptive field (Figure 2 shows examples with
both FTW Baseline and the Terramind models), leading to
patch tiling artifacts and low-quality field boundary maps.
Several prior studies have noted similar robustness failures
in geospatial deep learning pipelines [11, 12, 49].
To overcome these obstacles to creating accurate, readily-
updated global field boundary datasets with AI, we need
robust and scalable models that are:

- Invariant to changes in brightness, spatial scale, seasonal
    differences, and translations in the input window;
- Computationally efficient (low cost) at inference time;
    - Performant across diverse agricultural contexts (e.g., very
       small to very large fields).
          To develop a model that meets these requirements, we
    performed a systematic evaluation of diverse segmentation
    and geospatial foundation model (GFM) architectures for
    field boundary delineation. We conducted experiments that
    compared an array of GFMs and semantic and instance seg-
    mentation architectures, as well as a broad sweep of U-Net
    model settings using both established performance metrics
    and a suite of new measures designed to assess the robustness
    of field segmentation under real-world conditions. These
    analyses result in three key contributions:
    1.A new state-of-the-art model for field boundary seg-
       mentation (PRUE), which outperforms 18 other models
       against the FTW benchmark, and has high zero-shot per-
       formance. Our model code and weights are available at
       https://github.com/fieldsoftheworld/ftw-prue.
    2.A new set of metrics for evaluating the robustness of field
       boundary segmentation models to real-world distribution
       shifts (e.g., changes in brightness or translation).
    3. Country-scale, multi-year field boundary maps
       for Japan, Mexico, Rwanda, South Africa, and
       Switzerland, which are publicly available online at
       https://source.coop/wherobots/fields-of-the-world.
          These demonstrate the real-world benefits of PRUE over
             previous models: maps predicted using PRUE are more
             accurate and reveal important landscape change.
          These contributions advance reliable, scalable, and trans-
    ferable models for delineating field boundaries, enabling
    the creation of globally consistent, automatically updated
    datasets that can strengthen equitable and sustainable agri-
    cultural decision-making [41].

### 2. Background and Related Work

```
Early approaches to delineate field boundaries from Earth
observations (EO) relied on rule-based image processing,
such as edge detection techniques and region-based segmen-
tation [ 51 , 69 ]. Although simple and computationally effi-
cient, these approaches are often ineffective, as their reliance
on low-level features makes them sensitive to illumination,
noise, and image heterogeneity [ 66 ]. These limitations moti-
vated a shift towards data-driven methods that capture spatial
context and semantic structure [44, 66].
The success of deep learning for image segmentation
has made it the dominant approach for agricultural field
delineation [ 44 , 63 ], which is fundamentally an instance
segmentation task focused on identifying individual field
polygons. The unique characteristics of satellite imagery
have led to diverse methodological approaches, which we
organize by their segmentation paradigm.
Semantic segmentation with post-processing. Many
pipelines adopt a semantic segmentation framework, classi-
```

fying each pixel into two or three classes (e.g., field interior,
boundary, background) using models such as random forests
[ 16 , 20 ], CNNs [ 44 ], and U-Nets [ 34 , 65 ]. The resulting
raster masks are then post-processed with thresholding, con-
nected component analysis, and/or watershed segmentation
to recover field instances [ 45 , 64 , 65 ]. Recent refinements in-
corporate multitask objectives, distance maps, or edge-aware
losses that learn contour features, better separate adjacent
fields, and provide more accurate polygonization [ 64 ]. These
developments have established U-Net and its variants as the
most common field delineation framework [34, 45, 64, 65].
This predominance of semantic segmentation stems from
the challenges of applying standard instance segmentation
methods to crop fields, which are often irregularly shaped
and densely packed [ 40 , 65 , 70 ], and thus lack the clear struc-
ture that object detection methods exploit through bounding-
box regression and non-maximum suppression [ 53 , 74 ]. Con-
versely, semantic outputs may be fragmented, and post-
processing introduces hyperparameters (e.g. connectivity
thresholds) that can affect instance quality.
Instance and panoptic segmentation. A smaller body of
work applies instance or panoptic segmentation architectures
directly, eliminating post-processing steps. Mask R-CNN
has been adapted for field delineation [ 28 , 39 ], but its re-
liance on bounding boxes and NMS may be suboptimal for
irregular geometries. Recently, Delineate Anything [ 37 ] fine-
tuned YOLOv11-seg for resolution-agnostic field boundary
prediction across multi-sensor European imagery. Field-
Seg used Segment Anything (SAM) to delineate Sentinel-
imagery in 8 study areas across 6 continents [ 24 ]. These stud-
ies showed that modern instance segmentation architectures
can perform well across diverse landscapes. The PASTIS
dataset [ 26 ] benchmarked panoptic methods on French field
parcels from Sentinel-2 time series, while PanopticFPN has
been applied to map land cover in aerial imagery [15].
Universal segmentation. Task-specific segmentation mod-
els lack flexibility to generalize across tasks. Universal seg-
mentation models such as Mask2Former [ 8 ] and OneFormer
[ 31 ] are capable of semantic, instance, and panoptic segmen-
tation, but are underexplored for satellite imagery.
Geospatial Foundation Models (GFMs). GFMs such as
SatMAE [ 10 ], DeCUR [ 67 ], Satlas [ 4 ], CROMA [ 25 ], Soft-
Con [ 68 ], DOFA-v1 [ 73 ], AnySat [ 1 ], Galileo [ 62 ], Prithvi
2.0 [ 58 ], Clay [ 9 ], TerraFM [ 14 ], TerraMind [ 32 ], and Al-
phaEarth Foundations [ 6 ] are pretrained on global-scale EO
archives, using contrastive or masked modeling objectives.
Their pretrained encoders provide general-purpose repre-
sentations that capture land-cover semantics, phenological
patterns, and surface-texture variations across geographies.
GFM embeddings can be leveraged to enhance semantic
segmentation performance [ 62 ], particularly under domain
shift or limited labeled data. Previous work has not evalu-

```
ated GFMs for field boundary segmentation across FTW’s
diverse regions.
```
### 3. Methods

```
To address the challenges to robust, large-scale inference, we
systematically evaluated a range of models against the estab-
lished FTW benchmark dataset, introducing augmentations
and metrics to improve and assess deployment reliability.
```
#### 3.1. Dataset

```
The Fields of The World (FTW) dataset [ 34 ] is a globally
distributed benchmark for agricultural field boundary delin-
eation, with over 1.5 million geo-referenced, manually vali-
dated field polygons from 24 countries across four continents.
Each polygon is paired with bi-temporal RGB-NIR (RGBN)
Sentinel-2 imagery from the planting and harvest seasons.
We used FTW’s predefined datataset splits, which reduce
spatial autocorrelation and enable robust cross-regional eval-
uation. FTW provides a training, validation, and test split
for each of the 24 countries in the dataset. Kerner et al.[33]
recommends benchmarking models using the average of the
individual test accuracies for each country.
```
#### 3.2. Model architecture search

```
We approached model development as a “bake-off,” system-
atically evaluating an extensive set of semantic segmentation,
instance segmentation, and GFM model configurations to
find the most practical and robust recipe for field boundary
segmentation at scale. We named the winner of the bake-off
PRUE: A Practical Recipe for Field Boundary Segmentation
at Scale. For each model configuration, we used the orig-
inal training settings, including reported hyperparameters
and preprocessing, and swept learning rates to ensure fair
and reproducible comparison across model families. This
approach allowed us to identify the strongest baseline model
for subsequent development.
Semantic segmentation baselines. We used the highest-
performing model reported by Kerner et al.[34]on the FTW
benchmark, a U-Net with EfficientNet-B3 backbone, as the
“FTW baseline”. We also evaluated DECODE [ 64 ], which
uses a multitask FracTAL-ResUNet to jointly predict field
extent, boundaries, and distance maps. As in Kerner et al.
[33], we masked pixels with unknown labels during training
for presence-only examples in FTW. We post-processed all
outputs using connected components to extract individual
field instances for object-level evaluation (see § 3.4).
Instance and panoptic segmentation baselines. We evalu-
ated Delineate Anything [ 37 ], SAM [ 35 ], and Mask2Former
(M2F) [ 8 ]. Since Delineate Anything is pretrained for field
boundary segmentation and intended to be used in a zero-
shot setting, we evaluated it and its smaller variant Del-Any
```

S directly on FTW using the RGB channels from the plant-
ing season image. We evaluated SAM in both zero-shot and
fine-tuned settings. SAM and M2F (panoptic task) were
fine-tuned on the same 8-channel bitemporal input as the
U-Net baseline. Due to differences in how instance seg-
mentation models handle training objectives compared to
semantic models, we did not mask presence-only examples
in training. See supplement A for additional results.
Geospatial foundation models (GFMs). We computed
embeddings from pretrained GFM encoders: Galileo [ 62 ],
CROMA [ 25 ], SoftCon [ 68 ], Prithvi 2.0 [ 58 ], DOFA-v1 [ 73 ],
DeCUR [ 67 ], Satlas [ 4 ], Clay [ 9 ], DINOv3 [ 57 ], TerraFM
[ 14 ], and TerraMind [ 32 ]. We obtained token features from
pretrained, frozen GFMs for each temporal window. We
fused the two windows by concatenating tokens along the
feature dimension, and passed the fused tokens through a
3-layer MLP, allowing the model to integrate information
from both frames at the token level before decoding. To
overcome poor segmentation performance resulting from a
1×1 convolution followed by bilinear upsampling (see Sup-
plemental B), we adopted a decoder composed of a 3×3 con-
volutional projection layer, two residual refinement blocks,
and a multi-scale convolutional module that expands spatial
context, followed by pixel-shuffle upsampling [ 55 ] to pro-
duce dense segmentation masks. We used this MLP-based
fusion combined with the enhanced convolutional decoder
for all GFM experiments with results reported in Table 1.

#### 3.3. Model design space exploration

Building on the best-performing, most parameter-efficient
baseline with high per-km^2 throughput, we explored the
model design space to identify components critical for accu-
rate, scalable, and robust field boundary delineation. Rather
than exhaustively re-tuning every architecture, we used this
baseline as a controlled reference point, ensuring that ob-
served trends reflect genuine design effects rather than differ-
ences in capacity or optimization. We systematically varied
architectural, data, and optimization factors to isolate their
individual impacts on benchmark accuracy and real-world
robustness, with each sweep varying a single factor at a time.
Architectures and backbones. We compared multiple en-
coder–decoder architectures, including FCN [ 38 ], UPer-
Net [ 71 ], FCSiam [ 7 ], and U-Net [ 50 ] variants. We evaluated
all U-Net variants with EfficientNet backbones (B3-B7) [ 59 ]
and Mix Vision Transformers (B2-B5) [ 72 ] to evaluate the
impact of increasing capacity relative to the FTW baseline
(which used an EfficientNet B3 backbone).

```
Loss functions. We evaluated a range of losses commonly
used for field boundary and class imbalance segmentation
problems, including cross-entropy (CE), Dice, log-cosh Dice,
focal, Tversky, Jaccard, and Fractal Tanimoto (FTNMT) loss
functions, as well as their weighted variants [ 2 , 17 , 30 , 52 ].
```
```
Class weights. We further examined the impact of class
reweighting by varying the boundary class importance factor
ωin steps of 0.05 within the range[0. 60 , 0 .85]. The normal-
ized class weights for the three output classes (background,
interior, and boundary) were defined as[0. 05 , 0. 95 − ω,ω],
respectively. This sweep allowed us to assess the sensitivity
of model training to the relative emphasis placed on thin
boundary regions versus field interiors.
```
```
Learning rate. We swept learning rates logarithmically
to identify stable optimization regimes, testing values in
{ 10 −^4 , 3 × 10 −^4 , 3 × 10 −^3 , 10 −^2 , 3 × 10 −^2 }(orders of mag-
nitude common for Adam optimizers in segmentation tasks).
We trained each under identical conditions to evaluate con-
vergence stability and sensitivity to step size.
```
```
Data augmentations. To address the observed sensitivity
of models to variations in brightness, spatial scale, and tiling
boundaries, we performed a systematic sweep of data aug-
mentations designed to improve robustness along four key
dimensions: input order invariance, brightness robustness,
scale robustness, and spatial consistency. We applied each
augmentation independently or in combination to evaluate its
effect on these robustness properties, which were quantified
using the deployment-oriented metrics from § 3.5.
```
#### 3.4. Evaluation metrics for model comparison

```
Beyond pixel-level metrics of IoU and F1-score, we report
object-level precision and recall from polygonized predic-
tions. We averaged each metric over the individual country
test sets in FTW. We excluded “presence-only” countries
from our evaluation, since only recall can be computed for
those countries. To measure computational efficiency, we
report the inference throughput (in km^2 /s) and number of
parameters for each model.
We computed object precision/recall at a 0.5 confidence
threshold, which indicates expected performance under de-
fault inference settings. To evaluate the precision-recall
tradeoff of changing this threshold, we also computed COCO
AP 0. 5 and AP 0 .5:0. 95. AP 0. 5 integrates precision across all
confidence thresholds at IoU=0.5, and AP 0 .5:0. 95 averages
this across multiple IoU thresholds ({ 0. 5 , 0. 55 ,.. ., 0. 95 }) to
evaluate both confidence calibration and localization quality.
For semantic segmentation models, which do not pro-
duce per-instance confidence scores, we used the mean soft-
max probability across all pixels in each polygon created
by argmaxing the probability map as a proxy for instance
confidence. Instance segmentation models such as Delin-
eate Anything produce confidence scores with each object
detection. SAM has a proxy score of predicted IoU, which is
the model’s own prediction of mask quality. Similarly, mod-
els with a panoptic inference head, such as Mask2Former,
output a score for each predicted ‘thing’ and ‘stuff’ segment.
```

#### 3.5. Deployment-oriented model metrics

In geospatial machine learning (GeoML), models are typ-
ically trained on patch-based datasets and evaluated with
the standard computer vision metrics described above (e.g.
precision, recall, IoU, and average precision) on held-out
patches. However, at deployment time, artifacts can arise
when these models are used to construct large maps from
entire image scenes [ 29 , 78 ]. Model performance beyond
patch-level measures are critical [ 49 ]. Specifically, models
must be robust to tiling artifacts, invariant to input ordering
and preprocessing conventions, and stable under moderate
changes in spatial scale.
We propose deployment-oriented metrics to complement
standard performance metrics. These metrics aim to char-
acterize model behavior at deployment time. These metrics
can be computed on the predefined dataset splits used during
model development, giving practitioners insight into how
models are likely to behave when tiled over large scenes at
inference time.

Consistency under translations. Modern CNNs and ViTs
are not translation equivariant in practice [ 5 , 18 , 27 ]. Small
shifts in input can lead to large changes in output due to
padding choices, aliasing from strided downsampling, and
architectural elements, such as absolute positional encod-
ings [ 3 , 56 , 76 ]. In patch-based prediction pipelines, this
sensitivity manifests as visible grid artifacts when indepen-
dently processed patches are stitched into a large map [ 29 ].
Prior work recommends several approaches to reduce trans-
lation sensitivity, including strategies that average overlap-
ping logits, and variants of sliding-window inference with
buffered borders that are discarded during stitching [29].
We follow work that measures translation robustness via
consistency of model outputs under shifts [ 76 , 77 ] and adapt
it to patch-based geospatial semantic segmentation. Specifi-
cally, we extend the idea of mean Average Semantic Segmen-
tation Consistency from [ 77 ] and compute prediction agree-
ment across four overlapping corner crops of each patch,
rather than two global crops, to better mimic the tiling setup
used to create maps at inference time.
Letx∈RC×S×Sdenote an input patch with height and
widthS. We choose a crop sizepsuch thatS/ 2 < p < Sand
take four crops ofx, one anchored at each corner, constructed
so that they share a central overlapping regionΩbut differ
in their surrounding spatial context. Let ̃(yk)denote the hard
per-pixel predictions (e.g., after argmax over logits) obtained
from a segmentation model on thek-th crop and further
cropping the result to the areaΩ. We define the consistency
ofxas the fraction of pixels in the overlapping region whose
predicted labels agree across all four crops:

```
1
|Ω|
```
##### X

```
u∈Ω
```
##### 1

```
h
̃y(1)(u) = ̃y(2)(u) = ̃(3)y (u) = ̃y(4)(u)
```
```
i
(1)
```
```
Figure 3. Spatial consistency. Four overlapping crops from each
image corner are independently segmented. The consistency mask
shows pixel-level agreement, with yellow indicating unanimity
across all four predictions, and purple disagreement. This metric
quantifies grid artifact resistance for large-scale field delineation.
```
```
whereu ∈ Ωindexes over the pixels in the shared area. A
consistency of 1 implies the model is perfectly translation-
equivariant within the range of shifts implied by the corner
crops, while lower values correspond to stronger sensitivity
to small translations of the input. Figure 3 illustrates the
cropping scheme and overlapping regionΩused to compute
Eq. 1.
Sensitivity to input ordering. Many map prediction prob-
lems can be defined to operate on small sets of co-registered
observations (e.g., multi-temporal Sentinel-2 scenes or multi-
sensor stacks) that are most naturally viewed as unordered
sets rather than ordered sequences. In such cases, we would
like model predictions to be permutation invariant with re-
spect to the ordering of the input elements [ 54 , 75 ]. For
example, the FTW dataset [ 34 ] provides paired Sentinel-
observations from the planting and harvesting stages of the
growing season. In the official implementation accompany-
ing the dataset, the input tensor is constructed by stacking
the bands in a canonical (planting, harvest) ordering [ 16 , 20 ].
At inference time, however, practitioners may select different
scenes based on data availability, cloud coverage, and pro-
cessing level, and may inadvertently permute the temporal
order, causing models that are trained in the standard way to
fail. For GeoML tasks where inputs consist of observations
collected at different times, but preserving temporal order-
ing is not important, we argue that models should ideally
be insensitive to such permutations. Thus, to quantify this
property we define an input-order sensitivity metric.
Letπ 0 denote the “reference” ordering of input channels
(e.g., the training-time convention), andΠbe a set of alterna-
tive permutations of those channels, i.e. the reverse ordering
for a pair of observations. For a given evaluation sample
(x,y)and performance metricm(e.g., IoU), we compute
```
```
mref(x,y) = m
```
##### 

```
f (xπ^0 ), y
```
##### 

##### , (2)

```
mperm(x,y) =
```
##### 1

##### |Π|

##### X

```
π∈Π
```
```
m
```
##### 

```
f (xπ), y
```
##### 

##### , (3)


wherexπdenotes the input with channels reordered accord-
ing to permutationπ. The per-sample order sensitivity is
then∆order(x,y) = mref(x,y)−mperm(x,y)and the dataset-
level input-order sensitivity is the average absolute drop in
performance across samples.
Robustness to preprocessing conventions. Satellite im-
agery is distributed and processed under a variety of ra-
diometric conventions. For example, Sentinel-2 Level-2A
products are typically stored as quantized digital numbers
with scale factors (e.g., division by 10 , 000 ). Starting from
Processing Baseline 04.00 (in February 2022), an additive ra-
diometric offset (e.g.,BOA_ADD_OFFSET) must be applied
when converting to physical reflectance [ 22 , 23 ]. Down-
stream pipelines may additionally re-scale or normalize the
data (e.g., dividing by 3 , 000 instead of 10 , 000 , or dividing
per band by a dataset percentile).
In practice, training and inference pipelines for GeoML
models often do not share identical preprocessing steps, es-
pecially when models are reused across organizations, code-
bases, or data providers. To assess how brittle a model is to
such variations, we define a preprocessing invariance metric.
Letgrefdenote the “reference” normalization used dur-
ing training (e.g., radiometric offset correction followed by
division by 10 , 000 ), and let{gj}Jj=1denoteJalternative
normalizations that reflect plausible deploy-time choices
(e.g., different scale factors, omission of offsets, or simple
min–max scaling). For each (x,y), we compute

```
mref(x,y) = m
```
##### 

```
f (gref(x)), y
```
##### 

##### , (4)

```
mj(x,y) = m
```
##### 

```
f (gj(x)), y
```
##### 

```
for j = 1,...,J. (5)
```
We then define the per-sample preprocessing sensitivity as
∆prep(x,y) =J^1

##### PJ

```
j=
```
(^) mref(x,y)−mj(x,y)
(^). The dataset-
level preprocessing sensitivity is the average absolute drop
in performance across samples.
Sensitivity to spatial scale. GeoML models are frequently
trained at a fixed spatial resolution (e.g., 10 m Sentinel-
pixels) but may be deployed on imagery with different effec-
tive resolution (e.g., resampled Sentinel-2, or PlanetScope
mosaics). Recent self-supervised pretraining approaches for
remote sensing explicitly condition ViT models on spatial
scale [ 48 ]. We measure scale sensitivity under test-time
resizes in a method similar to the previous metrics. We com-
pute the performance difference between the model run with
standard inputs versus resized inputs, ∆scale(x,y).

### 4. Results and Analysis

Our results reveal several key insights regarding the perfor-
mance of different model architectures and design choices
(§4.1) based on traditional metrics and our proposed
deployment-oriented metrics (§4.2).

#### 4.1. Model performance & architecture comparison

```
Table 1 compares our model against semantic, instance, and
GFM model baselines on the FTW benchmark.
Architecture families reveal task-dependent strengths.
Semantic segmentation models, particularly U-Net variants,
remain strong performers on pixel-level and object-level
metrics, outperforming instance-based and GFM approaches
in pixel-level IoU, precision and recall. Instance segmen-
tation models (Delineate Anything, SAM) achieve reason-
able performance in specific contexts with zero-shot set-
tings. M2F shows competitive object-level performance
(precision=0.62, F1=0.39) but lower pixel-level predictions
(IoU=0.68). Given their slower inference speeds, these mod-
els are less practical for operational deployment; we focused
on optimizing the U-Net baseline which exhibits both high
accuracy metrics and high throughput (623.28 km^2 /s).
GFMs paired with semantic decoders achieved moderate
performance, but generally underperformed specialized ar-
chitectures, despite having 3-10×more parameters than the
U-Net baseline. Clay (ViT-L) was the best GFM performer
(IoU=0.67, F1=0.36), but was still 9% and 11% lower than
our optimized U-Net (PRUE). This discrepancy is likely due
to the lower effective resolution of GFM encoders, which out-
put coarse-scale patch-wise embeddings. Extended results
from GFM experiments are shared in the Supplement A.
Systematic optimization matters more than architectural
choice. The architecture design search detailed in §3.
shows that increasing encoder depth and choosing an ap-
propriate loss function can substantially improve delineation
quality. Log-cosh Dice loss produces smoother optimiza-
tion and superior boundary completeness compared to other
losses, while moderate boundary class weighting (ω = 0. 75 )
yields the most balanced precision–recall trade-off. Com-
bining these with brightness and resize augmentations pro-
duces consistent performance across domains, confirming
that robustness emerges from the interaction of architecture,
training objective, and data design choices.
```
#### 4.2. Final model selection and robustness evaluation

```
Our final model, PRUE, integrates the best-performing
design choices: U-Net decoder with EfficientNet-B7 en-
coder, channel shuffling for input-order invariance, bright-
ness and resize augmentations, log-cosh Dice loss, and
boundary weightingω = 0. 75. Table 1 compares our final
model against other backbone architectures; PRUE achieves
IoU=0.76 and object F1=0.47, representing+6%and+9%
improvements over the FTW baseline. We selected U-Net
over FCSiam due to higher temporal consistency.
Table 2 shows that PRUE is the most robust configuration
across all deployment-oriented perturbations (§ 3.5). Bright-
ness and resize augmentations reduce sensitivity to illumina-
tion and scale changes, while channel shuffling eliminates
```

Table 1. Performance comparison across model families on FTW test set (excluding presence-only countries). Semantic baselines:
Post-processed with connected components, using presence-only label masking in training [ 34 ]. Instance/panoptic models: Fine-tuned
for 8-channel input and FTW-specific classes but without presence-only masking due to architecture constraints. GFM models: Frozen
encoders with our trained convolutional decoder. Total parameter count includes both the frozen encoder (28M–300M) and our learnable
components, which consist of a frames-fusion module and a convolutional decoder (an additional 30M–110M parameters, depending on the
encoder’s output dimensionality). Bold indicates best performance,underlineindicates second-best. Throughput measured on a V100-32GB
GPU with batch size 64. *Galileo did not fit into VRAM at any batch size.

```
Model Backbone Pixel-level Object-level #Params Throughput
IoU↑ Prec↑ Recall↑ Prec↑ Recall↑ F1↑ AP 0 .5:0. 95 ↑ AP 0. 5 ↑ (M)↓ (km^2 /s)↑
Semantic segmentation baselines
FTW-Baseline U-Net+EfficientNet-B3 0.70 0.90 0.72 0.40 0.37 0.38 0.22 0.39 13.2 623.
DECODE FracTALResUNet 0.71 0.83 0.83 0.27 0.17 0.21 0.09 0.17 64.8 113.
Instance & panoptic segmentation
Mask2Former Swin-S (fine-tuned, 8ch) 0.68 0.88 0.75 0.62 0.30 0.39 0.28 0.44 68.8 26.
SAM ViT-Huge (fine-tuned, 8ch) 0.45 0.73 0.54 0.56 0.34 0.37 0.21 0.19 642.7 0.
SAM ViT-Huge (zero-shot, 3ch) 0.32 0.36 0.82 0.14 0.34 0.17 0.06 0.12 641.1 0.
Del-Any YOLOv11 (zero-shot, 3ch, win A) 0.37 0.53 0.56 0.25 0.05 0.09 0.05 0.10 56.9 87.
Del-Any S YOLOv11 (zero-shot, 3ch, win A) 0.44 0.52 0.73 0.15 0.06 0.08 0.07 0.14 2.6 389.
Geospatial foundation models
Clay ViT-Large 0.67 0.91 0.72 0.38 0.36 0.36 0.24 0.41 363.8 10.
Galileo ViT-Base 0.66 0.86 0.72 0.29 0.36 0.32 0.21 0.37 119.0 *
DINOv3 ViT-Large 0.60 0.90 0.64 0.39 0.27 0.31 0.20 0.35 412.2 46.
TerraMind ViT-Base 0.57 0.88 0.62 0.30 0.24 0.26 0.17 0.31 189.1 123.
Prithvi 2.0 ViT-Large 0.56 0.88 0.60 0.29 0.23 0.25 0.16 0.30 439.5 63.
TerraFM ViT-Base 0.57 0.86 0.62 0.29 0.23 0.25 0.16 0.29 218.6 110.
CROMA ViT-Base 0.52 0.86 0.56 0.26 0.18 0.21 0.13 0.25 137.0 133.
SoftCon ViT-Small 0.52 0.85 0.57 0.24 0.18 0.21 0.12 0.24 101.8 234.
DeCUR ViT-Small 0.49 0.85 0.53 0.23 0.16 0.19 0.11 0.22 120.2 271.
DOFA-v1 ViT-Large 0.49 0.85 0.53 0.21 0.15 0.17 0.10 0.19 446.0 64.
Satlas Swin-Tiny 0.45 0.79 0.50 0.13 0.11 0.12 0.07 0.14 131.7 79.
PRUE (ours) U-Net+EfficientNet-B7 0.76 0.89 0.83 0.62 0.40 0.47 0.26 0.40 67.1 306.
```
Table 2. Ablation results for controlled experiments on FTW test set (excluding presence-only countries) in which each row varies a
single design choice (data augmentations, class weighting, encoder, loss function, or architecture). The Combination rows report the
best-performing joint configurations for the FCSiam and U-Net models. Bold indicates best performance,underlineindicates second-best.

```
Category Ablation Performance Input order Brightness Scale Agree.
Object F1↑ Pixel IoU↑ F1|∆|↓ IoU|∆|↓ F1|∆|↓ IoU|∆|↓ F1|∆|↓ IoU|∆|↓ Avg↑
FTW-Baseline 0.39± 0.08 0.68± 0.08 0.07 0.11 0.04 0.05 0.15 0.12 0.
Data augs Brightness+Resize 0.38± 0.08 0.66± 0.09 0.06 0.10 0.02 0.03 0.00 0.01 0.
Data augs Channel shuffle 0.39± 0.07 0.68± 0.09 0.00 0.00 0.04 0.05 0.17 0.14 0.
Class weights ω = 0. 75 0.42± 0.06 0.74± 0.07 0.08 0.11 0.07 0.07 0.29 0.15 0.
Encoder EfficientNet-B7 0.42± 0.07 0.71± 0.08 0.07 0.09 0.03 0.04 0.20 0.13 0.
Loss function log-cosh Dice 0.44± 0.07 0.77± 0.06 0.09 0.13 0.06 0.05 0.36 0.20 0.
Architecture FCSiam 0.40± 0.07 0.69± 0.08 0.00 0.00 0.05 0.06 0.22 0.14 0.
```
```
Combination FCSiam comboPRUE (U-Net) 0.440.47±± 0.07 0.07 0.750.76±± 0.07 0.08^ 0.000.00 0.000.00^ 0.040.00 0.050.00 0.050.01 0.020.01 0.940.
```
```
input-order dependency. Compared to the FTW baseline,
PRUE demonstrates negligible variance under input order
and brightness shifts, and Figure 2 shows PRUE’s improved
generalization compared to the baseline model. These re-
sults validate that robustness, accuracy, and scalability are
not competing objectives but can be co-optimized through
deliberate model and data design. Extended per-country
results and ablation analyses are in Supplement C.
```
### 5. AI-Derived Field-Boundaries at Scale

```
Country-scale field boundaries. We used PRUE to gen-
erate complete field boundary maps in 2023 and 2024 for
five countries: Japan, Mexico, Rwanda, South Africa, and
Switzerland, covering over 4.76 million km^2. These coun-
tries were selected to cover diverse climatic zones, farming
practices, field sizes, and agricultural systems, representing
realistic global deployment scenarios. Figure 1 shows exam-
```

Table 3. Agricultural field statistics derived using PRUE. The
table reports total land area, distributed inference cost, field counts,
and median field area in hectares for five diverse countries.

Country Area processed(million km (^2) ) Cost($) Year Fields(M) area (ha)Median
Rwanda 0.02 3.23^20232024 0.180.26 0.050.
Switzerland 0.09 5.11^20232024 0.360.36 0.320.
Japan 0.65 7.59^20232024 1.551.61 0.200.
South Africa 1.60 8.03^20232024 2.992.70 0.080.
Mexico 2.39 8.26^20232024 5.906.57 0.090.
ples from Japan. The resulting maps show that the model
preserves field topology, maintains coherence across tile
boundaries, and generalizes without retraining or regional
fine-tuning, thus demonstrating both scalability and zero-
shot transferability. Table 3 gives country-level statistics
derived from the mapped field boundaries.
Mosaicking and inference pipeline. To enable large-scale
mapping, we developed a pipeline for national-scale infer-
ence that emphasizes throughput, spatial consistency, cost
efficiency, and reproducibility. The first step generates cloud-
free, seasonally aligned Sentinel-2 mosaics using a tiling
framework and latitude-based planting/harvest season selec-
tion algorithms (see Supplement D). Patches of 256× 256
are then read and processed by the model with a 25% overlap,
with Gaussian-weighted averaging applied across overlaps
with an apodization kernel [ 61 ] that places greater weight on
predictions near patch centers, which reduces edge artifacts
and ensures consistent boundary logits near patch borders.
The stitched probability maps are then vectorized in a block-
wise manner using 4,096 × 4,096-pixel windows to maintain
memory efficiency. Finally, resulting polygons are serialized
in thefiboa[ 60 ] GeoParquet format, enabling efficient
downstream querying, temporal indexing, and large-scale
analytics. This pipeline enables production of contiguous,
artifact-free field boundary layers at a country scale.
Throughput and cost efficiency. Table 3 reports the total
land area processed and inference cost for each country. Cre-
ating the planting and harvest mosaics for two agricultural
seasons (2023/24-2024/25) takes 49.3 minutes. Our infer-
ence pipeline is optimized to maximize GPU utilization in
a cluster pool of up to 256 NVIDIA A10G GPUs (AWS
g5.xlarge instances). Processing the 2-year Mexico mosaic
resulted in an execution time of only 14.23 min at a cost
of $8.26 ($1.05× 10 −^6 /km^2 ), consisting of 4.8 GPU-Hrs,
77.9 Core-Hrs, and a throughput of 232.4 GB-Hrs. Using
the Machine Learning Impact calculator [ 36 ], we estimate
this run generated a total emission of 0.25 kgCO 2 eq, 100%
of which was offset by AWS.
Field boundary change segmentation To quantify struc-
tural changes in agricultural landscapes, we computed field-
level change directly from the model’s multi-year semantic
predictions, which consist of raster logits produced for the
field class for each country and year. We computed the
absolute difference between the two, resulting in a change
magnitude map. We then min–max normalized and thresh-
olded the change at 0.5 to obtain a binary change mask.
Figure 1 shows an example change map for Japan. Supple-
ment E and F provide additional visualizations and change
detection examples across five countries.

### 6. Conclusions

```
We presented a systematic study of model architectures, train-
ing strategies, and robustness evaluation metrics for large-
scale agricultural field boundary delineation. Our study
establishes a new state-of-the-art on the FTW benchmark,
introduces deployment-oriented metrics that better reflect
real-world behavior, and demonstrates operational viability
through country-scale deployments.
```
```
Design choices that matter. Loss functions, boundary
weighting, and targeted augmentations have the strongest im-
pact on accuracy and robustness. Log-cosh Dice and moder-
ate boundary weighting improve boundary completeness and
precision–recall balance, while augmentations targeting de-
ployment failures (brightness, scale, channel shuffling) yield
measurable gains. Though explored only for U-Net variants,
this design-space methodology is architecture-agnostic and
applicable to instance, panoptic, and GFM-based models.
GFMs need task-specific adaptation. Despite broad EO
pretraining, GFMs generally underperform specialized seg-
mentation models due to resolution limits and weaker local-
ization. They require high-resolution decoders and boundary-
aware objectives to match task-specific architectures.
Robustness metrics predict real-world performance.
Standard metrics do not capture translation sensitivity,
input-order dependence, or radiometric brittleness. Our
deployment-oriented metrics quantify these behaviors and
guide targeted improvements. Models optimized with these
metrics show reduced sensitivity to brightness, scale, and
translation. Country-scale deployments confirm that higher
robustness scores correlate with fewer artifacts and more
stable performance across millions of km^2.
```
### Acknowledgments

```
This project was supported by funding from Taylor Geospa-
tial. ZF was supported by funding from NASA’s Land Cover
Land-Use Change program, award #80NSSC23K0528. We
appreciate the feedback and suggestions on this work pro-
vided by Fuxin Li and Jamon Van Den Hoek.
```

### References

```
[1]Guillaume Astruc, Nicolas Gonthier, Clément Mallet, and
Loic Landrieu. AnySat: One Earth Observation Model for
Many Resolutions, Scales, and Modalities. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), pages 19530–19540, 2025. 3
[2]Reza Azad, Moein Heidary, Kadir Yilmaz, Michael Hütte-
mann, Sanaz Karimijafarbigloo, Yuli Wu, Anke Schmeink,
and Dorit Merhof. Loss Functions in the Era of Semantic
Segmentation: A Survey and Outlook, 2023. 4
[3]Aharon Azulay and Yair Weiss. Why do deep convolutional
networks generalize so poorly to small image transforma-
tions? Journal of Machine Learning Research, 20(184):1–25,
```
2019. 5
[4]Favyen Bastani, Piper Wolters, Ritwik Gupta, Joe Ferdinando,
and Aniruddha Kembhavi. SatlasPretrain: A Large-Scale
Dataset for Remote Sensing Image Understanding. In Pro-
ceedings of the IEEE/CVF International Conference on Com-
puter Vision (ICCV), pages 16772–16782, 2023. 3, 4, 1
[5]Valerio Biscione and Jeffrey S. Bowers. Convolutional neural
networks are not invariant to translation, but they can learn
to be. Journal of Machine Learning Research, 22(229):1–28,
2021. 5
[6]Christopher F. Brown, Michal R. Kazmierski, Valerie J.
Pasquarella, William J. Rucklidge, Masha Samsikova, Chen-
hui Zhang, Evan Shelhamer, Estefania Lahera, Olivia Wiles,
Simon Ilyushchenko, Noel Gorelick, Lihui Lydia Zhang,
Sophia Alj, Emily Schechter, Sean Askay, Oliver Guinan,
Rebecca Moore, Alexis Boukouvalas, and Pushmeet Kohli.
AlphaEarth Foundations: An Embedding Field Model for
Accurate and Efficient Global Mapping from Sparse Label
Data, 2025. 3
[7]Rodrigo Caye Daudt, Bertr Le Saux, and Alexandre Boulch.
Fully Convolutional Siamese Networks for Change Detec-
tion. In 2018 25th IEEE International Conference on Image
Processing (ICIP), pages 4063–4067, 2018. 4
[8]Bowen Cheng, Ishan Misra, Alexander G Schwing, Alexander
Kirillov, and Rohit Girdhar. Masked-Attention Mask Trans-
former for Universal Image Segmentation. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 1290–1299, 2022. 3, 1, 4
[9]Clay. The Clay Foundation Model - An open source AI model
and interface for Earth, 2025. Accessed: 2025-11-13. 3, 4, 1
[10]Yezhen Cong, Samar Khanna, Chenlin Meng, Patrick Liu,
Erik Rozi, Yutong He, Marshall Burke, David B. Lobell,
and Stefano Ermon. SatMAE: Pre-training transformers for
temporal and multi-spectral satellite imagery. In Advances in
Neural Information Processing Systems, 2022. 3
[11]Isaac Corley, Caleb Robinson, Rahul Dodhia, Juan M. Lav-
ista Ferres, and Peyman Najafirad. Revisiting pre-trained
remote sensing model benchmarks: resizing and normaliza-
tion matters. In 2024 IEEE/CVF Conference on Computer
Vision and Pattern Recognition Workshops (CVPRW), pages
3162–3172, 2024. 2
[12]Isaac Corley, Caleb Robinson, and Anthony Ortiz. A Change
Detection Reality Check. In ICLR 2024 Machine Learning
for Remote Sensing (ML4RS) Workshop, 2024. 2

```
[13]R. d’Andrimont, M. Claverie, P. Kempeneers, D. Muraro,
M. Yordanov, D. Peressutti, M. Batiˇc, and F. Waldner.
AI4Boundaries: An Open AI-Ready Dataset to Map Field
Boundaries with Sentinel-2 and Aerial Photography. Earth
System Science Data, 15(1):317–329, 2023. 1, 2
[14]Muhammad Sohail Danish, Muhammad Akhtar Munir, Syed
Roshaan Ali Shah, Muhammad Haris Khan, Rao Muhammad
Anwer, Jorma Laaksonen, Fahad Shahbaz Khan, and Salman
Khan. TerraFM: A scalable foundation model for unified mul-
tisensor earth observation. In The Fourteenth International
Conference on Learning Representations, 2026. 3, 4, 1
[15]Osmar Luiz Ferreira de Carvalho, Osmar Abílio de Car-
valho Júnior, Cristiano Rosa e Silva, Anesmar Olino de Albu-
querque, Nickolas Castro Santana, Dibio Leandro Borges,
Roberto Arnaldo Trancoso Gomes, and Renato Fontes
Guimarães. Panoptic Segmentation Meets Remote Sensing.
Remote Sensing, 14(4):965, 2022. 3
[16]Stephanie R Debats, Dee Luo, Lyndon D Estes, Thomas J
Fuchs, and Kelly K Caylor. A Generalized Computer Vision
Approach to Mapping Crop Fields in Heterogeneous Agri-
cultural Landscapes. Remote Sensing of Environment, 179:
210–221, 2016. 3, 5
[17]Foivos I. Diakogiannis, François Waldner, and Peter Caccetta.
Looking for Change? Roll the Dice and Demand Attention.
Remote Sensing, 13(18), 2021. 4
[18]Peijian Ding, Davit Soselia, Thomas Armstrong, Jiahao Su,
and Furong Huang. Reviving Shift Equivariance in Vision
Transformers, 2023. 5
[19]Hakan Erden, Murat Aslan, and Cemre Bahar Özcanli. To
establish a new subsidy system. In 2015 Fourth International
Conference on Agro-Geoinformatics (Agro-geoinformatics),
pages 57–60, 2015. 1
[20]Lyndon D Estes, Su Ye, Lei Song, Boka Luo, J Ronald
Eastman, Zhenhua Meng, Qi Zhang, Dennis McRitchie,
Stephanie R Debats, Justus Muhando, et al. High Resolution,
Annual Maps of Field Boundaries for Smallholder-Dominated
Croplands at National Scales. Frontiers in Artificial Intelli-
gence, 4:744863, 2022. 3, 5
[21]L. D. Estes, A. Wussah, M. Asipunu, M. Gathigi, P. KovaÄiÄ,
J. Muhando, B. V. Yeboah, F. K. Addai, E. S. Akakpo, M. K.
Allotey, P. Amkoya, E. Amponsem, K. D. Donkoh, N. Ha,
E. Heltzel, C. Juma, R. Mdawida, A. Miroyo, J. Mucha, J.
Mugami, F. Mwawaza, D. A. Nyarko, P. Oduor, K. N. Ohe-
meng, S. I. D. Segbefia, T. Tumbula, F. Wambua, G. H. Xe-
flide, S. Ye, and F. Yeboah. A Region-Wide, Multi-Year Set
of Crop Field Boundary Labels for Africa, 2024. 2
[22]European Space Agency. Introduction of Additional
Radiometric Offset in Processing Baseline 04.00 Prod-
ucts. https://forum.step.esa.int/t/info-
introduction-of-additional-radiometric-
offset-in-pb04-00-products/35431, 2022. Ac-
cessed 9 November 2025. 6
[23]European Space Agency. Sentinel-2 Processing Baseline and
Product Format.https://sentiwiki.copernicus.
eu/web/s2-processing, 2022. Accessed 9 November
```
2025. 6
[24]Lucas B Ferreira, Vitor S Martins, Uilson RV Aires, Nuwan
Wijewardane, Xin Zhang, and Sathish Samiappan. FieldSeg:


A scalable agricultural field extraction framework based on
the Segment Anything Model and 10-m Sentinel-2 imagery.
Computers and Electronics in Agriculture, 232:110086, 2025.
3
[25]Anthony Fuller, Koreen Millard, and James Green. CROMA:
Remote Sensing Representations with Contrastive Radar-
Optical Masked Autoencoders. Advances in Neural Infor-
mation Processing Systems, 36, 2024. 3, 4, 1
[26]Vivien Sainte Fare Garnot and Loic Landrieu. Panoptic
Segmentation of Satellite Image Time Series with Convo-
lutional Temporal Attention Networks. In Proceedings of
the IEEE/CVF International Conference on Computer Vision,
pages 4872–4881, 2021. 2, 3
[27]Nate Gruver, Marc Anton Finzi, Micah Goldblum, and An-
drew Gordon Wilson. The Lie Derivative for Measuring
Learned Equivariance. In The Eleventh International Confer-
ence on Learning Representations, 2023. 5
[28]Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Gir-
shick. Mask R-CNN. In Proceedings of the IEEE Interna-
tional Conference on Computer Vision, pages 2961–2969,

2017. 3
[29]Bohao Huang, Daniel Reichman, Leslie M Collins, Kyle
Bradbury, and Jordan M Malof. Tiling and stitching seg-
mentation output for remote sensing: Basic challenges and
recommendations. arXiv preprint arXiv:1805.12219, 2018. 5
[30]Shruti Jadon. A survey of loss functions for semantic segmen-
tation. In 2020 IEEE Conference on Computational Intelli-
gence in Bioinformatics and Computational Biology (CIBCB),
page 1–7. IEEE, 2020. 4
[31]Jitesh Jain, Jiachen Li, MangTik Chiu, Ali Hassani, Nikita
Orlov, and Humphrey Shi. OneFormer: One Transformer to
Rule Universal Image Segmentation. In CVPR, 2023. 3
[32]Johannes Jakubik, Felix Yang, Benedikt Blumenstiel, Erik
Scheurer, Rocco Sedona, Stefano Maurogiovanni, Jente
Bosmans, Nikolaos Dionelis, Valerio Marsocci, Niklas Kopp,
et al. Terramind: Large-scale generative multimodality for
earth observation. IEEE/CVF International Conference on
Computer Vision (ICCV), 2025. 3, 4, 1
[33]Hannah Kerner, Catherine Nakalembe, Adam Yang, Ivan
Zvonkov, Ryan McWeeny, Gabriel Tseng, and Inbal Becker-
Reshef. How Accurate are Existing Land Cover Maps for
Agriculture in Sub-Saharan Africa? Scientific Data, 11(1):
486, 2024. 3
[34]Hannah Kerner, Snehal Chaudhari, Aninda Ghosh, Caleb
Robinson, Adeel Ahmad, Eddie Choi, Nathan Jacobs, Chris
Holmes, Matthias Mohr, Rahul Dodhia, et al. Fields of the
World: A Machine Learning Benchmark Dataset For Global
Agricultural Field Boundary Segmentation. In Proceedings of
the AAAI Conference on Artificial Intelligence, pages 28151–
28159, 2025. 1, 2, 3, 5, 7, 4
[35]Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao,
Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer White-
head, Alexander C. Berg, Wan-Yen Lo, Piotr Dollar, and Ross
Girshick. Segment Anything, 2023. 3, 1
[36]Alexandre Lacoste, Alexandra Luccioni, Victor Schmidt, and
Thomas Dandres. Quantifying the Carbon Emissions of Ma-
chine Learning. arXiv preprint arXiv:1910.09700, 2019. 8

```
[37]Mykola Lavreniuk, Nataliia Kussul, Andrii Shelestov, Bo-
hdan Yailymov, Yevhenii Salii, Volodymyr Kuzin, and Zoltan
Szantoi. Delineate Anything: Resolution-Agnostic Field
Boundary Delineation on Satellite Imagery. arXiv preprint
arXiv:2504.02534, 2025. 3, 1
[38]Jonathan Long, Evan Shelhamer, and Trevor Darrell. Fully
convolutional networks for semantic segmentation. In 2015
IEEE Conference on Computer Vision and Pattern Recogni-
tion (CVPR), pages 3431–3440, 2015. 4
[39]Weiye Mei, Haoyu Wang, David Fouhey, Weiqi Zhou, Isabella
Hinks, Josh M Gray, Derek Van Berkel, and Meha Jain. Using
Deep Learning and Very-High-Resolution Imagery to Map
Smallholder Field Boundaries. Remote Sensing, 14(13):3046,
```
2022. 3
[40]Catherine Nakalembe and Hannah Kerner. Considerations for
AI-EO for Agriculture in Sub-Saharan Africa. Environmental
Research Letters, 18(4):041002, 2023. 1, 3
[41]Catherine Nakalembe, Hannah Kerner, Ivan Zvonkov, et al. A
Framework for EO-Based National Agricultural Monitoring
(EO-NAM) - For the African Context, 2024. preprint. 2
[42]Heather C. North, David Pairman, and Stella E. Belliss.
Boundary Delineation of Agricultural Fields in Multitem-
poral Satellite Imagery. IEEE Journal of Selected Topics
in Applied Earth Observations and Remote Sensing, 12(1):
237–251, 2019. 1
[43]Pontus Olofsson, Giles M Foody, Martin Herold, Stephen V
Stehman, Curtis E Woodcock, and Michael A Wulder. Good
practices for estimating area and assessing accuracy of land
change. Remote Sensing of Environment, 148:42–57, 2014. 7
[44]Claudio Persello, Valentyn A Tolpekin, J Ray Bergado, and
Rolf A De By. Delineation of Agricultural Fields in Small-
holder Farms from Satellite Images Using Fully Convolu-
tional Networks and Combinatorial Grouping. Remote Sens-
ing of Environment, 231:111253, 2019. 2, 3
[45]Claudio Persello, Jeroen Grift, Xinyan Fan, Claudia
Paris, Ronny Hänsch, Mila Koeva, and Andrew Nelson.
AI4SmallFarms: A Dataset for Crop Field Delineation in
Southeast Asian Smallholder Farms. IEEE Geoscience and
Remote Sensing Letters, 20:1–5, 2023. 2, 3
[46]Julien Radoux and Patrick Bogaert. Good Practices for Object-
Based Accuracy Assessment. Remote Sensing, 9(7):646, 2017.
7
[47]Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang
Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman
Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting
Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan
Wu, Ross Girshick, Piotr Dollár, and Christoph Feichtenhofer.
Sam 2: Segment anything in images and videos, 2024. 1
[48]Colorado J Reed, Ritwik Gupta, Shufan Li, Sarah Brockman,
Christopher Funk, Brian Clipp, Kurt Keutzer, Salvatore Can-
dido, Matt Uyttendaele, and Trevor Darrell. Scale-MAE: A
Scale-Aware Masked Autoencoder for Multiscale Geospatial
Representation Learning. In Proceedings of the IEEE/CVF
International Conference on Computer Vision, pages 4088–
4099, 2023. 6
[49]Esther Rolf, Konstantin Klemmer, Caleb Robinson, and Han-
nah Kerner. Position: Mission Critical–Satellite Data is a


```
Distinct Modality in Machine Learning. In 41st International
Conference on Machine Learning, 2024. 2, 5
[50]Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-Net:
Convolutional networks for biomedical image segmentation.
In International Conference on Medical image computing
and computer-assisted intervention, pages 234–241. Springer,
```
2015. 4, 1
[51]A. Rydberg and G. Borgefors. Integrated Method for Bound-
ary Delineation of Agricultural Fields in Multispectral Satel-
lite Images. IEEE Transactions on Geoscience and Remote
Sensing, 39(11):2514–2520, 2001. 2
[52]Seyed Sadegh Mohseni Salehi, Deniz Erdogmus, and Ali
Gholipour. Tversky loss function for image segmentation
using 3D fully convolutional deep networks, 2017. 4
[53]Philipp Schuegraf, Julian Schnell, Corentin Henry, and Kse-
nia Bittner. Building Section Instance Segmentation with
Combined Classical and Deep Learning Methods. ISPRS
Annals of the Photogrammetry, Remote Sensing and Spatial
Information Sciences, 2:407–414, 2022. 3
[54]Nimrod Segol and Yaron Lipman. On universal equivari-
ant set networks. In International Conference on Learning
Representations, 2020. 5
[55]Wenzhe Shi, Jose Caballero, Ferenc Huszár, Johannes Totz,
Andrew P Aitken, Rob Bishop, Daniel Rueckert, and Zehan
Wang. Real-Time Single Image and Video Super-Resolution
Using an Efficient Sub-Pixel Convolutional Neural Network.
In Proceedings of the IEEE Conference on Computer Vision
and Pattern Recognition, pages 1874–1883, 2016. 4
[56]Ofir Shifman and Yair Weiss. Lost in translation: Modern
neural networks still struggle with small realistic image trans-
formations. In European Conference on Computer Vision,
pages 231–247. Springer, 2024. 5
[57]Oriane Siméoni, Huy V. Vo, Maximilian Seitzer, Federico
Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc
Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, Francisco
Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, Tim-
othée Darcet, Théo Moutakanni, Leonel Sentana, Claire
Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille
Couprie, Julien Mairal, Hervé Jégou, Patrick Labatut, and
Piotr Bojanowski. DINOv3, 2025. 4, 1
[58]Daniela Szwarcman, Sujit Roy, Paolo Fraccaro, Þorsteinn Elí
Gíslason, Benedikt Blumenstiel, Rinki Ghosal, Pedro Hen-
rique de Oliveira, Joao Lucas de Sousa Almeida, Rocco Se-
dona, Yanghui Kang, Srija Chakraborty, Sizhe Wang, Car-
los Gomes, Ankur Kumar, Vishal Gaur, Myscon Truong,
Denys Godwin, Sam Khallaghi, Hyunho Lee, Chia-Yu Hsu,
Ata Akbari Asanjan, Besart Mujeci, Disha Shidham, Ru-
fai Omowunmi Balogun, Venkatesh Kolluru, Trevor Keenan,
Paulo Arevalo, Wenwen Li, Hamed Alemohammad, Pontus
Olofsson, Timothy Mayer, Christopher Hain, Robert Kennedy,
Bianca Zadrozny, David Bell, Gabriele Cavallaro, Campbell
Watson, Manil Maskey, Rahul Ramachandran, and Juan Bern-
abe Moreno. Prithvi-eo-2.0: A versatile multitemporal foun-
dation model for earth observation applications. IEEE Trans-
actions on Geoscience and Remote Sensing, 64:1–20, 2026.
3, 4, 1
[59] Mingxing Tan and Quoc Le. Efficientnet: Rethinking model
scaling for convolutional neural networks. In International

```
conference on machine learning, pages 6105–6114. PMLR,
```
2019. 4
[60]Taylor Geospatial Engine. Field boundaries for agriculture
(fiboa) — specification, tools and open data, 2025. 8
[61]Jamie Tolan, Hung-I Yang, Benjamin Nosarzewski, Guil-
laume Couairon, Huy V Vo, John Brandt, Justine Spore,
Sayantan Majumdar, Daniel Haziza, Janaki Vamaraju, et al.
Very high resolution canopy height maps from RGB imagery
using self-supervised vision transformer and convolutional de-
coder trained on aerial lidar. Remote Sensing of Environment,
300:113888, 2024. 8
[62]Gabriel Tseng, Anthony Fuller, Marlena Reil, Henry Herzog,
Patrick Beukema, Favyen Bastani, James R Green, Evan Shel-
hamer, Hannah Kerner, and David Rolnick. Galileo: Learning
global & local features of many remote sensing modalities.
In Proceedings of the 42nd International Conference on Ma-
chine Learning, pages 60280–60300. PMLR, 2025. 3, 4,
1
[63]François Waldner and Foivos I. Diakogiannis. Deep learning
on edge: Extracting field boundaries from satellite images
with a convolutional neural network. Remote Sensing of
Environment, 245:111741, 2020. 2
[64]François Waldner, Foivos I. Diakogiannis, Kathryn Batchelor,
Michael Ciccotosto-Camp, Elizabeth Cooper-Williams, Chris
Herrmann, Gonzalo Mata, and Andrew Toovey. Detect, Con-
solidate, Delineate: Scalable Mapping of Field Boundaries
Using Satellite Images. Remote Sensing, 13(11), 2021. 1, 2,
3
[65]Sherrie Wang, François Waldner, and David B. Lobell. Un-
locking Large-Scale Crop Field Delineation in Smallholder
Farming Systems with Transfer Learning and Weak Supervi-
sion. Remote Sensing, 14(22), 2022. 3
[66]Xuying Wang, Lei Shu, Ru Han, Fan Yang, Timothy Gor-
don, Xiaochan Wang, and Hongyu Xu. A survey of farmland
boundary extraction technology based on remote sensing im-
ages. Electronics, 12(5), 2023. 2
[67]Yi Wang, Conrad M. Albrecht, Nassim Ait Ali Braham,
Chenying Liu, Zhitong Xiong, and Xiao Xiang Zhu. De-
coupling common and unique representations for multimodal
self-supervised learning. In Eur. Conf. Comput. Vis., pages
286–303, 2024. 3, 4, 1
[68]Yi Wang, Conrad M Albrecht, and Xiao Xiang Zhu. Multi-
Label Guided Soft Contrastive Learning for Efficient Earth
Observation Pretraining. In IGARSS 2024 - 2024 IEEE Inter-
national Geoscience and Remote Sensing Symposium, pages
7568–7571, 2024. 3, 4, 1
[69]Barry Watkins and Adriaan van Niekerk. A Comparison of
Object-Based Image Analysis Approaches for Field Boundary
Delineation Using Multi-Temporal Sentinel-2 Imagery. Com-
puters and Electronics in Agriculture, 158:294–302, 2019.
2
[70]Michael J. Wellington and Luigi J. Renzullo. High-
Dimensional Satellite Image Compositing and Statistics for
Enhanced Irrigated Crop Mapping. Remote Sensing, 13(7),
2021. 3
[71]Tete Xiao, Yingcheng Liu, Bolei Zhou, Yuning Jiang, and Jian
Sun. Unified Perceptual Parsing for Scene Understanding. In
European Conference on Computer Vision. Springer, 2018. 4


[72]Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar,
Jose M Alvarez, and Ping Luo. Segformer: Simple and
efficient design for semantic segmentation with transformers.
Advances in neural information processing systems, 34:12077–
12090, 2021. 4
[73]Zhitong Xiong, Yi Wang, Fahong Zhang, Adam J Stewart,
Joëlle Hanna, Damian Borth, Ioannis Papoutsis, Bertrand Le
Saux, Gustau Camps-Valls, and Xiao Xiang Zhu. Neural
Plasticity-Inspired Foundation Model for Observing the Earth
Crossing Modalities. arXiv preprint arXiv:2403.15356, 2024.
3, 4, 1
[74]Xiaodong Yu, Dahu Shi, Xing Wei, Ye Ren, Tingqun Ye,
and Wenming Tan. SOIT: Segmenting Objects with Instance-
Aware Transformers. CoRR, abs/2112.11037, 2021. 3
[75]Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barn-
abás Póczos, Ruslan Salakhutdinov, and Alexander J. Smola.
Deep sets. In Advances in Neural Information Processing
Systems (NeurIPS), 2017. 5
[76]Richard Zhang. Making Convolutional Networks Shift-
Invariant Again. In Proceedings of the 36th International
Conference on Machine Learning, pages 7324–7334. PMLR,

2019. 5
[77]Xueyan Zou, Fanyi Xiao, Zhiding Yu, Yuheng Li, and
Yong Jae Lee. Delving Deeper into Anti-Aliasing in Con-
vNets. International Journal of Computer Vision, 131(1):
67–81, 2023. 5
[78]Ivan Zvonkov, Gabriel Tseng, Catherine Nakalembe, and
Hannah Kerner. OpenMapFlow: A Library for Rapid Map
Creation with Machine Learning and Remote Sensing Data.
Proceedings of the AAAI Conference on Artificial Intelligence,
37(12):14655–14663, 2023. 5


## PRUE: A Practical Recipe for Field Boundary Segmentation at Scale

## Supplementary Material

### A. Extended Experimental Details

Details on instance and panoptic segmentation model
baselines. Delineate Anything [ 37 ] is based on Ultralytics’
YOLOv11-seg and fine-tuned on FBIS-22M, a dataset of
RGB images from multiple remote sensing sources (Sentinel-
2, Planet, Maxar, Pleiades, orthophotos) over 9 European
countries with spatial resolution 0.25-10m. For Delin-
eate Anything, we perform a 1st-99th percentile normal-
ization following the pretraining dataset norms. SAM [ 35 ]
is a promptable instance segmentation model pretrained
on natural images, which we assessed in both zero-shot
and fine-tuned settings as described in the methods sec-
tion. Mask2Former (M2F) [ 8 ] is a universal segmenta-
tion architecture capable of semantic, instance, and panop-
tic segmentation depending on training configuration; we
adapted M2F with a Swin-S backbone to handle the 8-
channel RGBN bitemporal input and train it on the panoptic
task, which jointly predicts individual field instances (things)
and background land cover classes (stuff). Note that SAM
and Mask2Former were trained without presence-only label
masking–a data preprocessing strategy used by all seman-
tic baselines that filters out ambiguous background regions
in partially-labeled countries. The inability to implement
this masking for instance models (due to fundamental differ-
ences in how instance segmentation models handle training
objectives) means these models faced a harder optimization
landscape, being penalized for predicting fields in regions
that may contain unlabeled fields.

RGB-only comparison with Delineate Anything. To pro-
vide a fairer comparison with Delineate Anything (De-
lAny) [ 37 ], we trained new PRUE models with EfficientNet-
B3 (EF3) backbones on RGB-only data from a single time
step. This resulted in object F1 scores of 0. 38 ± 0. 06 (window
A) and 0. 37 ± 0. 06 (window B), both higher than DelAny’s
performance despite using the same RGB-only input. This
demonstrates that the performance gap between PRUE and
DelAny is not attributable to FTW’s additional spectral chan-
nel (NIR), but rather reflects differences in model design and
training data diversity.

SAM2 evaluation. We evaluated SAM2 in a zero-shot set-
ting on the window A RGB bands, resulting in a pixel IoU of
0. 31 and an object F1 of 0. 07. SAM2 is designed for video
segmentation where it expects continuous video frames in
which objects move or deform slightly with strong appear-
ance consistency [ 47 ]. This differs fundamentally from the
FTW setting, which provides two snapshots separated by
months that capture significant phenological changes. This

```
creates multiple failure modes: (1) appearance shifts be-
tween seasons violate SAM2’s visual consistency assump-
tion, and (2) fields do not “move” between timestamps—
they transform—leaving SAM2’s optical flow and corre-
spondence mechanisms without useful signal.
```
```
Geospatial foundation models (GFMs). The FTW bench-
mark provides four bands (RGB and NIR) [ 34 ], which is
fewer than the multi-spectral inputs used by most GFMs.
Since many GFMs expect 8 to 13 Sentinel-2 bands, we
used GFMs evaluation wrapper published in Galileo code-
base [ 62 ] to correctly prepare inputs for Galileo [ 62 ],
CROMA [ 25 ], SoftCon [ 68 ], Prithvi 2.0 [ 58 ], DOFA-v
[ 73 ], DeCUR [ 67 ], and Satlas [ 4 ]. This wrapper allowed us
to (1) construct the band set expected by each model (ap-
plying mask where applicable), (2) impute missing channels
in a model-consistent manner, and (3) apply each model’s
required normalization or standardization using its original
training statistics. For TerraFM [ 14 ], we assign zeros to all
missing spectral bands. DINOv3 [ 57 ] operates exclusively
on RGB inputs, so the FTW RGB bands are passed directly
without modification. For Clay [ 9 ] and TerraMind [ 32 ],
which are designed to handle partially missing spectral in-
formation, we provide the available four-band input with
the appropriate normalization for each model. Patch-level
embeddings are extracted from each pretrained GFM inde-
pendently for the two temporal windows defined in FTW.
```
```
GFM feature fusion and decoding. The patch embeddings
from the two temporal windows are fused by first concate-
nating them along the feature dimension and passing the
result through a three-layer MLP. Our objective is to evalu-
ate the representational quality of the frozen GFM features
themselves. A common evaluation strategy adopted for this
objective is linear probing. However, we argue that a single
linear transformation is often too limited to fully assess the
spatial and contextual information encoded in the pretrained
features. Conversely, using a specialized model such as
U-Net [ 50 ] at this stage would primarily test the ability of
the decoder rather than the underlying GFM embeddings. To
strike a balance between these two extremes, we employ a
simple decoder that provides moderate flexibility through a
3 × 3 projection layer, two residual refinement blocks, and a
multi-scale convolutional module, followed by pixel-shuffle
upsampling. Table 1 reports the results obtained with our
convolutional decoder, and Table 4 provides the complemen-
tary 1 × 1 convolution linear-probing results.
```

```
Austria
```
```
Image Clay Galileo DINOv3 Prithvi 2.0 TerraMind SoftCon Satlas DeCUR DOFA-v1 TerraFM CROMA
```
```
Brazil
```
```
Belgium
```
```
Rwanda
```
```
Cambodia
```
```
France
```
```
Figure 4. PCA visualization of frozen GFM encoder features for a representative subset of image examples. The top 3 principal components
of patch embeddings are displayed as RGB channels. Clay ( 8 × 8 patch sizes) and Galileo ( 4 × 4 patch size) capture finer spatial structure
and more distinct field boundaries compared to models using 16 × 16 patches, demonstrating how tokenization granularity affects feature
quality for segmentation tasks. See Table 4 for quantitative performance.
```
Table 4. GFM linear probing results using a lightweight decoder (1×1 convolution + bilinear upsampling), sorted by object-level F1. Clay
(ViT-Large, 8×8 patches) and Galileo (ViT-Base, 4×4 patches) outperform other GFMs that use coarser 16×16 patch sizes, due to the finer
patch resolutions as well as techniques intentionally designed to handle missing spectral bands.

```
Model Backbone Pixel-level Object-level
IoU↑ Prec↑ Recall↑ Prec↑ Recall↑ F1↑ AP 0 .5:0. 95 ↑ AP 0. 5 ↑
Clay ViT-Large 0.56 0.88 0.60 0.22 0.16 0.18 0.07 0.
Galileo ViT-Base 0.53 0.83 0.59 0.11 0.19 0.13 0.08 0.
DINOv3 ViT-Large 0.47 0.89 0.50 0.25 0.09 0.12 0.03 0.
Prithvi 2.0 ViT-Large 0.44 0.84 0.48 0.20 0.06 0.10 0.02 0.
TerraMind ViT-Base 0.44 0.85 0.47 0.19 0.07 0.10 0.02 0.
SoftCon ViT-Small 0.41 0.83 0.46 0.16 0.05 0.07 0.01 0.
Satlas Swin-Tiny 0.39 0.74 0.45 0.13 0.04 0.07 0.01 0.
DeCUR ViT-Small 0.42 0.80 0.46 0.15 0.04 0.07 0.01 0.
DOFA-v1 ViT-Large 0.39 0.77 0.44 0.14 0.04 0.06 0.01 0.
TerraFM ViT-Base 0.44 0.85 0.48 0.17 0.06 0.09 0.02 0.
CROMA ViT-Base 0.42 0.85 0.46 0.18 0.05 0.08 0.02 0.
```
### B. Extended Results

```
Our design choice is a result of extensive ablations. Ta-
ble 5 shows that boundary weighting, loss function, and
targeted augmentations have the strongest impact on perfor-
mance. Moderate class weights (ω ≈ 0. 75 ) and losses such
as LogCosh Dice, Tversky, and Local Tversky consistently
improve object F1 and pixel IoU, while brightness, scale, and
channel-shuffle augmentations provide additional robustness.
Larger EfficientNet backbones slightly enhance results, and
combining these components in PRUE yields the highest
```
```
accuracy and boundary agreement across all metrics.
```
```
Accuracy–throughput trade-off across model configura-
tions. To explicitly summarize the accuracy–cost trade-off
over all models, Figure 5 shows the Pareto front between ob-
ject F1 and throughput. Supplemental Table 5 compares our
methodology against the baseline using the same backbone
(EfficientNet-B3): the “U-Net LogCosh Dice 0.75-weight
Augs EN-B3” row shows that PRUE with an EF3 back-
bone achieves an object F1 of 0. 43 ± 0. 07 and field IoU
of 0. 74 ± 0. 07 , compared to PRUE-EF7 which achieves
```

Table 5. Ablation results for controlled experiments on the FTW test set (excluding presence-only countries) in which each row varies
a single design choice (data augmentations, class weighting, encoder, loss function, or architecture). Bold indicates best performance,
underlineindicates second-best.

```
Category Ablation Performance Input order Brightness Scale Agree.
Object F1↑ Pixel IoU↑ F1|∆|↓ IoU|∆|↓ F1|∆|↓ IoU|∆|↓ F1|∆|↓ IoU|∆|↓ Avg↑
FTW-v1 0.39± 0.08 0.68± 0.08 0.07 0.11 0.04 0.05 0.17 0.08 0.
Data augs Brightness 0.39± 0.08 0.68± 0.08 0.07 0.09 0.00 0.00 0.14 0.05 0.
Data augs Resize 0.38± 0.08 0.67± 0.09 0.07 0.10 0.04 0.05 0.00 0.01 0.
Data augs Brightness+Resize 0.38± 0.08 0.66± 0.09 0.06 0.10 0.03 0.03 0.00 0.02 0.
Data augs Channel shuffle 0.39± 0.07 0.68± 0.09 0.00 0.00 0.05 0.05 0.18 0.09 0.
Class weights ω = 0. 60 0.32± 0.06 0.76± 0.06 0.07 0.13 0.07 0.07 0.28 0.19 0.
Class weights ω = 0. 65 0.36± 0.06 0.76± 0.06 0.07 0.11 0.07 0.07 0.30 0.17 0.
Class weights ω = 0. 70 0.40± 0.06 0.75± 0.06 0.08 0.12 0.08 0.07 0.30 0.14 0.
Class weights ω = 0. 75 0.42± 0.06 0.74± 0.07 0.08 0.11 0.07 0.07 0.29 0.13 0.
Class weights ω = 0. 80 0.42± 0.07 0.73± 0.06 0.09 0.12 0.07 0.07 0.23 0.11 0.
Class weights ω = 0. 85 0.41± 0.07 0.70± 0.08 0.08 0.10 0.05 0.06 0.17 0.08 0.
Encoder EfficientNet-B4 0.40± 0.07 0.69± 0.09 0.07 0.09 0.04 0.05 0.15 0.06 0.
Encoder EfficientNet-B5 0.41± 0.07 0.70± 0.08 0.07 0.11 0.04 0.05 0.16 0.10 0.
Encoder EfficientNet-B6 0.41± 0.07 0.70± 0.08 0.07 0.11 0.04 0.05 0.18 0.14 0.
Encoder EfficientNet-B7 0.42± 0.07 0.71± 0.08 0.07 0.09 0.04 0.04 0.21 0.09 0.
Encoder MiT-B2 0.39± 0.08 0.67± 0.09 0.08 0.10 0.02 0.03 0.13 0.05 0.
Encoder MiT-B3 0.39± 0.08 0.67± 0.09 0.08 0.10 0.02 0.03 0.13 0.05 0.
Encoder MiT-B4 0.39± 0.08 0.68± 0.09 0.07 0.09 0.02 0.03 0.16 0.06 0.
Encoder MiT-B5 0.38± 0.08 0.67± 0.09 0.08 0.10 0.02 0.03 0.12 0.02 0.
Encoder ResNet-18 0.35± 0.07 0.67± 0.09 0.08 0.11 0.04 0.05 0.14 0.05 0.
Encoder VGG13-BN 0.38± 0.07 0.69± 0.08 0.08 0.10 0.04 0.05 0.27 0.20 0.
Learning rate 0.0001 0.34± 0.07 0.65± 0.09 0.05 0.07 0.03 0.04 0.15 0.08 0.
Learning rate 0.0003 0.37± 0.07 0.66± 0.09 0.06 0.08 0.04 0.04 0.17 0.10 0.
Learning rate 0.003 0.39± 0.08 0.68± 0.09 0.08 0.10 0.06 0.08 0.17 0.08 0.
Learning rate 0.01 0.39± 0.08 0.68± 0.09 0.08 0.11 0.06 0.06 0.18 0.08 0.
Learning rate 0.03 0.37± 0.08 0.67± 0.09 0.09 0.10 0.10 0.13 0.16 0.07 0.
Loss function CE (w/ Edge Agreement) 0.39± 0.08 0.68± 0.09 0.07 0.10 0.04 0.05 0.15 0.07 0.
Loss function CE + Dice 0.41± 0.07 0.70± 0.08 0.07 0.10 0.04 0.05 0.20 0.08 0.
Loss function CE + Dice (no class weights) 0.38± 0.07 0.77± 0.06 0.08 0.11 0.06 0.05 0.29 0.15 0.
Loss function CE + FTNMT 0.41± 0.07 0.70± 0.08 0.08 0.11 0.04 0.05 0.21 0.08 0.
Loss function CE (no class weights) 0.24± 0.06 0.77± 0.06 0.05 0.11 0.05 0.06 0.15 0.14 0.
Loss function Dice 0.42± 0.07 0.76± 0.07 0.08 0.13 0.06 0.06 0.31 0.16 0.
Loss function Dice (w/ Edge Agreement) 0.42± 0.07 0.77± 0.06 0.08 0.13 0.07 0.07 0.32 0.17 0.
Loss function Focal 0.18± 0.05 0.75± 0.06 0.04 0.10 0.04 0.06 0.15 0.21 0.
Loss function FTNMT 0.38± 0.06 0.79± 0.06 0.10 0.14 0.06 0.07 0.29 0.17 0.
Loss function Local Tversky 0.45± 0.07 0.74± 0.07 0.10 0.13 0.05 0.05 0.31 0.16 0.
Loss function LogCosh Dice 0.44± 0.07 0.77± 0.06 0.09 0.13 0.06 0.06 0.33 0.20 0.
Loss function LogCosh Dice + CE 0.39± 0.08 0.68± 0.08 0.08 0.10 0.04 0.05 0.14 0.06 0.
Loss function Tversky 0.43± 0.07 0.76± 0.06 0.09 0.12 0.06 0.05 0.34 0.15 0.
Loss function Tversky + CE 0.41± 0.07 0.71± 0.08 0.07 0.10 0.05 0.06 0.20 0.10 0.
Architecture FCN 0.14± 0.03 0.60± 0.08 0.04 0.09 0.03 0.07 0.10 0.00 0.
Architecture FCSiam 0.40± 0.07 0.69± 0.08 0.00 0.00 0.05 0.06 0.23 0.10 0.
Architecture UNETR 0.37± 0.07 0.69± 0.08 0.08 0.10 0.04 0.04 0.27 0.18 0.
Architecture UPerNet 0.34± 0.08 0.64± 0.10 0.07 0.09 0.03 0.04 0.13 0.04 0.
Combination FCSiam Combo 0.44± 0.07 0.75± 0.07 0.00 0.00 0.04 0.04 0.05 0.02 0.
Combination U-Net LogCosh Dice Augs EN-B3 0.42± 0.07 0.74± 0.07 0.00 0.00 0.00 0.00 0.01 0.01 0.
Combination U-Net LogCosh Dice 0.75-weight Augs EN-B3 0.43± 0.07 0.74± 0.07 0.00 0.00 0.00 0.00 0.02 0.00 0.
Combination U-Net LogCosh Dice 0.75-weight Augs EN-B5 0.46± 0.07 0.75± 0.07 0.00 0.00 0.00 0.00 0.01 0.01 0.
Combination PRUE 0.47± 0.07 0.76± 0.08 0.00 0.00 0.00 0.00 0.01 0.01 0.
```
```
0. 47 ± 0. 07 and 0. 76 ± 0. 08 , respectively. PRUE-EF3 is
a Pareto improvement over the FTW baseline, meaning it
achieves higher accuracy without sacrificing throughput. The
PRUE family forms a substantially stronger Pareto frontier
than any prior model and shows a clear throughput vs. perfor-
mance trade-off. As a practical reference, an EF3 backbone
can process the entire land area of the Earth in approximately
66 hours on a single V100 GPU, while EF7 would require
approximately 134 hours.
```
```
Full fine-tuning of Clay. To assess whether the perfor-
mance gap between GFMs and PRUE could be closed
with end-to-end training, we fully fine-tuned the best-
performing GFM, Clay, selecting the learning rate from
{ 1 , 3 }× 10 {−^5 ,−^4 ,−^3 }based on the best object F1 on the val-
idation set. Full fine-tuning increased Clay’s object F1 from
0. 36 to 0. 42 (see Figure 5) and pixel IoU from 0. 67 to 0. 73.
While these represent meaningful improvements over frozen-
encoder decoding, both metrics remain below PRUE ( 0. 47
object F1, 0. 76 pixel IoU), and Clay’s throughput is substan-
```

Figure 5. Pareto front between Object F1 and Throughput for all
Table 1 models, including PRUE-EF (B3/B5) throughput. PRUE-
EF3 is a Pareto improvement over the FTW baseline. The PRUE
family forms a substantially stronger Pareto frontier than any prior
model.

```
tially lower (Clay: 11 km^2 /s vs. PRUE: 307 km^2 /s). This
confirms that the performance gap is not solely attributable
to the frozen-encoder evaluation protocol, but reflects fun-
damental differences in spatial resolution and architectural
suitability for field boundary delineation.
```
```
1D convolution did not capture field boundary complex-
ity. Across all GFM experiments, we observed that decoding
using a single 1 × 1 convolution followed by bilinear upsam-
pling produced coarse, low-fidelity boundaries compared to
our convolutional decoder head. Table 4 shows these linear
probing results for all GFMs. Among the GFMs, Clay and
Galileo exhibit notably stronger performance. Both models
use smaller token patch sizes ( 8 × 8 for Clay and 4 × 4 for
Galileo), compared to most of the other evaluated models
that operate at a patch size of 16 × 16. These finer patch reso-
lutions produce higher-granularity spatial features that align
more naturally with field-level segmentation. In addition,
both models robustly handle missing spectral channels: Clay
is explicitly trained with missing-band augmentation, and
Galileo incorporates consistent band masking and normal-
ization statistics for partially observed inputs.
```
Consistency varies by geography and overlap window
size. Figure 6 illustrates how consistency changes with the
size of the overlapping region between the four corner crops
used to measure translation sensitivity. When the overlap
window is large, approaching the patch size, consistency
primarily reflects intrinsic translation sensitivity, which cap-
tures the model’s architectural tendency to produce different
outputs for identical content at different spatial positions.
When the overlap window is small, consistency also reflects
context dependency, indicating how much the model’s pre-
dictions vary based on surrounding image content.
PRUE achieved two to three times higher consistency

```
than the FTW baseline across all overlap window sizes,
demonstrating improvements in both intrinsic translation
robustness, an architectural property, and reduced context
dependency, a learned behavior. Consistency varied sub-
stantially across countries and correlates with test set dif-
ficulty. For example, our best model shows> 40 % pixel
disagreement in Kenya at 224-pixel context shifts, compared
to< 20 % in Switzerland. This suggests that consistency
metrics may serve as an out-of-distribution detection signal,
where low consistency during inference could indicate that
the model is operating on data that differs from its training
distribution, and may thus require human review or model
retraining.
Consistency as a reliability signal. To quantify the relation-
ship between consistency and performance, we computed
the average consistency over test samples per country and
examined its correlation with object F1 (See Figure7). We
observe that consistency is weakly correlated with perfor-
mance: the FTW baseline yieldsR^2 =0. 48 and PRUE-EF
yields R^2 =0. 30. This indicates that consistency metrics ex-
plain some drops in out-of-distribution (OOD) performance
but not all—a model can have high consistency but poor
performance. Consistency metrics may serve as one signal
among several for OOD detection, but should not be used
as a sole indicator of model reliability. In deployment, low
consistency scores are most useful for flagging regions that
exhibit gridding artifacts and spatial prediction instability,
warranting human review.
```
### C. Per-Country Evaluation

```
We report full per-country metrics for all FTW countries
with complete labels, excluding presence-only regions. For
each country, we provide pixel IoU and object-level F1.
To represent the major model families, we evaluated the
strongest-performing model from each category, as identified
in Table 1 of the main paper: the FTW baseline for semantic
segmentation [ 34 ], Mask2Former (M2F) with a Swin Small
backbone for panoptic segmentation [ 8 ], Clay with our con-
volutional decoder for geospatial foundation models [ 9 ], and
PRUE. All metrics were computed on each country’s official
FTW test split, following the evaluation protocol described
in §3.4.
Across regions, the results show consistent patterns. As
shown in Table 6, PRUE consistently achieved the highest
or second-highest Pixel IoU and Object F1 across nearly all
tested FTW countries, highlighting its robust and reliable
performance across diverse geographies and agricultural
systems.
```
### D. Mosaicking and Large Scale Inference

```
Deploying PRUE at the country scale requires constructing
spatially complete, cloud-free Sentinel-2 composites from
```

```
Figure 6. Consistency dependence on overlap size. Consistency scores as a function of overlap window size for the FTW baseline (left)
and PRUE (right). Larger overlaps measure intrinsic translation sensitivity, while smaller overlaps additionally capture context dependency.
Shown: top 5 and bottom 5 countries by mean consistency for each model. PRUE achieves higher consistency across all countries and
overlap sizes, with particularly strong improvements in challenging regions. Hard countries (Kenya, India) show low consistency even at
large overlaps, while well-represented regions (Finland, Netherlands) have high consistency, suggesting consistency metrics can identify
out-of-distribution samples at inference time.
```
Table 6. Per-country performance comparison for the top-performing models of each architecture family: FTW baseline (semantic),
Mask2Former with Swin-S (instance/panoptic), Clay (frozen GFM), Clay-FT (finetuned GFM), and PRUE (our optimized semantic model).
Bold indicates best performance per country. PRUE achieves the highest or second-highest pixel IoU scores across nearly all countries. All
models evaluated using the protocol in Section 3.4 on presence/absence labeled countries only.

```
Country Pixel IoU Object F
FTW M2F Clay Clay-FT PRUE FTW M2F Clay Clay-FT PRUE
Austria 0.71 0.74 0.69 0.76 0.78 0.41 0.38 0.39 0.47 0.
Belgium 0.75 0.78 0.73 0.79 0.82 0.57 0.57 0.57 0.62 0.
Cambodia 0.40 0.19 0.27 0.40 0.66 0.19 0.10 0.09 0.19 0.
Corsica 0.45 0.52 0.47 0.51 0.51 0.18 0.24 0.18 0.22 0.
Croatia 0.67 0.70 0.64 0.71 0.77 0.28 0.34 0.25 0.33 0.
Denmark 0.83 0.57 0.83 0.86 0.86 0.52 0.24 0.51 0.58 0.
Estonia 0.80 0.82 0.80 0.83 0.84 0.44 0.54 0.43 0.48 0.
Finland 0.83 0.85 0.81 0.85 0.87 0.55 0.54 0.53 0.59 0.
France 0.79 0.80 0.78 0.81 0.83 0.55 0.55 0.54 0.58 0.
Germany 0.79 0.77 0.78 0.80 0.79 0.41 0.44 0.39 0.42 0.
Latvia 0.81 0.84 0.81 0.84 0.85 0.44 0.54 0.44 0.49 0.
Lithuania 0.74 0.78 0.74 0.78 0.79 0.39 0.48 0.38 0.45 0.
Luxembourg 0.79 0.80 0.76 0.82 0.85 0.49 0.37 0.46 0.53 0.
Netherlands 0.75 0.78 0.74 0.81 0.81 0.48 0.51 0.48 0.54 0.
Portugal 0.12 0.21 0.23 0.37 0.10 0.03 0.08 0.04 0.07 0.
Slovakia 0.92 0.91 0.92 0.94 0.94 0.53 0.61 0.53 0.58 0.
Slovenia 0.58 0.66 0.55 0.65 0.68 0.24 0.28 0.20 0.27 0.
South Africa 0.80 0.80 0.78 0.81 0.82 0.53 0.56 0.50 0.56 0.
Spain 0.73 0.70 0.69 0.75 0.83 0.24 0.26 0.21 0.26 0.
Sweden 0.81 0.82 0.80 0.84 0.85 0.45 0.51 0.44 0.50 0.
Vietnam 0.46 0.30 0.31 0.46 0.67 0.15 0.09 0.08 0.15 0.
```
```
irregularly sampled, partially cloudy observations. This
section details our operational pipeline for scene selection,
temporal compositing, and imagery quality mosaicking us-
```
```
ing latitude-based season heuristics, greedy scene selection
to minimize redundancy, and cloud-optimized data formats
enabling scalable parallel inference.
```

```
Figure 7. Country-level consistency vs. object F1. Each point
represents one FTW country. Dashed lines show linear fits for the
FTW baseline (orange, R²=0.48) and PRUE-EF7 (blue, R²=0.30).
Consistency is weakly correlated with performance, suggesting it
may serve as a partial out-of-distribution signal but not a reliable
sole predictor of model accuracy.
```
Latitude-Based Season Heuristics. Planting and harvest
windows were estimated using latitude-dependent day-of-
year (DOY) ranges that account for hemispheric differences
and climatic zones. This heuristic approach provides rea-
sonable temporal constraints for scene selection, although
we acknowledge that date selection could be substantially
improved by integrating additional information on geograph-
ical variation in crop growth cycles, such as crop calendars,
phenological models, or ground-based information on local
planting and harvest periods.

1. Planting Season Heuristic Algorithm

```
function PLANTINGSEASONDOY(latitude)
abs_lat←|latitude|
if abs_lat > 45 then High latitudes
return (91, 151) if latitude > 0 else (274, 334)
else if 20 < abs_lat≤ 45 then Mid-latitudes
return (60, 120) if latitude > 0 else (244, 334)
else if 5 < abs_lat≤ 20 then Subtropics
return (121, 212) if latitude > 0 else (305, 365)
else Equatorial|lat|≤ 5
return (60, 121)
end if
end function
```
2. Harvest Season Heuristic Algorithm

```
function HARVESTSEASONDOY(latitude)
abs_lat←|latitude|
if abs_lat > 45 then High latitudes
return (244, 304) if latitude > 0 else (60, 151)
else if 20 < abs_lat≤ 45 then Mid-latitudes
return (213, 304) if latitude > 0 else (32, 120)
else if 5 < abs_lat≤ 20 then Subtropics
return (274, 365) if latitude > 0 else (91, 181)
else Equatorial|lat|≤ 5
return (182, 243)
end if
end function
```
```
Feature Selection via Greedy Search Optimal scene se-
lection maximizes spatial coverage while minimizing re-
dundancy. Input scenes were pre-filtered by cloud cover
(< 75%) and Scene Classification Layer (SCL) quality flags
(excluding classes 1, 3, 7, 8, 9, 10, and nodata=0). The
greedy approach prioritizes scenes that contribute valid ob-
servations to underrepresented spatial regions within a tile,
enabling the use of relaxed scene-level cloud cover thresh-
olds. Scenes with high overall cloud cover may still con-
tain substantial cloud-free areas that fill critical gaps in the
composite, thereby improving spatial completeness without
requiring additional acquisitions.
```
```
Cloud-Optimized GeoTIFF Storage. For each Sentinel-
2 grid tile and temporal period, median composites were
constructed from the selected scenes. Spectral bands (B02,
B03, B04, B08) at native 10 m resolution were masked using
SCL upsampled from 20 m via nearest-neighbor. Temporal
medians were computed alongside valid observation counts.
Outputs were stored as float32 Cloud-Optimized GeoTIFFs
with 1024 × 1024 internal tiling.
```
```
GTI-Based Reprojection, Resampling and Zarr Assem-
bly. GDAL Tile Index (GTI)^1 files provide virtual mosaics
referencing distributed COGs. During Zarr^2 construction,
```
(^1) https://gdal.org/en/latest/drivers/raster/gti.
html
(^2) https://zarr.readthedocs.io/en/stable/


3. Greedy Scene Selection Algorithm

```
function SELECTSCENESGREEDY(valid_mask,
target_coverage = 5, max_scenes = 10)
Input:valid_mask- boolean array of shape(T,H,W )
```
coverage_depth← (^0) H×W
remaining←{ 0 , 1 ,...,T − 1 }
selected← []
for i = 1→ max_scenes do
best_idx← NULL
best_gain←− 1
for each idx∈ remaining do
undercovered ← (coverage_depth <
target_coverage)
new_valid← valid_mask[idx]∧undercovered
gain←
P
new_valid
if gain > best_gain then
best_gain← gain
best_idx← idx
end if
end for
if best_gain = 0 then
break
end if
selected.APPEND(time[best_idx])
coverage_depth ← coverage_depth +
valid_mask[best_idx]
remaining.REMOVE(best_idx)
end for
return selected
end function
gdal_translateperforms windowed extraction with
on-the-fly reprojection to EPSG:3857 (Web Mercator) using
nearest neighbor resampling and writes to a temporary local
file. Reprojection to EPSG:3857 prior to inference elimi-
nates the need for downstream pipelines to perform coordi-
nate transformations, and enables global non-overlapping
results without tile artifacts. The windowed data is then
loaded and inserted directly into the Zarr store, with spatial
partitions written to Zarr v3 arrays in parallel with Ray^3.
This creates a robust and scalable approach to building large-
scale, cloud-optimized data ready for downstream analysis.

### E. Large Scale Inference Visual Samples

To qualitatively assess PRUE’s performance at operational
scale, we present in Figure 8 representative visual samples
from each country-scale deployment described in Section 5.
These samples illustrate model behavior across diverse agri-
cultural systems (Japan, Mexico, Rwanda, South Africa,
Switzerland) spanning a wide range of field sizes. The vi-
sualizations demonstrate PRUE’s ability to maintain spatial
consistency across large extents under atmospheric varia-
tion. Visual inspection reveals spatial patterns of success
and failure modes that inform future improvements.

(^3) https://docs.ray.io/en/latest/index.html

### F. Change Detection Analysis

```
The change detection visualizations presented in Figure 8
are intended to demonstrate how multi-year maps produced
by PRUE can signal probable field-scale changes. We leave
more detailed studies of change detection to future work.
The method computes the absolute difference between se-
mantic logits from consecutive years, applies min–max nor-
malization, and thresholds at 0.5 to obtain a binary change
mask. For well-calibrated models, this threshold highlights
high-confidence semantic shifts. Visual inspection confirms
that even small-scale detected changes are consistent with
cultivation shifts (e.g., fields appearing or disappearing be-
tween years), and that artifacts from misregistration and
atmospheric variation are uncommon. We note that lacking
ground truth change labels, we relied on photo-interpretation
of high-resolution basemap imagery rather than quantitative
accuracy assessments, which we leave as future work.
```
### G. Future Directions

```
Several directions remain open for future work, includ-
ing: (1) comprehensive object-level [ 46 ] and thematic accu-
racy [ 43 ] assessments on country-scale deployments with
independent reference data; (2) exploring deployment met-
rics as out-of-distribution detectors (our preliminary findings
suggest low consistency scores may signal when models en-
counter dissimilar data); (3) incorporating super-resolution
approaches for delineating smallholder fields from tempo-
ral stacks of imagery; and (4) systematic post-processing
ablations (morphological operations, topological cleaning,
confidence thresholding) to further improve boundary quality
in challenging regions.
```

```
2023 Predictions 2023 Predictions + 2024 change
```
```
Japan
```
```
Mexico
```
```
Rwanda
```
```
South Africa
```
```
Switzerland
```
```
Field class probability
```
```
Field class probability
```
```
Field class probability
```
```
Field class probability
```
```
Field class probability
```
```
2023 Planet Imagery +
2024 change
2024 Planet Imagery +
2024 change
```
Figure 8. Example visuals over each country in our large-scale inference set (Japan, Mexico, Rwanda, South Africa, and Switzerland). For
each region of interest, we show: (1) the PRUE field boundary predictions from 2023; (2) the 2023 predictions with changes detected in
2024 highlighted in bright green; (3) Planet monthly basemaps from 2023 with the vectorized change mask outlined in red; and (4) Planet
monthly basemaps from 2024 with the vectorized change mask outlined in red. The basemaps shown for each pair are from the same month
in consecutive years.


