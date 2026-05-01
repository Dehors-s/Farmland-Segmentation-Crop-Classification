Delinerate Anything：分辨率无关的场边界划定
卫星影像
米科拉·拉夫雷纽克^1，^2，娜塔莉娅·库苏尔^3，安德烈·谢列斯托夫^2，^4，博赫丹·亚伊利莫夫^2，叶夫亨尼·萨利^2，^4，
Volodymyr Kuzin^2 ,^4 , Zoltan Szantoi^1
（^1）欧洲航天局，（^2）NASU-SSAU空间研究院，（^3）马里兰大学，
（^4）乌克兰国立技术大学“伊戈尔·西科尔斯基基辅理工学院”
图1。Delineate Anything模型的工作流程，用于从任意分辨率
卫星图像中进行场域实例分割和场边界提取，该模型基于我们的大型场边界实例分割数据集（FBIS-22M）训练，该数据集包含2200万个场边界。

摘要
The accurate delineation of agricultural field boundaries
from satellite imagery is vital for land management and
crop monitoring. However, current methods face challenges
due to limited dataset sizes, resolution discrepancies, and
diverse environmental conditions. We address this by re-
formulating the task as instance segmentation and intro-
ducing the Field Boundary Instance Segmentation - 22M
dataset (FBIS-22M), a large-scale, multi-resolution dataset
comprising 672,909 high-resolution satellite image patches
(ranging from 0.25 m to 10 m) and 22,926,427 instance
masks of individual fields, significantly narrowing the gap
between agricultural datasets and those in other computer
vision domains. We further propose Delineate Anything, an
instance segmentation model trained on our new FBIS-22M
dataset. Our proposed model sets a new state-of-the-art,
achieving a substantial improvement of 88.5% in mAP@0.
and 103% in mAP@0.5:0.95 over existing methods, while
also demonstrating significantly faster inference and strong
zero-shot generalization across diverse image resolutions
and unseen geographic regions. Code, pre-trained mod-
els, and the FBIS-22M dataset are available athttps://
lavreniuk.github.io/Delineate-Anything.
1. 引言
The delineation of agricultural field boundaries from satel-
lite imagery is crucial for precision agriculture, land man-
agement, policymaking and crop monitoring. The European
Union’s Land Parcel Identification System (LPIS) serves as
a key tool for defining agricultural field boundaries to sup-
port land use monitoring and subsidy allocation [9]. How-
ever, many regions in the world lack such systems, result-
ing in outdated cadastral maps that prevent effective agri-
cultural management. The manual, labor-intensive creation
and maintenance of LPIS data [35] further highlight the
need for automated, scalable solutions to detect field bound-
aries from satellite data.
Traditional computer vision techniques, like edge detec-
tion and clustering [11, 34, 40, 42], often fail to generalize
across diverse field types, geographic regions, and environ-
mental conditions. The recent availability of datasets like
AI4Boundaries [7], along with others [1, 25, 39], has facil-
arXiv：2504.02534v1 [cs.CV] 2025年4月3日
推进深度学习（DL）方法的发展。
然而，当前DL方法在场
边界检测中的应用落后于其他
计算视觉领域的进步，主要原因是数据集的
大小和质量限制。与大型数据集如
ADE20K [43]、Open Images [18]、COCO [21]、SA-1B [16]和
LAION [29]相比，现有农业数据集规模明显
较小，这阻碍了模型的泛化和应用
性。
另一个挑战是许多数据集依赖10米
中分辨率Sentinel-2图像。
虽然对较大的田地来说足够，但对于
较小、不规则的田地（小农
户常见）则不适用。因此，仅用哨兵
2号成像训练的模型在应用于无
人机或其他卫星获取的高分辨率数据时，常常表现出显著的性能下降
。广泛使用的AI4Boundaries
数据集[7]虽然贡献宝贵，但也存在月度图像合成引入的人工
信息，如
边界模糊，进一步影响模型的完整性
和准确性。

关键的是，大多数现有的DL方法将场
边界检测视为语义分割问题，将
每个像素归类为场边界
元或背景[1， 39]。这种方法通常
通过编码-解码器架构（如
U-Net或其变体）实现，重点是检测连续
的边界线。然而，对于实际的农业管理和
地籍应用，识别个别田间
物件至关重要。即使是轻微的分割错误也可能导致
相邻田地的错误合并，导致面积计算和
地块识别存在
重大不准确。虽然有人
提出后处理步骤以缓解这一问题，但它们往往缺乏必要的
稳健性和适用于不同农业
景观和田块类型的普遍性[35]。

为克服这些限制，我们引入了一个新的大规模
数据集，其规模是现有
数据集的12倍以上，涵盖了来自多个来源
（Sentinel-2、Planet、Maxar、昴宿星团和正射照片）的
图像，分辨率范围广泛（从0.25米到10米）。
这使得能够训练一个单一且高度可推广的模型
，在不同分辨率和感知
类型中有效表现，从而提升农业环境中的可扩展性。
此外，我们还提出了一种新颖的分辨率无关立
场分割方法用于田间划分（见图
1），该方法通过将任务框架为识别单个
田块实例，改善复杂田块形状的处理，
防止田地合并，并为现实世界的农业管理和
土地管理提供更准确且
实用相关的结果。

我们将模型与新数据集上的最先进方法
进行比较，显示出显著改进——

ments in mean Average Precision (mAP): from 0.382 to
0.720 (+88.5%) for mAP@0.5 and from 0.235 to 0.
(+103%) for mAP@0.5:0.95. Furthermore, our method has
significantly faster inference times compared to its closest
rival, enhancing its practical usability. Notably, we also
demonstrate the strong zero-shot capabilities of our model
on geographically distinct locations not present in the train-
ing dataset.
In summary, our contributions are threefold:
一种新颖的任务表述，将场边界检测作为
实例分割问题，解决了该任务语义分割的固有
局限性。
一个新的大型多分辨率卫星影像
数据集，用于稳健的现场边界划定。
一个分辨率无关的模型，显著超越
当前最先进的场边界
检测方法，同时在不同分辨率
和地理位置中展现出卓越的推断速度和
强的零样品泛化能力。
2. Related Work
2.1. Traditional Methods
Early approaches employed classical image processing
techniques such as edge detection (e.g., Canny, Sobel,
LoG) and clustering based on spectral or textural fea-
tures (e.g., graph-based segmentation, Simple Linear Itera-
tive Clustering (SLIC) segmentation, watershed segmenta-
tion) [6, 11, 34, 40, 42]. These methods, while computation-
ally efficient, often produced non-closed boundaries, requir-
ing post-processing and filtering to remove irrelevant edges
not corresponding to agricultural fields using additional in-
formation from cropland and crop type maps. These meth-
ods are also inherently sensitive to noise and varying illumi-
nation conditions common in satellite imagery. These limi-
tations motivated the exploration of more robust techniques,
particularly with the rise of deep learning.
2.2. Deep Learning for Semantic Segmentation
Deep learning has shown promise in related remote sensing
tasks, such as building [37] and road extraction [4], as well
as general boundary detection [2, 19, 22, 41]. However,
these methods primarily focus on semantic boundary detec-
tion, often requiring post-processing to form closed objects
and failing to distinguish individual field instances. Sev-
eral works have applied deep learning directly to agricul-
tural field boundary delineation. Some early deep learning
approaches combined deep learning with classical meth-
ods such as adaptive graph-based growing contours for field
extraction [33]. Fully convolutional networks (FCNs) and
contour closing procedures have been explored for field de-
lineation, particularly in smallholder farms [25]. FCNs have
also been used for super-resolution contour detection [23].
ResUNet-a, a deep learning framework for semantic seg-
mentation of remotely sensed data, has been applied to field
boundary detection [8]. U-Net-based FCNs have been used
for specific crop types such as rice paddy delineation [38].
Recent works have improved segmentation models and loss
functions, such as the Residual and Recurrent Attention U-
Net (R2AttU-Net) with Lovasz-Softmax loss [30], and U- ́
Net with Kolmogorov-Arnold Networks [27].
A significant step towards addressing the limitations of
purely boundary-based methods was the introduction of
FracTAL ResUNet [35]. Recognizing the challenges in di-
rectly predicting closed boundaries, this work incorporated
a distance-to-boundary channel alongside hierarchical wa-
tershed segmentation as a post-processing step. This ap-
proach aimed to produce more complete and closed con-
tours, moving closer to instance-level segmentation as ex-
plicitly stated by the authors. Subsequent efforts built upon
this idea. Transfer learning with FracTAL ResUNet was
explored for smallholder farming systems [39], leverag-
ing the benefits of the distance-to-boundary representation.
Other works further developed this direction, employing
similar strategies of incorporating boundary distance infor-
mation within a multi-task learning framework to predict
field extent, boundaries, and distance to boundaries [14].
While these methods, including efforts focused on multi-
task learning, model architecture improvements, and loss
function modifications, improve boundary prediction, they
still operate within semantic segmentation and thus do not
inherently provide instance-level information. Although
post-processing steps are incorporated [35], they often rely
on heuristics and lack generalizability.

2.3. Moving Towards Instance-Level Segmentation
The core challenge for accurate field identification and
area calculation is transitioning from semantic to instance
segmentation. While instance segmentation has advanced
significantly in computer vision, from Mask R-CNN [12]
to state-of-the-art architectures like Co-DETR [44], ViT-
Adapter [5], EVA [10], EVP [20], and recent real-time
YOLO variants [15, 36], its application to agricultural
fields is limited by the lack of suitable, instance-annotated
datasets. Existing datasets [1, 7, 25, 39] are often limited in
size and resolution (e.g., 10m Sentinel-2).
The emergence of the Segment Anything Model
(SAM) [16] presented a promising new direction by offer-
ing impressive zero-shot segmentation capabilities. This
approach, explored in the context of satellite-based field
boundary detection [32], offered the potential to perform in-
stance segmentation without extensive annotated datasets.
However, as also highlighted in [3, 13, 32] and confirmed
by our own investigations, direct application of SAM to
agricultural fields reveals limitations. SAM tends to over-
segment, detecting irrelevant objects like roads and forests,

leading to low precision. Furthermore, its computational
cost limits large-scale applicability. While subsequent work
has explored refinements like multi-scale processing [13],
weakly supervised learning [31], and prompt engineer-
ing [24, 28], these methods require additional data such as
prompts or weak labels. These approaches can be effec-
tive for scenarios where such data is available and the goal
is to refine boundaries for specific fields. However, they
do not address the fundamental limitations of SAM’s zero-
shot transferability in general, particularly for large territo-
ries where no such prior information exists. Even with the
newer SAM2 model [26], we observed similar issues, in-
dicating that these core challenges persist even in updated
versions.
To overcome these limitations, our work directly ad-
dresses the data bottleneck and the need for efficient, ac-
curate instance segmentation. We introduce the Delin-
eate Anything framework, which includes an instance seg-
mentation model and the new large-scale, multi-resolution,
instance-annotated FBIS-22M dataset. This framework
achieves significant advancements over existing seman-
tic segmentation methods and demonstrates clear advan-
tages over zero-shot instance segmentation approaches like
SAM [16, 32] and SAM2 [26].
3. Methodology
In this section, we present our contributions to the field of
boundary delineation, beginning with a reformulation of the
task as instance segmentation, which addresses the limita-
tions of existing methods. We introduce FBIS-22M, a new
dataset specifically designed for this purpose, and demon-
strate its utility by training and evaluating Delineate Any-
thing, a model that sets a new state-of-the-art in field bound-
ary delineation.
3.1. Reframing Field Boundary Delineation as In-
stance Segmentation
Traditional semantic segmentation approaches for field
boundary detection encounter notable challenges, espe-
cially when assessed using boundary Intersection over
Union (IoU). As illustrated in Figure 2, boundary IoU
scores are highly sensitive to small misalignments, even
when predicted boundaries closely follow the ground truth.
For instance, a slight offset of only a few pixels results in
a boundary IoU of 0.08 (Figure 2b), excessively penalizing
the model for an error that has minimal practical impact.
In contrast, instance IoU remains more robust in such sce-
narios, yielding a score of 0.98 (Figure 2e), as it prioritizes
accurate field delineation rather than pixel-perfect boundary
alignment.
More critically, boundary IoU fails to account for seg-
mentation errors that lead to adjacent fields being incor-
rectly merged into a single object. As shown in Figure 2, a
Figure 2.Comparison of task formulations and evaluation met-
rics for field boundary delineation.The top row illustrates field
boundary masks (semantic segmentation), while the bottom row
shows individual field masks (instance segmentation). Ground
truth examples are shown in (a) and (d). Slightly misaligned
boundaries result in a boundary IoU of 0.08 (b) and an instance
IoU of 0.98 (e). Partially detected boundaries yield a boundary
IoU of 0.93 (c) and an instance IoU of 0.54 (f).

partially detected boundary results in a high boundary IoU
score of 0.93 (Figure 2c), despite significant merging of dis-
tinct fields. However, instance IoU more accurately reflects
the severity of this error, dropping to 0.54 (Figure 2f). This
discrepancy highlights the inadequacy of boundary IoU for
real-world agricultural applications, where preserving the
distinctness of individual fields is critical for tasks such as
crop monitoring and yield estimation.
To overcome these limitations, we reformulate the field
boundary delineation task as an instance segmentation prob-
lem. In this approach, each field is treated as a distinct in-
stance, and the goal is to predict closed-field masks, which
avoids common issues such as boundary misalignment and
field merging. As shown in Figure 1, these instance-level
masks can be easily converted into field boundaries using
simple post-processing techniques like contour extraction.
This reformulation aligns the evaluation metric (instance
IoU) with the practical requirements of field delineation,
providing a more robust methodology for both training and
model evaluation. Instance IoU offers several advantages: it
is less sensitive to minor boundary variations while penaliz-
ing the merging of fields, which significantly affects the ac-
curacy of the model. By reformulating the task as instance
segmentation, we advance the precision and reliability of
field boundary detection models, marking a significant step
forward in agricultural image analysis.
3.2. Field Boundary Instance Segmentation Dataset
Field boundary detection in agriculture faces challenges
due to the variability in field sizes, shapes, and image
Dataset Resolution # Images # Instances
General Computer Vision Datasets
LAION-5B [29] - 5.85B -
COCO [21] - 330K 1.5M
Open Images [18] - 998K 2.8M
SA-1B [16] - 11M 1.1B
Field Boundary Delineation Datasets
Farm Parcel [1] 10m 2K -
India10K [39] - - 10K
AI4SmallFarms [25] 10m 62 439K
AI4Boundaries [7] 1m & 10m 55K 2.5M
FBIS-22M 0.25m-10m 673K 22.9M
Table 1.Comparison of FBIS-22M with existing datasets.The
table compares FBIS-22M with general computer vision datasets
and existing field boundary delineation datasets based on satellite
imagery, highlighting FBIS-22M’s resolution range and scale.
resolutions. While general computer vision datasets such
as LAION-5B with 5.85 billion images [29] and SA-1B
with 1.1 billion instance masks [16] provide large-scale
resources for other vision tasks, agricultural datasets for
field boundary detection have been much smaller. Existing
datasets range from just 62 images in AI4SmallFarms [25]
to 55 thousands images in AI4Boundaries [7], limiting the
ability to train robust and generalizable models (Table 1).
To address this limitation, we introduce theFieldBoundary
InstanceSegmentation - 22M (FBIS-22M) dataset, which
is the largest dataset for field boundary instance segmen-
tation. It contains 672,909 high-resolution satellite image
patches and 22,926,427 instance masks of individual fields,
making it more than 12 times larger than the previously
largest dataset, AI4Boundaries [7].
To the best of our knowledge, FBIS-22M is the first
dataset to incorporate high-resolution imagery from com-
mercial satellites. This unique feature enhances its value
as a resource for field boundary detection in diverse agri-
cultural landscapes. FBIS-22M integrates data from multi-
ple satellite platforms, including Sentinel-2, Planet, Maxar,
Pleiades, and publicly available satellite sources, providing
diverse data types and enabling compatibility with different
sensor technologies.
FBIS-22M offers a broad range of resolutions from
0.25m to 10m, covering both smallholder and large-scale
agricultural applications. Specifically, the dataset includes
images with resolutions of 0.25m, 0.3m, 0.5m, 1m, 1.2m,
2m, 3m, and 10m. This diversity in resolutions enables the
accurate segmentation of both small, irregular fields as well
as larger, expansive agricultural areas, supporting general-
ization across different field types and environmental con-
ditions.
FBIS-22M also provides significant geographic diver-
Figure 3.Examples of field boundary instance segmentation from our FBIS-22M dataset.The FBIS-22M dataset contains over 670K+
multi-resolution satellite images (ranging from 0.25m to 10m) and 22M+ field instance masks. Images are grouped by the number of fields
to demonstrate the dataset’s diversity and scalability, and a challenge of separating fields across varying resolutions and geographies.

sity, covering several European countries, including Aus-
tria, France, Luxembourg, the Netherlands, Slovakia,
Slovenia, Spain, Sweden, and Ukraine. This broad geo-
graphic scope ensures that models trained on FBIS-22M can
adapt to varied agricultural practices, land types, and envi-
ronmental conditions. The dataset further demonstrates di-
versity in field densities, with images containing fewer than

10 fields to over 300 fields per image. This variability, il-
lustrated in Figure 3, highlights its ability to represent both
sparse and dense agricultural regions.
The construction of FBIS-22M prioritized quality and
completeness. Official LPIS (Land Parcel Identification
System) boundaries were utilized for most regions, while
high-resolution commercial satellite imagery was manually
annotated for regions where LPIS data was unavailable,
such as Ukraine, ensuring full coverage. Additionally, the
dataset was manually cleaned, by removing errors in field
boundaries and inconsistencies addressed to ensure accu-
racy.
The dataset is split into 636,784 training images and
36,125 test images, enabling effective model training and
evaluation. As shown in Table 1, FBIS-22M significantly
surpasses existing field boundary datasets in both image
count and instance masks. By closing this critical resource
gap, FBIS-22M provides a comprehensive foundation for
advancing precision agriculture and automated land parcel
identification, placing it on par with leading computer vi-
sion datasets.

3.3. Delineate Anything
We propose Delineate Anything (DelAny), a framework for
accurate and efficient field boundary delineation from di-
verse satellite imagery. DelAny focuses on using exist-
ing state-of-the-art instance segmentation techniques and a
large-scale dataset to achieve strong results, rather than in-
troducing new architectural designs. At the core of DelAny
is the YOLOv11 instance segmentation model, currently
the state-of-the-art in instance segmentation. YOLOv
provides exceptional accuracy and real-time performance,
making it ideal for handling the large volumes of data typi-
cal in remote sensing applications.
The DelAny pipeline (Figure 1) processes satellite im-
agery at their native resolutions, avoiding resizing artifacts
and preserving fine-grained boundary details. During train-
ing, the model utilizes images from a variety of sources, in-
cluding Sentinel-2, Planet, Maxar, Pleiades, and orthopho-
tos, as part of the FBIS-22M dataset. This ensures the
model’s ability to generalize across a wide range of resolu-
tions and imaging conditions. Once trained, the resolution-
agnostic design of DelAny allows it to handle imagery from
any source, maintaining high performance without addi-
tional fine-tuning.
Input images are processed by the pre-trained DelAny
model to generate instance masks, which are then trans-
formed into closed-field boundaries using simple post-
processing techniques like contour extraction. This stream-
lined approach simplifies the pipeline while ensuring preci-
sion in delineating field boundaries.

4. Experiments
4.1. Metrics
We evaluate our method using standard instance segmen-
tation metrics based on the Microsoft COCO evaluation
protocol [21], reporting Mean Average Precision (mAP) at
IoU thresholds of 0.5 (mAP@0.5) and from 0.5 to 0.
(mAP@0.5:0.95). mAP@0.5 averages the precision for

each class at an IoU of 0.5, while mAP@0.5:0.95 averages
precision across IoU thresholds from 0.5 to 0.95 in steps of
0.05. These metrics offer a comprehensive evaluation of our
method’s performance in accurately detecting and segment-
ing agricultural fields.
4.2. Implementation Details
The Delineate Anything model is trained with a batch size
of 320 (40 per GPU), a learning rate of 2e−^5 , and 30 epochs,
using the standard YOLO loss function [15, 36], which
includes components for bounding box regression, object-
ness, and classification, along with task alignment learning.
Model is initialized with COCO pretrained weights before
fine-tuning on our dataset. We use the AdamW optimizer
with exponential learning rate decay. For data augmenta-
tion, we employ standard techniques such as horizontal and
vertical flips, color jittering, mosaic, mixup, and copy-paste
augmentation, consistent with typical YOLO training prac-
tices [15, 36]. Mosaic augmentation was used for the first
20 epochs and then disabled for the final 10 epochs. All ex-
periments are conducted on 8 NVIDIA H100 GPUs. By de-
fault, we evaluate model performance using the final check-
point after training rather than selecting the best-performing
checkpoint. To ensure a fair comparison, other models com-
pared in this work are trained using their officially released
code bases on our dataset (or the AI4Boundaries [7] dataset
where applicable), except for the zero-shot evaluation, as
specified elsewhere in the paper.
4.3. Main Results
We evaluate the performance of our proposed Delineate
Anything (DelAny) model and its smaller variant (DelAny-
S) on the FBIS-22M test set, comparing them with state-of-
the-art methods, including MultiTLF [14], SAM [17], and
SAM2 [26]. The results are presented in Table 2.
Our DelAny model achieves a significant improvement
in both mAP@0.5 and mAP@0.5:0.95 metrics, with scores
of 0.720 and 0.477, respectively, surpassing SAM2, the pre-
vious best-performing model, by 88.5% in mAP@0.5 and
103% in mAP@0.5:0.95. This establishes DelAny as the
new state-of-the-art for field boundary delineation. Impor-
tantly, DelAny achieves this improvement while also being
415 times faster in inference than SAM2, highlighting its
efficiency and suitability for real-time applications. The
DelAny-S variant, despite its smaller size and faster infer-
ence speed, also outperforms SAM2 by a significant mar-
gin, achieving a 65.5% gain in mAP@0.5 and a 63% gain
in mAP@0.5:0.95. Furthermore, DelAny-S is significantly
more efficient, achieving inference speeds 617 times faster
than SAM2 and 1.49 times faster than DelAny.
Figure 4 presents qualitative comparisons of Delineate
Anything with MultiTLF [14], SAM [17], and SAM2 [26].
MultiTLF performs well in scenarios with large fields and
Figure 4.Qualitative results on the FBIS-22M test set.Delineate Anything is compared to MultiTLF [14], SAM [17], and SAM2 [26].
For a fair comparison, the MultiTLF model was retrained using our FBIS-22M dataset. Different samples are carefully selected and
presented, varying in the size and density of the fields, to better illustrate the performance of each model under diverse conditions.

sparse boundaries, but struggles in images with smaller or
densely packed fields, often merging or missing them due
to its semantic segmentation approach. SAM tends to over-
segment, detecting irrelevant objects like water, grassland
and forests, leading to reduced precision, especially in im-
ages with non-agricultural areas. SAM2 slightly improves
on SAM but still faces similar challenges.
In contrast, Delineate Anything outperforms all methods
in every scenario, maintaining high accuracy in both sparse
and dense agricultural environments. Its instance segmen-
tation approach enables reliable field boundary delineation,
even in complex agricultural settings. These results demon-
strate model’s robustness and suitability for large-scale,

real-world applications.
4.4. Zero-Shot Cross-Region Generalization
To evaluate the generalization capabilities of Delineate
Anything, we conduct zero-shot experiments on geographic
regions not included in the training set. Specifically, we vi-
sualize the model’s predictions on regions in Brazil, Cam-
bodia, New Zealand, Rwanda, USA, Vietnam, and South
Africa, while the training data was exclusively sourced from
Europe. Since ground truth annotations are unavailable for
these regions, we focus on a qualitative evaluation. Figure 5
presents examples of the model’s performance in these un-
seen geographic contexts.
Figure 5. Qualitative results of zero-shot predictions. Delineate Anything is applied to geographic regions with different climates,
terrains, and agricultural practices, highlighting its field boundary delineation capabilities outside the training data.

Method mAP@0.5 mAP@0.5:0.95 Latency (ms)
MultiTLF†[14] 0.257 0.110 55.
SAM [17] 0.339 0.197 13605
SAM2 [26] 0.382 0.235 10370
DelAny-S 0.632 0.383 16.
DelAny 0.720 0.477 25.
Table 2.Quantitative comparisons on the FBIS-22M test set.
We compare our DelAny model and its smaller variant (DelAny-
S) against other methods.†: Models retrained on our FBIS-22M
dataset for fair comparison. Latency (ms) represents the total time
required to generate field boundaries. Best results are inbold.

The results highlight the model’s ability to adapt to di-
verse terrains, field patterns, and agricultural practices, in-
cluding smallholder farms, large industrial fields, and vary-
ing crop arrangements. This shows strong robustness and
potential for deployment across different agricultural set-
tings. The model consistently identifies field boundaries
even under challenging conditions, such as irregular field
shapes, varying textures, and diverse layouts. These quali-
tative results strongly support DelAny’s zero-shot general-
ization ability, demonstrating its suitability for scalable field
boundary mapping across global agricultural landscapes.

4.5. Ablation Studies
To assess the impact of dataset size and diversity, we con-
ducted ablation studies by training our Delineate Anything
model on subsets of FBIS-22M and compared its perfor-
mance to a model trained on the AI4Boundaries dataset [7].
Table 3 presents the results.
The AI4Boundaries training dataset consists of 45,
images, primarily from Sentinel-2 imagery, but suffers from
artifacts due to monthly compositing and lacks resolution
and satellite diversity, limiting its robustness. Our exper-
iments demonstrate that model trained on AI4Boundaries

Dataset # Images mAP@0.5 mAP@0.5:0.
AI4Boundaries [7] 45K 0.358 0.
FBIS-22M (subset) 45K 0.597 0.
FBIS-22M (subset) 150K 0.678 0.
FBIS-22M 636K 0.720 0.
Table 3.Impact of dataset size and diversity on model perfor-
mance.Performance comparison of the DelAny model trained on
the AI4Boundaries dataset and subsets of the FBIS-22M dataset,
highlighting the effect of dataset scale and diversity.
achieves only 0.358 mAP@0.5 and 0.211 mAP@0.5:0.95,
highlighting these limitations. In contrast, training on a
45,212-image subset of FBIS-22M improves performance
to 0.597 mAP@0.5 and 0.335 mAP@0.5:0.95. Expand-
ing to 150,000 images boosts it further to 0.678 mAP@0.
and 0.429 mAP@0.5:0.95. The full FBIS-22M dataset
yields the highest scores: 0.720 mAP@0.5 and 0.
mAP@0.5:0.95. A similar trend was observed with Mul-
tiTLF [14] trained on AI4Boundaries, where performance
dropped to 0.097 mAP@0.5 and 0.040 mAP@0.5:0.95.
These results show that with the same number of images,
the diverse FBIS-22M dataset performs much better than
AI4Boundaries, highlighting that having variety in resolu-
tion and sensors is just as important as the size of the dataset
for accurate field boundary detection.
5. Conclusion
This work addresses the need for automated agricultural
field boundary delineation by reformulating it as instance
segmentation task and introducing a large-scale, multi-
resolution dataset essential for training models robust
to varying image sources and resolutions. This dataset
bridges the gap in size and diversity compared to others in
computer vision. Our Delineate Anything model, designed
to handle diverse resolutions, significantly outperforms
existing methods, achieving faster inference and strong
zero-shot generalization. While further improvements
in generalization across geographic regions are needed,
this work advances the state-of-the-art in automated
field boundary delineation for agricultural applications,
with potential for large-scale areas, such as country level.

References
[1] Han Lin Aung, Burak Uzkent, Marshall Burke, David Lo-
bell, and Stefano Ermon. Farm parcel delineation using
spatio-temporal convolutional networks. In2020 IEEE/CVF
Conference on Computer Vision and Pattern Recognition
Workshops (CVPRW), pages 340–349. IEEE, 2020. 1, 2, 3,
4
[2] Gedas Bertasius, Jianbo Shi, and Lorenzo Torresani.
Deepedge: A multi-scale bifurcated deep network for top-
down contour detection. In2015 IEEE Conference on Com-
puter Vision and Pattern Recognition (CVPR), pages 4380–
IEEE, 2015. 2
[3] Keyan Chen, Chenyang Liu, Hao Chen, Haotian Zhang,
Wenyuan Li, Zhengxia Zou, and Zhenwei Shi. Rsprompter:
Learning to prompt for remote sensing instance segmenta-
tion based on visual foundation model.IEEE Transactions
on Geoscience and Remote Sensing, 2024. 3
[4] Ziyi Chen, Liai Deng, Yuhua Luo, Dilong Li, Jose Mar- ́
cato Junior, Wesley Nunes Gonc ̧alves, Abdul Awal Md Nu-
runnabi, Jonathan Li, Cheng Wang, and Deren Li. Road
extraction in remote sensing data: A survey. International
Journal of Applied Earth Observation and Geoinformation,
112, 2022. 2
[5] Zhe Chen, Yuchen Duan, Wenhai Wang, Junjun He, Tong
Lu, Jifeng Dai, and Yu Qiao. Vision transformer adapter for
dense predictions. arXiv preprint arXiv:2205.08534, 2022.
3
[6] Sophie Crommelinck, Rohan Bennett, Markus Gerke,
Francesco Nex, Michael Yang, and George Vosselman. Re-
view of automatic feature extraction from high-resolution
optical sensor data for uav-based cadastral mapping.Remote
Sensing, 8(8), 2016. 2
[7] Raphael d’Andrimont, Martin Claverie, Pieter Kempeneers, ̈
Davide Muraro, Momchil Yordanov, Devis Peressutti, Matej
Batic, and Franc ̧ois Waldner. Ai4boundaries: an open ai-ˇ
ready dataset to map field boundaries with sentinel-2 and
aerial photography.Earth System Science Data, 15(1):317–
329, 2023. 1, 2, 3, 4, 6, 8
[8] Foivos I. Diakogiannis, Franc ̧ois Waldner, Peter Caccetta,
and Chen Wu. Resunet-a: A deep learning framework for se-
mantic segmentation of remotely sensed data.ISPRS Journal
of Photogrammetry and Remote Sensing, 162:94–114, 2020.
3
[9] Hakan Erden, Murat Aslan, and Cemre Bahar Ozcanli. To
establish a new subsidy system. In2015 Fourth International
Conference on Agro-Geoinformatics (Agro-geoinformatics),
pages 57–60. IEEE, 2015. 1
[10] Yuxin Fang, Wen Wang, Binhui Xie, Quan Sun, Ledell Wu,
Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue
Cao. Eva: Exploring the limits of masked visual repre-
sentation learning at scale. In2023 IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR), pages
19358–19369. IEEE, 2023. 3
[11] Jordan Graesser and Navin Ramankutty. Detection of crop-
land field parcels from landsat imagery.Remote Sensing of
Environment, 201:165–180, 2017. 1, 2
[12] Kaiming He, Georgia Gkioxari, Piotr Dollar, and Ross Gir-
shick. Mask r-cnn. In2017 IEEE International Conference
on Computer Vision (ICCV). IEEE, 2017. 3
[13] Zhongxin Huang, Haitao Jing, Yueming Liu, Xiaomei Yang,
Zhihua Wang, Xiaoliang Liu, Ku Gao, and Haofeng Luo.
Segment anything model combined with multi-scale seg-
mentation for extracting complex cultivated land parcels in
high-resolution remote sensing images.Remote Sensing, 16
(18), 2024. 3
[14] Hannah Kerner, Saketh Sundar, and Mathan Satish. Multi-
region transfer learning for segmentation of crop field bound-
aries in satellite images with limited labels. arXiv preprint
arXiv:2404.00179, 2024. 3, 6, 7, 8
[15] Rahima Khanam and Muhammad Hussain. Yolov11: An
overview of the key architectural enhancements. arXiv
preprint arXiv:2410.17725, 2024. 3, 6
[16] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao,
Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer White-
head, Alexander C. Berg, Wan-Yen Lo, Piotr Doll ́ar, and
Ross Girshick. Segment anything. In2023 IEEE/CVF In-
ternational Conference on Computer Vision (ICCV). IEEE,
2, 3, 4
[17] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao,
Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer White-
head, Alexander C Berg, Wan-Yen Lo, et al. Segment any-
thing. InICCV, pages 4015–4026, 2023. 6, 7, 8
[18] Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Ui-
jlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan
Popov, Matteo Malloci, Alexander Kolesnikov, Tom Duerig,
and Vittorio Ferrari. The open images dataset v4.Interna-
tional Journal of Computer Vision, 128(7):1956–1981, 2020.
2, 4
[19] Mykola Lavreniuk. Spidepth: Strengthened pose informa-
tion for self-supervised monocular depth estimation. arXiv
preprint arXiv:2404.12501, 2024. 2
[20] Mykola Lavreniuk, Shariq Farooq Bhat, Matthias Muller, ̈
and Peter Wonka. Evp: Enhanced visual perception us-
ing inverse multi-attentive feature refinement and regularized
image-text alignment. arXiv preprint arXiv:2312.08548,
3
[21] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays,
Pietro Perona, Deva Ramanan, Piotr Doll ́ar, and C. Lawrence
Zitnick. Microsoft COCO: Common Objects in Context,
pages 740–755. Springer International Publishing, Cham,
2， 4， 6
[22] 刘云、程明明、胡晓伟、王凯和
向白。边缘检测的更丰富的卷积特征。
2017年IEEE计算机视觉与模式
识别会议（CVPR），第5872–5881页。IEEE，2017年。2
[23] Khairiya Mudrik Masoud、Claudio Persello 和 Valentyn A.
Tolpekin。农业田地边界的划定
Sentinel-2 图像，采用基于全卷积网络的新型超分辨率等高线反
探器。《遥感
》，12（1），2019年。2
[24] 卢卡斯·普拉多·奥斯科、吴秋生、爱德华多·洛佩斯·德莱莫斯、
韦斯利·努内斯·贡茨·阿尔维斯、安娜·宝拉·马奎斯·拉莫斯、
乔纳森·李和何塞·马尔卡托，朱尼尔。任意片段
模型（SAM）用于遥感应用：从零到
单次。《国际应用地球观测
与地理信息杂志》，124期，2023年。3
[25] 克劳迪奥·佩尔塞洛、耶罗恩·格里夫特、范欣扬、克劳迪娅·
帕里斯、罗尼·汉施、米拉·科瓦和安德鲁·尼尔森。̈
Ai4smallfarms：东南亚小农场作物田间划分
数据集。IEEE地球科学与遥
感快报，20卷1–5期，2023年。1， 2， 3， 4
[26] 尼基拉·拉维、瓦伦丁·加布尔、胡元婷、
胡荣恒、柴坦尼亚·赖亚利、马腾宇、海瑟姆·凯德尔、罗曼
·拉德尔、克洛伊·罗兰、劳拉·古斯塔夫森等。“Sam 2：
在图像和视频中切割任何内容。arXiv 预印本
arXiv：2408.00714,2024年。3， 6， 7， 8
[27] 丹尼埃莱·雷格·坎布林、埃莱奥诺拉·波埃塔、埃莉安娜·帕斯托尔、塔尼娅
·塞尔基泰利、埃琳娜·巴拉利斯和保罗·加尔萨。Kan，你看到
了吗？kans 和 sentinel 用于有效且可解释的作物田
间分割。arXiv 预印本 arXiv：2408.07040,2024。3
[28] 西米奥·任、弗朗切斯科·卢齐、萨阿德·拉赫里奇、卡勒布·卡萨乌、
莱斯利·M·柯林斯、凯尔·布拉德伯里和乔丹·M·马洛夫。
从太空中切割任何东西？载于
IEEE/CVF计算机视觉应用
冬季会议（WACV）论文集，第8355–8365页，2024年。3
[29] 克里斯托夫·舒曼、罗曼·博蒙特、理查德·文库、
凯德·戈登、罗斯·怀特曼、梅赫迪·切尔蒂、西奥
·库姆斯、阿鲁什·卡塔、克莱顿·穆利斯、米切尔·沃茨曼
、帕特里克·施拉莫夫斯基、斯里瓦察·昆杜尔蒂、凯瑟琳
·克劳森、路德维希·施密特、罗伯特·卡兹马尔奇克和耶
- 尼亚·伊采夫。Laion-5b：一个用于训练
下一代图像-文本模型的开放大规模数据集。arXiv 预印本
arXiv：2210.08402,2022年。2，
4 [30] 罗德里戈·菲尔·兰格尔·西兹，V·伊托尔·纳西门托·洛伦斯·伊奥·
盖沃塔，卢卡斯·沃洛亨·奥尔多尼·希兹，安娜·弗拉维亚·卡尔-
拉拉·博纳米戈·希兹，瓦拉斯·桑托斯·盖沃塔，布鲁诺
·席尔瓦·奥利维拉·希兹，以及马特乌斯·内维斯·巴雷托·希兹。一个
用于农田边界检测与
分割的统一框架。2024年IEEE/CVF计算机视觉应用
研讨会冬季会议（WACVW），第636–644页
。IEEE，2024年。3
[31] 孙嘉林、帅炎、姚晓长、高冰波和杨
建宇。一种基于模型的片段式弱超
感知学习方法，用于使用哨兵
时间序列图像进行裁剪映射。《国际应用地球
观测与地理信息杂志》，133卷，2024年。3
[32] 普拉秋什·特里帕蒂、凯西·贝利斯、吴凯尔、贾尔斯·沃森
和江瑞哲。研究无训练标签绘制小农农业
田边界的任意
基础模型。arXiv 预印本
arXiv：2407.01846,2024。3
[33] 马蒂亚斯·P·瓦格纳和娜塔莎·奥佩尔特。深度学习和
基于图形的自适应种植轮廓用于农业田间
开采。《遥感》，12（12），2020年。2

[34] Matthias P. Wagner and Natascha Oppelt. Extracting agricul-
tural fields from remote sensing imagery using graph-based
growing contours.Remote Sensing, 12(7), 2020. 1, 2
[35] Franc ̧ois Waldner, Foivos I. Diakogiannis, Kathryn Batche-
lor, Michael Ciccotosto-Camp, Elizabeth Cooper-Williams,
Chris Herrmann, Gonzalo Mata, and Andrew Toovey. De-
tect, consolidate, delineate: Scalable mapping of field
boundaries using satellite images.Remote Sensing, 13(11),
1,2,3
[36] 奥王、辉辰、刘丽浩、陈凯、林子佳、韩俊
公和贵光丁。Yolov10：实时端
到端物体检测。arXiv 预印本 arXiv：2405.14458，
3， 6
[37] 王立博、方胜慧、孟晓良、李瑞。
用视觉变换器进行建筑提取。IEEE地球科学与遥感通讯
，60卷：1–11期，2022年。2
[38] 莫王、王景、崔云鹏、刘娟和李晨。
利用卫星图像
年龄分割进行农业田界划定，实现高分辨率作物绘图：水稻案例
研究。《农学》，12（10），2022年。3
[39] 王雪莉、弗朗索瓦·沃尔德纳和大卫·B·洛贝尔。
通过迁移学习和弱监督
机制，解锁小农
户系统中的大规模作物田间划分。《遥感》，14（22），2022年。1， 2， 3， 4
[40] 巴里·沃特金斯和阿德里安·范·尼克。基于
对象的图像分析方法在使用多时空哨兵-2图像进行场边界
划定的比较。《农业中的
计算机与电子》，158：294–302,2019年。
1， 2
[41] 谢赛宁与涂卓文。整体嵌套边缘
析理。2015年IEEE国际
计算视觉会议（ICCV）。IEEE，2015年。2
[42] L. Yan 和 D.P. Roy。基于多时段陆地卫星数据的美国作物田
面积量化。《环境遥
感》，172：67–86,2016年。1， 2
[43] 周博雷、赵恒、泽维尔·普伊格、桑贾·菲德勒、阿德拉
·巴里乌索和安东尼奥·托拉尔巴。通过
ade20k数据集进行场景解析。2017年IEEE计算机视觉
与模式识别会议（CVPR）。IEEE，2017年。2
[44] 卓凡宗、宋光禄和于柳。Detrs与协作
混合作业培训。2023年IEEE/CVF
国际计算机视觉会议（ICCV），第6725–6735页
。IEEE，2023年。3


