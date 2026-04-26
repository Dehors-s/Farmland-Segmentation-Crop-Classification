Delineate Anything: Resolution-Agnostic Field Boundary Delineation on
|     |     |     |     |     | Satellite | Imagery |     |     |     |
| --- | --- | --- | --- | --- | --------- | ------- | --- | --- | --- |
MykolaLavreniuk1,2,NataliiaKussul3,AndriiShelestov2,4,BohdanYailymov2,YevheniiSalii2,4,
VolodymyrKuzin2,4,ZoltanSzantoi1
1EuropeanSpaceAgency,2SpaceResearchInstituteNASU-SSAU,3UniversityofMaryland,
4NationalTechnicalUniversityofUkraine“IgorSikorskyKyivPolytechnicInstitute”
5202 rpA 3  ]VC.sc[  1v43520.4052:viXra
Figure1.WorkflowoftheDelineateAnythingmodelforfieldinstancesegmentationandfieldboundaryextractionfromarbitraryresolution
satelliteimagery,trainedonourlarge-scaleFieldBoundaryInstanceSegmentationdataset(FBIS-22M),containing22Mfieldboundaries.
|                                                     |             | Abstract |                 |                 |            | lavreniuk.github.io/Delineate-Anything. |     |     |     |
| --------------------------------------------------- | ----------- | -------- | --------------- | --------------- | ---------- | --------------------------------------- | --- | --- | --- |
| The accurate                                        | delineation |          | of agricultural | field           | boundaries |                                         |     |     |     |
| from satellite                                      | imagery     | is       | vital for       | land management | and        |                                         |     |     |     |
| cropmonitoring.However,currentmethodsfacechallenges |             |          |                 |                 |            | 1.Introduction                          |     |     |     |
| due to limited                                      | dataset     | sizes,   | resolution      | discrepancies,  | and        |                                         |     |     |     |
diverse environmental conditions. We address this by re- Thedelineationofagriculturalfieldboundariesfromsatel-
|             |     |         |                       |     |            | lite imagery | is crucial | for precision agriculture, | land man- |
| ----------- | --- | ------- | --------------------- | --- | ---------- | ------------ | ---------- | -------------------------- | --------- |
| formulating | the | task as | instance segmentation |     | and intro- |              |            |                            |           |
ducing the Field Boundary Instance Segmentation - 22M agement,policymakingandcropmonitoring.TheEuropean
Union’sLandParcelIdentificationSystem(LPIS)servesas
dataset(FBIS-22M),alarge-scale,multi-resolutiondataset
comprising672,909high-resolutionsatelliteimagepatches akeytoolfordefiningagriculturalfieldboundariestosup-
(ranging from 0.25 m to 10 m) and 22,926,427 instance portlandusemonitoringandsubsidyallocation[9]. How-
|          |            |         |               |           |         | ever, many | regions in | the world lack such systems, | result- |
| -------- | ---------- | ------- | ------------- | --------- | ------- | ---------- | ---------- | ---------------------------- | ------- |
| masks of | individual | fields, | significantly | narrowing | the gap |            |            |                              |         |
betweenagriculturaldatasetsandthoseinothercomputer ing in outdated cadastral maps that prevent effective agri-
visiondomains. WefurtherproposeDelineateAnything,an culturalmanagement. Themanual,labor-intensivecreation
instancesegmentationmodeltrainedonournewFBIS-22M and maintenance of LPIS data [35] further highlight the
dataset. Our proposed model sets a new state-of-the-art, needforautomated,scalablesolutionstodetectfieldbound-
ariesfromsatellitedata.
achievingasubstantialimprovementof88.5%inmAP@0.5
and 103% in mAP@0.5:0.95 over existing methods, while Traditionalcomputervisiontechniques,likeedgedetec-
alsodemonstratingsignificantlyfasterinferenceandstrong tionandclustering[11,34,40,42],oftenfailtogeneralize
zero-shot generalization across diverse image resolutions acrossdiversefieldtypes,geographicregions,andenviron-
and unseen geographic regions. Code, pre-trained mod- mental conditions. The recent availability of datasets like
els,andtheFBIS-22Mdatasetareavailableathttps:// AI4Boundaries[7],alongwithothers[1,25,39],hasfacil-
1

itated the development of deep learning (DL) approaches. ments in mean Average Precision (mAP): from 0.382 to
However, the applying of current DL methods for field 0.720 (+88.5%) for mAP@0.5 and from 0.235 to 0.477
boundarydetectionlagsbehindadvancementsinothercom- (+103%)formAP@0.5:0.95. Furthermore,ourmethodhas
putervisiondomains,primarilyduetolimitationsindataset significantly faster inference times compared to its closest
size and quality. Compared to large-scale datasets like rival, enhancing its practical usability. Notably, we also
ADE20K[43],OpenImages[18],COCO[21],SA-1B[16], demonstrate the strong zero-shot capabilities of our model
and LAION [29], existing agricultural datasets are signifi- ongeographicallydistinctlocationsnotpresentinthetrain-
| cantly smaller, | hindering | model | generalization |     | and perfor- | ingdataset.                             |     |     |     |
| --------------- | --------- | ----- | -------------- | --- | ----------- | --------------------------------------- | --- | --- | --- |
| mance.          |           |       |                |     |             | Insummary,ourcontributionsarethreefold: |     |     |     |
Another challenge arises from the reliance on 10m • Anoveltaskformulationoffieldboundarydetectionasan
medium-resolution Sentinel-2 imagery in many datasets. instance segmentation problem, addressing the inherent
limitationsofsemanticsegmentationforthistask.
| While sufficient |     | for larger | fields, this | resolution | fails for |     |     |     |     |
| ---------------- | --- | ---------- | ------------ | ---------- | --------- | --- | --- | --- | --- |
smaller, irregular fields, common in smallholder farm- • A new, large-scale, multi-resolution satellite imagery
ing. Consequently,modelstrainedexclusivelyonSentinel- datasetforrobustfieldboundarydelineation.
2 imagery often exhibit significant performance degrada- • A resolution-agnostic model that significantly outper-
tion when applied to higher-resolution data acquired from formscurrentstate-of-the-artmethodsforfieldboundary
dronesorothersatellites. ThewidelyusedAI4Boundaries detection, while exhibiting superior inference speed and
dataset [7], while a valuable contribution, suffers from ar- strongzero-shotgeneralizationacrossdiverseresolutions
tifacts introduced by monthly image compositing, such as andgeographiclocations.
| blurred           | boundaries, | which    | further impact | model | perfor-     |               |     |     |     |
| ----------------- | ----------- | -------- | -------------- | ----- | ----------- | ------------- | --- | --- | --- |
| manceandaccuracy. |             |          |                |       |             | 2.RelatedWork |     |     |     |
| Critically,       | most        | existing | DL approaches  |       | treat field |               |     |     |     |
2.1.TraditionalMethods
| boundary | detection | as a semantic | segmentation |     | problem, |     |     |     |     |
| -------- | --------- | ------------- | ------------ | --- | -------- | --- | --- | --- | --- |
classifyingeachpixelasbelongingtoeitherafieldbound- Early approaches employed classical image processing
ary or the background [1, 39]. This approach, typically techniques such as edge detection (e.g., Canny, Sobel,
implemented using encoder-decoder architectures such as LoG) and clustering based on spectral or textural fea-
U-Net or their variants, focuses on detecting continuous tures(e.g.,graph-basedsegmentation,SimpleLinearItera-
boundarylines.However,forpracticalagriculturalmanage- tive Clustering (SLIC) segmentation, watershed segmenta-
mentandcadastralapplications,identifyingindividualfield tion)[6,11,34,40,42].Thesemethods,whilecomputation-
objects is essential. Even minor segmentation errors can allyefficient,oftenproducednon-closedboundaries,requir-
leadtotheerroneousmergingofadjacentfields,resultingin ingpost-processingandfilteringtoremoveirrelevantedges
substantialinaccuraciesinareacalculationsandlandparcel notcorrespondingtoagriculturalfieldsusingadditionalin-
identification. While post-processing steps have been pro- formationfromcroplandandcroptypemaps. Thesemeth-
| posed to | mitigate | this issue, | they often | lack the | necessary |     |     |     |     |
| -------- | -------- | ----------- | ---------- | -------- | --------- | --- | --- | --- | --- |
odsarealsoinherentlysensitivetonoiseandvaryingillumi-
robustness and generalizability across diverse agricultural nationconditionscommoninsatelliteimagery. Theselimi-
landscapesandfieldtypes[35]. tationsmotivatedtheexplorationofmorerobusttechniques,
To overcome these limitations, we introduce a new, particularlywiththeriseofdeeplearning.
| large-scale | dataset, | more than | 12 times | larger | than ex- |     |     |     |     |
| ----------- | -------- | --------- | -------- | ------ | -------- | --- | --- | --- | --- |
2.2.DeepLearningforSemanticSegmentation
| isting ones, | incorporating | imagery | from | multiple | sources |     |     |     |     |
| ------------ | ------------- | ------- | ---- | -------- | ------- | --- | --- | --- | --- |
(Sentinel-2,Planet,Maxar,Pleiades,andorthophotos)with
Deeplearninghasshownpromiseinrelatedremotesensing
a wide range of high resolutions (from 0.25m to 10m). tasks,suchasbuilding[37]androadextraction[4],aswell
This enables training a single, highly generalizable model as general boundary detection [2, 19, 22, 41]. However,
thatperformseffectivelyacrossdiverseresolutionsandsen-
thesemethodsprimarilyfocusonsemanticboundarydetec-
sor types, enhancing scalability in agricultural contexts. tion,oftenrequiringpost-processingtoformclosedobjects
| Additionally, | we  | propose a | novel resolution-agnostic |     | in- |             |                           |                  |      |
| ------------- | --- | --------- | ------------------------- | --- | --- | ----------- | ------------------------- | ---------------- | ---- |
|               |     |           |                           |     |     | and failing | to distinguish individual | field instances. | Sev- |
stance segmentation approach for field delineation (Fig- eral works have applied deep learning directly to agricul-
ure1),which,byframingthetaskasidentifyingindividual turalfieldboundarydelineation. Someearlydeeplearning
fieldinstances,improveshandlingofcomplexfieldshapes,
|     |     |     |     |     |     | approaches | combined deep learning | with classical | meth- |
| --- | --- | --- | --- | --- | --- | ---------- | ---------------------- | -------------- | ----- |
preventsfieldmerging,anddeliversmoreaccurateandprac- odssuchasadaptivegraph-basedgrowingcontoursforfield
ticallyrelevantoutputsforreal-worldagriculturalmanage- extraction [33]. Fully convolutional networks (FCNs) and
mentandlandadministration.
contourclosingprocedureshavebeenexploredforfieldde-
We evaluate our model against state-of-the-art methods lineation,particularlyinsmallholderfarms[25].FCNshave
on our new dataset, demonstrating a substantial improve- alsobeenusedforsuper-resolutioncontourdetection[23].
2

ResUNet-a, a deep learning framework for semantic seg- leading to low precision. Furthermore, its computational
mentationofremotelysenseddata,hasbeenappliedtofield costlimitslarge-scaleapplicability.Whilesubsequentwork
boundarydetection[8]. U-Net-basedFCNshavebeenused has explored refinements like multi-scale processing [13],
forspecificcroptypessuchasricepaddydelineation[38]. weakly supervised learning [31], and prompt engineer-
Recentworkshaveimprovedsegmentationmodelsandloss ing[24,28], thesemethodsrequireadditionaldatasuchas
functions,suchastheResidualandRecurrentAttentionU- prompts or weak labels. These approaches can be effec-
Net (R2AttU-Net) with Lova´sz-Softmax loss [30], and U- tiveforscenarioswheresuchdataisavailableandthegoal
NetwithKolmogorov-ArnoldNetworks [27]. is to refine boundaries for specific fields. However, they
A significant step towards addressing the limitations of donotaddressthefundamentallimitationsofSAM’szero-
shottransferabilityingeneral,particularlyforlargeterrito-
| purely boundary-based |     |     | methods | was | the | introduction | of  |     |     |     |     |     |     |
| --------------------- | --- | --- | ------- | --- | --- | ------------ | --- | --- | --- | --- | --- | --- | --- |
FracTALResUNet[35]. Recognizingthechallengesindi- rieswherenosuchpriorinformationexists. Evenwiththe
rectlypredictingclosedboundaries,thisworkincorporated newer SAM2 model [26], we observed similar issues, in-
a distance-to-boundary channel alongside hierarchical wa- dicating that these core challenges persist even in updated
| tershed | segmentation |         | as a post-processing |          |     | step.      | This ap- | versions.   |                    |     |     |               |     |
| ------- | ------------ | ------- | -------------------- | -------- | --- | ---------- | -------- | ----------- | ------------------ | --- | --- | ------------- | --- |
|         |              |         |                      |          |     |            |          | To overcome | these limitations, |     | our | work directly | ad- |
| proach  | aimed to     | produce | more                 | complete |     | and closed | con-     |             |                    |     |     |               |     |
tours, moving closer to instance-level segmentation as ex- dresses the data bottleneck and the need for efficient, ac-
plicitlystatedbytheauthors. Subsequenteffortsbuiltupon curate instance segmentation. We introduce the Delin-
|            |          |          |     |      |         |         |     | eateAnythingframework, |     | whichincludesaninstanceseg- |     |     |     |
| ---------- | -------- | -------- | --- | ---- | ------- | ------- | --- | ---------------------- | --- | --------------------------- | --- | --- | --- |
| this idea. | Transfer | learning |     | with | FracTAL | ResUNet | was |                        |     |                             |     |     |     |
explored for smallholder farming systems [39], leverag- mentationmodelandthenewlarge-scale,multi-resolution,
ingthebenefitsofthedistance-to-boundaryrepresentation. instance-annotated FBIS-22M dataset. This framework
Other works further developed this direction, employing achieves significant advancements over existing seman-
similarstrategiesofincorporatingboundarydistanceinfor- tic segmentation methods and demonstrates clear advan-
tagesoverzero-shotinstancesegmentationapproacheslike
| mation | within | a multi-task |     | learning | framework | to  | predict |     |     |     |     |     |     |
| ------ | ------ | ------------ | --- | -------- | --------- | --- | ------- | --- | --- | --- | --- | --- | --- |
field extent, boundaries, and distance to boundaries [14]. SAM[16,32]andSAM2[26].
| While these | methods, |     | including | efforts | focused | on  | multi- |     |     |     |     |     |     |
| ----------- | -------- | --- | --------- | ------- | ------- | --- | ------ | --- | --- | --- | --- | --- | --- |
3.Methodology
| task learning, | model          |          | architecture |              | improvements, |             | and loss |                  |            |                   |     |     |              |
| -------------- | -------------- | -------- | ------------ | ------------ | ------------- | ----------- | -------- | ---------------- | ---------- | ----------------- | --- | --- | ------------ |
| function       | modifications, |          | improve      | boundary     |               | prediction, | they     |                  |            |                   |     |     |              |
|                |                |          |              |              |               |             |          | In this section, | we present | our contributions |     | to  | the field of |
| still operate  | within         | semantic |              | segmentation |               | and thus    | do not   |                  |            |                   |     |     |              |
boundarydelineation,beginningwithareformulationofthe
| inherently | provide | instance-level |     |     | information. | Although |     |                  |               |       |           |     |             |
| ---------- | ------- | -------------- | --- | --- | ------------ | -------- | --- | ---------------- | ------------- | ----- | --------- | --- | ----------- |
|            |         |                |     |     |              |          |     | task as instance | segmentation, | which | addresses |     | the limita- |
post-processingstepsareincorporated[35],theyoftenrely
|     |     |     |     |     |     |     |     | tionsofexistingmethods. |     | WeintroduceFBIS-22M,anew |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------------------- | --- | ------------------------ | --- | --- | --- |
onheuristicsandlackgeneralizability.
|     |     |     |     |     |     |     |     | dataset specifically | designed            | for            | this purpose, | and       | demon- |
| --- | --- | --- | --- | --- | --- | --- | --- | -------------------- | ------------------- | -------------- | ------------- | --------- | ------ |
|     |     |     |     |     |     |     |     | strate its           | utility by training | and evaluating |               | Delineate | Any-   |
2.3.MovingTowardsInstance-LevelSegmentation
thing,amodelthatsetsanewstate-of-the-artinfieldbound-
| The core         | challenge |                  | for accurate |              | field identification |              | and      | arydelineation. |                |     |             |     |        |
| ---------------- | --------- | ---------------- | ------------ | ------------ | -------------------- | ------------ | -------- | --------------- | -------------- | --- | ----------- | --- | ------ |
| area calculation |           | is transitioning |              | from         | semantic             | to           | instance |                 |                |     |             |     |        |
|                  |           |                  |              |              |                      |              |          | 3.1. Reframing  | Field Boundary |     | Delineation |     | as In- |
| segmentation.    |           | While            | instance     | segmentation |                      | has advanced |          |                 |                |     |             |     |        |
stanceSegmentation
| significantly | in  | computer | vision, | from | Mask | R-CNN | [12] |     |     |     |     |     |     |
| ------------- | --- | -------- | ------- | ---- | ---- | ----- | ---- | --- | --- | --- | --- | --- | --- |
to state-of-the-art architectures like Co-DETR [44], ViT- Traditional semantic segmentation approaches for field
Adapter [5], EVA [10], EVP [20], and recent real-time boundary detection encounter notable challenges, espe-
| YOLO | variants | [15, | 36], | its application |     | to agricultural |     |             |                |          |     |              |      |
| ---- | -------- | ---- | ---- | --------------- | --- | --------------- | --- | ----------- | -------------- | -------- | --- | ------------ | ---- |
|      |          |      |      |                 |     |                 |     | cially when | assessed using | boundary |     | Intersection | over |
fields is limited by the lack of suitable, instance-annotated Union (IoU). As illustrated in Figure 2, boundary IoU
datasets. Existingdatasets[1,7,25,39]areoftenlimitedin scores are highly sensitive to small misalignments, even
sizeandresolution(e.g.,10mSentinel-2). whenpredictedboundariescloselyfollowthegroundtruth.
The emergence of the Segment Anything Model For instance, a slight offset of only a few pixels results in
(SAM) [16] presented a promising new direction by offer- aboundaryIoUof0.08(Figure2b),excessivelypenalizing
ing impressive zero-shot segmentation capabilities. This the model for an error that has minimal practical impact.
approach, explored in the context of satellite-based field Incontrast, instanceIoUremainsmorerobustinsuchsce-
boundarydetection[32],offeredthepotentialtoperformin- narios,yieldingascoreof0.98(Figure2e),asitprioritizes
stance segmentation without extensive annotated datasets. accuratefielddelineationratherthanpixel-perfectboundary
| However, | as also | highlighted |     | in [3, | 13, 32] | and confirmed |     | alignment. |     |     |     |     |     |
| -------- | ------- | ----------- | --- | ------ | ------- | ------------- | --- | ---------- | --- | --- | --- | --- | --- |
by our own investigations, direct application of SAM to More critically, boundary IoU fails to account for seg-
agricultural fields reveals limitations. SAM tends to over- mentation errors that lead to adjacent fields being incor-
segment,detectingirrelevantobjectslikeroadsandforests, rectlymergedintoasingleobject. AsshowninFigure2,a
3

|     |     |     |     |     |     |     | Dataset |     |     | Resolution |     | #Images | #Instances |     |
| --- | --- | --- | --- | --- | --- | --- | ------- | --- | --- | ---------- | --- | ------- | ---------- | --- |
GeneralComputerVisionDatasets
|     |     |     |     |     |     |     | LAION-5B[29]   |     |     | -   |     | 5.85B |     | -    |
| --- | --- | --- | --- | --- | --- | --- | -------------- | --- | --- | --- | --- | ----- | --- | ---- |
|     |     |     |     |     |     |     | COCO[21]       |     |     | -   |     | 330K  |     | 1.5M |
|     |     |     |     |     |     |     | OpenImages[18] |     |     | -   |     | 998K  |     | 2.8M |
|     |     |     |     |     |     |     | SA-1B[16]      |     |     | -   |     | 11M   |     | 1.1B |
FieldBoundaryDelineationDatasets
|     |     |     |     |     |     |     | FarmParcel[1]                                     |     |     | 10m       |     | 2K   |     | -     |
| --- | --- | --- | --- | --- | --- | --- | ------------------------------------------------- | --- | --- | --------- | --- | ---- | --- | ----- |
|     |     |     |     |     |     |     | India10K[39]                                      |     |     | -         |     | -    |     | 10K   |
|     |     |     |     |     |     |     | AI4SmallFarms[25]                                 |     |     | 10m       |     | 62   |     | 439K  |
|     |     |     |     |     |     |     | AI4Boundaries[7]                                  |     |     | 1m&10m    |     | 55K  |     | 2.5M  |
|     |     |     |     |     |     |     | FBIS-22M                                          |     |     | 0.25m-10m |     | 673K |     | 22.9M |
|     |     |     |     |     |     |     | Table1. ComparisonofFBIS-22Mwithexistingdatasets. |     |     |           |     |      |     | The   |
tablecomparesFBIS-22Mwithgeneralcomputervisiondatasets
Figure2.Comparisonoftaskformulationsandevaluationmet-
andexistingfieldboundarydelineationdatasetsbasedonsatellite
| ricsforfieldboundarydelineation. |     |     |     | Thetoprowillustratesfield |     |     |     |     |     |     |     |     |     |     |
| -------------------------------- | --- | --- | --- | ------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
boundary masks (semantic segmentation), while the bottom row imagery,highlightingFBIS-22M’sresolutionrangeandscale.
| shows individual | field     | masks | (instance | segmentation). |          | Ground     |     |     |     |     |     |     |     |     |
| ---------------- | --------- | ----- | --------- | -------------- | -------- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- |
| truth examples   | are shown | in    | (a) and   | (d).           | Slightly | misaligned |     |     |     |     |     |     |     |     |
boundaries result in a boundary IoU of 0.08 (b) and an instance resolutions. While general computer vision datasets such
IoU of 0.98 (e). Partially detected boundaries yield a boundary as LAION-5B with 5.85 billion images [29] and SA-1B
IoUof0.93(c)andaninstanceIoUof0.54(f).
|     |     |     |     |     |     |     | with 1.1                                   | billion   | instance | masks  | [16]         | provide | large-scale |          |
| --- | --- | --- | --- | --- | --- | --- | ------------------------------------------ | --------- | -------- | ------ | ------------ | ------- | ----------- | -------- |
|     |     |     |     |     |     |     | resources                                  | for other | vision   | tasks, | agricultural |         | datasets    | for      |
|     |     |     |     |     |     |     | fieldboundarydetectionhavebeenmuchsmaller. |           |          |        |              |         |             | Existing |
partiallydetectedboundaryresultsinahighboundaryIoU
datasetsrangefromjust62imagesinAI4SmallFarms[25]
scoreof0.93(Figure2c),despitesignificantmergingofdis-
|                                                  |                                           |     |     |     |     |      | to55thousandsimagesinAI4Boundaries[7], |              |     |               |     |        | limitingthe |     |
| ------------------------------------------------ | ----------------------------------------- | --- | --- | --- | --- | ---- | -------------------------------------- | ------------ | --- | ------------- | --- | ------ | ----------- | --- |
| tinctfields.                                     | However,instanceIoUmoreaccuratelyreflects |     |     |     |     |      |                                        |              |     |               |     |        |             |     |
|                                                  |                                           |     |     |     |     |      | ability to                             | train robust | and | generalizable |     | models | (Table      | 1). |
| theseverityofthiserror,droppingto0.54(Figure2f). |                                           |     |     |     |     | This |                                        |              |     |               |     |        |             |     |
Toaddressthislimitation,weintroducetheFieldBoundary
discrepancyhighlightstheinadequacyofboundaryIoUfor
|              |               |               |        |             |            |         | Instance       | Segmentation |         | - 22M           | (FBIS-22M) |          | dataset,  | which   |
| ------------ | ------------- | ------------- | ------ | ----------- | ---------- | ------- | -------------- | ------------ | ------- | --------------- | ---------- | -------- | --------- | ------- |
| real-world   | agricultural  | applications, |        | where       | preserving | the     |                |              |         |                 |            |          |           |         |
|              |               |               |        |             |            |         | is the largest | dataset      | for     | field           | boundary   | instance |           | segmen- |
| distinctness | of individual |               | fields | is critical | for tasks  | such as |                |              |         |                 |            |          |           |         |
|              |               |               |        |             |            |         | tation. It     | contains     | 672,909 | high-resolution |            |          | satellite | image   |
cropmonitoringandyieldestimation.
patchesand22,926,427instancemasksofindividualfields,
| To overcome | these | limitations, |     | we reformulate |     | the field |           |      |         |       |        |      |                |     |
| ----------- | ----- | ------------ | --- | -------------- | --- | --------- | --------- | ---- | ------- | ----- | ------ | ---- | -------------- | --- |
|             |       |              |     |                |     |           | making it | more | than 12 | times | larger | than | the previously |     |
boundarydelineationtaskasaninstancesegmentationprob-
largestdataset,AI4Boundaries[7].
| lem. Inthisapproach, |     | eachfieldistreatedasadistinctin- |     |     |     |     |        |      |        |            |     |          |     |           |
| -------------------- | --- | -------------------------------- | --- | --- | --- | --- | ------ | ---- | ------ | ---------- | --- | -------- | --- | --------- |
|                      |     |                                  |     |     |     |     | To the | best | of our | knowledge, |     | FBIS-22M | is  | the first |
stance,andthegoalistopredictclosed-fieldmasks,which
|     |     |     |     |     |     |     | dataset to | incorporate |     | high-resolution |     | imagery | from | com- |
| --- | --- | --- | --- | --- | --- | --- | ---------- | ----------- | --- | --------------- | --- | ------- | ---- | ---- |
avoidscommonissuessuchasboundarymisalignmentand
field merging. As shown in Figure 1, these instance-level mercial satellites. This unique feature enhances its value
|           |           |           |      |       |            |       | as a resource | for | field | boundary | detection |     | in diverse | agri- |
| --------- | --------- | --------- | ---- | ----- | ---------- | ----- | ------------- | --- | ----- | -------- | --------- | --- | ---------- | ----- |
| masks can | be easily | converted | into | field | boundaries | using |               |     |       |          |           |     |            |       |
simple post-processing techniques like contour extraction. culturallandscapes. FBIS-22Mintegratesdatafrommulti-
This reformulation aligns the evaluation metric (instance plesatelliteplatforms,includingSentinel-2,Planet,Maxar,
Pleiades,andpubliclyavailablesatellitesources,providing
| IoU) with | the practical | requirements |     | of  | field | delineation, |     |     |     |     |     |     |     |     |
| --------- | ------------- | ------------ | --- | --- | ----- | ------------ | --- | --- | --- | --- | --- | --- | --- | --- |
providingamorerobustmethodologyforbothtrainingand diversedatatypesandenablingcompatibilitywithdifferent
sensortechnologies.
modelevaluation.InstanceIoUoffersseveraladvantages:it
islesssensitivetominorboundaryvariationswhilepenaliz- FBIS-22M offers a broad range of resolutions from
ingthemergingoffields,whichsignificantlyaffectstheac- 0.25m to 10m, covering both smallholder and large-scale
curacyofthemodel. Byreformulatingthetaskasinstance agricultural applications. Specifically, the dataset includes
segmentation, we advance the precision and reliability of images with resolutions of 0.25m, 0.3m, 0.5m, 1m, 1.2m,
|     |     |     |     |     |     |     | 2m,3m,and10m. |     | Thisdiversityinresolutionsenablesthe |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------- | --- | ------------------------------------ | --- | --- | --- | --- | --- |
fieldboundarydetectionmodels,markingasignificantstep
forwardinagriculturalimageanalysis. accuratesegmentationofbothsmall,irregularfieldsaswell
|     |     |     |     |     |     |     | as larger, | expansive | agricultural |     | areas, | supporting |     | general- |
| --- | --- | --- | --- | --- | --- | --- | ---------- | --------- | ------------ | --- | ------ | ---------- | --- | -------- |
3.2.FieldBoundaryInstanceSegmentationDataset
|                |           |     |                |     |       |            | ization across | different |     | field types | and | environmental |     | con- |
| -------------- | --------- | --- | -------------- | --- | ----- | ---------- | -------------- | --------- | --- | ----------- | --- | ------------- | --- | ---- |
| Field boundary | detection |     | in agriculture |     | faces | challenges | ditions.       |           |     |             |     |               |     |      |
due to the variability in field sizes, shapes, and image FBIS-22M also provides significant geographic diver-
4

Figure3.ExamplesoffieldboundaryinstancesegmentationfromourFBIS-22Mdataset.TheFBIS-22Mdatasetcontainsover670K+
multi-resolutionsatelliteimages(rangingfrom0.25mto10m)and22M+fieldinstancemasks.Imagesaregroupedbythenumberoffields
todemonstratethedataset’sdiversityandscalability,andachallengeofseparatingfieldsacrossvaryingresolutionsandgeographies.
sity, covering several European countries, including Aus- 10 fields to over 300 fields per image. This variability, il-
tria, France, Luxembourg, the Netherlands, Slovakia, lustratedinFigure3,highlightsitsabilitytorepresentboth
Slovenia, Spain, Sweden, and Ukraine. This broad geo- sparseanddenseagriculturalregions.
graphicscopeensuresthatmodelstrainedonFBIS-22Mcan
adapttovariedagriculturalpractices,landtypes,andenvi- The construction of FBIS-22M prioritized quality and
ronmentalconditions. Thedatasetfurtherdemonstratesdi- completeness. Official LPIS (Land Parcel Identification
versityinfielddensities,withimagescontainingfewerthan System) boundaries were utilized for most regions, while
high-resolutioncommercialsatelliteimagerywasmanually
5

annotated for regions where LPIS data was unavailable, eachclassatanIoUof0.5,whilemAP@0.5:0.95averages
such as Ukraine, ensuring full coverage. Additionally, the precisionacrossIoUthresholdsfrom0.5to0.95instepsof
dataset was manually cleaned, by removing errors in field 0.05.Thesemetricsofferacomprehensiveevaluationofour
boundaries and inconsistencies addressed to ensure accu- method’sperformanceinaccuratelydetectingandsegment-
| racy. |         |          |              |          |        |     | ingagriculturalfields. |     |     |     |     |     |     |
| ----- | ------- | -------- | ------------ | -------- | ------ | --- | ---------------------- | --- | --- | --- | --- | --- | --- |
| The   | dataset | is split | into 636,784 | training | images | and |                        |     |     |     |     |     |     |
36,125 test images, enabling effective model training and 4.2.ImplementationDetails
evaluation. As shown in Table 1, FBIS-22M significantly TheDelineateAnythingmodelistrainedwithabatchsize
surpasses existing field boundary datasets in both image of320(40perGPU),alearningrateof2e−5,and30epochs,
| countandinstancemasks. |     |     | Byclosingthiscriticalresource |     |     |     |           |          |      |      |          |           |       |
| ---------------------- | --- | --- | ----------------------------- | --- | --- | --- | --------- | -------- | ---- | ---- | -------- | --------- | ----- |
|                        |     |     |                               |     |     |     | using the | standard | YOLO | loss | function | [15, 36], | which |
gap, FBIS-22M provides a comprehensive foundation for includes components for bounding box regression, object-
| advancing | precision | agriculture |     | and automated | land | parcel |     |     |     |     |     |     |     |
| --------- | --------- | ----------- | --- | ------------- | ---- | ------ | --- | --- | --- | --- | --- | --- | --- |
ness,andclassification,alongwithtaskalignmentlearning.
identification, placing it on par with leading computer vi- Model is initialized with COCO pretrained weights before
siondatasets.
|     |     |     |     |     |     |     | fine-tuning      | on  | our dataset. | We use      | the AdamW |                | optimizer |
| --- | --- | --- | --- | --- | --- | --- | ---------------- | --- | ------------ | ----------- | --------- | -------------- | --------- |
|     |     |     |     |     |     |     | with exponential |     | learning     | rate decay. | For       | data augmenta- |           |
3.3.DelineateAnything
tion,weemploystandardtechniquessuchashorizontaland
WeproposeDelineateAnything(DelAny),aframeworkfor verticalflips,colorjittering,mosaic,mixup,andcopy-paste
accurate and efficient field boundary delineation from di- augmentation,consistentwithtypicalYOLOtrainingprac-
|                 |     |          |        |         |          |        | tices [15, | 36]. | Mosaic | augmentation | was | used for | the first |
| --------------- | --- | -------- | ------ | ------- | -------- | ------ | ---------- | ---- | ------ | ------------ | --- | -------- | --------- |
| verse satellite |     | imagery. | DelAny | focuses | on using | exist- |            |      |        |              |     |          |           |
ingstate-of-the-artinstancesegmentationtechniquesanda 20epochsandthendisabledforthefinal10epochs. Allex-
large-scaledatasettoachievestrongresults,ratherthanin- perimentsareconductedon8NVIDIAH100GPUs.Byde-
troducingnewarchitecturaldesigns. AtthecoreofDelAny fault,weevaluatemodelperformanceusingthefinalcheck-
is the YOLOv11 instance segmentation model, currently pointaftertrainingratherthanselectingthebest-performing
checkpoint.Toensureafaircomparison,othermodelscom-
| the state-of-the-art |     | in  | instance | segmentation. |     | YOLOv11 |     |     |     |     |     |     |     |
| -------------------- | --- | --- | -------- | ------------- | --- | ------- | --- | --- | --- | --- | --- | --- | --- |
provides exceptional accuracy and real-time performance, paredinthisworkaretrainedusingtheirofficiallyreleased
makingitidealforhandlingthelargevolumesofdatatypi- codebasesonourdataset(ortheAI4Boundaries[7]dataset
calinremotesensingapplications. where applicable), except for the zero-shot evaluation, as
The DelAny pipeline (Figure 1) processes satellite im- specifiedelsewhereinthepaper.
| ageryattheirnativeresolutions, |     |     |     | avoidingresizingartifacts |     |     |     |     |     |     |     |     |     |
| ------------------------------ | --- | --- | --- | ------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
4.3.MainResults
| andpreservingfine-grainedboundarydetails. |     |     |     |     | Duringtrain- |     |     |     |     |     |     |     |     |
| ----------------------------------------- | --- | --- | --- | --- | ------------ | --- | --- | --- | --- | --- | --- | --- | --- |
ing,themodelutilizesimagesfromavarietyofsources,in-
|     |     |     |     |     |     |     | We evaluate | the | performance | of  | our proposed |     | Delineate |
| --- | --- | --- | --- | --- | --- | --- | ----------- | --- | ----------- | --- | ------------ | --- | --------- |
cludingSentinel-2, Planet, Maxar, Pleiades, andorthopho- Anything(DelAny)modelanditssmallervariant(DelAny-
| tos, as | part of | the FBIS-22M |     | dataset. | This ensures | the |     |     |     |     |     |     |     |
| ------- | ------- | ------------ | --- | -------- | ------------ | --- | --- | --- | --- | --- | --- | --- | --- |
S)ontheFBIS-22Mtestset,comparingthemwithstate-of-
model’sabilitytogeneralizeacrossawiderangeofresolu- the-art methods, including MultiTLF [14], SAM [17], and
tionsandimagingconditions. Oncetrained,theresolution- SAM2[26]. TheresultsarepresentedinTable2.
agnosticdesignofDelAnyallowsittohandleimageryfrom
|     |     |     |     |     |     |     | Our | DelAny | model | achieves | a significant | improvement |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------ | ----- | -------- | ------------- | ----------- | --- |
any source, maintaining high performance without addi- inbothmAP@0.5andmAP@0.5:0.95metrics,withscores
tionalfine-tuning.
of0.720and0.477,respectively,surpassingSAM2,thepre-
Input images are processed by the pre-trained DelAny vious best-performing model, by 88.5% in mAP@0.5 and
model to generate instance masks, which are then trans- 103% in mAP@0.5:0.95. This establishes DelAny as the
| formed | into closed-field |     | boundaries | using | simple | post- |                                                 |     |     |     |     |     |        |
| ------ | ----------------- | --- | ---------- | ----- | ------ | ----- | ----------------------------------------------- | --- | --- | --- | --- | --- | ------ |
|        |                   |     |            |       |        |       | newstate-of-the-artforfieldboundarydelineation. |     |     |     |     |     | Impor- |
processingtechniqueslikecontourextraction. Thisstream- tantly,DelAnyachievesthisimprovementwhilealsobeing
linedapproachsimplifiesthepipelinewhileensuringpreci-
|     |     |     |     |     |     |     | 415 times | faster | in inference | than | SAM2, | highlighting | its |
| --- | --- | --- | --- | --- | --- | --- | --------- | ------ | ------------ | ---- | ----- | ------------ | --- |
sionindelineatingfieldboundaries. efficiency and suitability for real-time applications. The
|     |     |     |     |     |     |     | DelAny-S | variant, | despite | its smaller | size | and faster | infer- |
| --- | --- | --- | --- | --- | --- | --- | -------- | -------- | ------- | ----------- | ---- | ---------- | ------ |
4.Experiments
|     |     |     |     |     |     |     | ence speed,    | also | outperforms | SAM2            | by  | a significant | mar-     |
| --- | --- | --- | --- | --- | --- | --- | -------------- | ---- | ----------- | --------------- | --- | ------------- | -------- |
|     |     |     |     |     |     |     | gin, achieving |      | a 65.5%     | gain in mAP@0.5 |     | and a         | 63% gain |
4.1.Metrics
|     |     |     |     |     |     |     | inmAP@0.5:0.95. |     | Furthermore,DelAny-Sissignificantly |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --------------- | --- | ----------------------------------- | --- | --- | --- | --- |
We evaluate our method using standard instance segmen- moreefficient,achievinginferencespeeds617timesfaster
tation metrics based on the Microsoft COCO evaluation thanSAM2and1.49timesfasterthanDelAny.
protocol[21], reportingMeanAveragePrecision(mAP)at Figure 4 presents qualitative comparisons of Delineate
IoU thresholds of 0.5 (mAP@0.5) and from 0.5 to 0.95 AnythingwithMultiTLF[14],SAM[17],andSAM2[26].
(mAP@0.5:0.95). mAP@0.5 averages the precision for MultiTLF performs well in scenarios with large fields and
6

Figure4. QualitativeresultsontheFBIS-22Mtestset. DelineateAnythingiscomparedtoMultiTLF[14],SAM[17],andSAM2[26].
For a fair comparison, the MultiTLF model was retrained using our FBIS-22M dataset. Different samples are carefully selected and
presented,varyinginthesizeanddensityofthefields,tobetterillustratetheperformanceofeachmodelunderdiverseconditions.
sparse boundaries, but struggles in images with smaller or real-worldapplications.
densely packed fields, often merging or missing them due
4.4.Zero-ShotCross-RegionGeneralization
toitssemanticsegmentationapproach. SAMtendstoover-
segment, detecting irrelevant objects like water, grassland To evaluate the generalization capabilities of Delineate
andforests, leadingtoreducedprecision, especiallyinim- Anything,weconductzero-shotexperimentsongeographic
ages with non-agricultural areas. SAM2 slightly improves regionsnotincludedinthetrainingset. Specifically,wevi-
onSAMbutstillfacessimilarchallenges. sualize the model’s predictions on regions in Brazil, Cam-
Incontrast,DelineateAnythingoutperformsallmethods bodia, New Zealand, Rwanda, USA, Vietnam, and South
ineveryscenario,maintaininghighaccuracyinbothsparse Africa,whilethetrainingdatawasexclusivelysourcedfrom
and dense agricultural environments. Its instance segmen- Europe. Sincegroundtruthannotationsareunavailablefor
tationapproachenablesreliablefieldboundarydelineation, theseregions,wefocusonaqualitativeevaluation.Figure5
evenincomplexagriculturalsettings. Theseresultsdemon- presentsexamplesofthemodel’sperformanceintheseun-
strate model’s robustness and suitability for large-scale, seengeographiccontexts.
7

Figure 5. Qualitative results of zero-shot predictions. Delineate Anything is applied to geographic regions with different climates,
terrains,andagriculturalpractices,highlightingitsfieldboundarydelineationcapabilitiesoutsidethetrainingdata.
Method mAP@0.5 mAP@0.5:0.95 Latency(ms) Dataset #Images mAP@0.5 mAP@0.5:0.95
MultiTLF†[14]
|          |     | 0.257 | 0.110 |     | 55.8  | AI4Boundaries[7] |     | 45K  |     | 0.358 |     | 0.211 |
| -------- | --- | ----- | ----- | --- | ----- | ---------------- | --- | ---- | --- | ----- | --- | ----- |
| SAM[17]  |     | 0.339 | 0.197 |     | 13605 | FBIS-22M(subset) |     | 45K  |     | 0.597 |     | 0.335 |
| SAM2[26] |     | 0.382 | 0.235 |     | 10370 | FBIS-22M(subset) |     | 150K |     | 0.678 |     | 0.429 |
| DelAny-S |     | 0.632 | 0.383 |     | 16.8  | FBIS-22M         |     | 636K |     | 0.720 |     | 0.477 |
| DelAny   |     | 0.720 | 0.477 |     | 25.0  |                  |     |      |     |       |     |       |
Table3. Impactofdatasetsizeanddiversityonmodelperfor-
Table2. QuantitativecomparisonsontheFBIS-22Mtestset. mance.PerformancecomparisonoftheDelAnymodeltrainedon
WecompareourDelAnymodelanditssmallervariant(DelAny- theAI4BoundariesdatasetandsubsetsoftheFBIS-22Mdataset,
†:
S)againstothermethods. ModelsretrainedonourFBIS-22M highlightingtheeffectofdatasetscaleanddiversity.
datasetforfaircomparison.Latency(ms)representsthetotaltime
requiredtogeneratefieldboundaries.Bestresultsareinbold.
|     |     |     |     |     |     | achieves     | only  | 0.358 mAP@0.5 | and | 0.211        | mAP@0.5:0.95, |      |
| --- | --- | --- | --- | --- | --- | ------------ | ----- | ------------- | --- | ------------ | ------------- | ---- |
|     |     |     |     |     |     | highlighting | these | limitations.  |     | In contrast, | training      | on a |
The results highlight the model’s ability to adapt to di- 45,212-image subset of FBIS-22M improves performance
| verse terrains, | field | patterns, | and agricultural |     | practices, in- |          |         |           |               |     |     |         |
| --------------- | ----- | --------- | ---------------- | --- | -------------- | -------- | ------- | --------- | ------------- | --- | --- | ------- |
|                 |       |           |                  |     |                | to 0.597 | mAP@0.5 | and 0.335 | mAP@0.5:0.95. |     |     | Expand- |
cludingsmallholderfarms,largeindustrialfields,andvary-
ingto150,000imagesboostsitfurtherto0.678mAP@0.5
ing crop arrangements. This shows strong robustness and and 0.429 mAP@0.5:0.95. The full FBIS-22M dataset
| potential | for deployment |     | across different | agricultural | set- |            |         |         |       |         |     |           |
| --------- | -------------- | --- | ---------------- | ------------ | ---- | ---------- | ------- | ------- | ----- | ------- | --- | --------- |
|           |                |     |                  |              |      | yields the | highest | scores: | 0.720 | mAP@0.5 |     | and 0.477 |
tings. The model consistently identifies field boundaries mAP@0.5:0.95. A similar trend was observed with Mul-
even under challenging conditions, such as irregular field tiTLF [14] trained on AI4Boundaries, where performance
| shapes, varyingtextures, |     |     | anddiverselayouts. |     | Thesequali- |         |          |         |     |       |               |     |
| ------------------------ | --- | --- | ------------------ | --- | ----------- | ------- | -------- | ------- | --- | ----- | ------------- | --- |
|                          |     |     |                    |     |             | dropped | to 0.097 | mAP@0.5 | and | 0.040 | mAP@0.5:0.95. |     |
tative results strongly support DelAny’s zero-shot general- These results show that with the same number of images,
izationability,demonstratingitssuitabilityforscalablefield
|     |     |     |     |     |     | the diverse | FBIS-22M | dataset | performs |     | much | better than |
| --- | --- | --- | --- | --- | --- | ----------- | -------- | ------- | -------- | --- | ---- | ----------- |
boundarymappingacrossglobalagriculturallandscapes. AI4Boundaries, highlighting that having variety in resolu-
tionandsensorsisjustasimportantasthesizeofthedataset
4.5.AblationStudies
foraccuratefieldboundarydetection.
| To assess                                           | the impact | of dataset  | size | and diversity, | we con-     |              |           |     |          |           |              |     |
| --------------------------------------------------- | ---------- | ----------- | ---- | -------------- | ----------- | ------------ | --------- | --- | -------- | --------- | ------------ | --- |
| ductedablationstudiesbytrainingourDelineateAnything |            |             |      |                |             | 5.Conclusion |           |     |          |           |              |     |
| model on                                            | subsets    | of FBIS-22M | and  | compared       | its perfor- |              |           |     |          |           |              |     |
|                                                     |            |             |      |                |             | This work    | addresses | the | need for | automated | agricultural |     |
mancetoamodeltrainedontheAI4Boundariesdataset[7].
|     |     |     |     |     |     | field boundary |     | delineation | by reformulating |     | it  | as instance |
| --- | --- | --- | --- | --- | --- | -------------- | --- | ----------- | ---------------- | --- | --- | ----------- |
Table3presentstheresults.
|                   |     |          |         |          |           | segmentation |         | task and introducing |     | a large-scale, |        | multi- |
| ----------------- | --- | -------- | ------- | -------- | --------- | ------------ | ------- | -------------------- | --- | -------------- | ------ | ------ |
| The AI4Boundaries |     | training | dataset | consists | of 45,212 |              |         |                      |     |                |        |        |
|                   |     |          |         |          |           | resolution   | dataset | essential            | for | training       | models | robust |
images,primarilyfromSentinel-2imagery,butsuffersfrom
|     |     |     |     |     |     | to varying | image | sources | and resolutions. |     | This | dataset |
| --- | --- | --- | --- | --- | --- | ---------- | ----- | ------- | ---------------- | --- | ---- | ------- |
artifacts due to monthly compositing and lacks resolution bridgesthegapinsizeanddiversitycomparedtoothersin
and satellite diversity, limiting its robustness. Our exper- computervision. OurDelineateAnythingmodel,designed
iments demonstrate that model trained on AI4Boundaries to handle diverse resolutions, significantly outperforms
8

existing methods, achieving faster inference and strong Cao. Eva: Exploring the limits of masked visual repre-
zero-shot generalization. While further improvements sentationlearningatscale. In2023IEEE/CVFConference
in generalization across geographic regions are needed, onComputerVisionandPatternRecognition(CVPR),pages
| this work | advances | the | state-of-the-art |     | in  | automated |                        |     |     |     |     |     |     |
| --------- | -------- | --- | ---------------- | --- | --- | --------- | ---------------------- | --- | --- | --- | --- | --- | --- |
|           |          |     |                  |     |     |           | 19358–19369.IEEE,2023. |     |     | 3   |     |     |     |
field boundary delineation for agricultural applications, [11] JordanGraesserandNavinRamankutty. Detectionofcrop-
with potential for large-scale areas, such as country level. landfieldparcelsfromlandsatimagery. RemoteSensingof
|     |     |     |     |     |     |     | Environment,201:165–180,2017. |     |     |     | 1,2 |     |     |
| --- | --- | --- | --- | --- | --- | --- | ----------------------------- | --- | --- | --- | --- | --- | --- |
[12] KaimingHe,GeorgiaGkioxari,PiotrDollar,andRossGir-
References
|             |       |       |         |          |        |           | shick.                            | Maskr-cnn. | In2017IEEEInternationalConference |     |     |     |     |
| ----------- | ----- | ----- | ------- | -------- | ------ | --------- | --------------------------------- | ---------- | --------------------------------- | --- | --- | --- | --- |
|             |       |       |         |          |        |           | onComputerVision(ICCV).IEEE,2017. |            |                                   |     |     | 3   |     |
| [1] Han Lin | Aung, | Burak | Uzkent, | Marshall | Burke, | David Lo- |                                   |            |                                   |     |     |     |     |
[13] ZhongxinHuang,HaitaoJing,YuemingLiu,XiaomeiYang,
bell, and Stefano Ermon. Farm parcel delineation using Zhihua Wang, Xiaoliang Liu, Ku Gao, and Haofeng Luo.
spatio-temporalconvolutionalnetworks. In2020IEEE/CVF Segment anything model combined with multi-scale seg-
| Conference | on  | Computer | Vision | and | Pattern | Recognition |     |     |     |     |     |     |     |
| ---------- | --- | -------- | ------ | --- | ------- | ----------- | --- | --- | --- | --- | --- | --- | --- |
mentationforextractingcomplexcultivatedlandparcelsin
| Workshops(CVPRW),pages340–349.IEEE,2020. |     |     |     |     |     | 1,2,3, |                                     |     |     |     |                  |     |     |
| ---------------------------------------- | --- | --- | --- | --- | --- | ------ | ----------------------------------- | --- | --- | --- | ---------------- | --- | --- |
|                                          |     |     |     |     |     |        | high-resolutionremotesensingimages. |     |     |     | RemoteSensing,16 |     |     |
4
|     |     |     |     |     |     |     | (18),2024. | 3   |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ---------- | --- | --- | --- | --- | --- | --- |
[2] Gedas Bertasius, Jianbo Shi, and Lorenzo Torresani. [14] HannahKerner, SakethSundar, andMathanSatish. Multi-
| Deepedge: | A   | multi-scale | bifurcated |     | deep network | for top- |     |     |     |     |     |     |     |
| --------- | --- | ----------- | ---------- | --- | ------------ | -------- | --- | --- | --- | --- | --- | --- | --- |
regiontransferlearningforsegmentationofcropfieldbound-
| downcontourdetection. |     |     | In2015IEEEConferenceonCom- |     |     |     |                                          |     |     |     |     |               |     |
| --------------------- | --- | --- | -------------------------- | --- | --- | --- | ---------------------------------------- | --- | --- | --- | --- | ------------- | --- |
|                       |     |     |                            |     |     |     | ariesinsatelliteimageswithlimitedlabels. |     |     |     |     | arXivpreprint |     |
puterVisionandPatternRecognition(CVPR),pages4380–
|                 |     |     |     |     |     |     | arXiv:2404.00179,2024. |        |              | 3,6,7,8 |          |          |     |
| --------------- | --- | --- | --- | --- | --- | --- | ---------------------- | ------ | ------------ | ------- | -------- | -------- | --- |
| 4389.IEEE,2015. |     | 2   |     |     |     |     |                        |        |              |         |          |          |     |
|                 |     |     |     |     |     |     | [15] Rahima            | Khanam | and Muhammad |         | Hussain. | Yolov11: | An  |
[3] Keyan Chen, Chenyang Liu, Hao Chen, Haotian Zhang, overview of the key architectural enhancements. arXiv
| WenyuanLi,ZhengxiaZou,andZhenweiShi. |           |     |        |         |          | Rsprompter: |                                |     |     |     |     |     |     |
| ------------------------------------ | --------- | --- | ------ | ------- | -------- | ----------- | ------------------------------ | --- | --- | --- | --- | --- | --- |
|                                      |           |     |        |         |          |             | preprintarXiv:2410.17725,2024. |     |     |     | 3,6 |     |     |
| Learning                             | to prompt | for | remote | sensing | instance | segmenta-   |                                |     |     |     |     |     |     |
[16] AlexanderKirillov,EricMintun,NikhilaRavi,HanziMao,
| tionbasedonvisualfoundationmodel. |     |     |     |     | IEEETransactions |     |     |     |     |     |     |     |     |
| --------------------------------- | --- | --- | --- | --- | ---------------- | --- | --- | --- | --- | --- | --- | --- | --- |
ChloeRolland,LauraGustafson,TeteXiao,SpencerWhite-
onGeoscienceandRemoteSensing,2024. 3 head, Alexander C. Berg, Wan-Yen Lo, Piotr Dolla´r, and
[4] Ziyi Chen, Liai Deng, Yuhua Luo, Dilong Li, Jose´ Mar- Ross Girshick. Segment anything. In2023 IEEE/CVFIn-
catoJunior,WesleyNunesGonc¸alves,AbdulAwalMdNu-
|          |          |     |       |       |           |          | ternational | Conference |     | on Computer | Vision | (ICCV). | IEEE, |
| -------- | -------- | --- | ----- | ----- | --------- | -------- | ----------- | ---------- | --- | ----------- | ------ | ------- | ----- |
| runnabi, | Jonathan | Li, | Cheng | Wang, | and Deren | Li. Road |             |            |     |             |        |         |       |
|          |          |     |       |       |           |          | 2023.       | 2,3,4      |     |             |        |         |       |
International
extraction in remote sensing data: A survey. [17] AlexanderKirillov,EricMintun,NikhilaRavi,HanziMao,
JournalofAppliedEarthObservationandGeoinformation, ChloeRolland,LauraGustafson,TeteXiao,SpencerWhite-
| 112,2022.               | 2      |       |        |                             |        |          | head,AlexanderCBerg,Wan-YenLo,etal. |                             |        |      |               | Segmentany- |            |
| ----------------------- | ------ | ----- | ------ | --------------------------- | ------ | -------- | ----------------------------------- | --------------------------- | ------ | ---- | ------------- | ----------- | ---------- |
| [5] Zhe Chen,           | Yuchen | Duan, | Wenhai | Wang,                       | Junjun | He, Tong |                                     |                             |        |      |               |             |            |
|                         |        |       |        |                             |        |          | thing.                              | InICCV,pages4015–4026,2023. |        |      |               | 6,7,8       |            |
| Lu,JifengDai,andYuQiao. |        |       |        | Visiontransformeradapterfor |        |          |                                     |                             |        |      |               |             |            |
|                         |        |       |        |                             |        |          | [18] Alina                          | Kuznetsova,                 | Hassan | Rom, | Neil Alldrin, |             | Jasper Ui- |
densepredictions. arXivpreprintarXiv:2205.08534, 2022. jlings,IvanKrasin,JordiPont-Tuset,ShahabKamali,Stefan
| 3   |     |     |     |     |     |     | Popov,MatteoMalloci,AlexanderKolesnikov,TomDuerig, |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | -------------------------------------------------- | --- | --- | --- | --- | --- | --- |
[6] Sophie Crommelinck, Rohan Bennett, Markus Gerke, andVittorioFerrari. Theopenimagesdatasetv4. Interna-
| FrancescoNex,MichaelYang,andGeorgeVosselman. |     |     |     |     |     | Re- |     |     |     |     |     |     |     |
| -------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
tionalJournalofComputerVision,128(7):1956–1981,2020.
| view of | automatic | feature | extraction |     | from high-resolution |     | 2,4 |     |     |     |     |     |     |
| ------- | --------- | ------- | ---------- | --- | -------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
opticalsensordataforuav-basedcadastralmapping.Remote [19] Mykola Lavreniuk. Spidepth: Strengthened pose informa-
Sensing,8(8),2016. 2 tionforself-supervisedmonoculardepthestimation. arXiv
[7] Raphae¨ld’Andrimont,MartinClaverie,PieterKempeneers, preprintarXiv:2404.12501,2024. 2
DavideMuraro,MomchilYordanov,DevisPeressutti,Matej
|     |     |     |     |     |     |     | [20] Mykola | Lavreniuk, | Shariq | Farooq | Bhat, | Matthias | Mu¨ller, |
| --- | --- | --- | --- | --- | --- | --- | ----------- | ---------- | ------ | ------ | ----- | -------- | -------- |
Baticˇ, and Franc¸ois Waldner. Ai4boundaries: an open ai- and Peter Wonka. Evp: Enhanced visual perception us-
ready dataset to map field boundaries with sentinel-2 and inginversemulti-attentivefeaturerefinementandregularized
aerialphotography. EarthSystemScienceData,15(1):317– image-text alignment. arXiv preprint arXiv:2312.08548,
| 329,2023. | 1,2,3,4,6,8 |     |     |     |     |     | 2023. | 3   |     |     |     |     |     |
| --------- | ----------- | --- | --- | --- | --- | --- | ----- | --- | --- | --- | --- | --- | --- |
[8] Foivos I. Diakogiannis, Franc¸ois Waldner, Peter Caccetta, [21] Tsung-YiLin,MichaelMaire,SergeBelongie,JamesHays,
andChenWu.Resunet-a:Adeeplearningframeworkforse- PietroPerona,DevaRamanan,PiotrDolla´r,andC.Lawrence
manticsegmentationofremotelysenseddata.ISPRSJournal Zitnick. Microsoft COCO: Common Objects in Context,
ofPhotogrammetryandRemoteSensing,162:94–114,2020. pages 740–755. Springer International Publishing, Cham,
3
|     |     |     |     |     |     |     | 2014. | 2,4,6 |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ----- | ----- | --- | --- | --- | --- | --- |
[9] HakanErden, MuratAslan, andCemreBaharOzcanli. To [22] Yun Liu, Ming-Ming Cheng, Xiaowei Hu, Kai Wang, and
establishanewsubsidysystem.In2015FourthInternational XiangBai. Richerconvolutionalfeaturesforedgedetection.
ConferenceonAgro-Geoinformatics(Agro-geoinformatics), In2017IEEEConferenceonComputerVisionandPattern
pages57–60.IEEE,2015. 1 Recognition(CVPR),pages5872–5881.IEEE,2017. 2
[10] YuxinFang,WenWang,BinhuiXie,QuanSun,LedellWu, [23] KhairiyaMudrikMasoud,ClaudioPersello,andValentynA.
Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Tolpekin. Delineationofagriculturalfieldboundariesfrom
9

sentinel-2imagesusinganovelsuper-resolutioncontourde- [34] MatthiasP.WagnerandNataschaOppelt.Extractingagricul-
tectorbasedonfullyconvolutionalnetworks. RemoteSens- turalfieldsfromremotesensingimageryusinggraph-based
ing,12(1),2019. 2 growingcontours. RemoteSensing,12(7),2020. 1,2
[24] LucasPradoOsco,QiushengWu,EduardoLopesdeLemos, [35] Franc¸oisWaldner,FoivosI.Diakogiannis,KathrynBatche-
Wesley Nunes Gonc¸alves, Ana Paula Marques Ramos, lor, MichaelCiccotosto-Camp, ElizabethCooper-Williams,
JonathanLi,andJose´Marcato,Junior.Thesegmentanything Chris Herrmann, Gonzalo Mata, and Andrew Toovey. De-
model(sam)forremotesensingapplications: Fromzeroto tect, consolidate, delineate: Scalable mapping of field
RemoteSensing,13(11),
oneshot. InternationalJournalofAppliedEarthObserva- boundariesusingsatelliteimages.
| tionandGeoinformation,124,2023. |     |     |     |     | 3   |     | 2021. 1,2,3 |     |     |     |     |     |
| ------------------------------- | --- | --- | --- | --- | --- | --- | ----------- | --- | --- | --- | --- | --- |
[25] Claudio Persello, Jeroen Grift, Xinyan Fan, Claudia [36] Ao Wang, Hui Chen, Lihao Liu, Kai Chen, Zijia Lin, Jun-
|        |       |          |      |        |            |         | gong Han,     | and Guiguang |       | Ding.    | Yolov10: Real-time | end- |
| ------ | ----- | -------- | ---- | ------ | ---------- | ------- | ------------- | ------------ | ----- | -------- | ------------------ | ---- |
| Paris, | Ronny | Ha¨nsch, | Mila | Koeva, | and Andrew | Nelson. |               |              |       |          |                    |      |
|        |       |          |      |        |            |         | to-end object | detection.   | arXiv | preprint | arXiv:2405.14458,  |      |
Ai4smallfarms:Adatasetforcropfielddelineationinsouth-
| eastasiansmallholderfarms. |     |     |     | IEEEGeoscienceandRemote |     |     | 2024. 3,6 |     |     |     |     |     |
| -------------------------- | --- | --- | --- | ----------------------- | --- | --- | --------- | --- | --- | --- | --- | --- |
SensingLetters,20:1–5,2023. 1,2,3,4 [37] Libo Wang, Shenghui Fang, Xiaoliang Meng, and Rui Li.
|              |           |          |         |             |     |              | Buildingextractionwithvisiontransformer.        |     |     |     | IEEETransac- |     |
| ------------ | --------- | -------- | ------- | ----------- | --- | ------------ | ----------------------------------------------- | --- | --- | --- | ------------ | --- |
| [26] Nikhila | Ravi,     | Valentin | Gabeur, | Yuan-Ting   |     | Hu, Ronghang |                                                 |     |     |     |              |     |
|              |           |          |         |             |     |              | tionsonGeoscienceandRemoteSensing,60:1–11,2022. |     |     |     |              | 2   |
| Hu,          | Chaitanya | Ryali,   | Tengyu  | Ma, Haitham |     | Khedr, Roman |                                                 |     |     |     |              |     |
Ra¨dle, Chloe Rolland, Laura Gustafson, et al. Sam 2: [38] MoWang,JingWang,YunpengCui,JuanLiu,andLiChen.
Segment anything in images and videos. arXiv preprint Agricultural field boundary delineation with satellite im-
arXiv:2408.00714,2024. 3,6,7,8 agesegmentationforhigh-resolutioncropmapping: Acase
|     |     |     |     |     |     |     | studyofricepaddy. |     | Agronomy,12(10),2022. |     | 3   |     |
| --- | --- | --- | --- | --- | --- | --- | ----------------- | --- | --------------------- | --- | --- | --- |
[27] DanieleRegeCambrin,EleonoraPoeta,ElianaPastor,Tania
Cerquitelli, Elena Baralis, and Paolo Garza. Kan you see [39] Sherrie Wang, Franc¸ois Waldner, and David B. Lobell.
it? kansandsentinelforeffectiveandexplainablecropfield Unlocking large-scale crop field delineation in smallholder
segmentation. arXivpreprintarXiv:2408.07040,2024. 3 farming systems with transfer learning and weak supervi-
|     |     |     |     |     |     |     | sion. RemoteSensing,14(22),2022. |     |     |     | 1,2,3,4 |     |
| --- | --- | --- | --- | --- | --- | --- | -------------------------------- | --- | --- | --- | ------- | --- |
[28] SimiaoRen,FrancescoLuzi,SaadLahrichi,KalebKassaw,
Leslie M. Collins, Kyle Bradbury, and Jordan M. Malof. [40] BarryWatkinsandAdriaanvanNiekerk. Acomparisonof
Segment anything, from space? In Proceedings of the object-based image analysis approaches for field boundary
IEEE/CVFWinterConferenceonApplicationsofComputer delineationusingmulti-temporalsentinel-2imagery. Com-
|                                   |     |            |        |           |         |        | puters and | Electronics | in  | Agriculture, | 158:294–302, | 2019. |
| --------------------------------- | --- | ---------- | ------ | --------- | ------- | ------ | ---------- | ----------- | --- | ------------ | ------------ | ----- |
| Vision(WACV),pages8355–8365,2024. |     |            |        |           | 3       |        |            |             |     |              |              |       |
| [29] Christoph                    |     | Schuhmann, | Romain | Beaumont, | Richard | Vencu, | 1,2        |             |     |              |              |       |
Cade Gordon, Ross Wightman, Mehdi Cherti, Theo [41] SainingXieandZhuowenTu. Holistically-nestededgede-
|          |         |              |                |          |            |           | tection.                     | In 2015 IEEE | International |     | Conference | on Com- |
| -------- | ------- | ------------ | -------------- | -------- | ---------- | --------- | ---------------------------- | ------------ | ------------- | --- | ---------- | ------- |
| Coombes, |         | Aarush       | Katta, Clayton | Mullis,  | Mitchell   | Worts-    |                              |              |               |     |            |         |
|          |         |              |                |          |            |           | puterVision(ICCV).IEEE,2015. |              |               |     | 2          |         |
| man,     | Patrick | Schramowski, |                | Srivatsa | Kundurthy, | Katherine |                              |              |               |     |            |         |
Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Je- [42] L.YanandD.P.Roy. Conterminousunitedstatescropfield
niaJitsev. Laion-5b: Anopenlarge-scaledatasetfortrain- sizequantificationfrommulti-temporallandsatdata.Remote
ing next generation image-text models. arXiv preprint SensingofEnvironment,172:67–86,2016. 1,2
arXiv:2210.08402,2022. [43] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela
2,4
[30] Rodrigo Fill Rangel Seedz, V´ıtor Nascimento Lourenc¸o Barriuso, and Antonio Torralba. Scene parsing through
Gaivota, Lucas Volochen Oldoni Seedz, Ana Flavia Car- ade20kdataset. In2017IEEEConferenceonComputerVi-
rara Bonamigo Seedz, Wallas Santos Gaivota, Bruno sionandPatternRecognition(CVPR).IEEE,2017. 2
|       |          |        |            |       |         |          | [44] ZhuofanZong,GuangluSong,andYuLiu. |     |     |     | Detrswithcol- |     |
| ----- | -------- | ------ | ---------- | ----- | ------- | -------- | -------------------------------------- | --- | --- | --- | ------------- | --- |
| Silva | Oliveira | Seedz, | and Mateus | Neves | Barreto | Seedz. A |                                        |     |     |     |               |     |
unifiedframeworkforcroplandfieldboundarydetectionand laborativehybridassignmentstraining. In2023IEEE/CVF
segmentation.In2024IEEE/CVFWinterConferenceonAp- InternationalConferenceonComputerVision(ICCV),pages
plicationsofComputerVisionWorkshops(WACVW),pages 6725–6735.IEEE,2023. 3
| 636–644.IEEE,2024. |      |                                     | 3          |     |             |          |     |     |     |     |     |     |
| ------------------ | ---- | ----------------------------------- | ---------- | --- | ----------- | -------- | --- | --- | --- | --- | --- | --- |
| [31] Jialin        | Sun, | Shuai Yan,                          | Xiaochuang |     | Yao, Bingbo | Gao, and |     |     |     |     |     |     |
| JianyuYang.        |      | Asegmentanythingmodelbasedweaklysu- |            |     |             |          |     |     |     |     |     |     |
pervisedlearningmethodforcropmappingusingsentinel-2
| timeseriesimages.                         |            |               | InternationalJournalofAppliedEarth |               |             |               |     |     |     |     |     |     |
| ----------------------------------------- | ---------- | ------------- | ---------------------------------- | ------------- | ----------- | ------------- | --- | --- | --- | --- | --- | --- |
| ObservationandGeoinformation,133,2024.    |            |               |                                    |               |             | 3             |     |     |     |     |     |     |
| [32] Pratyush                             |            | Tripathy,     | Kathy                              | Baylis,       | Kyle Wu,    | Jyles Wat-    |     |     |     |     |     |     |
| son,                                      | and        | Ruizhe Jiang. |                                    | Investigating | the         | segment any-  |     |     |     |     |     |     |
| thing                                     | foundation | model         | for                                | mapping       | smallholder | agricul-      |     |     |     |     |     |     |
| turefieldboundarieswithouttraininglabels. |            |               |                                    |               |             | arXivpreprint |     |     |     |     |     |     |
| arXiv:2407.01846,2024.                    |            |               | 3                                  |               |             |               |     |     |     |     |     |     |
[33] MatthiasP.WagnerandNataschaOppelt.Deeplearningand
adaptivegraph-basedgrowingcontoursforagriculturalfield
| extraction. |     | RemoteSensing,12(12),2020. |     |     | 2   |     |     |     |     |     |     |     |
| ----------- | --- | -------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
10