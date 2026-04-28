PRUE: A Practical Recipe for Field Boundary Segmentation at Scale
GedeonMuhawenayo1∗#,CalebRobinson2∗,SubashKhanal3∗,ZhanpeiFang4∗,
IsaacCorley5,AlexanderWollam3,TianyiGao3,LeonardStrnad5,RyanAvery5,
LyndonEstes6,AnaM.Tárano1,NathanJacobs3,HannahKerner1
1ArizonaStateUniversity 2MicrosoftAIforGood 3WashingtonUniversityinSt. Louis
|     |     |     | 4OregonStateUniversity |     |     | 5Wherobots |     | 6ClarkUniversity |     |     |     |
| --- | --- | --- | ---------------------- | --- | --- | ---------- | --- | ---------------- | --- | --- | --- |
6202 raM 82  ]VC.sc[  1v10172.3062:viXra
Figure1.ExampleresultsfromthePRUEmodelinJapan.PRUEspecifiesapracticalrecipeforcountry-scalefieldboundarysegmentation
thatisrobusttoreflectancevariationandnoiseinSentinel-2L2Aimagery,improvingstandardmetricsandlarge-scalemapquality.
|             |      |          | Abstract   |     |           |     | 1.Introduction |     |     |     |     |
| ----------- | ---- | -------- | ---------- | --- | --------- | --- | -------------- | --- | --- | --- | --- |
| Large-scale | maps | of field | boundaries | are | essential | for |                |     |     |     |     |
Agriculturalfieldboundarymaps(digitizedpolygonsthat
| agricultural | monitoring |     | tasks. | Existing deep | learning | ap- |                   |      |        |                  |              |
| ------------ | ---------- | --- | ------ | ------------- | -------- | --- | ----------------- | ---- | ------ | ---------------- | ------------ |
|              |            |     |        |               |          |     | define individual | farm | plots) | are foundational | for agricul- |
proachesforsatellite-basedfieldmappingaresensitiveto
turalmonitoringanddecision-making,enablingdownstream
illumination,spatialscale,andchangesingeographicloca-
|     |     |     |     |     |     |     | applications | such as crop | type | mapping, | yield estimation, |
| --- | --- | --- | --- | --- | --- | --- | ------------ | ------------ | ---- | -------- | ----------------- |
tion. Weconductthefirstsystematicevaluationofsegmen-
pestanddiseasesurveillance,andtrackingofconservation
tationandgeospatialfoundationmodels(GFMs)forglobal
|                |     |             |       |            |        |       | andclimateprograms[40]. |     | Fieldboundarymapsenhance |     |     |
| -------------- | --- | ----------- | ----- | ---------- | ------ | ----- | ----------------------- | --- | ------------------------ | --- | --- |
| field boundary |     | delineation | using | the Fields | of The | World |                         |     |                          |     |     |
nationalagriculturalstatistics,providingspatiallyconsistent
| (FTW) benchmark. |     | We  | evaluate | 18 models | under | unified |     |     |     |     |     |
| ---------------- | --- | --- | -------- | --------- | ----- | ------- | --- | --- | --- | --- | --- |
unitsforanalysisacrossregionsandseasons[42].
experimentalsettings,showingthataU-Netsemanticseg-
mentationmodeloutperformsinstance-basedandGFMal- Accurate, up-to-date field boundary maps would trans-
formagricultureandfoodsecurityapplications,butarein-
ternativesonasuiteofperformanceanddeploymentmetrics.
We propose a new segmentation approach that combines feasiblewithexistingmethods. Manuallydelineatingfield
boundaries–typicallythroughimageinterpretation–isslow,
aU-Netbackbone,compositelossfunctions,andtargeted
labor-intensive,andmustberegularlyrepeatedasboundaries
dataaugmentationstoenhanceperformanceandrobustness
under real-world conditions. Our model achieves a 76% shiftduetolanduseandmanagementchanges[64],making
thisapproachimpracticalforlarge-scalemonitoring[19].Al-
| IoU and | 47% object-F1 |     | on FTW, | an increase | of 6% | and |     |     |     |     |     |
| ------- | ------------- | --- | ------- | ----------- | ----- | --- | --- | --- | --- | --- | --- |
9% over the previous baseline. Our approach provides a ternatively,machinelearningandcomputervisionmethods
forsatelliteimageryenablefast,automated,andrepeatable
practicalframeworkforreliable,scalable,andreproducible
extractionoffieldboundariesacrosslargeregionsandover
| field boundary |     | delineation | across | model | design, training, |     |     |     |     |     |     |
| -------------- | --- | ----------- | ------ | ----- | ----------------- | --- | --- | --- | --- | --- | --- |
and inference. We release all models and model-derived time [13, 34]. Satellite imagery offers wide spatial cover-
fieldboundarydatasetsforfivecountries.
#CorrespondingAuthor:gmuhawen@asu.edu
*EqualContribution
1

• Performantacrossdiverseagriculturalcontexts(e.g.,very
smalltoverylargefields).
|     |     |     |     | To develop | a model that | meets | these requirements, |     | we  |
| --- | --- | --- | --- | ---------- | ------------ | ----- | ------------------- | --- | --- |
performedasystematicevaluationofdiversesegmentation
|     |     |     |     | and geospatial            | foundation | model                      | (GFM) architectures |     | for |
| --- | --- | --- | --- | ------------------------- | ---------- | -------------------------- | ------------------- | --- | --- |
|     |     |     |     | fieldboundarydelineation. |            | Weconductedexperimentsthat |                     |     |     |
comparedanarrayofGFMsandsemanticandinstanceseg-
mentationarchitectures,aswellasabroadsweepofU-Net
modelsettingsusingbothestablishedperformancemetrics
Figure2.ExamplevisualizationofpredictionsoverIllinois,USA andasuiteofnewmeasuresdesignedtoassesstherobustness
(top,MGRStile16TDL)andMatoGrosso,Brazil(bottom,MGRS of field segmentation under real-world conditions. These
analysesresultinthreekeycontributions:
tile20LRQ)withtheFTWBaseline,TerramindandPRUE.The
FTWBaselineandTerramindmodelsshowstrongsensitivityto 1. A new state-of-the-art model for field boundary seg-
scene characteristics, producing discontinuous and noisy field mentation(PRUE),whichoutperforms18othermodels
boundaries.ThePRUEmodelachievesstableboundariesacross
againsttheFTWbenchmark,andhashighzero-shotper-
regionsandimagingconditions.
|     |     |     |     | formance. | Ourmodelcodeandweightsareavailableat |     |     |     |     |
| --- | --- | --- | --- | --------- | ------------------------------------ | --- | --- | --- | --- |
https://github.com/fieldsoftheworld/ftw-prue.
| age, frequent | revisit intervals, | and cost-effective | (or free) |     |     |     |     |     |     |
| ------------- | ------------------ | ------------------ | --------- | --- | --- | --- | --- | --- | --- |
2. Anewsetofmetricsforevaluatingtherobustnessoffield
access to decades of historical observations, making it an boundarysegmentationmodelstoreal-worlddistribution
| ideal data | source for scalable | agricultural | field boundary |     |     |     |     |     |     |
| ---------- | ------------------- | ------------ | -------------- | --- | --- | --- | --- | --- | --- |
shifts(e.g.,changesinbrightnessortranslation).
| mapping[13,64]. | Theseadvantageshavemotivatedthecre- |     |     |                   |            |     |                |     |      |
| --------------- | ----------------------------------- | --- | --- | ----------------- | ---------- | --- | -------------- | --- | ---- |
|                 |                                     |     |     | 3. Country-scale, | multi-year |     | field boundary |     | maps |
ationofseveralopen,machinelearning-readyfieldboundary for Japan, Mexico, Rwanda, South Africa, and
delineationdatasetstoaccelerateprogressandbenchmarking
|     |     |     |     | Switzerland, | which | are publicly | available | online | at  |
| --- | --- | --- | --- | ------------ | ----- | ------------ | --------- | ------ | --- |
[13,21,26,34,45]. https://source.coop/wherobots/fields-of-the-world.
However,reliablydelineatingfieldboundariesinsatellite
Thesedemonstratethereal-worldbenefitsofPRUEover
imageryremainsadifficultcomputervisionproblem.Unlike previousmodels: mapspredictedusingPRUEaremore
typicalinstancesegmentationtasks,fieldboundariesareof- accurateandrevealimportantlandscapechange.
tennarrowandpoorlydefined,varyingwithcropphenology, Thesecontributionsadvancereliable,scalable,andtrans-
managementpractices,andimagingconditions[34]. Field ferable models for delineating field boundaries, enabling
edgesmayappeardiscontinuousduetomixedpixels,cloud
|     |     |     |     | the creation | of globally | consistent, | automatically | updated |     |
| --- | --- | --- | --- | ------------ | ----------- | ----------- | ------------- | ------- | --- |
shadows,orlowcontrast,especiallyinsmallholdersystems datasetsthatcanstrengthenequitableandsustainableagri-
wherefieldsareheterogeneousandirregularlyshaped.These culturaldecision-making[41].
factorsmakefielddelineationachallengingandimperfectly
supervisedproblem,sinceevenexpertannotatorsmaydis- 2.BackgroundandRelatedWork
agreeontrueboundarylocations[21].
Benchmarks such as Fields of The World (FTW) [34], EarlyapproachestodelineatefieldboundariesfromEarth
|     |     |     |     | observations | (EO) relied | on rule-based | image | processing, |     |
| --- | --- | --- | --- | ------------ | ----------- | ------------- | ----- | ----------- | --- |
PASTIS[26],andAI4Boundaries[13]helpadvanceresearch
suchasedgedetectiontechniquesandregion-basedsegmen-
progressonfieldboundarysegmentation,butdonotcapture
errorsthatarisewhenmodelsaredeployedforlarge-scale tation[51,69]. Althoughsimpleandcomputationallyeffi-
cient,theseapproachesareoftenineffective,astheirreliance
| map-making. | Applying | the best FTW | model from [34] |     |     |     |     |     |     |
| ----------- | -------- | ------------ | --------------- | --- | --- | --- | --- | --- | --- |
to map large, out-of-domain regions shows it is sensitive onlow-levelfeaturesmakesthemsensitivetoillumination,
to changes in brightness, temporal ordering, pixel resolu- noise,andimageheterogeneity[66]. Theselimitationsmoti-
vatedashifttowardsdata-drivenmethodsthatcapturespatial
tion/scale,andreceptivefield(Figure2showsexampleswith
bothFTWBaselineandtheTerramindmodels),leadingto contextandsemanticstructure[44,66].
|     |     |     |     | The | success of deep | learning | for image | segmentation |     |
| --- | --- | --- | --- | --- | --------------- | -------- | --------- | ------------ | --- |
patchtilingartifactsandlow-qualityfieldboundarymaps.
Severalpriorstudieshavenotedsimilarrobustnessfailures has made it the dominant approach for agricultural field
ingeospatialdeeplearningpipelines[11,12,49]. delineation [44, 63], which is fundamentally an instance
Toovercometheseobstaclestocreatingaccurate,readily- segmentation task focused on identifying individual field
|                |                |          |                  | polygons. | The unique characteristics |     | of satellite | imagery |     |
| -------------- | -------------- | -------- | ---------------- | --------- | -------------------------- | --- | ------------ | ------- | --- |
| updated global | field boundary | datasets | with AI, we need |           |                            |     |              |         |     |
haveledtodiversemethodologicalapproaches,whichwe
robustandscalablemodelsthatare:
organizebytheirsegmentationparadigm.
• Invarianttochangesinbrightness,spatialscale,seasonal
differences,andtranslationsintheinputwindow; Semantic segmentation with post-processing. Many
• Computationallyefficient(lowcost)atinferencetime; pipelinesadoptasemanticsegmentationframework,classi-
2

fyingeachpixelintotwoorthreeclasses(e.g.,fieldinterior, atedGFMsforfieldboundarysegmentationacrossFTW’s
boundary,background)usingmodelssuchasrandomforests diverseregions.
[16, 20], CNNs [44], and U-Nets [34, 65]. The resulting
rastermasksarethenpost-processedwiththresholding,con- 3.Methods
nectedcomponentanalysis,and/orwatershedsegmentation
torecoverfieldinstances[45,64,65]. Recentrefinementsin- Toaddressthechallengestorobust,large-scaleinference,we
corporatemultitaskobjectives,distancemaps,oredge-aware systematicallyevaluatedarangeofmodelsagainsttheestab-
lossesthatlearncontourfeatures, betterseparateadjacent lishedFTWbenchmarkdataset,introducingaugmentations
fields,andprovidemoreaccuratepolygonization[64].These andmetricstoimproveandassessdeploymentreliability.
developmentshaveestablishedU-Netanditsvariantsasthe
3.1.Dataset
mostcommonfielddelineationframework[34,45,64,65].
Thispredominanceofsemanticsegmentationstemsfrom TheFieldsofTheWorld(FTW)dataset[34]isaglobally
thechallengesofapplyingstandardinstancesegmentation distributedbenchmarkforagriculturalfieldboundarydelin-
methodstocropfields,whichareoftenirregularlyshaped eation,withover1.5milliongeo-referenced,manuallyvali-
anddenselypacked[40,65,70],andthuslacktheclearstruc- datedfieldpolygonsfrom24countriesacrossfourcontinents.
turethatobjectdetectionmethodsexploitthroughbounding- Eachpolygonispairedwithbi-temporalRGB-NIR(RGBN)
boxregressionandnon-maximumsuppression[53,74].Con- Sentinel-2imageryfromtheplantingandharvestseasons.
versely, semantic outputs may be fragmented, and post- We used FTW’s predefined datataset splits, which reduce
processing introduces hyperparameters (e.g. connectivity spatialautocorrelationandenablerobustcross-regionaleval-
thresholds)thatcanaffectinstancequality. uation. FTWprovidesatraining,validation,andtestsplit
foreachofthe24countriesinthedataset. Kerneretal.[33]
Instanceandpanopticsegmentation. Asmallerbodyof
recommendsbenchmarkingmodelsusingtheaverageofthe
workappliesinstanceorpanopticsegmentationarchitectures
individualtestaccuraciesforeachcountry.
directly, eliminatingpost-processingsteps. MaskR-CNN
has been adapted for field delineation [28, 39], but its re-
3.2.Modelarchitecturesearch
lianceonboundingboxesandNMSmaybesuboptimalfor
irregulargeometries.Recently,DelineateAnything[37]fine- Weapproachedmodeldevelopmentasa“bake-off,”system-
tunedYOLOv11-segforresolution-agnosticfieldboundary aticallyevaluatinganextensivesetofsemanticsegmentation,
prediction across multi-sensor European imagery. Field- instance segmentation, and GFM model configurations to
SegusedSegmentAnything(SAM)todelineateSentinel-2 findthemostpracticalandrobustrecipeforfieldboundary
imageryin8studyareasacross6continents[24].Thesestud- segmentationatscale. Wenamedthewinnerofthebake-off
iesshowedthatmoderninstancesegmentationarchitectures PRUE:APracticalRecipeforFieldBoundarySegmentation
canperformwellacrossdiverselandscapes. ThePASTIS atScale. Foreachmodelconfiguration, weusedtheorig-
dataset[26]benchmarkedpanopticmethodsonFrenchfield inal training settings, including reported hyperparameters
parcelsfromSentinel-2timeseries,whilePanopticFPNhas andpreprocessing, andsweptlearningratestoensurefair
beenappliedtomaplandcoverinaerialimagery[15]. andreproduciblecomparisonacrossmodelfamilies. This
approachallowedustoidentifythestrongestbaselinemodel
Universalsegmentation. Task-specificsegmentationmod-
forsubsequentdevelopment.
elslackflexibilitytogeneralizeacrosstasks. Universalseg-
mentationmodelssuchasMask2Former[8]andOneFormer Semantic segmentation baselines. We used the highest-
[31]arecapableofsemantic,instance,andpanopticsegmen- performingmodelreportedbyKerneretal.[34]ontheFTW
tation,butareunderexploredforsatelliteimagery. benchmark,aU-NetwithEfficientNet-B3backbone,asthe
“FTWbaseline”. WealsoevaluatedDECODE[64],which
GeospatialFoundationModels(GFMs). GFMssuchas
usesamultitaskFracTAL-ResUNettojointlypredictfield
SatMAE[10],DeCUR[67],Satlas[4],CROMA[25],Soft-
extent,boundaries,anddistancemaps. AsinKerneretal.
Con[68],DOFA-v1[73],AnySat[1],Galileo[62],Prithvi
[33],wemaskedpixelswithunknownlabelsduringtraining
2.0[58],Clay[9],TerraFM[14],TerraMind[32],andAl-
forpresence-onlyexamplesinFTW.Wepost-processedall
phaEarthFoundations[6]arepretrainedonglobal-scaleEO
outputsusingconnectedcomponentstoextractindividual
archives,usingcontrastiveormaskedmodelingobjectives.
fieldinstancesforobject-levelevaluation(see§3.4).
Their pretrained encoders provide general-purpose repre-
sentationsthatcaptureland-coversemantics,phenological Instanceandpanopticsegmentationbaselines. Weevalu-
patterns,andsurface-texturevariationsacrossgeographies. atedDelineateAnything[37],SAM[35],andMask2Former
GFM embeddings can be leveraged to enhance semantic (M2F)[8]. SinceDelineateAnythingispretrainedforfield
segmentationperformance[62],particularlyunderdomain boundarysegmentationandintendedtobeusedinazero-
shiftorlimitedlabeleddata. Previousworkhasnotevalu- shotsetting,weevaluateditanditssmallervariantDel-Any
3

SdirectlyonFTWusingtheRGBchannelsfromtheplant- Class weights. We further examined the impact of class
ingseasonimage. WeevaluatedSAMinbothzero-shotand reweightingbyvaryingtheboundaryclassimportancefactor
fine-tuned settings. SAM and M2F (panoptic task) were ωinstepsof0.05withintherange[0.60,0.85]. Thenormal-
fine-tuned on the same 8-channel bitemporal input as the izedclassweightsforthethreeoutputclasses(background,
U-Net baseline. Due to differences in how instance seg- interior,andboundary)weredefinedas[0.05,0.95−ω,ω],
mentation models handle training objectives compared to respectively. Thissweepallowedustoassessthesensitivity
semanticmodels,wedidnotmaskpresence-onlyexamples of model training to the relative emphasis placed on thin
intraining. SeesupplementAforadditionalresults. boundaryregionsversusfieldinteriors.
Geospatial foundation models (GFMs). We computed Learning rate. We swept learning rates logarithmically
| embeddingsfrompretrainedGFMencoders: |     |     |     | Galileo[62], |     |             |        |              |     |          |         |        |     |
| ------------------------------------ | --- | --- | --- | ------------ | --- | ----------- | ------ | ------------ | --- | -------- | ------- | ------ | --- |
|                                      |     |     |     |              |     | to identify | stable | optimization |     | regimes, | testing | values | in  |
CROMA[25],SoftCon[68],Prithvi2.0[58],DOFA-v1[73], {10−4,3×10−4,3×10−3,10−2,3×10−2}(ordersofmag-
DeCUR[67],Satlas[4],Clay[9],DINOv3[57],TerraFM
nitudecommonforAdamoptimizersinsegmentationtasks).
[14],andTerraMind[32]. Weobtainedtokenfeaturesfrom Wetrainedeachunderidenticalconditionstoevaluatecon-
pretrained, frozen GFMs for each temporal window. We vergencestabilityandsensitivitytostepsize.
fusedthetwowindowsbyconcatenatingtokensalongthe
|                    |            |           |        |         |     | Dataaugmentations. |     | Toaddresstheobservedsensitivity |     |     |     |     |     |
| ------------------ | ---------- | --------- | ------ | ------- | --- | ------------------ | --- | ------------------------------- | --- | --- | --- | --- | --- |
| feature dimension, | and passed | the fused | tokens | through | a   |                    |     |                                 |     |     |     |     |     |
3-layerMLP,allowingthemodeltointegrateinformation ofmodelstovariationsinbrightness,spatialscale,andtiling
boundaries,weperformedasystematicsweepofdataaug-
| from both | frames at the | token level | before | decoding. | To  |     |     |     |     |     |     |     |     |
| --------- | ------------- | ----------- | ------ | --------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
mentationsdesignedtoimproverobustnessalongfourkey
overcomepoorsegmentationperformanceresultingfroma
1×1convolutionfollowedbybilinearupsampling(seeSup- dimensions: inputorderinvariance,brightnessrobustness,
|     |     |     |     |     |     | scalerobustness,andspatialconsistency. |     |     |     |     | Weappliedeach |     |     |
| --- | --- | --- | --- | --- | --- | -------------------------------------- | --- | --- | --- | --- | ------------- | --- | --- |
plementalB),weadoptedadecodercomposedofa3×3con-
volutionalprojectionlayer,tworesidualrefinementblocks, augmentationindependentlyorincombinationtoevaluateits
effectontheserobustnessproperties,whichwerequantified
andamulti-scaleconvolutionalmodulethatexpandsspatial
usingthedeployment-orientedmetricsfrom§3.5.
context,followedbypixel-shuffleupsampling[55]topro-
| ducedensesegmentationmasks. |     | WeusedthisMLP-based |     |     |     |     |     |     |     |     |     |     |     |
| --------------------------- | --- | ------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
fusioncombinedwiththeenhancedconvolutionaldecoder 3.4.Evaluationmetricsformodelcomparison
forallGFMexperimentswithresultsreportedinTable1.
Beyondpixel-levelmetricsofIoUandF1-score,wereport
3.3.Modeldesignspaceexploration object-levelprecisionandrecallfrompolygonizedpredic-
|     |     |     |     |     |     | tions. Weaveragedeachmetricovertheindividualcountry |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
Buildingonthebest-performing,mostparameter-efficient
|          |                   |             |     |          |     | test sets | in FTW. | We excluded |     | “presence-only” |     | countries |     |
| -------- | ----------------- | ----------- | --- | -------- | --- | --------- | ------- | ----------- | --- | --------------- | --- | --------- | --- |
| baseline | with high per-km2 | throughput, | we  | explored | the |           |         |             |     |                 |     |           |     |
fromourevaluation,sinceonlyrecallcanbecomputedfor
modeldesignspacetoidentifycomponentscriticalforaccu-
|                                                  |     |     |     |     |        | thosecountries. |           | Tomeasurecomputationalefficiency, |     |            |     |        | we  |
| ------------------------------------------------ | --- | --- | --- | --- | ------ | --------------- | --------- | --------------------------------- | --- | ---------- | --- | ------ | --- |
| rate,scalable,androbustfieldboundarydelineation. |     |     |     |     | Rather |                 |           |                                   |     |            |     |        |     |
|                                                  |     |     |     |     |        | report the      | inference | throughput                        |     | (in km2/s) | and | number | of  |
thanexhaustivelyre-tuningeveryarchitecture,weusedthis
parametersforeachmodel.
| baseline | as a controlled | reference point, | ensuring |     | that ob- |     |     |     |     |     |     |     |     |
| -------- | --------------- | ---------------- | -------- | --- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
Wecomputedobjectprecision/recallata0.5confidence
servedtrendsreflectgenuinedesigneffectsratherthandiffer-
threshold,whichindicatesexpectedperformanceunderde-
| encesincapacityoroptimization. |     | Wesystematicallyvaried |     |     |     |                 |     |           |             |     |                      |     |     |
| ------------------------------ | --- | ---------------------- | --- | --- | --- | --------------- | --- | --------- | ----------- | --- | -------------------- | --- | --- |
|                                |     |                        |     |     |     | fault inference |     | settings. | To evaluate |     | the precision-recall |     |     |
architectural,data,andoptimizationfactorstoisolatetheir
individualimpactsonbenchmarkaccuracyandreal-world tradeoffofchangingthisthreshold,wealsocomputedCOCO
|     |     |     |     |     |     | AP andAP |     | . AP | integratesprecisionacrossall |     |     |     |     |
| --- | --- | --- | --- | --- | --- | -------- | --- | ---- | ---------------------------- | --- | --- | --- | --- |
robustness,witheachsweepvaryingasinglefactoratatime. 0.5 0.5:0.95 0.5
|     |     |     |     |     |     | confidencethresholdsatIoU=0.5,andAP |     |     |     |     |     | averages |     |
| --- | --- | --- | --- | --- | --- | ----------------------------------- | --- | --- | --- | --- | --- | -------- | --- |
0.5:0.95
| Architecturesandbackbones. |     | Wecomparedmultipleen- |     |     |     |     |     |     |     |     |     |     |     |
| -------------------------- | --- | --------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
thisacrossmultipleIoUthresholds({0.5,0.55,...,0.95})to
coder–decoder architectures, including FCN [38], UPer- evaluatebothconfidencecalibrationandlocalizationquality.
Net[71],FCSiam[7],andU-Net[50]variants.Weevaluated
|     |     |     |     |     |     | For semantic |     | segmentation |     | models, | which | do  | not pro- |
| --- | --- | --- | --- | --- | --- | ------------ | --- | ------------ | --- | ------- | ----- | --- | -------- |
allU-NetvariantswithEfficientNetbackbones(B3-B7)[59]
duceper-instanceconfidencescores,weusedthemeansoft-
andMixVisionTransformers(B2-B5)[72]toevaluatethe
|     |     |     |     |     |     | max probability |     | across all | pixels | in each | polygon |     | created |
| --- | --- | --- | --- | --- | --- | --------------- | --- | ---------- | ------ | ------- | ------- | --- | ------- |
impactofincreasingcapacityrelativetotheFTWbaseline
byargmaxingtheprobabilitymapasaproxyforinstance
(whichusedanEfficientNetB3backbone).
|     |     |     |     |     |     | confidence. | Instance | segmentation |     | models | such | as  | Delin- |
| --- | --- | --- | --- | --- | --- | ----------- | -------- | ------------ | --- | ------ | ---- | --- | ------ |
Lossfunctions. Weevaluatedarangeoflossescommonly eateAnythingproduceconfidencescoreswitheachobject
usedforfieldboundaryandclassimbalancesegmentation detection. SAMhasaproxyscoreofpredictedIoU,whichis
problems,includingcross-entropy(CE),Dice,log-coshDice, themodel’sownpredictionofmaskquality. Similarly,mod-
focal,Tversky,Jaccard,andFractalTanimoto(FTNMT)loss elswithapanopticinferencehead,suchasMask2Former,
functions,aswellastheirweightedvariants [2,17,30,52]. outputascoreforeachpredicted‘thing’and‘stuff’segment.
4

3.5.Deployment-orientedmodelmetrics
Ingeospatialmachinelearning(GeoML),modelsaretyp-
| ically | trained | on patch-based | datasets |     | and evaluated | with |     |     |     |     |     |     |     |     |
| ------ | ------- | -------------- | -------- | --- | ------------- | ---- | --- | --- | --- | --- | --- | --- | --- | --- |
thestandardcomputervisionmetricsdescribedabove(e.g.
| precision, | recall,  | IoU, and          | average | precision) |                   | on held-out |     |     |     |     |     |     |     |     |
| ---------- | -------- | ----------------- | ------- | ---------- | ----------------- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- |
| patches.   | However, | atdeploymenttime, |         |            | artifactscanarise |             |     |     |     |     |     |     |     |     |
whenthesemodelsareusedtoconstructlargemapsfrom
| entireimagescenes[29,78].           |     |     | Modelperformancebeyond |                     |     |     |     |     |     |     |     |     |     |     |
| ----------------------------------- | --- | --- | ---------------------- | ------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| patch-levelmeasuresarecritical[49]. |     |     |                        | Specifically,models |     |     |     |     |     |     |     |     |     |     |
Figure3.Spatialconsistency.Fouroverlappingcropsfromeach
mustberobusttotilingartifacts,invarianttoinputordering
imagecornerareindependentlysegmented.Theconsistencymask
andpreprocessingconventions,andstableundermoderate
|     |     |     |     |     |     |     | shows | pixel-level | agreement, |     | with yellow | indicating | unanimity |     |
| --- | --- | --- | --- | --- | --- | --- | ----- | ----------- | ---------- | --- | ----------- | ---------- | --------- | --- |
changesinspatialscale.
acrossallfourpredictions,andpurpledisagreement.Thismetric
Weproposedeployment-orientedmetricstocomplement quantifiesgridartifactresistanceforlarge-scalefielddelineation.
| standardperformancemetrics. |     |     |     | Thesemetricsaimtochar- |     |     |     |     |     |     |     |     |     |     |
| --------------------------- | --- | --- | --- | ---------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
acterizemodelbehavioratdeploymenttime. Thesemetrics whereu ∈ Ωindexesoverthepixelsinthesharedarea. A
canbecomputedonthepredefineddatasetsplitsusedduring
consistencyof1impliesthemodelisperfectlytranslation-
model development, giving practitioners insight into how equivariantwithintherangeofshiftsimpliedbythecorner
modelsarelikelytobehavewhentiledoverlargescenesat crops,whilelowervaluescorrespondtostrongersensitivity
inferencetime.
|     |     |     |     |     |     |     | to small | translations |     | of the | input. | Figure | 3 illustrates | the |
| --- | --- | --- | --- | --- | --- | --- | -------- | ------------ | --- | ------ | ------ | ------ | ------------- | --- |
croppingschemeandoverlappingregionΩusedtocompute
| Consistencyundertranslations. |     |     |     | ModernCNNsandViTs |     |     |     |     |     |     |     |     |     |     |
| ----------------------------- | --- | --- | --- | ----------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Eq.1.
| arenottranslationequivariantinpractice[5,18,27]. |     |     |     |     |     | Small |     |     |     |     |     |     |     |     |
| ------------------------------------------------ | --- | --- | --- | --- | --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- |
shifts in input can lead to large changes in output due to Sensitivitytoinputordering. Manymappredictionprob-
paddingchoices,aliasingfromstrideddownsampling,and lemscanbedefinedtooperateonsmallsetsofco-registered
architectural elements, such as absolute positional encod- observations(e.g.,multi-temporalSentinel-2scenesormulti-
ings [3, 56, 76]. In patch-based prediction pipelines, this sensorstacks)thataremostnaturallyviewedasunordered
sensitivitymanifestsasvisiblegridartifactswhenindepen- setsratherthanorderedsequences. Insuchcases,wewould
dentlyprocessedpatchesarestitchedintoalargemap[29]. likemodelpredictionstobepermutationinvariantwithre-
Priorworkrecommendsseveralapproachestoreducetrans- spect to the ordering of the input elements [54, 75]. For
lationsensitivity,includingstrategiesthataverageoverlap- example,theFTWdataset[34]providespairedSentinel-2
pinglogits,andvariantsofsliding-windowinferencewith observationsfromtheplantingandharvestingstagesofthe
bufferedbordersthatarediscardedduringstitching[29]. growingseason. Intheofficialimplementationaccompany-
Wefollowworkthatmeasurestranslationrobustnessvia ingthedataset,theinputtensorisconstructedbystacking
thebandsinacanonical(planting,harvest)ordering[16,20].
consistencyofmodeloutputsundershifts[76,77]andadapt
ittopatch-basedgeospatialsemanticsegmentation. Specifi- Atinferencetime,however,practitionersmayselectdifferent
cally,weextendtheideaofmeanAverageSemanticSegmen- scenesbasedondataavailability,cloudcoverage,andpro-
cessinglevel,andmayinadvertentlypermutethetemporal
tationConsistencyfrom[77]andcomputepredictionagree-
ment across four overlapping corner crops of each patch, order,causingmodelsthataretrainedinthestandardwayto
ratherthantwoglobalcrops,tobettermimicthetilingsetup fail. ForGeoMLtaskswhereinputsconsistofobservations
collectedatdifferenttimes,butpreservingtemporalorder-
usedtocreatemapsatinferencetime.
Letx∈RC×S×S ing is not important, we argue that models should ideally
denoteaninputpatchwithheightand
|     |     |     |     |     |     |     | beinsensitivetosuchpermutations. |     |     |     |     | Thus,toquantifythis |     |     |
| --- | --- | --- | --- | --- | --- | --- | -------------------------------- | --- | --- | --- | --- | ------------------- | --- | --- |
widthS.WechooseacropsizepsuchthatS/2<p<Sand
takefourcropsofx,oneanchoredateachcorner,constructed propertywedefineaninput-ordersensitivitymetric.
|                                                    |     |     |     |     |     |     | Letπ | denotethe“reference”orderingofinputchannels |     |     |     |     |     |     |
| -------------------------------------------------- | --- | --- | --- | --- | --- | --- | ---- | ------------------------------------------- | --- | --- | --- | --- | --- | --- |
| sothattheyshareacentraloverlappingregionΩbutdiffer |     |     |     |     |     |     |      | 0                                           |     |     |     |     |     |     |
(e.g.,thetraining-timeconvention),andΠbeasetofalterna-
| intheirsurroundingspatialcontext. |     |     |     | Lety˜(k)denotethehard |     |     |     |     |     |     |     |     |     |     |
| --------------------------------- | --- | --- | --- | --------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
per-pixelpredictions(e.g.,afterargmaxoverlogits)obtained tivepermutationsofthosechannels,i.e. thereverseordering
|        |              |       |     |          |      |             | for a | pair of | observations. |     | For a given | evaluation |     | sample |
| ------ | ------------ | ----- | --- | -------- | ---- | ----------- | ----- | ------- | ------------- | --- | ----------- | ---------- | --- | ------ |
| from a | segmentation | model | on  | the k-th | crop | and further |       |         |               |     |             |            |     |        |
croppingtheresulttotheareaΩ. Wedefinetheconsistency (x,y) and performance metric m (e.g., IoU), we compute
ofxasthefractionofpixelsintheoverlappingregionwhose
predictedlabelsagreeacrossallfourcrops: (cid:0) f(xπ0), (cid:1)
|     |     |     |     |     |     |     |     |     | m (x,y)=m |     |     | y , |     | (2) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --------- | --- | --- | --- | --- | --- |
ref
|     |                    |                                     |     |     |     |           |     |     |        | 1   | (cid:88) | (cid:0) | (cid:1) |     |
| --- | ------------------ | ----------------------------------- | --- | --- | --- | --------- | --- | --- | ------ | --- | -------- | ------- | ------- | --- |
| 1   | (cid:88) (cid:104) |                                     |     |     |     | (cid:105) |     | m   | (x,y)= |     | m        | f(xπ),  | y ,     | (3) |
|     | 1                  | y˜(1)(u)=y˜(2)(u)=y˜(3)(u)=y˜(4)(u) |     |     |     | (1)       |     |     | perm   | |Π  | |        |         |         |     |
|Ω|
π∈Π
u∈Ω
5

wherexπ denotestheinputwithchannelsreorderedaccord- 4.1.Modelperformance&architecturecomparison
| ing to permutation |     | π.  | The per-sample |     | order sensitivity | is  |     |     |     |     |     |     |
| ------------------ | --- | --- | -------------- | --- | ----------------- | --- | --- | --- | --- | --- | --- | --- |
Table1comparesourmodelagainstsemantic,instance,and
| then∆ order | (x,y)=m | ref | (x,y)−m | perm | (x,y)andthedataset- |     |     |     |     |     |     |     |
| ----------- | ------- | --- | ------- | ---- | ------------------- | --- | --- | --- | --- | --- | --- | --- |
GFMmodelbaselinesontheFTWbenchmark.
levelinput-ordersensitivityistheaverageabsolutedropin
performanceacrosssamples. Architecture families reveal task-dependent strengths.
Semanticsegmentationmodels,particularlyU-Netvariants,
| Robustness | to          | preprocessing |           | conventions. | Satellite | im-    |               |            |     |             |     |              |
| ---------- | ----------- | ------------- | --------- | ------------ | --------- | ------ | ------------- | ---------- | --- | ----------- | --- | ------------ |
|            |             |               |           |              |           |        | remain strong | performers | on  | pixel-level | and | object-level |
| agery is   | distributed | and           | processed | under        | a variety | of ra- |               |            |     |             |     |              |
metrics,outperforminginstance-basedandGFMapproaches
| diometricconventions. |     |     | Forexample,Sentinel-2Level-2A |     |     |     |                |                |     |             |          |         |
| --------------------- | --- | --- | ----------------------------- | --- | --- | --- | -------------- | -------------- | --- | ----------- | -------- | ------- |
|                       |     |     |                               |     |     |     | in pixel-level | IoU, precision |     | and recall. | Instance | segmen- |
productsaretypicallystoredasquantizeddigitalnumbers
tationmodels(DelineateAnything,SAM)achievereason-
withscalefactors(e.g.,divisionby10,000). Startingfrom able performance in specific contexts with zero-shot set-
ProcessingBaseline04.00(inFebruary2022),anadditivera-
|     |     |     |     |     |     |     | tings. M2F | shows competitive |     | object-level |     | performance |
| --- | --- | --- | --- | --- | --- | --- | ---------- | ----------------- | --- | ------------ | --- | ----------- |
diometricoffset(e.g.,BOA_ADD_OFFSET)mustbeapplied
(precision=0.62,F1=0.39)butlowerpixel-levelpredictions
| when converting |     | to physical |     | reflectance | [22, 23]. | Down- |             |                                           |     |     |     |     |
| --------------- | --- | ----------- | --- | ----------- | --------- | ----- | ----------- | ----------------------------------------- | --- | --- | --- | --- |
|                 |     |             |     |             |           |       | (IoU=0.68). | Giventheirslowerinferencespeeds,thesemod- |     |     |     |     |
streampipelinesmayadditionallyre-scaleornormalizethe
elsarelesspracticalforoperationaldeployment;wefocused
data(e.g.,dividingby3,000insteadof10,000,ordividing
onoptimizingtheU-Netbaselinewhichexhibitsbothhigh
perbandbyadatasetpercentile).
accuracymetricsandhighthroughput(623.28km2/s).
Inpractice,trainingandinferencepipelinesforGeoML GFMspairedwithsemanticdecodersachievedmoderate
modelsoftendonotshareidenticalpreprocessingsteps,es-
performance,butgenerallyunderperformedspecializedar-
peciallywhenmodelsarereusedacrossorganizations,code-
chitectures,despitehaving3-10×moreparametersthanthe
bases,ordataproviders. Toassesshowbrittleamodelisto U-Netbaseline. Clay(ViT-L)wasthebestGFMperformer
suchvariations,wedefineapreprocessinginvariancemetric.
(IoU=0.67,F1=0.36),butwasstill9%and11%lowerthan
Let g denote the “reference” normalization used dur- ouroptimizedU-Net(PRUE).Thisdiscrepancyislikelydue
ref
ingtraining(e.g.,radiometricoffsetcorrectionfollowedby
tothelowereffectiveresolutionofGFMencoders,whichout-
|     | 10,000), |     | {g  | }J  | J   |     |     |     |     |     |     |     |
| --- | -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
division by and let j j=1 denote alternative putcoarse-scalepatch-wiseembeddings. Extendedresults
normalizations that reflect plausible deploy-time choices fromGFMexperimentsaresharedintheSupplementA.
(e.g.,differentscalefactors,omissionofoffsets,orsimple
Systematicoptimizationmattersmorethanarchitectural
| min–maxscaling). |     | Foreach(x,y),wecompute |     |     |     |     |         |                  |        |        |          |         |
| ---------------- | --- | ---------------------- | --- | --- | --- | --- | ------- | ---------------- | ------ | ------ | -------- | ------- |
|                  |     |                        |     |     |     |     | choice. | The architecture | design | search | detailed | in §3.3 |
(cid:0) (cid:1) shows that increasing encoder depth and choosing an ap-
| m ref | (x,y)=m | f(g | ref (x)), | y , |     | (4) |     |     |     |     |     |     |
| ----- | ------- | --- | --------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
propriatelossfunctioncansubstantiallyimprovedelineation
|     |     | (cid:0) |     | (cid:1) |     |     |     |     |     |     |     |     |
| --- | --- | ------- | --- | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
m (x,y)=m f(g (x)), y forj =1,...,J. (5) quality. Log-cosh Dice loss produces smoother optimiza-
| j   |     |     | j   |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
tionandsuperiorboundarycompletenesscomparedtoother
Wethendefinetheper-samplepreprocessingsensitivityas losses,whilemoderateboundaryclassweighting(ω =0.75)
|          |     | (cid:80)J (cid:12) |         |     | (cid:12)                    |     |     |     |     |     |     |     |
| -------- | --- | ------------------ | ------- | --- | --------------------------- | --- | --- | --- | --- | --- | --- | --- |
| ∆ (x,y)= | 1   | (cid:12)m          | (x,y)−m |     | (x,y) (cid:12). Thedataset- |     |     |     |     |     |     |     |
prep J j=1 ref j yields the most balanced precision–recall trade-off. Com-
levelpreprocessingsensitivityistheaverageabsolutedrop
biningthesewithbrightnessandresizeaugmentationspro-
inperformanceacrosssamples. ducesconsistentperformanceacrossdomains,confirming
Sensitivitytospatialscale. GeoMLmodelsarefrequently thatrobustnessemergesfromtheinteractionofarchitecture,
trainingobjective,anddatadesignchoices.
| trained | at a fixed | spatial | resolution | (e.g., | 10m | Sentinel-2 |     |     |     |     |     |     |
| ------- | ---------- | ------- | ---------- | ------ | --- | ---------- | --- | --- | --- | --- | --- | --- |
pixels)butmaybedeployedonimagerywithdifferenteffec-
4.2.Finalmodelselectionandrobustnessevaluation
tiveresolution(e.g.,resampledSentinel-2,orPlanetScope
mosaics). Recentself-supervisedpretrainingapproachesfor Our final model, PRUE, integrates the best-performing
remotesensingexplicitlyconditionViTmodelsonspatial design choices: U-Net decoder with EfficientNet-B7 en-
scale [48]. We measure scale sensitivity under test-time coder,channelshufflingforinput-orderinvariance,bright-
resizesinamethodsimilartothepreviousmetrics. Wecom- ness and resize augmentations, log-cosh Dice loss, and
|     |     |     |     |     |     |     | boundaryweightingω |     | =0.75. | Table1comparesourfinal |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------ | --- | ------ | ---------------------- | --- | --- |
putetheperformancedifferencebetweenthemodelrunwith
standardinputsversusresizedinputs,∆ (x,y). modelagainstotherbackbonearchitectures;PRUEachieves
scale
IoU=0.76andobjectF1=0.47,representing+6%and+9%
4.ResultsandAnalysis improvementsovertheFTWbaseline. WeselectedU-Net
overFCSiamduetohighertemporalconsistency.
Ourresultsrevealseveralkeyinsightsregardingtheperfor- Table2showsthatPRUEisthemostrobustconfiguration
manceofdifferentmodelarchitecturesanddesignchoices acrossalldeployment-orientedperturbations(§3.5). Bright-
(§4.1) based on traditional metrics and our proposed nessandresizeaugmentationsreducesensitivitytoillumina-
deployment-orientedmetrics(§4.2). tionandscalechanges,whilechannelshufflingeliminates
6

Table 1. Performance comparison across model families on FTW test set (excluding presence-only countries). Semantic baselines:
Post-processedwithconnectedcomponents,usingpresence-onlylabelmaskingintraining[34].Instance/panopticmodels:Fine-tuned
for8-channelinputandFTW-specificclassesbutwithoutpresence-onlymaskingduetoarchitectureconstraints.GFMmodels:Frozen
encoderswithourtrainedconvolutionaldecoder.Totalparametercountincludesboththefrozenencoder(28M–300M)andourlearnable
components,whichconsistofaframes-fusionmoduleandaconvolutionaldecoder(anadditional30M–110Mparameters,dependingonthe
encoder’soutputdimensionality).Boldindicatesbestperformance,underlineindicatessecond-best.ThroughputmeasuredonaV100-32GB
GPUwithbatchsize64.*GalileodidnotfitintoVRAMatanybatchsize.
Pixel-level Object-level #ParamsThroughput
Model Backbone
IoU↑ Prec↑ Recall↑ Prec↑ Recall↑ F1↑ AP0.5:0.95 ↑ AP0.5 ↑ (M)↓ (km2/s)↑
Semanticsegmentationbaselines
FTW-Baseline U-Net+EfficientNet-B3 0.70 0.90 0.72 0.40 0.37 0.38 0.22 0.39 13.2 623.28
DECODE FracTALResUNet 0.71 0.83 0.83 0.27 0.17 0.21 0.09 0.17 64.8 113.47
Instance&panopticsegmentation
Mask2Former Swin-S(fine-tuned,8ch) 0.68 0.88 0.75 0.62 0.30 0.39 0.28 0.44 68.8 26.66
SAM ViT-Huge(fine-tuned,8ch) 0.45 0.73 0.54 0.56 0.34 0.37 0.21 0.19 642.7 0.17
SAM ViT-Huge(zero-shot,3ch) 0.32 0.36 0.82 0.14 0.34 0.17 0.06 0.12 641.1 0.17
Del-Any YOLOv11(zero-shot,3ch,winA) 0.37 0.53 0.56 0.25 0.05 0.09 0.05 0.10 56.9 87.32
Del-AnyS YOLOv11(zero-shot,3ch,winA) 0.44 0.52 0.73 0.15 0.06 0.08 0.07 0.14 2.6 389.24
Geospatialfoundationmodels
Clay ViT-Large 0.67 0.91 0.72 0.38 0.36 0.36 0.24 0.41 363.8 10.98
Galileo ViT-Base 0.66 0.86 0.72 0.29 0.36 0.32 0.21 0.37 119.0 *
DINOv3 ViT-Large 0.60 0.90 0.64 0.39 0.27 0.31 0.20 0.35 412.2 46.59
TerraMind ViT-Base 0.57 0.88 0.62 0.30 0.24 0.26 0.17 0.31 189.1 123.12
Prithvi2.0 ViT-Large 0.56 0.88 0.60 0.29 0.23 0.25 0.16 0.30 439.5 63.38
TerraFM ViT-Base 0.57 0.86 0.62 0.29 0.23 0.25 0.16 0.29 218.6 110.68
CROMA ViT-Base 0.52 0.86 0.56 0.26 0.18 0.21 0.13 0.25 137.0 133.10
SoftCon ViT-Small 0.52 0.85 0.57 0.24 0.18 0.21 0.12 0.24 101.8 234.18
DeCUR ViT-Small 0.49 0.85 0.53 0.23 0.16 0.19 0.11 0.22 120.2 271.67
DOFA-v1 ViT-Large 0.49 0.85 0.53 0.21 0.15 0.17 0.10 0.19 446.0 64.85
Satlas Swin-Tiny 0.45 0.79 0.50 0.13 0.11 0.12 0.07 0.14 131.7 79.85
PRUE(ours) U-Net+EfficientNet-B7 0.76 0.89 0.83 0.62 0.40 0.47 0.26 0.40 67.1 306.94
Table2. AblationresultsforcontrolledexperimentsonFTWtestset(excludingpresence-onlycountries)inwhicheachrowvariesa
singledesignchoice(dataaugmentations,classweighting,encoder,lossfunction,orarchitecture). TheCombinationrowsreportthe
best-performingjointconfigurationsfortheFCSiamandU-Netmodels.Boldindicatesbestperformance,underlineindicatessecond-best.
Performance Inputorder Brightness Scale Agree.
Category Ablation
ObjectF1↑ PixelIoU↑ F1|∆|↓ IoU|∆|↓ F1|∆|↓ IoU|∆|↓ F1|∆|↓ IoU|∆|↓ Avg↑
FTW-Baseline 0.39±0.08 0.68±0.08 0.07 0.11 0.04 0.05 0.15 0.12 0.93
Dataaugs Brightness+Resize 0.38±0.08 0.66±0.09 0.06 0.10 0.02 0.03 0.00 0.01 0.95
Dataaugs Channelshuffle 0.39±0.07 0.68±0.09 0.00 0.00 0.04 0.05 0.17 0.14 0.94
Classweights ω=0.75 0.42±0.06 0.74±0.07 0.08 0.11 0.07 0.07 0.29 0.15 0.95
Encoder EfficientNet-B7 0.42±0.07 0.71±0.08 0.07 0.09 0.03 0.04 0.20 0.13 0.94
Lossfunction log-coshDice 0.44±0.07 0.77±0.06 0.09 0.13 0.06 0.05 0.36 0.20 0.94
Architecture FCSiam 0.40±0.07 0.69±0.08 0.00 0.00 0.05 0.06 0.22 0.14 0.92
FCSiamcombo 0.44±0.07 0.75±0.07 0.00 0.00 0.04 0.05 0.05 0.02 0.94
Combination
PRUE(U-Net) 0.47±0.07 0.76±0.08 0.00 0.00 0.00 0.00 0.01 0.01 0.95
input-order dependency. Compared to the FTW baseline, 5.AI-DerivedField-BoundariesatScale
PRUEdemonstratesnegligiblevarianceunderinputorder
andbrightnessshifts,andFigure2showsPRUE’simproved
Country-scalefieldboundaries. WeusedPRUEtogen-
generalization compared to the baseline model. These re-
eratecompletefieldboundarymapsin2023and2024for
sultsvalidatethatrobustness,accuracy,andscalabilityare
fivecountries: Japan,Mexico,Rwanda,SouthAfrica,and
notcompetingobjectivesbutcanbeco-optimizedthrough
Switzerland,coveringover4.76millionkm2. Thesecoun-
deliberate model and data design. Extended per-country
trieswereselectedtocoverdiverseclimaticzones,farming
resultsandablationanalysesareinSupplementC.
practices,fieldsizes,andagriculturalsystems,representing
realisticglobaldeploymentscenarios. Figure1showsexam-
7

|     |     |     |     |     |     |     | Fieldboundarychangesegmentation |     |     |     | Toquantifystruc- |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------------------- | --- | --- | --- | ---------------- | --- | --- |
Table3.AgriculturalfieldstatisticsderivedusingPRUE.The
turalchangesinagriculturallandscapes,wecomputedfield-
tablereportstotallandarea,distributedinferencecost,fieldcounts,
levelchangedirectlyfromthemodel’smulti-yearsemantic
andmedianfieldareainhectaresforfivediversecountries.
predictions,whichconsistofrasterlogitsproducedforthe
Areaprocessed Cost Fields Median field class for each country and year. We computed the
| Country |              |     |     | Year |     |          |     |     |     |     |     |     |     |
| ------- | ------------ | --- | --- | ---- | --- | -------- | --- | --- | --- | --- | --- | --- | --- |
|         | (millionkm2) |     | ($) |      | (M) | area(ha) |     |     |     |     |     |     |     |
absolutedifferencebetweenthetwo,resultinginachange
|     |     |     |     | 2023 | 0.18 | 0.05 |     |     |     |     |     |     |     |
| --- | --- | --- | --- | ---- | ---- | ---- | --- | --- | --- | --- | --- | --- | --- |
Rwanda 0.02 3.23 magnitudemap. Wethenmin–maxnormalizedandthresh-
|     |     |     |     | 2024 | 0.26 | 0.06 |     |     |     |     |     |     |     |
| --- | --- | --- | --- | ---- | ---- | ---- | --- | --- | --- | --- | --- | --- | --- |
2023 0.36 0.32 olded the change at 0.5 to obtain a binary change mask.
| Switzerland |     | 0.09 | 5.11 |     |     |     |     |     |     |     |     |     |     |
| ----------- | --- | ---- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
2024 0.36 0.28 Figure1showsanexamplechangemapforJapan. Supple-
mentEandFprovideadditionalvisualizationsandchange
| Japan |     | 0.65 | 7.59 | 2023 | 1.55 | 0.20 |     |     |     |     |     |     |     |
| ----- | --- | ---- | ---- | ---- | ---- | ---- | --- | --- | --- | --- | --- | --- | --- |
|       |     |      |      | 2024 | 1.61 | 0.19 |     |     |     |     |     |     |     |
detectionexamplesacrossfivecountries.
|             |     |      |      | 2023 | 2.99 | 0.08 |     |     |     |     |     |     |     |
| ----------- | --- | ---- | ---- | ---- | ---- | ---- | --- | --- | --- | --- | --- | --- | --- |
| SouthAfrica |     | 1.60 | 8.03 | 2024 | 2.70 | 0.07 |     |     |     |     |     |     |     |
6.Conclusions
|        |     |      |      | 2023 | 5.90 | 0.09 |     |     |     |     |     |     |     |
| ------ | --- | ---- | ---- | ---- | ---- | ---- | --- | --- | --- | --- | --- | --- | --- |
| Mexico |     | 2.39 | 8.26 | 2024 | 6.57 | 0.09 |     |     |     |     |     |     |     |
Wepresentedasystematicstudyofmodelarchitectures,train-
| plesfromJapan. |     | Theresultingmapsshowthatthemodel |     |     |     |     |     |     |     |     |     |     |     |
| -------------- | --- | -------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
ingstrategies,androbustnessevaluationmetricsforlarge-
| preserves | field topology, |     | maintains | coherence |     | across tile |                    |       |          |              |     |     |       |
| --------- | --------------- | --- | --------- | --------- | --- | ----------- | ------------------ | ----- | -------- | ------------ | --- | --- | ----- |
|           |                 |     |           |           |     |             | scale agricultural | field | boundary | delineation. |     | Our | study |
boundaries,andgeneralizeswithoutretrainingorregional
establishesanewstate-of-the-artontheFTWbenchmark,
| fine-tuning,          | thus | demonstrating |         | both          | scalability | and zero-  |            |                     |     |         |      |        |         |
| --------------------- | ---- | ------------- | ------- | ------------- | ----------- | ---------- | ---------- | ------------------- | --- | ------- | ---- | ------ | ------- |
|                       |      |               |         |               |             |            | introduces | deployment-oriented |     | metrics | that | better | reflect |
| shot transferability. |      | Table         | 3 gives | country-level |             | statistics |            |                     |     |         |      |        |         |
real-worldbehavior,anddemonstratesoperationalviability
derivedfromthemappedfieldboundaries.
throughcountry-scaledeployments.
| Mosaickingandinferencepipeline. |     |     |     | Toenablelarge-scale |     |     |                |      |         |      |            |          |     |
| ------------------------------- | --- | --- | --- | ------------------- | --- | --- | -------------- | ---- | ------- | ---- | ---------- | -------- | --- |
|                                 |     |     |     |                     |     |     | Design choices | that | matter. | Loss | functions, | boundary |     |
mapping,wedevelopedapipelinefornational-scaleinfer-
weighting,andtargetedaugmentationshavethestrongestim-
encethatemphasizesthroughput,spatialconsistency,cost
|     |     |     |     |     |     |     | pactonaccuracyandrobustness. |     |     | Log-coshDiceandmoder- |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ---------------------------- | --- | --- | --------------------- | --- | --- | --- |
efficiency,andreproducibility.Thefirststepgeneratescloud-
ateboundaryweightingimproveboundarycompletenessand
| free, seasonally |     | aligned | Sentinel-2 | mosaics |     | using a tiling |     |     |     |     |     |     |     |
| ---------------- | --- | ------- | ---------- | ------- | --- | -------------- | --- | --- | --- | --- | --- | --- | --- |
precision–recallbalance,whileaugmentationstargetingde-
frameworkandlatitude-basedplanting/harvestseasonselec-
ploymentfailures(brightness,scale,channelshuffling)yield
tionalgorithms(seeSupplementD).Patchesof256×256
|     |     |     |     |     |     |     | measurablegains. | ThoughexploredonlyforU-Netvariants, |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ---------------- | ----------------------------------- | --- | --- | --- | --- | --- |
arethenreadandprocessedbythemodelwitha25%overlap,
thisdesign-spacemethodologyisarchitecture-agnosticand
withGaussian-weightedaveragingappliedacrossoverlaps
applicabletoinstance,panoptic,andGFM-basedmodels.
withanapodizationkernel[61]thatplacesgreaterweighton
predictionsnearpatchcenters,whichreducesedgeartifacts GFMsneedtask-specificadaptation. DespitebroadEO
andensuresconsistentboundarylogitsnearpatchborders. pretraining,GFMsgenerallyunderperformspecializedseg-
Thestitchedprobabilitymapsarethenvectorizedinablock- mentationmodelsduetoresolutionlimitsandweakerlocal-
wisemannerusing4,096×4,096-pixelwindowstomaintain ization.Theyrequirehigh-resolutiondecodersandboundary-
memoryefficiency. Finally,resultingpolygonsareserialized awareobjectivestomatchtask-specificarchitectures.
| in the fiboa | [60]                                       | GeoParquet |     | format,   | enabling | efficient   |             |             |             |             |              |              |     |
| ------------ | ------------------------------------------ | ---------- | --- | --------- | -------- | ----------- | ----------- | ----------- | ----------- | ----------- | ------------ | ------------ | --- |
|              |                                            |            |     |           |          |             | Robustness  | metrics     | predict     | real-world  |              | performance. |     |
| downstream   | querying,                                  | temporal   |     | indexing, | and      | large-scale |             |             |             |             |              |              |     |
|              |                                            |            |     |           |          |             | Standard    | metrics do  | not capture | translation |              | sensitivity, |     |
| analytics.   | Thispipelineenablesproductionofcontiguous, |            |     |           |          |             |             |             |             |             |              |              |     |
|              |                                            |            |     |           |          |             | input-order | dependence, | or          | radiometric | brittleness. |              | Our |
artifact-freefieldboundarylayersatacountryscale.
deployment-orientedmetricsquantifythesebehaviorsand
Throughputandcostefficiency. Table3reportsthetotal guidetargetedimprovements. Modelsoptimizedwiththese
landareaprocessedandinferencecostforeachcountry. Cre- metrics show reduced sensitivity to brightness, scale, and
atingtheplantingandharvestmosaicsfortwoagricultural translation. Country-scaledeploymentsconfirmthathigher
seasons(2023/24-2024/25)takes49.3minutes. Ourinfer- robustness scores correlate with fewer artifacts and more
stableperformanceacrossmillionsofkm2.
encepipelineisoptimizedtomaximizeGPUutilizationin
| a cluster            | pool of | up to                           | 256 NVIDIA |     | A10G | GPUs (AWS |     |     |     |     |     |     |     |
| -------------------- | ------- | ------------------------------- | ---------- | --- | ---- | --------- | --- | --- | --- | --- | --- | --- | --- |
| g5.xlargeinstances). |         | Processingthe2-yearMexicomosaic |            |     |      |           |     |     |     |     |     |     |     |
Acknowledgments
| resulted | in an execution |     | time of | only | 14.23 | min at a cost |     |     |     |     |     |     |     |
| -------- | --------------- | --- | ------- | ---- | ----- | ------------- | --- | --- | --- | --- | --- | --- | --- |
of$8.26($1.05×10−6/km2),consistingof4.8GPU-Hrs, ThisprojectwassupportedbyfundingfromTaylorGeospa-
77.9Core-Hrs,andathroughputof232.4GB-Hrs. Using tial. ZFwassupportedbyfundingfromNASA’sLandCover
theMachineLearningImpactcalculator[36],weestimate Land-UseChangeprogram,award#80NSSC23K0528. We
thisrungeneratedatotalemissionof0.25kgCO eq,100% appreciatethefeedbackandsuggestionsonthisworkpro-
2
| ofwhichwasoffsetbyAWS. |     |     |     |     |     |     | videdbyFuxinLiandJamonVanDenHoek. |     |     |     |     |     |     |
| ---------------------- | --- | --- | --- | --- | --- | --- | --------------------------------- | --- | --- | --- | --- | --- | --- |
8

References [13] R. d’Andrimont, M. Claverie, P. Kempeneers, D. Muraro,
|     |     |     |     | M.  | Yordanov, | D. Peressutti, |     | M. Baticˇ, | and | F. Waldner. |
| --- | --- | --- | --- | --- | --------- | -------------- | --- | ---------- | --- | ----------- |
[1] GuillaumeAstruc, NicolasGonthier, ClémentMallet, and AI4Boundaries: AnOpenAI-ReadyDatasettoMapField
| LoicLandrieu.                         | AnySat: OneEarthObservationModelfor |                 |     |                                               |     |     |     |     |     |       |
| ------------------------------------- | ----------------------------------- | --------------- | --- | --------------------------------------------- | --- | --- | --- | --- | --- | ----- |
|                                       |                                     |                 |     | BoundarieswithSentinel-2andAerialPhotography. |     |     |     |     |     | Earth |
| ManyResolutions,Scales,andModalities. |                                     | InProceedingsof |     |                                               |     |     |     |     |     |       |
|                                       |                                     |                 |     | SystemScienceData,15(1):317–329,2023.         |     |     |     |     | 1,2 |       |
theIEEE/CVFConferenceonComputerVisionandPattern
[14] MuhammadSohailDanish,MuhammadAkhtarMunir,Syed
| Recognition(CVPR),pages19530–19540,2025. |     |     | 3   |     |     |     |     |     |     |     |
| ---------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
RoshaanAliShah,MuhammadHarisKhan,RaoMuhammad
[2] RezaAzad,MoeinHeidary,KadirYilmaz,MichaelHütte- Anwer,JormaLaaksonen,FahadShahbazKhan,andSalman
mann,SanazKarimijafarbigloo,YuliWu,AnkeSchmeink,
Khan.TerraFM:Ascalablefoundationmodelforunifiedmul-
| andDoritMerhof.                      | LossFunctionsintheEraofSemantic |     |     |                                           |     |     |                              |     |     |       |
| ------------------------------------ | ------------------------------- | --- | --- | ----------------------------------------- | --- | --- | ---------------------------- | --- | --- | ----- |
|                                      |                                 |     |     | tisensorearthobservation.                 |     |     | InTheFourteenthInternational |     |     |       |
| Segmentation:ASurveyandOutlook,2023. |                                 |     | 4   |                                           |     |     |                              |     |     |       |
|                                      |                                 |     |     | ConferenceonLearningRepresentations,2026. |     |     |                              |     |     | 3,4,1 |
[3] AharonAzulayandYairWeiss. Whydodeepconvolutional [15] Osmar Luiz Ferreira de Carvalho, Osmar Abílio de Car-
networks generalize so poorly to small image transforma- valhoJúnior,CristianoRosaeSilva,AnesmarOlinodeAlbu-
tions? JournalofMachineLearningResearch,20(184):1–25,
|     |     |     |     | querque, | Nickolas | Castro | Santana, | Dibio | Leandro | Borges, |
| --- | --- | --- | --- | -------- | -------- | ------ | -------- | ----- | ------- | ------- |
2019. 5
|     |     |     |     | Roberto | Arnaldo | Trancoso | Gomes, |     | and Renato | Fontes |
| --- | --- | --- | --- | ------- | ------- | -------- | ------ | --- | ---------- | ------ |
[4] FavyenBastani,PiperWolters,RitwikGupta,JoeFerdinando, Guimarães. PanopticSegmentationMeetsRemoteSensing.
and Aniruddha Kembhavi. SatlasPretrain: A Large-Scale RemoteSensing,14(4):965,2022. 3
DatasetforRemoteSensingImageUnderstanding. InPro- [16] StephanieRDebats,DeeLuo,LyndonDEstes,ThomasJ
ceedingsoftheIEEE/CVFInternationalConferenceonCom-
|                                          |     |     |       | Fuchs,andKellyKCaylor. |     |     | AGeneralizedComputerVision |     |     |     |
| ---------------------------------------- | --- | --- | ----- | ---------------------- | --- | --- | -------------------------- | --- | --- | --- |
| puterVision(ICCV),pages16772–16782,2023. |     |     | 3,4,1 |                        |     |     |                            |     |     |     |
ApproachtoMappingCropFieldsinHeterogeneousAgri-
[5] ValerioBiscioneandJeffreyS.Bowers. Convolutionalneural culturalLandscapes. RemoteSensingofEnvironment,179:
| networksarenotinvarianttotranslation,buttheycanlearn |     |     |     | 210–221,2016. |     | 3,5 |     |     |     |     |
| ---------------------------------------------------- | --- | --- | --- | ------------- | --- | --- | --- | --- | --- | --- |
tobe. JournalofMachineLearningResearch,22(229):1–28,
[17] FoivosI.Diakogiannis,FrançoisWaldner,andPeterCaccetta.
2021. 5
LookingforChange?RolltheDiceandDemandAttention.
[6] Christopher F. Brown, Michal R. Kazmierski, Valerie J. RemoteSensing,13(18),2021. 4
Pasquarella,WilliamJ.Rucklidge,MashaSamsikova,Chen- [18] PeijianDing,DavitSoselia,ThomasArmstrong,JiahaoSu,
huiZhang,EvanShelhamer,EstefaniaLahera,OliviaWiles, andFurongHuang. RevivingShiftEquivarianceinVision
| Simon Ilyushchenko, | Noel Gorelick,  | Lihui  | Lydia Zhang,   |                                                  |     |     |     |     |     |     |
| ------------------- | --------------- | ------ | -------------- | ------------------------------------------------ | --- | --- | --- | --- | --- | --- |
|                     |                 |        |                | Transformers,2023.                               |     | 5   |     |     |     |     |
| Sophia Alj, Emily   | Schechter, Sean | Askay, | Oliver Guinan, |                                                  |     |     |     |     |     |     |
|                     |                 |        |                | [19] HakanErden,MuratAslan,andCemreBaharÖzcanli. |     |     |     |     |     | To  |
RebeccaMoore,AlexisBoukouvalas,andPushmeetKohli. establishanewsubsidysystem. In2015FourthInternational
AlphaEarth Foundations: An Embedding Field Model for ConferenceonAgro-Geoinformatics(Agro-geoinformatics),
| AccurateandEfficientGlobalMappingfromSparseLabel |     |     |     | pages57–60,2015. |     | 1         |         |            |      |          |
| ------------------------------------------------ | --- | --- | --- | ---------------- | --- | --------- | ------- | ---------- | ---- | -------- |
| Data,2025. 3                                     |     |     |     |                  |     |           |         |            |      |          |
|                                                  |     |     |     | [20] Lyndon      | D   | Estes, Su | Ye, Lei | Song, Boka | Luo, | J Ronald |
[7] RodrigoCayeDaudt,BertrLeSaux,andAlexandreBoulch.
|     |     |     |     | Eastman, |     | Zhenhua Meng, | Qi  | Zhang, | Dennis | McRitchie, |
| --- | --- | --- | --- | -------- | --- | ------------- | --- | ------ | ------ | ---------- |
FullyConvolutionalSiameseNetworksforChangeDetec- StephanieRDebats,JustusMuhando,etal. HighResolution,
tion. In201825thIEEEInternationalConferenceonImage AnnualMapsofFieldBoundariesforSmallholder-Dominated
Processing(ICIP),pages4063–4067,2018. 4 CroplandsatNationalScales. FrontiersinArtificialIntelli-
[8] BowenCheng,IshanMisra,AlexanderGSchwing,Alexander
|     |     |     |     | gence,4:744863,2022. |     |     | 3,5 |     |     |     |
| --- | --- | --- | --- | -------------------- | --- | --- | --- | --- | --- | --- |
Kirillov,andRohitGirdhar. Masked-AttentionMaskTrans- [21] L.D.Estes,A.Wussah,M.Asipunu,M.Gathigi,P.KovaÄiÄ,
formerforUniversalImageSegmentation. InProceedingsof J.Muhando,B.V.Yeboah,F.K.Addai,E.S.Akakpo,M.K.
theIEEE/CVFConferenceonComputerVisionandPattern Allotey,P.Amkoya,E.Amponsem,K.D.Donkoh,N.Ha,
Recognition,pages1290–1299,2022. 3,1,4 E.Heltzel,C.Juma,R.Mdawida,A.Miroyo,J.Mucha,J.
[9] Clay.TheClayFoundationModel-AnopensourceAImodel
Mugami,F.Mwawaza,D.A.Nyarko,P.Oduor,K.N.Ohe-
andinterfaceforEarth,2025. Accessed:2025-11-13. 3,4,1 meng,S.I.D.Segbefia,T.Tumbula,F.Wambua,G.H.Xe-
[10] YezhenCong, SamarKhanna, ChenlinMeng, PatrickLiu, flide,S.Ye,andF.Yeboah. ARegion-Wide,Multi-YearSet
Erik Rozi, Yutong He, Marshall Burke, David B. Lobell, ofCropFieldBoundaryLabelsforAfrica,2024. 2
| andStefanoErmon. | SatMAE:Pre-trainingtransformersfor |     |     |               |     |               |     |              |     |            |
| ---------------- | ---------------------------------- | --- | --- | ------------- | --- | ------------- | --- | ------------ | --- | ---------- |
|                  |                                    |     |     | [22] European |     | Space Agency. |     | Introduction | of  | Additional |
temporalandmulti-spectralsatelliteimagery. InAdvancesin Radiometric Offset in Processing Baseline 04.00 Prod-
NeuralInformationProcessingSystems,2022. 3 ucts. https://forum.step.esa.int/t/info-
[11] IsaacCorley,CalebRobinson,RahulDodhia,JuanM.Lav- introduction-of-additional-radiometric-
ista Ferres, and Peyman Najafirad. Revisiting pre-trained offset-in-pb04-00-products/35431,2022. Ac-
remotesensingmodelbenchmarks:resizingandnormaliza- cessed9November2025. 6
tionmatters. In2024IEEE/CVFConferenceonComputer [23] EuropeanSpaceAgency. Sentinel-2ProcessingBaselineand
VisionandPatternRecognitionWorkshops(CVPRW),pages ProductFormat. https://sentiwiki.copernicus.
3162–3172,2024. 2 eu/web/s2-processing,2022. Accessed9November
| [12] IsaacCorley,CalebRobinson,andAnthonyOrtiz. |     |     | AChange | 2025. | 6   |     |     |     |     |     |
| ----------------------------------------------- | --- | --- | ------- | ----- | --- | --- | --- | --- | --- | --- |
DetectionRealityCheck. InICLR2024MachineLearning [24] LucasBFerreira,VitorSMartins,UilsonRVAires,Nuwan
forRemoteSensing(ML4RS)Workshop,2024. 2 Wijewardane,XinZhang,andSathishSamiappan. FieldSeg:
9

Ascalableagriculturalfieldextractionframeworkbasedon [37] MykolaLavreniuk, NataliiaKussul, AndriiShelestov, Bo-
theSegmentAnythingModeland10-mSentinel-2imagery. hdanYailymov,YevheniiSalii,VolodymyrKuzin,andZoltan
ComputersandElectronicsinAgriculture,232:110086,2025. Szantoi. Delineate Anything: Resolution-Agnostic Field
3 BoundaryDelineationonSatelliteImagery. arXivpreprint
[25] AnthonyFuller,KoreenMillard,andJamesGreen. CROMA: arXiv:2504.02534,2025. 3,1
Remote Sensing Representations with Contrastive Radar- [38] JonathanLong,EvanShelhamer,andTrevorDarrell. Fully
OpticalMaskedAutoencoders. AdvancesinNeuralInfor- convolutionalnetworksforsemanticsegmentation. In2015
mationProcessingSystems,36,2024. 3,4,1 IEEEConferenceonComputerVisionandPatternRecogni-
[26] Vivien Sainte Fare Garnot and Loic Landrieu. Panoptic tion(CVPR),pages3431–3440,2015. 4
Segmentation of Satellite Image Time Series with Convo- [39] WeiyeMei,HaoyuWang,DavidFouhey,WeiqiZhou,Isabella
lutional Temporal Attention Networks. In Proceedings of Hinks,JoshMGray,DerekVanBerkel,andMehaJain.Using
theIEEE/CVFInternationalConferenceonComputerVision, DeepLearningandVery-High-ResolutionImagerytoMap
pages4872–4881,2021. 2,3 SmallholderFieldBoundaries. RemoteSensing,14(13):3046,
[27] NateGruver,MarcAntonFinzi,MicahGoldblum,andAn- 2022. 3
drew Gordon Wilson. The Lie Derivative for Measuring [40] CatherineNakalembeandHannahKerner.Considerationsfor
LearnedEquivariance. InTheEleventhInternationalConfer- AI-EOforAgricultureinSub-SaharanAfrica. Environmental
enceonLearningRepresentations,2023. 5 ResearchLetters,18(4):041002,2023. 1,3
[28] KaimingHe,GeorgiaGkioxari,PiotrDollár,andRossGir- [41] CatherineNakalembe,HannahKerner,IvanZvonkov,etal.A
shick. MaskR-CNN. InProceedingsoftheIEEEInterna- FrameworkforEO-BasedNationalAgriculturalMonitoring
tional Conference on Computer Vision, pages 2961–2969, (EO-NAM)-FortheAfricanContext,2024. preprint. 2
2017. 3 [42] Heather C. North, David Pairman, and Stella E. Belliss.
[29] Bohao Huang, Daniel Reichman, Leslie M Collins, Kyle Boundary Delineation of Agricultural Fields in Multitem-
Bradbury, and Jordan M Malof. Tiling and stitching seg- poral Satellite Imagery. IEEE Journal of Selected Topics
mentationoutputforremotesensing: Basicchallengesand inAppliedEarthObservationsandRemoteSensing,12(1):
recommendations. arXivpreprintarXiv:1805.12219,2018. 5 237–251,2019. 1
[30] ShrutiJadon.Asurveyoflossfunctionsforsemanticsegmen- [43] PontusOlofsson,GilesMFoody,MartinHerold,StephenV
tation. In2020IEEEConferenceonComputationalIntelli- Stehman,CurtisEWoodcock,andMichaelAWulder. Good
genceinBioinformaticsandComputationalBiology(CIBCB), practicesforestimatingareaandassessingaccuracyofland
page1–7.IEEE,2020. 4 change. RemoteSensingofEnvironment,148:42–57,2014. 7
[31] JiteshJain,JiachenLi,MangTikChiu,AliHassani,Nikita [44] ClaudioPersello,ValentynATolpekin,JRayBergado,and
Orlov,andHumphreyShi. OneFormer:OneTransformerto RolfADeBy. DelineationofAgriculturalFieldsinSmall-
RuleUniversalImageSegmentation. InCVPR,2023. 3 holder Farms from Satellite Images Using Fully Convolu-
[32] JohannesJakubik,FelixYang,BenediktBlumenstiel,Erik tionalNetworksandCombinatorialGrouping. RemoteSens-
Scheurer, Rocco Sedona, Stefano Maurogiovanni, Jente ingofEnvironment,231:111253,2019. 2,3
Bosmans,NikolaosDionelis,ValerioMarsocci,NiklasKopp, [45] Claudio Persello, Jeroen Grift, Xinyan Fan, Claudia
etal. Terramind: Large-scalegenerativemultimodalityfor Paris, Ronny Hänsch, Mila Koeva, and Andrew Nelson.
earthobservation. IEEE/CVFInternationalConferenceon AI4SmallFarms: A Dataset for Crop Field Delineation in
ComputerVision(ICCV),2025. 3,4,1 SoutheastAsianSmallholderFarms. IEEEGeoscienceand
[33] Hannah Kerner, Catherine Nakalembe, Adam Yang, Ivan RemoteSensingLetters,20:1–5,2023. 2,3
Zvonkov,RyanMcWeeny,GabrielTseng,andInbalBecker- [46] JulienRadouxandPatrickBogaert.GoodPracticesforObject-
Reshef. HowAccurateareExistingLandCoverMapsfor BasedAccuracyAssessment.RemoteSensing,9(7):646,2017.
AgricultureinSub-SaharanAfrica? ScientificData,11(1): 7
486,2024. 3 [47] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang
[34] Hannah Kerner, Snehal Chaudhari, Aninda Ghosh, Caleb Hu,ChaitanyaRyali,TengyuMa,HaithamKhedr,Roman
Robinson,AdeelAhmad,EddieChoi,NathanJacobs,Chris Rädle,ChloeRolland,LauraGustafson,EricMintun,Junting
Holmes,MatthiasMohr,RahulDodhia,etal. Fieldsofthe Pan, KalyanVasudevAlwala, NicolasCarion, Chao-Yuan
World:AMachineLearningBenchmarkDatasetForGlobal Wu,RossGirshick,PiotrDollár,andChristophFeichtenhofer.
AgriculturalFieldBoundarySegmentation.InProceedingsof Sam2:Segmentanythinginimagesandvideos,2024. 1
theAAAIConferenceonArtificialIntelligence,pages28151– [48] ColoradoJReed,RitwikGupta,ShufanLi,SarahBrockman,
28159,2025. 1,2,3,5,7,4 ChristopherFunk,BrianClipp,KurtKeutzer,SalvatoreCan-
[35] AlexanderKirillov,EricMintun,NikhilaRavi,HanziMao, dido,MattUyttendaele,andTrevorDarrell. Scale-MAE:A
ChloeRolland,LauraGustafson,TeteXiao,SpencerWhite- Scale-AwareMaskedAutoencoderforMultiscaleGeospatial
head,AlexanderC.Berg,Wan-YenLo,PiotrDollar,andRoss RepresentationLearning. InProceedingsoftheIEEE/CVF
Girshick. SegmentAnything,2023. 3,1 InternationalConferenceonComputerVision,pages4088–
[36] AlexandreLacoste,AlexandraLuccioni,VictorSchmidt,and 4099,2023. 6
ThomasDandres. QuantifyingtheCarbonEmissionsofMa- [49] EstherRolf,KonstantinKlemmer,CalebRobinson,andHan-
chineLearning. arXivpreprintarXiv:1910.09700,2019. 8 nahKerner. Position: MissionCritical–SatelliteDataisa
10

DistinctModalityinMachineLearning. In41stInternational conferenceonmachinelearning,pages6105–6114.PMLR,
ConferenceonMachineLearning,2024. 2,5 2019. 4
[50] OlafRonneberger,PhilippFischer,andThomasBrox. U-Net: [60] TaylorGeospatialEngine. Fieldboundariesforagriculture
Convolutionalnetworksforbiomedicalimagesegmentation. (fiboa)—specification,toolsandopendata,2025. 8
In International Conference on Medical image computing [61] Jamie Tolan, Hung-I Yang, Benjamin Nosarzewski, Guil-
andcomputer-assistedintervention,pages234–241.Springer, laume Couairon, Huy V Vo, John Brandt, Justine Spore,
2015. 4,1 SayantanMajumdar,DanielHaziza,JanakiVamaraju,etal.
[51] A.RydbergandG.Borgefors. IntegratedMethodforBound- VeryhighresolutioncanopyheightmapsfromRGBimagery
aryDelineationofAgriculturalFieldsinMultispectralSatel- usingself-supervisedvisiontransformerandconvolutionalde-
liteImages. IEEETransactionsonGeoscienceandRemote codertrainedonaeriallidar. RemoteSensingofEnvironment,
Sensing,39(11):2514–2520,2001. 2 300:113888,2024. 8
[52] Seyed Sadegh Mohseni Salehi, Deniz Erdogmus, and Ali [62] GabrielTseng,AnthonyFuller,MarlenaReil,HenryHerzog,
Gholipour. Tversky loss function for image segmentation PatrickBeukema,FavyenBastani,JamesRGreen,EvanShel-
using3Dfullyconvolutionaldeepnetworks,2017. 4 hamer,HannahKerner,andDavidRolnick.Galileo:Learning
[53] PhilippSchuegraf,JulianSchnell,CorentinHenry,andKse- global&localfeaturesofmanyremotesensingmodalities.
nia Bittner. Building Section Instance Segmentation with InProceedingsofthe42ndInternationalConferenceonMa-
Combined Classical and Deep Learning Methods. ISPRS chine Learning, pages 60280–60300. PMLR, 2025. 3, 4,
AnnalsofthePhotogrammetry,RemoteSensingandSpatial 1
InformationSciences,2:407–414,2022. 3 [63] FrançoisWaldnerandFoivosI.Diakogiannis. Deeplearning
[54] Nimrod Segol and Yaron Lipman. On universal equivari- onedge: Extractingfieldboundariesfromsatelliteimages
antsetnetworks. InInternationalConferenceonLearning with a convolutional neural network. Remote Sensing of
Representations,2020. 5 Environment,245:111741,2020. 2
[55] WenzheShi,JoseCaballero,FerencHuszár,JohannesTotz, [64] FrançoisWaldner,FoivosI.Diakogiannis,KathrynBatchelor,
AndrewPAitken,RobBishop,DanielRueckert,andZehan MichaelCiccotosto-Camp,ElizabethCooper-Williams,Chris
Wang. Real-TimeSingleImageandVideoSuper-Resolution Herrmann,GonzaloMata,andAndrewToovey. Detect,Con-
UsinganEfficientSub-PixelConvolutionalNeuralNetwork. solidate,Delineate: ScalableMappingofFieldBoundaries
InProceedingsoftheIEEEConferenceonComputerVision UsingSatelliteImages. RemoteSensing,13(11),2021. 1,2,
andPatternRecognition,pages1874–1883,2016. 4 3
[56] OfirShifmanandYairWeiss. Lostintranslation: Modern [65] SherrieWang,FrançoisWaldner,andDavidB.Lobell. Un-
neuralnetworksstillstrugglewithsmallrealisticimagetrans- lockingLarge-ScaleCropFieldDelineationinSmallholder
formations. InEuropeanConferenceonComputerVision, FarmingSystemswithTransferLearningandWeakSupervi-
pages231–247.Springer,2024. 5 sion. RemoteSensing,14(22),2022. 3
[57] OrianeSiméoni, HuyV.Vo, MaximilianSeitzer, Federico [66] XuyingWang, LeiShu, RuHan, FanYang, TimothyGor-
Baldassarre,MaximeOquab,CijoJose,VasilKhalidov,Marc don,XiaochanWang,andHongyuXu. Asurveyoffarmland
Szafraniec,SeungeunYi,MichaëlRamamonjisoa,Francisco boundaryextractiontechnologybasedonremotesensingim-
Massa,DanielHaziza,LucaWehrstedt,JianyuanWang,Tim- ages. Electronics,12(5),2023. 2
othée Darcet, Théo Moutakanni, Leonel Sentana, Claire [67] Yi Wang, Conrad M. Albrecht, Nassim Ait Ali Braham,
Roberts,AndreaVedaldi,JamieTolan,JohnBrandt,Camille Chenying Liu, Zhitong Xiong, and Xiao Xiang Zhu. De-
Couprie, JulienMairal, HervéJégou, PatrickLabatut, and couplingcommonanduniquerepresentationsformultimodal
PiotrBojanowski. DINOv3,2025. 4,1 self-supervisedlearning. InEur.Conf.Comput.Vis.,pages
[58] DanielaSzwarcman,SujitRoy,PaoloFraccaro,ÞorsteinnElí 286–303,2024. 3,4,1
Gíslason,BenediktBlumenstiel,RinkiGhosal,PedroHen- [68] YiWang,ConradMAlbrecht,andXiaoXiangZhu. Multi-
riquedeOliveira,JoaoLucasdeSousaAlmeida,RoccoSe- LabelGuidedSoftContrastiveLearningforEfficientEarth
dona,YanghuiKang,SrijaChakraborty,SizheWang,Car- ObservationPretraining. InIGARSS2024-2024IEEEInter-
los Gomes, Ankur Kumar, Vishal Gaur, Myscon Truong, nationalGeoscienceandRemoteSensingSymposium,pages
DenysGodwin,SamKhallaghi,HyunhoLee,Chia-YuHsu, 7568–7571,2024. 3,4,1
Ata Akbari Asanjan, Besart Mujeci, Disha Shidham, Ru- [69] BarryWatkinsandAdriaanvanNiekerk. AComparisonof
faiOmowunmiBalogun,VenkateshKolluru,TrevorKeenan, Object-BasedImageAnalysisApproachesforFieldBoundary
PauloArevalo,WenwenLi,HamedAlemohammad,Pontus DelineationUsingMulti-TemporalSentinel-2Imagery. Com-
Olofsson,TimothyMayer,ChristopherHain,RobertKennedy, putersandElectronicsinAgriculture, 158:294–302, 2019.
BiancaZadrozny,DavidBell,GabrieleCavallaro,Campbell 2
Watson,ManilMaskey,RahulRamachandran,andJuanBern- [70] Michael J. Wellington and Luigi J. Renzullo. High-
abeMoreno. Prithvi-eo-2.0:Aversatilemultitemporalfoun- DimensionalSatelliteImageCompositingandStatisticsfor
dationmodelforearthobservationapplications. IEEETrans- EnhancedIrrigatedCropMapping. RemoteSensing,13(7),
actionsonGeoscienceandRemoteSensing,64:1–20,2026. 2021. 3
3,4,1 [71] TeteXiao,YingchengLiu,BoleiZhou,YuningJiang,andJian
[59] MingxingTanandQuocLe. Efficientnet:Rethinkingmodel Sun. UnifiedPerceptualParsingforSceneUnderstanding. In
scalingforconvolutionalneuralnetworks. InInternational EuropeanConferenceonComputerVision.Springer,2018. 4
11

[72] EnzeXie,WenhaiWang,ZhidingYu,AnimaAnandkumar,
Jose M Alvarez, and Ping Luo. Segformer: Simple and
efficientdesignforsemanticsegmentationwithtransformers.
Advancesinneuralinformationprocessingsystems,34:12077–
12090,2021. 4
[73] ZhitongXiong,YiWang,FahongZhang,AdamJStewart,
JoëlleHanna,DamianBorth,IoannisPapoutsis,BertrandLe
Saux, Gustau Camps-Valls, and Xiao Xiang Zhu. Neural
Plasticity-InspiredFoundationModelforObservingtheEarth
CrossingModalities. arXivpreprintarXiv:2403.15356,2024.
3,4,1
[74] Xiaodong Yu, Dahu Shi, Xing Wei, Ye Ren, Tingqun Ye,
andWenmingTan. SOIT:SegmentingObjectswithInstance-
AwareTransformers. CoRR,abs/2112.11037,2021. 3
[75] ManzilZaheer,SatwikKottur,SiamakRavanbakhsh,Barn-
abásPóczos,RuslanSalakhutdinov,andAlexanderJ.Smola.
Deepsets. InAdvancesinNeuralInformationProcessing
Systems(NeurIPS),2017. 5
[76] Richard Zhang. Making Convolutional Networks Shift-
InvariantAgain. InProceedingsofthe36thInternational
ConferenceonMachineLearning,pages7324–7334.PMLR,
2019. 5
[77] Xueyan Zou, Fanyi Xiao, Zhiding Yu, Yuheng Li, and
YongJaeLee. DelvingDeeperintoAnti-AliasinginCon-
vNets. International Journal of Computer Vision, 131(1):
67–81,2023. 5
[78] Ivan Zvonkov, Gabriel Tseng, Catherine Nakalembe, and
HannahKerner. OpenMapFlow: ALibraryforRapidMap
CreationwithMachineLearningandRemoteSensingData.
ProceedingsoftheAAAIConferenceonArtificialIntelligence,
37(12):14655–14663,2023. 5
12

PRUE: A Practical Recipe for Field Boundary Segmentation at Scale
|     |     |     |     |     | Supplementary | Material |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | ------------- | -------- | --- | --- | --- | --- | --- | --- | --- |
A.ExtendedExperimentalDetails creates multiple failure modes: (1) appearance shifts be-
tweenseasonsviolateSAM2’svisualconsistencyassump-
Details on instance and panoptic segmentation model tion, and (2) fields do not “move” between timestamps—
baselines. DelineateAnything[37]isbasedonUltralytics’ they transform—leaving SAM2’s optical flow and corre-
YOLOv11-seg and fine-tuned on FBIS-22M, a dataset of spondencemechanismswithoutusefulsignal.
RGBimagesfrommultipleremotesensingsources(Sentinel-
| 2, Planet, | Maxar, | Pleiades, | orthophotos) |     | over 9 European |     |     |     |     |     |     |     |     |
| ---------- | ------ | --------- | ------------ | --- | --------------- | --- | --- | --- | --- | --- | --- | --- | --- |
countries with spatial resolution 0.25-10m. For Delin- Geospatialfoundationmodels(GFMs). TheFTWbench-
| eate Anything,                              | we      | perform  | a            | 1st-99th | percentile normal- |               |                    |        |        |               |       |        |       |
| ------------------------------------------- | ------- | -------- | ------------ | -------- | ------------------ | ------------- | ------------------ | ------ | ------ | ------------- | ----- | ------ | ----- |
|                                             |         |          |              |          |                    | mark provides | four               | bands  | (RGB   | and NIR)      | [34], | which  | is    |
| izationfollowingthepretrainingdatasetnorms. |         |          |              |          | SAM[35]            |               |                    |        |        |               |       |        |       |
|                                             |         |          |              |          |                    | fewer than    | the multi-spectral |        | inputs | used          | by    | most   | GFMs. |
| is a promptable                             |         | instance | segmentation |          | model pretrained   |               |                    |        |        |               |       |        |       |
|                                             |         |          |              |          |                    | Since many    | GFMs               | expect | 8 to   | 13 Sentinel-2 |       | bands, | we    |
| on natural                                  | images, | which    | we           | assessed | in both zero-shot  |               |                    |        |        |               |       |        |       |
usedGFMsevaluationwrapperpublishedinGalileocode-
and fine-tuned settings as described in the methods sec- base [62] to correctly prepare inputs for Galileo [62],
| tion. Mask2Former |     | (M2F) | [8] | is a universal | segmenta- |       |               |       |         |     |           |         |     |
| ----------------- | --- | ----- | --- | -------------- | --------- | ----- | ------------- | ----- | ------- | --- | --------- | ------- | --- |
|                   |     |       |     |                |           | CROMA | [25], SoftCon | [68], | Prithvi |     | 2.0 [58], | DOFA-v1 |     |
tionarchitecturecapableofsemantic,instance,andpanop-
|     |     |     |     |     |     | [73],DeCUR[67],andSatlas[4]. |     |     |     | Thiswrapperallowedus |     |     |     |
| --- | --- | --- | --- | --- | --- | ---------------------------- | --- | --- | --- | -------------------- | --- | --- | --- |
tic segmentation depending on training configuration; we to (1) construct the band set expected by each model (ap-
| adapted M2F | with | a Swin-S |     | backbone | to handle the 8- |     |     |     |     |     |     |     |     |
| ----------- | ---- | -------- | --- | -------- | ---------------- | --- | --- | --- | --- | --- | --- | --- | --- |
plyingmaskwhereapplicable),(2)imputemissingchannels
channelRGBNbitemporalinputandtrainitonthepanoptic inamodel-consistentmanner,and(3)applyeachmodel’s
task,whichjointlypredictsindividualfieldinstances(things)
requirednormalizationorstandardizationusingitsoriginal
| andbackgroundlandcoverclasses(stuff). |     |     |     |     | NotethatSAM |                     |     |                                   |     |     |     |     |     |
| ------------------------------------- | --- | --- | --- | --- | ----------- | ------------------- | --- | --------------------------------- | --- | --- | --- | --- | --- |
|                                       |     |     |     |     |             | trainingstatistics. |     | ForTerraFM[14],weassignzerostoall |     |     |     |     |     |
andMask2Formerweretrainedwithoutpresence-onlylabel missingspectralbands. DINOv3[57]operatesexclusively
masking–adatapreprocessingstrategyusedbyallseman-
onRGBinputs,sotheFTWRGBbandsarepasseddirectly
ticbaselinesthatfiltersoutambiguousbackgroundregions without modification. For Clay [9] and TerraMind [32],
| in partially-labeled |     | countries. |     | The inability | to implement |     |     |     |     |     |     |     |     |
| -------------------- | --- | ---------- | --- | ------------- | ------------ | --- | --- | --- | --- | --- | --- | --- | --- |
whicharedesignedtohandlepartiallymissingspectralin-
thismaskingforinstancemodels(duetofundamentaldiffer-
|     |     |     |     |     |     | formation, | we provide | the | available | four-band |     | input | with |
| --- | --- | --- | --- | --- | --- | ---------- | ---------- | --- | --------- | --------- | --- | ----- | ---- |
encesinhowinstancesegmentationmodelshandletraining theappropriatenormalizationforeachmodel. Patch-level
objectives)meansthesemodelsfacedaharderoptimization
embeddingsareextractedfromeachpretrainedGFMinde-
landscape,beingpenalizedforpredictingfieldsinregions pendentlyforthetwotemporalwindowsdefinedinFTW.
thatmaycontainunlabeledfields.
| RGB-onlycomparisonwithDelineateAnything. |     |     |     |     | Topro- |     |     |     |     |     |     |     |     |
| ---------------------------------------- | --- | --- | --- | --- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
vide a fairer comparison with Delineate Anything (De- GFMfeaturefusionanddecoding. Thepatchembeddings
fromthetwotemporalwindowsarefusedbyfirstconcate-
lAny)[37],wetrainednewPRUEmodelswithEfficientNet-
|     |     |     |     |     |     | nating them | along | the feature | dimension |     | and | passing | the |
| --- | --- | --- | --- | --- | --- | ----------- | ----- | ----------- | --------- | --- | --- | ------- | --- |
B3(EF3)backbonesonRGB-onlydatafromasingletime
step.ThisresultedinobjectF1scoresof0.38±0.06(window resultthroughathree-layerMLP.Ourobjectiveistoevalu-
atetherepresentationalqualityofthefrozenGFMfeatures
A)and0.37±0.06(windowB),bothhigherthanDelAny’s
performancedespiteusingthesameRGB-onlyinput. This themselves. Acommonevaluationstrategyadoptedforthis
|     |     |     |     |     |     | objectiveislinearprobing. |     |     | However,wearguethatasingle |     |     |     |     |
| --- | --- | --- | --- | --- | --- | ------------------------- | --- | --- | -------------------------- | --- | --- | --- | --- |
demonstratesthattheperformancegapbetweenPRUEand
lineartransformationisoftentoolimitedtofullyassessthe
DelAnyisnotattributabletoFTW’sadditionalspectralchan-
nel(NIR),butratherreflectsdifferencesinmodeldesignand spatialandcontextualinformationencodedinthepretrained
|     |     |     |     |     |     | features. | Conversely, | using | a specialized |     | model | such | as  |
| --- | --- | --- | --- | --- | --- | --------- | ----------- | ----- | ------------- | --- | ----- | ---- | --- |
trainingdatadiversity.
U-Net[50]atthisstagewouldprimarilytesttheabilityof
SAM2evaluation. WeevaluatedSAM2inazero-shotset- thedecoderratherthantheunderlyingGFMembeddings. To
tingonthewindowARGBbands,resultinginapixelIoUof strikeabalancebetweenthesetwoextremes,weemploya
0.31andanobjectF1of0.07. SAM2isdesignedforvideo simpledecoderthatprovidesmoderateflexibilitythrougha
segmentationwhereitexpectscontinuousvideoframesin 3×3projectionlayer,tworesidualrefinementblocks,anda
whichobjectsmoveordeformslightlywithstrongappear- multi-scaleconvolutionalmodule,followedbypixel-shuffle
anceconsistency[47]. Thisdiffersfundamentallyfromthe upsampling. Table1reportstheresultsobtainedwithour
FTW setting, which provides two snapshots separated by convolutionaldecoder,andTable4providesthecomplemen-
monthsthatcapturesignificantphenologicalchanges. This tary1×1convolutionlinear-probingresults.
1

Image Clay Galileo DINOv3 Prithvi 2.0 TerraMind SoftCon Satlas DeCUR DOFA-v1 TerraFM CROMA
airtsuA
lizarB
muigleB
adnawR
aidobmaC
ecnarF
Figure4.PCAvisualizationoffrozenGFMencoderfeaturesforarepresentativesubsetofimageexamples.Thetop3principalcomponents
ofpatchembeddingsaredisplayedasRGBchannels.Clay(8×8patchsizes)andGalileo(4×4patchsize)capturefinerspatialstructure
andmoredistinctfieldboundariescomparedtomodelsusing16×16patches,demonstratinghowtokenizationgranularityaffectsfeature
qualityforsegmentationtasks.SeeTable4forquantitativeperformance.
Table4.GFMlinearprobingresultsusingalightweightdecoder(1×1convolution+bilinearupsampling),sortedbyobject-levelF1.Clay
(ViT-Large,8×8patches)andGalileo(ViT-Base,4×4patches)outperformotherGFMsthatusecoarser16×16patchsizes,duetothefiner
patchresolutionsaswellastechniquesintentionallydesignedtohandlemissingspectralbands.
|     | Pixel-level |     | Object-level |     |     |
| --- | ----------- | --- | ------------ | --- | --- |
Model Backbone
|                      | IoU↑ Prec↑ | Recall↑ Prec↑ | Recall↑ F1↑ | AP 0.5:0.95 ↑ | AP 0.5 ↑ |
| -------------------- | ---------- | ------------- | ----------- | ------------- | -------- |
| Clay ViT-Large       | 0.56 0.88  | 0.60 0.22     | 0.16 0.18   | 0.07          | 0.17     |
| Galileo ViT-Base     | 0.53 0.83  | 0.59 0.11     | 0.19 0.13   | 0.08          | 0.18     |
| DINOv3 ViT-Large     | 0.47 0.89  | 0.50 0.25     | 0.09 0.12   | 0.03          | 0.08     |
| Prithvi2.0 ViT-Large | 0.44 0.84  | 0.48 0.20     | 0.06 0.10   | 0.02          | 0.06     |
| TerraMind ViT-Base   | 0.44 0.85  | 0.47 0.19     | 0.07 0.10   | 0.02          | 0.06     |
| SoftCon ViT-Small    | 0.41 0.83  | 0.46 0.16     | 0.05 0.07   | 0.01          | 0.04     |
| Satlas Swin-Tiny     | 0.39 0.74  | 0.45 0.13     | 0.04 0.07   | 0.01          | 0.03     |
| DeCUR ViT-Small      | 0.42 0.80  | 0.46 0.15     | 0.04 0.07   | 0.01          | 0.03     |
| DOFA-v1 ViT-Large    | 0.39 0.77  | 0.44 0.14     | 0.04 0.06   | 0.01          | 0.03     |
| TerraFM ViT-Base     | 0.44 0.85  | 0.48 0.17     | 0.06 0.09   | 0.02          | 0.05     |
| CROMA ViT-Base       | 0.42 0.85  | 0.46 0.18     | 0.05 0.08   | 0.02          | 0.05     |
B.ExtendedResults accuracyandboundaryagreementacrossallmetrics.
Ourdesignchoiceisaresultofextensiveablations. Accuracy–throughputtrade-offacrossmodelconfigura-
Ta-
ble 5 shows that boundary weighting, loss function, and tions. Toexplicitlysummarizetheaccuracy–costtrade-off
targetedaugmentationshavethestrongestimpactonperfor- overallmodels,Figure5showstheParetofrontbetweenob-
mance. Moderateclassweights(ω ≈0.75)andlossessuch jectF1andthroughput. SupplementalTable5comparesour
asLogCoshDice,Tversky,andLocalTverskyconsistently methodologyagainstthebaselineusingthesamebackbone
improveobjectF1andpixelIoU,whilebrightness,scale,and (EfficientNet-B3): the“U-NetLogCoshDice0.75-weight
channel-shuffleaugmentationsprovideadditionalrobustness. Augs EN-B3” row shows that PRUE with an EF3 back-
LargerEfficientNetbackbonesslightlyenhanceresults,and bone achieves an object F1 of 0.43±0.07 and field IoU
combining these components in PRUE yields the highest of 0.74 ± 0.07, compared to PRUE-EF7 which achieves
2

Table5. AblationresultsforcontrolledexperimentsontheFTWtestset(excludingpresence-onlycountries)inwhicheachrowvaries
asingledesignchoice(dataaugmentations,classweighting,encoder,lossfunction,orarchitecture). Boldindicatesbestperformance,
underlineindicatessecond-best.
Performance Inputorder Brightness Scale Agree.
Category Ablation
ObjectF1↑ PixelIoU↑ F1|∆|↓ IoU|∆|↓ F1|∆|↓ IoU|∆|↓ F1|∆|↓ IoU|∆|↓ Avg↑
FTW-v1 0.39±0.08 0.68±0.08 0.07 0.11 0.04 0.05 0.17 0.08 0.93
Dataaugs Brightness 0.39±0.08 0.68±0.08 0.07 0.09 0.00 0.00 0.14 0.05 0.93
Dataaugs Resize 0.38±0.08 0.67±0.09 0.07 0.10 0.04 0.05 0.00 0.01 0.94
Dataaugs Brightness+Resize 0.38±0.08 0.66±0.09 0.06 0.10 0.03 0.03 0.00 0.02 0.95
Dataaugs Channelshuffle 0.39±0.07 0.68±0.09 0.00 0.00 0.05 0.05 0.18 0.09 0.94
Classweights ω=0.60 0.32±0.06 0.76±0.06 0.07 0.13 0.07 0.07 0.28 0.19 0.96
Classweights ω=0.65 0.36±0.06 0.76±0.06 0.07 0.11 0.07 0.07 0.30 0.17 0.96
Classweights ω=0.70 0.40±0.06 0.75±0.06 0.08 0.12 0.08 0.07 0.30 0.14 0.95
Classweights ω=0.75 0.42±0.06 0.74±0.07 0.08 0.11 0.07 0.07 0.29 0.13 0.95
Classweights ω=0.80 0.42±0.07 0.73±0.06 0.09 0.12 0.07 0.07 0.23 0.11 0.95
Classweights ω=0.85 0.41±0.07 0.70±0.08 0.08 0.10 0.05 0.06 0.17 0.08 0.96
Encoder EfficientNet-B4 0.40±0.07 0.69±0.09 0.07 0.09 0.04 0.05 0.15 0.06 0.93
Encoder EfficientNet-B5 0.41±0.07 0.70±0.08 0.07 0.11 0.04 0.05 0.16 0.10 0.93
Encoder EfficientNet-B6 0.41±0.07 0.70±0.08 0.07 0.11 0.04 0.05 0.18 0.14 0.94
Encoder EfficientNet-B7 0.42±0.07 0.71±0.08 0.07 0.09 0.04 0.04 0.21 0.09 0.94
Encoder MiT-B2 0.39±0.08 0.67±0.09 0.08 0.10 0.02 0.03 0.13 0.05 0.95
Encoder MiT-B3 0.39±0.08 0.67±0.09 0.08 0.10 0.02 0.03 0.13 0.05 0.95
Encoder MiT-B4 0.39±0.08 0.68±0.09 0.07 0.09 0.02 0.03 0.16 0.06 0.94
Encoder MiT-B5 0.38±0.08 0.67±0.09 0.08 0.10 0.02 0.03 0.12 0.02 0.95
Encoder ResNet-18 0.35±0.07 0.67±0.09 0.08 0.11 0.04 0.05 0.14 0.05 0.93
Encoder VGG13-BN 0.38±0.07 0.69±0.08 0.08 0.10 0.04 0.05 0.27 0.20 0.94
Learningrate 0.0001 0.34±0.07 0.65±0.09 0.05 0.07 0.03 0.04 0.15 0.08 0.89
Learningrate 0.0003 0.37±0.07 0.66±0.09 0.06 0.08 0.04 0.04 0.17 0.10 0.90
Learningrate 0.003 0.39±0.08 0.68±0.09 0.08 0.10 0.06 0.08 0.17 0.08 0.95
Learningrate 0.01 0.39±0.08 0.68±0.09 0.08 0.11 0.06 0.06 0.18 0.08 0.95
Learningrate 0.03 0.37±0.08 0.67±0.09 0.09 0.10 0.10 0.13 0.16 0.07 0.94
Lossfunction CE(w/EdgeAgreement) 0.39±0.08 0.68±0.09 0.07 0.10 0.04 0.05 0.15 0.07 0.94
Lossfunction CE+Dice 0.41±0.07 0.70±0.08 0.07 0.10 0.04 0.05 0.20 0.08 0.93
Lossfunction CE+Dice(noclassweights) 0.38±0.07 0.77±0.06 0.08 0.11 0.06 0.05 0.29 0.15 0.95
Lossfunction CE+FTNMT 0.41±0.07 0.70±0.08 0.08 0.11 0.04 0.05 0.21 0.08 0.94
Lossfunction CE(noclassweights) 0.24±0.06 0.77±0.06 0.05 0.11 0.05 0.06 0.15 0.14 0.96
Lossfunction Dice 0.42±0.07 0.76±0.07 0.08 0.13 0.06 0.06 0.31 0.16 0.96
Lossfunction Dice(w/EdgeAgreement) 0.42±0.07 0.77±0.06 0.08 0.13 0.07 0.07 0.32 0.17 0.95
Lossfunction Focal 0.18±0.05 0.75±0.06 0.04 0.10 0.04 0.06 0.15 0.21 0.96
Lossfunction FTNMT 0.38±0.06 0.79±0.06 0.10 0.14 0.06 0.07 0.29 0.17 0.93
Lossfunction LocalTversky 0.45±0.07 0.74±0.07 0.10 0.13 0.05 0.05 0.31 0.16 0.94
Lossfunction LogCoshDice 0.44±0.07 0.77±0.06 0.09 0.13 0.06 0.06 0.33 0.20 0.94
Lossfunction LogCoshDice+CE 0.39±0.08 0.68±0.08 0.08 0.10 0.04 0.05 0.14 0.06 0.93
Lossfunction Tversky 0.43±0.07 0.76±0.06 0.09 0.12 0.06 0.05 0.34 0.15 0.95
Lossfunction Tversky+CE 0.41±0.07 0.71±0.08 0.07 0.10 0.05 0.06 0.20 0.10 0.92
Architecture FCN 0.14±0.03 0.60±0.08 0.04 0.09 0.03 0.07 0.10 0.00 0.99
Architecture FCSiam 0.40±0.07 0.69±0.08 0.00 0.00 0.05 0.06 0.23 0.10 0.92
Architecture UNETR 0.37±0.07 0.69±0.08 0.08 0.10 0.04 0.04 0.27 0.18 0.94
Architecture UPerNet 0.34±0.08 0.64±0.10 0.07 0.09 0.03 0.04 0.13 0.04 0.91
Combination FCSiamCombo 0.44±0.07 0.75±0.07 0.00 0.00 0.04 0.04 0.05 0.02 0.94
Combination U-NetLogCoshDiceAugsEN-B3 0.42±0.07 0.74±0.07 0.00 0.00 0.00 0.00 0.01 0.01 0.94
Combination U-NetLogCoshDice0.75-weightAugsEN-B3 0.43±0.07 0.74±0.07 0.00 0.00 0.00 0.00 0.02 0.00 0.94
Combination U-NetLogCoshDice0.75-weightAugsEN-B5 0.46±0.07 0.75±0.07 0.00 0.00 0.00 0.00 0.01 0.01 0.94
Combination PRUE 0.47±0.07 0.76±0.08 0.00 0.00 0.00 0.00 0.01 0.01 0.95
0.47±0.07 and 0.76±0.08, respectively. PRUE-EF3 is Full fine-tuning of Clay. To assess whether the perfor-
aParetoimprovementovertheFTWbaseline, meaningit mance gap between GFMs and PRUE could be closed
achieveshigheraccuracywithoutsacrificingthroughput.The with end-to-end training, we fully fine-tuned the best-
PRUEfamilyformsasubstantiallystrongerParetofrontier performing GFM, Clay, selecting the learning rate from
thananypriormodelandshowsaclearthroughputvs.perfor- {1,3}×10{−5,−4,−3}basedonthebestobjectF1ontheval-
mancetrade-off. Asapracticalreference,anEF3backbone idationset. Fullfine-tuningincreasedClay’sobjectF1from
canprocesstheentirelandareaoftheEarthinapproximately 0.36to0.42(seeFigure5)andpixelIoUfrom0.67to0.73.
66hoursonasingleV100GPU,whileEF7wouldrequire Whiletheserepresentmeaningfulimprovementsoverfrozen-
approximately134hours. encoderdecoding,bothmetricsremainbelowPRUE(0.47
objectF1,0.76pixelIoU),andClay’sthroughputissubstan-
3

|     |     |     |     |     |     |     | than the      | FTW baseline             |           | across     | all overlap       | window    | sizes,        |
| --- | --- | --- | --- | --- | --- | --- | ------------- | ------------------------ | --------- | ---------- | ----------------- | --------- | ------------- |
|     |     |     |     |     |     |     | demonstrating | improvements             |           | in         | both              | intrinsic | translation   |
|     |     |     |     |     |     |     | robustness,   | anarchitecturalproperty, |           |            | andreducedcontext |           |               |
|     |     |     |     |     |     |     | dependency,   | a learned                | behavior. |            | Consistency       |           | varied sub-   |
|     |     |     |     |     |     |     | stantially    | across countries         |           | and        | correlates        | with      | test set dif- |
|     |     |     |     |     |     |     | ficulty.      | For example,             | our       | best model |                   | shows     | > 40% pixel   |
disagreementinKenyaat224-pixelcontextshifts,compared
|     |     |     |     |     |     |     | to < 20% | in Switzerland. |     | This | suggests | that | consistency |
| --- | --- | --- | --- | --- | --- | --- | -------- | --------------- | --- | ---- | -------- | ---- | ----------- |
metricsmayserveasanout-of-distributiondetectionsignal,
wherelowconsistencyduringinferencecouldindicatethat
themodelisoperatingondatathatdiffersfromitstraining
distribution,andmaythusrequirehumanreviewormodel
retraining.
Figure5.ParetofrontbetweenObjectF1andThroughputforall
|     |     |     |     |     |     |     | Consistencyasareliabilitysignal. |     |     |     | Toquantifytherelation- |     |     |
| --- | --- | --- | --- | --- | --- | --- | -------------------------------- | --- | --- | --- | ---------------------- | --- | --- |
Table1models,includingPRUE-EF(B3/B5)throughput.PRUE-
|     |     |     |     |     |     |     | shipbetweenconsistencyandperformance, |     |     |     |     | wecomputed |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------------------------- | --- | --- | --- | --- | ---------- | --- |
EF3isaParetoimprovementovertheFTWbaseline.ThePRUE
theaverageconsistencyovertestsamplespercountryand
familyformsasubstantiallystrongerParetofrontierthananyprior
|     |     |     |     |     |     |     | examineditscorrelationwithobjectF1(SeeFigure7). |     |     |     |     |     | We  |
| --- | --- | --- | --- | --- | --- | --- | ----------------------------------------------- | --- | --- | --- | --- | --- | --- |
model.
|     |     |     |     |     |     |     | observe | that consistency |     | is weakly | correlated |     | with perfor- |
| --- | --- | --- | --- | --- | --- | --- | ------- | ---------------- | --- | --------- | ---------- | --- | ------------ |
mance: theFTWbaselineyieldsR2=0.48andPRUE-EF7
| tiallylower(Clay: |     | 11km2/svs.PRUE:307km2/s). |     |     |     | This |                |                                        |     |     |     |     |     |
| ----------------- | --- | ------------------------- | --- | --- | --- | ---- | -------------- | -------------------------------------- | --- | --- | --- | --- | --- |
|                   |     |                           |     |     |     |      | yieldsR2=0.30. | Thisindicatesthatconsistencymetricsex- |     |     |     |     |     |
confirmsthattheperformancegapisnotsolelyattributable
plainsomedropsinout-of-distribution(OOD)performance
tothefrozen-encoderevaluationprotocol,butreflectsfun-
|     |     |     |     |     |     |     | but not | all—a model | can | have | high consistency |     | but poor |
| --- | --- | --- | --- | --- | --- | --- | ------- | ----------- | --- | ---- | ---------------- | --- | -------- |
damentaldifferencesinspatialresolutionandarchitectural
|     |     |     |     |     |     |     | performance. | Consistencymetricsmayserveasonesignal |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------ | ------------------------------------- | --- | --- | --- | --- | --- |
suitabilityforfieldboundarydelineation.
amongseveralforOODdetection,butshouldnotbeused
|     |     |     |     |     |     |     | asasoleindicatorofmodelreliability. |     |     |     |     | Indeployment,low |     |
| --- | --- | --- | --- | --- | --- | --- | ----------------------------------- | --- | --- | --- | --- | ---------------- | --- |
1Dconvolutiondidnotcapturefieldboundarycomplex-
ity. AcrossallGFMexperiments,weobservedthatdecoding consistencyscoresaremostusefulforflaggingregionsthat
usingasingle1×1convolutionfollowedbybilinearupsam- exhibitgriddingartifactsandspatialpredictioninstability,
warrantinghumanreview.
plingproducedcoarse,low-fidelityboundariescomparedto
| ourconvolutionaldecoderhead. |     |     | Table4showstheselinear |     |     |     |     |     |     |     |     |     |     |
| ---------------------------- | --- | --- | ---------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
C.Per-CountryEvaluation
| probingresultsforallGFMs.                 |       |             | AmongtheGFMs,Clayand |          |            |         |                                                   |                  |     |         |     |         |           |
| ----------------------------------------- | ----- | ----------- | -------------------- | -------- | ---------- | ------- | ------------------------------------------------- | ---------------- | --- | ------- | --- | ------- | --------- |
| Galileoexhibitnotablystrongerperformance. |       |             |                      |          | Bothmodels |         |                                                   |                  |     |         |     |         |           |
|                                           |       |             |                      |          |            |         | We report                                         | full per-country |     | metrics | for | all FTW | countries |
| use smaller                               | token | patch sizes | (8×8                 | for Clay | and        | 4×4 for |                                                   |                  |     |         |     |         |           |
|                                           |       |             |                      |          |            |         | withcompletelabels,excludingpresence-onlyregions. |                  |     |         |     |         | For       |
Galileo),comparedtomostoftheotherevaluatedmodels
eachcountry,weprovidepixelIoUandobject-levelF1.
| thatoperateatapatchsizeof16×16. |     |     |     | Thesefinerpatchreso- |     |     |     |     |     |     |     |     |     |
| ------------------------------- | --- | --- | --- | -------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Torepresentthemajormodelfamilies,weevaluatedthe
lutionsproducehigher-granularityspatialfeaturesthatalign
strongest-performingmodelfromeachcategory,asidentified
| more naturally                                   | with | field-level | segmentation. |     | In  | addition, |                         |     |     |                           |     |     |     |
| ------------------------------------------------ | ---- | ----------- | ------------- | --- | --- | --------- | ----------------------- | --- | --- | ------------------------- | --- | --- | --- |
|                                                  |      |             |               |     |     |           | inTable1ofthemainpaper: |     |     | theFTWbaselineforsemantic |     |     |     |
| bothmodelsrobustlyhandlemissingspectralchannels: |      |             |               |     |     | Clay      |                         |     |     |                           |     |     |     |
segmentation[34],Mask2Former(M2F)withaSwinSmall
| isexplicitlytrainedwithmissing-bandaugmentation, |     |     |     |     |     | and |     |     |     |     |     |     |     |
| ------------------------------------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
backboneforpanopticsegmentation[8],Claywithourcon-
Galileoincorporatesconsistentbandmaskingandnormal-
volutionaldecoderforgeospatialfoundationmodels[9],and
izationstatisticsforpartiallyobservedinputs.
PRUE.Allmetricswerecomputedoneachcountry’sofficial
FTWtestsplit,followingtheevaluationprotocoldescribed
| Consistency | varies | by geography |     | and overlap |     | window |     |     |     |     |     |     |     |
| ----------- | ------ | ------------ | --- | ----------- | --- | ------ | --- | --- | --- | --- | --- | --- | --- |
in§3.4.
size. Figure6illustrateshowconsistencychangeswiththe
sizeoftheoverlappingregionbetweenthefourcornercrops Acrossregions,theresultsshowconsistentpatterns. As
showninTable6,PRUEconsistentlyachievedthehighest
| used to measure |     | translation | sensitivity. | Whenthe |     | overlap |     |     |     |     |     |     |     |
| --------------- | --- | ----------- | ------------ | ------- | --- | ------- | --- | --- | --- | --- | --- | --- | --- |
orsecond-highestPixelIoUandObjectF1acrossnearlyall
| window | is large, | approaching | the patch | size, | consistency |     |            |            |              |     |     |        |              |
| ------ | --------- | ----------- | --------- | ----- | ----------- | --- | ---------- | ---------- | ------------ | --- | --- | ------ | ------------ |
|        |           |             |           |       |             |     | tested FTW | countries, | highlighting |     | its | robust | and reliable |
primarilyreflectsintrinsictranslationsensitivity,whichcap-
|     |     |     |     |     |     |     | performance | across | diverse | geographies |     | and | agricultural |
| --- | --- | --- | --- | --- | --- | --- | ----------- | ------ | ------- | ----------- | --- | --- | ------------ |
turesthemodel’sarchitecturaltendencytoproducedifferent
| outputs for | identical | content | at different | spatial |     | positions. | systems. |     |     |     |     |     |     |
| ----------- | --------- | ------- | ------------ | ------- | --- | ---------- | -------- | --- | --- | --- | --- | --- | --- |
Whentheoverlapwindowissmall,consistencyalsoreflects
D.MosaickingandLargeScaleInference
contextdependency,indicatinghowmuchthemodel’spre-
dictionsvarybasedonsurroundingimagecontent. DeployingPRUEatthecountryscalerequiresconstructing
PRUE achieved two to three times higher consistency spatiallycomplete,cloud-freeSentinel-2compositesfrom
4

Figure6.Consistencydependenceonoverlapsize.ConsistencyscoresasafunctionofoverlapwindowsizefortheFTWbaseline(left)
andPRUE(right).Largeroverlapsmeasureintrinsictranslationsensitivity,whilesmalleroverlapsadditionallycapturecontextdependency.
Shown: top5andbottom5countriesbymeanconsistencyforeachmodel. PRUEachieveshigherconsistencyacrossallcountriesand
overlapsizes,withparticularlystrongimprovementsinchallengingregions.Hardcountries(Kenya,India)showlowconsistencyevenat
largeoverlaps,whilewell-representedregions(Finland,Netherlands)havehighconsistency,suggestingconsistencymetricscanidentify
out-of-distributionsamplesatinferencetime.
Table 6. Per-country performance comparison for the top-performing models of each architecture family: FTW baseline (semantic),
Mask2FormerwithSwin-S(instance/panoptic),Clay(frozenGFM),Clay-FT(finetunedGFM),andPRUE(ouroptimizedsemanticmodel).
Boldindicatesbestperformancepercountry.PRUEachievesthehighestorsecond-highestpixelIoUscoresacrossnearlyallcountries.All
modelsevaluatedusingtheprotocolinSection3.4onpresence/absencelabeledcountriesonly.
|     |     | PixelIoU |     | ObjectF1 |     |
| --- | --- | -------- | --- | -------- | --- |
Country
|            | FTW M2F   | Clay Clay-FT | PRUE FTW  | M2F Clay  | Clay-FT PRUE |
| ---------- | --------- | ------------ | --------- | --------- | ------------ |
| Austria    | 0.71 0.74 | 0.69 0.76    | 0.78 0.41 | 0.38 0.39 | 0.47 0.50    |
| Belgium    | 0.75 0.78 | 0.73 0.79    | 0.82 0.57 | 0.57 0.57 | 0.62 0.66    |
| Cambodia   | 0.40 0.19 | 0.27 0.40    | 0.66 0.19 | 0.10 0.09 | 0.19 0.36    |
| Corsica    | 0.45 0.52 | 0.47 0.51    | 0.51 0.18 | 0.24 0.18 | 0.22 0.24    |
| Croatia    | 0.67 0.70 | 0.64 0.71    | 0.77 0.28 | 0.34 0.25 | 0.33 0.45    |
| Denmark    | 0.83 0.57 | 0.83 0.86    | 0.86 0.52 | 0.24 0.51 | 0.58 0.65    |
| Estonia    | 0.80 0.82 | 0.80 0.83    | 0.84 0.44 | 0.54 0.43 | 0.48 0.54    |
| Finland    | 0.83 0.85 | 0.81 0.85    | 0.87 0.55 | 0.54 0.53 | 0.59 0.64    |
| France     | 0.79 0.80 | 0.78 0.81    | 0.83 0.55 | 0.55 0.54 | 0.58 0.63    |
| Germany    | 0.79 0.77 | 0.78 0.80    | 0.79 0.41 | 0.44 0.39 | 0.42 0.47    |
| Latvia     | 0.81 0.84 | 0.81 0.84    | 0.85 0.44 | 0.54 0.44 | 0.49 0.56    |
| Lithuania  | 0.74 0.78 | 0.74 0.78    | 0.79 0.39 | 0.48 0.38 | 0.45 0.50    |
| Luxembourg | 0.79 0.80 | 0.76 0.82    | 0.85 0.49 | 0.37 0.46 | 0.53 0.56    |
Netherlands 0.75 0.78 0.74 0.81 0.81 0.48 0.51 0.48 0.54 0.57
| Portugal | 0.12 0.21 | 0.23 0.37 | 0.10 0.03 | 0.08 0.04 | 0.07 0.03 |
| -------- | --------- | --------- | --------- | --------- | --------- |
| Slovakia | 0.92 0.91 | 0.92 0.94 | 0.94 0.53 | 0.61 0.53 | 0.58 0.65 |
| Slovenia | 0.58 0.66 | 0.55 0.65 | 0.68 0.24 | 0.28 0.20 | 0.27 0.33 |
SouthAfrica 0.80 0.80 0.78 0.81 0.82 0.53 0.56 0.50 0.56 0.54
| Spain   | 0.73 0.70 | 0.69 0.75 | 0.83 0.24 | 0.26 0.21 | 0.26 0.33 |
| ------- | --------- | --------- | --------- | --------- | --------- |
| Sweden  | 0.81 0.82 | 0.80 0.84 | 0.85 0.45 | 0.51 0.44 | 0.50 0.55 |
| Vietnam | 0.46 0.30 | 0.31 0.46 | 0.67 0.15 | 0.09 0.08 | 0.15 0.22 |
irregularly sampled, partially cloudy observations. This inglatitude-basedseasonheuristics,greedysceneselection
sectiondetailsouroperationalpipelineforsceneselection, tominimizeredundancy,andcloud-optimizeddataformats
temporalcompositing,andimageryqualitymosaickingus- enablingscalableparallelinference.
5

2.HarvestSeasonHeuristicAlgorithm
functionHARVESTSEASONDOY(latitude)
abs_lat←|latitude|
|     |     |     |     | ifabs_lat>45then |     |     | Highlatitudes |
| --- | --- | --- | --- | ---------------- | --- | --- | ------------- |
return(244,304)iflatitude>0else(60,151)
|     |     |     |     | elseif20<abs_lat≤45then |     |     | Mid-latitudes |
| --- | --- | --- | --- | ----------------------- | --- | --- | ------------- |
return(213,304)iflatitude>0else(32,120)
|     |     |     |     | elseif5<abs_lat≤20then |     |     | Subtropics |
| --- | --- | --- | --- | ---------------------- | --- | --- | ---------- |
return(274,365)iflatitude>0else(91,181)
Equatorial|lat|≤5
else
return(182,243)
endif
endfunction
|     | Country-levelconsistencyvs. | objectF1. |     |     |     |     |     |
| --- | --------------------------- | --------- | --- | --- | --- | --- | --- |
Figure7. Eachpoint Feature Selection via Greedy Search Optimal scene se-
representsoneFTWcountry.Dashedlinesshowlinearfitsforthe
|     |     |     |     | lection maximizes | spatial | coverage while | minimizing re- |
| --- | --- | --- | --- | ----------------- | ------- | -------------- | -------------- |
FTWbaseline(orange,R²=0.48)andPRUE-EF7(blue,R²=0.30).
|     |     |     |     | dundancy. Input | scenes were | pre-filtered | by cloud cover |
| --- | --- | --- | --- | --------------- | ----------- | ------------ | -------------- |
Consistencyisweaklycorrelatedwithperformance,suggestingit
mayserveasapartialout-of-distributionsignalbutnotareliable (<75%)andSceneClassificationLayer(SCL)qualityflags
solepredictorofmodelaccuracy. (excluding classes 1, 3, 7, 8, 9, 10, and nodata=0). The
greedyapproachprioritizesscenesthatcontributevalidob-
Latitude-BasedSeasonHeuristics. Plantingandharvest servationstounderrepresentedspatialregionswithinatile,
enablingtheuseofrelaxedscene-levelcloudcoverthresh-
| windows | were estimated | using latitude-dependent | day-of- |     |     |     |     |
| ------- | -------------- | ------------------------ | ------- | --- | --- | --- | --- |
year(DOY)rangesthataccountforhemisphericdifferences olds. Scenes with high overall cloud cover may still con-
tainsubstantialcloud-freeareasthatfillcriticalgapsinthe
| and climatic | zones. This | heuristic approach | provides rea- |     |     |     |     |
| ------------ | ----------- | ------------------ | ------------- | --- | --- | --- | --- |
composite,therebyimprovingspatialcompletenesswithout
sonabletemporalconstraintsforsceneselection,although
weacknowledgethatdateselectioncouldbesubstantially requiringadditionalacquisitions.
improvedbyintegratingadditionalinformationongeograph-
icalvariationincropgrowthcycles,suchascropcalendars,
phenologicalmodels,orground-basedinformationonlocal Cloud-OptimizedGeoTIFFStorage. ForeachSentinel-
plantingandharvestperiods. 2 grid tile and temporal period, median composites were
|     |     |     |     | constructedfromtheselectedscenes. |     | Spectralbands(B02, |     |
| --- | --- | --- | --- | --------------------------------- | --- | ------------------ | --- |
1.PlantingSeasonHeuristicAlgorithm
B03,B04,B08)atnative10mresolutionweremaskedusing
|     |     |     |     | SCLupsampledfrom20mvianearest-neighbor. |     |     | Temporal |
| --- | --- | --- | --- | --------------------------------------- | --- | --- | -------- |
functionPLANTINGSEASONDOY(latitude)
medianswerecomputedalongsidevalidobservationcounts.
abs_lat←|latitude|
ifabs_lat>45then Highlatitudes Outputswerestoredasfloat32Cloud-OptimizedGeoTIFFs
return(91,151)iflatitude>0else(274,334) with1024×1024internaltiling.
|     | elseif20<abs_lat≤45then |     | Mid-latitudes |     |     |     |     |
| --- | ----------------------- | --- | ------------- | --- | --- | --- | --- |
return(60,120)iflatitude>0else(244,334)
|     | elseif5<abs_lat≤20then |     | Subtropics |     |     |     |     |
| --- | ---------------------- | --- | ---------- | --- | --- | --- | --- |
GTI-BasedReprojection,ResamplingandZarrAssem-
return(121,212)iflatitude>0else(305,365)
bly. GDALTileIndex(GTI)1filesprovidevirtualmosaics
|     | else | Equatorial|lat|≤5 |     |     |     |     |     |
| --- | ---- | ----------------- | --- | --- | --- | --- | --- |
DuringZarr2construction,
|     | return(60,121) |     |     | referencingdistributedCOGs. |     |     |     |
| --- | -------------- | --- | --- | --------------------------- | --- | --- | --- |
endif
endfunction
1https://gdal.org/en/latest/drivers/raster/gti.
html
2https://zarr.readthedocs.io/en/stable/
6

F.ChangeDetectionAnalysis
3.GreedySceneSelectionAlgorithm
ThechangedetectionvisualizationspresentedinFigure8
| function |     | SELECTSCENESGREEDY(valid_mask, |     |     |     |     |     |     |     |
| -------- | --- | ------------------------------ | --- | --- | --- | --- | --- | --- | --- |
areintendedtodemonstratehowmulti-yearmapsproduced
target_coverage=5,max_scenes=10)
|     |     |     |     |     | byPRUEcansignalprobablefield-scalechanges. |     |     | Weleave |     |
| --- | --- | --- | --- | --- | ------------------------------------------ | --- | --- | ------- | --- |
Input:valid_mask-booleanarrayofshape(T,H,W)
moredetailedstudiesofchangedetectiontofuturework.
|     | coverage_depth←0 | H×W |     |     |     |     |     |     |     |
| --- | ---------------- | --- | --- | --- | --- | --- | --- | --- | --- |
Themethodcomputestheabsolutedifferencebetweense-
|     | remaining←{0,1,...,T |     | −1} |     |     |     |     |     |     |
| --- | -------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
selected←[] manticlogitsfromconsecutiveyears,appliesmin–maxnor-
fori=1→max_scenesdo malization,andthresholdsat0.5toobtainabinarychange
best_idx←NULL mask. Forwell-calibratedmodels,thisthresholdhighlights
best_gain←−1
|     |     |     |     |     | high-confidencesemanticshifts. |     | Visualinspectionconfirms |     |     |
| --- | --- | --- | --- | --- | ------------------------------ | --- | ------------------------ | --- | --- |
foreachidx∈remainingdo thatevensmall-scaledetectedchangesareconsistentwith
|     | undercovered | ←   | (coverage_depth | <   |     |     |     |     |     |
| --- | ------------ | --- | --------------- | --- | --- | --- | --- | --- | --- |
cultivationshifts(e.g.,fieldsappearingordisappearingbe-
target_coverage)
|     |     |     |     |     | tween years), | and that | artifacts from | misregistration | and |
| --- | --- | --- | --- | --- | ------------- | -------- | -------------- | --------------- | --- |
new_valid←valid_mask[idx]∧undercovered
|     |       |                    |     |     | atmosphericvariationareuncommon. |     |     | Wenotethatlacking |     |
| --- | ----- | ------------------ | --- | --- | -------------------------------- | --- | --- | ----------------- | --- |
|     | gain← | (cid:80) new_valid |     |     |                                  |     |     |                   |     |
ifgain>best_gainthen groundtruthchangelabels,wereliedonphoto-interpretation
ofhigh-resolutionbasemapimageryratherthanquantitative
best_gain←gain
accuracyassessments,whichweleaveasfuturework.
best_idx←idx
endif
|     | endfor |     |     |     | G.FutureDirections |     |     |     |     |
| --- | ------ | --- | --- | --- | ------------------ | --- | --- | --- | --- |
ifbest_gain=0then
break Several directions remain open for future work, includ-
|     | endif |     |     |     | ing: (1)comprehensiveobject-level[46]andthematicaccu- |     |     |     |     |
| --- | ----- | --- | --- | --- | ----------------------------------------------------- | --- | --- | --- | --- |
selected.APPEND(time[best_idx]) racy [43] assessments on country-scale deployments with
coverage_depth ← coverage_depth + independentreferencedata;(2)exploringdeploymentmet-
valid_mask[best_idx]
ricsasout-of-distributiondetectors(ourpreliminaryfindings
remaining.REMOVE(best_idx)
suggestlowconsistencyscoresmaysignalwhenmodelsen-
endfor
counterdissimilardata);(3)incorporatingsuper-resolution
returnselected
approachesfordelineatingsmallholderfieldsfromtempo-
endfunction
|     |     |     |     |     | ral stacks | of imagery; | and (4) systematic | post-processing |     |
| --- | --- | --- | --- | --- | ---------- | ----------- | ------------------ | --------------- | --- |
gdal_translate
|     |     | performs windowed |     | extraction with |     |     |     |     |     |
| --- | --- | ----------------- | --- | --------------- | --- | --- | --- | --- | --- |
ablations(morphologicaloperations,topologicalcleaning,
on-the-flyreprojectiontoEPSG:3857(WebMercator)using
confidencethresholding)tofurtherimproveboundaryquality
nearestneighborresamplingandwritestoatemporarylocal
inchallengingregions.
| file. Reprojection | to  | EPSG:3857 | prior to | inference elimi- |     |     |     |     |     |
| ------------------ | --- | --------- | -------- | ---------------- | --- | --- | --- | --- | --- |
natestheneedfordownstreampipelinestoperformcoordi-
| nate transformations, |                 | and enables | global   | non-overlapping |     |     |     |     |     |
| --------------------- | --------------- | ----------- | -------- | --------------- | --- | --- | --- | --- | --- |
| results without       | tile artifacts. | The         | windowed | data is then    |     |     |     |     |     |
loadedandinserteddirectlyintotheZarrstore,withspatial
Ray3.
| partitions | written to | Zarr v3 arrays | in parallel | with |     |     |     |     |     |
| ---------- | ---------- | -------------- | ----------- | ---- | --- | --- | --- | --- | --- |
Thiscreatesarobustandscalableapproachtobuildinglarge-
scale,cloud-optimizeddatareadyfordownstreamanalysis.
E.LargeScaleInferenceVisualSamples
ToqualitativelyassessPRUE’sperformanceatoperational
scale,wepresentinFigure8representativevisualsamples
fromeachcountry-scaledeploymentdescribedinSection5.
Thesesamplesillustratemodelbehavioracrossdiverseagri-
| cultural                                    | systems (Japan, | Mexico, | Rwanda, | South Africa, |     |     |     |     |     |
| ------------------------------------------- | --------------- | ------- | ------- | ------------- | --- | --- | --- | --- | --- |
| Switzerland)spanningawiderangeoffieldsizes. |                 |         |         | Thevi-        |     |     |     |     |     |
sualizationsdemonstratePRUE’sabilitytomaintainspatial
| consistency  | across large | extents | under atmospheric | varia-     |     |     |     |     |     |
| ------------ | ------------ | ------- | ----------------- | ---------- | --- | --- | --- | --- | --- |
| tion. Visual | inspection   | reveals | spatial patterns  | of success |     |     |     |     |     |
andfailuremodesthatinformfutureimprovements.
3https://docs.ray.io/en/latest/index.html
7

2023 Predictions + 2023 Planet Imagery + 2024 Planet Imagery +
2023 Predictions
2024 change 2024 change 2024 change
Japan
Field class probability
Mexico
Field class probability
Rwanda
Field class probability
South Africa
Field class probability
Switzerland
Field class probability
Figure8.Examplevisualsovereachcountryinourlarge-scaleinferenceset(Japan,Mexico,Rwanda,SouthAfrica,andSwitzerland).For
eachregionofinterest,weshow:(1)thePRUEfieldboundarypredictionsfrom2023;(2)the2023predictionswithchangesdetectedin
2024highlightedinbrightgreen;(3)Planetmonthlybasemapsfrom2023withthevectorizedchangemaskoutlinedinred;and(4)Planet
monthlybasemapsfrom2024withthevectorizedchangemaskoutlinedinred.Thebasemapsshownforeachpairarefromthesamemonth
inconsecutiveyears.
8