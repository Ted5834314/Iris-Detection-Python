*When running the script on windows terminal I need to close the previous graphs to view later graphs. I don't know if this would be necessary in other environments.


Design:

Localization：

First locate the pupil center through Canny edge detection and Hough transformation as the center of our localized graph, then take the 240x240
area around the center as our ROI(this works better than the method in the paper). Then detect the pupil again in the ROI to get a more accurate measure of the pupil center and radius. Turn pixels inside the pupil and around the pupil boundary to black to better detect outer boundary of iris. Create two sets of edges(one for fuzzy edges, one for clear edges) by Canny detection and individually detect outer boundary using Hough transformation. Pick the one whose center is closer to the pupil center to be our final outer boundary.

Normalization:

Following the same technique described in the paper, map each pixel in the 64x512 normalized image to the corresponding position of the original image

Enhancement:

Following the same technique described in the paper. First blur by taking mean of 8x8 blocks, then resize using bicubic interpolation, then do histogram equalization to each 32x32 blocks

Feature Extraction:

Following the same formula described in the papaer, use a kernel of size 9x9 to do convolution to the upper 48x512 region of the enhanced image.
Then compute mean and absolute deviation by 8x8 blocks. Use two different sets of sigmax and sigmay, setting f=1/sigmax. Glue these two feature vectors to get
final feature vector of length 1536

Iris Matching(Model Training):

The 7 images of different shift angles are created so in each class there are 3x7=21 images in the training set. The min distance idea in the paper doesn't work well,
so we just view these 21 images as one class and use ordinary distance functions.
Use LDA to do dimension reduction, then train nearest center models.

Performance Evaluation:

CRR is evaluated for 3 different distances by applying the models directly. Cos similarity has the best performance.
To evaluate FMR, the label of test set is randomly shuffled multiple times since we need unmatching labels.
Thresholds are calculated for FMR=[0.01,0.05,0.1,0.15,0.2,0.25,0.3]
Apply these thresholds to calculate FNMR.
Tables and graphs are drawn accordingly.


Limitation:

Even if two cycles are computed, we may still fail to localize accurately for some outer boundaries. Also sometimes the more distant cycle is the real outer boundary, and the algorithm chooses the closer one. Many images have heavy eyelashes even in the ROI(48x512) which would influence the localization and reduce the model performance.

Improvement：

For outer boundary detection, we may need to enhace the image first to make some fuzzy boundaries clearer so that we can use only one edge detection criterion to detect edges not two. For eyelash issue, we may need to use histogram to detect and remove them.



Peer Evaluation Form:

xw2747: IrisLocalization IrisNormalization ImageEnhancement IrisRecognition(main function)

hj2593: FeatureExtraction IrisMatching PerformanceEvaluation

by2325: FeatureExtraction IrisMatching PerformanceEvaluation


