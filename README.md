# 1 Introduction

This is a biomedical data set built by Dr. Henrique da Mota during a medical residence period in the Group of Applied Research in Orthopaedics (GARO) of the Centre Médico-Chirurgical de Réadaptation des Massues in Lyon, France. In this project, I will apply both unsupervised and supervised analysis to the task of classifying whether a patient is normal or abnormal. In some other datasets there are three/four classes but we are working with the simpler dataset.

This dataset is public and can be found at: https://www.kaggle.com/datasets/caesarlupum/vertebralcolumndataset or https://archive.ics.uci.edu/dataset/212/vertebral+column

# 2 Data Pre-processing and Exploratory Data Analysis

The dataset had no column names, so I used the metadata to rename all variables. Their unit of measurement along with the new names are specified in the table below [1], apart from the last variable which is the class.

| Variable Name    | Full Name                  | Unit of Measurement              |
|------------------|----------------------------|----------------------------------|
| Pelvic_Inc       | Pelvic incidence           | Degrees (°)                      |
| Pelvic_Tilt      | Pelvic tilt                | Degrees (°)                      |
| Lumb_Lard_Angle  | Lumbar lordosis angle      | Degrees (°)                      |
| Sacral_Slope     | Sacral slope               | Degrees (°)                      |
| Pelvic_Rad       | Pelvic radius              | Millimeters or centimeters (mm/cm) |
| Spond_Grade      | Grade of spondylolisthesis | Percentage (of slippage, %)      |


The next figure shows the scatter matrix of all covariates, coloured by the two classes (blue for abnormal, red for normal). One thing to notice is the data is not linearly separable; in all combinations of variables there is a great deal of overlap. Additionally, in the Spond_Grade column/row there is a major outlier.


The next plot shows the density plots of each covariate. Clearly, Spond_Grade
is extremely right-skewed with multiple peaks, and other variables like Pelvic_Inc, Lumb_Lard_Angle and Sacral_Slope suggest that they have two or three peaks. Pelvic_Tilt and Pelvic_Rad seem the most approximately normal, with the former being slightly right-skewed. The multiple peaks may be an indicator of class separation, simplifying the classification process.

Now, to detect outliers we will use boxplots - the figure below shows these for each covariate. Like we saw in the scatter matrix, a huge outlier can be seen in Spond_Grade - this is found to be observation 116 with a value of 418.54. We elect to omit only this observation; the other outliers seem marginal compared to 116. This is further supported by the (scaled) PCA biplot.

# 3 Unsupervised Analysis

We wish to capture the underlying structure of the data. One way to do this is through a clustering method.
K-means clustering is a type of partitional clustering algorithm, whereby data objects are subsetted into non-overlapping groups, known as clusters. Each cluster has a centre, called a centroid, not necessarily an actual data object. The algorithm evaluates each data object by calculating the Eu- clidean distance from that and the centroids and assigns it to the nearest cluster. When a cluster gains or loses an observation, the algorithm then recalculates the centroid, and does this until there are no observations left to assign. It tries to minimise an objective function, the within cluster sum of squares [2]:


where $K$ is the total number of clusters, $S_k$ is the set of all points in cluster $k$, $w_i$ is the i-th point in $S_k$ and $c_k$ is the centroid of cluster $k$. This equation calculates the square of the Euclidean distance between a datapoint and the clusters’ centroids, summing over all clusters and all data points $i$ belonging to $S_k$.

There are many ways to choose k. The elbow method is the oldest and still one of the most popular ways - this requires plotting the WCSS against the number of clusters, and finding the ‘elbow’, i.e the point where the WCSS starts to decrease much slower. From the plot below, we determine that it is $k$ = 3.

Looking at the clusters in the next plot, we see that the dimensions explain approximately 76% of the variance. The clusters themselves have slight overlap, and clusters 0 and 3 are tightly grouped, suggesting a need to change the number of clusters. Based on my analysis, 3 clusters could be meaningful in terms of identifying subgroups within the two class labels, i.e there may be levelling degrees of normality/abnormality in the orthopaedic research. This remains true, as the original dataset actually has 4 classes: DH (Disk Hernia), Spondylolisthesis (SL), Normal (NO) and Abnormal (AB), but we are utilising a simpler dataset for the purposes of this analysis.

# 4 Supervised Analysis

Using the class labels, we’ll now attempt to classify patients between nor- mality and abnormality. Note that 209 observations are labelled abnormal, and 100 are normal, showing a class imbalance in the data - we need to be aware of this as some models perform worse with imbalances. One way to combat this is to apply class weights to our model which is just:


where $n_{samples}$ is the total number of observations in the dataset, and nclasses is the total number of classes, and $n_{{samples}_j}$ are the total number of observations in class $j$ [3].
Although considered better for when number of attributes are larger than observations, Support Vector Machines (SVM) were chosen for this dataset. SVM uses regularisation parameters so it’s robust to overfitting, and when appropriately tuned, can be beneficial for imbalanced datasets. SVM use hyperplanes to best divide data into two classes. The support vectors are the data points closest to these hyperplanes [4]. These can be described as:

Considering that the scatter plot showed no linear separation, we use the
radial basis function (RBF) kernel [4]:


with the kernel, we now have the decision function:


where $v_i$ are constants to be found as part of the quadratic solving problem [4].

For classification, a 70/30 train/test split was applied with 10 fold cross-validation. Cross validation is seen to be not recommended for imbalanced datasets, but with the class weights it adds more robustness to our end result. The table below shows the confusion matrix - the accuracy is 90.3%, with 95.2% sensitivity and 80% specificity. These are solid values in the context of diagnostic research; it means that the model’s ability to detect positive cases is great. There is always a trade off between sensitivity and specificity, and in this context we should prioritise finding abnormality. Note that we also scaled and centered the data, as from the first table we see that there are different units of measurements.

|  Actual/Predicted  |      AB      |      NO      |
|--------------------|--------------|--------------|
| AB                 | 60           | 6            |
| NO                 | 3            | 24           |

The Reciever Operating Characteristic (ROC) curve, which plots the true positive rate against the false positive rate, can also be seen below. We want the curve to be closest to the top left for the Area Under the Curve (AUC) close to 1, anything lower than 0.5 is worthless. The area is calculated as 0.9317, which is an excellent result and shows that the model has great discrimination ability.

# 4 Discussion

The unsupervised analysis and EDA gave insight into the structure of the data which didn’t necessarily align with the class labels. This informed the supervised process about challenges such as non-linear separability, which helped us choose the RBF kernel for SVM. Other observations such as the class imbalance were tackled; adding class weights greatly improved the accuracy of the model. This was highlighted also from the clusters - K-means could’ve made clusters that were heavily skewed towards the dominant class.

# References


[1] J C Le Huec, S Aunoble, Leijssen Philippe, and Pellet Nicolas. Pelvic pa- rameters: origin and significance. Eur Spine J, 20 Suppl 5(Suppl 5):564–571, Sep 2011.

[2] Trupti Kodinariya and Prashant Makwana. Review on determining of cluster in k-means clustering. International Journal of Advance Research in Computer Science and Management Studies, 1:90–95, 01 2013.

[3] Kamaldeep Singh. How to improve class imbalance using class weights in machine learning?, Jul 2023. Accessed: 09/03/2024.

[4] Marti A. Hearst, Susan T Dumais, Edgar Osuna, John Platt, and Bernhard Scholkopf. Support vector machines. IEEE Intelligent Systems and their applications, 13(4):18–28, 1998.

