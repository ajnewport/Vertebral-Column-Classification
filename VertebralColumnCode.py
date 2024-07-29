# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.utils.class_weight import compute_class_weight
import os
path = '/Users/Milly/Documents/DATA70132'
os.chdir(path)

# Set working directory and read data
df = pd.read_csv("vertebral_column_data.txt", delim_whitespace=True, header=None)

# Renaming columns
df.columns = ["Pelvic_Inc", "Pelvic_Tilt", "Lumb_Lard_Angle", "Sacral_Slope", "Pelvic_Rad", "Spond_Grade", "Class"]

# Check for duplicated data
df.duplicated().sum() # No duplicated data

# Count classes
df['Class'].value_counts() # Check original counts

# Scatter plot matrix
sns.pairplot(df, hue="Class", palette={"AB": "blue", "NO": "red"})
plt.show()

# Reshape data for density plots
df_long = pd.melt(df, id_vars='Class', var_name='Variable', value_name='Value')

# Density plots
g = sns.FacetGrid(df_long, col="Variable", hue="Variable", sharex=False, sharey=False)
g.map(sns.kdeplot, "Value", shade=True, alpha=0.5)
g.add_legend()
plt.subplots_adjust(top=0.9)
plt.show()

# Boxplots
df.drop('Class', axis=1).plot(kind='box', subplots=True, layout=(2, 3), figsize=(12, 8))
plt.tight_layout()
plt.show()

# Identify and remove outlier
df = df[df['Spond_Grade'] <= 400]

# PCA
X = df.drop('Class', axis=1)
y = df['Class']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)
print(pca.explained_variance_ratio_)
pca_df = pd.DataFrame(data = pca.components_, columns = X.columns)
print(pca_df)

# Plot PCA biplot
pca_results = pd.DataFrame(data = X_pca, columns = [f'PC{i}' for i in range(1, len(X.columns)+1)])
pca_results['Class'] = y

# Create the plot
plt.figure(figsize=(10, 7))

# Scatter plot of the scores
sns.scatterplot(x='PC1', y='PC2', hue='Class', data=pca_results, s=60, alpha=0.7, edgecolor='k', palette={"AB": "blue", "NO": "red"})

# Add vectors for the loadings
feature_vectors = pca.components_.T
n_features = feature_vectors.shape[0]
scaling_factor = 3  # Adjust for visualization clarity

for i in range(n_features):
    plt.arrow(0, 0, feature_vectors[i, 0] * scaling_factor, feature_vectors[i, 1] * scaling_factor,
              color='black', alpha=0.5, head_width=0.1)
    plt.text(feature_vectors[i, 0] * scaling_factor * 1.15, feature_vectors[i, 1] * scaling_factor * 1.15,
             X.columns[i], color='black', ha='center', va='center')

plt.title("PCA Biplot")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} Variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} Variance)")
plt.grid()
plt.show()

# K-means clustering
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# K-means with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
pca_results['Cluster'] = kmeans.labels_
sns.scatterplot(x='Dim1', y='Dim2', hue='Cluster', data=pca_results, palette="Set1")
plt.title("K-means Clustering")
plt.show()

# Convert class labels to binary: 'AB' as 1 and 'NO' as 0
y_binary = (df['Class'] == 'AB').astype(int)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_binary, test_size=0.3, random_state=42, stratify=y_binary)

# Train SVM with RBF kernel
svm = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001]}
grid = GridSearchCV(svm, param_grid, refit=True, cv=10)
grid.fit(X_train, y_train)

# Best model
best_svm = grid.best_estimator_
print("Best Parameters:", grid.best_params_)

# Predictions and evaluation
y_pred = best_svm.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ROC Curve
y_pred_prob = best_svm.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.clf()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random guess
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

