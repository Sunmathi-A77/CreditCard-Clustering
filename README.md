## 🧩 Clustering Analysis with PCA | KMeans • Hierarchical • DBSCAN

### 📘 Project Overview

This project performs Exploratory Data Analysis (EDA) and Unsupervised Clustering using three key algorithms:

#### KMeans Clustering

#### Hierarchical (Agglomerative) Clustering

#### DBSCAN (Density-Based Spatial Clustering)

It applies Principal Component Analysis (PCA) to reduce dimensionality and visualize the clusters in 2D space.
The goal is to identify natural groupings in the dataset and compare clustering performance using silhouette scores.

### 🧠 Key Objectives

Perform full EDA to understand the dataset structure and feature distributions.

Scale features and apply PCA for dimensionality reduction.

Implement and evaluate KMeans, Hierarchical, and DBSCAN algorithms.

Compare clustering results using Silhouette Score, Noise fraction, and Cluster count.

Visualize clusters and dendrograms for interpretability.

### 🧾 Project Structure
```
ClusteringProject/
│
├── data/
│   └── your_dataset.csv
│
├── notebook/
│   └── clustering_analysis.ipynb
│
├── models/
│   ├── scaler.pkl
│   ├── pca.pkl
│   ├── kmeans.pkl
│   ├── hier.pkl
│   └── dbscan.pkl
│
├── plots/
│   ├── elbow_method.png
│   ├── dendrogram.png
│   ├── dbscan_knn.png
│   └── cluster_visualizations.png
│
├── README.md
└── requirements.txt
```
### ⚙️ Technologies Used

| Category                     | Library / Tool                          |
| ---------------------------- | --------------------------------------- |
| **Language**                 | Python 3.x                              |
| **Data Handling**            | pandas, numpy                           |
| **Visualization**            | matplotlib, seaborn                     |
| **Preprocessing**            | StandardScaler                          |
| **Dimensionality Reduction** | PCA (sklearn.decomposition)             |
| **Clustering Algorithms**    | KMeans, AgglomerativeClustering, DBSCAN |
| **Evaluation Metric**        | Silhouette Score                        |
| **Distance Measure**         | Euclidean Distance                      |

### 📊 Exploratory Data Analysis (EDA)

#### 1. Data Cleaning

Checked for missing values and duplicates

Filled missing values with median

Removed duplicates

#### 2. Statistical Summary

Visualized feature distributions (histograms, boxplots)

#### 3. Correlation Analysis

Heatmap of feature correlations to detect redundancy

#### 4. Scaling

Standardized features using StandardScaler

#### 5. PCA

Reduced dimensions to 2 components

Explained Variance Ratio: ~0.60

### 🔍 Model Implementation & Results

#### 🔹 KMeans Clustering

Optimal k: 4 (from Elbow Method)

Silhouette Score: 0.4132

Interpretation: Good cluster separation and compactness.

#### 🔹 Hierarchical Clustering

Linkage Method: Ward

Optimal clusters: 2 (from Dendrogram)

Silhouette Score: 0.4242

Interpretation: Slightly better cluster cohesion than KMeans.

#### 🔹 DBSCAN Clustering

eps: 0.35

min_samples: 5

Estimated clusters: 3

Noise points: 37

Silhouette Score: 0.160

Interpretation: DBSCAN found meaningful density-based clusters, but with lower overall separation.

### 📈 Visualizations

| Visualization         | Description                                          |
| --------------------- | ---------------------------------------------------- |
| **Elbow Plot**        | Shows optimal k for KMeans                           |
| **Dendrogram**        | Determines cut point for Hierarchical clusters       |
| **k-distance Graph**  | Used to find suitable `eps` for DBSCAN               |
| **PCA Scatter Plots** | Visualizes 2D projection of clusters for all methods |

### 🧮 Model Comparison

| Algorithm        | Silhouette Score | Clusters | Noise Fraction |
| ---------------- | ---------------- | -------- | -------------- |
| **KMeans**       | 0.4132           | 4        | -              |
| **Hierarchical** | 0.4242           | 4        | -              |
| **DBSCAN**       | 0.160            | 3        | ~5%            |

🟩 Best Performing Model: Hierarchical Clustering (Ward linkage)
It achieved the highest silhouette score and balanced cluster separation.

### 🚀 How to Run

#### 1️⃣ Clone the Repository

git clone https://github.com/yourusername/ClusteringProject.git

cd ClusteringProject

#### 2️⃣ Install Dependencies

pip install -r requirements.txt

#### 3️⃣ Run the Notebook

jupyter notebook notebook/clustering_analysis.ipynb

### 📦 requirements.txt

pandas

numpy

matplotlib

seaborn

scikit-learn

scipy

### 📜 Key Insights

PCA captured 60% of total variance, allowing clear 2D visualization.

Hierarchical Clustering provided slightly better performance than KMeans.

DBSCAN effectively detected a small set of noise points, revealing less dense groups.

Cluster quality metrics confirm meaningful structure in data.

### 🏁 Final Conclusion

Best Model: Hierarchical Clustering (Ward linkage)

Reason: Highest silhouette score (0.4242), stable cluster formation, and interpretable structure.

KMeans remains a strong alternative with similar performance.
DBSCAN is useful for datasets with irregular density or outliers.
