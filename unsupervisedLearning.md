# Unsupervised Learning Overview

## 1. Clustering Techniques
   - **K-Means Clustering:** K-Means partitions data into 'k' clusters by minimizing the within-cluster sum of squares. It assigns each data point to the cluster with the nearest mean.
   - **Hierarchical Clustering:** Hierarchical clustering builds a tree-like structure of clusters. It starts by considering each data point as a separate cluster and merges them iteratively based on their similarity.

## 2. Dimensionality Reduction Methods
   - **Principal Component Analysis (PCA):** PCA identifies orthogonal axes (principal components) along which the data varies the most. It then projects the data onto these components, effectively reducing the dimensionality.
   - **t-Distributed Stochastic Neighbor Embedding (t-SNE):** t-SNE maps high-dimensional data to a lower-dimensional space, emphasizing the preservation of pairwise similarities between data points.

## 3. Association Rule Learning
   - **Apriori Algorithm:** Apriori identifies frequent itemsets in a transactional database and generates association rules based on these sets, revealing relationships between different features.

## 4. Generative Models
   - **Gaussian Mixture Models (GMM):** GMM assumes that the data is a mixture of multiple Gaussian distributions. It is particularly useful for modeling complex data with multiple underlying patterns.
   - **Autoencoders:** Autoencoders consist of an encoder and a decoder, learning a compressed representation of data in an unsupervised manner. They are adept at capturing latent features.

## 5. Anomaly Detection Techniques
   - **Isolation Forest:** Isolation Forest isolates anomalies efficiently by randomly selecting features and using isolation trees.
   - **One-Class SVM:** One-Class SVM learns a decision boundary around the normal data, classifying instances outside this boundary as anomalies.

## 6. Density Estimation
   - **Kernel Density Estimation (KDE):** KDE estimates the probability density function of a random variable by placing a kernel at each data point and summing them to create a smooth density estimate.

## 7. Graph-based Approaches
   - **Community Detection:** Community detection algorithms identify densely connected subgraphs in a network, revealing underlying structures or groups.

## 8. Word Embeddings
   - **Word2Vec, GloVe:** These techniques represent words as vectors in a continuous vector space. Word2Vec captures semantic relationships by predicting context words, while GloVe focuses on word co-occurrence statistics.

## 9. Deep Learning in Unsupervised Settings
   - **Variational Autoencoders (VAE):** VAE combines autoencoders with probabilistic modeling, allowing for the generation of new data points within the learned distribution.
   - **Generative Adversarial Networks (GAN):** GAN consists of a generator and a discriminator. The generator creates realistic data, and the discriminator distinguishes between real and generated data, leading to the improvement of both over time.

## 10. Market Segmentation Applications
   - **Customer Segmentation:** In marketing, customer segmentation involves categorizing customers based on shared characteristics, allowing businesses to tailor strategies to specific customer groups for more effective engagement.

Unsupervised learning techniques are foundational in data analysis, enabling the discovery of hidden structures and patterns within datasets without relying on labeled information. Each method has its strengths and applications, and the choice depends on the specific characteristics and goals of the data analysis task at hand.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
```
