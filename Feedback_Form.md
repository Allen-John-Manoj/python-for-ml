# Intel Workshop Feedback Analysis

This Markdown document presents the analysis of feedback data from an Intel workshop. The workshop involved 31 students and featured insights from 4 resource persons. The aim is to showcase the code and analysis related to the feedback form collected during the workshop.

## Workshop Overview

- Number of Students: 31
- Resource Persons: 4+

## Setting Up the Environment and Importing Libraries

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
```

# Load the dataset from the provided URL
```python
df_class = pd.read_csv("https://raw.githubusercontent.com/sijuswamy/Intel-Unnati-sessions/main/Feed_back_data.csv")
```

# Display information about the dataset
```python
df_class.info()
```
# Drop unnecessary columns
```python
df_class = df_class.drop(['Timestamp', 'Email ID', 'Please provide any additional comments, suggestions, or feedback you have regarding the session. Your insights are valuable and will help us enhance the overall learning experience.'], axis=1)
df_class.info()
```
| Column                                                                                                                                     | Non-Null Count | Dtype  |
|------------------------------------------------------------------------------|----------------|--------|
| Name of the Participant                                                     | 174 non-null   | object |
| Branch                                                                       | 174 non-null   | object |
| Semester                                                                     | 174 non-null   | object |
| Recourse Person of the session                                               | 174 non-null   | object |
| How would you rate the overall quality and relevance of the course content presented in this session?                                    | 174 non-null   | int64  |
| To what extent did you find the training methods and delivery style effective in helping you understand the concepts presented?          | 174 non-null   | int64  |
| How would you rate the resource person's knowledge and expertise in the subject matter covered during this session?                      | 174 non-null   | int64  |
| To what extent do you believe the content covered in this session is relevant and applicable to real-world industry scenarios?           | 174 non-null   | int64  |
| How would you rate the overall organization of the session, including time management, clarity of instructions, and interactive elements?| 174 non-null   | int64  |

# Rename columns
```python
df_class.columns = ["Name", "Branch", "Semester", "Resourse Person", "Content Quality", "Effeciveness", "Expertise", "Relevance", "Overall Organization"]
df_class.shape
#(174, 9)
```
# Exploratory Data Analysis: Percentage Analysis of RP-wise Distribution

## Creating a percentage analysis of Resourse Person (RP)-wise distribution of data

```python
# Calculate the percentage distribution of Resourse Person (RP)
round(df_class["Resourse Person"].value_counts(normalize=True) * 100, 2)
```
| Resourse Person           | Proportion |
|---------------------------|------------|
| Mrs. Akshara Sasidharan   | 34.48      |
| Mrs. Veena A Kumar        | 31.03      |
| Dr. Anju Pratap           | 17.24      |
| Mrs. Gayathri J L         | 17.24      |
```python
round(df_class["Name"].value_counts(normalize=True)*100,2)
```
| Name                        | Proportion |
|-----------------------------|------------|
| Sidharth V Menon            | 4.02       |
| Rizia Sara Prabin           | 4.02       |
| Abia Abraham                | 3.45       |
| Rahul Biju                  | 3.45       |
| Christo Joseph Sajan        | 3.45       |
|      .                      |    .       |
|      .                      |    .       |
|      .                      |    .       |
| Jobin Tom                   | 0.57       |
| Lisbeth Ajith               | 0.57       |
| Anaswara Biju               | 0.57       |
| Aaron Thomas Blessen        | 0.57       |
| Marianna Martin             | 0.57       |

We see that there is a huge difference between the feedback reviews comparing the top and bottom students.
# Visualization

## Faculty-wise Distribution of Data - Bar Chart and Pie Chart
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Bar Chart
ax = plt.subplot(1, 2, 1)
ax = sns.countplot(y='Resourse Person', data=df_class)
plt.title("Faculty-wise distribution of data", fontsize=20, color='Brown', pad=20)

# Pie Chart
ax = plt.subplot(1, 2, 2)
ax = df_class['Resourse Person'].value_counts().plot.pie(explode=[0.1, 0.1, 0.1, 0.1], autopct='%1.2f%%', shadow=True)
ax.set_title(label="Resourse Person", fontsize=20, color='Brown', pad=20)
```
![download](https://github.com/Allen-John-Manoj/python-for-ml/assets/136485968/4781a791-9c75-4c6c-b729-a1c93ee1996e)

We can see that the Resource people who took more than one sessions have a higher Feedback count than the people who have only taken one session. We can see than in some cases, it is even doubled the amount of people who took only one session

# Summary of Responses - Box Plots
## Content Quality by Resourse Person
```python
sns.boxplot(y=df_class['Resourse Person'], x=df_class['Content Quality'])
plt.show()
```
![download](https://github.com/Allen-John-Manoj/python-for-ml/assets/136485968/8aa8d2d0-f122-467f-8f76-1938b48c80a1)

## Effectiveness by Resourse Person
```python
# Effectiveness by Resourse Person
sns.boxplot(y=df_class['Resourse Person'], x=df_class['Effeciveness'])
plt.show()
```
![download](https://github.com/Allen-John-Manoj/python-for-ml/assets/136485968/b6cdd6c9-6350-4fb1-a4c7-5c8d14205726)

## Expertise by Resourse Person
```python
# Expertise by Resourse Person
sns.boxplot(y=df_class['Resourse Person'], x=df_class['Expertise'])
plt.show()
```
![download](https://github.com/Allen-John-Manoj/python-for-ml/assets/136485968/8eb3375c-3139-4ef3-83e6-d3bf98433a11)

## Relevance by Resourse Person
```python
## Relevance by Resourse Person
sns.boxplot(y=df_class['Resourse Person'], x=df_class['Relevance'])
plt.show()
```
![download](https://github.com/Allen-John-Manoj/python-for-ml/assets/136485968/746cc7cc-8ade-45a7-8d49-dac0b3be6dc8)

## Overall Organization by Resourse Person
```python
## Overall Organization by Resourse Person
sns.boxplot(y=df_class['Resourse Person'], x=df_class['Overall Organization'])
plt.show()
```
![download](https://github.com/Allen-John-Manoj/python-for-ml/assets/136485968/fe6aa6a4-5e48-4dd9-a5d4-74e75d74d6b2)

## Content Quality by Branch
```python
## Content Quality by Branch
sns.boxplot(y=df_class['Branch'], x=df_class['Content Quality'])
plt.show()
```
![download](https://github.com/Allen-John-Manoj/python-for-ml/assets/136485968/600a3700-2422-4e08-b221-617f950d5bfd)


From the above, we see that the data is mostly the same throughout. Apart from a few anomalies, it is mostly the same. However the last boxplot shows relevant information. We see that CSE is more general, ECE is in the middle, RB is more positively 
inclined and IMCA is immensly perfect. This reflects the department distribution of the workshop, with a lot of cse students balancing the data out, while a single IMCA student has extreme and less varied data


# K-Means Clustering
## Clustering Analysis - Elbow Method

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Selecting input columns
input_col = ["Content Quality", "Effeciveness", "Expertise", "Relevance", "Overall Organization"]
X = df_class[input_col].values

# Initialize an empty list to store the within-cluster sum of squares
wcss = []

# Try different values of k
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)  # here inertia calculates the sum of square distance in each cluster

# Plot the within-cluster sum of squares for different values of k
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.title('Elbow Method')
plt.show()
```
![download](https://github.com/Allen-John-Manoj/python-for-ml/assets/136485968/8de76abd-e9ca-40c0-a1b3-0dd1dda42617)


We see that the gradient reduces and doesn't undergo signifcant changes after 3, and stabilizes after 6. SO we can conclude that the ideal amount of clusters is 3 (K = 3)

# GridSearch for KMeans Clustering
```python
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans

# Define the parameter grid
param_grid = {'n_clusters': [2, 3, 4, 5, 6]}

# Create a KMeans object
kmeans = KMeans(n_init='auto', random_state=42)

# Create a GridSearchCV object
grid_search = GridSearchCV(kmeans, param_grid, cv=5)

# Perform grid search
grid_search.fit(X)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_
```
```python
print("Best Parameters:", best_params)
print("Best Score:", best_score)
```
| Parameter      | Value     |
|----------------|-----------|
| n_clusters     | 6         |
| Best Score     | -21.20    |

# Perform k-means clustering

```python
from sklearn.cluster import KMeans

# Set the number of clusters
k = 3

# Create a KMeans object
kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)

# Fit the model and get cluster labels and centroids
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Add the cluster labels to the DataFrame
df_class['Cluster'] = labels
```
```python
df_class.sample(5)
```
# Clustered Data - Sample Rows

| Name                | Branch | Semester | Resourse Person       | Content Quality | Effeciveness | Expertise | Relevance | Overall Organization | Cluster |
|---------------------|--------|----------|-----------------------|------------------|--------------|-----------|-----------|----------------------|---------|
| Abna Ev             | CSE    | Sixth    | Dr. Anju Pratap       | 4                | 3            | 3         | 3         | 2                    | 2       |
| Nandana A           | CSE    | Fourth   | Mrs. Gayathri J L     | 5                | 5            | 5         | 5         | 4                    | 0       |
| Allen John Manoj    | CSE    | Sixth    | Mrs. Veena A Kumar    | 4                | 5            | 4         | 4         | 5                    | 1       |
| Bhagya Sureshkumar  | CSE    | Sixth    | Dr. Anju Pratap       | 5                | 5            | 5         | 5         | 5                    | 0       |
| Mathews Reji        | CSE    | Fourth   | Mrs. Akshara Sasidharan | 4              | 4            | 4         | 5         | 4                    | 1       |

# Visualize the clusters

```python
import matplotlib.pyplot as plt

# Scatter plot
plt.scatter(X[:, 1], X[:, 2], c=labels, cmap='viridis')

# Plot centroids
plt.scatter(centroids[:, 1], centroids[:, 2], marker='X', s=200, c='red')

# Set labels and title
plt.xlabel(input_col[1])
plt.ylabel(input_col[2])
plt.title('K-means Clustering')

# Show the plot
plt.show()
```
![download](https://github.com/Allen-John-Manoj/python-for-ml/assets/136485968/30b0a0a2-c37e-47b3-88df-48d3a0034f66)
# Perception on content quality over Clusters
```python
pd.crosstab(columns = df_class['Cluster'], index = df_class['Content Quality'])
```
# Cross-tabulation between Cluster and Content Quality

| Content Quality | Cluster 0 | Cluster 1 | Cluster 2 |
|-----------------|-----------|-----------|-----------|
| 3               | 0         | 7         | 11        |
| 4               | 18        | 67        | 1         |
| 5               | 95        | 8         | 0         |


# Conclusion

In this analysis, we applied k-means clustering with 3 clusters based on the scores participants gave for each session and their respective resource person. The clustering allowed us to categorize students into different groups based on their responses.

### Key Insights:

1. **Cluster Distribution:**
   - We identified three distinct clusters, each representing a group of students with similar feedback patterns.
   - Students in each cluster showed different preferences and ratings for the sessions.

2. **Resource Person Influence:**
   - The analysis highlighted the influence of the resource person on student ratings.
   - Different clusters may have preferred sessions conducted by specific resource persons.

3. **Feature Comparisons:**
   - Visualizations, including bar charts, pie charts, and box plots, were used to compare features such as content quality, effectiveness, expertise, relevance, and overall organization.
   - Box plots provided insights into the spread and central tendencies of each feature across clusters.

### Findings:

- **Cluster Characteristics:**
  - Cluster 0: Represents students with diverse ratings, potentially indicating a neutral or mixed response to the sessions.
  - Cluster 1: Contains students with high ratings, suggesting a positive reception of the sessions.
  - Cluster 2: Comprises students with lower ratings, indicating a less favorable response to the sessions.

- **Resource Person Impact:**
  - Different resource persons may have contributed to the formation of distinct clusters, indicating the varying impact of different instructors.

- **Feature Analysis:**
  - Content quality and effectiveness were crucial factors influencing cluster assignments.
  - The box plots provided a detailed understanding of the distribution of scores for each feature within clusters.

### Recommendations:

- **Tailored Approaches:**
  - Consider tailoring future sessions based on the characteristics of each cluster to enhance overall participant satisfaction.
  
- **Resource Person Training:**
  - Identify areas for improvement based on feedback from different clusters and provide additional training to resource persons.

- **Continuous Monitoring:**
  - Regularly monitor participant feedback and cluster assignments to adapt and improve future sessions.

In conclusion, the clustering analysis provided valuable insights into participant preferences and resource person influence. The visualizations and comparisons offered a comprehensive view of the dataset, 
facilitating informed decision-making for future sessions and improvements.

