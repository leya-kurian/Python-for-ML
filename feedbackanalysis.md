# Feedback Analysis and Clustering for Session Enhancement

## Objective:
The primary goal of this analysis is to delve into the feedback received from participants after a session, leveraging machine learning techniques to uncover underlying patterns. By understanding these patterns, we aim to gain insights that can guide improvements in future sessions, tailoring them to better meet the needs and expectations of the participants.

## Dataset Overview:
The dataset comprises responses gathered from participants, encompassing their ratings on various aspects of the session's quality and additional details like participant names, branches, semesters, resource persons, and additional comments. This diverse dataset serves as a rich source of information for understanding participant perceptions and preferences.

## Steps in Detail:
<br/>

1. **Data Loading and Preparation:**  
   - Load the dataset and perform initial data cleanup by removing irrelevant columns such as 'Timestamp', 'Email ID', and 'Additional Comments'.
  
 ```python
df_class=pd.read_csv("https://raw.githubusercontent.com/sijuswamy/Intel-Unnati-sessions/main/Feed_back_data.csv")
df_class = df_class.drop(['Timestamp','Email ID','Please provide any additional comments, suggestions, or feedback you have regarding the session. Your insights are valuable and will help us enhance the overall learning experience.'],axis=1)
```
<br/>

2. **Exploratory Data Analysis (EDA):**  
   - Check for null values in the dataset.
   - Perform percentage analysis to understand the distribution of feedback across different resource persons.
   - Visualize Faculty-wise distribution of data using count plots and pie charts.
   - Explore the relationship between various attributes and 'Resource Person' using appropriate visualizations (e.g., boxplot).

```python
df_class.isnull().sum().sum()
round(df_class["Resourse Person"].value_counts(normalize=True)*100,2)
round(df_class["Name"].value_counts(normalize=True)*100,2)
ax = plt.subplot(1,2,1)
ax = sns.countplot(x='Resourse Person', data=df_class)
#ax.bar_label(ax.containers[0])
plt.title("Faculty-wise distribution of data", fontsize=20,color = 'Brown',pad=20)
ax =plt.subplot(1,2,2)
ax=df_class['Resourse Person'].value_counts().plot.pie(explode=[0.1, 0.1,0.1,0.1],autopct='%1.2f%%',shadow=True);
ax.set_title(label = "Resourse Person", fontsize = 20,color='Brown',pad=20);
sns.boxplot(y=df_class['Resourse Person'],x=df_class['Content Quality'])
plt.show()
```
<br/>

3. **Data Preprocessing:**  
   - Select relevant input columns for clustering analysis, including ratings on aspects like 'Content Quality', 'Effectiveness', 'Expertise', 'Relevance', and 'Overall Organization'.
   - Standardize the input data to ensure that all features are on the same scale, preventing biases in the clustering process.
```python
input_col=["Content Quality","Effeciveness","Expertise","Relevance","Overall Organization"]
X=df_class[input_col].values
```
<br/>

4. **Elbow Method for Optimal K:**  
   - Implement the Elbow Method to determine the optimal number of clusters (k) for KMeans clustering.
   - Calculate the Within-Cluster Sum of Squares (WCSS) for different values of k.
   - Plot the WCSS values against the number of clusters to identify the 'elbow point' which indicates the optimal k.
```python
# Initialize an empty list to store the within-cluster sum of squares
from sklearn.cluster import KMeans
wcss = []
# Try different values of k
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k,n_init='auto', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)# here inertia calculate sum of square distance in each cluster
# Plot the within-cluster sum of squares for different values of k
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.title('Elbow Method')
plt.show()
```
<br/>

5. **Hyperparameter Tuning with Grid Search:**  
   - Define a parameter grid with different values of k for KMeans clustering.
   - Use Grid Search Cross Validation to find the best parameters for KMeans clustering, optimizing for performance metrics such as mean squared error.
```python
# Define the parameter grid
from sklearn.model_selection import GridSearchCV

param_grid = {'n_clusters': [2, 3, 4, 5, 6]}

# Create a KMeans object
kmeans = KMeans(n_init='auto',random_state=42)

# Create a GridSearchCV object
grid_search = GridSearchCV(kmeans, param_grid, cv=5)

# Perform grid search
grid_search.fit(X)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print("Best Parameters:", best_params)
print("Best Score:", best_score)
```
<br/>

6. **Clustering Analysis:**  
   - Apply the KMeans clustering algorithm to the preprocessed data with the optimal parameters obtained from Grid Search.
   - Assign cluster labels to each data point based on their similarity, grouping participants with similar feedback patterns together.
   - Visualize the clusters using scatter plots or other appropriate visualizations, considering relevant attributes.
```python
# Perform k-means clustering
k = 3 # Number of clusters
kmeans = KMeans(n_clusters=k,n_init='auto', random_state=42)
kmeans.fit(X)
# Get the cluster labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
# Add the cluster labels to the DataFrame
df_class['Cluster'] = labels
# Visualize the clusters
plt.scatter(X[:, 1], X[:, 2], c=labels, cmap='viridis')
plt.scatter(centroids[:,1], centroids[:, 2], marker='X', s=200, c='red')
plt.xlabel(input_col[1])
plt.ylabel(input_col[2])
plt.title('K-means Clustering')
plt.show()
pd.crosstab(columns = df_class['Cluster'], index = df_class['Content Quality'])
```
<br/>

## Conclusion:
Through feedback analysis and clustering, we can gain valuable insights into participant perceptions and preferences regarding session quality. These insights enable us to identify key areas for improvement and tailor future sessions to better meet participant expectations, ultimately enhancing the overall learning experience.

