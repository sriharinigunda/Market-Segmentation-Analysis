import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
import statsmodels.api as sm
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.tree import DecisionTreeClassifier, plot_tree


data=pd.read_csv('mcdonalds.csv')
col=data.columns.tolist()
print(col)

dim= data.shape
print(dim)

head_data= data.head(3)
print(head_data)

MD_x= data.iloc[:, 0:11].values
MD_x= (MD_x == "Yes").astype(int)
column_means= np.round(np.mean(MD_x, axis=0), 2)
print(column_means)

# Assuming x is a pandas DataFrame containing your data

# Instantiate the PCA object
MD_pca = PCA()

# Perform PCA on MD.x
MD_pca.fit(MD_x)

# Get the summary
summary = pd.DataFrame({
    'Standard deviation': MD_pca.explained_variance_ ** 0.5,
    'Proportion of Variance': MD_pca.explained_variance_ratio_,
    'Cumulative Proportion': np.cumsum(MD_pca.explained_variance_ratio_)
})
print(summary)
summary=summary.round(1)

plt.scatter(MD_pca.transform(x)[:, 0], MD_pca.transform(MD_x)[:, 1], c='grey')
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.show()

# Perform K-means clustering
k_values = range(2, 9)
results = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=1234)
    kmeans.fit(MD_x)
    results.append(kmeans)

# Plot number of segments vs. within-cluster sum of squares
plt.bar(k_values, [res.inertia_ for res in results])
plt.xlabel("Number of segments")
plt.show()

# Perform bootstrapped K-means clustering
bootstrap_results = []
for k in k_values:
    bootstrap_scores = []
    for _ in range(100):
        bootstrap_sample = np.random.choice(range(MD_x.shape[0]), size=x.shape[0], replace=True)
        bootstrap_data = MD_x[bootstrap_sample]
        kmeans = KMeans(n_clusters=k, random_state=1234)
        kmeans.fit(bootstrap_data)
        bootstrap_scores.append(kmeans.inertia_)
    bootstrap_results.append(bootstrap_scores)

# Plot number of segments vs. adjusted Rand index
plt.boxplot(bootstrap_results)
plt.xlabel("Number of segments")
plt.ylabel("Adjusted Rand index")
plt.xticks(range(1, len(k_values) + 1), k_values)
plt.show()

# Assuming MD.x is the input data matrix and MD.k4 is the cluster labels
# Assuming MD.km28 is the KMeans clustering result

# Plot histogram of cluster membership probabilities
plt.hist([res.labels_ for res in results], bins=10, range=(0, 1))
plt.xlabel("Probability")
plt.ylabel("Frequency")
plt.xlim(0, 1)
plt.show()

# Extract the cluster labels for the desired segment
MD_k4 = [res.labels_ for res in results]

# Calculate segment stability
# Assuming x is the input data matrix and MD_k4 is the cluster labels

# Fit K-means clustering with the desired number of segments
k = 4
kmeans = KMeans(n_clusters=k, random_state=1234)
kmeans.fit(MD_x)

# Calculate silhouette scores for each data point
silhouette_vals = silhouette_samples(x, kmeans.labels_)

# Sort the silhouette scores and extract the average silhouette score for each segment
segment_stability = []
for cluster in range(k):
    cluster_silhouette_vals = silhouette_vals[MD_k4 == cluster]
    segment_stability.append(cluster_silhouette_vals.mean())

# Plot segment stability
plt.box(segment_stability)
plt.xlabel("Segment number")
plt.ylabel("Segment stability")
plt.ylim(0, 1)
plt.show()


# Assuming MD.x is the input data matrix

# Fit Gaussian mixture models for different number of components (2 to 8)
k_values = range(2, 9)
models = []
for k in k_values:
    model = GaussianMixture(n_components=k)
    model.fit(MD_x)
    models.append(model)

# Print the fitted mixture models
for i, model in enumerate(models):
    print(f"Model {i+2}:")
    print(model)

# Plot information criteria
AIC = [model.aic(MD_x) for model in models]
BIC = [model.bic(MD_x) for model in models]
ICL = [model.lower_bound_ for model in models]
plt.plot(k_values, AIC, label="AIC")
plt.plot(k_values, BIC, label="BIC")
plt.plot(k_values, ICL, label="ICL")
plt.ylabel("Value of information criteria")
plt.xlabel("Number of components")
plt.legend()
plt.show()

# Extract the model with the desired number of components
desired_k = 4
desired_model = models[desired_k - 2]

# Obtain the cluster assignments from the desired model
MD_m4_clusters = desired_model.predict(MD_x)

# Compare cluster assignments from K-means and Gaussian mixture model
kmeans = KMeans(n_clusters=desired_k, random_state=1234)
kmeans_clusters = kmeans.fit_predict(MD_x)
table = pd.crosstab(kmeans_clusters, MD_m4_clusters, rownames=["K-means"], colnames=["Mixture"])
print(table)

# Fit a Gaussian mixture model with fixed cluster assignments
MD_m4a = GaussianMixture(n_components=desired_k)
MD_m4a.fit(MD_x, MD_m4_clusters)

# Compare cluster assignments from K-means and the fitted Gaussian mixture model
MD_m4a_clusters = MD_m4a.predict(MD_x)
table = pd.crosstab(kmeans_clusters, MD_m4a_clusters, rownames=["K-means"], colnames=["Mixture"])
print(table)

# Calculate log-likelihoods of the fitted Gaussian mixture models
loglik_m4a = MD_m4a.score(MD_x)
loglik_m4 = desired_model.score(MD_x)
print(f"Log-Likelihood (Mixture Model with Fixed Clusters): {loglik_m4a}")
print(f"Log-Likelihood (Fitted Mixture Model): {loglik_m4}")

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Assuming 'data' is a pandas DataFrame with the required columns

# Reverse levels of the 'Like' variable
like_table = data['Like'].value_counts().sort_index(ascending=False)
like_table_rev = like_table.iloc[::-1]
print(like_table_rev)

# Create a new variable 'Like.n' by subtracting 'Like' from 6
data['Like.n'] = 6 - pd.to_numeric(data['Like'])

# Create a frequency table of 'Like.n'
like_n_table = data['Like.n'].value_counts().sort_index()
print(like_n_table)

# Construct the formula for the regression model
f = 'Like.n ~ ' + ' + '.join(['yummy', 'convenient', 'spicy', 'fattening', 'greasy', 'fast', 'cheap','tasty', 'expensive', 'healthy', 'disgusting'])
f = sm.families.Binomial(link=sm.genmod.families.links.logit())
f = sm.formula(formula=f)

# Set the random seed
np.random.seed(1234)

# Fit a flexmix model using stepwise selection
MD.reg2 = None
for _ in range(10):  # nrep = 10
    model = sm.GLM.from_formula(formula=f, data=data, family=family)
    result = model.fit()
    if MD.reg2 is None or result.aic < MD.reg2.aic:
        MD.reg2 = result

print(MD.reg2.summary())

# Refit the best model
MD.ref2 = MD.reg2.refit()

# Summary of the refitted model
print(MD.ref2.summary())

# Plot the refitted model
plt.figure()
sm.graphics.plot_partregress_grid(MD.ref2)
plt.show()






# Calculate the pairwise distance matrix
MD_x = np.transpose(MD_x)  # Transpose MD.x for distance calculation
distance_matrix = pdist(MD_x)

# Perform hierarchical clustering
MD_vclust = linkage(distance_matrix)

# Plot barchart
plt.figure(figsize=(8, 6))
dendrogram(MD_vclust, orientation='left', no_labels=True, color_threshold=0)
plt.gca().set_axis_off()
plt.show()

# Perform PCA
MD_pca = PCA(n_components=2)
MD_x_pca = MD_pca.fit_transform(MD_x)

# Perform K-means clustering
k = 4
kmeans = KMeans(n_clusters=k, random_state=1234)
kmeans.fit(MD_x)
labels = kmeans.labels_

# Plot clusters on PCA
plt.scatter(MD_x_pca[:, 0], MD_x_pca[:, 1], c=labels, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Plot projection axes
plt.figure(figsize=(6, 6))
plt.scatter(MD_x_pca[:, 0], MD_x_pca[:, 1], c=labels, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Draw projection axes
for length, vector in zip(MD_pca.explained_variance_, MD_pca.components_):
    v = vector * 2 * np.sqrt(length)
    plt.plot([0, v[0]], [0, v[1]], '-k', lw=1)

plt.show()


# Calculate k4 clusters
k4 = MD_k4  # Replace with the actual variable containing k4 clusters

# Create mosaic plot for Like and segment number
like_table = pd.crosstab(k4, data['Like'])
plt.figure(figsize=(8, 6))
plt.imshow(like_table, cmap='viridis', aspect='auto')
plt.colorbar(label='Frequency')
plt.xlabel('Segment number')
plt.ylabel('Like')
plt.title('Mosaic Plot - Like vs Segment number')
plt.show()

# Create mosaic plot for Gender and segment number
gender_table = pd.crosstab(k4, data['Gender'])
plt.figure(figsize=(8, 6))
plt.imshow(gender_table, cmap='viridis', aspect='auto')
plt.colorbar(label='Frequency')
plt.xlabel('Segment number')
plt.ylabel('Gender')
plt.title('Mosaic Plot - Gender vs Segment number')
plt.show()

# Build decision tree using ctree
X = data[['Like.n', 'Age', 'VisitFrequency', 'Gender']]
y = (k4 == 3).astype(int)  # Convert to binary outcome
tree = DecisionTreeClassifier()
tree.fit(X, y)

# Plot decision tree
plt.figure(figsize=(10, 8))
plot_tree(tree, feature_names=X.columns, class_names=['False', 'True'],
          filled=True, rounded=True)
plt.show()

# Calculate average visit frequency for each segment
visit = np.array([np.mean(data.loc[k4 == i, 'VisitFrequency'].astype(float)) for i in range(1, 5)])

# Calculate average Like for each segment
like = np.array([np.mean(data.loc[k4 == i, 'Like.n']) for i in range(1, 5)])

# Plot scatter plot
female = data['Gender'] == 'female'  # Replace 'female' with the actual value for female gender
plt.scatter(visit, like, s=10 * female, color='blue')
plt.xlim(2, 4.5)
plt.ylim(-3, 3)
plt.xlabel('Average Visit Frequency')
plt.ylabel('Average Like')
plt.title('Scatter Plot of Visit Frequency vs Like')
plt.grid(True)

# Add text labels
for i in range(1, 5):
    plt.text(visit[i - 1], like[i - 1], str(i))

plt.show()
