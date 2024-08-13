import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Path to the directory containing the images
image_dir = 'path_to_your_images'

# Function to extract image features (color and quality)
def extract_image_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))  # Resize to reduce dimensionality
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Flatten the image and normalize pixel values
    flat_image = image.flatten() / 255.0
    
    # You can add other features like edges, texture, etc.
    return flat_image

# Load and process all images
image_features = []
image_files = []
for filename in os.listdir(image_dir):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        image_path = os.path.join(image_dir, filename)
        features = extract_image_features(image_path)
        image_features.append(features)
        image_files.append(filename)

image_features = np.array(image_features)

# Optional: Reduce dimensionality with PCA
pca = PCA(n_components=50)
image_features_pca = pca.fit_transform(image_features)

# Cluster the images using K-Means
n_clusters = 50
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(image_features_pca)
labels = kmeans.labels_

# Display the clustering result
for i in range(n_clusters):
    print(f"Cluster {i+1}:")
    cluster_files = [image_files[j] for j in range(len(labels)) if labels[j] == i]
    print(cluster_files)
    print("\n")

# Optional: Visualize the clusters
plt.figure(figsize=(12, 8))
plt.scatter(image_features_pca[:, 0], image_features_pca[:, 1], c=labels, cmap='viridis')
plt.title('Image Clustering with K-Means')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar()
plt.show()
