import os
import cv2
import numpy as np
import argparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Cluster images based on style.')
    parser.add_argument('image_dir', type=str, help='Path to the directory containing the images')
    parser.add_argument('--n_components', type=int, default=50, help='Number of components for PCA')
    parser.add_argument('--n_clusters', type=int, default=50, help='Number of clusters for K-Means')
    args = parser.parse_args()

    image_dir = args.image_dir
    n_components = args.n_components
    n_clusters = args.n_clusters

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

    # Reduce dimensionality with PCA
    pca = PCA(n_components=n_components)
    image_features_pca = pca.fit_transform(image_features)

    # Cluster the images using K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(image_features_pca)
    labels = kmeans.labels_

    # Store the found images in the folder './data/sty'
    os.makedirs('./data/sty', exist_ok=True)
    for i in range(n_clusters):
        cluster_files = [image_files[j] for j in range(len(labels)) if labels[j] == i]
        for file in cluster_files:
            os.rename(os.path.join(image_dir, file), os.path.join('./data/sty', file))

if __name__ == '__main__':
    main()
