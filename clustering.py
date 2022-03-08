import concurrent.futures
import csv
import os
import pickle

import h5py
import sklearn
from sklearn.cluster import KMeans


def kmeans_sklearn(feature_vectors):
    kmeans = KMeans(
        n_clusters=num_clusters,
        init='k-means++',
        n_init='auto',
        max_iter=9000,
        tol=0.00001,
        verbose=0,
        random_state=1,
        copy_x=False).fit(feature_vectors)

    return kmeans


def dbscan_sklearn(points):
    dbscan = sklearn.cluster.OPTICS(min_samples=15, max_eps=9000, metric='minkowski', p=2, metric_params=None,
                                    cluster_method='xi', eps=None, xi=0.05, predecessor_correction=True,
                                    min_cluster_size=None, algorithm='auto', leaf_size=30, memory=None,
                                    n_jobs=None).fit(points)
    labels = dbscan.labels_
    return labels


def move_file(img_path, label):
    src_path = os.path.join(original_img_path, img_path)
    dst_path = os.path.join(results_path, str(label), os.path.basename(img_path))
    os.link(src_path, dst_path)


def move_files(img_paths, labels_pred):
    print("Save result")
    for i in range(0, max(labels_pred) + 1):
        os.makedirs(os.path.join(results_path, str(i)))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(move_file, img_path, label) for img_path, label in zip(img_paths, labels_pred)]

        for index, future in enumerate(concurrent.futures.as_completed(futures)):
            future.result()
            print(index, "/", len(img_paths))


def save(img_paths, labels_true, labels_pred, kmeans):
    print("Save labels")
    rows = [list(item) for item in zip(img_paths, labels_true, labels_pred)]

    with open(os.path.join(results_path, "labels.csv"), 'w') as f:
        write = csv.writer(f)
        write.writerows(rows)

    print("Save model")
    with open(os.path.join(results_path, "kmeans.pkl"), "wb") as f:
        pickle.dump(kmeans, f)


def load_class(class_name, hf):
    hf_group = hf[class_name]
    feature_vectors = []
    img_paths = []

    for file_name in hf_group:
        feature_vector = hf_group[file_name][()]
        img_paths.append(os.path.join(class_name, file_name))
        feature_vectors.append(feature_vector)

    return (feature_vectors, img_paths, class_name)


def main():
    labels_true = []
    img_paths = []
    feature_vectors = []

    print("Load vectors")
    with h5py.File(vectors_path, 'r') as hf:
        classes = list(hf.keys())
        num_classes = len(classes)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(load_class, class_name, hf) for class_name in classes]

            for future in concurrent.futures.as_completed(futures):
                (class_feature_vectors, class_img_paths, class_name) = future.result()

                feature_vectors += class_feature_vectors
                labels_true += [classes.index(class_name)] * len(class_img_paths)
                img_paths += class_img_paths
                print(classes.index(class_name), "/", num_classes)

    print("Clustering")
    kmeans = kmeans_sklearn(feature_vectors)
    labels_pred = kmeans.labels_

    print("Save result")
    os.makedirs(results_path, exist_ok=True)
    save(img_paths, labels_true, labels_pred, kmeans)


vectors_path = 'results/feature_vectors/xception.hdf'
results_path = "results/clustered_img/xception_432"
num_clusters = 432
main()
