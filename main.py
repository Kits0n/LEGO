import os
import pickle
import datetime
import threading

import pandas as pd
import tensorflow as tf
import numpy as np
from joblib.numpy_pickle_utils import xrange
from keras.api.keras.preprocessing import image
from keras.applications.xception import preprocess_input
from sklearn.decomposition import PCA
import sklearn.metrics as metrics
import sklearn.cluster as cluster
from PIL import Image


def create_model_xception():
    base_model = tf.keras.applications.xception.Xception(
        include_top=True, weights='imagenet', input_tensor=None,
        input_shape=None, pooling=None, classes=1000,
        classifier_activation='softmax'
    )
    layer_size = base_model.get_layer('avg_pool').output.shape[1]
    img_size = 299
    return tf.keras.Model(inputs=base_model.input,
                          outputs=base_model.get_layer('avg_pool').output), layer_size, img_size


def create_model_VGG16():
    base_model = tf.keras.applications.VGG16(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation="softmax",
    )

    layer_size = base_model.get_layer('fc1').output.shape[1]
    img_size = 224
    return tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output), layer_size, img_size


def create_model_NASNetLarge():
    base_model = tf.keras.applications.nasnet.NASNetLarge(
        input_shape=None, include_top=True, weights='imagenet',
        input_tensor=None, pooling=None, classes=1000
    )

    layer_size = base_model.get_layer('global_average_pooling2d').output.shape[1]
    img_size = 331
    return tf.keras.Model(inputs=base_model.input,
                          outputs=base_model.get_layer('global_average_pooling2d').output), layer_size, img_size


def get_images_paths(dir_path):
    images_paths = []
    correct_labels = []
    dirs_depth_1 = os.listdir(dir_path)
    for i, dir_depth_1 in enumerate(dirs_depth_1):
        paths = [dir_path + "/" + dir_depth_1 + "/" + x for x in os.listdir(dir_path + "/" + dir_depth_1)]
        correct_labels.extend([i] * len(paths))
        images_paths.extend(paths)
    num_clusters = len(dirs_depth_1)
    return images_paths, correct_labels, num_clusters


def split_list(alist, parts):
    length = len(alist)
    return [alist[i * length // parts: (i + 1) * length // parts]
            for i in range(parts)]


def process_images(images_paths, model, layer_size, img_size, num_threads):
    splited_images_paths = split_list(images_paths, num_threads)
    results = [None] * num_threads
    threads = []
    for i in range(0, num_threads):
        thread = threading.Thread(target=process_images_thread,
                                  args=(splited_images_paths[i], model, img_size, layer_size, results, i))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    points = np.concatenate(results)
    return points


def process_images_thread(images_paths, model, img_size, layer_size, results, index):
    points = np.zeros((0, layer_size))
    for i, img_path in enumerate(images_paths):
        img = image.load_img(img_path, target_size=(img_size, img_size))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        avg_pool = model.predict(x)
        points = np.append(points, avg_pool, axis=0)
        if index == 0:
            print(i / len(images_paths) * 100, "%")
    results[index] = points


def pca(points, n_components):
    pca = PCA(n_components=min(min(points.shape), n_components))
    pca.fit(points)
    points = pca.transform(points)
    return points


def input_fn():
    return tf.compat.v1.train.limit_epochs(
        tf.convert_to_tensor(global_points, dtype=tf.float32), num_epochs=1)


def kmeans_keras(num_clusters, num_iterations, points):
    kmeans = tf.compat.v1.estimator.experimental.KMeans(
        num_clusters=num_clusters, use_mini_batch=False)

    # Nie można przekazać argumentów do input_fn więc używam zmiennej globalnej
    global global_points
    global_points = points

    for i in xrange(num_iterations):
        kmeans.train(input_fn)
        kmeans.cluster_centers()
        print(i / num_iterations * 100, "%")
    print(kmeans.cluster_centers())

    result = []
    # map the input points to their clusters
    cluster_indices = list(kmeans.predict_cluster_index(input_fn))
    for i, point in enumerate(points):
        cluster_index = cluster_indices[i]
        result.append(cluster_index)

    return result


def kmeans_sklearn(num_clusters, points):
    kmeans = cluster.KMeans(n_clusters=num_clusters, init='k-means++', n_init=500, max_iter=9000, tol=0.0001, verbose=0,
                            random_state=None, copy_x=False, algorithm='auto').fit(points)
    return kmeans


def v_measure(labels, correct_labels):
    print("v_measure_score", "%.6f" % metrics.v_measure_score(correct_labels, labels))


def create_confusion_matrix(labels, correct_labels, num_clusters):
    confusion_matrix = np.zeros((num_clusters, num_clusters))
    for i, correct_label in enumerate(correct_labels):
        confusion_matrix[correct_label][labels[i]] += 1

    print("True label in rows, clustering label in columns")
    print(pd.DataFrame(confusion_matrix, columns=list(range(0, num_clusters))))


def serialize(object2, filename):
    with open(filename, "wb") as f:
        pickle.dump(object2, f)


def deserialize(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def delete_points(kmeans, points, result, points_types):
    indexes = []
    for i, point in enumerate(points):
        dist = np.linalg.norm(kmeans.cluster_centers_[result[i]] - point)
        if dist > 14:
            indexes.append(i)
    points = np.delete(points, indexes, axis=0)
    points_types = np.delete(points_types, indexes, axis=0)
    return points, points_types


def add_pad(img_paths):
    for img_path in img_paths:
        img = Image.open(img_path)
        img_size = img.size
        width = img_size[0]
        height = img_size[1]
        if width != height:
            bigside = max(width, height)
            background = Image.new('RGB', (bigside, bigside), (255, 255, 255))
            offset = (int(round(((bigside - width) / 2), 0)), int(round(((bigside - height) / 2), 0)))
            background.paste(img, offset)
            background.save(img_path)


def resize_images(img_paths, size):
    for i, img_path in enumerate(img_paths):
        img = Image.open(img_path)
        image_size = img.size
        width = image_size[0]
        height = image_size[1]
        proportion = size / max(width, height)
        resized_img = img.resize((int(width * proportion), int(height * proportion)))
        resized_img.save(img_path)


def main():
    print("Create file paths")
    images_paths, correct_labels, num_clusters = get_images_paths("../data/dataset_1")

    if os.path.exists("./points.bin"):
        points = deserialize("points.bin")
    else:
        print("Create model")
        model, layer_size, img_size = create_model_xception()
        print("Process files")
        a = datetime.datetime.now()
        points = process_images(images_paths, model, layer_size, img_size, 4)
        print(datetime.datetime.now()-a)
        serialize(points, "points.bin")

    # print("PCA")
    points = pca(points, n_components=1000)

    print("Kmeans")
    # num_iterations = 20
    # result = kmeans_keras(num_clusters, num_iterations, points)
    kmeans = kmeans_sklearn(num_clusters, points)
    labels = kmeans.labels_

    print("V-measure")
    v_measure(labels, correct_labels)

    print("Confusion matrix")
    create_confusion_matrix(labels, correct_labels, num_clusters)

    print("Predict")
    # kmeans.predict([new_points])

    while True:
        val = input("True Label: ")
        val2 = input("Label: ")
        for i, images_path in enumerate(images_paths):
            if correct_labels[i] == int(val) and labels[i] == int(val2):
                print(images_path)


global_points = None
main()

