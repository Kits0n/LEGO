import concurrent.futures
import csv
import os
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import sklearn


def load():
    print("Load labels")
    img_paths, labels_true, labels_pred = [], [], []

    with open(os.path.join(labels_path, "labels.csv"), 'r') as f:
        read = csv.reader(f)
        for row in read:
            img_paths.append(row[0])
            labels_true.append(int(row[1]))
            labels_pred.append(int(row[2]))

    print("Load model")
    with open(os.path.join(labels_path, "kmeans.pkl"), "rb") as f:
        kmeans = pickle.load(f)

    return img_paths, labels_true, labels_pred, kmeans


def confusion_matrix(labels_pred, labels_true):
    cf_matrix = sklearn.metrics.confusion_matrix(labels_true, labels_pred)
    sns.heatmap(cf_matrix, annot=True, cmap='Blues')
    plt.show()


def move_file(img_path, label):
    src_path = os.path.join(original_img_path, img_path)
    dst_path = os.path.join(results_path, str(label), os.path.basename(img_path))
    os.link(src_path, dst_path)


def move_files(img_paths, labels_pred):
    os.makedirs(results_path, exist_ok=True)
    print("Save result")
    for i in range(0, max(labels_pred) + 1):
        os.makedirs(os.path.join(results_path, str(i)))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(move_file, img_path, label) for img_path, label in zip(img_paths, labels_pred)]

        for index, future in enumerate(concurrent.futures.as_completed(futures)):
            future.result()
            print(index, "/", len(img_paths))


def main():
    print("Load")
    img_paths, labels_true, labels_pred, kmeans = load()

    os.makedirs(results_path, exist_ok=True)

    v_measure_score = sklearn.metrics.v_measure_score(labels_pred, labels_true)
    print("v_measure_score", v_measure_score)

    adjusted_rand_score = sklearn.metrics.adjusted_rand_score(labels_true, labels_pred)
    print("adjusted_rand_score", adjusted_rand_score)

    with open(os.path.join(os.path.dirname(results_path), "scores.csv"), 'a') as f:
        writer = csv.writer(f)
        #writer.writerow(['id', 'v_measure_score', 'adjusted_rand_score'])
        writer.writerow([os.path.basename(results_path), v_measure_score, adjusted_rand_score])

    print("Confusion matrix")
    # confusion_matrix(labels_pred, labels_true)

    print("Move files")
    # move_files(img_paths, labels_pred)


original_img_path = 'results/preprocessed-img/padd+gray'
labels_path = 'results/clustered_img/xception_432'
results_path = 'results/evaluation/xception_432'

main()
