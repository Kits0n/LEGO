import cv2
import os
import multiprocessing


def load_img_paths():
    img_paths = []
    for index, (dir_path, dir_names, file_names) in enumerate(os.walk(img_path)):
        img_paths.extend([dir_path + "/" + file_name for file_name in file_names])
    return img_paths


def padding(img):
    height, width = img.shape
    vertical = int(abs(max(width, height) - height) / 2)
    horizontal = int(abs(max(width, height) - width) / 2)
    return cv2.copyMakeBorder(img, vertical, vertical, horizontal, horizontal, cv2.BORDER_REPLICATE)


def preprocess_images(path):
    print(path)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = padding(img)
    dir_path = os.path.join(result_path, os.path.basename(os.path.dirname(path)))
    file_name = os.path.basename(path)
    os.makedirs(dir_path, exist_ok=True)
    cv2.imwrite(os.path.join(dir_path, file_name), img)


def main():
    os.makedirs(result_path, exist_ok=True)

    print("Load img paths")
    img_paths = load_img_paths()

    print("Preprocess images")
    with multiprocessing.Pool() as pool:
        pool.map(preprocess_images, img_paths)


img_path = '../data/original/images_real'
result_path = './results/preprocessed-img/padd+gray'
main()
