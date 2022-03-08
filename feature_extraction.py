import gc
import os

import h5py
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_model_xception():
    base_model = tf.keras.applications.xception.Xception(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation='softmax'
    )
    global input_size
    input_size = 299
    model = tf.keras.Model(
        inputs=base_model.input,
        outputs=base_model.get_layer('avg_pool').output)

    return model


def process_path(file_path):
    img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [input_size, input_size])

    return file_path, img


def extract_features(model):
    class_num = os.listdir(preprocessed_img_path)

    with h5py.File(results_path, 'a') as hf:
        for index, (dir_path, dir_names, file_names) in enumerate(os.walk(preprocessed_img_path)):
            if index == 0:
                continue

            dataset = tf.data.Dataset.list_files(os.path.join(dir_path, "*"), shuffle=False)\
                .map(process_path, num_parallel_calls=tf.data.AUTOTUNE)\
                .batch(16)

            dir_name = os.path.basename(dir_path)

            for path_batch, img_batch in dataset.take(-1):

                img_batch = tf.keras.applications.xception.preprocess_input(img_batch.numpy().astype("uint8"))
                vectors_batch = model.predict(img_batch)

                for vector, path in zip(vectors_batch, path_batch.numpy().tolist()):
                    hf.create_dataset(os.path.join(dir_name, os.path.basename(path.decode('utf-8'))), data=vector)

            gc.collect()
            print("Progress: ", index, "/", len(class_num))


def main():
    print("Get model")
    model = get_model_xception()

    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    print("Extract features")
    extract_features(model)


input_size = 299
preprocessed_img_path = './results/preprocessed-img/padd+gray'
results_path = './results/feature_vectors/xception.hdf'
main()
