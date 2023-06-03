from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np

model = None
interpreter = None
input_index = None
output_index = None

BUCKET_NAME = "ank_model"


def download_blob(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")


def predict(request):
    global model
    if model is None:
        download_blob(
            BUCKET_NAME,
            "models/dh_model2.h5",
            "/tmp/dh_model2.h5",
        )
        model = tf.keras.models.load_model("/tmp/dh_model2.h5", compile=False)

    img = request.files["file"]
    img = np.array(Image.open(img).convert("RGB").resize((224, 224)))
    img = img / 255
    img_arr = tf.expand_dims(img, 0)

    pred = model.predict(img_arr)
    predicted_value = pred[0][0]

    if predicted_value > 0.56:
        return {
            "pred": "Melanoma not detected",
            "prec": "No further action required. Monitor the situation for any changes over time.",
        }
    else:
        return {
            "pred": "Melanoma Detected",
            "prec": "Take immediate action. Consult with a healthcare professional or dermatologist for further evaluation and possible treatment options.",
        }
