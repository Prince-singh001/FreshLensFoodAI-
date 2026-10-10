import os
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROD_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../models/production"))

models_to_convert = [
    {
        "h5": os.path.join(PROD_DIR, "foodlens_mobilenetv2_model.h5") if os.path.exists(os.path.join(PROD_DIR, "foodlens_mobilenetv2_model.h5")) else os.path.join(PROD_DIR, "freshlens_mobilenetv2_model.h5"),
        "tflite": os.path.join(PROD_DIR, "foodlens_mobilenetv2_model.tflite"),
        "name": "FoodLens Freshness Classifier"
    },
    {
        "h5": os.path.join(PROD_DIR, "food_validator_model.h5"),
        "tflite": os.path.join(PROD_DIR, "food_validator_model.tflite"),
        "name": "Food vs Non-Food Validator"
    }
]

for item in models_to_convert:
    h5_path = item["h5"]
    tflite_path = item["tflite"]
    name = item["name"]

    if not os.path.exists(h5_path):
        print(f"Skipping {name}: {h5_path} does not exist.")
        continue

    print(f"\nLoading Keras model for {name} from {h5_path}...")
    model = tf.keras.models.load_model(h5_path, compile=False)

    print("Converting to TFLite with size optimization...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    print(f"Saving TFLite model to {tflite_path} ({len(tflite_model) // 1024} KB)...")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

print("\nTFLite conversion complete for all available models!")
