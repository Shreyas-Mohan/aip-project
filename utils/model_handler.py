import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import Model
import cv2
import PIL.Image
import glob

MODEL_PATH = "./final_model_export.keras"
CLASS_LABELS = ["Normal (Grade 0)", "Doubtful (Grade 1)", "Mild (Grade 2)", "Moderate (Grade 3)", "Severe (Grade 4)"]

def load_cnn_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((128, 128))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # img_array /= 255.0  # Removing normalization as the base model was trained on 0-255 RGB values in this specific notebook
    return img_array

def generate_gradcam(img_array, full_model):
    base_model = full_model.layers[0] 
    
    last_conv_layer_name = None
    for layer in reversed(base_model.layers):
        # We only want the last valid Conv2D-like layer (excludes dense, flatten)
        if layer.__class__.__name__ == 'Conv2D':
            last_conv_layer_name = layer.name
            break
            
    if not last_conv_layer_name:
        return None
        
    grad_model = Model(
        inputs=base_model.inputs,  # Fixed nested list bug
        outputs=[base_model.get_layer(last_conv_layer_name).output, base_model.output]
    )
    
    with tf.GradientTape() as tape:
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        conv_outputs, base_output = grad_model(img_tensor, training=False)
        tape.watch(conv_outputs)
        
        # Pass the base model output through the remaining classifier layers
        preds = base_output
        for layer in full_model.layers[1:]:
            preds = layer(preds, training=False)
            
        # Target the top predicted class
        top_pred_index = tf.argmax(preds[0])
        loss = preds[:, top_pred_index]
        
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def apply_heatmap(original_img, heatmap, alpha=0.4):
    original_img = np.array(original_img.convert('RGB'))
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    superimposed_img = heatmap * alpha + original_img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return PIL.Image.fromarray(superimposed_img)

def get_reference_healthy_image():
    """Finds a healthy knee image from the local dataset folder."""
    # We use a static asset for the cloud deployed app, as dataset-kaggle isn't pushed
    asset_path = './assets/reference_healthy_knee.png'
    if os.path.exists(asset_path):
        return asset_path
        
    # Fallback to local dataset if asset isn't created yet
    files = glob.glob('./dataset-kaggle/train/0/*.*')
    if files:
        return files[0]
    return None
