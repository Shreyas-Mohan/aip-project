# AI Knee Osteoarthritis Diagnostician

An end-to-end Machine Learning pipeline and diagnostic dashboard for classifying the severity of Knee Osteoarthritis based on the Kellgren-Lawrence (K-L) grading scale (Grades 0-4).

## Model Architecture & Training
The core of this repository is a Convolutional Neural Network (CNN) trained to classify X-ray images into 5 discrete categories:
- **Grade 0**: Normal
- **Grade 1**: Doubtful (Minute osteophytes, normal joint space)
- **Grade 2**: Mild (Definite osteophytes, normal joint space)
- **Grade 3**: Moderate (Moderate joint space narrowing)
- **Grade 4**: Severe (Large osteophytes, severe joint space narrowing)

The model was adapted from a local training pipeline utilizing an advanced pre-trained backbone architecture (e.g., Xception/DenseNet base), fine-tuned specifically to detect radiographic osteoarthritic features. The final weights and graph architecture are exported natively in the modern Keras 3 format (`final_model_export.keras` tracked via Git LFS).

## Image Processing Pipeline
During inference, raw user-uploaded X-rays pass through a strict preprocessing pipeline engineered to precisely match the training data distribution:
1. **Format Standardization**: Images are converted from grayscale/RGBA to standard 3-channel RGB to maintain architectural compatibility with ImageNet-based convolution layers.
2. **Resizing**: Images are fixedly resized to `128x128` pixels natively before being expanded with a batch dimension `(1, 128, 128, 3)`.
3. **Value Scaling Mapping**: Rather than strictly normalizing by `/ 255.0`, pixel intensities are fed directly as raw 0-255 inputs to perfectly align with the `tf.keras.preprocessing.image_dataset_from_directory` outputs utilized during the initial training cycle. This prevents zero-activation hallucination bias across the grading scale.

## Explainability: Grad-CAM Integration
To ensure clinical transparency and combat the "black box" nature of deep learning, we implemented **Gradient-weighted Class Activation Mapping (Grad-CAM)** logic natively inside `utils/model_handler.py`:
- We instantiate a `GradientTape` to isolate the final generic `Conv2D` layer in the nested CNN structure.
- We calculate the gradients of the top predicted classification output with respect to the spatial feature map activations.
- A global-average-pooled weighting is applied to create a localized 2D heatmap.
- This heatmap is resized, mapped via a Jet colormap, and alpha-blended over the original X-ray. It highlights the exact regions (typically joint space narrowing or asymmetrical cartilage wear) that the model fixated on to calculate the K-L grade.

## Features & Integration Overview
While wrapped in a lightweight, accessible Streamlit web interface, the application boasts heavy clinical and programmatic integrations:
- **Multi-Modal AI Context**: Connects the vision capabilities of the Gemini 2.5 Pro LLM to automatically synthesize the visual X-ray and the math-based CNN classification into a holistic clinical report.
- **Adaptive Reporting Tone**: Prompts are dynamically engineered to translate findings into either professional medical terminology (Doctor Mode) or empathetic, accessible language (Patient Mode).
- **Automated Document Generation**: Serializes Streamlit states into an interactive, downloadable clinical PDF record.
- **Modular Design Structure**: Organized into discrete micro-services—isolating model inference (`utils/model_handler.py`), LLM integration (`utils/llm_handler.py`), and document synthesis from the main routing logic (`app.py`).