"""
Pneumonia Detection - Streamlit application.

Accepts a chest X-ray image and returns the predicted class with probabilities.
Run locally with:  streamlit run app.py
"""
import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from preprocess import prepare_image

HERE = Path(__file__).parent
CONFIG = json.loads((HERE / "labels.json").read_text())
CLASS_NAMES = CONFIG["class_names"]
BACKBONE = CONFIG["backbone"]
PNEUMONIA_IDX = CONFIG["pneumonia_class_index"]

st.set_page_config(page_title="Pneumonia Detection", page_icon="🫁", layout="centered")


@st.cache_resource
def load_model():
    """Load the model once and keep it in memory across reruns."""
    import tensorflow as tf
    return tf.keras.models.load_model(HERE / "model" / "best_model.keras")


st.title("Pneumonia Detection from Chest X-Rays")
st.caption(
    "Decision-support tool. Classifies a chest radiograph as Normal, "
    "abnormal-but-not-pneumonia, or Lung Opacity (pneumonia)."
)

st.warning(
    "**Not a diagnostic device.** This is a screening aid intended to support, not replace, "
    "radiologist interpretation. All predictions require clinical confirmation.",
    icon="⚠️",
)

uploaded = st.file_uploader(
    "Upload a chest X-ray image", type=["png", "jpg", "jpeg"],
    help="The model expects a frontal chest radiograph.",
)

if uploaded is not None:
    image = Image.open(uploaded)
    col_img, col_res = st.columns([1, 1])

    with col_img:
        st.image(image, caption="Uploaded radiograph", use_container_width=True)

    with st.spinner("Analysing..."):
        model = load_model()
        batch = prepare_image(np.array(image), BACKBONE)
        probabilities = model.predict(batch, verbose=0)[0]

    predicted_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_idx])

    with col_res:
        st.subheader("Prediction")
        if predicted_idx == PNEUMONIA_IDX:
            st.error(f"**{CLASS_NAMES[predicted_idx]}**", icon="🔴")
        else:
            st.success(f"**{CLASS_NAMES[predicted_idx]}**", icon="🟢")
        st.metric("Confidence", f"{confidence:.1%}")

    st.subheader("Probability across all classes")
    for name, p in zip(CLASS_NAMES, probabilities):
        st.write(f"**{name}** - {p:.2%}")
        st.progress(float(p))

    pneumonia_p = float(probabilities[PNEUMONIA_IDX])
    st.info(
        f"Probability of pneumonia: **{pneumonia_p:.1%}**. "
        "Higher values warrant prioritised radiologist review."
    )
else:
    st.info("Upload a chest X-ray image above to see a prediction.")

with st.expander("About this model"):
    st.write(f"""
    - **Backbone:** {BACKBONE}, pre-trained on ImageNet
    - **Task:** three-class classification
    - **Input:** 320 x 320, greyscale converted to three channels
    - **Classes:** {", ".join(CLASS_NAMES)}

    Trained on 21,347 chest radiographs. Selected on recall and F1 rather than accuracy,
    because a missed pneumonia case is considerably more costly than a false alarm.
    """)
