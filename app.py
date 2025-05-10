import streamlit as st
from PIL import Image
import os
import zipfile
import tempfile
import numpy as np
import io
import cv2
import json
import logging
from registry import get_model
from core.describe_scene import describe_scene

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Streamlit Page Config
st.set_page_config(page_title="UVIS - Unified Visual Intelligence System", layout="wide")

# Logo and Title
st.sidebar.image("assets/ui/logo.png", width=180)
st.sidebar.markdown("## Visual Intelligence Engine")

# File Selection - Sample or Upload
SAMPLE_DIR = "assets/sample_images"
sample_images = [f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
sample_choice = st.sidebar.selectbox(" Select Sample Image or Upload", ["None"] + sample_images)
uploaded_file = None

if sample_choice != "None":
    # User chose a sample
    image_path = os.path.join(SAMPLE_DIR, sample_choice)
    image = Image.open(image_path).convert("RGB")
    st.image(image, caption=f"Sample: {sample_choice}", width=400)
    uploaded_file = None  # Ensure this is cleared

else:
    # User uploads their own image
    uploaded_file = st.file_uploader("Or Upload Your Own Image", type=["jpg", "jpeg", "png"])
    if not uploaded_file:
        st.info("Please upload an image (JPG, JPEG or PNG) to begin.")
        st.stop()
    if not uploaded_file.type.startswith("image"):
        st.error(f"Unsupported file type: {uploaded_file.type}. Please upload a valid image file.")
        st.stop()

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        input_path = tmp_file.name
    image = Image.open(input_path).convert("RGB")
    st.image(image, caption="Uploaded Image", width=400)

st.sidebar.subheader(" Select Tasks and Models")

# Detection Task
run_detection = st.sidebar.checkbox("Run Object Detection")
detection_model = None
if run_detection:
    detection_model = st.sidebar.selectbox("Detection Model", ["YOLOv5Nano", "YOLOv5"], index=0)

# Segmentation Task
run_segmentation = st.sidebar.checkbox("Run Semantic Segmentation")
segmentation_model = None
if run_segmentation:
    segmentation_model = st.sidebar.selectbox("Segmentation Model", ["FastSeg", "BiSeNetV2"], index=0)

# Depth Task
run_depth = st.sidebar.checkbox("Run Depth Estimation")
depth_model = None
if run_depth:
    depth_model = st.sidebar.selectbox("Depth Model", ["MiDaSSmall", "DPTLite"], index=0)


# Sidebar: Task Selection and Blend Control
st.sidebar.markdown("##  Perception Options")
blend_strength = st.sidebar.slider(" Overlay Blend Strength", 0.0, 1.0, 0.5)

# Pre-Run Tabs
tab1, tab2, tab3 = st.tabs([" Scene JSON", " Scene Blueprint", " Metrics"])

with tab1:
    st.subheader("Scene JSON")
    st.info(" Select tasks and click 'Run Analysis' to generate a structured scene description here.")

with tab2:
    st.subheader("Scene Blueprint")
    st.image(image, caption="Preview of Uploaded or Sample Image", use_container_width=True)

with tab3:
    st.subheader("Scene Complexity Rating")
    st.info(" Select tasks and click 'Run Analysis' to compute scene metrics here.")

# Process Button
if st.sidebar.button("Run Analysis"):
    combined_np = np.array(image)
    outputs_to_zip = {}
    scene_data = {}

    # Detection
    if run_detection:
        try:
            st.sidebar.markdown(f"🟡 **Running Object Detection with {detection_model}...**")
            with st.spinner(f"Processing {detection_model}..."):
                model = get_model("detection", detection_model, device="cpu")
                boxes = model.predict(image)
                overlay = model.draw(image, boxes)
                combined_np = np.array(overlay)
                buf = io.BytesIO()
                overlay.save(buf, format="PNG")
                outputs_to_zip["detection.png"] = buf.getvalue()
                scene_data["detection"] = boxes
                st.sidebar.markdown(f"✅ **Completed Object Detection with {detection_model}**")
        except Exception as e:
            st.sidebar.error(f"Error during Object Detection: {e}")
            logger.error(f"Error during Object Detection: {e}")

    # Segmentation
    if run_segmentation:
        try:
            st.sidebar.markdown(f"🟡 **Running Semantic Segmentation with {segmentation_model}...**")
            with st.spinner(f"Processing {segmentation_model}..."):
                model = get_model("segmentation", segmentation_model, device="cpu")
                mask = model.predict(image)
                overlay = model.draw(image, mask, alpha=blend_strength)
                combined_np = cv2.addWeighted(combined_np, 1 - blend_strength, np.array(overlay), blend_strength, 0)
                buf = io.BytesIO()
                overlay.save(buf, format="PNG")
                outputs_to_zip["segmentation.png"] = buf.getvalue()
                scene_data["segmentation"] = mask.tolist()
                st.sidebar.markdown(f"✅ **Completed Semantic Segmentation with {segmentation_model}**")
        except Exception as e:
            st.sidebar.error(f"Error during Semantic Segmentation: {e}")
            logger.error(f"Error during Semantic Segmentation: {e}")

    # Depth
    if run_depth:
        try:
            st.sidebar.markdown(f"🟡 **Running Depth Estimation with {depth_model}...**")
            with st.spinner(f"Processing {depth_model}..."):
                model = get_model("depth", depth_model, device="cpu")
                depth_map = model.predict(image)
                depth_img = ((depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map)) * 255).astype(np.uint8)
                depth_pil = Image.fromarray(depth_img)
                combined_np = cv2.addWeighted(combined_np, 1 - blend_strength, np.array(depth_pil.convert("RGB")), blend_strength, 0)
                buf = io.BytesIO()
                depth_pil.save(buf, format="PNG")
                outputs_to_zip["depth_map.png"] = buf.getvalue()
                scene_data["depth"] = depth_map.tolist()
                st.sidebar.markdown(f"✅ **Completed Depth Estimation with {depth_model}**")
        except Exception as e:
            st.sidebar.error(f"Error during Depth Estimation: {e}")
            logger.error(f"Error during Depth Estimation: {e}")

    # Scene Blueprint
    combined_pil = Image.fromarray(combined_np)
    buf = io.BytesIO()
    combined_pil.save(buf, format="PNG")
    outputs_to_zip["scene_blueprint.png"] = buf.getvalue()

    # Scene Description
    try:
        scene_json = describe_scene(**scene_data)
    except Exception as e:
        logger.warning(f"Scene description generation failed: {e}")
        scene_json = {"error": str(e)}

    outputs_to_zip["scene_description.json"] = json.dumps(scene_json, indent=2).encode("utf-8")

    # Tabs for Results
    tab1, tab2, tab3 = st.tabs([" Scene JSON", " Image with Overlays", " Metrics"])

    with tab1:
        st.subheader("Scene JSON")
        st.json(scene_json)

    with tab2:
        st.subheader("Scene Blueprint")
        st.image(combined_pil, caption="Unified Overlay", use_container_width=True)

    with tab3:
        st.subheader("Scene Complexity Rating")
        score = len(scene_data.get("detection", [])) + len(np.unique(scene_data.get("segmentation", [])))
        rating = "High" if score > 10 else "Medium" if score > 5 else "Low"
        st.markdown(f"###  Scene Complexity: **{rating}**")

        if "detection" in scene_data:
            st.markdown("###  Agent-Ready Summary")
            for obj in scene_data["detection"]:
                st.write(f"- Detected **{obj.get('class_name')}** with confidence {obj.get('confidence'):.2f}")

    # ZIP Download
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zipf:
        for filename, data in outputs_to_zip.items():
            zipf.writestr(filename, data)

    st.download_button(
        label="📦 Download Results as ZIP",
        data=zip_buf.getvalue(),
        file_name="uvis_results.zip",
        mime="application/zip"
    )
