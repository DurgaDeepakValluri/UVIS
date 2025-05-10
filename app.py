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
st.sidebar.image("assets/ui/uvis_logo.png", width=180)
st.sidebar.markdown("## Visual Intelligence Engine")

# File Selection - Sample or Upload
SAMPLE_DIR = "assets/sample_images"
sample_images = [f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
sample_choice = st.sidebar.selectbox("🎨 Select Sample Image or Upload", ["None"] + sample_images)
uploaded_file = None

if sample_choice != "None":
    image_path = os.path.join(SAMPLE_DIR, sample_choice)
    image = Image.open(image_path).convert("RGB")
    st.sidebar.image(image, caption=f"Sample: {sample_choice}", width=180)
else:
    uploaded_file = st.sidebar.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            image_path = tmp_file.name
        image = Image.open(image_path).convert("RGB")
        st.sidebar.image(image, caption="Uploaded Image", width=180)
    else:
        st.stop()

# Sidebar: Task Selection and Blend Control
st.sidebar.markdown("## 🛠️ Perception Options")
selected_tasks = st.sidebar.multiselect("Select Tasks", ["Object Detection", "Semantic Segmentation", "Depth Estimation"], default=["Object Detection"])
blend_strength = st.sidebar.slider("🔧 Overlay Blend Strength", 0.0, 1.0, 0.5)

# Process Button
if st.sidebar.button("🚀 Run Analysis"):
    combined_np = np.array(image)
    outputs_to_zip = {}
    scene_data = {}

    for task in selected_tasks:
        try:
            st.sidebar.markdown(f"🟡 **Running {task}...**")
            with st.spinner(f"Processing {task}..."):
                if task == "Object Detection":
                    model = get_model("detection", "YOLOv5Nano", device="cpu")
                    boxes = model.predict(image)
                    overlay = model.draw(image, boxes)
                    combined_np = np.array(overlay)
                    outputs_to_zip["detection.png"] = io.BytesIO()
                    overlay.save(outputs_to_zip["detection.png"], format="PNG")
                    outputs_to_zip["detection.png"] = outputs_to_zip["detection.png"].getvalue()
                    scene_data["detection"] = boxes

                elif task == "Semantic Segmentation":
                    model = get_model("segmentation", "FastSeg", device="cpu")
                    mask = model.predict(image)
                    overlay = model.draw(image, mask, alpha=blend_strength)
                    combined_np = cv2.addWeighted(combined_np, 1 - blend_strength, np.array(overlay), blend_strength, 0)
                    outputs_to_zip["segmentation.png"] = io.BytesIO()
                    overlay.save(outputs_to_zip["segmentation.png"], format="PNG")
                    outputs_to_zip["segmentation.png"] = outputs_to_zip["segmentation.png"].getvalue()
                    scene_data["segmentation"] = mask.tolist()

                elif task == "Depth Estimation":
                    model = get_model("depth", "MiDaSSmall", device="cpu")
                    depth_map = model.predict(image)
                    depth_img = ((depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map)) * 255).astype(np.uint8)
                    depth_pil = Image.fromarray(depth_img)
                    combined_np = cv2.addWeighted(combined_np, 1 - blend_strength, np.array(depth_pil.convert("RGB")), blend_strength, 0)
                    outputs_to_zip["depth_map.png"] = io.BytesIO()
                    depth_pil.save(outputs_to_zip["depth_map.png"], format="PNG")
                    outputs_to_zip["depth_map.png"] = outputs_to_zip["depth_map.png"].getvalue()
                    scene_data["depth"] = depth_map.tolist()

                st.sidebar.markdown(f"✅ **Completed {task}**")
        except Exception as e:
            st.sidebar.error(f"Error during {task}: {e}")
            logger.error(f"Error during {task}: {e}")
            continue

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
    tab1, tab2, tab3 = st.tabs(["📝 Scene JSON", "🖼️ Image with Overlays", "📊 Metrics"])

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
        st.markdown(f"### 🏆 Scene Complexity: **{rating}**")

        if "detection" in scene_data:
            st.markdown("### 🤖 Agent-Ready Summary")
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
