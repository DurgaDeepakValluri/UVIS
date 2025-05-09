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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


st.set_page_config(page_title="UVIS - Unified Visual Intelligence System (Beta)", layout="wide")
st.image("assets/ui/uvis_logo.png", use_column_width=False)
st.title("UVIS - Unified Visual Intelligence System (Beta)")
st.write("Welcome to UVIS! Upload an image, select one or more tasks, and get a unified scene understanding. Currently supports lighter models of Depth Estimation, Object Detection, Semantic Segmentation and for images only. Will soon support heavier models, SLAM and video input too!")

# File upload
SAMPLE_DIR = "assets/sample_images"
sample_images = [f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

st.subheader("Or Select a Sample Image")
sample_choice = st.selectbox("Choose a Sample Image", ["None"] + sample_images)

image = None  # Initialize image

if sample_choice != "None":
    image_path = os.path.join(SAMPLE_DIR, sample_choice)
    image = Image.open(image_path).convert("RGB")
    st.image(image, caption=f"Sample: {sample_choice}", use_column_width=True)
else:
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

    try:
        image = Image.open(input_path).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
    except Exception as e:
        st.error(f"Failed to load image: {e}")
        st.stop()


with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
    tmp_file.write(uploaded_file.read())
    input_path = tmp_file.name

try:
    image = Image.open(input_path).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
except Exception as e:
    st.error(f"Failed to load image: {e}")
    st.stop()

blend_strength = st.slider("🔧 Adjust Overlay Blend Strength", 0.0, 1.0, 0.5)

# Multi-Task Selection
selected_tasks = st.multiselect("Select Tasks", ["Depth Estimation", "Object Detection", "Semantic Segmentation"])
if not selected_tasks:
    st.warning("Please select at least one task.")
    st.stop()

if st.button("Run Analysis"):
    combined_np = np.array(image)
    outputs_to_zip = {}
    scene_data = {}

    for task in selected_tasks:
        try:

            st.markdown(f" 🟡 **Starting {task}...**")

            with st.spinner(f"Running {task}..."):
                logger.info(f"Starting task: {task}")

                if task == "detection":
                    model = get_model("detection", "YOLOv5Nano", device="cpu")
                    boxes = model.predict(image)
                    overlay = model.draw(image, boxes)
                    combined_np = np.array(overlay)
                    buf = io.BytesIO()
                    overlay.save(buf, format="PNG")
                    outputs_to_zip["detectio.png"] = buf.getvalue()
                    scene_data["detection"] = boxes

                elif task == "segmentation":
                    model = get_model("segmentation", "FastSeg", device="cpu")
                    mask = model.predict(image)
                    overlay = model.draw(image, mask, alpha=0.5)
                    combined_np = cv2.addWeighted(combined_np, 1 - blend_strength, np.array(overlay), blend_strength, 0)
                    buf = io.BytesIO()
                    overlay.save(buf, format="PNG")
                    outputs_to_zip["segmentation.png"] = buf.getvalue()
                    scene_data["segmentation"] = mask.tolist()

                elif task == "depth":
                    model = get_model("depth", "MiDaSSmall", device="cpu")
                    depth_map = model.predict(image)
                    depth_img = ((depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map)) * 255).astype(np.uint8)
                    depth_pil = Image.fromarray(depth_img)
                    combined_np = cv2.addWeighted(combined_np, 1 - blend_strength, np.array(overlay), blend_strength, 0)
                    buf = io.BytesIO()
                    depth_pil.save(buf, format="PNG")
                    outputs_to_zip["depth_map.png"] = buf.getvalue()
                    scene_data["depth"] = depth_map.tolist()

                logger.info(f"Completed task: {task}")
                st.markdown(f" ✅ **Completed {task}**")


        except Exception as e:
            st.error(f"Error during {task}: {e}")
            logger.error(f"Error during {task}: {e}")
            continue


# Show Combined Preview
combined_pil = Image.fromarray(combined_np)
st.subheader("Scene Blueprint - Unified View of All Selected Tasks")
st.image(combined_pil, caption="Scene Blueprint", use_column_width=True)

# Save Combined Overlay
buf = io.BytesIO()
combined_pil.save(buf, format="PNG")
outputs_to_zip["Scene_Blueprint.png"] = buf.getvalue()

# Scene Description
try:
    scene_json = describe_scene(**scene_data)
except Exception as e:
    logger.warning(f"describe_scene() failed or not ready: {e}")
    scene_json = {"tasks_completed": list(scene_data.keys()), "note": "Scene description not implemented yet."}

# Show Scene Metrics to User
with st.expander("View Scene Metrics"):
    st.json(scene_json)

agent_summary = []
if "detection" in scene_data:
    for obj in scene_data["detection"]:
        agent_summary.append(f"Detected {obj.get('class_name')} with confidence {obj.get('confidence'):.2f}")

if agent_summary:
    st.subheader("Agent-Ready Summary")
    for line in agent_summary:
        st.write(line)

# Save Scene Description
outputs_to_zip["scene_description.json"] = json.dumps(scene_json, indent=2).encode("utf-8")

# Scene Complexity Rating
score = len(scene_data.get("detection", [])) + len(np.unique(scene_data.get("segmentation", [])))
rating = "High" if score > 10 else "Medium" if score > 5 else "Low"
st.markdown(f"### Scene Complexity: **{rating}**")


# Create ZIP Download
try:
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zipf:
        for filename, data in outputs_to_zip.items():
            zipf.writestr(filename, data)

    st.download_button(
        label=" Download Results as ZIP",
        data=zip_buf.getvalue(),
        file_name="uvis_results.zip",
        mime="application/zip"
    )
    logger.info("ZIP download prepared successfully.")
except Exception as e:
    st.error(f"Failed to create ZIP: {e}")
    logger.error(f"Failed to create ZIP: {e}")


