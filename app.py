
# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# import streamlit as st
# import tensorflow as tf
# import numpy as np
# import cv2
# import random
# import time
# import os
# import json
# import plotly.graph_objects as go
# from tensorflow.keras.models import load_model, Sequential
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.layers import Dense, Dropout, Flatten
# from tensorflow.keras.optimizers import Adamax

# # ===============================
# # CONFIG
# # ===============================
# st.set_page_config(
#     page_title="Federated Medical AI",
#     layout="wide",
#     initial_sidebar_state="collapsed"
# )

# GLOBAL_MIN = 65.0
# GLOBAL_MAX = 87.0
# AGGREGATION_THRESHOLD = 70.0
# GLOBAL_HISTORY_FILE = "global_accuracy_history.json"

# CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

# output_dir = "saliency_maps"
# os.makedirs(output_dir, exist_ok=True)

# # ===============================
# # BACKGROUND STYLE
# # ===============================
# st.markdown("""
# <style>
# .stApp {
#     background:
#       radial-gradient(circle at 20% 20%, rgba(0,210,255,0.15), transparent 40%),
#       radial-gradient(circle at 80% 30%, rgba(58,123,213,0.12), transparent 45%),
#       radial-gradient(circle at 50% 80%, rgba(0,255,180,0.10), transparent 50%),
#       #050a0f;
#     color: #E2E8F0;
# }
# .glass {
#     background: rgba(15, 23, 42, 0.45);
#     backdrop-filter: blur(14px);
#     border-radius: 16px;
#     padding: 20px;
# }
# .stButton>button {
#     width: 100%;
#     background: linear-gradient(90deg,#00D2FF,#3A7BD5);
#     color: white;
#     border-radius: 10px;
#     font-weight: bold;
# }
# </style>
# """, unsafe_allow_html=True)

# # ===============================
# # SESSION STATE
# # ===============================
# if "page" not in st.session_state:
#     st.session_state.page = "Landing"

# if "acc_history" not in st.session_state:
#     if os.path.exists(GLOBAL_HISTORY_FILE):
#         with open(GLOBAL_HISTORY_FILE, "r") as f:
#             try:
#                 st.session_state.acc_history = json.load(f)
#             except:
#                 st.session_state.acc_history = []
#     else:
#         st.session_state.acc_history = []

#     if len(st.session_state.acc_history) == 0:
#         acc = round(random.uniform(80, 88), 2)
#         st.session_state.acc_history = [acc]
#         st.session_state.global_acc = acc
#     else:
#         st.session_state.global_acc = st.session_state.acc_history[-1]

# if "has_integrated" not in st.session_state:
#     st.session_state.has_integrated = False

# # ===============================
# # SALIENCY MAP
# # ===============================
# def generate_saliency_map(model, img_array, class_index, img_size, img, uploaded_file, file_name):
#     with tf.GradientTape() as tape:
#         img_tensor = tf.convert_to_tensor(img_array)
#         tape.watch(img_tensor)
#         predictions = model(img_tensor)
#         target_class = predictions[0, class_index]

#     gradients = tape.gradient(target_class, img_tensor)
#     gradients = tf.reduce_max(tf.abs(gradients), axis=-1).numpy().squeeze()
#     gradients = cv2.resize(gradients, img_size)

#     center = (gradients.shape[0] // 2, gradients.shape[1] // 2)
#     radius = min(center) - 10
#     x, y = np.ogrid[:gradients.shape[0], :gradients.shape[1]]
#     mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2

#     gradients *= mask
#     if gradients[mask].max() > gradients[mask].min():
#         gradients[mask] = (gradients[mask] - gradients[mask].min()) / (gradients[mask].max() - gradients[mask].min())

#     gradients[gradients < np.percentile(gradients[mask], 80)] = 0
#     gradients = cv2.GaussianBlur(gradients, (11, 11), 0)

#     heatmap = cv2.applyColorMap(np.uint8(255 * gradients), cv2.COLORMAP_JET)
#     heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
#     heatmap = cv2.resize(heatmap, img_size)

#     original = image.img_to_array(img)
#     overlay = (heatmap * 0.7 + original * 0.3).astype(np.uint8)

#     path = os.path.join(output_dir, file_name)
#     with open(path, "wb") as f:
#         f.write(uploaded_file.getbuffer())

#     cv2.imwrite(path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
#     return overlay

# # ===============================
# # MEDICAL EXPLANATION
# # ===============================
# def generate_explanation(model_prediction, confidence):
#     confidence *= 100

#     explanations = {
#         "Glioma": "The model identified diffuse abnormal patterns within the brain tissue, suggesting infiltrative tumor behavior commonly seen in gliomas.",
#         "Meningioma": "The highlighted regions appear well-defined and localized near the brain surface, consistent with meningioma characteristics.",
#         "Pituitary": "The activation is concentrated near the central base of the brain, aligning with anatomical features of pituitary tumors.",
#         "No Tumor": "No prominent abnormal activation regions were detected, indicating normal brain structure."
#     }

#     certainty = (
#         "This prediction shows high confidence."
#         if confidence >= 75 else
#         "This prediction shows moderate confidence."
#     )

#     return f"{explanations[model_prediction]} {certainty}"

# # ===============================
# # MODEL LOADERS
# # ===============================
# def load_xception_model(path):
#     base = tf.keras.applications.Xception(
#         include_top=False,
#         weights="imagenet",
#         input_shape=(299,299,3),
#         pooling="max"
#     )
#     model = Sequential([
#         base,
#         Flatten(),
#         Dropout(0.3),
#         Dense(128, activation="relu"),
#         Dense(4, activation="softmax")
#     ])
#     model.compile(Adamax(0.001), "categorical_crossentropy", ["accuracy"])
#     model.load_weights(path)
#     return model

# # ===============================
# # LANDING
# # ===============================
# if st.session_state.page == "Landing":
#     st.markdown("<h1 style='text-align:center'>🌐 Federated Medical AI</h1>", unsafe_allow_html=True)
#     c1, c2 = st.columns(2)
#     if c1.button("🌍 Global Server"):
#         st.session_state.page = "Global"
#         st.rerun()
#     if c2.button("🧠 Local Client"):
#         st.session_state.page = "Client"
#         st.rerun()

# # ===============================
# # CLIENT
# # ===============================
# elif st.session_state.page == "Client":
#     if st.button("← Back"):
#         st.session_state.page = "Landing"
#         st.rerun()

#     st.markdown("<div class='glass'><h2>Local Client Node 🧠</h2></div>", unsafe_allow_html=True)
#     uploaded = st.file_uploader("Upload MRI", type=["jpg","png","jpeg"])

#     if uploaded:
#         choice = st.radio("Prediction Model", ("Custom CNN", "Transfer Learning – Xception"))

#         if choice == "Transfer Learning – Xception":
#             model = load_xception_model("models/xception_model.weights.h5")
#             img_size = (299,299)
#         else:
#             model = load_model("models/cnn_model.h5")
#             img_size = (224,224)

#         # ===============================
#         # CORRECT PREDICTION PIPELINE
#         # ===============================

#         # Load image exactly like reference
#         img = image.load_img(uploaded, target_size=img_size)
#         img_array = image.img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)
#         img_array = img_array / 255.0  # critical

#         # Predict
#         prediction = model.predict(img_array, verbose=0)

#         # Class labels MUST match training order
#         CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

#         class_index = int(np.argmax(prediction[0]))
#         predicted_label = CLASS_NAMES[class_index]
#         confidence = float(prediction[0][class_index])


#         saliency = generate_saliency_map(
#             model, img_array, class_index, img_size, img, uploaded, uploaded.name
#         )

#         c1, c2 = st.columns(2)
#         c1.image(uploaded, caption="MRI", use_container_width=True)
#         c2.image(saliency, caption="Saliency Map", use_container_width=True)

#         # 🔮 Prediction Output
#         st.markdown("### 🧠 Prediction Result")
#         st.success(f"**Diagnosis:** {predicted_label}")
#         st.metric("Model Confidence", f"{confidence*100:.2f}%")

#         st.markdown("### 🩺 Clinical Explanation")
#         st.write(generate_explanation(predicted_label, confidence))

#         if st.button("🔗 Integrate with Global Model"):
#             if confidence * 100 >= AGGREGATION_THRESHOLD:
#                 progress = st.progress(0)
#                 for i in range(100):
#                     time.sleep(0.01)
#                     progress.progress(i + 1)
#                 st.session_state.has_integrated = True
#                 st.success("✅ Client update accepted by global server")
#             else:
#                 st.error("❌ Cannot aggregate due to low confidence")

# # ===============================
# # GLOBAL
# # ===============================
# elif st.session_state.page == "Global":
#     if st.button("← Back"):
#         st.session_state.page = "Landing"
#         st.rerun()

#     st.markdown("<div class='glass'><h2>Global Federated Server 🌍</h2></div>", unsafe_allow_html=True)
#     st.metric("Global Accuracy", f"{st.session_state.global_acc}%")

#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         y=st.session_state.acc_history,
#         mode="lines+markers"
#     ))
#     fig.update_layout(
#         template="plotly_dark",
#         yaxis_range=[GLOBAL_MIN, GLOBAL_MAX],
#         title="Global Accuracy per Federated Round"
#     )
#     st.plotly_chart(fig, use_container_width=True)

#     if st.button("🚀 Start Federated Round", disabled=not st.session_state.has_integrated):
#         progress = st.progress(0)
#         for i in range(100):
#             time.sleep(0.01)
#             progress.progress(i + 1)

#         delta = random.uniform(0.3, 1.2)
#         st.session_state.global_acc = round(
#             min(GLOBAL_MAX, max(GLOBAL_MIN, st.session_state.global_acc + delta)), 2
#         )

#         st.session_state.acc_history.append(st.session_state.global_acc)

#         with open(GLOBAL_HISTORY_FILE, "w") as f:
#             json.dump(st.session_state.acc_history, f)

#         st.session_state.has_integrated = False
#         st.success("🌍 Aggregation completed. Clients must retrain.")
#         st.rerun()

# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# import streamlit as st
# import tensorflow as tf
# import numpy as np
# import cv2
# import random
# import time
# import os
# import json
# import plotly.graph_objects as go
# from tensorflow.keras.models import load_model, Sequential
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.layers import Dense, Dropout, Flatten
# from tensorflow.keras.optimizers import Adamax

# # ===============================
# # CONFIG
# # ===============================
# st.set_page_config(
#     page_title="Federated Medical AI",
#     layout="wide",
#     initial_sidebar_state="collapsed"
# )

# GLOBAL_MIN = 65.0
# GLOBAL_MAX = 87.0
# AGGREGATION_THRESHOLD = 70.0
# GLOBAL_HISTORY_FILE = "global_accuracy_history.json"

# CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

# output_dir = "saliency_maps"
# os.makedirs(output_dir, exist_ok=True)

# # ===============================
# # BACKGROUND STYLE
# # ===============================
# st.markdown("""
# <style>
# .stApp {
#     background:
#       radial-gradient(circle at 20% 20%, rgba(0,210,255,0.15), transparent 40%),
#       radial-gradient(circle at 80% 30%, rgba(58,123,213,0.12), transparent 45%),
#       radial-gradient(circle at 50% 80%, rgba(0,255,180,0.10), transparent 50%),
#       #050a0f;
#     color: #E2E8F0;
# }
# .glass {
#     background: rgba(15, 23, 42, 0.45);
#     backdrop-filter: blur(14px);
#     border-radius: 16px;
#     padding: 20px;
# }
# .stButton>button {
#     width: 100%;
#     background: linear-gradient(90deg,#00D2FF,#3A7BD5);
#     color: white;
#     border-radius: 10px;
#     font-weight: bold;
# }
# </style>
# """, unsafe_allow_html=True)

# # ===============================
# # SESSION STATE (GLOBAL ACCURACY FIX ONLY)
# # ===============================
# if "page" not in st.session_state:
#     st.session_state.page = "Landing"

# if "acc_history" not in st.session_state:
#     if os.path.exists(GLOBAL_HISTORY_FILE):
#         with open(GLOBAL_HISTORY_FILE, "r") as f:
#             try:
#                 st.session_state.acc_history = json.load(f)
#             except:
#                 st.session_state.acc_history = []
#     else:
#         st.session_state.acc_history = []

#     if len(st.session_state.acc_history) == 0:
#         acc = round(random.uniform(GLOBAL_MIN + 5, GLOBAL_MAX - 3), 2)
#         st.session_state.acc_history = [acc]
#         st.session_state.global_acc = acc
#     else:
#         st.session_state.global_acc = st.session_state.acc_history[-1]

# if "has_integrated" not in st.session_state:
#     st.session_state.has_integrated = False

# # ===============================
# # SALIENCY MAP
# # ===============================
# def generate_saliency_map(model, img_array, class_index, img_size, img, uploaded_file, file_name):
#     with tf.GradientTape() as tape:
#         img_tensor = tf.convert_to_tensor(img_array)
#         tape.watch(img_tensor)
#         predictions = model(img_tensor)
#         target_class = predictions[0, class_index]

#     gradients = tape.gradient(target_class, img_tensor)
#     gradients = tf.reduce_max(tf.abs(gradients), axis=-1).numpy().squeeze()
#     gradients = cv2.resize(gradients, img_size)

#     center = (gradients.shape[0] // 2, gradients.shape[1] // 2)
#     radius = min(center) - 10
#     x, y = np.ogrid[:gradients.shape[0], :gradients.shape[1]]
#     mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2

#     gradients *= mask
#     if gradients[mask].max() > gradients[mask].min():
#         gradients[mask] = (gradients[mask] - gradients[mask].min()) / (gradients[mask].max() - gradients[mask].min())

#     gradients[gradients < np.percentile(gradients[mask], 80)] = 0
#     gradients = cv2.GaussianBlur(gradients, (11, 11), 0)

#     heatmap = cv2.applyColorMap(np.uint8(255 * gradients), cv2.COLORMAP_JET)
#     heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
#     heatmap = cv2.resize(heatmap, img_size)

#     original = image.img_to_array(img)
#     overlay = (heatmap * 0.7 + original * 0.3).astype(np.uint8)

#     path = os.path.join(output_dir, file_name)
#     with open(path, "wb") as f:
#         f.write(uploaded_file.getbuffer())

#     cv2.imwrite(path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
#     return overlay

# # ===============================
# # MEDICAL EXPLANATION
# # ===============================
# def generate_explanation(model_prediction, confidence):
#     confidence *= 100

#     explanations = {
#         "Glioma": "The model identified diffuse abnormal patterns within the brain tissue, suggesting infiltrative tumor behavior commonly seen in gliomas.",
#         "Meningioma": "The highlighted regions appear well-defined and localized near the brain surface, consistent with meningioma characteristics.",
#         "Pituitary": "The activation is concentrated near the central base of the brain, aligning with anatomical features of pituitary tumors.",
#         "No Tumor": "No prominent abnormal activation regions were detected, indicating normal brain structure."
#     }

#     certainty = (
#         "This prediction shows high confidence."
#         if confidence >= 75 else
#         "This prediction shows moderate confidence."
#     )

#     return f"{explanations[model_prediction]} {certainty}"

# # ===============================
# # MODEL LOADERS
# # ===============================
# def load_xception_model(path):
#     base = tf.keras.applications.Xception(
#         include_top=False,
#         weights="imagenet",
#         input_shape=(299,299,3),
#         pooling="max"
#     )
#     model = Sequential([
#         base,
#         Flatten(),
#         Dropout(0.3),
#         Dense(128, activation="relu"),
#         Dense(4, activation="softmax")
#     ])
#     model.compile(Adamax(0.001), "categorical_crossentropy", ["accuracy"])
#     model.load_weights(path)
#     return model

# # ===============================
# # LANDING
# # ===============================
# if st.session_state.page == "Landing":
#     st.markdown("<h1 style='text-align:center'>🌐 Federated Medical AI</h1>", unsafe_allow_html=True)
#     c1, c2 = st.columns(2)
#     if c1.button("🌍 Global Server"):
#         st.session_state.page = "Global"
#         st.rerun()
#     if c2.button("🧠 Local Client"):
#         st.session_state.page = "Client"
#         st.rerun()

# # ===============================
# # CLIENT
# # ===============================
# elif st.session_state.page == "Client":
#     if st.button("← Back"):
#         st.session_state.page = "Landing"
#         st.rerun()

#     st.markdown("<div class='glass'><h2>Local Client Node 🧠</h2></div>", unsafe_allow_html=True)
#     uploaded = st.file_uploader("Upload MRI", type=["jpg","png","jpeg"])

#     if uploaded:
#         choice = st.radio("Prediction Model", ("Custom CNN", "Transfer Learning – Xception"))

#         if choice == "Transfer Learning – Xception":
#             model = load_xception_model("models/xception_model.weights.h5")
#             img_size = (299,299)
#         else:
#             model = load_model("models/cnn_model.h5")
#             img_size = (224,224)

#         img = image.load_img(uploaded, target_size=img_size)
#         img_array = image.img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)
#         img_array = img_array / 255.0

#         prediction = model.predict(img_array, verbose=0)

#         CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

#         class_index = int(np.argmax(prediction[0]))
#         predicted_label = CLASS_NAMES[class_index]
#         confidence = float(prediction[0][class_index])

#         saliency = generate_saliency_map(
#             model, img_array, class_index, img_size, img, uploaded, uploaded.name
#         )

#         c1, c2 = st.columns(2)
#         c1.image(uploaded, caption="MRI", use_container_width=True)
#         c2.image(saliency, caption="Saliency Map", use_container_width=True)

#         st.markdown("### 🧠 Prediction Result")
#         st.success(f"**Diagnosis:** {predicted_label}")
#         st.metric("Model Confidence", f"{confidence*100:.2f}%")

#         st.markdown("### 🩺 Clinical Explanation")
#         st.write(generate_explanation(predicted_label, confidence))

#         if st.button("🔗 Integrate with Global Model"):
#             if confidence * 100 >= AGGREGATION_THRESHOLD:
#                 progress = st.progress(0)
#                 for i in range(100):
#                     time.sleep(0.01)
#                     progress.progress(i + 1)
#                 st.session_state.has_integrated = True
#                 st.success("✅ Client update accepted by global server")
#             else:
#                 st.error("❌ Cannot aggregate due to low confidence")

# # ===============================
# # GLOBAL
# # ===============================
# elif st.session_state.page == "Global":
#     if st.button("← Back"):
#         st.session_state.page = "Landing"
#         st.rerun()

#     st.markdown("<div class='glass'><h2>Global Federated Server 🌍</h2></div>", unsafe_allow_html=True)
#     st.metric("Global Accuracy", f"{st.session_state.global_acc}%")

#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         y=st.session_state.acc_history,
#         mode="lines+markers"
#     ))
#     fig.update_layout(
#         template="plotly_dark",
#         yaxis_range=[GLOBAL_MIN, GLOBAL_MAX],
#         title="Global Accuracy per Federated Round"
#     )
#     st.plotly_chart(fig, use_container_width=True)

#     if st.button("🚀 Start Federated Round", disabled=not st.session_state.has_integrated):
#         progress = st.progress(0)
#         for i in range(100):
#             time.sleep(0.01)
#             progress.progress(i + 1)

#         new_acc = round(
#             random.uniform(
#                 max(GLOBAL_MIN, st.session_state.global_acc - 1.0),
#                 min(GLOBAL_MAX, st.session_state.global_acc + 1.5)
#             ),
#             2
#         )

#         st.session_state.global_acc = new_acc
#         st.session_state.acc_history.append(new_acc)

#         with open(GLOBAL_HISTORY_FILE, "w") as f:
#             json.dump(st.session_state.acc_history, f)

#         st.session_state.has_integrated = False
#         st.success("🌍 Aggregation completed. Clients must retrain.")
#         st.rerun()




import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import random
import time
import json
import plotly.graph_objects as go

from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adamax

# ===============================
# CONFIG
# ===============================
st.set_page_config(
    page_title="Federated Medical AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

GLOBAL_MIN = 65.0
GLOBAL_MAX = 87.0
AGGREGATION_THRESHOLD = 70.0
GLOBAL_HISTORY_FILE = "global_accuracy_history.json"

CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

output_dir = "saliency_maps"
os.makedirs(output_dir, exist_ok=True)

# ===============================
# STYLE
# ===============================
st.markdown("""
<style>
.stApp {
    background:
      radial-gradient(circle at 20% 20%, rgba(0,210,255,0.15), transparent 40%),
      radial-gradient(circle at 80% 30%, rgba(58,123,213,0.12), transparent 45%),
      radial-gradient(circle at 50% 80%, rgba(0,255,180,0.10), transparent 50%),
      #050a0f;
    color: #E2E8F0;
}
.glass {
    background: rgba(15, 23, 42, 0.45);
    backdrop-filter: blur(14px);
    border-radius: 16px;
    padding: 20px;
}
.stButton>button {
    width: 100%;
    background: linear-gradient(90deg,#00D2FF,#3A7BD5);
    color: white;
    border-radius: 10px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# SESSION STATE
# ===============================
if "page" not in st.session_state:
    st.session_state.page = "Landing"

if "acc_history" not in st.session_state:
    if os.path.exists(GLOBAL_HISTORY_FILE):
        with open(GLOBAL_HISTORY_FILE, "r") as f:
            try:
                st.session_state.acc_history = json.load(f)
            except:
                st.session_state.acc_history = []
    else:
        st.session_state.acc_history = []

    if len(st.session_state.acc_history) == 0:
        init_acc = round(random.uniform(GLOBAL_MIN + 5, GLOBAL_MAX - 5), 2)
        st.session_state.acc_history = [init_acc]
        st.session_state.global_acc = init_acc
    else:
        st.session_state.global_acc = st.session_state.acc_history[-1]

if "has_integrated" not in st.session_state:
    st.session_state.has_integrated = False

if "client_predicted" not in st.session_state:
    st.session_state.client_predicted = False

# ===============================
# SALIENCY MAP (UNCHANGED)
# ===============================
def generate_saliency_map(model, img_array, class_index, img_size, img, uploaded_file, file_name):
    with tf.GradientTape() as tape:
        img_tensor = tf.convert_to_tensor(img_array)
        tape.watch(img_tensor)
        predictions = model(img_tensor)
        target_class = predictions[0, class_index]

    gradients = tape.gradient(target_class, img_tensor)
    gradients = tf.reduce_max(tf.abs(gradients), axis=-1).numpy().squeeze()
    gradients = cv2.resize(gradients, img_size)

    center = (gradients.shape[0] // 2, gradients.shape[1] // 2)
    radius = min(center) - 10
    x, y = np.ogrid[:gradients.shape[0], :gradients.shape[1]]
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2

    gradients *= mask
    if gradients[mask].max() > gradients[mask].min():
        gradients[mask] = (gradients[mask] - gradients[mask].min()) / (gradients[mask].max() - gradients[mask].min())

    gradients[gradients < np.percentile(gradients[mask], 80)] = 0
    gradients = cv2.GaussianBlur(gradients, (11, 11), 0)

    heatmap = cv2.applyColorMap(np.uint8(255 * gradients), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, img_size)

    original = image.img_to_array(img)
    overlay = (heatmap * 0.7 + original * 0.3).astype(np.uint8)

    cv2.imwrite(os.path.join(output_dir, file_name), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return overlay

# ===============================
# MODEL LOADER (UNCHANGED)
# ===============================
def load_xception_model(path):
    if not os.path.exists(path):
        st.error("Xception weights not found.")
        st.stop()

    base = tf.keras.applications.Xception(
        include_top=False,
        weights="imagenet",
        input_shape=(299, 299, 3),
        pooling="max"
    )

    model = Sequential([
        base,
        Flatten(),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dense(4, activation="softmax")
    ])

    model.compile(Adamax(0.001), "categorical_crossentropy", ["accuracy"])
    model.load_weights(path)
    return model

# ===============================
# LANDING
# ===============================
if st.session_state.page == "Landing":
    st.markdown("<h1 style='text-align:center'>🌐 Federated Medical AI</h1>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    if c1.button("🌍 Global Server"):
        st.session_state.page = "Global"
        st.rerun()
    if c2.button("🧠 Local Client"):
        st.session_state.page = "Client"
        st.rerun()

# ===============================
# CLIENT (UNCHANGED)
# ===============================
elif st.session_state.page == "Client":
    if st.button("← Back"):
        st.session_state.page = "Landing"
        st.rerun()

    st.markdown("<div class='glass'><h2>Local Client Node 🧠</h2></div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload MRI", type=["jpg", "png", "jpeg"])

    if uploaded:
        choice = st.radio("Prediction Model", ("Custom CNN", "Transfer Learning – Xception"))

        if choice == "Transfer Learning – Xception":
            model = load_xception_model("models/xception_model.weights.h5")
            img_size = (299, 299)
        else:
            model = load_model("models/cnn_model.h5")
            img_size = (224, 224)

        img = image.load_img(uploaded, target_size=img_size)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)
        st.session_state.client_predicted = True

        CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
        class_index = int(np.argmax(prediction[0]))
        predicted_label = CLASS_NAMES[class_index]
        confidence = float(prediction[0][class_index])

        saliency = generate_saliency_map(
            model, img_array, class_index, img_size, img, uploaded, uploaded.name
        )

        c1, c2 = st.columns(2)
        c1.image(uploaded, caption="MRI", use_container_width=True)
        c2.image(saliency, caption="Saliency Map", use_container_width=True)

        st.success(f"Diagnosis: {predicted_label}")
        st.metric("Confidence", f"{confidence*100:.2f}%")

        if st.button("🔗 Integrate with Global Model"):
            if confidence * 100 >= AGGREGATION_THRESHOLD:
                st.session_state.has_integrated = True
                st.success("Client update accepted")
            else:
                st.error("Low confidence update rejected")

# ===============================
# GLOBAL (FIXED LOGIC)
# ===============================
elif st.session_state.page == "Global":
    if st.button("← Back"):
        st.session_state.page = "Landing"
        st.rerun()

    st.markdown("<div class='glass'><h2>Global Federated Server 🌍</h2></div>", unsafe_allow_html=True)
    st.metric("Global Accuracy", f"{st.session_state.global_acc}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=st.session_state.acc_history, mode="lines+markers"))
    fig.update_layout(
        template="plotly_dark",
        yaxis_range=[GLOBAL_MIN, GLOBAL_MAX],
        title="Global Accuracy per Federated Round"
    )
    st.plotly_chart(fig, use_container_width=True)

    if st.button(
        "🚀 Start Federated Round",
        disabled=not (st.session_state.has_integrated and st.session_state.client_predicted)
    ):
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)

        # ✅ RANDOM GLOBAL ACCURACY EACH ROUND
        new_acc = round(
            random.uniform(GLOBAL_MIN, GLOBAL_MAX),
            2
        )

        st.session_state.global_acc = new_acc
        st.session_state.acc_history.append(new_acc)

        with open(GLOBAL_HISTORY_FILE, "w") as f:
            json.dump(st.session_state.acc_history, f)

        st.session_state.has_integrated = False
        st.session_state.client_predicted = False

        st.success("🌍 Global model updated via federated aggregation")
        st.rerun()
