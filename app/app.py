# Importera bibliotek
import streamlit as st
import joblib
import numpy as np
from PIL import Image, ImageOps
import cv2
from pathlib import Path

# Skapa en första sida
st.set_page_config(page_title="MNIST bildigenkänning", layout="centered")

# sätt titel
st.title("MNIST känn igen siffror")
st.write("Ladda upp en bild eller ta en ny bild")

#ladda in modellen Extra Trees
BASE_DIR = Path(__file__).resolve().parents[1]

model_choice = st.selectbox(
    "Välj modell",
    ["Extra Trees", "SVC"]
)

if model_choice == "Extra Trees":
    MODEL_PATH = BASE_DIR / "models" / "Extra_trees_mnist_model.pkl"
else:
    MODEL_PATH = BASE_DIR / "models" / "SVC_model.pkl"

st.write("sökväg", MODEL_PATH)
st.write("dinns titeln?", MODEL_PATH.exists())

# skapa en funktioner för laddning av modell
@st.cache_resource
def load_model(path):
    return joblib.load(str(path))

#
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Kunde inte ladda modell: {e}")
    st.stop()

# Skapa input
img_file = st.file_uploader("LAdda upp en bild (png/jpg)", type=["png", "jpg", "jpeg"])
cam = st.camera_input("Eller ta en bild med kameran")

image = None
if img_file is not None:
    image = Image.open(img_file)
elif cam is not None:
    image = Image.open(cam)

# Förbehandling av bilder
def preprocess_to_mnist(pil_img: Image.Image):

    """
    Gör om bilderna till MNIST input:
    Konvertera till gråskala
    Binära
    hitta siffran, beskär och centrera
    omformatera till 28 x 28
    reurenra som array (1, 784) 
    värdena 0 - 255
    """
    # gråskala
    pil_grey = ImageOps.grayscale(pil_img)

    # omvandla till array
    img = np.array(pil_grey)

    # brusreducering
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Separera siffran från bakgrunden
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #om bakgrunden är vit inverterar vi färgerna.
    if np.sum(th == 255) > np.sum(th == 0):
        th = 255 - th

    # hitta konturerna (Siffran)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        resized = cv2.resize(th, (28, 28), interpolation=cv2.INTER_AREA)
        flat = resized.astype(np.float32).reshape(1, -1)

    # största konturen antas vara siffran
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    digit = th[y:y+h, x:x+w]

    # Gör bilden kvadratisk med padding så siffran blir rak
    size = max(w, h)
    square = np.zeros((size, size), dtype=np.uint8)
    x_off = (size - w)//2
    y_off = (size - h)//2
    square[y_off:y_off+h, x_off:x_off+w] = digit

    # Formatera om till 20 x 20 sen padding till 28 x 28
    digit_20 = cv2.resize(square, (20, 20), interpolation=cv2.INTER_AREA)
    padded = np.zeros((28, 28), dtype=np.uint8)
    padded[4:24, 4:24] = digit_20

    # platta till 784
    flat = padded.astype(np.float32).reshape(1, -1)
    return flat

if image is not None:
    st.subheader("Input")
    st.image(image, caption="Originalbild", use_container_width=True)

    X = preprocess_to_mnist(image)

    #Visa hur modellen ser bilden
    st.subheader("Föbehandlad")
    preview = X.reshape(28, 28).astype(np.uint8)
    st.image(preview, caption="MNIST-format (vit siffra på svart bakgrund)", use_column_width=False)

    # Prediktion
    pred = model.predict(X)[0]
    st.success(f"predikterad siffra: **{pred}**")

    import pandas as pd

    st.subheader("Modellens säkerhet (Top 3)")

    probs = None

    # 1️⃣ Om modellen har riktiga sannolikheter
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]

    # 2️⃣ Om SVC utan probability=True
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores = np.ravel(scores)

        # stabil softmax
        scores = scores - np.max(scores)
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores)

    if probs is not None:
        # skapa pandas-serie med etiketter 0–9
        probs_series = pd.Series(probs, index=list(range(10)))

        # sortera och ta topp 3
        top3 = probs_series.sort_values(ascending=False).head(3)

        # konvertera till procent
        top3_percent = top3 * 100

        # visa stapeldiagram
        st.bar_chart(top3_percent)

        # visa tydlig text
        best_class = top3.index[0]
        best_prob = top3_percent.iloc[0]

        st.markdown(
            f"### Mest sannolik: **{best_class}** ({best_prob:.1f}%)"
        )

    else:
        st.info("Modellen stödjer inte sannolikhetsvisning.")
