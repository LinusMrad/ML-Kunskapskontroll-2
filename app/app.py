# Importera bibliotek
import streamlit as st
import joblib
import numpy as np
from PIL import Image, ImageOps
import cv2
from pathlib import Path
import pandas as pd

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
img_file = None
cam = None

img_choice = st.radio(
    "Välj metod",
    ["📁 Ladda upp en bild", "📷 Ta en bild"]
)

if img_choice == "📁 Ladda upp en bild":
    img_file = st.file_uploader("Ladda upp en bild (png/jpg)", type=["png", "jpg", "jpeg"])
else:    
    cam = st.camera_input("Ta en bild med kameran")

image = None
if img_file is not None:
    image = Image.open(img_file)
elif cam is not None:
    image = Image.open(cam)

def remove_ruled_lines(gray: np.ndarray):
    """Försöker ta bort linjer på linjerat papper
    reutnerar en bild i gråskala där linjer är dämpade eller borta"""

    # Blur för att minska brus
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Binär, inverterad ("bläck" blir vitt)
    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 7
    )

    # Hitta linjer med morfologi
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    h_lines = cv2.morphologyEx(th, cv2.MORPH_OPEN, h_kernel, iterations=1)

    # Hitta vertikala linjer
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    v_lines = cv2.morphologyEx(th, cv2.MORPH_OPEN, v_kernel, iterations=1)

    # mask av linjer
    lines = cv2.bitwise_or(h_lines, v_lines)

    # Ta bort linjer från threshold bild
    th_no_lines = cv2.bitwise_and(th, cv2.bitwise_not(lines))

    #
    cleaned = cv2.bitwise_not(th_no_lines)
    return cleaned

# Förbehandling av bilder
def preprocess_to_mnist(pil_img: Image.Image, mode="Bas (foton)", debug=False):
    pil_grey = ImageOps.grayscale(pil_img)
    img = np.array(pil_grey)

    # blur hjälper mot pappersstruktur
    img_blur = cv2.GaussianBlur(img, (7, 7), 0)

    # Bas: adaptive threshold (robust för mobilfoto)
    th = cv2.adaptiveThreshold(
        img_blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        15
    )

    # Linjerat papper: aggressivare borttagning av horisontella linjer
    if mode == "Linjerat papper":
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1))
        h_lines = cv2.morphologyEx(th, cv2.MORPH_OPEN, h_kernel, iterations=2)
        th = cv2.bitwise_and(th, cv2.bitwise_not(h_lines))

    # Rensa prickar + gör streck lite tydligare
    kernel = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    th = cv2.dilate(th, kernel, iterations=2)

    if debug:
        st.image(th, caption="Debug: threshold", clamp=True)

    # hitta konturer
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        resized = cv2.resize(th, (28, 28), interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32).reshape(1, -1)

    # välj största (baseline igen – stabilt när th är bra)
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    digit = th[y:y+h, x:x+w]

    # kvadrat + resize + padding
    size = max(w, h)
    square = np.zeros((size, size), dtype=np.uint8)
    x_off = (size - w)//2
    y_off = (size - h)//2
    square[y_off:y_off+h, x_off:x_off+w] = digit

    digit_20 = cv2.resize(square, (20, 20), interpolation=cv2.INTER_AREA)
    padded = np.zeros((28, 28), dtype=np.uint8)
    padded[4:24, 4:24] = digit_20

    #Centre of mass centrering
    coords = np.column_stack(np.where(padded > 0))

    if len(coords) > 0:
        cy, cx = coords.mean(axis=0)

        shift_x = int(np.round(14 - cx))
        shift_y = int(np.round(14 - cy))

        M = np.float32([[1, 0, shift_x],
                        [0, 1, shift_y]])
        
        padded = cv2.warpAffine(padded, M, (28, 28))


    return padded.astype(np.float32).reshape(1, -1)



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

        st.caption("För SVC utan probability=True visas en normaliserad 'säkerhet' baserad på decision_function (inte en kalibrerad sannolikhet).")


    else:
        st.info("Modellen stödjer inte sannolikhetsvisning.")
