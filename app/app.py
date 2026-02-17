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
    MODEL_PATH = BASE_DIR / "models" / "EXT_produktion.pkl"
else:
    MODEL_PATH = BASE_DIR / "models" / "SVC_produktion.pkl"

st.write("sökväg", MODEL_PATH)
st.write("Finns titeln?", MODEL_PATH.exists())

# skapa en funktion för laddning av modell
@st.cache_resource # den här ser till att jag itne behöver läsa in modellen varje gång. 
def load_model(path):
    return joblib.load(str(path))

# felmeddelande om modellen itne kan laddas.
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

mode = st.radio("Bildtyp", ["Bas (vanliga foton)", "Linjerat papper"])

# Förbehandling av bilder
def preprocess_to_mnist(pil_img: Image.Image, mode: str):
    pil_grey = ImageOps.grayscale(pil_img)
    img = np.array(pil_grey)

    H, W = img.shape[:2]
    max_side = 900
    if max(H, W) > max_side:
        scale = max_side / max(H, W)
        img = cv2.resize(img, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)

    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    kernel = np.ones((3, 3), np.uint8)

    if mode == "Bas (vanliga foton)":
        th = cv2.adaptiveThreshold(
        img_blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35, 7
    )
        # mild efterbehandling
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
        th = cv2.dilate(th, kernel, iterations=1)

    else:  # Linjerat papper
        th = cv2.adaptiveThreshold(
            img_blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            35, 7
        )
        """
        Behåll för att testa senare. 
        W = th.shape[1]
        k = max(25, min(80, W // 25))   # typ W/25 men clampad mellan 25 och 80

        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
        h_lines = cv2.morphologyEx(th, cv2.MORPH_OPEN, h_kernel, iterations=1)

        th = cv2.bitwise_and(th, cv2.bitwise_not(h_lines))
        """

        # dynamisk kernel så den faktiskt hittar linjer i stora bilder
        W = th.shape[1]
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(50, W // 2), 1))
        h_lines = cv2.morphologyEx(th, cv2.MORPH_OPEN, h_kernel, iterations=1)
        th = cv2.bitwise_and(th, cv2.bitwise_not(h_lines))

        # lite starkare efterbehandling
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
        th = cv2.dilate(th, kernel, iterations=2)

    th_celan = th.copy()
    h, w = th_celan.shape
    mask = np.zeros((h + 2, w +2), np.uint8)
    cv2.floodFill(th, mask, (0, 0), 0)
    cv2.floodFill(th, mask, (w-1, 0), 0)
    cv2.floodFill(th, mask, (0, h-1), 0)
    cv2.floodFill(th, mask, (w-1, h-1), 0)
    th = th_celan

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

    digit_20 = cv2.resize(square, (20, 20), interpolation=cv2.INTER_NEAREST)
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

    X = preprocess_to_mnist(image, mode)

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
