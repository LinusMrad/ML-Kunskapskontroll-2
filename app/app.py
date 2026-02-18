# Importera bibliotek
import streamlit as st
import joblib
import numpy as np
from PIL import Image, ImageOps
import cv2
from pathlib import Path
import pandas as pd
from streamlit_drawable_canvas import st_canvas
#==========================================
# --------------- Funktioner---------------
#==========================================
# Funktion för förbehandling av bilder
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

# Funmktion för att visa sannolikheten som modellerna gissar på siffran
# predict_proba för Extra tress och decision_function + softmax för SVC utan probabnility=True
def get_probs(model, X):
    #Riktiga sannolikheter
    if hasattr(model, "predict_proba"):
        try:
            p = model.predict_proba(X)
            p = np.asarray(p)
            if p.ndim == 2:
                p = p[0]
                return p
        except Exception:
            pass
    
    # Softmax
    if hasattr(model, "decision_function"):
        try:
            scores = model.decision_function(X)
            scores = np.asarray(scores)
            scores = scores.reshape(-1)
            if scores.shape[0] != 10:
                return None
            scores = scores - np.max(scores)
            exp_scores = np.max(scores)
            p = exp_scores / np.sum(exp_scores)
            return p
        except Exception:
            pass
    
    return None


# Skapa en första sida
st.set_page_config(page_title="MNIST bildigenkänning", layout="centered")

# sätt titel
st.title("MNIST känn igen siffror")


# Sökvägen för modellerna
BASE_DIR = Path(__file__).resolve().parents[1]

# Modelfilerna
EXT_PATH = BASE_DIR / "models" / "EXT_produktion.pkl"
SVC_PATH = BASE_DIR / "models" / "SVC_produktion.pkl"

# skapa en funktion för laddning av modell
@st.cache_resource # den här ser till att jag itne behöver läsa in modellen varje gång. 
def load_model(path):
    return joblib.load(str(path))

# felmeddelande om modellen itne kan laddas.
try:
    ext_model = load_model(EXT_PATH)
    svc_model = load_model(SVC_PATH)
except Exception as e:
    st.error(f"Kunde inte ladda modell: {e}")
    st.stop()


# Huvudflöde med bildhantering. uppladning/ta en bild/rita själv
st.markdown("---")
st.header("Välj hur du vill mata in en siffra")

input_choice = st.radio(
    "Välj metod",
    ["Rita själv", "📁 Ladda upp en bild", "📷 Ta en bild"]
)

# Skapa input
img_file = None
cam = None

if input_choice == "📁 Ladda upp en bild":
    img_file = st.file_uploader("Ladda upp en bild (png/jpg)", type=["png", "jpg", "jpeg"])
elif input_choice == "📷 Ta en bild":    
    cam = st.camera_input("Ta en bild med kameran")
elif input_choice == "Rita själv":
    st.subheader("Rita din siffra")

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas"
    )

    if canvas_result.image_data is not None and np.any(canvas_result.image_data[:, :, 3] > 0):
        image = Image.fromarray(
            canvas_result.image_data.astype("uint8"),
            mode="RGBA"
        ).convert("RGB")
    else:
        image = None
        st.info("Rita en siffra")

image = None
if img_file is not None:
    image = Image.open(img_file)
elif cam is not None:
    image = Image.open(cam)

mode = st.radio("Bildtyp", ["Bas (vanliga foton)", "Linjerat papper"])


if image is not None:
    st.subheader("Input")
    st.image(image, caption="Originalbild", use_container_width=True)

    X = preprocess_to_mnist(image, mode)

    #Visa hur modellen ser bilden
    st.subheader("Föbehandlad")
    preview = X.reshape(28, 28).astype(np.uint8)
    st.image(preview, caption="MNIST-format (vit siffra på svart bakgrund)", use_column_width=False)

    # Prediktion
    pred_ext = ext_model.predict(X)[0]
    pred_svc = svc_model.predict(X)[0]
    #st.success(f"predikterad siffra: **{pred}**")

    st.subheader("Modelljämförelse")
    c1, c2 = st.columns(2)

    with c1:
        st.metric("Extra Trees", int(pred_ext))
    with c2:
        st.metric("SVC", int(pred_svc))

    if pred_ext != pred_svc:
        st.warning("Modellerna är inte överens")

    st.subheader("Moddelernas säkerhet(top 3)")

    p_ext = get_probs(ext_model, X)
    p_svc = get_probs(svc_model, X)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Extra Trees**")
        if p_ext is not None:
            s = pd.Series(p_ext, index=list(range(10)))
            s = s.sort_values(ascending=False).head(3) * 100
            st.bar_chart(s)
            st.caption(f"Mest sannolik: {int(s.index[0])} ({s.iloc[0]:.1f}%)")
        else:
            st.info("Ingen sannolikhetsvisning.")
    
    with c2:
        st.markdown("**SVC**")
        if p_svc is not None:
            s = pd.Series(p_svc, index=list(range(10)))
            s = s.sort_values(ascending=False).head(3) * 100
            st.bar_chart(s)
            st.caption(f"Mest sannolik: {int(s.index[0])} ({s.iloc[0]:.1f}%)")
        else:
            st.info("Ingen sannolikhetsvisning.")