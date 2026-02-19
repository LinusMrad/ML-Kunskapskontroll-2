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

# Funktion för förbehandling av bilder till MNIST-format
def preprocess_to_mnist(pil_img: Image.Image, mode: str):
    pil_grey = ImageOps.grayscale(pil_img)
    img = np.array(pil_grey)

    H, W = img.shape[:2]
    max_side = 900
    if max(H, W) > max_side:
        scale = max_side / max(H, W)
        img = cv2.resize(img, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)

    Clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = Clahe.apply(img)

    if np.mean(img) < 80:
        img = 255 - img

    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    kernel = np.ones((3, 3), np.uint8)

    block = max(35, (min(img.shape) // 20) | 1)
    C = 7

    if mode == "Bas (vanliga foton)":
        th = cv2.adaptiveThreshold(
        img_blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block, C
    )
        # mild efterbehandling
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
        th = cv2.dilate(th, kernel, iterations=1)

    else:  # Linjerat papper
        th = cv2.adaptiveThreshold(
            img_blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block, C
        )

        #Hough för att hitta linjer och maskera
        W = th.shape[1]
        lines = cv2.HoughLinesP(th, 1, np.pi/180, threshold=150, minLineLength=W//3, maxLineGap=30)
        line_mask = np.zeros_like(th)
        if lines is not None:
            for l in lines:
                x1, y1, x2, y2 = l[0]
                if abs(y2 -y1) < 15:
                    cv2.line(line_mask, (x1, y1), (x2, y2), 255, 8)
        th = cv2.bitwise_and(th, cv2.bitwise_not(line_mask))

        # Efterbehandling för att försöka hitta där linjerna skär genom isffran och fylla igen.
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
        th = cv2.dilate(th, kernel, iterations=2)

    th_clean = th.copy()
    h, w = th_clean.shape
    mask = np.zeros((h + 2, w +2), np.uint8)
    cv2.floodFill(th, mask, (0, 0), 0)
    cv2.floodFill(th, mask, (w-1, 0), 0)
    cv2.floodFill(th, mask, (0, h-1), 0)
    cv2.floodFill(th, mask, (w-1, h-1), 0)
    th = th_clean

    # hitta konturer
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        resized = cv2.resize(th, (28, 28), interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32).reshape(1, -1)

    img_h, img_w = th.shape
    total_area = img_h * img_w

    valid = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        aspect = w / max(h, 1)

        if area < 500: continue # hanterar litet brus
        if area > total_area * 0.3: continue # stor bakgrund = bakgrund/skugga
        if aspect > 5: continue # för platt = horisontell skuggkant
        if y < img_h * 0.1: continue # för nära toppen = skuggkant
        valid.append(c)

    if valid:
        if mode == "Linjerat papper":
            # Hitta den största konturen som referens
            main = max(valid, key=cv2.contourArea)
            mx, my, mw, mh = cv2.boundingRect(main)
            
            # Inkludera bara konturer som överlappar eller är nära huvudkonturen
            nearby = []
            for c in valid:
                x, y, w, h = cv2.boundingRect(c)
                # Konturen måste vara inom 1.5x huvudkonturens höjd
                if y < my + mh * 1.5 and y + h > my - mh * 0.5:
                    nearby.append(c)
            
            all_points = np.vstack(nearby)
            x, y, w, h = cv2.boundingRect(all_points)
        else:
            c = max(valid, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
    else:
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

# Funmktion för att visa sannolikheten som modellerna gissar på siffran
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
        
    return None

#==============================================
#------------- Ui: Layout ------------------
#==============================================

# Skapa en första sida
st.set_page_config(page_title="MNIST känn igen siffror", layout="centered")


# sätt titel
st.title("MNIST känn igen siffror")
st.caption("Jämförelse mellan Extra Trees och SVC på MNIST")

#==============================================
#------------- Hämta modeller------------------
#==============================================

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

#==============================================
#------------- Sidebar -----------------------
#==============================================

with st.sidebar:
    st.header("⚙️ Inställningar")
    show_debug = st.checkbox("Visa tekniska detaljer (debug)", value=False)
    conf_threshold = st.slider("Osäkerhetsgräns (%)", 30, 90, 60, 5)

    mode_choice = st.radio("bildtyp", ["Bas (Vanliga foton)", "Linjerat papper"])



#==============================================
#------------- Skapa input --------------------
#==============================================


# Huvudflöde med bildhantering. uppladning/ta en bild/rita själv
st.markdown("---")
st.header("Input")

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = "canvas_0"

input_choice = st.radio(
    "Välj metod",
    ["🎨 Rita själv", "📁 Ladda upp en bild", "📷 Ta en bild"],
    horizontal=True
)

img_file = None
cam = None
image = None

#layout för input/resultat
left, right = st.columns([1, 1])

with left:
    if input_choice == "📁 Ladda upp en bild":
        img_file = st.file_uploader("Ladda upp en bild (png/jpg)", type=["png", "jpg", "jpeg"])
    elif input_choice == "📷 Ta en bild":    
        cam = st.camera_input("Ta en bild med kameran")
    elif input_choice == "🎨 Rita själv":
        st.subheader("🎨 Rita din siffra")

        cbtn1, cbtn2 = st.columns([1, 2])
        with cbtn1:
            if st.button("🧽 Rensa"):
                st.session_state.canvas_key = f"canvas_{np.random.randint(1_000_000)}"
        
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=18,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key=st.session_state.canvas_key
        )

        if canvas_result.image_data is not None:
            rgba = canvas_result.image_data.astype("uint8")
            rgb = rgba[:, :, :3]
            ink = np.any(np.min(rgb, axis=2) <200)

            if ink:
                image = Image.fromarray(rgba, mode="RGBA").convert("RGB")
            else:
                image = None
                st.info("Rita en siffra")

if image is None:
    if img_file is not None:
        image = Image.open(img_file)
    elif cam is not None:
        image = Image.open(cam)

if input_choice == "🎨 Rita själv":
    effective_mode = "Bas (vanliga foton)"
else:
    effective_mode = mode_choice

#==============================================
#------------- prediktion + UI ----------------
#==============================================

with right:
    st.subheader("Resultat")
    if image is None:
        st.info("Välj en metod och ge en bild/siffra")
        st.stop()

# Visa input i vänsterkolumnen
with left:
    st.subheader("Förhandsvisning")
    st.image(image, caption="Originalbild", use_container_width=True)

    # Preprocess och predict
    X = preprocess_to_mnist(image, effective_mode)


    # Prediktion
    pred_ext = ext_model.predict(X)[0]
    pred_svc = svc_model.predict(X)[0]
    #st.success(f"predikterad siffra: **{pred}**")

    p_ext = get_probs(ext_model, X)
    p_svc = get_probs(svc_model, X)

    # Resultat UI
with right:
    # Status: överens/inte
    if pred_ext != pred_svc:
        st.error("⚠️ Modellerna är inte överens")
    else:
        st.success("✅ Modellerna är överens")

    st.markdown("### Modelljämförelse")
    m1, m2 = st.columns(2)
    with m1:
        st.metric("Extra Trees", int(pred_ext))
    with m2:
        st.metric("SVC", int(pred_svc))

    # Osäkerhetsvarning (prioritera SVC om den finns, annars EXT)
    primary_probs = p_svc if p_svc is not None else p_ext
    if primary_probs is not None:
        max_prob = float(np.max(primary_probs))
        if max_prob < (conf_threshold / 100.0):
            st.warning(f"Osäker prediktion ({max_prob*100:.1f}%). Testa bättre ljus / zooma in.")

    st.markdown("### Modellernas säkerhet (Top 3)")
    b1, b2 = st.columns(2)

    with b1:
        st.markdown("**Extra Trees**")
        if p_ext is not None:
            s = pd.Series(p_ext, index=list(range(10))).sort_values(ascending=False).head(3) * 100
            st.bar_chart(s)
            st.caption(f"Mest sannolik: {int(s.index[0])} ({s.iloc[0]:.1f}%)")
        else:
            st.info("Ingen sannolikhetsvisning.")

    with b2:
        st.markdown("**SVC**")
        if p_svc is not None:
            s = pd.Series(p_svc, index=list(range(10))).sort_values(ascending=False).head(3) * 100
            st.bar_chart(s)
            st.caption(f"Mest sannolik: {int(s.index[0])} ({s.iloc[0]:.1f}%)")
        else:
            st.info("Ingen sannolikhetsvisning.")

    # Debug expander
    if show_debug:
        with st.expander("Visa tekniska detaljer (debug)"):
            preview = X.reshape(28, 28).astype(np.uint8)
            st.image(preview, caption="Förbehandlad 28x28", width=200)
            st.write("Effective mode:", effective_mode)
            st.write("Input method:", input_choice)