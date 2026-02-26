""" 
MNIST sifferigenkänning app

Input: rita själv / ladda upp bild / ta bild
Förbehandling: mobilbilder och linjerat papper
Modeller: Extra Trees och SVC för jämförelse direkt i appen
"""

#==========================================
# --------------- Import-------------------
#==========================================
from typing import Literal, Optional
import streamlit as st
import joblib
import numpy as np
from PIL import Image, ImageOps
import cv2
from pathlib import Path
import pandas as pd
from streamlit_drawable_canvas import st_canvas
import gdown

#==========================================
# -------Konstanter/konfigurering----------
#==========================================
# här samlar jag alla tal som används
# i appen för att senare använda som variabler, det underlättar konfigueringen

# låser valet mellan två alternativ
Mode = Literal["Bas (vanliga foton)", "Linjerat papper"] 

# bilder större än 900 opixlar skalas ner, för snabbhet och stabilitet
max_side_px = 900 

# styr kontrastutjämning
clahe_clip_limit = 2.0 
clahe_tile_grid = (8, 8) 

# om bilden är väldigt mörk inverteras den.
dark_image_mean_threshold = 80 

gauss_kernel = (5, 5) # Styr blur
morph_kernel = (3, 3) # styr morfologi

# Reglerar filtrerringen för att filtrerera bvrus och kontrast
min_area = 500 
max_area_ratio = 0.30
max_aspect = 5.0
top_margin_ratio = 0.10

# MNIST format
mnist_size = 28
mnist_inner_size = 20
mnist_padding = 4

# Canvs storlek och pennan tjocklek
canvas_size = 280
canvas_stroke = 18

#==============================================
#------------- Ladda modeller------------------
#==============================================
@st.cache_resource # Gör att streamlit inte laddar modellerna varje gång appen behöver köras om. 
def load_model(model_path: Path):
    """
    Laddar en sparad modell från lokalt minne 
    """
    return joblib.load(str(model_path))

def ensure_models_downloaded(ext_id: str, svc_id: str, models_dir: Path) -> tuple[Path, Path]:
    models_dir.mkdir(parents=True, exist_ok=True)

    ext_path = models_dir / "EXT_produktion.pkl"
    svc_path = models_dir / "SVC_produktion.pkl"

    # Om filen finns men är misstänkt liten (ofta en HTML-sida), ta bort den
    if ext_path.exists() and ext_path.stat().st_size < 200_000:
        ext_path.unlink()
    if svc_path.exists() and svc_path.stat().st_size < 200_000:
        svc_path.unlink()

    if not ext_path.exists():
        gdown.download(id=ext_id, output=str(ext_path), quiet=False)

    if not svc_path.exists():
        gdown.download(id=svc_id, output=str(svc_path), quiet=False)

    return ext_path, svc_path

def get_probs(model, X: np.ndarray):
    """
    Returnerar en endimensionel array med sannolikheter om modellen stödjer det.
    """
    if not hasattr(model, "predict_proba"):
        return None
    try:
        proba = model.predict_proba(X)
        proba= np.asarray(proba)
        if proba.ndim == 2:
            return proba[0]
    except Exception:
        return None
    return None

#==============================================
#------------- Bildbehandlig mnist format------
#==============================================

def resize(grey: np.ndarray, max_side: int = max_side_px):
    """
    Skala stora bilder för stabilare och snabbare app
    """
    # Är bilden mindre eller lika med max_side återger den samam bild
    h, w = grey.shape[:2]
    if max(h, w) <= max_side:
        return grey

    # Om bilden är större än max_side återger den en nerskalad bild proportioneligt
    scale = max_side / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(grey, (new_w, new_h), interpolation=cv2.INTER_AREA)

def apply_clahe(grey: np.ndarray) -> np.ndarray:
    """
    Kontrastutjämning för ojämnt ljus och skuggor, delar upp bilden i rutmänster
    och returenrar en förbättrad bild
    """
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid)
    return clahe.apply(grey)

def adaptive_threshold(grey_blur: np.ndarray):
    """
    Adaptiv tröskling och dynamisk blockstorlek för att fungera på olika upplösningar.
    Gör bilden binär med vitt= 255 och svart = 0
    """
    block = max(35, (min(grey_blur.shape) // 20) | 1) # Alltid udda tal
    c = 7
    return cv2.adaptiveThreshold(
        grey_blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block,
        c,
    )

def process_base(th: np.ndarray):
    """
    Bearbetning för vanliga foton(olinjerat papper) en mild open/close med dialation
    """
    kernel = np.ones(morph_kernel, np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1) # Tar bort små vita prickar för att minska brus
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1) # Fyller hål i siffran.
    th = cv2.dilate(th, kernel, iterations=1) # Gör siffran tjockare för att hjälpa vid tunna streck
    return th

def remove_lines(th: np.ndarray):
    """
    Hitta horisontella linjer med hough och maska dem
    """

    # Identifiera hosrisontella linjer
    w = th.shape[1]
    lines = cv2.HoughLinesP(
        th, 
        1, 
        np.pi / 180, 
        threshold=150, 
        minLineLength=w // 3, # bara linjer som är minst en tredjedel av bilden accepteras
        maxLineGap=30 # Linjer som är lite brutna accepteras ändå pga skuggor och kontraster.
    )

    # Skapr en tom maskering
    line_mask = np.zeros_like(th)

    # Ritar linjerna i den tomma masken
    if lines is not None:
        for (x1, y1, x2, y2) in lines[:, 0]:
            if abs(y2 - y1) < 15:  # ungefär horisontell
                cv2.line(line_mask, (x1, y1), (x2, y2), 255, 8) # maskera linjerna genom att rita vita streck över

    # kombinerar den maskerade bilden med originalbilden för att maskera linjerna. 
    return cv2.bitwise_and(th, cv2.bitwise_not(line_mask))

def process_ruled(th: np.ndarray):
    """
    Försöka fyll aigen där linjer skär siffran
    """
    kernel = np.ones(morph_kernel, np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2) # Stänger gapet i siffran där linjer tagits bort
    th = cv2.dilate(th, kernel, iterations=2) # gör siffran tjockare för lättare identifirering
    return th

def find_bounding_box(th: np.ndarray, mode: Mode):
    """
    Hittar den mest sannolika siffran genom att välja största rimliga konturen,
    i linjerat läge slår den ihop närliggande konturer eftersom siffrran kan bli splittrad av borttagna linjer
    """
    # Hitta sammanhängande vita områden
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Tar bara ytterkonturer och nte konturer i hål för tex 8 och 0
    if not contours:
        return 0, 0, th.shape[1], th.shape[0] # Hittas inga konturer återfaller den till att ha hela bilden som boudning box

    # Filtreerar bort konturer som är för stora relativt till bilden.
    img_h, img_w = th.shape
    total_area = img_h * img_w

    # Filtrera bort brus
    valid = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        aspect = w / max(h, 1)

        if area < min_area: # ta bort småpartiklar och brus
            continue
        if area > total_area * max_area_ratio: # Tar bort stora områden, tex skugga, papperskanter
            continue
        if aspect > max_aspect: # Tar bort långsmala konturer som är för breda i förhållande till höjden
            continue
        if y < img_h * top_margin_ratio: # tar bort saker nära toppen tex skuggkant, papperskant 
            continue

        valid.append(c)
    
    #skydda från att krasha om nearby blir tom


    # Om inget blev "valid" tar vi största konturen att falla tillbaka på
    candidates = valid if valid else contours

    if mode == "Linjerat papper" and valid:
        # Huvudkontur och närliggande konturer för streck som brutits av linjerna
        main = max(valid, key=cv2.contourArea)
        mx, my, mw, mh = cv2.boundingRect(main)

        # hitta och samla konturer som ligger i ugnefär samma vertikala spannn som huvudkonturen ifall siffran blivit splittrad.
        nearby = []
        for c in valid:
            x, y, w, h = cv2.boundingRect(c)
            if y < my + mh * 1.5 and (y + h) > my - mh * 0.5: # hittar konturer som ligger lagom långt från huvudkonturen
                nearby.append(c)
        
        # Säkerhetsfallback om inga nearby hittas, använd huvudkonturen
        nearby = nearby if nearby else [main]

        all_points = np.vstack(nearby)
        x, y, w, h = cv2.boundingRect(all_points)
        return x, y, w, h

    # om inte linjerat papper hitta största konturen och anväönd som bounding box.
    c = max(candidates, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return x, y, w, h

def mnist_28x28(th: np.ndarray, bbox: tuple[int, int, int, int]):
    """
    Beskär siffran formatera till 20x20 med padding 28x28 och centrering som centre of mass.
    """
    
    x, y, w, h = bbox # beskär ut boundingbox
    digit = th[y : y + h, x : x + w] # beskär regionen där siffran är

    # skalar ner bilden och behåller proportionerna
    size = max(w, h) # största sidan av boxen
    square = np.zeros((size, size), dtype=np.uint8) 
    # beräknar offset så siffran blir centrerad i kvadraten
    x_off = (size - w) // 2 
    y_off = (size - h) // 2
    square[y_off : y_off + h, x_off : x_off + w] = digit # 

    # omstrukturterar till 20x20 och lägger till padding till 28x28, lägger siffran i mitten med marginal runt som MNISt
    digit_20 = cv2.resize(square, (mnist_inner_size, mnist_inner_size), interpolation=cv2.INTER_AREA)
    padded = np.zeros((mnist_size, mnist_size), dtype=np.uint8)
    padded[mnist_padding : mnist_padding + mnist_inner_size, mnist_padding : mnist_padding + mnist_inner_size] = digit_20

    # centrerar siffran baserat på pixlar med bläcks tyngdpunkt
    coords = np.column_stack(np.where(padded > 0)) # Alla kordinater där det finns bläck(vita pixlar)
    if len(coords) > 0: # återger medelpostionen för alla pixlar med bläck
        cy, cx = coords.mean(axis=0)
        # räkna hur mycket bilden måste förskjutas för att hamna i mitten
        shift_x = int(np.round((mnist_size // 2) - cx)) # rättar upp bilden i x axeln
        shift_y = int(np.round((mnist_size // 2) - cy)) # rättar upp bilden i y axeln
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]]) 
        padded = cv2.warpAffine(padded, M, (mnist_size, mnist_size))

    return padded

def preprocess_to_mnist(pil_img: Image.Image, mode: Mode):
    """
    konvertera PIL-bild till 28x28 med MNIST-likande array 1x784 och gråskala
    """
    grey = np.array(ImageOps.grayscale(pil_img))

    grey = resize(grey)
    grey = apply_clahe(grey)

    # Om väldigt mörk bild: invertera så att siffran blir ljus mot mörk bakgrund,
    if np.mean(grey) < dark_image_mean_threshold:
        grey = 255 - grey

    grey_blur = cv2.GaussianBlur(grey, gauss_kernel, 0)
    th = adaptive_threshold(grey_blur)

    # beroende på mode valt i appen hanteras bilden olika
    if mode == "Bas (vanliga foton)":
        th = process_base(th)
    else:
        th = remove_lines(th)
        th = process_ruled(th)

    bbox = find_bounding_box(th, mode)
    mnist_28 = mnist_28x28(th, bbox)

    return mnist_28.astype(np.float32).reshape(1, -1)

#==============================================
#------------- App UI -------------------------
#==============================================

def main() -> None:
    """
    Startpunkt för appen
    """
    st.set_page_config(page_title="MNIST känn igen siffror", layout="centered")
    st.title("MNIST känn igen siffror")
    st.caption("Jämförelse mellan Extra Trees och SVC på MNIST")

    # sökvägar till modellerna
    base_dir = Path(__file__).resolve().parents[1]
    models_dir = base_dir / "models"

    EXT_ID = "1x7D76TIunyXw0p6SSmEC_DQ9oa67KfN-"
    SVC_ID = "1e77q5w9IHJTR6jZpwmBQQZkM6KWFTKoZ"

    ext_path, svc_path = ensure_models_downloaded(EXT_ID, SVC_ID, models_dir)

    try:
        ext_model = load_model(ext_path)
        svc_model = load_model(svc_path)
    except Exception as e:
        st.error(f"Kunde inte ladda modell: {e}") # om modellen inte laddas korrekt får vi felmeddelande. 
        st.stop()

    #sidobar för inställningar.
    with st.sidebar:
        st.header("⚙️ Inställningar")
        show_debug = st.checkbox("Visa tekniska detaljer (debug)", value=False)
        conf_threshold = st.slider("Osäkerhetsgräns (%)", 30, 90, 60, 5)
        mode_choice: Mode = st.radio("Bildtyp", ["Bas (vanliga foton)", "Linjerat papper"])

    st.markdown("---")
    st.header("Input")

    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = "canvas_0"

    # bälj metod för input, rita, ladda upp eller ta bild
    input_choice = st.radio(
        "Välj metod",
        ["🎨 Rita själv", "📁 Ladda upp en bild", "📷 Ta en bild"],
        horizontal=True,
    )

    predict_clicked = st.button("🔍 Prediktera")
    left, right = st.columns([1, 1])

    image: Optional[Image.Image] = None

    with left:
        if input_choice == "📁 Ladda upp en bild":
            img_file = st.file_uploader("Ladda upp en bild (png/jpg)", type=["png", "jpg", "jpeg"])
            if img_file is not None:
                image = Image.open(img_file)

        elif input_choice == "📷 Ta en bild":
            cam = st.camera_input("Ta en bild med kameran")
            if cam is not None:
                image = Image.open(cam)

        else:
            st.subheader("🎨 Rita din siffra")

            cbtn1, _ = st.columns([1, 2])
            with cbtn1:
                if st.button("🧽 Rensa"):
                    st.session_state.canvas_key = f"canvas_{np.random.randint(1_000_000)}"

            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 0, 0)",
                stroke_width=canvas_stroke,
                stroke_color="#000000",
                background_color="#FFFFFF",
                height=canvas_size,
                width=canvas_size,
                drawing_mode="freedraw",
                key=st.session_state.canvas_key,
            )

            if canvas_result.image_data is not None:
                rgba = canvas_result.image_data.astype("uint8")
                rgb = rgba[:, :, :3]
                ink = np.any(np.min(rgb, axis=2) < 200)

                if ink:
                    image = Image.fromarray(rgba, mode="RGBA").convert("RGB")
                else:
                    st.info("🎨 Rita en siffra")

    # ser till att modellerna inte körs om det inte finns en bild och användaren tryckt "prediktera"
    with right:
        st.subheader("Resultat")
        if image is None:
            st.info("Välj en metod och ge en bild/siffra")
            st.stop()
        if not predict_clicked:
            st.stop()

    with left:
        st.subheader("Förhandsvisning")
        st.image(image, caption="Originalbild", use_container_width=True)

        effective_mode: Mode = "Bas (vanliga foton)" if input_choice == "🎨 Rita själv" else mode_choice
        X = preprocess_to_mnist(image, effective_mode)

        pred_ext = int(ext_model.predict(X)[0])
        pred_svc = int(svc_model.predict(X)[0])

        p_ext = get_probs(ext_model, X)
        p_svc = get_probs(svc_model, X)

    with right:
        if pred_ext != pred_svc:
            st.error("⚠️ Modellerna är inte överens")
        else:
            st.success("✅ Modellerna är överens")

        st.markdown("### Modelljämförelse")
        m1, m2 = st.columns(2)
        m1.metric("Extra Trees", pred_ext)
        m2.metric("SVC", pred_svc)

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

        if show_debug:
            with st.expander("Visa tekniska detaljer (debug)"):
                preview = X.reshape(28, 28).astype(np.uint8)
                st.image(preview, caption="Förbehandlad 28x28", width=200)
                st.write("Effective mode:", effective_mode)
                st.write("Input method:", input_choice)


if __name__ == "__main__":
    main()