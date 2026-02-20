""" 
MNIST sifferigenkänning app

Input: rita själv / ladda upp bild / ta bild
Förbehandling: mobilbilder och linjerat papper
Modeller: Extra Trees och SVC för jämförelse direkt i appen
"""

#==========================================
# --------------- Import---------------
#==========================================
from __future__ import annotations
from typing import Literal, Optional
import streamlit as st
import joblib
import numpy as np
from PIL import Image, ImageOps
import cv2
from pathlib import Path
import pandas as pd
from streamlit_drawable_canvas import st_canvas
#==========================================
# --------------- Konstanter---------------
#==========================================
""" här samlar jag alla tal som används
i appen för att senare använda som variabler, det underlättar konfigueringen
"""
mode = Literal["Bas (vanliga foton)", "Linjerat papper"]

max_side_px = 900

clahe_clip_limit = 2.0
clahe_tile_grid = (8, 8)

dark_image_mean_threshold = 80 # om bilden är väldigt mörk inverteras den.

gauss_kernel = (5, 5)
morph_kernel = np.ones((3, 3), np.uint8)

min_area = 500
max_area_ratio = 0.30
max_aspect = 5.0
top_margin_ratio = 0.10

mnist_size = 28
mnist_inner_size = 20
mnist_padding = 4

canvas_size = 280
canvas_stroke = 18

#==============================================
#------------- Ladda modeller------------------
#==============================================
@st.cache_resource
def load_model(model_path: Path)
    """Laddar en sparad modell från hårddisk (cache för att slippa ladda modellerna varej gång sidan laddas)"""
    return joblib.load(str(model_path))


def get_probs(model, X: np.ndarray) -> Optional[np.ndarray]:
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

def resize(grey: np.ndarray, max_side: int = max_side_px) -> np.ndarray:
    """
    Skala stora bilder för stabilare och snabbare app
    """
    h, w = grey.shape[:2]
    if max(h, w) <= max_side:
        return grey

    scale = max_side / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(grey, (new_w, new_h), interpolation=cv2.INTER_AREA)

def apply_clahe(grey: np.ndarray) -> np.ndarray:
    """
    Kontrastutjämning för ojämnt ljus och skuggor
    """
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid)
    return clahe.apply(grey)

def adaptive_threshold(grey_blur: np.ndarray) -> np.ndarray:
    """
    Adaptiv tröskling och dynamisk blockstorlek för att fungera på olika upplösningar.
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

def process_base(th: np.ndarray) -> np.ndarray:
    """
    Bearbetning för vanliga foton(olinjerat papper) en mild open/close med dialation
    """
    kernel = np.ones(morph_kernel, np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    th = cv2.dilate(th, kernel, iterations=1)
    return th

def remove_lines(th: np.ndarray) -> np.ndarray:
    """
    Hitta horisontella linjer med hough och maska dem
    """
    w = th.shape[1]
    lines = cv2.HoughLinesP(
        th, 1, np.pi / 180, threshold=150, minLineLength=w // 3, maxLineGap=30
    )
    line_mask = np.zeros_like(th)

    if lines is not None:
        for (x1, y1, x2, y2) in lines[:, 0]:
            if abs(y2 - y1) < 15:  # ungefär horisontell
                cv2.line(line_mask, (x1, y1), (x2, y2), 255, 8)

    return cv2.bitwise_and(th, cv2.bitwise_not(line_mask))

def process_ruled(th: np.ndarray) -> np.ndarray:
    """
    Försöka fyll aigen där linjer skär siffran
    """
    kernel = np.ones(morph_kernel, np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    th = cv2.dilate(th, kernel, iterations=2)
    return th

def find_bounding_box(th: np.ndarray, mode: Mode) -> tuple[int, int, int, int]:
    """
    Hitta siffrans bounding box och filtrera för att undvika brus samt skuggkanter
    """
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, th.shape[1], th.shape[0]

    img_h, img_w = th.shape
    total_area = img_h * img_w

    valid = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        aspect = w / max(h, 1)

        if area < min_area:
            continue
        if area > total_area * max_area_ratio:
            continue
        if aspect > max_aspect:
            continue
        if y < img_h * top_margin_ratio:
            continue

        valid.append(c)

    # Om inget blev "valid" tar vi största konturen att falla tillbaka på
    candidates = valid if valid else contours

    if mode == "Linjerat papper" and valid:
        # Huvudkontur + närliggande konturer för streck som brutits av linjerna
        main = max(valid, key=cv2.contourArea)
        mx, my, mw, mh = cv2.boundingRect(main)

        nearby = []
        for c in valid:
            x, y, w, h = cv2.boundingRect(c)
            if y < my + mh * 1.5 and (y + h) > my - mh * 0.5:
                nearby.append(c)

        all_points = np.vstack(nearby)
        x, y, w, h = cv2.boundingRect(all_points)
        return x, y, w, h

    c = max(candidates, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return x, y, w, h

def mnist_28x28(th: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """
    Beskär siffran formatera till 20x20 med padding 28x28 och centrering som centre of mass.
    """
    x, y, w, h = bbox
    digit = th[y : y + h, x : x + w]

    size = max(w, h)
    square = np.zeros((size, size), dtype=np.uint8)
    x_off = (size - w) // 2
    y_off = (size - h) // 2
    square[y_off : y_off + h, x_off : x_off + w] = digit

    digit_20 = cv2.resize(square, (mnist_inner_size, mnist_inner_size), interpolation=cv2.INTER_AREA)
    padded = np.zeros((mnist_size, mnist_size), dtype=np.uint8)
    padded[mnist_padding : mnist_padding + mnist_inner_size, mnist_padding : mnist_padding + mnist_inner_size] = digit_20

    # Center of mass centrering för stabilare beabrbetning är bara bounding box
    coords = np.column_stack(np.where(padded > 0))
    if len(coords) > 0:
        cy, cx = coords.mean(axis=0)
        shift_x = int(np.round((mnist_size // 2) - cx))
        shift_y = int(np.round((mnist_size // 2) - cy))
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        padded = cv2.warpAffine(padded, M, (mnist_size, mnist_size))

    return padded

def preprocess_to_mnist(pil_img: Image.Image, mode: Mode) -> np.ndarray:
    """
    konvertera PIL-bild till 28x28 med MNIST-likande array 1x784
    """
    grey = np.array(ImageOps.grayscale(pil_img))

    grey = resize(grey)
    grey = apply_clahe(grey)

    # Om väldigt mörk bild: invertera så att siffran blir ljus mot mörk bakgrund,
    # men eftersom vi använder THRESH_BINARY_INV kan detta hjälpa i vissa fall.
    if np.mean(grey) < dark_image_mean_threshold:
        grey = 255 - grey

    grey_blur = cv2.GaussianBlur(grey, gauss_kernel, 0)
    th = adaptive_threshold(grey_blur)

    if mode == "Bas (vanliga foton)":
        th = process_base(th)
    else:
        th = remove_lines(th)
        th = process_ruled(th)

    bbox = find_bounding_box(th, mode)
    mnist_28 = mnist_28x28(th, bbox)

    return mnist_28.astype(np.float32).reshape(1, -1)

#==============================================
#------------- Bildbehandlig mnist format------
#==============================================