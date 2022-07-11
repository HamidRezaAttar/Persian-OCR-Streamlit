import streamlit as st
import pandas as pd
import numpy as np
from ocr_utils import *

import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Persian OCR",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📖 Persian OCR")


def scan(file_name):
    img = Image.open(file_name)
    img.thumbnail((800, 800), Image.ANTIALIAS)
    image = np.array(img)
    resize_ratio = 500 / image.shape[0]
    original = image.copy()
    image = opencv_resize(image, resize_ratio)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilated = cv2.dilate(blurred, rectKernel)
    edged = cv2.Canny(dilated, 100, 200, apertureSize=3)
    contours, hierarchy = cv2.findContours(
        edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    image_with_contours = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 3)
    largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    image_with_largest_contours = cv2.drawContours(
        image.copy(), largest_contours, -1, (0, 255, 0), 3
    )
    receipt_contour = get_receipt_contour(largest_contours)
    image_with_receipt_contour = cv2.drawContours(
        image.copy(), [receipt_contour], -1, (0, 255, 0), 2
    )
    scanned = wrap_perspective(
        original.copy(), contour_to_rect(receipt_contour, resize_ratio)
    )
    result = bw_scanner(scanned)
    return result


input_col, output_col = st.columns(2)
with input_col:
    with st.form("form1", clear_on_submit=True):
        content_file = st.file_uploader("Upload your image here")
        submit = st.form_submit_button("Submit")
        if submit:
            if content_file is not None:
                result = scan(content_file)
                with output_col:
                    st.image(
                        result,
                        caption="scanned image",
                        use_column_width="always",
                        output_format="PNG",
                    )
