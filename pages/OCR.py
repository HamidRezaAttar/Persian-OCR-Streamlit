from cProfile import label
from distutils import extension
import streamlit as st
import pytesseract
from pytesseract import Output
from ocr_utils import *
from parsivar import Normalizer

my_normalizer = Normalizer(statistical_space_correction=True)

import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Persian OCR",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📖 Persian OCR")


def ocr(file_name):
    image = Image.open(file_name)
    image = np.array(image)
    d = pytesseract.image_to_data(image, output_type=Output.DICT)
    n_boxes = len(d["level"])
    boxes = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    for i in range(n_boxes):
        (x, y, w, h) = (d["left"][i], d["top"][i], d["width"][i], d["height"][i])
        boxes = cv2.rectangle(boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
    extracted_text = pytesseract.image_to_string(image, lang="fas")
    # pdf = pytesseract.image_to_pdf_or_hocr(image, lang="fas", extension="pdf")
    # hocr = pytesseract.image_to_pdf_or_hocr(image, lang="fas", extension="hocr")
    return boxes, extracted_text


input_col, output_col = st.columns(2)
with input_col:
    with st.form("form2", clear_on_submit=True):
        content_file = st.file_uploader("Upload your image here")
        submit = st.form_submit_button("Submit")
        if submit:
            if content_file is not None:
                result_image, extracted_text = ocr(content_file)
                with output_col:
                    st.image(
                        result_image,
                        caption="OCR",
                        use_column_width="always",
                        output_format="PNG",
                    )
                    with input_col:
                        output_text = f"""<p dir="rtl" align="justify">{my_normalizer.normalize(extracted_text)}</p>"""
                        st.markdown(output_text, unsafe_allow_html=True)
                        download_image = st.button("Download Image")
                        download_pdf = st.button("Download Report")
                        download_hocr = st.button("Download HOCR")
