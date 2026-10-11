import os

from pdf_handler import read_pdf
from vector_store import add_document

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def ingest_file(path):
    """Read a PDF or image file, add it to the vector index, return the number of chunks."""
    name = os.path.basename(path)
    lower = name.lower()

    if lower.endswith(".pdf"):
        text = read_pdf(path)
    elif lower.endswith(IMAGE_EXTENSIONS):
        from image_handler import read_image  # imported here: EasyOCR loads slowly
        text = read_image(path)
    else:
        raise ValueError("Only PDF, PNG and JPG files are supported.")

    return add_document(text, name)
