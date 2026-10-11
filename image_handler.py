import easyocr

reader = easyocr.Reader(['en'])

def read_image(image_path):

    result = reader.readtext(image_path)

    text = ""

    for item in result:
        text += item[1] + " "

    return text