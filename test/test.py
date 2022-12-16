import pytesseract as pt
from PIL import Image, ImageDraw, ImageFont
from pytesseract import Output

global OCR_config
OCR_config = "--oem 3 --dpi 600"
IMG = Image.open("images2/Doc3_move.png")
d = pt.image_to_data(IMG, lang="eng", output_type=Output.DICT, config=OCR_config)
items = list()
tokens = list()
for token, left, top, width, height in zip(
    d["text"], d["left"], d["top"], d["width"], d["height"]
):
    if token.strip() != "":
        item = {"token": token, "bbox": [left, top, left + width, top + height]}
        items.append(item)

for i, item in enumerate(items):
    print(i, "\t", item["token"], "\t", item["bbox"])
print(len(items))
draw = ImageDraw.Draw(IMG)
font = ImageFont.load_default()
for item in items:
    draw.rectangle(item["bbox"], outline="red")
IMG.save("test/inference.jpg")
