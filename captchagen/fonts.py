# list all ttf fonts in the font directory
import os

font_dir = "fonts"
fonts = []
for root, dirs, files in os.walk(font_dir):
    for file in files:
        if file.endswith(".ttf"):
            fonts.append(os.path.join(root, file))