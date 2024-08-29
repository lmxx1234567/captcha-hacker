import random
from captcha.image import ImageCaptcha,random_color,SMOOTH,ColorTuple
from PIL import Image
from PIL.ImageDraw import Draw
from models import vocab
from .fonts import fonts

def create_noise_line(image: Image, color: ColorTuple) -> Image:
    w, h = image.size
    x1 = random.randint(0, w)
    x2 = random.randint(0, w)
    y1 = random.randint(0, h)
    y2 = random.randint(0, h)
    points = [x1, y1, x2, y2]
    Draw(image).line(points, fill=color, width=1)
    return image

def generate_image(chars: str) -> Image:
    """Generate the image of the given characters.

    :param chars: text to be generated.
    """
    image = ImageCaptcha(width=112,height=35,fonts=fonts)
    # background = random_color(238, 255)
    color = random_color(1, 200, 255)
    im = image.create_captcha_image(chars, color, (255, 255, 255) )
    image.create_noise_dots(im, color,1,30)
    create_noise_line(im, color)
    create_noise_line(im, color)
    # im = im.filter(SMOOTH)
    return im

def generate_captcha(captcha = None):
    if captcha is None:
        captcha = ''.join(random.choices(vocab, k=4))
    image = generate_image(captcha)
    return image, captcha

def generate_simple_captcha(captcha = None):
    if captcha is None:
        captcha = ''.join(random.choices(vocab, k=4))
    image = ImageCaptcha(width=112,height=35,font_sizes=[36]).generate_image(captcha)
    return image, captcha
