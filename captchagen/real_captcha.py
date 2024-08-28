import requests
from PIL import Image
import io
import string

vocab = string.ascii_letters + string.digits


def get_captcha_image():
    response = requests.get(
        "https://apply.jtw.beijing.gov.cn/apply/app/common/validCodeImage"
    )
    cookie = response.cookies.get_dict()  # 获取cookie
    image = Image.open(io.BytesIO(response.content))  # 获取图片
    return image, cookie


def check_captcha(captcha_text, cookie):
    headers = {"cookie": f"JSESSIONID={cookie['JSESSIONID']}"}
    url = f"https://apply.jtw.beijing.gov.cn/apply/app/common/checkValidCode?validCode={captcha_text}"
    response = requests.get(url, headers=headers)
    return response.json()


