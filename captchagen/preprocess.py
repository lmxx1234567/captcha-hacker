# 图像预处理
def preprocess_image(image):
    image = image.convert("L")  # 灰度化
    threshold = 200
    table = []
    for i in range(256):
        if i < threshold:
            table.append(0)
        else:
            table.append(1)
    image = image.point(table, '1')  # 二值化
    return image