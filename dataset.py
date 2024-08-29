# generate captcha dataset
from torch.utils.data import Dataset
import concurrent.futures
from torchvision import transforms
import torch
from captchagen import generate_captcha, generate_simple_captcha, preprocess_image, vocab
from tqdm import tqdm
import math
import random
from typing import List, Tuple

def generate_and_preprocess(repeat=1,simple=False)->Tuple[List[torch.Tensor], torch.Tensor]:
    captcha = ''.join(random.choices(vocab, k=4))
    image_tensor_list = []
    for i in range(repeat):
        if simple:
            image, captcha = generate_simple_captcha(captcha)
        else:
            image, captcha = generate_captcha(captcha)
        image = preprocess_image(image)
        image_tensor = transforms.ToTensor()(image)
        image_tensor_list.append(image_tensor)

    # Generate one-hot encoded label
    label = torch.zeros(4, 62)
    for i, char in enumerate(captcha):
        label[i][vocab.index(char)] = 1

    return image_tensor_list, label


class CaptchaDataset(Dataset):
    def __init__(self, n=1000,repeat=1, simple=False):
        self.data = []
        self.labels = []

        with concurrent.futures.ProcessPoolExecutor() as executor:
            real_n = math.ceil(n / repeat)
            results = executor.map(generate_and_preprocess, [repeat]*real_n, [simple]*real_n)
            with tqdm(total=n) as pbar:
                for image, label in results:
                    self.data.extend(image)
                    self.labels.extend([label]*repeat)
                    pbar.update(repeat)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]