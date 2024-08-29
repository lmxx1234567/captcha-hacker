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
import h5py
import os

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
    def __init__(self,name,n=1000,repeat=1, simple=False):
        self.n = n
        self.name = name
        if not os.path.exists('runs/cache/'):
            os.makedirs('runs/cache/')
        with h5py.File(f'runs/cache/{self.name}_captcha_dataset.h5', 'w') as f:
            data = f.create_dataset('data', (n, 1, 35, 112), dtype='float32')
            labels = f.create_dataset('labels', (n, 4, 62), dtype='float32')
            self.data = data
            self.labels = labels

            with concurrent.futures.ProcessPoolExecutor() as executor:
                real_n = math.ceil(n / repeat)
                results = executor.map(generate_and_preprocess, [repeat]*real_n, [simple]*real_n)
                with tqdm(total=n) as pbar:
                    idx = 0
                    for image, label in results:
                        for img in image:
                            data[idx] = img.numpy()
                            labels[idx] = label.numpy()
                            idx += 1
                            pbar.update(1)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        with h5py.File(f'runs/cache/{self.name}_captcha_dataset.h5', 'r') as f:
            data = f['data'][idx]
            label = f['labels'][idx]
        return torch.tensor(data), torch.tensor(label)