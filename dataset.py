# generate captcha dataset
from torch.utils.data import Dataset
import concurrent.futures
from torchvision import transforms
import torch
from captchagen import (
    generate_captcha,
    generate_simple_captcha,
    preprocess_image,
    vocab,
)
from tqdm import tqdm
import math
import random
from typing import List, Tuple
import h5py
import os
import string


def generate_and_preprocess(
    repeat=1, simple=False, case_sensitive=False
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    global vocab
    captcha = "".join(random.choices(vocab, k=4))
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
    if case_sensitive:
        label = torch.zeros(4, 62)
    else:
        label = torch.zeros(4, 36)
        vocab = string.ascii_lowercase + string.digits
    for i, char in enumerate(captcha):
        if case_sensitive:
            label[i][vocab.index(char)] = 1
        else:
            label[i][vocab.index(char.lower())] = 1

    return image_tensor_list, label


class CaptchaDataset(Dataset):
    def __init__(
        self, name, n=1000, repeat=1, simple=False, case_sensitive=False, force=False
    ):
        self.n = n
        self.name = name
        self.case_sensitive = case_sensitive
        if not force:
            if os.path.exists(f"runs/cache/{self._get_h5_filename()}"):
                with h5py.File(f"runs/cache/{self._get_h5_filename()}", "r") as f:
                    self.data = f["data"]
                    self.labels = f["labels"]
                    return
        if not os.path.exists("runs/cache/"):
            os.makedirs("runs/cache/")
        with h5py.File(f"runs/cache/{self._get_h5_filename()}", "w") as f:
            data = f.create_dataset("data", (n, 1, 35, 112), dtype="float32")
            if case_sensitive:
                labels = f.create_dataset("labels", (n, 4, 62), dtype="float32")
            else:
                labels = f.create_dataset("labels", (n, 4, 36), dtype="float32")
            self.data = data
            self.labels = labels

            with concurrent.futures.ProcessPoolExecutor() as executor:
                real_n = math.ceil(n / repeat)
                results = executor.map(
                    generate_and_preprocess, [repeat] * real_n, [simple] * real_n
                )
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
        with h5py.File(f"runs/cache/{self._get_h5_filename()}", "r") as f:
            data = f["data"][idx]
            label = f["labels"][idx]
        return torch.tensor(data), torch.tensor(label)
    
    def _get_h5_filename(self):
        return f"{self.name}_{self.n}_{'case_sensitive' if self.case_sensitive else 'case_insensitive'}.h5"
