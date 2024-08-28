# generate captcha dataset
from torch.utils.data import Dataset
import concurrent.futures
from torchvision import transforms
import torch
from captchagen import generate_captcha, generate_simple_captcha, preprocess_image, vocab
from tqdm import tqdm

def generate_and_preprocess(simple=False):
    if simple:
        image, captcha = generate_simple_captcha()
    else:
        image, captcha = generate_captcha()
    image = preprocess_image(image)
    image_tensor = transforms.ToTensor()(image)

    # Generate one-hot encoded label
    label = torch.zeros(4, 62)
    for i, char in enumerate(captcha):
        label[i][vocab.index(char)] = 1

    return image_tensor, label


class CaptchaDataset(Dataset):
    def __init__(self, n=1000,simple=False):
        self.data = []
        self.labels = []

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(generate_and_preprocess, [simple] * n)
            for image, label in tqdm(results, total=n):
                self.data.append(image)
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]