# OCR model assuming the image is 112x35 and the captcha is 4 characters long
import torch
import torch.nn.init as init
from . import vocab


class CaptchaOCR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(7168, 256),
            torch.nn.ReLU(),
            # 4 characters in one-hot encoding
            torch.nn.Linear(256, 4 * 62),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                init.xavier_uniform_(m.weight)  # Xavier initialization for Conv layers
                if m.bias is not None:
                    init.zeros_(m.bias)  # Set bias to zero
            elif isinstance(m, torch.nn.Linear):
                init.xavier_uniform_(
                    m.weight
                )  # Xavier initialization for Linear layers
                init.zeros_(m.bias)  # Set bias to zero

    def forward(self, x):
        return self.model(x).view(-1, 4, 62)

    def predict(self, x):
        with torch.no_grad():
            return self.model(x).view(-1, 4, 62).argmax(dim=-1)

    def predict_all_possible(self, x):
        with torch.no_grad():
            # Get each character's probabilities
            probs = self.model(x).view(-1, 4, 62).softmax(dim=-1)

            # Generate all combinations of indices for each character position
            all_indices = torch.cartesian_prod(
                torch.arange(62),  # For first character
                torch.arange(62),  # For second character
                torch.arange(62),  # For third character
                torch.arange(62),  # For fourth character
            ).to(
                probs.device
            )  # Ensure this tensor is on the same device as probs

            # Gather the probabilities for each combination
            word_probs = probs[0, torch.arange(4).unsqueeze(1), all_indices.T].prod(
                dim=0
            )

            # Sort the word combinations by probability
            sorted_probs, sorted_idx = word_probs.sort(descending=True)

            words = []
            # Return the top 10 word list with probabilities
            for i in range(10):
                word = "".join(
                    [vocab[idx.item()] for idx in all_indices[sorted_idx[i]]]
                )
                words.append((word, sorted_probs[i].item()))

            return words