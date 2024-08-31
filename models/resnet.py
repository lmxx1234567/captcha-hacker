import torch
import torch.nn as nn
import torch.nn.functional as F
from . import vocab
import string


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class ResNetCaptchaOCR(nn.Module):
    def __init__(
        self,
        num_classes=62,
        num_characters=4,
        block=ResidualBlock,
        layers=[2, 2, 2, 2, 2],
        dropout_prob=0.5,
    ):
        super(ResNetCaptchaOCR, self).__init__()
        self.num_classes = num_classes
        if num_classes == 36:
            self.vocab = string.ascii_lowercase + string.digits
        else:
            self.vocab = vocab
        self.in_channels = 32

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(
            1, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)

        # Residual layers
        self.res_layers = []
        for i, num_blocks in enumerate(layers):
            stride = 2 if i > 0 else 1
            self.res_layers.append(
                self._make_layer(block, 32 * 2**i, num_blocks, stride)
            )
        self.res_layers = nn.Sequential(*self.res_layers)

        # Global Average Pooling layer
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(
            32 * 2 ** (len(layers) - 1), 256
        )  # Output of the last residual layer
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(256, num_characters * num_classes)

        self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.res_layers(x)

        x = self.gap(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.view(-1, 4, self.num_classes)
        return x

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x).view(-1, 4, self.num_classes).argmax(dim=-1)

    def predict_all_possible(self, x):
        with torch.no_grad():
            # Get each character's probabilities
            probs = self.forward(x).view(-1, 4, self.num_classes).softmax(dim=-1)

            # Generate all combinations of indices for each character position
            all_indices = torch.cartesian_prod(
                torch.arange(self.num_classes),  # For first character
                torch.arange(self.num_classes),  # For second character
                torch.arange(self.num_classes),  # For third character
                torch.arange(self.num_classes),  # For fourth character
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
                    [self.vocab[idx.item()] for idx in all_indices[sorted_idx[i]]]
                )
                words.append((word, sorted_probs[i].item()))

            return words

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
