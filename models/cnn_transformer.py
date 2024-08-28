# use a CNN for patch generation

import torch
import torch.nn as nn
from . import vocab
import math

class CaptchaSolverWithCNNTransformer(nn.Module):
    def __init__(
        self,
        img_size=(112, 35),
        num_channels=1,
        num_classes=62,
        num_characters=4,
        d_model=256,
        nhead=8,
        num_encoder_layers=3,
    ):
        super(CaptchaSolverWithCNNTransformer, self).__init__()
        self.img_size = img_size
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.num_characters = num_characters
        self.d_model = d_model

        # CNN for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Downsample by 2
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # Downsample by 2
            nn.ReLU(),
            nn.Conv2d(256, d_model, kernel_size=3, stride=2, padding=1),  # Downsample by 2
            nn.ReLU(),
        )

        # Calculate the output dimensions after CNN layers
        self.num_patches = math.ceil(img_size[0] / 8) * math.ceil(img_size[1] / 8)

        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, self.num_patches, d_model)
        )

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers,
        )

        # Final linear layer to predict character probabilities
        self.classifier = nn.Linear(d_model, num_classes * num_characters)

        # Initialize weights
        self.apply(self._initialize_weights)

    def forward(self, x):
        batch_size = x.size(0)

        # Extract patches using CNN
        x = self.cnn(x)  # Output shape: [batch_size, d_model, height, width]
        x = x.flatten(2).transpose(1, 2)  # Reshape to [batch_size, num_patches, d_model]

        # Add positional encoding
        x = x + self.positional_encoding

        # Transformer encoder
        x = self.transformer_encoder(
            x.permute(1, 0, 2)
        )  # (sequence_length, batch_size, embedding_dim)

        # Final classification
        x = x.mean(dim=0)  # Average pooling across the sequence dimension
        x = self.classifier(x)
        x = x.view(-1, self.num_characters, self.num_classes)
        return x
    
    def predict_all_possible(self, x):
        with torch.no_grad():
            # Get each character's probabilities
            probs = self.forward(x).view(-1, 4, 62).softmax(dim=-1)

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
    
    def _initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)