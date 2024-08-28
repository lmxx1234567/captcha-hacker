import torch
import torch.nn as nn
import torch.nn.init as init
from . import vocab


class CaptchaSolverWithTransformer(nn.Module):
    def __init__(
        self,
        img_size=(112, 35),
        patch_size=(16, 16),
        num_channels=1,
        num_classes=62,
        num_characters=4,
        d_model=256,
        nhead=8,
        num_encoder_layers=3,
    ):
        super(CaptchaSolverWithTransformer, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[1]
        )
        self.d_model = d_model
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.num_characters = num_characters

        # Linear projection of flattened patches
        self.patch_embed = nn.Linear(
            patch_size[0] * patch_size[1] * num_channels, d_model
        )

        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, self.num_patches, d_model)
        )

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # Final linear layer to predict character probabilities
        self.classifier = nn.Linear(d_model, num_classes * num_characters)

        # Initialize weights
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            init.xavier_uniform_(module.weight)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.TransformerEncoderLayer):
            init.xavier_uniform_(module.self_attn.in_proj_weight)
            init.xavier_uniform_(module.linear1.weight)
            init.zeros_(module.self_attn.in_proj_bias)
            init.zeros_(module.linear1.bias)

    def forward(self, x):
        # Create patches
        batch_size = x.size(0)
        x = x.unfold(2, self.patch_size[0], self.patch_size[0]).unfold(
            3, self.patch_size[1], self.patch_size[1]
        )
        x = x.contiguous().view(
            batch_size, self.num_channels, -1, self.patch_size[0] * self.patch_size[1]
        )
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, self.num_patches, -1)

        # Linear embedding of patches
        x = self.patch_embed(x)

        # Add positional encoding
        x = x + self.positional_encoding

        # Transformer encoder
        x = self.transformer_encoder(
            x.permute(1, 0, 2)
        )  # (sequence_length, batch_size, embedding_dim)

        # Final classification
        x = x.mean(dim=0)  # Average pooling across the sequence dimension
        x = self.classifier(x)
        x= x.view(-1, self.num_characters, self.num_classes)
        return x
    
    def predict(self, x):
        with torch.no_grad():
            return self.forward(x).argmax(dim=-1)

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
