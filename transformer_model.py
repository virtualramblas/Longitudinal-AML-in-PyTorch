import torch.nn as nn
import torch

class PyTorchMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(PyTorchMultiHeadAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=False) # Keras input (batch, seq, features), PyTorch (seq, batch, features) by default

    def forward(self, x, return_attention_weights=False):
        x_permuted = x.permute(1, 0, 2)

        # Pass the permuted input to nn.MultiheadAttention
        # query, key, value are all the same for self-attention
        attn_output, attn_output_weights = self.multihead_attn(query=x_permuted, key=x_permuted, value=x_permuted, need_weights=return_attention_weights)

        # Permute the output back to (batch_size, sequence_length, features)
        attn_output_original_dim = attn_output.permute(1, 0, 2)

        if return_attention_weights:
            return attn_output_original_dim, attn_output_weights
        else:
            return attn_output_original_dim
    
class PyTorchFeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim):
        super(PyTorchFeedForwardNetwork, self).__init__()
        # First linear layer: input_dim -> 64
        self.dense1 = nn.Linear(embed_dim, 64)
        # Second linear layer: 64 -> output_dim
        self.dense2 = nn.Linear(64, embed_dim)

    def forward(self, x):
        # Pass through the first dense layer and apply ReLU activation
        x = self.dense1(x)
        x = torch.relu(x)
        # Pass through the second dense layer
        x = self.dense2(x)
        return x

class PyTorchLayerNormalization(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super(PyTorchLayerNormalization, self).__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, x):
        return self.layer_norm(x)

class PyTorchGlobalAveragePooling1D(nn.Module):
    def __init__(self):
        super(PyTorchGlobalAveragePooling1D, self).__init__()

    def forward(self, x):
        return x.mean(dim=1)
    
class PyTorchTransformerModel(nn.Module):
    def __init__(self, embed_dim, num_heads, num_classes, dropout_rate=0.1):
        super(PyTorchTransformerModel, self).__init__()
        self.multi_head_attention = PyTorchMultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.layer_norm1 = PyTorchLayerNormalization(normalized_shape=embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.feed_forward_network = PyTorchFeedForwardNetwork(embed_dim=embed_dim)
        self.layer_norm2 = PyTorchLayerNormalization(normalized_shape=embed_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.global_average_pooling = PyTorchGlobalAveragePooling1D()
        self.classification_head = nn.Linear(embed_dim, num_classes)

    def forward(self, inputs, return_attention_weights=False):
        # Multi-Head Attention Layer
        if return_attention_weights:
            attn_output, attn_weights = self.multi_head_attention(inputs, return_attention_weights=True)
        else:
            attn_output = self.multi_head_attention(inputs)

        # LayerNormalization with residual connection
        attn_output = self.dropout1(attn_output)
        attention_output_norm = self.layer_norm1(attn_output + inputs)

        # Feedforward Network
        ffn_output = self.feed_forward_network(attention_output_norm)

        # LayerNormalization with residual connection
        ffn_output = self.dropout2(ffn_output)
        ffn_output_norm = self.layer_norm2(ffn_output + attention_output_norm)

        # Global Average Pooling
        pooled_output = self.global_average_pooling(ffn_output_norm)

        # Final Classification Layer
        outputs = self.classification_head(pooled_output)
        # Apply softmax for classification probabilities
        outputs = torch.softmax(outputs, dim=-1)

        if return_attention_weights:
            return outputs, attn_weights
        else:
            return outputs
    