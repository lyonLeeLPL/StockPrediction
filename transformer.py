'''
Transformer script based on the work of A. Vaswani et. al. (2017) in
"Attention is all you need" and inspired from the work of Frank Odom
in https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51
'''


import torch
import torch.nn as nn
import torch.nn.functional as f

def scaled_dot_product_attention(query, key, value):
    '''
    Computes the local fields and the attention of the inputs as described in Vaswani et. al.
    and then scale it for a total sum of 1

    INPUT: query, key, value - input data of size (batch_size, seq_length, num_features)
    '''

    temp = query.bmm(key.transpose(1,2))
    scale = query.size(-1)**0.5
    softmax = f.softmax(temp / scale, dim = -1)
    attention = softmax.bmm(value)
    return attention


class MultiHeadAttention(nn.Module):
    '''
    Computes the multihead head consisting of a feedforward layer for each input value
    where the attention for all of these are computed for each head and then concatenated and projected
    as described in Vaswani et. al.

    INPUT: dimensions of the three matrices (where the key and query matrix has the same dimensions) and the nr of heads
    OUTPUT: the projected output of the multihead attention
    '''

    def __init__(self, num_heads, input_dim, key_dim, value_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, key_dim)
        self.key = nn.Linear(input_dim, key_dim)
        self.value = nn.Linear(input_dim, value_dim)
        self.num_heads = num_heads

        self.linear = nn.Linear(num_heads * value_dim, input_dim)

    def forward(self, query, key, value):
        multiheads_out = [
            scaled_dot_product_attention(self.query(query), self.key(key), self.value(value)) for _ in range(self.num_heads)
        ]
        out = self.linear(torch.cat(multiheads_out, dim=-1))
        return out

def positioning_encoding(seq_length, model_dim):
    '''
    Computes the positional encoding for the current state of the elements in the input sequence as
    there is no recurrence or convolution. Using the same encoding with sinusosoidal functions as in Vaswani et. al.
    as the motivations of linearly dependency of the relative positions and the ability to extrapolate to sequence lengths
    longer than encountered in training holds strong.
    Code copied from Frank Odom

    INPUT: length of the input sequence and the dimension of the model
    OUTPUT: Encoded relative positions of the data points in the input sequence
    '''
    position = torch.arange(seq_length, dtype=torch.float).reshape(1, -1, 1)
    dimension = torch.arange(model_dim, dtype=torch.float).reshape(1, 1, -1)
    phase = (position / 1e4) ** (dimension // model_dim)
    return torch.where(dimension.long() % 2 == 0, -torch.sin(phase), torch.cos(phase))


def forward(input_dim = 512, forward_dim = 2048):
    '''
    Forward class for the feed-forward layer that is following the multihead
    attention layers

    INPUT: input dimension and the layer size of the forward layer
    OUTPUT: feed-forward layer (nn.Module)
    '''
    forward_layer = nn.Sequential(
        nn.Linear(input_dim, forward_dim),
        nn.ReLU(),
        nn.Linear(forward_dim, input_dim)
    )
    return forward_layer

class ResidualConnection(nn.Module):
    '''
    Class for the residual connections for the encoder and the decoder, used for each multihead attention layer
    and for each feed-forward layer

    INPUT: type of layer, dimension for the layer normalization and dropout probability factor
    OUTPUT: Normalized and processed tensors added to the input tensors
    '''

    def __init__(self, layer, dimension, dropout = 0.2):
        super().__init__()
        self.layer = layer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *X):
        return self.norm(X[-1] + self.dropout(self.layer(*X)))

class Encoder(nn.Module):
    '''
    The encoder of the transformer model, first computes the relative positions of the inputs, then feeds it into
    the multihead attention followed by the feed-forward layer, both with normalized residual connections
    '''

    def __init__(self, n_layers = 6, model_dim = 512, num_heads = 8, forward_dim = 2048, dropout = 0.2):
        super().__init__()

        self.n_layers = n_layers
        key_dim = value_dim = model_dim // num_heads

        # Multihead attention layer with normalized residual connections and dropout
        self.multihead_attention = ResidualConnection(
            MultiHeadAttention(num_heads, model_dim, key_dim, value_dim),
            dimension=model_dim,
            dropout=dropout
        )
        # Feed-forward layer with normalized residual connections and dropout
        self.feed_forward = ResidualConnection(
            forward(model_dim, forward_dim),
            dimension=model_dim,
            dropout=dropout
        )

    def forward(self, X):
        seq_length, dimension = X.size(1), X.size(2)
        out = X
        # Computes the positional encodings
        out += positioning_encoding(seq_length, dimension)
        # Feeds the input to the multihead attention layer followed by the feed-forward
        # layer for 'n_layers' many layers
        for _ in range(self.n_layers):
            att_out = self.multihead_attention(out, out, out)
            out = self.feed_forward(att_out)

        return out



class Decoder(nn.Module):
    '''
    The decoder of the transformer model, first computes the relative positions of the inputs, then feeds it into
    the first multihead attention layer followed by the second multihead attention layer which inputs the output of the
    encoder as the key and query matrices and the output of the first multihead attention layer as value matrix. This
    output is then fed to a feed-forward layer, all of these with normalized residual connections, and the further fed
    to the linear output layer.
    '''

    def __init__(self, n_layers = 6, model_dim = 512, output_dim = 512, num_heads = 8, forward_dim = 2048, dropout = 0.2):
        super().__init__()

        self.n_layers = n_layers
        key_dim = model_dim // num_heads
        value_dim = model_dim // num_heads

        # First multihead attention layer
        self.first_attention = ResidualConnection(
            MultiHeadAttention(num_heads, model_dim, key_dim, value_dim),
            dimension=model_dim,
            dropout=dropout
        )
        # Second multihead attention layer
        self.second_attention = ResidualConnection(
            MultiHeadAttention(num_heads, model_dim, key_dim, value_dim),
            dimension=model_dim,
            dropout=dropout
        )
        # Feed-forward layer
        self.feed_forward = ResidualConnection(
            forward(model_dim, forward_dim),
            dimension=model_dim,
            dropout=dropout
        )
        # Linear output layer
        self.linear = nn.Linear(model_dim, output_dim)

    def forward(self, X_dec, Y_enc):
        seq_len, dimension = X_dec.size(1), X_dec.size(2)
        # Computes the positional encodings
        X_dec += positioning_encoding(seq_len, dimension)

        for _ in range(self.n_layers):
            # All inputs to the first multihead attention layer
            X_dec = self.first_attention(X_dec, X_dec, X_dec)
            # Using the outputs of the encoder as the query and key matrices in
            # the scaled dot product attention and the input as the value matrix
            X_dec = self.second_attention(Y_enc, Y_enc, X_dec)
            # Feeds the output to the feed forward layer
            X_dec = self.feed_forward(X_dec)

        # output linear layer
        out = self.linear(X_dec)
        return out


class TransformerModel(nn.Module):
    '''
    Transformer model that combines the encoder and the decoder
    "model_dim" must be the same size as "num_features" in the input data (i.e size last dimension),
    otherwise freely tunable parameters
    '''

    def __init__(self, n_layers_enc = 6, n_layers_dec = 6, model_dim = 512, output_dim = 512,
                 num_heads = 6, forward_dim = 2048, dropout = 0.2):
        super().__init__()
        self.encoder = Encoder(n_layers_enc, model_dim, num_heads, forward_dim, dropout)
        self.decoder = Decoder(n_layers_dec, model_dim, output_dim, num_heads, forward_dim, dropout)

    def forward(self, X, Y):
        enc_out = self.encoder(X)
        dec_out = self.decoder(Y, enc_out)
        return dec_out


# Test with random tensors
X = torch.rand(32, 64, 32)
Y = torch.rand(32, 64, 32)
out = TransformerModel(model_dim = 32, output_dim=50)(X, Y)
print(out.shape)

