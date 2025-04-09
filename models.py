import torch
import torch.nn as nn

class LSTMwithoutAttn(nn.Module):
    def __init__(self, vocab_size, embd_dims, hidden_size, word2vec_embd, num_layers = 2, dropout = 0.1):
        super(LSTMwithoutAttn, self).__init__()
        
        self._dummy = nn.Parameter(torch.Tensor(0), requires_grad=False)
        self.embedding = nn.Embedding.from_pretrained(word2vec_embd, freeze=False, padding_idx=0)
        self.lstm = nn.LSTM(embd_dims, hidden_size, batch_first= True, num_layers= num_layers, dropout= dropout)
        self.output = nn.Linear(hidden_size, 1)

    @property
    def device(self):
        return self._dummy.device

    def forward(self, x):
        x = self.embedding(x)
        out, hidden = self.lstm(x)
        # out = self.output(out[:,-1,:]).squeeze(dim = 1)
        out = self.output(torch.mean(out, dim=1)).squeeze(dim=1)
        return out
    

class LSTMwithAttention(nn.Module):
    def __init__(self, vocab_size, embd_dims, hidden_size, word2vec_embd, num_layers = 2, dropout = 0.1):
        super(LSTMwithAttention, self).__init__()
        
        self._dummy = nn.Parameter(torch.Tensor(0), requires_grad=False)
        self.embedding = nn.Embedding.from_pretrained(word2vec_embd, freeze=False, padding_idx=0)
        self.lstm = nn.LSTM(embd_dims, hidden_size, batch_first= True, num_layers= num_layers, dropout= dropout)
        self.attention = nn.Linear(hidden_size, 1, bias = False)
        self.output = nn.Linear(hidden_size, 1)

    @property
    def device(self):
        return self._dummy.device

    def forward(self, x):
        x = self.embedding(x)
        out, hidden = self.lstm(x)
        attn_scores = self.attention(out)
        out = out.permute(0,2,1)
        weighted_out = torch.bmm(out, attn_scores).squeeze(dim = 2)
        out = self.output(weighted_out).squeeze(dim=1)
        return out
    
# %%
# MULTIHEAD SELF ATTENTION BLOCK

class MSA(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_dropout):
        super().__init__()

        self.multi_head_self_attn = nn.MultiheadAttention(embed_dim = embed_dim, num_heads = num_heads,
                                                          dropout = attn_dropout, batch_first= True)
        self.layernorm = nn.LayerNorm(normalized_shape= embed_dim)

    def forward(self, x):
        x = self.layernorm(x)
        x, _ = self.multi_head_self_attn(query = x, key = x, value = x, need_weights = False) # it returns both attn output and attn weights
        return x
    
# %%
# MULTI LAYER PERCEPTRON BLOCK
class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_size, mlp_dropout):
        super().__init__()
        self.layernorm = nn.LayerNorm(normalized_shape = embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_size),
            nn.ReLU(),
            nn.Dropout(p = mlp_dropout),
            nn.Linear(mlp_size, embed_dim),
            nn.Dropout(p = mlp_dropout)
        )

    def forward(self, x):
        x = self.layernorm(x)
        x = self.mlp(x)
        return x

# %%
# TRANSFORMER ENCODER USING MULTI-HEAD SELF ATTENTION
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, mlp_size, num_heads, attn_dropout, mlp_dropout):
        super().__init__()

        self.msa_block = MSA(embed_dim, num_heads, attn_dropout)
        self.mlp_block = MLP(embed_dim, mlp_size, mlp_dropout)

    def forward(self, x):
        x = self.msa_block(x) + x
        x = self.mlp_block(x) + x
        return x 
    
# %%
# COMPLETE TRANSFORMER MODEL

class TransformerModel(nn.Module):
    def __init__(self, word2vec_embd, embed_dim, hidden_dim, mlp_size, max_seq_len, num_heads, num_layers, attn_dropout, mlp_dropout):
        super(TransformerModel, self).__init__()

        self._dummy = nn.Parameter(torch.Tensor(0), requires_grad=False)
        self.position_embd = nn.Embedding(max_seq_len, embedding_dim= hidden_dim)
        self.max_seq_len = max_seq_len
        self.word_embd = nn.Embedding.from_pretrained(word2vec_embd, freeze=False, padding_idx=0)
        self.embd_projection = nn.Linear(embed_dim, hidden_dim)
        self.transformer_layer = nn.Sequential(*[
            TransformerEncoder(hidden_dim, mlp_size, num_heads, attn_dropout, mlp_dropout ) for _ in range(num_layers)
        ])
        self.output = nn.Linear(hidden_dim, 1)

    @property
    def device(self):
        return self._dummy.device

    def forward(self, x):
        x = self.word_embd(x)
        x = self.embd_projection(x)
        x = self.position_embd(torch.arange(self.max_seq_len).to(self.device)) + x
        x = self.transformer_layer(x)
        x = self.output(torch.mean(x, dim = 1)).squeeze(dim=1)
        return x


# %%
# CNN Model for Q-3(a)
class AudioCNN(nn.Module):
    def __init__(self, filter_depth, hidden_dims, num_classes, norm_type = None):
        super(AudioCNN, self).__init__()

        self._dummy = nn.Parameter(torch.Tensor(0), requires_grad=False)

        if norm_type is not None and norm_type == 'batchnorm':
            self.norm1 = nn.BatchNorm2d(filter_depth)
            self.norm2 = nn.BatchNorm2d(filter_depth)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, filter_depth, 3, 1),
            self.norm1,
            nn.ReLU(),
            nn.MaxPool2d(3,3),
            nn.Conv2d(filter_depth, 16, 3, 1),
            self.norm2,
            nn.ReLU(),
            nn.MaxPool2d(3, 3)
        )
        self.flatten = nn.Flatten(start_dim= 1, end_dim= 3)

        if (norm_type is not None and norm_type == 'layernorm'):
            self.norm3 = nn.LayerNorm(normalized_shape= filter_depth*13*54)
        else:
            self.norm3 = nn.Identity()

        self.fcn = nn.Sequential(
            nn.Linear(filter_depth*13*54, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, num_classes),
        )

    @property
    def device(self):
        return self._dummy.device

    def forward(self,x):
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.norm3(x)
        x = self.fcn(x)
        return x


# %%
# ENSEMBLE MODEL TRAINING
class EnsembleModel(nn.Module):
    def __init__(self, model1, model2, model3):
        super(EnsembleModel, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self._dummy = nn.Parameter(torch.Tensor(0), requires_grad=False)

        for param in model1.parameters():
            param.requires_grad = False
        for param in model2.parameters():
            param.requires_grad = False
        for param in model3.parameters():
            param.requires_grad = False

        self.weights = nn.Parameter(data = torch.tensor([1/3, 1/3, 1/3]), requires_grad = True)

    @property
    def device(self):
        return self._dummy.device
    
    def forward(self, x):
        y1 = self.model1(x)
        y2 = self.model2(x)
        y3 = self.model3(x)
        alpha = torch.nn.functional.softmax(self.weights)
        return (alpha[0]*y1 + alpha[1]*y2 + alpha[2]*y3)

