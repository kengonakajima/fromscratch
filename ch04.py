# Chapter 4: Implementing a GPT model from Scratch To Generate Text
# 4章はGPTみたいなやつをつくる。次の章はそれをTrainする

# [1]

from importlib.metadata import version

print("matplotlib version:", version("matplotlib"))
print("torch version:", version("torch"))
print("tiktoken version:", version("tiktoken"))

# [2] ここでは124mのGPT2に準拠した設定をする

GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size, BPE, in ch02
    "context_length": 1024, # Context length モデルのインプット xの最大長
    "emb_dim": 768,         # Embedding dimension 
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers  この後説明する
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias q,k,vを計算するときに使うバイアス。 ch5で説明。
}


# text > tokens > token ids > token embeddings (768) > GPT MODEL > output vectors (768) > postprocess > text

# [3]
import torch
import torch.nn as nn


class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        # Use a placeholder for TransformerBlock
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        # Use a placeholder for LayerNorm
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # A simple placeholder

    def forward(self, x):
        # This block does nothing and just returns its input.
        return x


class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        # The parameters here are just to mimic the LayerNorm interface.

    def forward(self, x):
        # This layer does nothing and just returns its input.
        return x

    
# [4]

import tiktoken
import torch

tokenizer = tiktoken.get_encoding("gpt2")

batch = []

txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)

# [5]

torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)

logits = model(batch)
print("Output shape:", logits.shape)
print(logits)


# 4.2 Normalizing activations with layer normalization

# Layer normalization, also known as LayerNorm (Ba et al. 2016), centers the activations of a neural network layer around a mean of 0 and normalizes their variance to 1
# This stabilizes training and enables faster convergence to effective weights
# Layer normalization is applied both before and after the multi-head attention module within the transformer block, which we will implement later; it's also applied before the final output layer

# NN層の活性化を、平均0、分散1に正規化する。こうすることで安定化する。

# [6] Let's see how layer normalization works by passing a small input sample through a simple neural network layer:

torch.manual_seed(123)

# create 2 training examples with 5 dimensions (features) each
batch_example = torch.randn(2, 5) 

layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
print(out)

# [7] Let's compute the mean and variance for each of the 2 inputs above:


mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)

print("Mean:\n", mean)  # 平均
print("Variance:\n", var) # 分差

# [8] Subtracting the mean and dividing by the square-root of the variance (standard deviation) centers the inputs to have a mean of 0 and a variance of 1 across the column (feature) dimension:

out_norm = (out - mean) / torch.sqrt(var)
print("Normalized layer outputs:\n", out_norm)

mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)
print("Mean:\n", mean)
print("Variance:\n", var)


# [9] Each input is centered at 0 and has a unit variance of 1; to improve readability, we can disable PyTorch's scientific notation:

torch.set_printoptions(sci_mode=False)
print("Mean:\n", mean)
print("Variance:\n", var)

# Above, we normalized the features of each input
# [10] Now, using the same idea, we can implement a LayerNorm class:


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5 # 分散が0のときにゼロ除算エラーを防ぐため
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
    
# 平均を引いて分散で割る正規化処理に加えて、学習可能な2つのパラメータ（スケールとシフト） を導入しています。
#スケール: 正規化後の値に掛け算する係数（初期値は1）
#シフト: 正規化後の値に足すバイアス（初期値は0）

# [11] Let's now try out LayerNorm in practice:


ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)

# [12]

mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, unbiased=False, keepdim=True)

print("Mean:\n", mean)
print("Variance:\n", var)

# 4.3 Implementing a feed forward network with GELU activations
# 4.3 GELU 活性化関数を用いたフィードフォワードネットワークの実装
# このセクションでは、LLM の Transformer ブロックの一部として使われる小さなニューラルネットワークのサブモジュールを実装します。

# まずは活性化関数から始めます。
# ディープラーニングでは、ReLU（Rectified Linear Unit） がそのシンプルさと多様なニューラルネットワークにおける有効性のため、一般的に使われています。

# しかし LLM では、従来の ReLU 以外のさまざまな活性化関数が使われます。その代表的なものが GELU（Gaussian Error Linear Unit） と SwiGLU（Swish-Gated Linear Unit） です。
# ReLUはカクッと折れるが、GELU, SwiGLUは曲線になる。
# 誤差関数、ガウス分布、

# 実際には近似式が使われる。GPT2もそうなってる

# [15]  Next, let's implement the small neural network module, FeedForward, that we will be using in the LLM's transformer block later:
# feed forwardは、 transformer blockで使う予定なのね。


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))

    
# [16]

print(GPT_CONFIG_124M["emb_dim"]) # 768

# [17]
ffn = FeedForward(GPT_CONFIG_124M)

# input shape: [batch_size, num_token, emb_size]
x = torch.rand(2, 3, 768) 
out = ffn(x)
print(out.shape)


# 4.4 Adding shortcut connections

# もともとショートカット接続は、コンピュータビジョンの深いネットワーク（残差ネットワーク, ResNet）において、勾配消失問題を緩和するために提案されました。
# ショートカット接続は、勾配がネットワークを流れる際に通ることができるより短い経路を提供します。
# これは、ある層の出力を、（間に1層以上を飛ばして）後の層の出力に加えることで実現されます。

# [18]

class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
        ])

    def forward(self, x):
        for layer in self.layers:
            # Compute the output of the current layer
            layer_output = layer(x)
            # Check if shortcut can be applied
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x


def print_gradients(model, x):
    # Forward pass
    output = model(x)
    target = torch.tensor([[0.]])

    # Calculate loss based on how close the target
    # and output are
    loss = nn.MSELoss()
    loss = loss(output, target)
    
    # Backward pass to calculate the gradients
    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            # Print the mean absolute gradient of the weights
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

# 例示用の、deep な NN
# gradient : 勾配

# [19] Let's print the gradient values first without shortcut connections:

layer_sizes = [3, 3, 3, 3, 3, 1]  

sample_input = torch.tensor([[1., 0., -1.]])

torch.manual_seed(123)
model_without_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=False
)
print_gradients(model_without_shortcut, sample_input)

# [20] Next, let's print the gradient values with shortcut connections:

torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=True
)
print_gradients(model_with_shortcut, sample_input)



# 上の出力から分かるように、ショートカット接続は初期の層（layer.0 付近）で勾配が消失するのを防ぎます。
# shortcutなしだと、0.00020173587836325169とかになってしまう。
# まあようするに偏るということだなぁ。
# デメリットは、恒等変換になりやすいこと、計算コストちょっと高くなること。


# 4.5 Connecting attention and linear layers in a transformer block

# Transformer ブロックは、前の章で扱った 因果的マルチヘッドアテンションモジュール と、さらに以前のセクションで実装した 線形層およびフィードフォワードニューラルネットワーク を組み合わせたものです。
# 線形層 : nn.Linear
# さらに、Transformer ブロックでは ドロップアウト と ショートカット接続（残差接続） も利用します。

# [21]

# If the `previous_chapters.py` file is not available locally,
# you can import it from the `llms-from-scratch` PyPI package.
# For details, see: https://github.com/rasbt/LLMs-from-scratch/tree/main/pkg
# E.g.,
# from llms_from_scratch.ch03 import MultiHeadAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        # As in `CausalAttention`, for inputs where `num_tokens` exceeds `context_length`, 
        # this will result in errors in the mask creation further below. 
        # In practice, this is not a problem since the LLM (chapters 4-7) ensures that inputs  
        # do not exceed `context_length` before reaching this forwar

        keys = self.W_key(x) # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2) 
        
        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec

    


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x

# LayerNormは「各トークンの特徴ベクトルを均等に整える下ごしらえ」みたいな役割

# 仮に、2つの入力サンプルがあって、それぞれ6トークンを含んでいるとする。
# その各トークンは、768次元の埋め込みベクトルで表されている。
# このトランスフォーマーブロックは、まず自己注意（Self-Attention）を適用し、その後に線形層を通す。
# そして、入力とほぼ同じサイズの出力を生成する。
# その出力は、前章で説明した「コンテキストベクトル」の拡張版と考えることができる。

# 入力：2サンプル × 6トークン × 768次元ベクトル
# 処理：Self-Attention → 線形層
# 出力：入力と同じ形状（2 × 6 × 768）、ただし「文脈情報が組み込まれた強化版ベクトル」

# [22]
torch.manual_seed(123)

x = torch.rand(2, 4, 768)  # Shape: [batch_size, num_tokens, emb_dim]
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)

print("Input shape:", x.shape) # torch.Size([2,4,768])
print("Output shape:", output.shape) # torch.Size([2,4,768])

