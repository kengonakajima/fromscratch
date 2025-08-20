import torch

# [2]
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)


# [3] : step 1. queryとそれぞれのinputs との attention score を計算する。正規化前。

query = inputs[1]  # 2nd input token is the query

attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query) # dot product (transpose not necessary here since they are 1-dim vectors)

print(attn_scores_2) 

# [4] 

res = 0.

for idx, element in enumerate(inputs[0]):
    res += inputs[0][idx] * query[idx]

print(res)
print(torch.dot(inputs[0], query)) # それ自身とのdot

# [5]

attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()

print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())


# [6]

def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)

print("Attention weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())

# [7] step 2. attention score を正規化して attention weightsを計算. softmaxを使う

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())


# [8]

query = inputs[1] # inputs と attention weightsをかけてcontext vectorを得る   2nd input token is the query

context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i

print(context_vec_2)

# 3.3.2 Computing attention weights for all input tokens


# [9] Apply previous step 1 to all pairwise elements to compute the unnormalized attention score matrix:
# まずすべてのinputs 6個について、  attention score を計算する. 6x6の行列が必要。

attn_scores = torch.empty(6, 6)

for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)

print(attn_scores) # inputsが6なので 6x6の行列になる。計算いっぱい必要だねぇ

# [10] We can achieve the same as above more efficiently via matrix multiplication:
# inputsのベクトルとその転置ベクトルをかけると 6x6になる。[9]と同じ結果になることを確認。

attn_scores = inputs @ inputs.T
print(attn_scores)


# [11] Similar to step 2 previously, we normalize each row so that the values in each row sum to 1:
# step2、合計したら1になるように正規化する

attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)

# [12] Quick verification that the values in each row indeed sum to 1:
# 簡単に確認しとく

row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("Row 2 sum:", row_2_sum)

print("All row sums:", attn_weights.sum(dim=-1)) # [6]がぜんぶ1.000になってるで


# [13] Apply previous step 3 to compute all context vectors:
# すべてのトークンについてcontext vectorを求めるで

all_context_vecs = attn_weights @ inputs
print(all_context_vecs)

# [14] As a sanity check, the previously computed context vector  can be found in the 2nd row in above:

print("Previous 2nd context vector:", context_vec_2)  # 2つ眼にみつかったよ




# 3.4 Implementing self-attention with trainable weights
# 訓練可能な自己注意を実装する

# 3.4.1 Computing the attention weights step by step

# LLMで使われる自己注意は、 scaled dot-product attention とも呼ばれる。
# 3.3までの単純な注意機構との違いは、
#  - モデルのトレーニング中に更新される weight行列sを導入すること。
#  - このweight matricesは、よりよい context vectors を生成するために学習できる。


# Wq, Wk, Wv の3つの training weight matricesを導入する。
# これらの3つのtraining weight matricesは,
# q(i) = x(i)Wq
# k(i) = x(i)Wk
# v(i) = x(i)Wv
# というように、q,k,vの3つをそれぞれx(i)に対してかけ算して生成する。

# The embedding dimensions of the input x  and the query vector q  can be the same or different, depending on the model's design and specific implementation
# xとqの次数は違っててもいいよ

# [15] In GPT models, the input and output dimensions are usually the same, but for illustration purposes, to better follow the computation, we choose different input and output dimensions here:
# GPTとかだと大体同じ次数になるが、ここでは理解のために次数を異なるようにしてみる

x_2 = inputs[1] # second input element
d_in = inputs.shape[1] # the input embedding size, d=3
d_out = 2 # the output embedding size, d=2

# Below, we initialize the three weight matrices;  次に3種類のWを初期化する。
# note that we are setting requires_grad=False to reduce clutter in the outputs for illustration purposes, but if we were to use the weight matrices for model training, we would set requires_grad=True to update these matrices during model training
# requires_grad は、係数の更新が可能ならTrue.

torch.manual_seed(123)

W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# [17] Next we compute the query, key, and value vectors:
# q,k,v を計算する

query_2 = x_2 @ W_query # _2 because it's with respect to the 2nd input element
key_2 = x_2 @ W_key 
value_2 = x_2 @ W_value

print(query_2)

# [18] As we can see below, we successfully projected the 6 input tokens from a 3D onto a 2D embedding space:
#

keys = inputs @ W_key 
values = inputs @ W_value

print("keys.shape:", keys.shape) #  ここはあえて d_outが2になってて、 d_inの1とは違う値にしている。LLMはだいたい同じなんだけど。
print("values.shape:", values.shape)



# [19] In the next step, step 2, we compute the unnormalized attention scores by computing the dot product between the query and each key vector:

# 次は、 query vectorとkey vector の内積を計算して、 attention scoresを計算する

keys_2 = keys[1] # Python starts index at 0
attn_score_22 = query_2.dot(keys_2)
print(attn_score_22)

# [20] Since we have 6 inputs, we have 6 attention scores for the given query vector:
#
attn_scores_2 = query_2 @ keys.T # All attention scores for given query
print(attn_scores_2) # 6次のベクトルになった。

# [21] Next, in step 3, we compute the attention weights (normalized attention scores that sum up to 1) using the softmax function we used earlier.
# The difference to earlier is that we now scale the attention scores by dividing them by the square root of the embedding dimension,   (i.e., d_k**0.5):

d_k = keys.shape[1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print(attn_weights_2) # softmaxで正規化する。次元数の平方根で割る。1なら1だけどここでは2なので sqrt(2) で割ってる





# [22] In step 4, we now compute the context vector for input query vector 2:

context_vec_2 = attn_weights_2 @ values
print(context_vec_2)  # d_out=2なので、2次のベクトルがcontext vectorが出力。これで3種類の Wを用いたcontext vectorが生成された。


# 3.4.2 Implementing a compact SelfAttention class
# [23] Putting it all together, we can implement the self-attention mechanism as follows:

import torch.nn as nn

class SelfAttention_v1(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )

        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))

# [24] nn.Linearをつかったらもっとスッキリできます

class SelfAttention_v2(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))  #なんか値がだいぶ違うけど。。

# Note that SelfAttention_v1 and SelfAttention_v2 give different outputs because they use different initial weights for the weight matrices
# 初期Wが違うから値が違うと。
# nn.Linearは乱数を用いて初期化される。それが、torch.randとは異なる。

# 3.5 Hiding future words with causal attention


# In causal attention, the attention weights above the diagonal are masked, ensuring that for any given input, the LLM is unable to utilize future tokens while calculating the context vectors with the attention weight
# 因果的（causal）アテンションでは、対角線より上のアテンション重みがマスクされる。これにより、任意の入力に対して、LLM は文脈ベクトルをアテンション重みで計算する際に未来のトークンを利用できないようにしている。
# ここでいう 因果的 (causal) とは、「原因があって結果が生じる」 という時間的な一方向性を守る、という意味です。

# attention weightの 6x6の行列の右上部分についてマスクしてしまう。

# 3.5.1 Applying a causal attention mask


# このセクションでは、前の自己注意メカニズムを 因果的自己注意メカニズム に変換します。
# 因果的自己注意は、系列中のある位置に対するモデルの予測が、その位置より後のトークンではなく、既知の前の位置の出力のみに依存するようにするものです。
# 簡単に言えば、次の単語の予測は常に直前までの単語にのみ依存するようにします。
# これを実現するために、各トークンについて、そのトークン以降（入力テキスト中で後に出現するトークン）をマスクします。

# [25] To illustrate and implement causal self-attention, let's work with the attention scores and weights from the previous section:
#

# Reuse the query and key weight matrices of the
# SelfAttention_v2 object from the previous section for convenience
queries = sa_v2.W_query(inputs)  # 計算済みのWqの値
keys = sa_v2.W_key(inputs)    # 計算済みの Wkの値
attn_scores = queries @ keys.T

attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
print(attn_weights)


# [26] The simplest way to mask out future attention weights is by creating a mask via PyTorch's tril function with elements below the main diagonal (including the diagonal itself) set to 1 and above the main diagonal set to 0:

context_length = attn_scores.shape[0] # 6
mask_simple = torch.tril(torch.ones(context_length, context_length)) # 右上だけ0にするやつ
print(mask_simple)

# [27] Then, we can multiply the attention weights with this mask to zero out the attention scores above the diagonal:
# 1,0のマスクをかけたらOK

masked_simple = attn_weights*mask_simple
print(masked_simple)

#マスクすると正規化が壊れるので、re-normalizeする必要ある。

# [28] To make sure that the rows sum to 1, we can normalize the attention weights as follows:

row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)


# [29] つまり、対角線より上の注意重みをゼロにして結果を再正規化する代わりに、softmax関数に入力する前に、対角線より上の非正規化注意スコアを負の無限大でマスクすることができます。

mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked) # これはスコアです



# [30] As we can see below, now the attention weights in each row correctly sum to 1 again:

attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)
print(attn_weights) # 正規化して重みになった。


# 3.5.2 Masking additional attention weights with dropout
# ドロップアウトによって過学習を防ぐ。不必要なつながりをなくせる。　面白いなぁ
# ここではランダムドロップアウト率を0.5とするが、GPT2だと0.1とか0.2。

# If we apply a dropout rate of 0.5 (50%), the non-dropped values will be scaled accordingly by a factor of 1/0.5 = 2
# The scaling is calculated by the formula 1 / (1 - dropout_rate)
# [31] ドロップアウトするので、その率に応じてスケーリングする必要がある。

torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5) # dropout rate of 50%
example = torch.ones(6, 6) # create a matrix of ones

print(dropout(example)) # ランダムな2と0からなる6x6行列


# [32]

torch.manual_seed(123)
print(dropout(attn_weights)) #なんか合計が1越えてる

# 3.5.3 Implementing a compact causal self-attention class
# casual じゃなくて、 causalだった。。因果的。
# causalマスク(右上0)　と、ドロップアウトマスクを適用した。

# バッチを作る必要がある。


# [33] まず単純に同じのを重ねてみる

batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape) # 2 inputs with 6 tokens each, and each token has embedding dimension 3




# [34] classにまとめる

class CausalAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length,
                 dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout) # New
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)) # New

    def forward(self, x):
        b, num_tokens, d_in = x.shape # New batch dimension b
        # For inputs where `num_tokens` exceeds `context_length`, this will result in errors
        # in the mask creation further below.
        # In practice, this is not a problem since the LLM (chapters 4-7) ensures that inputs  
        # do not exceed `context_length` before reaching this forward method. 
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2) # Changed transpose
        attn_scores.masked_fill_(  # New, _ ops are in-place
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)  # `:num_tokens` to account for cases where the number of tokens in the batch is smaller than the supported context_size
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights) # New

        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(123)

context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)

context_vecs = ca(batch)

print(context_vecs) # とりあえず2つ出る
print("context_vecs.shape:", context_vecs.shape)

# Note that dropout is only applied during training, not during inference
# droopoutは学習中のみなんだね


# 3.6 Extending single-head attention to multi-head attention
# シングルヘッドをマルチヘッドに拡張する

# [35] 「マルチヘッドアテンションの主要なアイデアは、異なる学習された線形射影を用いて、注意機構を複数回（並列に）実行することです。これにより、モデルは異なる位置における異なる表現部分空間からの情報に同時に注意を向けることができます。」
# ランダム性を用いて、いろんな意味を学習するんだなぁ!

# ここまでにつくったシングルヘッド注意を重ねていくだけ。

class MultiHeadAttentionWrapper(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) 
             for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


torch.manual_seed(123)

context_length = batch.shape[1] # This is the number of tokens
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(
    d_in, d_out, context_length, 0.0, num_heads=2
)

context_vecs = mha(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)




# [36] CausalAttention を単純に2つ重ねると、Wが無駄に多いので減らす。
# wrapperの効率を改善しただけ。

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

torch.manual_seed(123)

batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)

context_vecs = mha(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)


# もっと効率よくした、 torch.nn.MultiheadAttention というのもあるよ。

# [37] Since the above implementation may look a bit complex at first glance, let's look at what happens when executing attn_scores = queries @ keys.transpose(2, 3):

# (b, num_heads, num_tokens, head_dim) = (1, 2, 3, 4)
a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573],
                    [0.8993, 0.0390, 0.9268, 0.7388],
                    [0.7179, 0.7058, 0.9156, 0.4340]],

                   [[0.0772, 0.3565, 0.1479, 0.5331],
                    [0.4066, 0.2318, 0.4545, 0.9737],
                    [0.4606, 0.5159, 0.4220, 0.5786]]]])

print(a @ a.transpose(2, 3))


# [38] For instance, the following becomes a more compact way to compute the matrix multiplication for each head separately:

first_head = a[0, 0, :, :]
first_res = first_head @ first_head.T
print("First head:\n", first_res)

second_head = a[0, 1, :, :]
second_res = second_head @ second_head.T
print("\nSecond head:\n", second_res)



