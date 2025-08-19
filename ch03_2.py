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




