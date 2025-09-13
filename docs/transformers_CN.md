
## 🔑 第一层：必答八股文（重点掌握）

### 1. Transformer 架构

**是什么：**
- Transformer是一种基于自注意力机制的神经网络架构，由Google在2017年提出
- 核心组件：Encoder-Decoder结构、Self-Attention、Multi-Head Attention、Feed Forward、Residual Connection、Layer Normalization

**怎么做：**
```
Encoder: Input → Embedding + Position Encoding → Multi-Head Self-Attention → Add&Norm → Feed Forward → Add&Norm
Decoder: 类似Encoder，但增加了Masked Self-Attention和Cross-Attention
```

**核心机制详解：**
- **Self-Attention**: 每个位置的token都能关注到序列中所有位置的信息
- **Multi-Head**: 将注意力分成多个头，捕获不同类型的关系
- **Position Encoding**: 正弦/余弦编码为序列添加位置信息
- **Residual + LayerNorm**: 解决梯度消失，稳定训练

**优缺点：**
- 优点：并行化训练、长程依赖处理好、表现力强
- 缺点：计算复杂度O(n²)、内存消耗大、需要大量数据

**什么时候用：**
机器翻译、文本生成、语言理解任务，现在是LLM的主流架构


#### Layer Normalization
- 个人心得：
这里的add&norm的norm是Layer Normalization。他跟神经网络里经常出现的范数norm有什么区别？



**Layer Normalization vs 范数(Norm)**

**Layer Normalization（层归一化）**
**是什么：**
- 是一种**归一化技术**，用于稳定神经网络训练
- 对每个样本的特征维度进行标准化处理

**怎么做：**
```python
# 对单个样本的所有特征维度归一化
mean = x.mean(dim=-1, keepdim=True)  # 计算均值
var = x.var(dim=-1, keepdim=True)    # 计算方差
x_norm = (x - mean) / sqrt(var + eps) # 标准化
output = gamma * x_norm + beta        # 可学习的缩放和偏移
```

**目的：**
- 解决梯度消失/爆炸问题
- 加速收敛
- 提高训练稳定性

**范数(Norm)**
**是什么：**
- 是**数学概念**，衡量向量"大小"的度量
- 常见的有L1范数、L2范数、无穷范数等

**怎么做：**
```python
# L1范数：所有元素绝对值之和
L1_norm = |x1| + |x2| + ... + |xn|

# L2范数：所有元素平方和的开方
L2_norm = sqrt(x1² + x2² + ... + xn²)

# 无穷范数：最大绝对值
L∞_norm = max(|x1|, |x2|, ..., |xn|)
```

**目的：**
- 正则化（防止过拟合）
- 梯度裁剪（防止梯度爆炸）
- 相似度计算

**关键区别对比**

| 维度 | Layer Normalization | 范数(Norm) |
|------|-------------------|-----------|
| **性质** | 数据预处理/归一化操作 | 数学度量概念 |
| **作用对象** | 特征维度（跨通道） | 向量整体 |
| **输出** | 归一化后的向量（维度不变） | 标量数值 |
| **用途** | 稳定训练过程 | 正则化、度量、约束 |
| **可学习** | 有可学习参数(γ, β) | 无参数，纯计算 |

**实际例子说明**

假设有一个向量 `x = [1, 100, 0.01]`

**Layer Normalization：**
```python
mean = (1 + 100 + 0.01) / 3 ≈ 33.67
var = ((1-33.67)² + (100-33.67)² + (0.01-33.67)²) / 3
x_norm ≈ [-0.71, 1.41, -0.71]  # 归一化后的向量
```

**L2范数：**
```python
L2_norm = sqrt(1² + 100² + 0.01²) ≈ 100.005  # 一个标量
```

**为什么Transformer用Layer Norm？**

1. **跨特征归一化**：每个token的所有特征维度一起归一化
2. **独立处理**：每个样本独立处理，不依赖batch
3. **梯度稳定**：防止不同层之间的激活值差异过大

而范数更多用于：
- **权重衰减**：L2正则化 `loss += λ * ||W||²`
- **梯度裁剪**：`if ||grad|| > threshold: grad = grad * threshold / ||grad||`

所以虽然都叫"norm"，但Layer Normalization是一种**数据处理技术**，而范数是一种**数学测量工具**，应用场景完全不同！

#### Batch Normalization VS Layer Normalization
##### Batch Normalization的些许缺陷
要讲Layer Normalization，先讲讲Batch Normalization存在的一些问题：即不适用于什么场景。

1. BN在mini-batch较小的情况下不太适用。
BN是对整个mini-batch的样本统计均值和方差，当训练样本数很少时，样本的均值和方差不能反映全局的统计分布信息，从而导致效果下降。
2. BN无法应用于RNN。
RNN实际是共享的MLP，在时间维度上展开，每个step的输出是(bsz, hidden_dim)。由于不同句子的同一位置的分布大概率是不同的，所以应用BN来约束是没意义的。注：而BN应用在CNN可以的原因是同一个channel的特征图都是由同一个卷积核产生的。

LN原文的说法是：在训练时，对BN来说需要保存每个step的统计信息（均值和方差）。在测试时，由于变长句子的特性，测试集可能出现比训练集更长的句子，所以对于后面位置的step，是没有训练的统计量使用的。（不过实践中的话都是固定了max len，然后padding的。）
当然还有一种说法是，不同句子的长度不一样，对所有的样本统计均值是无意义的，因为某些样本在后面的timestep时其实是padding。
还有一种说法是（Normalization helps training of quantized lstm.）：应用BN层的话，每个timestep都需要去保存和计算batch统计量，耗时又耗力，后面就有人提出across timestep去shared BN的统计量，这明显不对，因为不同timestep的分布明显是不同的。
最后，大家发现LN的效果还很不错，比BN好，所以就变成NLP data里面的default config了。

##### Layer Normalization的原理

一言以蔽之。BN是对batch的维度去做归一化，也就是针对不同样本的同一特征做操作。LN是对hidden的维度去做归一化，也就是针对单个样本的不同特征做操作。因此LN可以不受样本数的限制。

具体而言，BN就是在每个维度上统计所有样本的值，计算均值和方差；LN就是在每个样本上统计所有维度的值，计算均值和方差（注意，这里都是指的简单的MLP情况，输入特征是（bsz，hidden_dim））。所以BN在每个维度上分布是稳定的，LN是每个样本的分布是稳定的。
```
# 计算均值和方差
x = torch.randn(bsz, hidden_dim)
mu = x.mean(dim=1) # 注意！要统计的是每个样本所有维度的值，所以应该是dim=1上求均值
sigma = x.std(dim=1)
```
Transformer中Layer Normalization的实现
对于一个输入tensor：(batch_size, max_len, hidden_dim) 应该如何应用LN层呢？

注意，和Batch Normalization一样，同样会施以线性映射的。区别就是操作的维度不同而已！公式都是统一的：减去均值除以标准差，施以线性映射。同时LN也有BN的那些个好处！
```
# features: (bsz, max_len, hidden_dim)
# 
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
	super(LayerNorm, self).__init__()
	self.a_2 = nn.Parameter(torch.ones(features))
	self.b_2 = nn.Parameter(torch.zeros(features))
	self.eps = eps
	
    def forward(self, x):
	# 就是在统计每个样本所有维度的值，求均值和方差，所以就是在hidden dim上操作
	# 相当于变成[bsz*max_len, hidden_dim], 然后再转回来, 保持是三维
	mean = x.mean(-1, keepdim=True) # mean: [bsz, max_len, 1]
	std = x.std(-1, keepdim=True) # std: [bsz, max_len, 1]
        # 注意这里也在最后一个维度发生了广播
	return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
```
讨论：Transformer 为什么使用 Layer normalization，而不是其他的归一化方法？
当然这个问题还没有啥定论，包括BN和LN为啥能work也众说纷纭。这里先列出一些相关的研究论文。

Leveraging Batch Normalization for Vision Transformers
PowerNorm: Rethinking Batch Normalization in Transformers
Understanding and Improving Layer Normalization
Understanding and Improving Layer Normalization
这篇文章主要研究LN为啥work，除了一般意义上认为可以稳定前向输入分布，加快收敛快，还有没有啥原因。最后的结论有：

相比于稳定前向输入分布，反向传播时mean和variance计算引入的梯度更有用，可以稳定反向传播的梯度（让 
 梯度的均值趋于0，同时降低其方差，相当于re-zeros和re-scales操作），起名叫gradient normalization（其实就是ablation了下，把mean和variance的梯度断掉，看看效果）
去掉 gain和bias这两个参数可以在很多数据集上有提升，可能是因为这两个参数会带来过拟合，因为这两个参数是在训练集上学出来的
注：Towards Stabilizing Batch Statistics in Backward Propagation 也讨论了额外两个统计量：mean和variance的梯度的影响。实验中看到了对于小的batch size，在反向传播中这两个统计量的方差甚至大于前向输入分布的统计量的方差，其实说白了就是这两个与梯度相关的统计量的不稳定是BN在小batch size下不稳定的关键原因之一。

**总结**
Layer Normalization和Batch Normalization一样都是一种归一化方法，因此，BatchNorm的好处LN也有，当然也有自己的好处：比如稳定后向的梯度，且作用大于稳定输入分布。然而BN无法胜任mini-batch size很小的情况，也很难应用于RNN。LN特别适合处理变长数据，因为是对channel维度做操作(这里指NLP中的hidden维度)，和句子长度和batch大小无关。BN比LN在inference的时候快，因为不需要计算mean和variance，直接用running mean和running variance就行。BN和LN在实现上的区别仅仅是：BN是对batch的维度去做归一化，也就是针对不同样本的同一特征做操作。LN是对hidden的维度去做归一化，也就是针对单个样本的不同特征做操作。因此，他们都可以归结为：减去均值除以标准差，施以线性映射。对于NLP data来说，Transformer中应用BN并不好用，原因是前向和反向传播中，batch统计量及其梯度都不太稳定。而对于VIT来说，BN也不是不能用，但是需要在FFN里面的两层之间插一个BN层来normalized。

#### Cross Attention vs Self-Attention

是什么：

不同序列之间的注意力机制
Q来自一个序列，K和V来自另一个序列

怎么做：
```
python# Query来自目标序列（比如要生成的文本）
Q = target_sequence  
 Key和Value来自源序列（比如要翻译的原文）
K = V = source_sequence

attention_weights = softmax(Q @ K.T / sqrt(d))
output = attention_weights @ V  # 用源序列的信息来更新目标序列
```

**关键区别对比**
维度Self-AttentionCross Attention信息来源序列内部不同序列之间Q来源当前序列目标序列K,V来源当前序列源序列作用内部依赖建模跨序列信息传递典型位置Encoder/Decoder内部Decoder中
具体应用场景
1. 机器翻译（最经典的例子）
源语言：["I", "love", "you"]
目标语言：["我", "爱", "你"]

Cross Attention让"我"能够关注到"I"
让"爱"能够关注到"love"
让"你"能够关注到"you"
工作流程：
```
python

# Encoder处理源语言
encoder_output = encoder(["I", "love", "you"])  # 得到源语言的表示

Decoder生成目标语言时
for i in range(target_length):
    # Self-Attention: 目标序列内部注意力
    decoder_hidden = self_attention(target_so_far)
    
    # Cross Attention: 关注源语言信息
    Q = decoder_hidden  # 来自目标语言
    K = V = encoder_output  # 来自源语言
    cross_attended = cross_attention(Q, K, V)
```

2. 文档问答系统
文档：["今天", "天气", "很好", "适合", "出门"]
问题：["今天", "天气", "怎么样", "？"]

Cross Attention让问题中的"天气"关注到文档中的"天气很好"
3. 图像描述生成
图像特征：[feature1, feature2, feature3, ...]
描述文本：["一只", "可爱的", "小猫", "在", "草地上"]

Cross Attention让"小猫"这个词关注到图像中猫的特征区域
Transformer Decoder中的三种Attention
pythonclass TransformerDecoderLayer:
    def forward(self, target, source):
        # 1. Masked Self-Attention (目标序列内部)
        target = self.masked_self_attention(target)
        
        # 2. Cross Attention (目标→源)
        target = self.cross_attention(
            query=target,      # 来自目标序列
            key=source,        # 来自源序列  
            value=source       # 来自源序列
        )
        
        # 3. Feed Forward
        target = self.feed_forward(target)
        
        return target
现代应用中的Cross Attention
1. RAG系统中
python# 用户问题作为Query
question = "什么是机器学习？"

**检索到的文档作为Key/Value**
retrieved_docs = ["机器学习是人工智能的分支...", "它使用算法..."]

**Cross Attention让问题关注到相关文档片段**
answer = cross_attention(question, retrieved_docs, retrieved_docs)
2. Vision-Language模型中
python# 图像特征作为Key/Value
image_features = vision_encoder(image)

**文本作为Query**
text_tokens = ["描述", "这张", "图片"]

**Cross Attention实现图文对齐**
output = cross_attention(text_tokens, image_features, image_features)
什么时候用Cross Attention？
典型场景：

序列到序列任务：机器翻译、文本摘要
条件生成：基于图像生成描述、基于问题生成答案
多模态融合：图像+文本、音频+文本
信息检索增强：RAG、知识图谱问答

判断标准：

需要在两个不同信息源之间建立对应关系时
一个序列需要"参考"另一个序列的信息时
条件生成任务（有条件输入+目标输出）

面试回答要点
核心理解：

Self-Attention是"自己看自己"
Cross Attention是"我看你，学习你的信息"
本质都是注意力机制，只是信息来源不同

实际意义：
Cross Attention是实现跨模态、跨序列信息传递的关键机制，没有它就无法实现真正的序列到序列建模！

---

### 2. Q, K, V 含义与计算

**是什么：**
- Q (Query): 查询向量，表示"我要找什么"
- K (Key): 键向量，表示"我能提供什么信息"  
- V (Value): 值向量，表示"我实际包含的内容"

**怎么做：**
```
Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V

步骤：
1. 计算相似度：Q与所有K做点积 → QK^T
2. 缩放：除以sqrt(d_k)防止梯度消失
3. 归一化：softmax得到注意力权重
4. 加权求和：权重乘以V得到输出
```

**物理意义：**
- QK^T：衡量query与每个key的相关性
- softmax：将相关性转为概率分布
- 加权V：根据相关性获取对应的value信息

**什么时候用：**
Transformer中的核心计算，包括Self-Attention和Cross-Attention

---

### 3. Transformer vs RNN/LSTM优势

**并行化优势：**
- RNN/LSTM：必须顺序处理，t时刻依赖t-1时刻
- Transformer：所有位置可以同时计算，GPU友好

**长程依赖处理：**
- RNN/LSTM：信息需要逐步传递，长距离容易丢失
- Transformer：任意两个位置直接计算attention，O(1)距离

**表达能力：**
- RNN：单一隐藏状态，信息压缩严重
- Transformer：Multi-Head机制捕获多维度关系

**训练效率：**
- RNN：梯度消失/爆炸问题严重
- Transformer：Residual连接，训练更稳定

---

### 4. LLM微调方法对比

#### 全量微调 (Full Fine-Tuning)

**是什么：**
更新模型的所有参数来适应特定任务

**怎么做：**
```python
# 伪代码
model = load_pretrained_model()
for epoch in range(num_epochs):
    for batch in dataloader:
        loss = model(batch)
        loss.backward()  # 所有参数都更新
        optimizer.step()
```

**优缺点：**
- 优点：效果最好，完全适配目标任务
- 缺点：显存需求大、训练时间长、易过拟合

**什么时候用：**
数据量充足且计算资源充裕时

#### PEFT方法详解

### LoRA (Low-Rank Adaptation)

**是什么：**
在预训练权重上添加低秩分解的可训练参数

**怎么做：**
```
原权重：W ∈ R^(d×k)
LoRA：ΔW = A·B，其中 A ∈ R^(d×r), B ∈ R^(r×k), r << min(d,k)
新输出：h = W·x + ΔW·x = W·x + A·B·x
```

**优缺点：**
- 优点：参数量少（通常<1%）、显存友好、训练快
- 缺点：表达能力有限，复杂任务可能效果不如全量微调

**关键参数：**
- rank r：通常4-64，越大表达能力越强但参数越多
- alpha：缩放系数，控制LoRA的影响程度

### QLoRA

**是什么：**
结合4-bit量化的LoRA，进一步减少显存占用

**怎么做：**
1. 将预训练模型量化为4-bit（显存减少75%）
2. 在量化模型上应用LoRA
3. 训练时只更新LoRA参数

**优势：**
单GPU就能微调大模型，显存需求降低90%+

### P-Tuning v2

**是什么：**
在每一层都加入可训练的prompt embedding

**怎么做：**
```
原输入：[CLS] token1 token2 ... [SEP]
P-Tuning v2：[P1] [P2] ... [Pk] token1 token2 ... [SEP]
```

**优缺点：**
- 优点：参数极少、适合理解类任务
- 缺点：生成任务效果一般

### RAFT (Reward-Augmented Fine-Tuning)

**是什么：**
结合奖励信号的参数高效微调方法，类似轻量版RLHF

**怎么做：**
1. 收集人类反馈构建奖励模型
2. 使用奖励信号指导PEFT训练
3. 平衡生成质量和计算效率

**应用场景：**
对话系统、代码生成等需要对齐人类偏好的任务

---

## 🎯 实际应用场景

### RAG + LoRA 组合

**为什么常用：**
- RAG解决知识更新问题
- LoRA解决领域适配问题  
- 成本可控，效果显著

**典型架构：**
```
用户问题 → 检索相关文档 → 构建prompt → LoRA微调的LLM → 生成答案
```

### 选择微调方法的决策树

```
数据量大 + GPU充足 → 全量微调
数据量中等 + GPU有限 → LoRA/QLoRA  
数据量小 + 理解任务 → P-Tuning v2
需要人类对齐 → RAFT
```

---

## 🔥 面试高频问题预测

1. **"解释一下Transformer的Self-Attention机制"**
   - 重点讲Q,K,V的计算过程和物理意义

2. **"LoRA为什么能用这么少参数就有效果？"** 
   - 低秩假设：大模型的权重变化通常在低维空间

3. **"什么情况下用LoRA，什么情况下全量微调？"**
   - 考虑数据量、计算资源、任务复杂度

4. **"RAG和微调有什么区别？"**
   - RAG：外部知识检索，无需训练
   - 微调：内化知识到模型参数

---

## ⏰ 3分钟回答模板

**开头**：简洁定义核心概念（30秒）
**展开**：关键技术细节和计算过程（90秒）  
**对比**：优缺点和适用场景（45秒）
**总结**：实际应用中的选择建议（15秒）

记住：**逻辑清晰 > 细节完整**，面试官更看重你的理解深度和表达能力！