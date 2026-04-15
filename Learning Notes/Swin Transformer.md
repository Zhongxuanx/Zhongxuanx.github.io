# Swin Transformer

**Swin Transformer** 是一种基于 Transformer 的视觉模型，由 Microsoft 研究团队提出，旨在解决传统 Transformer 模型在计算机视觉任务中的高计算复杂度问题。其全称是Shifted Window Transformer，通过引入分层架构和滑动窗口机制，Swin Transformer 在性能和效率之间取得了平衡，广泛应用于图像分类、目标检测、分割等视觉任务，称为新一代的backbone，可直接套用在各项下游任务中。在Swin Transformer中，提供大、中、小不同版本模型，可以进行自由选择合适的使用。

论文原文：[Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2103.14030)

![](https://pic2.zhimg.com/v2-de8f518e1542ff23bfea83ea124e7985_r.jpg)

## 1. **介绍**

Transformer 最初在自然语言处理（NLP）领域大获成功，但直接将 Transformer 应用于计算机视觉任务存在很多挑战。传统Transformer中，拿到了图像数据，将图片进行划分成一个个patch，尽可能patch细一些。但是图像中像素点太多了，如果需要更多的特征，就必须构建很长的序列。而越长的序列算起注意力肯定越慢，自注意力机制的计算复杂度是 $O(n^2)$ ，当处理高分辨率图像时，这种复杂度会快速增长，这就导致了效率问题。

而且图像中许多视觉信息依赖于局部关系，而标准 Transformer 处理的是全局关系，可能无法有效捕获局部特征。Swin Transformer便采用窗口和分层的形式来替代长序列的方法，CNN中经常提到感受野，在Transformer中对应的就是分层。也就是说，我们可以将当前这件事做L次（Lx），每次都会两两进行合并，向量数越来越小（400个token-200个token-100个token），窗口的大小也会增大。分层操作也就是，第一层的时候token很多，第二层合并token，第三层合并token，就像我们的卷积和池化的操作。而在传统的Transformer中，第一层怎么做，第二层第三层也会采用同样的尺寸进行，都是一样的操作。

![](https://pic1.zhimg.com/v2-b6dc8e62452bd33e277da1dc661e988c_1440w.jpg)

2. **Swin Transformer 整体架构**

![](https://picx.zhimg.com/v2-090ccdde2313150d71bbea629c369fef_r.jpg)

## 2.1. Patch Embedding

在 **Swin Transformer** 中，**Patch Embedding** 负责将输入图像分割成多个小块（patches），并将这些小块的像素值嵌入到一个高维空间中，形成适合 Transformer 处理的特征表示。在传统的卷积神经网络（CNN）中，卷积操作可以用来提取局部特征。在 **Swin Transformer** 中，为了将输入图像转化为适合 Transformer 模型处理的 patch 序列，首先对输入图像进行分块。假设输入图像的大小为 `224x224x3`​，其通过一个卷积操作实现。卷积操作可以将每个局部区域的像素值映射为一个更高维的特征向量。假设输入图像大小为 `224x224x3`​，应用一个卷积层，参数为 `Conv2d(3, 96, kernel_size=(4, 4), stride=(4, 4))`​，这表示卷积核的大小是 `4x4`​，步长是 `4`​，输入的通道数是 `3`​（RGB图像），输出的通道数是 `96`​。卷积后，图像的空间维度会变小，输出的特征图的尺寸会变为 `56x56`​（通过计算：`(224 - 4) / 4 + 1 = 56`​）。所以，卷积后的输出大小是 `56x56x96`，这表示每个空间位置（56x56）都有一个96维的特征向量。

在 Swin Transformer 中，通常将图像通过卷积操作分割成不重叠的小块（patches）。每个小块对应一个特征向量。例如，`56x56x96`​ 的输出可以视为有 `3136`​ 个 patch，每个 patch 是一个 96 维的向量。这些特征向量将作为 Transformer 模型的输入序列。根据不同的卷积参数（如 kernel\_size 和 stride），你可以控制生成的 patch 的数量和每个 patch 的维度。例如，如果使用更小的卷积核和步长，可以得到更细粒度的 patch，反之则可以得到较大的 patch。

- ​`kernel_size` 决定了每个 patch 的空间大小。
- ​`stride` 决定了每个 patch 之间的间隔，即步长。

## 2.2. window\_partition

在 **Swin Transformer** 中，图像的特征表示不仅仅是通过 `Patch Embedding`​ 来获得，还通过 **窗口划分（**​**[Window Partition](https://zhida.zhihu.com/search?content_id=251464052&content_type=Article&match_order=1&q=Window+Partition&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzYwNjA2MjAsInEiOiJXaW5kb3cgUGFydGl0aW9uIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6MjUxNDY0MDUyLCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.sFn-BN7gzE_LmS_atLhv6YupjujeHLfRKNi64F83rP4&zhida_source=entity)**​ **）**  来进一步细化和处理，通过窗口内的局部注意力机制来增强计算效率并捕捉局部特征。

假设输入的图像经过卷积处理后得到了大小为 `56x56x96`​ 的特征图，将这个特征图划分为多个小窗口（window），每个窗口包含一部分局部信息，其中窗口大小为`7x7`​，特征图大小为`56x56。`​为了将特征图划分成大小为 `7x7`​ 的窗口，我们首先计算在空间维度（高和宽）上可以分成多少个窗口，水平和垂直方向上，每个 `7x7`​ 窗口可以覆盖 `56 / 7 = 8`​ 个窗口（总共 `8x8 = 64`​ 个窗口），窗口内部的特征图由 `96`​ 个通道组成。因此，在划分后，特征图的维度将变为 `(64, 7, 7, 96)`，其中：

- ​`64`​ 表示窗口的数量（即 `8x8 = 64` 个窗口）。
- ​`7x7` 是每个窗口的空间维度。
- ​`96` 是每个窗口内的特征通道数。

在 Swin Transformer 中，**Token** 通常指的是图像中的局部特征，每个 Token 是图像的一个小区域。在 **Window Partition** 过程中，我们将整个图像的 Token 重新组织成窗口（Window）。之前每个 Token 对应一个图像位置，现在每个 Token 对应一个窗口的内部特征。所以，原来每个 Token（如卷积后的每个空间位置）代表了图像的一部分信息，现在我们通过窗口划分来捕捉更大范围的局部信息。这种划分​**有助于模型专注于图像的局部结构，同时减少计算量**，因为每个窗口只在局部范围内进行注意力计算。

## 2.3. [W-MSA](https://zhida.zhihu.com/search?content_id=251464052&content_type=Article&match_order=1&q=W-MSA&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzYwNjA2MjAsInEiOiJXLU1TQSIsInpoaWRhX3NvdXJjZSI6ImVudGl0eSIsImNvbnRlbnRfaWQiOjI1MTQ2NDA1MiwiY29udGVudF90eXBlIjoiQXJ0aWNsZSIsIm1hdGNoX29yZGVyIjoxLCJ6ZF90b2tlbiI6bnVsbH0.RX1TwxOrICC42RhPBYNbPimpL7b9D5-wC6-XY1iO3bo&zhida_source=entity)（Window multi-head self attention）

在 **Swin Transformer** 中，**W-MSA (Window Multi-Head Self Attention)**  是关键的注意力机制，它通过在每个窗口内部独立地计算自注意力（Self-Attention）来减少计算复杂度，并捕捉局部特征。

通过 **Window Partition** 将特征图划分为 `64`​ 个窗口，每个窗口的尺寸为 `7x7`​，并且每个位置的特征通道数为 `96`​，因此每个窗口的形状为 `(7, 7, 96)，`​这些窗口将作为 **W-MSA** 的输入。在 **Multi-Head Self-Attention** 中，首先需要将输入特征矩阵（窗口内的特征）通过三个不同的矩阵进行线性变换，得到 ​**查询（Q）** ​、**键（K）**  和 ​**值（V）** ​，这三个矩阵用于计算注意力得分。对于每个头（Head），计算过程是独立的。假设有 `3`​ 个头，那么每个头的输入特征维度为 `96 / 3 = 32`​，因为 `96`​ 维的输入被平均分成了 3 个头，每个头负责 32 维的特征。在 **W-MSA** 中，针对每个窗口独立计算自注意力得分，计算方法如下：

- 对每个窗口中的 `49` 个像素点（即每个位置的特征向量）进行查询Q、键K、值V的计算。
- **自注意力得分（Attention Score）**  是通过计算查询与键的点积（或者其他相似度度量）得到的，这可以表示为：

![](https://picx.zhimg.com/v2-fe5c9e2720e10cec2d315ff6c44a1b37_1440w.jpg)

其中， $d_k$ 是每个头的维度（在这里是 32），Q 和 K 的乘积衡量了每个位置之间的相似性。

- ​**Softmax**：通过 Softmax 操作将得分归一化，使其成为概率分布，得到每个位置与其他位置的相关性。
- ​**加权值（Weighted Sum）** ：使用得分对值V进行加权求和，得到每个位置的最终输出表示。

每个头的自注意力计算都会产生一个形状为 `(64, 3, 49, 49)`​ 的结果，其中，`64`​ 表示窗口的数量，`3`​ 表示头的数量，`49`​ 是每个窗口中位置的数量（`7x7`​），`49`​ 代表每个位置对其他位置的注意力得分（自注意力矩阵）。因此，每个头会计算出每个窗口内所有位置之间的自注意力得分，输出的形状为 `(64, 3, 49, 49)`。

## 2.4. window\_reverse

​`Window Reverse`​ 操作的目的是将计算得到的 `(64, 49, 96)`​ 特征图恢复回原始的空间维度 `(56, 56, 96)`​。为此，我们需要将每个窗口的 `49`​ 个位置（`7x7`）重新排列到原始的图像空间中。步骤：

- **Reshape 操作：**  每个窗口的特征图形状是 `(49, 96)`​，我们将其转换成 `(7, 7, 96)`​ 的形状，表示每个窗口中的每个像素点都有一个 `96` 维的特征向量。
- **按窗口拼接：**  将所有 `64`​ 个窗口按照它们在特征图中的位置重新排列成 `56x56`​ 的大特征图。原始的输入特征图大小是 `56x56`​，这意味着 `64`​ 个窗口将按照 `8x8`​ 的网格排列，并恢复到一个 `(56, 56, 96)` 的特征图。

在 **Window Reverse** 操作后，恢复得到的特征图形状是 `(56, 56, 96)`​，这与卷积后的特征图的形状一致。`56x56`​ 是恢复后的空间维度，代表每个像素点在特征图中的位置；`96` 是每个像素点的特征维度，表示每个位置的特征信息。

## 2.5. SW-MASA

### 为什么要滑动窗口（Shifted Window）？

原始的 **Window MSA** 将图像划分为固定的窗口（例如 `7x7`），并在每个窗口内计算自注意力。这样做的一个问题是每个窗口内部的信息相对封闭，没有与相邻窗口之间的信息交流。因此，模型容易局限于各自的小区域，无法充分捕捉不同窗口之间的关联。

通过引入 ​**滑动窗口**​（​**Shifted Window**）机制，窗口在原来位置的基础上向四个方向移动一部分，重叠区域与原窗口有交集。这样，原本相互独立的窗口就可以共享信息，增强了模型的表达能力和全局感知。

### 位移操作（Shift Operation）

位移操作的细节如下：

- 初始的窗口被划分为 `4x4`​ 的块（例如 `7x7` 窗口），每个块进行独立的自注意力计算。
- 在进行位移时，原来 `4x4`​ 的窗口将被平移，变成新的大小为 `9x9` 的窗口，窗口重叠区域包含了不同窗口之间的信息。
- 通过平移，模型能获取到更广泛的信息，使得窗口之间能够通过共享信息来融合彼此的特征，避免局部化。

![](https://pic2.zhimg.com/v2-4c551fe5962be037de7efa2922553779_1440w.jpg)

**Shifted Window MSA** 会导致计算量的增加，特别是在窗口滑动后，窗口数量从 `4x4`​ 变为 `9x9`​，计算量几乎翻倍。为了控制计算量的增长，可以通过 **mask 操作** 来减少不必要的计算。在位移后，窗口之间会重叠。为了避免重复计算，我们可以使用 **mask** 来屏蔽掉不需要计算的部分。在计算注意力时，对于每个位置的 **Q** 和 **K** 的匹配，使用 **softmax** 时，设置不需要计算的位置的值为负无穷，这样对应位置的注意力值将接近零，不会对结果产生影响。

![](https://pic2.zhimg.com/v2-abd42b57db5553021b28a66f5f398911_r.jpg)

![](https://pic3.zhimg.com/v2-48c523b2b6f99333778730acedef9020_1440w.jpg)

在进行 **SW-MSA** 后，输出的特征图的形状仍然是 `56x56x96`​，与输入特征图的大小一致。通过 **shifted window** 和 ​**mask 操作**，模型不仅保留了原始的窗口内的自注意力计算，还增强了窗口之间的信息交换和融合。即使窗口被移动了，经过计算后的特征也需要回到其原本的位置，也就是还原平移，保持图像的完整性。

## 2.6. [PatchMerging](https://zhida.zhihu.com/search?content_id=251464052&content_type=Article&match_order=1&q=PatchMerging&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzYwNjA2MjAsInEiOiJQYXRjaE1lcmdpbmciLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNTE0NjQwNTIsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.tCbTwMi7OAXW8d_1kfPfmEfotU4Ea-98dHaluT3Is74&zhida_source=entity)

**PatchMerging** 是 **Swin Transformer** 中的一种下采样操作，但是不同于池化，这个相当于间接的（对H和W维度进行间隔采样后拼接在一起，得到H/2，W/2，C\*4），目的是将输入特征图的空间维度（即高和宽）逐渐减小，同时增加通道数，从而在保持计算效率的同时获得更高层次的特征表示。它是下采样的过程，但与常规的池化操作不同，**PatchMerging** 通过将相邻的 patch 拼接在一起，并对拼接后的特征进行线性变换，从而实现下采样。具体来说，在 **Swin Transformer** 中，随着网络层数的加深，输入的特征图会逐渐减小其空间尺寸（即 H 和 W 维度），而同时增加其通道数（即 C 维度），以便模型可以捕捉到更为复杂的高层次信息。

![](https://pic3.zhimg.com/v2-f2158e62819217981f347de5f0335714_r.jpg)

假设输入的特征图形状为 `H x W x C`​，`PatchMerging` 通过以下步骤来实现下采样和通道数扩展：

- ​**分割和拼接（Splitting and Concatenation）** ：

  - 输入的特征图会按照一定的步长（通常是 2）进行分割，即对每个 `2x2` 的 patch 进行合并。
  - 这样原本的 `H x W`​ 的空间尺寸会缩小一半，变成 `H/2 x W/2`。
  - 然后，将每个 `2x2`​ 的 patch 内部的特征进行拼接，得到新的特征维度。假设原始通道数为 `C`​，拼接后的通道数为 `4C`。
- ​**卷积操作**：

  - 对拼接后的特征进行 ​**卷积**，以进一步增强特征表达。卷积操作用于转换特征空间，虽然通道数增加了，但通过卷积，特征能够更加丰富。

## 2.7. 分层计算

在 **Swin Transformer** 中，模型的每一层都会进行下采样操作，同时逐步增加通道数。每次 PatchMerging 后的特征图会作为输入进入下一层的 Attention 计算。通过这种方式，**Swin Transformer** 能够逐渐提取到越来越复杂的特征，同时保持计算效率。每一层的 **PatchMerging** 操作实际上是将输入的特征图通过 ​**线性变换**（通常是卷积）合并成更高维度的特征图，从而为后续的注意力计算提供更丰富的表示。

从图中可以得到，通道数在每层中并不是从C变成4C而是2C，这是因为中间又加了一层卷积操作。

![](https://picx.zhimg.com/v2-090ccdde2313150d71bbea629c369fef_r.jpg)

## 3. **实验结果**

在ImageNet22K数据集上，准确率能达到惊人的86.4%。另外在检测，分割等任务上表现也很优异。

![](https://pica.zhimg.com/v2-fddc2dee6831c48fe787f043f6cca900_r.jpg)

---

  
**参考资料：**

[【深度学习】详解 Swin Transformer (SwinT)-CSDN博客](https://link.zhihu.com/?target=https%3A//blog.csdn.net/qq_39478403/article/details/120042232)

[深度学习之Swin Transformer学习篇(详细 - 附代码）_swintransformer训练-CSDN博客](https://link.zhihu.com/?target=https%3A//blog.csdn.net/weixin_44956153/article/details/142698557)

[图解Swin Transformer - 知乎](https://zhuanlan.zhihu.com/p/367111046)

[【论文精读】Swin Transformer - 知乎](https://zhuanlan.zhihu.com/p/681944361)

[ICCV2021最佳论文：Swin Transformer论文解读+源码复现，迪哥带你从零解读霸榜各大CV任务的Swin Transformer模型！_哔哩哔哩_bilibili](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1Nm4y1Y7U4/%3Fspm_id_from%3D333.337.search-card.all.click%26vd_source%3D0dc0c2075537732f2b9a894b24578eed)
