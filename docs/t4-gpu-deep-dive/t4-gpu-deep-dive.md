---
title: "T4 GPU 深度解析：16GB 能跑什么模型，为什么 decode 吃满 320GB/s 带宽"
slug: t4-gpu-deep-dive
date: 2026-10-31
author: AI Infra 学习笔记
description: 用 256-bit×10Gbps 推导 T4 的 320GB/s，算清 16GB 显存能装多大模型，推导 7B INT4 约 91 token/s 的 decode 带宽上限，并对比 L4/A10 判断 2026 年 T4 还值不值得学；并以 AI Infra 视角逐行解读完整规格表，每个参数对应一条决策依据。
tags:
  - GPU
  - 推理优化
  - CUDA
  - 量化
  - AI Infra
lang: zh-CN
kicker: AI Infra 学习笔记
site_name: AI Infra 学习笔记
---

# T4 GPU 深度解析：16GB 能跑什么模型，为什么 decode 吃满 320GB/s 带宽

如果你在 Colab 或 Kaggle 上用过免费的 T4，多半见过这样的场景：加载一个 7B 量化模型，生成速度只有十几 token/s，等待的时间比思考的时间还长。问题通常不在模型，也不在你的代码，而在这张 2018 年发布的卡：它的显存带宽和 70W 功耗早就写好了天花板。

规格表谁都会查，但查完往往更糊涂：8.1 TFLOPS、65 TFLOPS、130 TOPS，哪个才是真实算力？datasheet 写带宽 300GB/s，产品页写 320+GB/s，听谁的？16GB 显存到底装得下多大的模型？搜索结果里几乎全是结论式回答，很少有人给出推导。

这篇是我研究 T4 规格后的完整学习笔记。读完它，你能从一张规格表自己算出三个预算：显存带宽、模型容量、decode（解码阶段）吞吐。推导过程全部保留，每一步都能拿计算器复算。

> **TL;DR**: T4 是一张"算力平平、带宽和 INT8 是亮点"的入门数据中心推理卡。FP32 只有 8.1 TFLOPS，但 INT8 达 130 TOPS（[NVIDIA 官方产品页](https://www.nvidia.com/en-us/data-center/tesla-t4/)）；16GB 显存装得下 7B INT4（约 3.5GB 权重），decode（解码阶段）受 320GB/s 带宽限制，理论吞吐上限约 91 token/s。本文给出全部推导，并解释 NVIDIA 官方 [datasheet](https://images.nvidia.com/aem-dam/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf)（300GB/s）与产品页（320+GB/s）的口径矛盾。文中附 AI Infra 视角的逐行规格表解读：每个参数如何翻译成一条 infra 决策依据。

## T4 是一张什么卡：定位与关键规格

一句话定位：T4 是 2018 年随 Turing 架构发布的入门级数据中心推理卡，设计目标不是单卡极致速度，而是在 70W 功耗内高效服务更多推理任务。

| 项目 | 数值 | 口径说明 |
|---|---|---|
| 架构 | Turing，TU104 核心，40 个 SM（流多处理器） | 白皮书 |
| CUDA Core | 2560 | 产品页 |
| Tensor Core | 320 | 产品页 |
| 显存 | 16GB GDDR6，256-bit 总线 | 产品页 |
| 显存带宽 | 320+ GB/s（产品页）/ 300 GB/s（datasheet） | 下文带宽一节裁决 |
| FP32 | 8.1 TFLOPS | 产品页 |
| FP16 Tensor Core | 65 TFLOPS | 产品页 |
| INT8 | 130 TOPS | 产品页 |
| INT4 | 260 TOPS | 产品页 |
| TDP | 70W | 产品页 |

上表数值来自 [NVIDIA T4 官方产品页](https://www.nvidia.com/en-us/data-center/tesla-t4/) 与 [Turing 架构白皮书](https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf)。有两点值得先记下。其一，70W 低于 PCIe x16 插槽 75W 的供电上限，所以 T4 不需要外接电源线（这是我的推断，官方未直接写明，但功率数字支持这个结论）。其二，带宽一栏官方自己就有两个说法，下文带宽一节会给出来龙去脉。

## 为什么不能只看 TFLOPS：三个数字，三种电路

8.1 TFLOPS（FP32）、65 TFLOPS（FP16 Tensor Core）、130 TOPS（INT8）不是同一份算力的三种写法，而是三种不同计算电路各自的峰值。拿哪个数字做预算，取决于你的算子走哪条电路。

8.1 TFLOPS 是 2560 个 CUDA Core 的标量浮点峰值，走这条路的是普通算术和 elementwise 运算。如果 FP16 运算不走 Tensor Core，峰值只有 16.2 TFLOPS（白皮书口径），只是 FP32 的两倍，不是八倍。65 TFLOPS 才是 Tensor Core 矩阵路径的 FP16 峰值：

```
65 / 8.1 ≈ 8.02 倍
```

白话解释：同样标着 FP16，走没走 Tensor Core 差了 4 倍；和 FP32 比，差了 8 倍。再看整数路径：INT8 是 130 TOPS，INT4 是 260 TOPS，单位已经是 TOPS（每秒万亿次整数操作）而不是 TFLOPS（每秒万亿次浮点操作），所以柱状图上的高度不能当成同等工作量来比较。

还要泼一盆冷水：峰值不等于实际推理性能，算子能否用上 Tensor Core、batch 大小、访存效率都会让实际数字远低于标称。这套"口径决定认知"的逻辑后面还会出现两次：L4 的稀疏对密集，Orin 的 TOPS 对 TFLOPS（见后文三卡对比一节）。

<figure class="blog-chart">
<svg viewBox="0 0 560 380" style="max-width: 100%; height: auto; font-family: 'Inter', system-ui, sans-serif;" role="img" aria-label="T4 五种性能口径的峰值对比。T4 在五种精度口径下的峰值：FP32 标量 8.1 TFLOPS、FP16 非张量路径 16.2 TFLOPS、FP16 Tensor Core 65 TFLOPS、INT8 130 TOPS、INT4 260 TOPS。前三个为浮点口径，后两个为整数口径，单位不同不能直接等同工作量。数据来自 NVIDIA T4 官方产品页与 Turing 架构白皮书。">
  <text x="280.0" y="29" text-anchor="middle" font-size="18" font-weight="800" fill="currentColor">T4 五种性能口径的峰值对比</text>
  <text x="280.0" y="48.0" text-anchor="middle" font-size="12" fill="var(--chart-muted, currentColor)"><tspan x="280.0" dy="0">TFLOPS 与 TOPS 单位不同，柱高仅示意相对倍数（正文解释口径差异）</tspan></text>
  <line x1="145" y1="298.0" x2="510" y2="298.0" stroke="currentColor" opacity="0.08" />
<line x1="145" y1="243.0" x2="510" y2="243.0" stroke="currentColor" opacity="0.08" />
<line x1="145" y1="188.0" x2="510" y2="188.0" stroke="currentColor" opacity="0.08" />
<line x1="145" y1="133.0" x2="510" y2="133.0" stroke="currentColor" opacity="0.08" />
<line x1="145" y1="78.0" x2="510" y2="78.0" stroke="currentColor" opacity="0.08" />
  <text x="136.0" y="102.4" text-anchor="end" font-size="11" fill="currentColor" opacity="0.8"><tspan x="136.0" dy="0">INT4｜TOPS</tspan></text>
  <rect x="145" y="78.0" width="365.0" height="37.6" rx="4" fill="#f97316" />
  <text x="518.0" y="102.4" font-size="11" fill="currentColor">260</text>
  <text x="136.0" y="148.0" text-anchor="end" font-size="11" fill="currentColor" opacity="0.8"><tspan x="136.0" dy="0">INT8｜TOPS</tspan></text>
  <rect x="145" y="123.6" width="182.5" height="37.6" rx="4" fill="#38bdf8" />
  <text x="335.5" y="148.0" font-size="11" fill="currentColor">130</text>
  <text x="136.0" y="193.6" text-anchor="end" font-size="11" fill="currentColor" opacity="0.8"><tspan x="136.0" dy="0">FP16 Tensor</tspan><tspan x="136.0" dy="12">Core｜TFLOPS</tspan></text>
  <rect x="145" y="169.2" width="91.2" height="37.6" rx="4" fill="#a78bfa" />
  <text x="244.2" y="193.6" font-size="11" fill="currentColor">65</text>
  <text x="136.0" y="239.2" text-anchor="end" font-size="11" fill="currentColor" opacity="0.8"><tspan x="136.0" dy="0">FP16 非张量｜TFLOPS</tspan></text>
  <rect x="145" y="214.8" width="22.7" height="37.6" rx="4" fill="#22c55e" />
  <text x="175.7" y="239.2" font-size="11" fill="currentColor">16.2</text>
  <text x="136.0" y="284.8" text-anchor="end" font-size="11" fill="currentColor" opacity="0.8"><tspan x="136.0" dy="0">FP32｜TFLOPS</tspan></text>
  <rect x="145" y="260.4" width="11.4" height="37.6" rx="4" fill="#facc15" />
  <text x="164.4" y="284.8" font-size="11" fill="currentColor">8.1</text>
  <text x="280.0" y="366" text-anchor="middle" font-size="10" fill="var(--chart-muted, currentColor)">Source: NVIDIA T4 官方产品页 / Turing 架构白皮书 (2018)</text>
</svg>
<figcaption>Source: <a href="https://www.nvidia.com/en-us/data-center/tesla-t4/">NVIDIA T4 官方产品页 / Turing 架构白皮书</a>, 2018.</figcaption>
</figure>

## CUDA Core 与 Tensor Core：SM 里的分工

CUDA Core 是通用标量单元，什么都能算但单项不快；Tensor Core 是专用的混合精度矩阵乘累加单元，Turing 这一代支持 FP16 输入与 INT8/INT4 输入，累加走 FP32/INT32（白皮书口径）。LLM 推理的吞吐主要看后者。

T4 一共 40 个 SM（流多处理器），每个 SM 里 64 个 CUDA Core 和 8 个 Tensor Core：

```
CUDA Core 总数   = 40 SM × 64 = 2560
Tensor Core 总数 = 40 SM × 8  = 320
```

白话解释：把 SM 想成一个车间，CUDA Core 是通用机床，什么零件都能加工但速度有限；Tensor Core 是专门做矩阵乘的自动生产线，只干一件事，但快得多。

为什么 GEMM（通用矩阵乘）是 LLM 的心脏？因为 Transformer 里的 Linear、MLP、Attention 投影底层全是矩阵乘法。CUDA Core 和 Tensor Core 也不是替代关系，而是融合量化 GEMM 算子里的流水线分工：CUDA Core 做地址计算和索引，Load 指令把数据从显存搬进寄存器或共享内存，Tensor Core 完成矩阵乘，CUDA Core 再做 bias、激活、scale 后处理。理解这个分工，是研究 CUTLASS 和 TensorRT plugin 的硬件基础。

顺便纠正三个我一开始记错的细节。第一，65 TFLOPS 是 FP16 Tensor Core 的峰值口径，不是任何 FP16 算子都能摸到，不走 Tensor 路径就只有 16.2。第二，Turing（计算能力 7.5）的 Max blocks per SM 通常是 16，不是我最初抄来的 32（官方依据见下文规格表逐行解读一节）。第三，单 block 默认共享内存上限 48KB，但 Turing 每个 SM 的共享内存可配置到 64KB，用到更高需要显式 opt-in。这些限制直接决定 occupancy：每个 warp（线程束）32 个线程，每 SM 最多驻留 1024 个线程，寄存器共 65536 个 32-bit 槽位。假设一个 block 256 线程、每线程 64 个寄存器：

```
每 block 寄存器需求 = 256 线程 × 64 = 16384
可驻留 block 数     = 65536 / 16384 = 4
```

白话解释：寄存器用得越凶，能同时驻留的 block 越少，GPU 用来隐藏访存延迟的"兵力"就不足。

<figure class="blog-chart">
<svg viewBox="0 0 560 320" style="max-width: 100%; height: auto; font-family: 'Inter', system-ui, sans-serif;" role="img" aria-label="一个 SM 内部的简化结构（T4 共 40 个）。SM（流多处理器）内部包含 64 个 CUDA Core（通用标量单元，图中蓝色方格阵列）、8 个 Tensor Core（橙色块，混合精度矩阵乘累加）、65536 个 32-bit 寄存器与 64KB 共享内存（紫色条）。数据路径：显存经 Load 到共享内存，再到寄存器，由 CUDA Core / Tensor Core 计算后输出。">
  <text x="95" y="20" font-size="12" font-weight="700" fill="currentColor">SM（流多处理器）× 40</text>
  <rect x="90" y="28" width="380" height="140" rx="10" fill="none" stroke="currentColor" stroke-width="1.5" opacity="0.5"/>
  <g opacity="0.75">
    <!-- 64 CUDA cores: 8x8 grid -->
    <g fill="#38bdf8">
      <rect x="112" y="44" width="9" height="9"/><rect x="123" y="44" width="9" height="9"/><rect x="134" y="44" width="9" height="9"/><rect x="145" y="44" width="9" height="9"/><rect x="156" y="44" width="9" height="9"/><rect x="167" y="44" width="9" height="9"/><rect x="178" y="44" width="9" height="9"/><rect x="189" y="44" width="9" height="9"/>
      <rect x="112" y="55" width="9" height="9"/><rect x="123" y="55" width="9" height="9"/><rect x="134" y="55" width="9" height="9"/><rect x="145" y="55" width="9" height="9"/><rect x="156" y="55" width="9" height="9"/><rect x="167" y="55" width="9" height="9"/><rect x="178" y="55" width="9" height="9"/><rect x="189" y="55" width="9" height="9"/>
      <rect x="112" y="66" width="9" height="9"/><rect x="123" y="66" width="9" height="9"/><rect x="134" y="66" width="9" height="9"/><rect x="145" y="66" width="9" height="9"/><rect x="156" y="66" width="9" height="9"/><rect x="167" y="66" width="9" height="9"/><rect x="178" y="66" width="9" height="9"/><rect x="189" y="66" width="9" height="9"/>
      <rect x="112" y="77" width="9" height="9"/><rect x="123" y="77" width="9" height="9"/><rect x="134" y="77" width="9" height="9"/><rect x="145" y="77" width="9" height="9"/><rect x="156" y="77" width="9" height="9"/><rect x="167" y="77" width="9" height="9"/><rect x="178" y="77" width="9" height="9"/><rect x="189" y="77" width="9" height="9"/>
      <rect x="112" y="88" width="9" height="9"/><rect x="123" y="88" width="9" height="9"/><rect x="134" y="88" width="9" height="9"/><rect x="145" y="88" width="9" height="9"/><rect x="156" y="88" width="9" height="9"/><rect x="167" y="88" width="9" height="9"/><rect x="178" y="88" width="9" height="9"/><rect x="189" y="88" width="9" height="9"/>
      <rect x="112" y="99" width="9" height="9"/><rect x="123" y="99" width="9" height="9"/><rect x="134" y="99" width="9" height="9"/><rect x="145" y="99" width="9" height="9"/><rect x="156" y="99" width="9" height="9"/><rect x="167" y="99" width="9" height="9"/><rect x="178" y="99" width="9" height="9"/><rect x="189" y="99" width="9" height="9"/>
      <rect x="112" y="110" width="9" height="9"/><rect x="123" y="110" width="9" height="9"/><rect x="134" y="110" width="9" height="9"/><rect x="145" y="110" width="9" height="9"/><rect x="156" y="110" width="9" height="9"/><rect x="167" y="110" width="9" height="9"/><rect x="178" y="110" width="9" height="9"/><rect x="189" y="110" width="9" height="9"/>
      <rect x="112" y="121" width="9" height="9"/><rect x="123" y="121" width="9" height="9"/><rect x="134" y="121" width="9" height="9"/><rect x="145" y="121" width="9" height="9"/><rect x="156" y="121" width="9" height="9"/><rect x="167" y="121" width="9" height="9"/><rect x="178" y="121" width="9" height="9"/><rect x="189" y="121" width="9" height="9"/>
    </g>
  </g>
  <text x="155" y="152" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.8">64 × CUDA Core</text>
  <g fill="#f97316" opacity="0.85">
    <rect x="238" y="48" width="30" height="14" rx="2"/><rect x="274" y="48" width="30" height="14" rx="2"/>
    <rect x="238" y="68" width="30" height="14" rx="2"/><rect x="274" y="68" width="30" height="14" rx="2"/>
    <rect x="238" y="88" width="30" height="14" rx="2"/><rect x="274" y="88" width="30" height="14" rx="2"/>
    <rect x="238" y="108" width="30" height="14" rx="2"/><rect x="274" y="108" width="30" height="14" rx="2"/>
  </g>
  <text x="271" y="136" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.8">8 × Tensor Core</text>
  <rect x="348" y="44" width="26" height="88" rx="3" fill="#a78bfa" opacity="0.6"/>
  <text x="361" y="146" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.8">65536 寄存器</text>
  <rect x="404" y="44" width="26" height="64" rx="3" fill="#a78bfa" opacity="0.4"/>
  <text x="417" y="122" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.8">共享内存</text>
  <text x="417" y="136" text-anchor="middle" font-size="10" fill="var(--chart-muted, currentColor)">64KB</text>
  <g font-size="11.5" fill="currentColor">
    <rect x="22" y="196" width="84" height="34" rx="6" fill="none" stroke="currentColor" opacity="0.4"/>
    <text x="64" y="217" text-anchor="middle">显存</text>
    <rect x="154" y="196" width="84" height="34" rx="6" fill="none" stroke="currentColor" opacity="0.4"/>
    <text x="196" y="217" text-anchor="middle">共享内存</text>
    <rect x="286" y="196" width="84" height="34" rx="6" fill="none" stroke="currentColor" opacity="0.4"/>
    <text x="328" y="217" text-anchor="middle">寄存器</text>
    <rect x="418" y="196" width="84" height="34" rx="6" fill="none" stroke="currentColor" opacity="0.4"/>
    <text x="460" y="217" text-anchor="middle">计算单元</text>
    <rect x="454" y="258" width="84" height="34" rx="6" fill="none" stroke="currentColor" opacity="0.4"/>
    <text x="496" y="279" text-anchor="middle">输出</text>
    <g stroke="currentColor" opacity="0.55" stroke-width="1.5">
      <line x1="106" y1="213" x2="128" y2="213"/>
      <line x1="238" y1="213" x2="260" y2="213"/>
      <line x1="370" y1="213" x2="392" y2="213"/>
      <line x1="502" y1="230" x2="502" y2="256"/>
    </g>
    <g fill="currentColor" opacity="0.55">
      <polygon points="128,213 121,209.5 121,216.5"/>
      <polygon points="260,213 253,209.5 253,216.5"/>
      <polygon points="392,213 385,209.5 385,216.5"/>
      <polygon points="502,256 498.5,249 505.5,249"/>
    </g>
    <text x="280" y="310" text-anchor="middle" font-size="10" fill="var(--chart-muted, currentColor)">数据路径：显存 → 共享内存 → 寄存器 → CUDA Core / Tensor Core → 输出（示意）</text>
  </g>
</svg>
<figcaption>Turing SM 结构简化示意。来源：<a href="https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf">NVIDIA Turing 架构白皮书</a>，2018（本文自绘简化版）。</figcaption>
</figure>

## 320GB/s 是怎么算出来的：兼解官方口径矛盾

显存带宽可以自己算：总线宽度除以 8 变成字节，再乘以有效数据速率。T4 是 256-bit 总线配 10Gbps 速率，乘出来正好 320GB/s；datasheet 的 300 是保守写法，白皮书按 10Gbps 推导就是 320。

```
BW = (256 bit / 8) × 10 Gbps
   = 32 Byte × 10 G 次/秒
   = 320 GB/s
```

白话解释（本文推导）：显存总线有 256 根数据线，一次传 256 bit，即 32 字节；每秒传输 10G 次，32 Byte × 10G/s，正好等于 320 GB/s。

这里最容易踩坑的是规格表里的"显存时钟 5001MHz"。它不是"每秒 5001M 次传输"。GDDR6 是双倍数据率内存，时钟的上升沿和下降沿各传一次，有效速率约等于物理时钟的两倍：

```
有效数据速率 ≈ 5001 MHz × 2 ≈ 10 Gbps
```

白话解释：把 5001MHz 直接当传输速率，会算出 160GB/s 的腰斩答案；先翻倍，再进公式。

<!-- ORIGINAL DATA: 320GB/s 带宽推导为本文原创整理，推导口径基于 Turing 白皮书表 5（256-bit @ 10Gbps）；2026-10-31 中文 SERP 调研未见公开的逐步推导与口径裁决 -->

接着说官方口径矛盾。NVIDIA 的 [T4 datasheet](https://images.nvidia.com/aem-dam/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf)（2018 年 12 月）写 300GB/s，产品页写 320+GB/s，而[白皮书](https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf)按 256-bit @ 10Gbps 给出 320GB/s。我的处理方式：引用时写 320GB/s 并注明"白皮书与产品页口径"，看到 300 不要慌，那是 datasheet 的保守写法，两个数字描述的是同一套硬件。

<figure class="blog-chart">
<svg viewBox="0 0 560 330" style="max-width: 100%; height: auto; font-family: 'Inter', system-ui, sans-serif;" role="img" aria-label="320 GB/s 是怎么算出来的。两条输入汇合成最终带宽：其一，总线宽度 256 bit 除以 8 等于每次传输 32 Byte；其二，显存时钟 5001 MHz 乘以 2（GDDR6 双倍数据率）约等于有效速率 10 Gbps。两者相乘，32 Byte 每秒 10G 次，得到 320 GB/s。角标注明：NVIDIA T4 datasheet 写 300 GB/s（保守口径），产品页写 320+ GB/s，白皮书按 256-bit @ 10Gbps 推导为 320 GB/s，同一套硬件。">
  <text x="280" y="24" text-anchor="middle" font-size="15" font-weight="800" fill="currentColor">320 GB/s 是怎么算出来的</text>
  <rect x="30" y="48" width="180" height="80" rx="8" fill="none" stroke="#38bdf8" stroke-width="1.5"/>
  <text x="120" y="74" text-anchor="middle" font-size="12" fill="currentColor">总线宽度 256 bit</text>
  <text x="120" y="92" text-anchor="middle" font-size="11" fill="var(--chart-muted, currentColor)">÷ 8</text>
  <text x="120" y="114" text-anchor="middle" font-size="13" font-weight="700" fill="currentColor">= 32 Byte / 次</text>
  <rect x="30" y="164" width="180" height="80" rx="8" fill="none" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="120" y="190" text-anchor="middle" font-size="12" fill="currentColor">显存时钟 5001 MHz</text>
  <text x="120" y="208" text-anchor="middle" font-size="11" fill="var(--chart-muted, currentColor)">× 2（GDDR6 双倍数据率）</text>
  <text x="120" y="230" text-anchor="middle" font-size="13" font-weight="700" fill="currentColor">≈ 10 Gbps</text>
  <g stroke="currentColor" opacity="0.45" stroke-width="1.5">
    <line x1="210" y1="88" x2="316" y2="136"/>
    <line x1="210" y1="204" x2="316" y2="152"/>
  </g>
  <g fill="currentColor" opacity="0.45">
    <polygon points="320,138 311,134 315,128"/>
    <polygon points="320,150 315,160 311,154"/>
  </g>
  <rect x="322" y="112" width="208" height="68" rx="10" fill="none" stroke="#f97316" stroke-width="2"/>
  <text x="426" y="140" text-anchor="middle" font-size="13" fill="currentColor">32 Byte × 10 G / s</text>
  <text x="426" y="168" text-anchor="middle" font-size="20" font-weight="800" fill="#f97316">= 320 GB/s</text>
  <rect x="30" y="262" width="500" height="52" rx="8" fill="none" stroke="currentColor" stroke-dasharray="4 3" opacity="0.3"/>
  <text x="280" y="283" text-anchor="middle" font-size="11" fill="currentColor">datasheet：300 GB/s（保守口径）｜ 产品页：320+ GB/s</text>
  <text x="280" y="302" text-anchor="middle" font-size="11" fill="var(--chart-muted, currentColor)">白皮书按 256-bit @ 10Gbps 推导 = 320 GB/s，同一套硬件（本文推导可视化）</text>
</svg>
<figcaption>推导口径：<a href="https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf">NVIDIA Turing 架构白皮书</a>（256-bit @ 10Gbps），本文自绘。</figcaption>
</figure>

## 16GB 能装多大模型：显存算术

FP16 最多装 7B 且几乎没有余量，INT8 7B 有运行空间，INT4 7B 是舒适区，14B 只有 INT4 能装下权重且要精打细算。方法只有一句话：参数量乘以每个参数的字节数。

```
权重显存 ≈ 参数量 × 每参数字节数

7B × 2 Byte   (FP16) = 14 GB
7B × 1 Byte   (INT8) = 7 GB
7B × 0.5 Byte (INT4) = 3.5 GB
```

白话解释：一个参数存成什么精度，就占多少字节；模型多大、用什么精度存，显存账单当场就能算出来。扩展到常见规模（单位 GB，仅权重理论值，本文推导）：

| 模型规模 | FP32 | FP16 | INT8 | INT4 |
|---|---|---|---|---|
| 1B | 4 | 2 | 1 | 0.5 |
| 3B | 12 | 6 | 3 | 1.5 |
| 7B | 28 | 14 | 7 | 3.5 |
| 8B | 32 | 16 | 8 | 4 |
| 14B | 56 | 28 | 14 | 7 |

<!-- ORIGINAL DATA: 16GB 容量对照表为本文推导（十进制参数量 × 精度字节数），方法透明，读者可自行复算；实际占用恒高于理论值 -->

但这张表只是"理论列"，现实列至少还要加三笔账。第一，INT4 不是干净的 0.5 Byte 每参数，还要存 scale、zero point 和对齐元数据，实际权重略高于表值。第二，KV 缓存是第二个大头，随上下文长度和并发请求数线性增长，长对话场景下可能反超权重。第三，CUDA context、框架运行时和激活值还要再占一块固定开销。所以"T4 能跑的最强模型"没有单一答案：单看权重，14B INT4 是上限；算上 KV 缓存和运行时，日常舒适区是 7B INT4。

<figure class="blog-chart">
<svg viewBox="0 0 560 380" style="max-width: 100%; height: auto; font-family: 'Inter', system-ui, sans-serif;" role="img" aria-label="不同规模模型的权重显存 vs T4 的 16GB 上限。分组条形图，横轴为模型规模 1B、3B、7B、14B，每组三根柱子分别为 FP16、INT8、INT4 的权重显存理论值：1B 为 2、1、0.5 GB；3B 为 6、3、1.5 GB；7B 为 14、7、3.5 GB；14B 为 28、14、7 GB。图中虚线为 T4 的 16GB 显存上限：7B FP16 贴线几乎无余量，14B 仅 INT4 装得下权重。数据为本文推导（参数量 × 精度字节数），未含 KV 缓存与运行时开销。">
  <text x="280" y="29" text-anchor="middle" font-size="16" font-weight="800" fill="currentColor">不同规模模型的权重显存 vs T4 的 16GB 上限</text>
  <text x="280" y="46" text-anchor="middle" font-size="11" fill="var(--chart-muted, currentColor)">仅权重理论值（GB），未含 KV 缓存、运行时与激活</text>
  <g font-size="11" fill="currentColor" opacity="0.8">
    <rect x="330" y="56" width="10" height="10" fill="#f97316"/><text x="344" y="65">FP16</text>
    <rect x="390" y="56" width="10" height="10" fill="#38bdf8"/><text x="404" y="65">INT8</text>
    <rect x="446" y="56" width="10" height="10" fill="#a78bfa"/><text x="460" y="65">INT4</text>
  </g>
  <g stroke="currentColor" opacity="0.08">
    <line x1="60" y1="70" x2="530" y2="70"/><line x1="60" y1="110" x2="530" y2="110"/>
    <line x1="60" y1="150" x2="530" y2="150"/><line x1="60" y1="190" x2="530" y2="190"/>
    <line x1="60" y1="230" x2="530" y2="230"/><line x1="60" y1="270" x2="530" y2="270"/>
  </g>
  <g font-size="10" fill="currentColor" opacity="0.8" text-anchor="end">
    <text x="54" y="314">0</text><text x="54" y="274">5</text><text x="54" y="234">10</text>
    <text x="54" y="194">15</text><text x="54" y="154">20</text><text x="54" y="114">25</text><text x="54" y="74">30</text>
  </g>
  <line x1="60" y1="310" x2="530" y2="310" stroke="currentColor" opacity="0.3"/>
  <line x1="60" y1="182" x2="530" y2="182" stroke="currentColor" stroke-width="1.5" stroke-dasharray="6 4" opacity="0.45"/>
  <text x="526" y="176" text-anchor="end" font-size="10.5" fill="currentColor" opacity="0.8">T4 显存上限 16GB</text>
  <g>
    <rect x="81.75" y="294" width="22" height="16" rx="2" fill="#f97316"/>
    <rect x="107.75" y="302" width="22" height="8" rx="2" fill="#38bdf8"/>
    <rect x="133.75" y="306" width="22" height="4" rx="2" fill="#a78bfa"/>
    <rect x="199.25" y="262" width="22" height="48" rx="2" fill="#f97316"/>
    <rect x="225.25" y="286" width="22" height="24" rx="2" fill="#38bdf8"/>
    <rect x="251.25" y="298" width="22" height="12" rx="2" fill="#a78bfa"/>
    <rect x="316.75" y="198" width="22" height="112" rx="2" fill="#f97316"/>
    <rect x="342.75" y="254" width="22" height="56" rx="2" fill="#38bdf8"/>
    <rect x="368.75" y="282" width="22" height="28" rx="2" fill="#a78bfa"/>
    <rect x="434.25" y="86" width="22" height="224" rx="2" fill="#f97316"/>
    <rect x="460.25" y="198" width="22" height="112" rx="2" fill="#38bdf8"/>
    <rect x="486.25" y="254" width="22" height="56" rx="2" fill="#a78bfa"/>
  </g>
  <g font-size="9.5" fill="currentColor" text-anchor="middle">
    <text x="92.75" y="289">2</text><text x="118.75" y="297">1</text><text x="144.75" y="301">0.5</text>
    <text x="210.25" y="257">6</text><text x="236.25" y="281">3</text><text x="262.25" y="293">1.5</text>
    <text x="327.75" y="193">14</text><text x="353.75" y="249">7</text><text x="379.75" y="277">3.5</text>
    <text x="445.25" y="81">28</text><text x="471.25" y="193">14</text><text x="497.25" y="249">7</text>
  </g>
  <g font-size="12" fill="currentColor" opacity="0.8" text-anchor="middle">
    <text x="118.75" y="330">1B</text><text x="236.25" y="330">3B</text>
    <text x="353.75" y="330">7B</text><text x="471.25" y="330">14B</text>
  </g>
  <text x="60" y="352" font-size="10.5" fill="var(--chart-muted, currentColor)">14B FP16（28GB）与 FP32 各柱超出 16GB 上限，装不进 T4。</text>
  <text x="280" y="372" text-anchor="middle" font-size="10" fill="var(--chart-muted, currentColor)">Source: 本文推导（参数量 × 精度字节数） (2026)</text>
</svg>
<figcaption>Source: 本文推导（参数量 × 精度字节数，方法见上文，可复算），2026-10-31。</figcaption>
</figure>

## decode 为什么吃带宽：91 token/s 是怎么推出来的

自回归生成的 decode（解码阶段）每产出一个 token 都要把全部权重读一遍，所以单请求吞吐的上限由显存带宽决定：7B INT4 权重 3.5GB，除以 320GB/s，每个 token 约 10.94ms，倒过来就是约 91 token/s。prefill（预填充阶段）恰好相反，是算力瓶颈。

```
W = 7B 参数 × 0.5 Byte (INT4) = 3.5 GB

T_weight = 3.5 GB ÷ 320 GB/s = 0.01094 s ≈ 10.94 ms / token

TPS_上限 = 1 ÷ 0.01094 ≈ 91 token/s
```

白话解释（本文推导，理想化估算）：decode 就像开卷考试，每答一道题都要把整本教材（权重）从头翻一遍，翻书的速度（显存带宽）决定了每分钟能答几题，跟写字多快（算力）关系不大。

<!-- ORIGINAL DATA: 91 token/s 为本文推导的理想上限（3.5GB ÷ 320GB/s），方法透明可复算，非实测数据；实际吞吐恒低于此值 -->

把算力和带宽画进同一张图，就是 roofline 模型。横轴是计算强度 AI，意思是每读 1 字节显存要配多少次计算：

```
AI = FLOPs / Bytes

AI_FP32        = 8.1e12 ÷ 320e9 ≈ 25.3 FLOP/Byte
AI_FP16_Tensor = 65e12  ÷ 320e9 ≈ 203.1 FLOP/Byte
```

白话解释：一个算子的计算强度低于拐点，就落在带宽墙下，再高的峰值算力也是摆设；高于拐点，才轮到算力说话。decode 的 GEMV 计算强度远低于 25.3，是带宽受限；prefill 大批量并行、矩阵规整，容易吃满 Tensor Core，是算力受限。一张卡、两个瓶颈，这就是推理引擎要区别对待两个阶段的根本原因。

91 token/s 是理想上限，现实至少有八项折扣：反量化开销、算子启动开销、KV 缓存访问、非连续访存、GEMV/GEMM 利用率、batch size 影响、算子调度间隙、有效带宽低于标称值。层层叠下来，实际数字会明显低于 91。

那为什么 Colab/Kaggle 的免费 T4 连这个数都跑不到？我的理解是三重额外折扣：70W 功耗墙让 GPU 无法长时间顶在标称频率；T4 是被动散热，依赖服务器风道，云平台条件未必理想；免费平台普遍还会限频控制整机成本。推导给出理论天花板，平台条件决定你在天花板下面的哪一层。

<!-- UNIQUE INSIGHT: "免费 T4 为什么慢"的系统解释（70W 功耗墙 + 被动散热 + 云平台限频），2026-10-31 调研确认大量用户困惑但缺少系统回答；此处为基于规格的合理解释，非实测归因 -->

<figure class="blog-chart">
<svg viewBox="0 0 560 380" style="max-width: 100%; height: auto; font-family: 'Inter', system-ui, sans-serif;" role="img" aria-label="T4 Roofline：decode 与 prefill 落在两堵不同的墙下。T4 的 roofline 示意图，双对数坐标。横轴为计算强度 AI（FLOP/Byte），纵轴为可达吞吐。共享带宽斜线为 320 GB/s：斜线段上吞吐随计算强度线性上升，直到撞上算力屋顶。FP32 屋顶为 8.1 TFLOPS，拐点在 AI 约 25.3；INT8 屋顶为 130 TOPS（密集），拐点在 AI 约 406。decode 的 GEMV 计算强度低，落在带宽墙下（绿色圆点）；prefill 大批量矩阵乘计算强度高，落在 INT8 算力墙下（紫色菱形）。来源：NVIDIA 产品页峰值与本文 320GB/s 推导。">
  <text x="280" y="29" text-anchor="middle" font-size="16" font-weight="800" fill="currentColor">T4 Roofline：decode 与 prefill 落在两堵不同的墙下</text>
  <text x="280" y="46" text-anchor="middle" font-size="11" fill="var(--chart-muted, currentColor)">双对数坐标；两屋顶单位不同（TFLOPS vs TOPS），仅示意位置关系</text>
  <g stroke="currentColor" opacity="0.08">
    <line x1="70" y1="60" x2="70" y2="320"/><line x1="220" y1="60" x2="220" y2="320"/>
    <line x1="370" y1="60" x2="370" y2="320"/><line x1="520" y1="60" x2="520" y2="320"/>
    <line x1="70" y1="60" x2="520" y2="60"/><line x1="70" y1="125" x2="520" y2="125"/>
    <line x1="70" y1="190" x2="520" y2="190"/><line x1="70" y1="255" x2="520" y2="255"/>
  </g>
  <g font-size="10" fill="currentColor" opacity="0.8" text-anchor="middle">
    <text x="70" y="336">1</text><text x="220" y="336">10</text><text x="370" y="336">100</text><text x="520" y="336">1000</text>
  </g>
  <g font-size="10" fill="currentColor" opacity="0.8" text-anchor="end">
    <text x="62" y="324">0.1</text><text x="62" y="259">1</text><text x="62" y="194">10</text>
    <text x="62" y="129">100</text><text x="62" y="64">1000</text>
  </g>
  <line x1="70" y1="320" x2="520" y2="320" stroke="currentColor" opacity="0.3"/>
  <line x1="70" y1="320" x2="70" y2="60" stroke="currentColor" opacity="0.3"/>
  <text x="295" y="356" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.8">计算强度 AI（FLOP/Byte，对数刻度）</text>
  <text x="20" y="190" font-size="11" fill="currentColor" opacity="0.8" transform="rotate(-90 20 190)" text-anchor="middle">可达吞吐（对数刻度）</text>
  <line x1="70" y1="287" x2="461" y2="118" stroke="currentColor" stroke-width="2" opacity="0.5"/>
  <polyline points="280.5,196 520,196" fill="none" stroke="#38bdf8" stroke-width="2.5"/>
  <polyline points="461,117.6 520,117.6" fill="none" stroke="#f97316" stroke-width="2.5"/>
  <text x="410" y="212" text-anchor="middle" font-size="11" font-weight="700" fill="#38bdf8">FP32 屋顶 8.1 TFLOPS</text>
  <text x="490" y="140" text-anchor="middle" font-size="11" font-weight="700" fill="#f97316">INT8 屋顶 130 TOPS</text>
  <text x="180" y="235" font-size="10.5" fill="var(--chart-muted, currentColor)" transform="rotate(-23.4 180 235)">共享带宽斜线 320 GB/s</text>
  <circle cx="115" cy="267.6" r="5" fill="#22c55e"/>
  <text x="126" y="262" font-size="10.5" fill="#22c55e" font-weight="700">decode（GEMV）带宽受限</text>
  <rect x="410.7" y="133.3" width="9" height="9" fill="#a78bfa" transform="rotate(45 415.2 137.8)"/>
  <text x="415" y="128" text-anchor="middle" font-size="10.5" fill="#a78bfa" font-weight="700">prefill 算力受限</text>
  <text x="280" y="372" text-anchor="middle" font-size="10" fill="var(--chart-muted, currentColor)">Source: NVIDIA T4 产品页峰值 + 本文 320GB/s 推导 (2026)</text>
</svg>
<figcaption>Source: <a href="https://www.nvidia.com/en-us/data-center/tesla-t4/">NVIDIA T4 官方产品页</a>（峰值）+ 本文带宽推导（320GB/s），2026-10-31。拐点：FP32 约 25.3、INT8 约 406 FLOP/Byte（本文推导）。</figcaption>
</figure>

## AI Infra 视角：逐行读完规格表，每个参数对应一条决策依据

前面几节是三笔账的深挖。这一节把方法反过来用：假设你刚接手一批 T4，手里只有一张完整规格表，AI Infra 工程师的工作不是背数字，而是把每一行翻译成一条决策依据。我的读法是把全表拆成五张预算表：架构代次行决定软件生态边界，算力行决定 kernel 走哪条电路，显存与缓存行决定容量和吞吐，SM 资源行决定 kernel 调优空间，互连与功耗行决定机型和部署密度。深挖过的参数给结论和回链，前面没展开的这里补齐。

### 架构代次行：软件生态的边界条件

| 参数 | T4 规格 | infra 决策依据 |
|---|---|---|
| Architecture | Turing | 特性集代名词：这一代没有 BF16、TF32、结构化稀疏，也没有 MIG |
| Compute Capability | 7.5 | 编译目标 sm_75，判断 kernel 库支持矩阵的第一道筛子 |
| Manufacturing Process | 12 nm FFN | 同代频率与功耗的物理解释，跨代对比时用 |
| Release Year | 2018 | 折旧与生态维护期的粗估输入 |

白话解释：看到 Turing 和 7.5，有经验的工程师脑内自动展开一张"这卡没有什么"的清单——BF16、TF32、2:4 结构化稀疏、cp.async 异步拷贝都是 Ampere（CC 8.0）及以后的特性，MIG 从 A100 开始才有。对 infra 的直接含义：从 A100/H100 生态迁来的 BF16 权重和按新特性写的 kernel，到 T4 上要么转 FP16 重新校准，要么直接编译不过。所以选型除了"算力够不够"，永远要加一条"特性集够不够"；年份和制程则进成本账——2018 年的卡在 2026 年还值不值得用，先看它缺的特性能否用软件路径绕过去（INT8 就是 T4 的绕行通道）。

### 算力行：prefill 吞吐预算

| 参数 | T4 规格 | infra 决策依据 |
|---|---|---|
| CUDA Cores | 2560 | 标量路径算力的分子 |
| SM Count | 40 | 任务划分的颗粒度，40 个"车间" |
| Tensor Cores | 320 | 矩阵路径算力的分子（40 SM × 8） |
| Clock Speed | 585 MHz base / 1590 MHz boost | 标称算力按 boost 口径算，长时负载稳不住 |
| FP32 Performance | 8.1 TFLOPS | = 2560 × 2 × 1.59 GHz，非张量算子的预算 |
| Tensor Performance | 65 TFLOPS (FP16) | 量化档位选择：FP16/INT8/INT4 = 65/130/260 |

峰值算力本身就该手算验证一遍（本文推导）：

```
FP32 峰值 = 2560 核 × 2 FLOP/周期 × 1.59 GHz ≈ 8.1 TFLOPS
```

白话解释：官方标称用的是 boost 频率，而 70W 功耗墙下长时间推理大概率稳不住 1590 MHz——这就是"预算按标称、实际打折旧"的根源（免费 T4 为什么慢的三重折扣，见上文 decode 一节）。算力行最终汇成一个优化决策：prefill 吃 Tensor Core，量化档位每升一级峰值翻倍；decode 却只吃带宽，不吃这份红利（roofline 推导见上文）。所以"T4 上量化先救 prefill"是动手优化前的第一判断。

### 显存与缓存行：容量与带宽预算

| 参数 | T4 规格 | infra 决策依据 |
|---|---|---|
| Memory | 16 GB GDDR6 | 模型规模 × 精度的容量账（见上文） |
| Memory Bus Width | 256-bit | 带宽公式的一半：÷8 变 32 Byte/次 |
| Maximum Memory Clock | 5001 MHz | DDR 翻倍后 ≈ 10 Gbps，直接当速率会算出腰斩的 160GB/s（见上文） |
| Peak Memory Bandwidth | 320 GB/s | decode 吞吐天花板：7B INT4 ≈ 91 token/s（见上文） |
| L2 Cache Size | 4 MB（TU104 架构口径，T4 官方页未单列） | decode 流式读权重基本无效，batch 内复用才受益 |

L2 是规格表里最容易被高估的一行。decode 每个 token 都要把 3.5GB 权重从头读一遍，4MB 的 L2 相对权重是九牛一毛，而且这些数据在"下一次读取"之前早就被挤出去了——真正能吃到 L2 的是同一批次内的数据复用：多请求共享的 system prompt KV、融合算子反复访问的激活值。白话解释：小 batch 场景别指望 cache 把带宽墙变矮，容量规划按零缓存收益做最坏预算。

### SM 资源行：kernel 调优的硬约束

| 参数 | T4 规格 | infra 决策依据 |
|---|---|---|
| Shared Memory per SM | 64 KB | tile（分块）数据的驻地，直接限 tile 上限 |
| Max Shared Memory per Block | 48 KB（可 opt-in 到 64 KB） | tile 尺寸的硬上限 |
| Registers per SM | 65536 × 32-bit | occupancy 公式的分母 |
| Max Registers per Thread | 255 | 编译期就要守的每线程上限 |
| Warp Size | 32 | 调度最小单位，不足 32 线程也占一整个 warp |
| Max Threads per Block | 1024 | 启动配置上限 |
| Max Threads per SM | 1024 | = 32 warps/SM，驻留线程上限 |
| Max Blocks per SM | 16 | 小 block 多驻留策略的上限 |

先纠一个口径：最后一行在一些聚合网站上写着 32，那是 Volta（CC 7.0）的数字；按 [NVIDIA CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) 的 compute capability 规格表，7.5 是 16。聚合站的表永远要回官方文档核对一遍——这和上文带宽的 300/320 口径矛盾是同一种病。

这组数字合起来就是 occupancy（占用率）三公式：寄存器、共享内存、线程与块数各算一遍能驻留多少 block，取最小值（演算见上文 SM 一节）。它们直接决定 kernel 写法的三个选择：

- **block 开多大**：1024 线程的 block 顶满 SM 上限，一个 block 就吃光驻留名额。
- **tile 开多大**：128×128 的 FP16 分块占 32 KB，48 KB 预算里放不下两份，要么缩 tile，要么 opt-in 到 64 KB。
- **寄存器敢用多少**：每线程 64 个寄存器时 1024 线程刚好用满 65536 个槽位，再往上就得牺牲驻留数——`__launch_bounds__` 和 `-maxrregcount` 就是在这笔账上做交易。

白话解释：Nsight Compute 报 occupancy 上不去时，答案几乎总在这四行规格里；它是手写算子、读 CUTLASS、写 TensorRT plugin 时真正的语法。

### 互连与功耗行：机型与部署密度

| 参数 | T4 规格 | infra 决策依据 |
|---|---|---|
| Interconnect | PCIe Gen3 x16 | 单向约 16 GB/s：权重加载与 CPU offload 的过路费 |
| TDP | 70 W | 低于 75 W 插槽供电，可高密度部署（免外接电源为推断，见开头定位一节） |
| 散热（规格表不写） | 被动散热 | 依赖服务器风道，上架检查项 |

PCIe 的 16 GB/s 主要影响两件事。一是模型加载：16 GB 权重理论上一秒出头就能过完 PCIe，实际加载慢通常是存储和反序列化——先量再猜，别急着怪接口。二是 CPU offload 方案：每 token 需要过 PCIe 的字节直接进延迟，7B INT4 的 3.5GB 权重若整体放在 CPU，单搬运就是 3.5 ÷ 16 ≈ 220 ms/token 的量级，GPU 上 10.94 ms 的 GEMV 预算完全被淹没——这就是"能跑起来"和"能提供服务"的差距（本文推导）。70W 则是部署侧的红利：不超过 PCIe 插槽 75W 供电上限，免外接电源线，2U 服务器插 8 张以上不稀奇；代价是功耗墙加被动散热，机箱风道不达标就是稳定性事故（回链免费 T4 慢的三重折扣）。

<!-- UNIQUE INSIGHT: 规格表逐行映射到五类 infra 决策（生态边界 / prefill 预算 / 容量带宽 / kernel 调优 / 部署密度）的解读框架为本文原创整理，覆盖全表 25 个参数；2026-10-31 中文 SERP 未见同类逐行系统化解读 -->

一句话总结：规格表不是宣传页，是预算表——算力行给 prefill 立预算，显存行给容量和 decode 立预算，SM 资源行给 kernel 立预算，架构行和互连行圈定生态与部署的边界。换一张新卡（L4、A10 还是更新的 Blackwell），这套逐行翻译的流程原样可用。

## 2026 年还值得学 T4 吗：与 L4、A10 和 Orin 对比

作为生产卡，T4 的位置已经被 L4 和 A10 取代；但作为"学习推理硬件的第一张卡"，它仍是我心目中性价比最高的教具之一，因为它的规格简单到可以手算，而手算能力正是推理优化的第一课。

| 指标 | T4 | L4 | A10 |
|---|---|---|---|
| 架构 | Turing（2018） | Ada | Ampere |
| 显存 | 16GB GDDR6 | 24GB | 24GB |
| 显存带宽 | 320 GB/s | 300 GB/s | 600 GB/s |
| INT8 峰值 | 130 TOPS（密集） | 485 TOPS（稀疏）/ 242.5（密集） | 250 TOPS（密集） |
| 功耗 | 70W | 72W | 150W |

数据来自 [NVIDIA L4 官方产品页](https://www.nvidia.com/en-us/data-center/l4/) 与 [A10 官方产品页](https://www.nvidia.com/en-us/data-center/products/a10-gpu/)。口径陷阱又来了：L4 宣传页最显眼的 485 TOPS 是稀疏值，密集口径只有 242.5；A10 的 250 与 T4 的 130 都是密集值，同口径可比，对 485 则不可比。稀疏和密集混着比，会得出 L4 比 A10 强的错觉。L4 官方页还给出相对 T4 约 2.5 倍的 AI 图像生成性能提升，引用时要连同场景口径一起写。

软件支持边界：经典 TensorRT 对 T4 一直有完整支持，是主流部署路径；TensorRT-LLM 面向更新的架构，对 Turing 的支持情况需以官方支持矩阵为准（方向性判断：不要假设 T4 能用上其最新特性）。所谓"TensorRT 不支持 T4"是以讹传讹，经典版本在 T4 上至今可复现。

再把另一张常被拿来和 T4 混谈的卡放进来：Jetson Orin Nano Super。

| 指标 | Tesla T4 | Jetson Orin Nano Super |
|---|---|---|
| 架构 | Turing | Ampere |
| CUDA Core | 2560 | 1024 |
| Tensor Core | 320 | 32 |
| 显存 | 16GB GDDR6，320GB/s | 8GB LPDDR5（与 CPU 共享），102.4GB/s |
| 功耗 | 70W | 约 25W |
| 形态 | PCIe 数据中心加速卡 | 边缘 SoC |

最后一个口径陷阱：Orin Nano Super 宣传的 67 TOPS 是 INT8 口径，T4 的 65 TFLOPS 是 FP16 Tensor 口径，数字看着只差 2，单位完全不同，不能直接比。

<!-- PERSONAL EXPERIENCE: 作者有一手 Orin Nano Super（Ampere）部署经验，用于本节的 T4/Orin 学习互补对比；T4 部分未做实测，全文性能数字均为官方规格或本文推导 -->

我手上正好有 Orin Nano Super 的部署经验，这张表对我不只是纸面对比：T4 教你数据中心卡的经典玩法（TensorRT、INT8 校准、多租户 serving），Orin 教你在功耗墙与共享内存约束下做边缘部署。两张卡是学习路径的两条腿，跨架构对比正好训练"先问口径、再问架构、最后问数字"的习惯。

<figure class="blog-chart">
<svg viewBox="0 0 560 380" style="max-width: 100%; height: auto; font-family: 'Inter', system-ui, sans-serif;" role="img" aria-label="T4 vs L4 vs A10：带宽与 INT8 峰值（同口径对比）。双面板条形图对比 T4、L4、A10。左面板为显存带宽：T4 320 GB/s、L4 300 GB/s、A10 600 GB/s。右面板为 INT8 峰值，统一用密集口径：T4 130 TOPS、L4 242.5 TOPS、A10 250 TOPS；虚线标出 L4 宣传页常用的稀疏口径 485 TOPS，与密集值不可直接比较。颜色在两个面板中保持卡与色的对应：T4 橙、L4 蓝、A10 紫。数据来自 NVIDIA 三张卡的官方产品页。">
  <text x="280" y="29" text-anchor="middle" font-size="16" font-weight="800" fill="currentColor">T4 vs L4 vs A10：带宽与 INT8 峰值（同口径对比）</text>
  <text x="145" y="62" text-anchor="middle" font-size="12" font-weight="700" fill="currentColor">显存带宽（GB/s）</text>
  <text x="415" y="62" text-anchor="middle" font-size="12" font-weight="700" fill="currentColor">INT8 峰值（TOPS，密集口径）</text>
  <g stroke="currentColor" opacity="0.08">
    <line x1="30" y1="195" x2="260" y2="195"/><line x1="30" y1="142.5" x2="260" y2="142.5"/><line x1="30" y1="247.5" x2="260" y2="247.5"/>
    <line x1="300" y1="258" x2="530" y2="258"/><line x1="300" y1="216" x2="530" y2="216"/>
    <line x1="300" y1="174" x2="530" y2="174"/><line x1="300" y1="132" x2="530" y2="132"/>
  </g>
  <line x1="30" y1="300" x2="260" y2="300" stroke="currentColor" opacity="0.3"/>
  <line x1="300" y1="300" x2="530" y2="300" stroke="currentColor" opacity="0.3"/>
  <g>
    <rect x="53" y="188" width="44" height="112" rx="3" fill="#f97316"/>
    <rect x="123" y="195" width="44" height="105" rx="3" fill="#38bdf8"/>
    <rect x="193" y="90" width="44" height="210" rx="3" fill="#a78bfa"/>
    <rect x="323" y="245.4" width="44" height="54.6" rx="3" fill="#f97316"/>
    <rect x="393" y="198.2" width="44" height="101.8" rx="3" fill="#38bdf8"/>
    <rect x="463" y="195" width="44" height="105" rx="3" fill="#a78bfa"/>
  </g>
  <g font-size="10.5" fill="currentColor" text-anchor="middle">
    <text x="75" y="182">320</text><text x="145" y="189">300</text><text x="215" y="84">600</text>
    <text x="345" y="239">130</text><text x="415" y="192">242.5</text><text x="485" y="189">250</text>
  </g>
  <g font-size="12" fill="currentColor" opacity="0.8" text-anchor="middle">
    <text x="75" y="318">T4</text><text x="145" y="318">L4</text><text x="215" y="318">A10</text>
    <text x="345" y="318">T4</text><text x="415" y="318">L4</text><text x="485" y="318">A10</text>
  </g>
  <line x1="300" y1="96.3" x2="530" y2="96.3" stroke="currentColor" stroke-width="1.2" stroke-dasharray="5 4" opacity="0.45"/>
  <text x="415" y="91" text-anchor="middle" font-size="10" fill="var(--chart-muted, currentColor)">L4 稀疏口径 485 TOPS（不同口径，不参与同口径比较）</text>
  <text x="280" y="345" text-anchor="middle" font-size="10.5" fill="var(--chart-muted, currentColor)">稀疏与密集混比会得出"L4 比 A10 强"的错觉；本图右面板统一为密集口径。</text>
  <text x="280" y="372" text-anchor="middle" font-size="10" fill="var(--chart-muted, currentColor)">Source: NVIDIA T4 / L4 / A10 官方产品页 (2018-2026)</text>
</svg>
<figcaption>Source: <a href="https://www.nvidia.com/en-us/data-center/tesla-t4/">NVIDIA T4</a>、<a href="https://www.nvidia.com/en-us/data-center/l4/">L4</a>、<a href="https://www.nvidia.com/en-us/data-center/products/a10-gpu/">A10</a> 官方产品页。</figcaption>
</figure>

## 我会用 T4 学什么：一条可复制的闭环

不要只拿 T4 跑一次 YOLO 或者部署一个 LLM 就完事。把它当成一台 AI Infra 实验设备，走完这条闭环：PyTorch 模型，ONNX 导出，量化，自定义算子，TensorRT 引擎，GPU Serving，性能 Profiling。这条链路把部署、量化、CUDA 优化、推理引擎、服务调度和性能分析全部串起来。

这条路线值得走的证据来自官方教程。[GCP 的 T4 推理教程](https://docs.cloud.google.com/compute/docs/tutorials/ml-inference-t4)在同一张卡上做 ResNet 优化：原生 TensorFlow 314 fps，转 FP16 后 988 fps，TensorRT INT8 后 1524 fps，三步走完接近 4.9 倍提升，说明在 T4 上"INT8 值得做"不是口号。顺带一个口径提醒：教程原文同时写"延迟降至 0.08 ms"，但按 1524 fps 折算单张约 0.66 ms，两个数字对不上；0.08 s 恰好等于 batch 128 的整批耗时（128 ÷ 1524 ≈ 0.084 s），原文大概率把"秒"标成了"毫秒"（我的推断）。引用延迟数字前，先拿吞吐倒推一遍，这是防止口径陷阱的最后一道工序。LLM 侧还有旁证：[Microsoft 开源博客](https://opensource.microsoft.com/blog/2022/05/02/optimizing-and-deploying-transformer-int8-inference-with-onnx-runtime-tensorrt-on-nvidia-gpus)（2022-05-02）报告 T4 上 BERT-Large INT8 延迟低至 1.2ms，注意这是 Bing/Azure、TensorRT 8 语境的数据，别和同文里 A100 的数字混用。

闭环里的每个环节都值得单独深挖：roofline 用 Nsight 实测、手写 INT4 weight-only GEMM、KV 缓存管理、连续批处理（continuous batching）、多租户 GPU serving。本文先把地基打牢。

## 常见问题

### T4 相当于什么级别的显卡？

T4 与 GeForce RTX 2080 同为 TU104 核心，同属 2018 年的 Turing 架构（RTX 20 系内各型号核心不同：2080 Ti 用 TU102，2070/2060 用 TU106）。但它是一张数据中心形态的卡：无显示输出、被动散热、显存带 ECC，定位是服务器推理而不是桌面使用。把它当"老款游戏卡"看待，会误解它的设计目标。

### 为什么 Colab/Kaggle 的免费 T4 跑大模型很慢？

先撞带宽墙：7B INT4 的 decode 理论上限约 91 token/s（推导见上文），这是 320GB/s 显存带宽决定的物理天花板，再叠加反量化、KV 缓存访问等现实折扣。云平台还有 70W 功耗墙、被动散热和限频三重折扣，实测十几 token/s 属于正常现象，不是环境坏了。

### T4 和 P40 做推理选哪个？

追求 INT8/INT4 吞吐选 T4：它有完整的 Tensor Core 整数路径（130 TOPS 口径）。P40 没有 Tensor Core，INT8 靠 CUDA Core 的 DP4A 指令实现，量化收益有限；它的讨论价值主要在更大的显存装更大的 FP16 权重。老卡选型的关键动作，就是先问"我的算子走哪条电路"。

### 如何在 T4 上用 TensorRT 做 INT8？

按上文 GCP 官方教程走即可复现：模型导出 ONNX，TensorRT 构建引擎，执行 INT8 校准，教程里 ResNet 从 314 fps 提升到 1524 fps。记住口径提醒：130 TOPS 是 Tensor Core 峰值，实际收益取决于算子覆盖率和校准质量，不是标称值的固定折扣。

### T4 能跑 14B 模型吗？

能装下，但仅限 INT4：14B × 0.5 Byte = 7GB 权重（推导见上文容量算术），16GB 显存还剩 9GB 给 KV 缓存、运行时和激活值。上下文一长、并发一多就会触顶，所以"能装下权重"和"能稳定服务"是两回事，上车前先做一遍显存预算。

## 结论

T4 教给推理工程师的是三笔账和两个警觉：带宽账（256-bit 与 10Gbps 推出 320GB/s）、容量账（参数量乘精度字节数）、吞吐账（3.5GB 除以带宽得到约 91 token/s）；警觉口径陷阱（稀疏对密集、TOPS 对 TFLOPS）和峰值幻觉（65 TFLOPS 是 Tensor 路径峰值）。这些手算能力比任何单卡结论都长寿。文中每个推导都给了完整算式，拿计算器就能复算与勘误。
