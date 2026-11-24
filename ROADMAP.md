# 开放集增量自进化电磁调制识别（OS-SE-AMR）技术路线与研发规划

## 一、 总体技术路线图

本研究旨在解决传统自动调制识别（AMR）模型在开放电磁环境下“遇到未知样式无法拒识”以及“吸收新样式时产生灾难性遗忘与极小样本过拟合”的双重瓶颈。整体技术路线按照 **“基础设施搭设 $\rightarrow$ 静态开集拒识 $\rightarrow$ 动态解耦复现 $\rightarrow$ 基线对齐测试 $\rightarrow$ 闭环融合与创新”** 五轮演化推进。

```text
                    [ 电磁信号流 X(t) ]
                            │
                            ▼
           ┌────────────────────────────────┐
           │ 阶段 0: 骨干与开放集划分器     │
           │  (TLDNN / IQFormer 适配)       │
           └────────────────┬───────────────┘
                            │
                            ▼
           ┌────────────────────────────────┐
           │ 阶段 1: 静态开集拒识底座       │
           │ (MLS / ARPL / 原型度量体系)    │
           └────────────────┬───────────────┘
                            │
             ┌──────────────┴──────────────┐
             ▼                             ▼
 ┌───────────────────────┐     ┌────────────────────────┐
 │ 阶段 2 (主线):        │     │ 阶段 2 (支线):         │
 │ 未知发现与原型聚类    │     │ 抗遗忘与重放蒸馏       │
 │ (HDBSCAN + 轮廓门控)  │     │ (Herding + LwF 约束)   │
 └───────────┬───────────┘     └───────────┬────────────┘
             │                             │
             └──────────────┬──────────────┘
                            │
                            ▼
           ┌────────────────────────────────┐
           │ 阶段 3: 多维度对齐与环境评估   │
           │  (SSD-AMC / iCaRL / DOCO 漂移) │
           └────────────────┬───────────────┘
                            │
                            ▼
           ┌────────────────────────────────┐
           │ 阶段 4: 自进化闭环与 SS 破局   │
           │ (TLDNN-SS + 盲 SNR 双感知门控) │
           └────────────────────────────────┘
```

> 详细算法协议、数学推导边界与软件工程接口契约请参阅 [`docs/TECHNICAL_DESIGN.md`](docs/TECHNICAL_DESIGN.md)。

---

## 二、 阶段化推进路线与里程碑

### 阶段 0：基础设施搭建与开放度划分构建（周期：1 周）

#### 0.1 研发目标
完成数据集清洗、无泄露开放集数据划分器（Open-Set Splitter）编写，并将 **TLDNN**、**IQFormer** 接入统一的数据加载与 64 维特征暴露接口。

#### 0.2 实验协议与开放度定义
采用标杆数据集 **RadioML2016.10a**（11 类调制样式，20 档信噪比 $-20 \sim +18\text{ dB}$，每类每档 1,000 条样本，张量维度 $2 \times 128$）。主实验流程筛选信噪比 $\text{SNR} \ge 0\text{ dB}$（共 10 档 SNR，每类 10,000 条、总计 110,000 条样本）保证信号特征可分性；负信噪比样本（$-10 \sim -2\text{ dB}$）保留用于盲 SNR 回归训练及阶段 3 环境漂移压力测试。

按每个 `(mod, snr)` 组合独立做 `60% Train / 20% Val / 20% Test` 确定性分层切分（详见 [`docs/TECHNICAL_DESIGN.md`](docs/TECHNICAL_DESIGN.md#1-数据集切分与流式到达协议)）。开放度（Openness）按标准公式定量控制：

$$
\text{Openness} = 1 - \sqrt{\frac{2 \times C_{\text{train}}}{C_{\text{target}} + C_{\text{train}}}}
$$

其中 $C_{\text{train}}$ 为已知类数量，$C_{\text{target}}$ 为测试时总类别数（包含已知与未知类，本数据集全集 $C_{\text{target}} = 11$）。

* **数据划分矩阵（精确公式校准版）**：

| 阶段划分 | 已知类数量 ($C_{\text{train}}$) | 已知类包含样式 | 未知类数量 ($C_{\text{unknown}}$) | 未知类包含样式 | 精确开放度 ($\text{Openness}$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **测试配置 A** | 7 | 8PSK, BPSK, CPFSK, GFSK, PAM4, QAM16, QPSK | 4 | QAM64, WBFM, AM-DSB, AM-SSB | **$11.81\%$** |
| **测试配置 B** | 5 | 8PSK, BPSK, CPFSK, GFSK, PAM4 | 6 | QAM16, QPSK, QAM64, WBFM, AM-DSB, AM-SSB | **$20.94\%$** |

> **口径校准说明**：配置 A 代入公式得 $1 - \sqrt{14/18} \approx 11.81\%$，配置 B 代入公式得 $1 - \sqrt{10/16} \approx 20.94\%$（早期规划草稿中估算值分别为 $13.4\%$ 与 $22.5\%$，现统一校准为严格公式解析值）。

#### 0.3 交付物与阶段通过标准
* **交付物**：`rml2016.py`、`splitter.py`、统一特征抽取接口 `TLDNN` 与 `IQFormer`、封闭集基准权重及切分防泄露单元测试。
* **通过标准**：在封闭集配置 A（$C_{\text{train}}=7, \text{SNR} \ge 0\text{ dB}$）下，TLDNN 测试集分类准确率达到 $88.5\% \pm 0.5\%$；网络稳定输出 $\mathbb{R}^{64}$ 特征向量 $\mathbf{z}$；`Train/Val/Test` 零重叠且所有单元测试通过。

---

### 阶段 1：静态开集拒识基线搭建与指标固化（周期：1~2 周）

#### 1.1 研发目标
引入标准化的离群打分与特征空间边界约束机制，建立静态开集拒识（OSR）测试管线，固化全部打分极性与评估指标口径。

#### 1.2 核心技术机制区分
为避免概念混淆并保证基线复现严谨性，将本阶段的三种核心 OSR 机制严格区分如下：

1. **极大 Logit 打分（MLS, ICLR 2022）**：无需重新训练，直接基于分类器未归一化的最大 Logit 输出计算已知度得分：
   $$
   S_{\text{MLS}}(\mathbf{x}) = \max_{c \in \{1, \dots, C_{\text{train}}\}} g_c(\mathbf{x})
   $$

2. **对抗互反点学习（ARPL, TPAMI 2022 严格基线）**：在标准 `iCGY96/ARPL` 框架中，互反点（Reciprocal Point）$\mathbf{P}_c$ 代表**类外（非 $c$ 类）空间**，通过最大化类 $c$ 样本特征到互反点 $\mathbf{P}_c$ 的距离实现分类，同时利用可学习的开放空间半径 $R$ 限制已知类嵌入到有界区域以抑制开放空间风险：
   $$
   \mathcal{L}_{\text{ARPL}} = \mathcal{L}_{\text{CE}}^{\text{RP}}(\mathbf{z}, y; \{\mathbf{P}_c\}) + \lambda \, \mathcal{L}_{\text{margin}}\big(d(\mathbf{z}, \mathbf{P}_y), R\big)
   $$

3. **类内原型距离度量（Distance-Prototype, Ours）**：在特征空间为每个已知类维护**类内原型中心** $\boldsymbol{\mu}_c$，联合交叉熵与类内紧凑损失（可选配合类间余弦边际约束），迫使已知类特征向各自中心紧缩：
   $$
   \mathcal{L}_{\text{proto}} = \mathcal{L}_{\text{CE}} + \lambda \sum_{i} \|\mathbf{z}_i - \boldsymbol{\mu}_{y_i}\|_2^2
   $$
   推理时以负最小原型距离 $S_{\text{proto}}(\mathbf{x}) = -\min_c \|\mathbf{z} - \boldsymbol{\mu}_c\|_2$ 作为已知度打分。该原型库可直接无缝对接阶段 2 与阶段 4 的新类聚类扩展。

#### 1.3 评估指标固化
统一规定所有 OSR 打分器满足 **“得分越高越倾向于已知类，越低越倾向于未知类”** 的单调极性：
* **AUROC**：衡量已知类与未知类样本得分分布的全局可分性。
* **FPR@TPR95**：在已知类真阳性率（TPR）达到 $95\%$ 时，未知类被误放行为已知类的假阳性率（越低越好）。
* **OSCR（Open-Set Classification Rate）**：随阈值变化同时刻画已知类正确分类率（CCR）与未知类误入率（FPR）的曲线下面积。

#### 1.4 交付物与阶段通过标准
* **通过标准**：在配置 A 下，静态 OSR 基线（ARPL 或 Distance-Prototype）的未知类检测 $\text{AUROC} \ge 0.86$，`OSCR` 与 `FPR@TPR95` 计算经过合成边界单元测试验证无逻辑错误。

---

### 阶段 2：未知发现与抗遗忘增量双线拆解复现（周期：2~3 周）

#### 2.1 研发目标
将自进化闭环解耦为“**未知发现**”与“**抗遗忘增量**”两条支线独立开发调试，在排除前端误拒噪声干扰的情况下验证各自极限性能，并确立流式到达顺序与无监督匹配评测规范。

#### 2.2 模块拆解路线

##### 【主线】未知发现与动态原型扩展模块
* **数据流与到达顺序**：明确 $7 \rightarrow 9 \rightarrow 11$ 的多批次到达顺序（默认协议 **Order A**：Session 1 新增 `QAM64, AM-SSB`，Session 2 新增 `WBFM, AM-DSB`；顺序敏感对照协议 **Order B** 将易混类 `WBFM` 与 `AM-DSB` 拆至不同阶段到达）。
* **聚类与质量门控**：拒识缓冲区样本 $\mathcal{B}_{\text{unk}} \rightarrow$ 提取 L2 归一化特征 $\hat{\mathbf{z}} \rightarrow \text{HDBSCAN}$ 密度聚类 $\rightarrow$ 过滤噪点（`label == -1`）与筛选核心样本 $\rightarrow$ 轮廓系数（Silhouette Coefficient）门控：仅当有效簇数 $\ge 1$、簇规模达标且 $S_{\text{sil}} > 0.45$ 时触发新类生成与新原型 $\boldsymbol{\mu}_{\text{new}}$ 注册，否则暂存样本等待下一批次。
* **无监督评测协议**：采用**锁定已知类的匈牙利 1-to-1 匹配（Locked-Known Hungarian Matching）** 将新生成簇头映射至真实未知类标签，严禁修改已知类头映射，具体见 [`docs/TECHNICAL_DESIGN.md`](docs/TECHNICAL_DESIGN.md#3-匈牙利评测协议与全套指标定义)。

##### 【支线】抗遗忘与连续学习模块
* **旧类回放保持**：构建轻量级回放缓冲区 $\mathcal{M}_{\text{old}}$，采用基于特征空间的 **Herding** 算法为每个旧类固定保留 30 条代表性原始 IQ 样本。
* **双层蒸馏约束**：增量微调时联合回放样本与新类样本交叉熵损失，以及特征空间与旧头 Logits 双层 LwF（Learning without Forgetting）知识蒸馏约束：
  $$
  \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}(\theta) + \beta_{\text{feat}} \|\mathbf{z}_{\text{new}} - \mathbf{z}_{\text{old}}\|_2^2 + \beta_{\text{logit}} \mathcal{L}_{\text{KD}}^{\text{logit}}(g_{\text{old}}^{\text{new}}, g_{\text{old}}^{\text{old}}; T=2.0)
  $$
* **抗漂移原型同步（Prototype Refresh）**：每次增量更新完成后，利用 $\mathcal{M}_{\text{old}}$ 前向重算全部旧类原型 $\boldsymbol{\mu}_c$ 并校准新类原型，消除骨干微调引起的度量空间漂移。

#### 2.3 交付物与阶段通过标准
* **通过标准**：在阶段 2 解耦流式切换（$7 \rightarrow 9$，Order A）下，经匈牙利匹配后的新类识别准确率 $A_{\text{new}} \ge 75\%$，且基础 7 类的准确率降幅（基础类遗忘 $\Delta A_{\text{old}}$）控制在 $5\%$ 以内。

---

### 阶段 3：对照基线补齐与开放动态环境评估（周期：1~2 周）

#### 3.1 研发目标
补充完整且统一口径的横向对比方法（包含通用 OSR/CIL 方法与电磁领域专有基线），并评估信噪比环境漂移对开集决策边界与原型稳定性的影响。

#### 3.2 对照方法与评估矩阵
1. **静态开集对照**：`Softmax-Threshold`、`MLS (ICLR 2022)`、`ARPL (TPAMI 2022)`、`SSD-AMC-Open-set (RadarConf 2026)`。
2. **增量学习对照**：`Naive Fine-Tune`（暴力微调下界）、`iCaRL`（经典重放+NME）、`SimpleCIL`（无微调原型扩展）。
3. **动态环境鲁棒性（参照 DOCO 思想）**：在 High-SNR（$\ge 10\text{ dB}$）数据流上完成模型更新与原型扩张后，分别在 Mid/Low-SNR（$0 \sim 6\text{ dB}$）以及 Hold-Out 负信噪比测试集（$-10 \sim -2\text{ dB}$）上评估已知类误拒率漂移与分类准确率退化幅度。

#### 3.3 交付物与阶段通过标准
* **通过标准**：在统一种子与数据划分下产出全部对照基线的结构化 `metrics.json`，验证静态 OSR 方法随未知类增加全局准确率 $A_{\text{all}}$ 呈阶梯式骤降，而自进化框架呈稳步吸收特性。

---

### 阶段 4：自进化闭环融合、TLDNN-SS 破局与论文写作（周期：2~3 周）

#### 4.1 研发目标
打通“**开集拦截 $\rightarrow$ 盲 SNR 与置信度双感知门控 $\rightarrow$ HDBSCAN 未知发现 $\rightarrow$ 簇内 SS 增强 $\rightarrow$ 增量微调与原型同步**”的端到端真实流式闭环，攻克极小样本与低信噪比噪声污染难题，完成全量消融实验与论文成稿。

#### 4.2 核心电磁创新点设计

##### 1. 基于簇内 TLDNN Segment Substitution (SS) 的极小样本扩充
新发现的未知类经过高置信核心样本过滤后样本量往往有限，直接参与增量微调极易过拟合。借助 TLDNN 的离散片段替换（Segment Substitution）机理，严格在同一高纯度核心簇 $\Omega_k^{\text{core}}$ 内部进行时域半段置换，使新类有效训练样本扩充 $3 \sim 5$ 倍：
$$
\tilde{\mathbf{x}}_{\text{aug}} = \text{Concat}\left(\mathbf{x}_i[:, 0:L/2], \; \mathbf{x}_j[:, L/2:L]\right), \quad \mathbf{x}_i, \mathbf{x}_j \in \Omega_k^{\text{core}}, \; i \neq j
$$
同时针对连续相位/调频信号（如 `CPFSK`、`GFSK`、`WBFM`）在 $L/2$ 拼接处可能产生的相位跳变伪影，设计 `SS-IQ`（直接拼接后转 A/P）与 `SS-PhaseAligned`（拼接处相位连续对齐）对照消融。

##### 2. 置信度与盲 SNR 双重感知演化门控
**严禁在在线自进化过程中读取数据集真实 SNR 标签**。门控依赖实时的开集得分与**盲 SNR 估计器**输出 $\widehat{\text{SNR}}(\mathbf{x})$（提供 **M2M4 高阶矩物理估计** 与 **仅在 Session 0 已知类上训练的轻量化辅助 SNR 回归头** 两种实现，详见 [`docs/TECHNICAL_DESIGN.md`](docs/TECHNICAL_DESIGN.md#2-盲-snr-估计方案与三阶双感知门控)）：
$$
\text{Gate}_{\text{sample}}(\mathbf{x}) = \mathbb{I}\left( S_{\text{OSR}}(\mathbf{x}) < \tau_{\text{low}} \;\land\; \widehat{\text{SNR}}(\mathbf{x}) \ge \tau_{\text{snr}} \right)
$$

```text
                     [ 真实开放流：未知与已知混合样本 ]
                                     │
                                     ▼
                     [ OSR 得分拦截 + 盲 SNR 双感知门控 ]
                                     │
                           ┌─────────┴─────────┐
                           ▼ (Pass)            ▼ (Reject / Low-SNR)
               [ 存入拒识缓冲区 B_unk ]   [ 丢弃噪声或保留待观察 ]
                           │
                           ▼
             [ HDBSCAN 聚类 + 轮廓系数门控 (>0.45) ]
                           │
                 ┌─────────┴─────────┐
                 ▼ (触发新类)        ▼ (未成簇)
      [ 核心样本筛选 + 簇内 SS 扩充 ] [ 暂存 B_unk ]
                 │
                 ▼
      [ 增量微调 + 双层 LwF + Herding 回放 ]
                 │
                 ▼
      [ 原型库同步刷新 P_0 -> P_new + 阈值重校准 ]
```

---

## 三、 实验结果记录表模板

> **说明**：表 1 已更新为阶段 1 实测结果（RadioML2016.10a，配置 A，`SNR >= 0 dB`，`seed=42`）；表 2 与表 3 已全量更新为阶段 4（`outputs/stage4/summary_stage4.json`）在含已知类与负信噪比（$-10 \sim -2\text{ dB}$）干扰的**真实混流**下的实测闭环数据。

### 表 1：阶段 1 静态开集拒识性能实测对比表（RadioML2016.10a，配置 A，$\text{SNR} \ge 0\text{ dB}$，`seed=42`）

> **技术演进说明**：实测发现封闭集分类训练在末层 `LayerNorm` 与各向同性度量下产生“各向异性神经坍缩”，使无边界或各向同性方法（Softmax/MLS/ARPL/欧氏原型）将 `WBFM` 与 `AM-DSB` 坍缩至已知类 `GFSK` 附近（AUROC `52%~69%`）；移除末层归一化约束并引入收缩协方差感知与多尺度融合原型（`MS-MahaProto`）后，AUROC 跃升至 **`91.68%`**（`OSCR=90.96%`）。

| 骨干网络 | 拒识算法 | Known Class Acc (%) | AUROC (%) $\uparrow$ | FPR@TPR95 (%) $\downarrow$ | OSCR (%) $\uparrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **TLDNN** | Softmax-Threshold | 98.76 | 57.45 | 73.76 | 57.26 |
| **TLDNN** | MLS (ICLR 2022) | 98.76 | 52.28 | 74.25 | 52.08 |
| **TLDNN** | ARPL (TPAMI 2022) | 98.81 | 63.53 | 70.03 | 63.30 |
| **IQFormer** | MLS (ICLR 2022) | **98.94** | 69.54 | 70.53 | 69.35 |
| **TLDNN** | Euclidean-Prototype (Ablation) | 98.79 | 62.28 | 65.53 | 62.01 |
| **TLDNN** | 64d-MahaProto (Ablation) | 98.76 | 87.30 | 41.90 | 87.04 |
| **TLDNN (Ours)** | **MS-MahaProto** | 98.76 | **91.68** | 26.80 | **90.96** |
| **IQFormer (Ours)** | **MS-MahaProto** | **98.94** | 91.44 | **25.96** | 90.76 |

### 表 2：阶段 4 多阶段自进化闭环实测对比表（配置 A，$7 \rightarrow 9 \rightarrow 11$ 真实混流闭环，`seed=42`）

> 注：真实混流按 $50\%$ 当前已知 + $50\%$ 新增未知采样，且混入 $-10\sim -2\text{ dB}$ 负信噪比干扰；全系统未知类均通过无标签 OSR + 盲 SNR（`AuxSNR >= 4 dB`）双感知门控、`HDBSCAN (stem_preln)` + 碎片子簇合并与质量剪枝自动发现，并由质量/规模自适应 `TLDNN-SS` 扩充。另外，`RML2016.10a` 中 `AM-SSB` 在全 SNR 下呈平直白噪声统计缺陷（见表 3 注释），故主闭环以物理双新类 **Order B** 为主对照并同步给出 **Order A** 与无缺陷 **Clean10（表 2B）** 完整轨迹。

| 到达协议 | 方法机制 | 阶段 0 ($C=7$) | 阶段 1 ($C=9$) | | | 阶段 2 ($C=11$) | | | 终态基础类遗忘 $\Delta A_{\text{old}}$ (%) $\downarrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| | | $A_{\text{all}}$ (%) | $A_{\text{all}}$ (%) | $A_{\text{old}}$ (%) | $A_{\text{new}}$ (%) | $A_{\text{all}}$ (%) | $A_{\text{old}}$ (%) | $A_{\text{new}}$ (%) | |
| **Order B** | **Static Baseline**（仅拒识不进化） | 98.76 | 76.81 | **98.76** | 0.00 | 62.85 | **98.76** | 0.00 | 0.00 |
| **Order B** | **Naive Fine-Tune**（暴力微调） | 98.76 | 32.58 | 15.19 | **93.47** | 16.80 | 1.65 | **43.30** | 97.11 |
| **Order B** | **iCaRL (`m=100`)**（经典增量重放） | 98.77 | 83.58 | 90.14 | 60.62 | 36.23 | 39.11 | 31.19 | 59.66 |
| **Order B** | **Ours (W/O SS Aug)** | 98.76 | 85.20 | 93.78 | 55.17 | 71.90 | 93.09 | 34.83 | 5.67 |
| **Order B** | **Ours (Full Model + SS)** | 98.76 | **86.37** | 93.54 | 61.27 | **72.51** | 93.44 | 35.88 | **5.31** |
| **Order A** | **Static Baseline**（仅拒识不进化） | 98.76 | 76.81 | **98.76** | 0.00 | 62.85 | **98.76** | 0.00 | 0.00 |
| **Order A** | **Naive Fine-Tune**（暴力微调） | 98.76 | 64.28 | 69.46 | **46.17** | 20.82 | 0.04 | **57.20** | 98.72 |
| **Order A** | **iCaRL (`m=100`)**（经典增量重放） | 98.77 | 77.12 | 94.63 | 15.82 | 41.57 | 44.15 | 37.06 | 54.62 |
| **Order A** | **Ours (W/O SS Aug)** | 98.76 | 77.84 | **95.59** | 15.72 | 72.10 | **94.89** | 32.24 | **3.87** |
| **Order A** | **Ours (Full Model + SS)** | 98.76 | **78.03** | 93.74 | 23.05 | **72.52** | 94.86 | 33.42 | 3.89 |

### 表 2B：`RML2016.10a-Clean10` 无缺陷 10 类物理有效闭环对比（$7 \rightarrow 9 \rightarrow 10$，`Openness = 9.25%`，`seed=42`）

| 方法机制 | S0 $A_{\text{all}}$ (%) | S1 $A_{\text{all}} / A_{\text{old}} / A_{\text{new}}$ (%) | S2 $A_{\text{all}} / A_{\text{old}} / A_{\text{new}}$ (%) | 终态 $\Delta A_{\text{old}}$ (%) $\downarrow$ | 簇数 $K_{\text{hat}}$ (raw→kept) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Static Baseline** | 98.76 | 76.81 / **98.76** / 0.00 | 69.13 / **98.76** / 0.00 | 0.00 | S1: 0→0, S2: 0→0 |
| **Naive Fine-Tune** | 98.76 | 32.58 / 15.19 / **93.47** | 18.84 / 0.02 / **62.75** | 98.74 | S1: 2→2, S2: 5→4 |
| **iCaRL (`m=100`)** | 98.77 | 83.58 / 90.14 / 60.62 | 41.74 / 42.86 / 39.13 | 55.91 | S1: 2→2, S2: 10→8 |
| **Ours (W/O SS Aug)** | 98.76 | 85.20 / 93.78 / 55.17 | 79.49 / 93.51 / 46.77 | 5.24 | S1: 2→2, S2: 2→1 |
| **Ours (Full Model + SS)** | 98.76 | **86.37** / 93.54 / 61.27 | **79.78** / 93.59 / 47.57 | **5.16** | S1: 2→2 ($\Delta K=0$), S2: 3→1 ($\Delta K=0$) |

### 表 3：阶段 4 关键四维消融实验实测结论矩阵（`outputs/stage4/summary_stage4.json`）

> **数据集物理缺陷记录**：直接观测 `RML2016.10a` 原始波形确认 `AM-SSB` 在 `-10~+18 dB` 均表现为平直白噪声统计（`ac1≈0.08`, 左右边带 PSD 比 `0.992~0.998`）；在其余 3 个物理有效未见未知类（`QAM64, WBFM, AM-DSB`）上，`AuxSNRRegressor` 跨调制泛化实现 **MAE = `1.630 dB`、$\ge 4\text{ dB}$ 门控 F1 = `96.76%`**（对比 `M2M4`: `MAE=5.337 dB, F1=81.27%`）。

| 消融维度 | 对照设置 | 实测核心指标 | 实证结论 |
| :--- | :--- | :--- | :--- |
| **盲 SNR 门控（Order B S1）** | `No-SNR-Gate` vs `M2M4` vs **`Aux-SNR (Ours)`** vs `Oracle-SNR` | S1 $\mathcal{B}_{\text{unk}}$ 纯度：`66.27%` / `84.96%` / **`85.85%`** / `84.50%`<br>S1 负 SNR 漏入：`45.28%` / `7.50%` / **`0.05%`** / `0.00%`<br>S1 $A_{\text{all}} / A_{\text{new}}$：`85.04/54.05` vs **`86.37/61.27`** vs `85.78/57.53` | `AuxSNR` 将负 SNR 噪声泄漏由 `45.28%` 压至 **`0.05%`**，S1 聚类纯度与增量精度均全面超越无门控与物理 M2M4 基线 |
| **SS 拼接与质量/规模门控** | `W/O SS` vs `SS-IQ` vs **`SS-PhaseAligned (Adaptive)`** | **Order A**：S1 $A_{\text{new}}$ `15.72% -> 23.05%`，S2 $A_{\text{all}}$ **`72.52%`**（高于 W/O SS `72.10%` 与 SS-IQ `72.37%`）<br>**Order B**：S1 $A_{\text{all}}$ **`86.37%`**，S2 $A_{\text{all}} / A_{\text{new}}$ **`72.51% / 35.88%`**（均优于 W/O SS `71.90% / 34.83%` 与 SS-IQ `71.79% / 34.74%`） | 子簇余弦合并（`5→1` / `4→1`）+ 质量/规模自适应门控彻底消除了 S2 脏簇过采样的负效应，实现 S1 与 S2 双阶段一致正增益 |
| **原型同步机制** | `W/O Prototype Sync` vs **`With Prototype Sync (Ours)`** | S2 流内 Known FRR：Order A **`98.55% -> 24.92%`**、Order B **`98.57% -> 19.82%`**<br>S2 $\mathcal{B}_{\text{unk}}$ 纯度：`53.99%/34.10% ->` **`89.23%/64.81%`**<br>S2 $A_{\text{all}} / \Delta A_{\text{old}}$：`57.32/26.46% ->` **`72.52/3.89%`**（Order B：`53.23/33.18% ->` **`72.51/5.31%`**） | 实证不刷新原型与阈值将导致二轮演进时 Known 样本几乎全被误拒（`98.6%`），原型同步是维持多轮闭环纯度与低遗忘的核心支柱 |
| **到达顺序敏感性** | `Order A` vs `Order B` | **Order B**：S1 `K=2(raw=2), S_sil=0.535, A_all=86.37%`；S2 `4→1` 合并后 `A_all=72.51%, ΔA_old=5.31%`<br>**Order A**：S1 `1→1 (85.0% core)`；S2 `5→1` 合并并剪枝杂散 FP 簇后 `A_all=72.52%, ΔA_old=3.89%` | 碎片子簇余弦合并使两种极端到达顺序终态均收拢至 `~72.5%` 稳健平台，`Order B` S1 `K=2` 可实现双新类精准无混叠隔离 |

### 多种子统计稳定性汇总（`seed ∈ {42, 1024, 2025}`，`ddof=1`，见 [`outputs/multiseed/summary_multiseed.md`](outputs/multiseed/summary_multiseed.md)）

> 每个种子均独立划分 `Train/Val/Test`、独立从零训练 `TLDNN`、拟合 `MS-MahaProto` 并训练 `AuxSNRRegressor`，全程零跨种子与零测试集泄露。

| 模块 / 基准 | 方法 | 核心指标 (`mean ± std`) | 对照意义 |
| :--- | :--- | :--- | :--- |
| **Stage 1 静态 OSR (`Config A`)** | `MLS` vs **`MS-MahaProto (Ours)`** | AUROC: `56.58 ± 12.65%` vs **`91.29 ± 0.34%`**<br>OSCR: `56.34 ± 12.56%` vs **`90.59 ± 0.32%`** | Logit 基线受神经坍缩影响跨种子剧烈波动，本文方法稳定提升 `+34.7%` 且标准差低至 `0.34%` |
| **盲 SNR 跨调制泛化** | `M2M4` vs **`AuxSNRRegressor (Ours)`** | Valid Unseen MAE: `5.337 ± 0.004` vs **`1.677 ± 0.052 dB`**<br>Gate F1@4dB: `81.45 ± 0.15%` vs **`96.28 ± 0.43%`** | 在未见调制上稳定维持 $< 1.7\text{ dB}$ 误差与 $> 96\%$ 门控 F1 |
| **表 2 `Order B` ($7 \rightarrow 9 \rightarrow 11$)** | `Static` / `Naive` / `iCaRL` / `W/O SS` / **`Full+SS`** | S2 $A_{\text{all}}$: `62.88 ± 0.04` / `18.63 ± 1.70` / `37.32 ± 1.68` / `70.67 ± 2.27` / **`73.20 ± 0.97%`**<br>S2 $\Delta A_{\text{old}}$: **`5.40 ± 0.31%`**（S1 $A_{\text{all}}$ **`85.26 ± 0.96%`**） | 自适应 SS + 原型合并使完整模型 S2 均值领先无 SS **`+2.53%`** 且方差减半 |
| **表 2B `Clean10` ($7 \rightarrow 9 \rightarrow 10$)** | `Static` / `Naive` / `iCaRL` / `W/O SS` / **`Full+SS`** | S2 $A_{\text{all}}$: `69.17 ± 0.05` / `20.16 ± 2.46` / `44.69 ± 4.41` / `78.94 ± 1.01` / **`79.11 ± 1.20%`**<br>S2 $A_{\text{old}}$: **`93.44 ± 0.44%`**, $\Delta A_{\text{old}}$: **`5.37 ± 0.43%`**, 核心纯度 **`95.37 ± 1.03%`** | 排除 `AM-SSB` 缺陷后系统稳定达成 **~80%** 闭环准确率与 **`>95%`** 核心发现纯度 |
| **原型同步消融 (`Order B S2`)** | `W/O Sync` vs **`With Sync (Ours)`** | Known FRR: `98.92 ± 0.74%` vs **`18.53 ± 1.20%`**<br>S2 $A_{\text{all}}$: `53.25 ± 1.63%` vs **`73.20 ± 0.97%`** | 跨三种子一致实证原型同步消除了 98.9% 级已知类边界雪崩 |

---

## 四、 关键开源工具与代码仓库映射

| 技术模块 | 推荐仓库 / 工具 | 应用目的 | 工程落地改造点 |
| :--- | :--- | :--- | :--- |
| **轻量骨干网络** | `qyp2000/TLDNN` | 特征抽取、A/P 变换、SS 数据增强 | 改造并封装为 `BaseBackbone`，暴露 LSTM 末时间步 64 维特征与可扩展分类头 |
| **强特征对比骨干** | `WestdoorSad/IQFormer` | 双模态融合特征空间验证 | 提取 GlobalAvgPool 后特征并映射/对齐至 64 维统一特征空间 |
| **开集评估管线** | `sgvaze/osr_closed_set_all_you_need` | 提取 MLS 得分与 OSCR 评测逻辑 | 移植并向量化 `eval_osr.py`，严格对齐高分代表 Known 的极性规范 |
| **特征边界压缩** | `iCGY96/ARPL` | 互反点损失约束开放空间风险 | 适配 1D 电磁信号特征维度，作为阶段 1/3 严格 ARPL 基线 |
| **连续学习/抗遗忘** | `LAMDA-CL/PyCIL` | Herding 样本筛选与 LwF/iCaRL 基线 | 抽取核心重放与蒸馏逻辑，适配 1D I/Q 波形与原型刷新模块 |
| **聚类与匹配引擎** | `hdbscan` / `scikit-learn` / `scipy` | 无监督新类发现与匈牙利匹配 | 调用 `HDBSCAN` 聚类与 `silhouette_score`，使用 `linear_sum_assignment` 做锁定已知类匹配 |
