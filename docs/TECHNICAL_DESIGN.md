# OS-SE-AMR 核心技术与工程设计规范

本文档将研发规划中的关键实验边界、数学定义、无标签自进化协议及软件模块接口契约固化为可执行标准，确保各阶段代码实现零歧义、无数据/标签泄露、可复现。

---

## 1. 数据集切分与流式到达协议

### 1.1 数据集规格与防泄露分层切分
* **原始数据结构**：`RML2016.10a_dict.pkl` 以 `(modulation, snr)` 为键映射至形状为 `(1000, 2, 128)` 的 `float32` 数组（共 11 种调制样式 $\times$ 20 档信噪比 $-20, -18, \dots, +18\text{ dB}$，总计 220,000 条样本）。
* **信噪比子集划分**：
  * **主实验集（Main Set）**：$\text{SNR} \in \{0, 2, 4, 6, 8, 10, 12, 14, 16, 18\}\text{ dB}$（10 档，每类 10,000 条，共 110,000 条）。
  * **环境压力测试集（Low/Negative-SNR Hold-Out）**：$\text{SNR} \in \{-10, -8, -6, -4, -2\}\text{ dB}$，仅用于盲 SNR 回归头训练（限定已知类）与阶段 3 跨 SNR 漂移鲁棒性评测。
* **分层切分铁律**：对每一个 `(mod, snr)` 单元使用固定随机种子（默认 `seed = 42`）做 **60% Train / 20% Val / 20% Test** 无重叠切分，对应每个 `(mod, snr)` 精确切分为 `600 / 200 / 200` 条样本：
  * **Train 集**：用于阶段 0/1 已知类有监督训练，以及阶段 2/4 流式到达过程中无标签样本池构造（读取时剥离分类标签与真实 SNR 标签）。
  * **Val 集**：Session 0 已知类的 Val 集是**唯一允许**用于标定 OSR 拒识阈值 $\tau_{\text{low}}$ 与 M2M4 偏置校准的验证集合。
  * **Test 集**：跨全生命周期严格只读，严禁参与任何参数更新、聚类发现或阈值选择。

### 1.2 开放度（Openness）精确口径
按标准公式：
$$
\text{Openness} = 1 - \sqrt{\frac{2 \times C_{\text{train}}}{C_{\text{target}} + C_{\text{train}}}}
$$
* **配置 A**：$C_{\text{train}}=7$（`8PSK, BPSK, CPFSK, GFSK, PAM4, QAM16, QPSK`），$C_{\text{unknown}}=4$（`QAM64, WBFM, AM-DSB, AM-SSB`），$C_{\text{target}}=11$，精确开放度为 $1 - \sqrt{14/18} \approx \mathbf{11.81\%}$。
* **配置 B**：$C_{\text{train}}=5$（`8PSK, BPSK, CPFSK, GFSK, PAM4`），$C_{\text{unknown}}=6$（`QAM16, QPSK, QAM64, WBFM, AM-DSB, AM-SSB`），$C_{\text{target}}=11$，精确开放度为 $1 - \sqrt{10/16} \approx \mathbf{20.94\%}$。

### 1.3 多阶段流式到达协议（配置 A：$7 \rightarrow 9 \rightarrow 11$）
RadioML2016.10a 的未知类具有鲜明的电磁领域挑战：`QAM64` 与已知类 `QAM16` 星座高度嵌套属于**近邻未知（Near-OOD）**；而 `WBFM` 与 `AM-DSB` 因原始音频存在大量静音段，在 $N=128$ 短帧下天然存在显著的类间重叠。为此设立两套到达顺序协议：

| 协议版本 | Session 0（有监督底座，$C=7$） | Session 1（第一轮自进化，$7 \rightarrow 9$） | Session 2（第二轮自进化，$9 \rightarrow 11$） | 科学验证目的 |
| :--- | :--- | :--- | :--- | :--- |
| **Order A（默认协议 / 难易分层）** | `8PSK, BPSK, CPFSK, GFSK, PAM4, QAM16, QPSK` | `+ QAM64, AM-SSB` | `+ WBFM, AM-DSB` | 首轮检验 Near-OOD（`QAM64` vs `QAM16`）与模拟调制分离；次轮直面 `WBFM/AM-DSB` 强混淆聚类压力测试 |
| **Order B（顺序敏感性对照）** | 同上 7 类 | `+ QAM64, WBFM` | `+ AM-SSB, AM-DSB` | 将强混淆的 `WBFM` 与 `AM-DSB` 拆分至不同阶段到达，验证先验学得 `WBFM` 能否改善后续 `AM-DSB` 的特征分离度 |

### 1.4 流式样本生成与两种评测模式
`StreamSession` 生成器在每个增量阶段 $t$ 从对应类别的 `Train` 切片中按配置比例抽取无标签信号帧：
* 混合当前已知类的历史数据（默认占比 $\rho_{\text{known}} = 0.5$）与新到达未知类数据（占比 $1 - \rho_{\text{known}} = 0.5$），按批次（如每批 1,000 条）推入推理引擎。
* **Mode 1（阶段 2 解耦调试模式）**：向未知发现模块注入纯净的未知类流，单独压测 `HDBSCAN` 聚类精度、轮廓系数门控与 `LwF` 增量上限。
* **Mode 2（阶段 4 端到端闭环模式）**：全混流经过静态 OSR 拒识器真实拦截后进入 $\mathcal{B}_{\text{unk}}$，包含已知类误拒样本（False Positives）以及漏检的未知样本，全面检验闭环系统在噪声污染下的自进化鲁棒性。

---

## 2. 盲 SNR 估计方案与三阶双感知门控

> **严禁 Oracle 泄露准则**：任何在线推理、拒识缓存过滤、伪标签筛选或增量训练流程均**禁止读取**数据集中附带的真实 SNR 标签。真实 SNR 仅允许在阶段 4 消融实验中作为显式标注的 `Oracle-SNR Upper Bound` 理论参照行出现。

### 2.1 盲 SNR 估计器设计
所有 SNR 估计器实现统一抽象接口：
```python
BaseSNREstimator.predict_snr(x_iq: Tensor[B, 2, 128]) -> Tensor[B]  # 返回 dB 估计值
```

#### 方案 A：M2M4 无训练高阶矩物理基线
将输入 I/Q 帧视为复基带信号 $y[n] = I[n] + j Q[n]$（$n=0,\dots,L-1$，$L=128$），计算二阶与四阶经验矩：
$$
M_2 = \frac{1}{L}\sum_{n=0}^{L-1} |y[n]|^2, \qquad M_4 = \frac{1}{L}\sum_{n=0}^{L-1} |y[n]|^4
$$
在加性高斯白噪声（峰度 $k_w = 2$）与信号独立假设下，分离信号功率 $\hat{S}$ 与噪声功率 $\hat{N}$：
$$
\hat{S} = \sqrt{\max\left(0, \; \frac{2 M_2^2 - M_4}{2 - k_s}\right)}, \qquad \hat{N} = \max\left(\epsilon, \; M_2 - \hat{S}\right), \qquad \widehat{\text{SNR}}_{\text{dB}} = 10 \log_{10}\left(\frac{\hat{S}}{\hat{N}}\right)
$$
* **物理局限与校准**：默认采用恒模假设 $k_s = 1$（适用于 `BPSK/QPSK/8PSK/CPFSK/GFSK`）；对于非恒模调制（`QAM16/64`、`PAM4`、`AM-DSB`），由于真实信号峰度 $k_s > 1$，在 128 点短帧上直接计算会产生系统性负向估计偏置（以及随机采样方差导致的 $\hat{S}=0$ 截断）。工程实现需加入幅度截断保护，并支持基于 Session 0 Val 集拟合全局线性仿射校准参数 $(a, b)$ 以修正系统偏置。

#### 方案 B：轻量化辅助 SNR 回归头（推荐主方案）
* **结构与训练约束**：在 Session 0 骨干特征 $\mathbf{z} \in \mathbb{R}^{64}$ 之上挂载一个轻量级 2 层 MLP 回归头（`64 -> 32 -> 1`, ReLU），或基于冻结特征的微型独立 1D-CNN。**严格限定仅使用 Session 0 已知类样本**（覆盖 `-10 ~ +18 dB` 范围）通过 Smooth-L1 (Huber) 损失完成监督训练，确保训练过程完全不接触任何未知类波形。
* **跨调制泛化评测**：在阶段 4 测试之前，单独统计该回归头在**未见未知类**（`QAM64, WBFM, AM-DSB, AM-SSB`）上的 MAE、RMSE，以及在门控阈值 `0 dB` 处的二分类精确率（Precision）与召回率（Recall）。

### 2.2 三级自进化门控链（阶段 4）
1. **第一级：样本级 OSR + 盲 SNR 双感知门控**
   $$
   \text{Gate}_{\text{sample}}(\mathbf{x}) = \mathbb{I}\Big( S_{\text{OSR}}(\mathbf{x}) < \tau_{\text{low}} \;\land\; \widehat{\text{SNR}}(\mathbf{x}) \ge \tau_{\text{snr}} \Big)
   $$
   其中 $\tau_{\text{low}}$ 默认在 Session 0 Val 集按已知类 $\text{TPR}=95\%$ 标定（默认 $\tau_{\text{snr}} = 0\text{ dB}$）；工程上同时支持采用更严格的收集阈值 $\tau_{\text{strict}}$（例如拒识分位数下界或 $\text{TPR}=98\%$ 对应阈值），以在未知类入库环节主动压低已知类误入（False Positive）比例。
2. **第二级：簇级密度与轮廓系数门控**
   * 对缓冲区 $\mathcal{B}_{\text{unk}}$ 内通过第一级门控的样本提取 L2 归一化特征 $\hat{\mathbf{z}} = \mathbf{z} / \|\mathbf{z}\|_2$。
   * 运行 `HDBSCAN(min_cluster_size=30, min_samples=10, metric="euclidean")`（L2 归一化后的欧氏距离等价于余弦空间度量）。
   * 剔除噪声点（`cluster_id == -1`），在其余非噪点样本上计算全局轮廓系数 $S_{\text{sil}}$ 及单簇平均轮廓系数 $s_k$。
   * **触发新类条件**：有效簇数量 $\ge 1$，每个候选簇有效样本数 $\ge N_{\min}$（默认 30），且整体 $S_{\text{sil}} > 0.45$（或逐簇校验 $s_k > 0.40$）。若不满足，则保留高质量缓冲区样本继续累积至下一流式批次。
3. **第三级：高纯度核心样本筛选（伪标签与 SS 来源）**
   * 在通过检验的新簇 $k$ 中，仅提取 HDBSCAN 隶属概率 $p_{\text{mem}}(\mathbf{x}) \ge 0.75$（或距离簇质心最近的 Top $60\%$）样本构成高纯度核心集 $\Omega_k^{\text{core}}$，用于后续簇内 SS 扩充、原型初始化及增量微调。

---

## 3. 匈牙利评测协议与全套指标定义

### 3.1 阶段 1 静态开集指标
统一约定打分器输出 $S(\mathbf{x})$ 满足 **值越大表示越偏向已知类（Known = 1, Unknown = 0）**：
* **$\text{Acc}_{\text{known}}$**：在 Known Test 集上的标准闭集分类准确率。
* **AUROC**：以所有测试样本的 $S(\mathbf{x})$ 计算 Known(1) vs Unknown(0) 的 ROC 曲线下面积。
* **FPR@TPR95**：在 Known 检出率（TPR）精确达到 $0.95$ 的阈值点（ROC 线性插值）上对应的 Unknown 误判率（FPR）。
* **OSCR**：遍历所有候选阈值 $\tau$，计算横轴 $\text{FPR}(\tau)$（Unknown 得分 $\ge \tau$ 的比例）与纵轴正确分类率 $\text{CCR}(\tau)$（Known 得分 $\ge \tau$ 且分类标签正确的比例）构成的曲线下面积。

### 3.2 阶段 2/4 锁定已知类的匈牙利 1-to-1 匹配协议
无监督聚类生成的新分类头下标 $\{C_{\text{old}}, \dots, C_{\text{old}} + \hat{K} - 1\}$ 与数据集原始类别编号没有自然对应关系，严禁随意置换已知类标签。本协议确立**锁定已知类（Locked-Known）约束**：

1. **已知类头锁定**：基座已知类头 $\{0, \dots, C_{\text{old}} - 1\}$ 与其真实调制标签永久绑定，**绝不参与**匈牙利算法重排。
2. **新增簇头最优匹配**：在测试集的新增未知类子集 $\mathcal{D}_{\text{test}}^{\text{new}}$（含真实标签集 $\mathcal{Y}_{\text{new}}$，类别数 $K_{\text{true}}$）上执行前向推理，仅统计预测落入新增头 $\mathcal{K}_{\text{pred}} = \{C_{\text{old}}, \dots, C_{\text{old}} + \hat{K} - 1\}$ 的样本，构建 $\hat{K} \times K_{\text{true}}$ 列联混淆矩阵：
   $$
   W_{i,j} = \sum_{\mathbf{x} \in \mathcal{D}_{\text{test}}^{\text{new}}} \mathbb{I}\big[\hat{y}(\mathbf{x}) = C_{\text{old}} + i \;\land\; y(\mathbf{x}) = y_j\big]
   $$
   调用 `scipy.optimize.linear_sum_assignment(-W)` 求解最大化匹配总命中数的 1-to-1 映射 $\pi$。
3. **欠聚类与过聚类处理口径**：
   * **欠聚类（$\hat{K} < K_{\text{true}}$）**：未能获得匹配簇头的真实新类计为 $0\%$ 识别率，如实反映在 $A_{\text{new}}$ 与 $A_{\text{all}}$ 惩罚中。
   * **过聚类（$\hat{K} > K_{\text{true}}$）**：主评测口径（**Strict 1-to-1**）中未匹配的多余簇头预测一律判错；同时在实验日志中附带记录无监督聚类诊断指标：`NMI`、`ARI`、`Many-to-1 Purity` 以及簇数量误差 $\Delta K = |\hat{K} - K_{\text{true}}|$。

### 3.3 自进化准确率与遗忘率 $FM$ 定义
在阶段 $t$ 全量测试集（覆盖截至阶段 $t$ 已出现的全部 $C_t$ 个类别）上进行端到端全分类头预测（已知样本若被误分至新类头同样计错）：
* **$A_{\text{old}}^{(t)}$**：Session 0 最初的 $C_0$ 个已知类在全 $C_t$ 头下的平均准确率。
* **$A_{\text{new}}^{(t)}$**：截至阶段 $t$ 累计引入的 $(C_t - C_0)$ 个新增类在经锁定已知类匈牙利映射后的平均准确率。
* **$A_{\text{all}}^{(t)}$**：全量 $C_t$ 个类别的总准确率。
* **灾难性遗忘指标（双口径同时记录）**：
  1. **基础类准确率跌幅（$\Delta A_{\text{old}}^{(t)}$，对应规划表 2 的 $FM$ 列）**：
     $$
     \Delta A_{\text{old}}^{(t)} = A_{\text{old}}^{(0)} - A_{\text{old}}^{(t)}
     $$
  2. **标准逐类平均遗忘率（$FM_{\text{cls}}^{(t)}$，参考增量学习标准文献）**：
     $$
     FM_{\text{cls}}^{(t)} = \frac{1}{C_0} \sum_{c=1}^{C_0} \max_{s \in \{0,\dots,t-1\}} \left( \text{Acc}_{s,c} - \text{Acc}_{t,c} \right)
     $$

---

## 4. 抗遗忘增量、原型同步与 TLDNN-SS 电磁增强设计

### 4.1 Replay Buffer 与双层 LwF 蒸馏
* **Replay Buffer ($\mathcal{M}_{\text{old}}$)**：按每个旧类固定存储 30 条原始 `(2, 128)` I/Q 波形（总占用内存极小），默认采用 **iCaRL Herding** 在 $\mathbb{R}^{64}$ 特征空间选取最贴近类质心的代表样本，并与随机采样策略做对照。
* **增量联合损失**：微调阶段冻结上一阶段教师模型参数 $\theta_{\text{old}}$，对当前新类伪标签集与 $\mathcal{M}_{\text{old}}$ 重放样本混合构造小批量（Mini-Batch），优化：
  $$
  \mathcal{L}_{\text{inc}}(\theta) = \mathcal{L}_{\text{CE}}\big(g_{0:C_t}(\mathbf{x}; \theta), \tilde{y}\big) + \beta_{\text{feat}} \|\mathbf{z}_{\theta}(\mathbf{x}) - \mathbf{z}_{\theta_{\text{old}}}(\mathbf{x})\|_2^2 + \beta_{\text{logit}} \, T^2 \, D_{\text{KL}}\Big(\sigma(g_{0:C_{\text{old}}}^{\theta_{\text{old}}}/T) \;\big\|\; \sigma(g_{0:C_{\text{old}}}^{\theta}/T)\Big)
  $$
  默认蒸馏温度 $T = 2.0$，新类样本可根据聚类隶属度 $p_{\text{mem}}(\mathbf{x})$ 加权或统一加权。

### 4.2 增量后原型同步与阈值重校准（Anti-Drift Sync）
即使在 LwF 特征蒸馏约束下，骨干网络权重的微调仍会导致旧类嵌入中心发生轻微偏移，进而破坏基于原型距离的后续 OSR 判决。因此每次增量更新后必须执行两步无梯度同步：
1. **旧类原型刷新**：用更新后的骨干对 $\mathcal{M}_{\text{old}}$ 全部样本前向提取特征，重新计算每个旧类原型 $\boldsymbol{\mu}_c^{\text{sync}}$；同时用核心集 $\Omega_k^{\text{core}}$ 初始化新类原型 $\boldsymbol{\mu}_{\text{new}}$。
2. **OSR 阈值重校准**：在 Session 0 Val 集及当前新类核心样本/重放集上重算已知类得分分布，更新下一流式阶段使用的 $\tau_{\text{low}}$。

### 4.3 簇内 TLDNN-SS 扩充及拼接边界消融设计
为避免跨类拼接污染，**SS 操作严格限制在单个已认证核心簇 $\Omega_k^{\text{core}}$ 内部**，随机抽取不同样本对 $(\mathbf{x}_i, \mathbf{x}_j)$（$i \neq j$）按 $L/2 = 64$ 拼接，将每个新簇样本扩充至 $3\times \sim 5\times$。

针对 `CPFSK`、`GFSK`、`WBFM` 等连续相位/调频信号在时域中点直接拼接可能引入瞬时相位跳变与伪频脉冲的问题，代码库实现并对比两种 SS 变体：
1. **`SS-IQ`（标准时域片段替换基线）**：
   $$
   \tilde{\mathbf{x}}_{\text{IQ}} = \text{Concat}\big(\mathbf{x}_i[:, 0:64], \; \mathbf{x}_j[:, 64:128]\big)
   $$
   拼接后再经由标准化管道送入 TLDNN（内部转为幅度/相位 A/P 特征）。
2. **`SS-PhaseAligned`（边界相位连续对齐改进版）**：
   计算接缝点两侧复基带瞬时相位差 $\Delta\theta = \angle z_i[63] - \angle z_j[64]$（其中 $z[n] = I[n] + j Q[n]$），将后半段信号乘以复旋转因子 $e^{j\Delta\theta}$ 实现接缝处零阶相位连续：
   $$
   \tilde{z}[n] = \begin{cases} z_i[n], & 0 \le n < 64 \\ z_j[n] \cdot e^{j\Delta\theta}, & 64 \le n < 128 \end{cases}
   $$
   再分解为 `(I, Q)` 并入后续特征抽取与微调流程，从物理波形层面消除拼接导致的异常相位导数尖峰。

---

## 5. 仓库架构、统一接口契约与工程规范

### 5.1 规范化工程目录树
与根目录 [`README.md`](../README.md#三-仓库架构设计总览) 保持完全一致：`configs/` 管理切分与五阶段配置；`src/os_se_amr/` 划分为 `data`、`models`、`osr`、`discovery`、`continual`、`engine`、`eval`、`utils` 八大内聚子包；`scripts/` 提供各阶段命令行入口；`tests/` 覆盖关键数学与防泄露断言。

### 5.2 核心 Python 接口契约

为保证 `TLDNN` 与 `IQFormer`、多种 OSR 打分器及盲 SNR 估计器可热插拔组合，各模块必须遵守以下类型签名与张量不变量：

```python
class BaseBackbone(nn.Module):
    feature_dim: int = 64

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: 原始 I/Q 输入张量，shape=(B, 2, 128), dtype=float32
        Returns:
            logits: 当前已知分类头未归一化对数概率, shape=(B, C_current)
            features: L2 可归一化或标准化前 64 维隐特征 z, shape=(B, 64)
        """

    def expand_classifier(
        self, num_new_classes: int, init_weights: Optional[Tensor] = None
    ) -> None:
        """在不改变现有 0..C_old-1 权重的前提下，原子化扩展分类头输出维度。"""


class BaseOSRScorer(ABC):
    def fit(self, features: Tensor, labels: Tensor) -> None:
        """基于训练集特征或原型初始化内部统计量。"""

    def score(self, logits: Tensor, features: Tensor) -> Tensor:
        """
        极性不变量：返回 shape=(B,) 的标量得分，值越大越偏向已知类（Known），越小越偏向未知类（Unknown）。
        """


class BaseSNREstimator(ABC):
    def predict_snr(self, x_iq: Tensor) -> Tensor:
        """
        Args:
            x_iq: 原始 I/Q 波形（非归一化前物理尺度或固定协议尺度），shape=(B, 2, 128)
        Returns:
            snr_db: 每条样本的盲估计信噪比（dB），shape=(B,)
        """


class UnknownDiscoverer(ABC):
    def discover(
        self, features: Tensor, x_iq: Tensor, snr_est: Tensor
    ) -> "DiscoveryResult":
        """
        执行门控筛选、HDBSCAN 密度聚类、轮廓系数验证及核心集提取。
        返回 DiscoveryResult(accepted, cluster_ids, membership_probs, silhouette, prototypes, core_mask)。
        """


class ContinualLearner(ABC):
    def evolve(
        self,
        backbone: BaseBackbone,
        replay_buffer: "ReplayBuffer",
        new_samples: Tensor,
        new_pseudo_labels: Tensor,
        sample_weights: Optional[Tensor] = None,
    ) -> "EvolutionReport":
        """执行增量微调、双层蒸馏与原型刷新，返回训练损失与原型偏移统计。"""
```

### 5.3 实验可复现性与工程卫生规范
1. **全局确定性控制**：所有实验入口强制调用 `seed_everything(seed)` 固定 `random`、`numpy`、`torch`、`torch.cuda` 种子（主实验默认跑 `seed in {42, 1024, 2025}` 报告均值 $\pm$ 标准差）。
2. **标准化输出落盘**：每次运行自动在 `outputs/<stage>/<exp_id>/` 下生成完整快照：
   * `config.yaml`（解析后的完整参数）
   * `metrics.json`（严格结构化评测指标，含 AUROC、OSCR、匈牙利映射表、$\Delta A_{\text{old}}$ 与 $FM_{\text{cls}}$）
   * `checkpoints/`（阶段模型与原型库）
   * `figures/`（OSCR 曲线、t-SNE/UMAP 聚类演化图、混淆矩阵）
3. **强制单元测试门槛**：任何算法代码提交前须通过 `pytest tests/`，重点保障开放度解析值校验、Split 零重叠断言、AUROC/OSCR 极性校验及锁定已知类匈牙利匹配算法在过聚类/欠聚类场景下的正确性。
