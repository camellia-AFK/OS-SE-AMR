# OS-SE-AMR：开放集增量自进化电磁调制识别系统

**OS-SE-AMR**（Open-Set Self-Evolving Automatic Modulation Recognition）面向开放复杂电磁环境下的自动调制识别（AMR）任务，旨在同时突破传统深度学习识别模型的双重核心瓶颈：

1. **开放集未知拒识能力缺失**：面对训练集中未曾出现过的新型调制样式（Unknown Modulations），封闭集 Softmax 分类器易产生高置信度误判。
2. **持续学习中的灾难性遗忘与极小样本过拟合**：在流式捕获未知信号并拓展分类边界时，直接微调模型会导致已有调制样式识别率剧烈衰退，同时初期截获的未知样本量较少且伴随低信噪比干扰，极易过拟合。

---

## 一、 总体技术路线图

本项目采用 **“基础设施搭设 $\rightarrow$ 静态开集拒识 $\rightarrow$ 动态解耦复现 $\rightarrow$ 基线对齐测试 $\rightarrow$ 闭环融合与创新”** 的五阶段渐进式研发架构：

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
           │ (SSD-AMC / iCaRL / SNR 漂移)   │
           └────────────────┬───────────────┘
                            │
                            ▼
           ┌────────────────────────────────┐
           │ 阶段 4: 自进化闭环与 SS 破局   │
           │ (TLDNN-SS + 盲 SNR 双感知门控) │
           └────────────────────────────────┘
```

---

## 二、 文档导航

| 文档 | 核心内容 |
| :--- | :--- |
| [`ROADMAP.md`](ROADMAP.md) | 5 阶段研发里程碑、经数学校准的开放度划分矩阵、阶段通过门槛、实验对比记录表模板与开源仓库映射 |
| [`docs/TECHNICAL_DESIGN.md`](docs/TECHNICAL_DESIGN.md) | 数据集分层切分与防泄露铁律、多阶段流式到达协议（Order A/B）、盲 SNR 估计方案（M2M4 与辅助回归头）、锁定已知类的匈牙利评测协议、原型同步与 TLDNN-SS 拼接消融设计、模块接口契约 |

---

## 三、 仓库架构设计总览

整个工程采用配置驱动（YAML-driven）与接口解耦设计，规划目录结构如下：

```text
OS-SE-AMR/
├── README.md                          # 项目总览与工程导航
├── ROADMAP.md                         # 五阶段研发规划与里程碑清单
├── pyproject.toml                     # 依赖管理与包安装配置
├── .gitignore                         # 忽略数据集、模型权重与实验输出
├── configs/                           # YAML 实验配置文件
│   ├── splits/
│   │   ├── config_a.yaml              # 配置 A：7 已知 + 4 未知 (Openness=11.81%)
│   │   ├── config_a_clean10.yaml      # 配置 A-Clean10：7 已知 + 3 有效未知 (Openness=9.25%)
│   │   └── config_b.yaml              # 配置 B：5 已知 + 6 未知 (Openness=20.94%)
│   ├── stage0_base.yaml               # 阶段 0：骨干封闭集训练与特征校验
│   ├── stage1_osr.yaml                # 阶段 1：MLS / ARPL / Distance-Prototype 对比
│   ├── stage2_discovery_cil.yaml      # 阶段 2：未知发现与抗遗忘解耦测试
│   ├── stage3_baselines.yaml          # 阶段 3：跨基线对齐与 SNR 漂移评估
│   └── stage4_closed_loop.yaml        # 阶段 4：自进化闭环与消融实验
├── data/                              # 原始数据集目录（Git 忽略）
│   └── RML2016.10a_dict.pkl           # RadioML2016.10a 原始数据文件
├── docs/
│   └── TECHNICAL_DESIGN.md            # 核心算法与评测协议详细设计规范
├── src/
│   └── os_se_amr/
│       ├── __init__.py
│       ├── data/                      # 数据加载、开放集划分与流式仿真
│       │   ├── rml2016.py             # RML2016.10a 加载与 SNR 过滤
│       │   ├── splitter.py            # 分层切分与开放度精确计算
│       │   ├── stream.py              # 多阶段无标签流式批次生成器
│       │   └── transforms.py          # I/Q 标准化与 A/P（幅度/相位）转换
│       ├── models/                    # 骨干网络与辅助估计器
│       │   ├── base.py                # BaseBackbone 与 BaseSNREstimator 抽象基类
│       │   ├── tldnn.py               # TLDNN 轻量骨干（暴露 64 维标准化特征）
│       │   ├── iqformer.py            # IQFormer 对比骨干（GlobalAvgPool 特征）
│       │   ├── heads.py               # 线性分类头、余弦原型头与可扩展头
│       │   └── snr_estimator.py       # M2M4 物理矩估计与轻量化辅助 SNR 回归头
│       ├── osr/                       # 静态开集拒识评分与损失约束
│       │   ├── softmax.py             # Softmax Maximum Probability 基线
│       │   ├── mls.py                 # Maximum Logit Score (MLS)
│       │   ├── arpl.py                # 互反点学习损失与打分 (ARPL)
│       │   ├── prototype.py           # 类内原型紧凑损失与距离原型打分
│       │   └── threshold.py           # 基于 Val 集的 TPR95/TPR98 阈值标定
│       ├── discovery/                 # 未知发现与伪样本扩充
│       │   ├── buffer.py              # 拒识样本缓冲区 B_unk 管理
│       │   ├── hdbscan_cluster.py     # HDBSCAN 密度聚类与轮廓系数门控
│       │   ├── gating.py              # 置信度 + 盲 SNR 双感知采样门控
│       │   └── ss_augment.py          # 簇内 TLDNN-SS（SS-IQ / SS-PhaseAligned）
│       ├── continual/                 # 抗遗忘增量学习与对照方法
│       │   ├── replay.py              # Herding / 随机回放缓冲区 M_old
│       │   ├── lwf.py                 # 特征 + Logits 双层 LwF 蒸馏约束
│       │   ├── icarl.py               # iCaRL 基线实现
│       │   ├── simplecil.py           # SimpleCIL 基线实现
│       │   ├── finetune.py            # Naive Fine-Tune 基线实现
│       │   └── prototype_sync.py      # 增量微调后旧类原型刷新与阈值重校准
│       ├── engine/                    # 训练与自进化调度引擎
│       │   ├── trainer.py             # 阶段 0/1 有监督与 OSR 训练循环
│       │   └── evolver.py             # 阶段 2/4 流式自进化闭环调度器
│       ├── eval/                      # 统一评测指标与协议
│       │   ├── osr_metrics.py         # AUROC、FPR@TPR95、OSCR 计算
│       │   ├── hungarian.py           # 锁定已知类的匈牙利 1-to-1 簇匹配评测
│       │   ├── cil_metrics.py         # A_all、A_old、A_new、FM 与辅助聚类指标
│       │   └── robustness.py          # DOCO 思想跨 SNR 漂移鲁棒性评测
│       └── utils/                     # 工程基础工具
│           ├── seed.py                # 全局确定性随机种子固定
│           ├── config.py              # YAML 配置解析与合并
│           ├── logger.py              # 结构化日志与实验指标落盘
│           └── checkpoint.py          # 模型权重与原型库保存/加载
├── scripts/                           # 一键复现与评测入口脚本
│   ├── run_stage0.py
│   ├── run_stage1.py
│   ├── run_stage2.py
│   ├── run_stage3.py
│   ├── run_stage4.py
│   ├── run_multiseed.py               # 优先级 2：多种子（42/1024/2025）独立统计评估
│   └── eval_all.py
├── tests/                             # 核心逻辑单元测试
│   ├── test_splitter_openness.py      # 数据零泄露与 Openness 公式校验
│   ├── test_osr_metrics.py            # AUROC / FPR@TPR95 / OSCR 边界校验
│   ├── test_hungarian.py              # 锁定已知类的匈牙利匹配与过/欠聚类校验
│   ├── test_snr_estimator.py          # M2M4 与辅助 SNR 回归头接口校验
│   └── test_ss_augment.py             # TLDNN-SS 簇内约束与相位对齐校验
└── outputs/                           # 实验日志、指标 JSON、权重与图表（Git 忽略）
```

### 核心模块与研发阶段对应关系

| 子包路径 | 核心职责 | 主要服务阶段 |
| :--- | :--- | :--- |
| `os_se_amr.data` | RML2016.10a 读取、分层防泄露切分、流式到达队列与 A/P 变换 | 全阶段 |
| `os_se_amr.models` | TLDNN / IQFormer 统一 64 维特征暴露、可扩展分类头与盲 SNR 估计 | 阶段 0、1、4 |
| `os_se_amr.osr` | 统一“高分=已知”极性的 OSR 打分器、ARPL/原型损失与阈值标定 | 阶段 1、3、4 |
| `os_se_amr.discovery` | 拒识样本缓存、盲 SNR/置信度门控、HDBSCAN 聚类与簇内 SS 扩充 | 阶段 2（主线）、4 |
| `os_se_amr.continual` | Herding 重放缓存、双层 LwF 蒸馏、原型同步刷新与增量对照算法 | 阶段 2（支线）、3、4 |
| `os_se_amr.eval` | AUROC / OSCR、锁定已知类匈牙利匹配、遗忘率 FM 与 SNR 漂移评测 | 阶段 1、2、3、4 |

---

## 四、 当前研发状态

- [x] **前期设计**：完成五阶段研发路线梳理，并沉淀方案设计文档（`README.md`、`ROADMAP.md`、`docs/TECHNICAL_DESIGN.md`）。
- [x] **阶段 0**：基础设施搭建、数据集复制、开放度划分器与 TLDNN（配置 A 封闭集 Test Acc **`98.76%`**）/ IQFormer（配置 A 封闭集 Test Acc **`98.94%`**）骨干适配与单元测试全通过。
- [x] **阶段 1**：静态开集拒识基线与指标固化完成；揭示 Logit/欧氏基线在模拟调制上的各向异性坍缩问题，升级提出 **MS-MahaProto**（`TLDNN` 达成 **AUROC `91.68%` / OSCR `90.96%` / FPR@TPR95 `26.80%`**，`IQFormer` 达成 **AUROC `91.44%` / OSCR `90.76%` / FPR@TPR95 `25.96%`**，8/8 单测全通过）。
- [x] **阶段 2**：未知发现与抗遗忘增量双线解耦完成；主线 HDBSCAN（`stem_preln` 空间）在协议 A S1 达成 **NMI `0.970` / 1-to-1 Acc `99.65%` / 核心纯度 `99.59%`**、协议 B S1/S2 均达成 **NMI `1.000` / 1-to-1 Acc `100.00%`**；支线 **Dual-LwF + BiC（m=100）** 在协议 A（7→9）达成 **`A_all 90.14%` / `A_old 94.06%` / `A_new 76.45%` / `ΔA_old 4.70%`**，协议 B（7→9→11）最终达成 **`A_all 78.22%` / `A_old 93.16%` / `A_new 52.06%` / `ΔA_old 5.59%`**（远优于 Naive Fine-Tune 的 `A_all 17.36%` / `ΔA_old 98.59%`），11/11 单测全通过。
- [x] **阶段 3**：跨领域静态/增量基线补齐与 SNR 环境漂移评估完成；补齐 **SSD-AMC-Open-set**（配置 A AUROC `82.07%`）与高开放度**配置 B（Openness `20.94%`）**泛化矩阵，**`MS-MahaProto`** 达成 **AUROC `91.73%` / OSCR `91.50%` / FPR@TPR95 `23.49%`**；增量横向对比中 **`Dual-LwF + BiC (m=100)`** 在协议 A/B 均领先 `Static` / `Naive` / `SimpleCIL` / `iCaRL`（`C=11` `A_all` 分别达 **`79.58%`** 与 **`78.22%`**）；DOCO 跨 SNR 漂移实验验证 **High-SNR（≥10 dB）** 下 **`90.79%`** 与 **Mid-Low（0–6 dB）** 下 **`76.26%`** 的稳健优势及 **Prototype Refresh** 对抑制 OSR 边界崩塌（FRR 从 `90.32%` 降至 `5.19%`）的必要性，15/15 单测全通过。
- [x] **阶段 4（含优先级 1 闭环精修 + Clean10）**：端到端闭环融合、盲 SNR 双感知门控、过分裂子簇语义合并与质量/规模自适应 `TLDNN-SS` 完成；`AuxSNRRegressor` 在 Known Test 达 **MAE `0.94 dB` / F1 `96.26%`**、有效未见未知类（`QAM64/WBFM/AM-DSB`）达 **MAE `1.63 dB` / F1 `96.76%`**；真实混流闭环（`seed=42`）在 `Order A` / `Order B` / `Clean10` 上 **`Ours (Full Model + SS)` 在 S1 与 S2 均全面严格超越 `Ours (W/O SS)` 及 `Static` / `Naive` / `iCaRL`**——`Order B` S1 **`86.37%`** -> S2 **`72.51%`（`A_old 93.44%`, `ΔA_old 5.31%`, `4→1` 合并）**、`Order A` S2 **`72.52%`（`A_old 94.86%`, `ΔA_old 3.89%`, `5→1` 合并）**、`Clean10`（无 `AM-SSB` 缺陷，`Openness=9.25%`）达成 **`79.78%`（`A_old 93.59%`, `ΔA_old 5.16%`, S1 `ΔK=0`, S2 `ΔK=0`）**。
- [x] **多种子统计验证（优先级 2）**：完成 `seed ∈ {42, 1024, 2025}` 无跨种子泄露全链路（Stage 0 → 1 → 4 → Clean10 → 原型同步消融）统计验证；`MS-MahaProto` 获 **AUROC `91.29 ± 0.34%` / OSCR `90.59 ± 0.32%`**（对比 `MLS` `56.58 ± 12.65%`），`Order B` S2 获 **`73.20 ± 0.97%`**（超越 `W/O SS` `70.67 ± 2.27%`），`Clean10` S2 获 **`79.11 ± 1.20%`（核心纯度 `95.37 ± 1.03%`，`ΔA_old 5.37 ± 0.43%`）**，`eval_all.py` 已汇总全基准报告（**23/23** 项单元测试全通过）。
