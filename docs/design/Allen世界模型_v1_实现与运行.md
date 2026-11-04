# Allen 世界模型 v1：实现、验证与服务器运行

> 后续讨论修订：本文件的“物种→类群汇总”L2 没有建立原 H-JEPA 的时间层级对应，保留为历史实验，不再推荐运行此处的双层命令。当前候选与过夜入口见 `Allen世界模型_v2_候选与过夜实验.md`。v2 明确比较形态输出结构，不把它称为原生 H-JEPA 时间层级。

2026-11-04。状态：开发版本；本地接口与计算检查通过，尚未完成服务器模拟恢复或真实数据训练。本文是本轮前瞻性开发记录，不称为公开预注册，也不把已有 Allen 分析重新称为独立验证。

## 科学问题和层级选择

主问题：在控制体重、发育方式及近缘形态背景后，温度与运动信息是否帮助预测鸟腿形态；模型的条件响应是否与观测到的发育方式差异、骨段差异相容。

- **L1 主模型：物种之间的条件形态预测。** 状态为外测跗跖及 Eliason 股骨、胫跗骨、跗跖骨的 log10 长度和观察掩码。条件包括来源种及目标种的温度、运动、体重、发育方式等，及两者差值、观察掩码和树路径距离。
- **L2 可选实验：近缘单系类群的形态汇总预测。** 在原树上按冠群年龄不超过 20 Myr 的最大单系群划分，划分之后才过滤训练／留出成员。20 Myr 是本轮固定的开发尺度，不是已证明最佳尺度。最多抽 16 个成员；本次状态与条件采用相同抽样成员。
- 不启用旧的“运动方式 → 陆水空”分组，也不把温度、发育方式、运动方式分成上下层。这些因素彼此交叉；按运动方式分组会使“遮蔽运动信息”后仍从分组结构得到运动信息。
- 叶序与树路径距离不是实测祖先后代轨迹。L2 也不是恢复出的祖先状态或更长时间的实测轨迹。
- 先评价 L2 联合训练是否改善**同一批留出物种的 L1 预测与条件响应**。不能拿类群平均误差直接和单物种误差比较来宣称架构胜出。若无增益，文章采用 L1，保留 H-JEPA 技术来源和未来扩展接口。

这种安排让模型服务于 Allen 主线：物种层的形态响应是否能够跨类群成立。模型层级不预设“远端强、近端弱”，也不把零响应或正响应写进模型损失。力学仍是辅助验证，不参与本轮训练或模型优选。

## 保留源码与变更边界

直接扩展现有 `EvoWM`，没有另行 clone，没有覆盖 Claude 的分析、旧配置、稿件和结果。单独的目录会分裂当前 Git 同步路径，因此本轮采用新的 `allen_world_v1_*` 配置、`scripts/allen_world/` 和独立输出目录。

复用上游 JEPA/HJEPA 对象、MLP 编码器、Transformer 预测器、SIGReg、原训练入口，以及已有的残差预测和单位平均扩展。上游只增加两个 opt-in 接口：

1. `hjepa/h_jepa/data.py`：`evowm_allen` 数据类型。
2. `hjepa/h_jepa/main_hjepa.py`：Allen 配置显式打开时，在模型创建前设随机种子；旧配置无变化。

变更已登记 `hjepa/UPSTREAM.md`。没有重新实现一套冒充原架构的网络。

## 数据口径

输入生成在 `outputs/allen_world_v1/input/`，不改已有 `data/processed/species_traits.csv`。

| 内容 | 处理 |
|---|---|
| 温度 | bio1、bio5、bio6，保留原始 °C，在每折训练物种上标准化；同时提供来源、目标和差值 |
| 运动信息 | AVONET 5 个生活方式哑变量 + 7 个觅食层比例；这些是生态代理，不等于实测步态或机械负荷 |
| 固定背景 | log10 体重、P/SP/SA/A 四个发育方式哑变量、栖境密度、迁徙等级 |
| 形态输出 | 外测跗跖 + Eliason 的三骨，比例由预测的两段 log 长度相减得到 |
| 发育编码 | 沿用按科赋值及雀形目=A 的既有口径；不是个体生长测量 |
| 骨长来源 | Eliason 单独接入；不把 Skelevision、Watanabe、Eliason 的骨长直接拼成同一无来源列 |
| 留出 | 沿用整科五折；所有历史物种、目标物种和 L2 成员均属于各自 split；归一化和读数探针只拟合训练物种 |

本地输入包含 9,993 行；满足体重、外测跗跖、三温度和发育方式的 9,172 种，其中 Eliason 三骨完整 1,096 种、168 科、40 目。原 EvoWM 的 k_* 骨数据在当前匹配子集中主要为雀形目与鸭科，来源、目和发育方式混杂，因此不承担本次广泛发育方式比较。

Eliason 加入训练后，不能再称为模型独立外部验证。已有生物学统计仍保留其原来的数据来源说明；模型结果必须来自留出科。新表没有插补缺失骨长。

## 一个模型的四种输入条件

一个模型在训练时以各 25% 概率见到：全部条件、遮蔽温度、遮蔽运动、两者均遮蔽。背景和距离始终提供。遮蔽同时清除该组的来源值、目标值、差值、观察掩码和 offered 标志，区别“未知”和“已知恰好为零”。L1/L2 对同一样本使用相同遮蔽。

这些是**同一组参数**的四种输入，不是四个真实数据主模型。它们比较新增信息的预测价值；由于因素相关且来源形态已有历史信息，不能把贡献换算成各因素的纯因果百分比。没有加入 X 的重复通道；X 只是已有觅食比例之和，留作审计。

状态 8 维（4 值 + 4 掩码）；动作 115 维（22 条件列 × 5 + 3 组 offered 标志 + 2 距离）。残差预测器、128 维嵌入、原正则和原训练损失保持原实现。

## 评估范围与进入主分析的条件

`evaluate.py` 在每个 H+1 窗口只评分最后目标；目标形态不进入预测器。训练物种上拟合 Ridge 读数探针；同输入历史、同四种遮蔽训练的 Ridge 为简单对照，另有复制前一种形态的对照。提供每种方法可用样本和所有方法共同支持样本两个口径，先在物种内平均平方误差，再在科内平均，再对科等权；同时输出窗口数、独立物种数和科数。

条件温度诊断同时把最终目标的三温度升／降 1°C，固定体重、发育、运动和来源形态，计算每 10°C 响应。报告 P+SP 与 SA+A，及同一目标上的跗跖骨−胫跗骨、胫跗骨−股骨、跗跖骨−股骨响应差。这是三温度平行移动，不等于 bio1 单独系数，也不等于冷热端分工检验。

当前只检查扰动温度是否在训练集各温度边际范围内，**没有把边际范围当作联合环境支持或因果识别**。输出为开发诊断，无窗口伪重复置信区间；正式论文所需的跨折、跨种子、按科配对区间和多树检验需在模拟校准结果复核后固定。

模拟提供 null / temperature / motion / joint 四种已知条件效应，叠加树上布朗残差和测量噪声，保留实际缺失模式。每个场景是独立的合成数据集，不是要求真实任务训练四个相互竞争的解释模型。生成模型独立于 JEPA；系数、树和输入哈希写入 truth.json。它测试可恢复性，不宣称该生成过程就是鸟类演化。

开发检查通过不自动放行真实主分析。首先检查零温度场景是否虚报温度响应、温度场景是否恢复发育调节和骨段差、运动单独场景是否错误归因给温度，并检查 L2 是否在匹配 L1 任务上有稳定增益。当前脚本保留失败与零结果，没有“有利才输出”的分支；正式数值判据和锁定分析需在本轮开发后另行记录，不能事后声称预注册全部通过。

## 同步到 GitHub

以下在本地 PowerShell 执行。只暂存本轮代码和设计，避免把尚在阅读的 v4/v5 稿件混入提交；若另有已暂存修改，先在 `git diff --cached --stat` 中检查。

```powershell
Set-Location C:\Users\ZihanWANG\Desktop\FFY\EvoWM
git add -- src/evowm/data/allen.py src/evowm/data/allen_hierarchy.py scripts/allen_world tests/test_allen_dataset.py tests/test_allen_hierarchy.py tests/test_allen_table.py tests/test_allen_evaluate.py configs/hjepa/allen_world_v1_l1.yaml configs/hjepa/allen_world_v1_l2.yaml hjepa/h_jepa/data.py hjepa/h_jepa/main_hjepa.py hjepa/UPSTREAM.md "docs/design/Allen世界模型_v1_实现与运行.md"
git diff --cached --stat
git commit -m "Add Allen JEPA conditions and optional phylogenetic hierarchy"
git push
```

若 Git 报 dubious ownership，使用逐命令 `git -c safe.directory=C:/Users/ZihanWANG/Desktop/FFY/EvoWM ...`，不必修改全局 Git 配置。助手本轮未提交或推送。

## 服务器先做端到端短检查

在现有 evowm 环境、仓库根目录运行。先确认可用卡，示例 `--gpus 0` 必须换成实际空闲卡。最多四张卡；CPU 检查最多 32 个任务（本次仅 10 项所以最多实际 10 个），评估默认 32 个进程，每进程 1 数值线程，避免 32×8 过度并行。

```bash
cd /mnt/zihanw/EvoWM
git pull --ff-only
python scripts/extract_trees.py
python scripts/allen_world/build_table.py
python scripts/allen_world/check.py --workers 32 --model-smoke
nvidia-smi
python scripts/allen_world/run_simulation.py --gpus 0 --folds 0 --seeds 13 --scenarios null joint --steps 20 --eval-workers 4
```

最后一条只跑 4 个短任务（2 模拟场景 × L1/L2），用于验证服务器完整训练、保存和评估链路，不判断生物学恢复。日志和输入存于打印的 `outputs/allen_world_sim_时间戳/`，任何失败返回非零退出码；失败时不静默进入下一阶段。

短检查成功后，完整开发模拟命令如下。默认四个模拟场景、两个随机种子、五折、两种层级，共 80 个训练任务；一次最多四个任务并发，不是 80 个模型同时占卡。每个模型 4,000 步。没有可靠实测前不估计总耗时。

```bash
python scripts/allen_world/run_simulation.py --gpus 0 1 2 3 --eval-workers 32 --loader-workers 8 --threads-per-job 4
```

所有训练依旧调用 `scripts/train_hjepa.sh` 和原 `main_hjepa.py`。此入口只接受模拟场景，不会自动开始真实数据主分析。运行目录包含每任务日志、训练配置、模型检查点、输入表、真值、原始评估窗口、结果 JSON、代码哈希、Git 状态及退出状态。下载整个运行目录即可回传；不能只下载一个旧 `allen_v1` 目录。

## 本地验证记录

- 19 项测试通过：来源匹配、缺失保留、单系划分、split 隔离、训练统计独立于留出值、四遮蔽、配对比例、条件扰动和物种／科权重。
- L1/L2 × 五折共 10 项真实输入结构检查通过：`outputs/allen_world_v1/check/check.json`。
- L1 和 L2 各执行 2 步原 JEPA/SIGReg CPU 前向、反向和参数更新，损失及梯度有限。训练损失数值不作生物学结果。
- 模拟启动器 dry-run 通过；真实上游模型对象在独立合成表上跑通保存后评估 CLI，含哈希和压缩窗口输出。
- 本地没有运行完整 Lightning/stable_pretraining 的服务器训练调度，也没有产生新的真实数据模型效果。CPU 冒烟测试使用隔离的本地依赖；A100 环境按既有 environment.yml 使用，无须复制本地依赖目录。
