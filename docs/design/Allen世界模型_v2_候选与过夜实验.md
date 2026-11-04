# Allen 世界模型 v2：候选安排与过夜实验

2026-11-04。开发性比较；尚未产生新的 A100 模型结果。旧的物种→类群汇总 L2 不参与此矩阵。

## 推荐次序与文章问题

现有完整阶梯证据限定于 Eliason 非雀形目早成 P 的 259 种、31 科：温度关联从股骨经胫跗骨到跗跖骨增强。P+SP、鸭科和全鸟类的结果不允许直接套用同一个完整排序。因此模型必须能给出无梯度、相等或反向响应。

| 候选 | 具体预测组织 | 推荐用途 |
|---|---|---|
| allocation，首选 | 先预测三骨长度之和，再预测各骨段分配比例；每段长度由两者共同决定 | 区分整体腿部尺寸变化与骨段之间的重新分配，最贴近 Allen 主线和骨段阶梯 |
| flat，必要对照 | 三骨及外测跗跖平行预测，共享同一 JEPA 预测表示 | 判断结构化输出是否真的优于联合预测；如果它最好，保留它 |
| proximal，探索候选 | 预测股骨，再让胫跗骨参考预测股骨，最后让跗跖骨参考前两段的预测 | 检查按解剖近端→远端组织是否有预测价值；不是预设生长或因果传递顺序 |
| distal，反向对照 | 跗跖骨→胫跗骨→股骨 | 判断串联优势是否只是更灵活的计算，而不是特定解剖顺序 |

“三骨长度之和”不是鸟的站立高度。allocation 的三个比例经 softmax 为正、和为一，但不限制骨段温度斜率的正负或顺序。串联模块使用自己先前的预测，不使用目标物种实测骨长。外测跗跖始终是独立输出，不与跗跖骨当作同一测量。

推荐基于问题与数据的对应关系，不保证正向结果。若 allocation 能改善留出预测，并在模拟和真实留出 P 样本上恢复骨段差，才有理由强调“整体尺寸与骨段分配”这一解释；若 flat 一样好，就无需给文章增加输出层级。不能按哪种方法最容易得到显著梯度来挑模型。

## 与 H-JEPA/EvoWM 的关系

本轮是**单层 JEPA 上的解剖输出组织比较**，不是原始 H-JEPA 的不同时间跨度层级，也没有声称实现了祖先—后代动态。核心仍由原 `hjepa_utils.create_world_model` 构建，使用原编码器、条件预测器、残差扩展、JEPA/SIGReg 训练。新增模块仅在 `src/evowm/models/allen_readout.py`；原训练入口通过 opt-in 分支加载。

四种候选都增加同样权重的、带缺失掩码的骨长读数监督，以使预测表示保留论文所需的骨段信息。每个骨段先按训练集尺度标准化，再按有效标签平均；四个性状等权。原 JEPA 损失没有被删除，但不能称新目标函数与旧版完全相同。

所有候选使用同样四个标量 MLP 头、同样隐层宽度及输入槽位，当前总可训练参数均为 **1,617,668**。无先前预测的槽位填零。总参数量相同不意味着函数类完全等价；这也是为什么需要平行与反向对照。SIGReg、优化器、批量、训练步数、种子、五折与同一物种窗口保持同口径。

上游训练函数不返回预测表示，新增观测监督使用相同上下文再调用一次预测器；训练时有另一份 dropout 随机抽样，所有候选一致。日志 `train/allen_total_loss` 才是组合目标，旧 `train/loss` 仍是原 JEPA 损失。

## 输入与共同样本

输入分为五组：

- 温度：bio1、bio5、bio6。
- 暴露代理：7 个觅食层比例，包括地面、水面、水下等。不是实测裸露面积或热剂量。
- 运动代理：5 个 AVONET 生活方式类别，与暴露分组不重复传入同一列，但统计相关仍然存在。
- 发育方式：P、SP、SA、A，保留既有按科编码口径。
- 背景：log10 体重、栖境密度、迁徙等级。

同一个模型训练八种信息模式：全部、无温度、无运动、无温度及运动、无暴露、无发育、无温度及暴露、无温度及发育，各概率 1/8。隐藏组同时隐藏原值、差值、观察掩码和提供标志。体重背景与树距离始终提供。信息增益不等于因素的纯因果贡献。

所有架构使用同一份 9,172 种资格样本及冻结整科五折；三骨观测 1,096 种、168 科、40 目。不会为了某个结构单独换一批更有利的物种。缺骨保持缺失，不用插补标签训练。新输入保存在 `outputs/allen_world_v2/input/`，v1 表及原始表不改。训练、评估均校验冻结折分配。

## 一夜的实验矩阵

默认第一阶段模拟：4 候选 × 6 场景 × 2 种子 × 5 折 = **240 个训练任务**，每个 2,000 步。

六场景：温度和运动均无作用；只有运动；只有温度；温度与运动并存且暴露调节温度；三骨温度响应相等；骨段响应反向。每个模拟明确给出每个物种的真值，保留原观察缺失、树上相关残差和测量噪声。正向和反向梯度只赋给 P，不能将 SP 混入检验主组。

随后是真实数据**开发性**比较：4 候选 × 2 种子 × 5 折 = **40 个训练任务**，每个 4,000 步。整个矩阵共 280 个任务，最多四个 GPU 训练任务并行，不是同时启动 280 个。每个真实任务仍输出全部结果，不因模拟失败而删掉；模拟未恢复的模型不得据其条件响应作演化解释。本轮结果会参与开发和选择，不能直接称为锁定后的确认性主分析。正式主分析仍需经过恢复复核、模型锁定和多树等检查。

每个模型使用同一批留出窗口，最后目标形态不进预测器。原尺度骨长、比例、三骨总长度及响应差都输出。骨段直接比较使用相同完整三骨目标样本。重复窗口先在物种内汇总，再按科等权，记录窗口、物种与科三个计数。

P 单独报告，另报 SP、P+SP、SA+A、雀形目、非雀形目。温度诊断包括三温度共同变化以及 bio1、bio5、bio6 各自变化。共同变化不当作 bio1 单变量 PGLS 系数；留出预测的匹配 OLS 也只作开发诊断，不替代既有 PGLS 证据。

汇总脚本使用正向、反向、等响应与零效应场景检查三骨斜率及两个相邻差是否恢复。固定开发容许误差为 `max(0.003, 0.5 × |真值|)`，单位每 10°C 的 log10 长度响应；同一实际支持集、同一科权重比较预测和真值。要求六场景、两个种子、五折完整，且每个场景/种子支持的 P 非雀形目至少 30 种、5 科。这个 flag 是开发诊断，不是正式错误率控制或论文结论放行。缺结果不会按零值补齐，也不会当作通过。

## 本地同步

在 PowerShell 执行。仅列出本轮 EvoWM 代码和设计，保留尚在阅读的稿件修改不混入提交。先检查已暂存内容。

```powershell
Set-Location C:\Users\ZihanWANG\Desktop\FFY\EvoWM
git add -- src/evowm/data/allen.py src/evowm/data/allen_hierarchy.py src/evowm/models/allen_readout.py scripts/allen_world tests configs/hjepa/allen_world_v1_l1.yaml configs/hjepa/allen_world_v1_l2.yaml configs/hjepa/allen_v2_flat.yaml configs/hjepa/allen_v2_allocation.yaml configs/hjepa/allen_v2_proximal.yaml configs/hjepa/allen_v2_distal.yaml hjepa/h_jepa/data.py hjepa/h_jepa/main_hjepa.py hjepa/UPSTREAM.md "docs/design/Allen世界模型_v1_实现与运行.md" "docs/design/Allen世界模型_v2_候选与过夜实验.md"
git diff --cached --stat
git commit -m "Add matched anatomical JEPA experiments and overnight runner"
git push
```

若遇到 dubious ownership，用每条命令的 `git -c safe.directory=C:/Users/ZihanWANG/Desktop/FFY/EvoWM ...`，无需改全局设置。助手尚未提交或推送。

## 服务器启动

仍在现有 evowm 环境和 tmux 中运行。先用 `nvidia-smi` 确认所填的 GPU 当前可用。示例 0/1/2/3 是占位选择；脚本不擅自抢占其他任务。

```bash
cd /mnt/zihanw/EvoWM
git pull --ff-only
nvidia-smi
GPUS="0 1 2 3" HOURS=8 bash scripts/allen_world/run_overnight_v2.sh
```

入口会先提取树、准备输入，做四配置×五折检查及真实骨长标签的 CPU 前后向短检查，然后运行矩阵。最多四张卡；默认每训练任务 8 个 DataLoader worker、4 个数值线程，CPU 评估最多 32 个独立任务、每任务 1 数值线程。可用 `EVAL_WORKERS=16` 等覆盖。

**8 小时是停止启动新任务的预算，不是保证全部跑完，也不是强制中途杀掉当前模型。**预算到时完成正在训练/评估的任务，写清剩余任务。单个训练或评估子进程默认 90 分钟超时，失败保留日志；不无限挂住。第一批实际耗时会逐个打印，尚无可靠依据保证整矩阵需要几小时。

如果只想先验证服务器完整链路：

```bash
GPUS="0" HOURS=1 bash scripts/allen_world/run_overnight_v2.sh --seeds 13 --folds 0 --scenarios null joint --sim-steps 20 --real-steps 20
```

这会跑 12 个短任务，不会满足完整恢复判据，不作生物学解释。

只跑模拟可加 `--stages simulation`；只运行候选子集可加 `--variants flat allocation`。保留完整矩阵是默认推荐，避免按初步有利结果挑选。

## 续跑与回传

启动时会打印 `RUN_DIRECTORY=.../outputs/allen_overnight_v2_时间戳/`。过夜后续跑相同计划：

```bash
GPUS="0 1 2 3" HOURS=8 bash scripts/allen_world/run_overnight_v2.sh --resume outputs/allen_overnight_v2_实际时间戳
```

已经成功评估的任务跳过。训练成功但评估未完成的检查点复用；失败任务默认保留，排除环境原因后可显式加 `--retry-failed`。续跑恢复原计划、输入、步数，其他矩阵参数不会替换原计划；只调整卡、资源和本次时间预算。代码或输入哈希改变时拒绝混合续跑，应开新运行目录。

运行目录有 Linux 进程锁；原 tmux 任务仍在运行时，第二次启动相同目录会被拒绝，防止互相覆盖检查点和状态。代码哈希和快照包括实际 Bash 训练入口。

目录中主要文件：

- `manifest.json`：完整计划、输入与代码哈希；`source/` 为代码快照。
- `status.json`、`completion.json`、`jobs.csv`：每任务状态，失败与未运行任务均列出。
- `summary.csv`：全部已完成模型的预测误差和样本数，不按是否支持梯度筛选。
- `calibration.json`：各候选的模拟恢复诊断；矩阵未完成时明确不通过。
- `evaluation/<任务>/pilot_results.json`：预测、分组与温度诊断。
- `evaluation/<任务>/*windows.csv.gz`：原始逐窗口记录，便于后续科级配对区间与物种核对。
- `logs/`、`training/`：训练/评估日志和模型检查点。

下载整个 `allen_overnight_v2_实际时间戳` 目录。若检查点太大，可先打包 `manifest.json status.json completion.json summary.csv jobs.csv calibration.json evaluation logs`；完整模型和模拟真值仍保留在服务器。实时查看进度可读 `status.json` 或终端的 START/DONE/FAILED 记录。

## 本地验证范围

已通过四模型×五折数据检查，每模型有真实骨长标签的两步 CPU 前向、反向、各头梯度与参数更新，以及保存后直接读数的评估 CLI。四模型参数量相同，输出分配能够表示任意响应排序。另有输入、冻结折、缺失、分组遮蔽、模拟真值、完整矩阵、汇总与部分失败等测试。

本地只执行 CPU 实现检查、合成评估和队列 dry-run，没有启动远程 A100，也没有完整运行服务器 Lightning 调度。所有真实生物学结论仍以前述已核验分析为准，新模型结论等待服务器输出。
