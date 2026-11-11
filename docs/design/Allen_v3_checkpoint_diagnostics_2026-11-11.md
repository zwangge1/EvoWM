# Allen v3 已训练模型诊断与服务器运行

2026-11-11。针对已完成的 `outputs/allen_overnight_v3_20261110_150810_454903`，复用服务器保留的模型权重。独立生成诊断结果，不重新训练，不修改原始运行、输入、模型或评估记录。

本轮正式结果：联合骨段比例RMSE full 0.079817、single 0.080038，两者差异尚不明确；full相对Ridge低18.2%。诊断目的为识别层间信息的实际使用方式和误差来源，而非预设两层应当胜出。可将论文重点放在生物学发现及模型预测价值，并完整报告架构比较；未测量的自然约束不能作为缺少架构增益的已证实原因。没有明确差异也不等同于正式不劣。

**这次检查什么**

1. 重载保存权重，核对checkpoint、配置、输入、原始推理代码及窗口身份；先复现原评估预测，最大绝对偏差须不超过0.00001 log10单位，超过即停止该任务。
2. 固定模型参数、下层基础预测及用于三骨重构的预测总长，分别将上层丰富状态置零、替换为训练窗口均值，或在相似预测总长的窗口间乱序。乱序做5次，跨完整评估窗口取供体，不依赖计算batch划分；输入不使用目标实测值。
3. 另外切掉上层状态及总长对下层的条件输入，但仍保留原预测总长用于三骨重构。原先no_link、scalar中的相应遮罩必须继续生效，作为负对照。
4. 将每段骨的log误差分解为总长误差、该骨比例误差，以及两者的交叉项。交叉项可以为负，不能写成三个非负“贡献比例”。
5. 仅在结果分析中进行真值替换：分别把总长或比例换成观测值，检查误差如何变化。它们使用目标答案，不是可部署预测，也不是必然可达的误差下限；取消误差抵消有时会让误差变大。
6. 在固定的训练窗口子集上检查观测/坐标损失与潜表示预测损失的梯度方向、量级，以及上层表示的变异。这是局部诊断：eval模式、完整条件、只看最终目标，不含SIGReg和训练时其它位置/条件遮蔽，不能当成完整训练过程的因果解释。

冻结权重干预衡量当前模型依赖何种信息，与重新训练消融不同；置零/乱序仍可能形成训练分布外输入。须结合原有重训对照、训练均值替换及同尺寸乱序共同判断。即便乱序使预测变差，也不能单凭此证明两层优于单层或存在生物学因果关系。

**计算与输出**

默认CPU 32个进程、每进程2个计算线程；可用 `WORKERS=64 THREADS_PER_JOB=2` 调整，合计不超过128线程。本轮不使用GPU。smoke运行真实数据full和single的seed13/fold0，共2项；full读取原真实数据五配置×两种子×五折，共50项。

所有指标沿用相同完整三骨支持，窗口先按物种聚合、再按科等权，两个训练种子平均MSE后开平方。各次乱序同样平均误差，不把它们当独立模型。科重抽样区间条件于这批固定训练结果，不包含重新训练不确定性，不是多重比较校正或不劣检验。

自动生成独立 `outputs/allen_diagnose_v3_时间戳/`，包括原任务清单、诊断源码、运行环境、日志、逐模型诊断JSON和小型窗口表，以及总体/逐种子预测误差、配对差异、误差分解等汇总。结束时打印 `DOWNLOAD=outputs/allen_diagnose_v3_时间戳_review.tar.gz`；该包不含模型权重和原始大目录。失败时也打包已完成的材料和失败日志，退出码非零，不能按成功结果解读。

**本地同步到GitHub**

在Windows PowerShell执行以下内容。明确列出本次文件，避免将项目里其它稿件修改混入提交。

```powershell
cd C:\Users\ZihanWANG\Desktop\FFY\EvoWM
$files = @(
  'scripts/allen_world/diagnose_v3.py',
  'scripts/allen_world/diagnose_v3_checkpoint.py',
  'scripts/allen_world/summarize_diagnose_v3.py',
  'scripts/allen_world/run_diagnose_v3.sh',
  'tests/test_allen_diagnose_v3_runner.py',
  'tests/test_allen_diagnose_v3_checkpoint.py',
  'tests/test_allen_diagnose_v3_summary.py',
  'docs/design/Allen_v3_checkpoint_diagnostics_2026-11-11.md'
)
git add -- $files
git --no-pager diff --cached --stat -- $files
git commit -m "Add frozen-checkpoint diagnostics for Allen v3" -- $files
git push origin main
```

**服务器运行**

在现有evowm环境的tmux终端中：

```bash
cd /mnt/zihanw/EvoWM
git pull --ff-only origin main

SOURCE_RUN=outputs/allen_overnight_v3_20261110_150810_454903
WORKERS=2 bash scripts/allen_world/run_diagnose_v3.sh --source-run "$SOURCE_RUN" --smoke &&
WORKERS=32 bash scripts/allen_world/run_diagnose_v3.sh --source-run "$SOURCE_RUN" --full
```

smoke退出码为0才会开始full。请下载最后打印的full诊断包，目录是 `allen_diagnose_v3_...`，无需重新下载原2.7GB过夜包。服务器必须仍保留原 `training/ckpts/任务名/allen_v3_object.ckpt`；本地review包排除了这些文件，因此正式权重诊断应在服务器运行。

只想检查文件清单而不加载权重，可使用：

```bash
bash scripts/allen_world/run_diagnose_v3.sh --source-run "$SOURCE_RUN" --full --plan-only
```

**X问题与本轮诊断的范围**

已完成的v3真实模型把地面、水面、水下、林下、中层、树冠和空中七项觅食比例分别输入，没有强迫前三项作用相同。此前做过的是将七项整组遮蔽：联合比例RMSE从0.079817到0.077315，变化的区间包含零，尚未证明这组特征稳定提高当前模型预测。

尚未检验的是“地面、水面、水下能否合理合并为X，以及为何排除空中”。按X生成的模拟只能检验模型能否恢复该设定，不能证明生态分组正确。本次checkpoint诊断也不承担这一问题。

后续若开展X检验，应在相同样本、同基础主效应和协变量下比较分量各有温度交互的模型与合并交互模型；合并相当于给这些分量施加相同交互系数的约束。比例之和的约束也必须处理。统计上未检出差异并不自动证明它们具有相同热交换机制；腿部热暴露仍需独立测量或机制证据。

**本地验证记录**

- 21项新增测试通过，另有16项既有解剖模型测试通过，覆盖原始输入/源码篡改拒绝、目标隔离、连接负对照、部分观测、误差分解和配对聚合。
- 两份既有本地16步模型权重实际完成各3,664窗口的端到端诊断；full与single相对原预测最大差分别为3.58e−7和4.77e−7。两份输出的联合汇总及梯度/内部状态摘要生成成功。这些短训练只验证流程，不作为正式模型诊断结论。
- 正式50任务原预测的40项汇总RMSE，与前一轮独立审计最大差9.71e−17。
- 正式任务的只读计划核验通过：52份输入、114份源码快照、67份运行时源码一致；本地没有正式checkpoint，服务器50权重诊断尚未运行。
- Bash入口语法检查通过，文件为LF换行。所有新增代码独立于模型训练源码，本次未自动commit或push。
