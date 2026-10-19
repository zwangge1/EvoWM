# 上游来源

- 仓库：https://github.com/kevinghst/H-JEPA
- 提交：97054ecaff83e13839195ec4b92d74c08a0127c5（2026-10-09，"Use original teaser in README"）
- 论文：H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning（arXiv 2610.06805）
- 许可：MIT，见本目录 LICENSE，保留原版权声明
- 导入：2026-10-13，用 `git archive` 原样导出，169 个文件，未做任何改动

## 改动规则

1. 第一次提交为原样快照，之后每处改动单独提交，提交信息以 `hjepa:` 开头。
2. 每处改动在下表登记：文件、改了什么、为什么。
3. 鸟类专用的新代码放 `../src/evowm/`，尽量不改上游文件；必须改时改动最小。
4. 暂时不用的部分（规划、机器人环境、DROID）先保留不动，跑通后再决定删不删。

## 改动登记

| 日期 | 文件 | 改动 | 原因 |
|---|---|---|---|
| 2026-10-13 | h_jepa/data.py | `build_hdf5_dataset` 增加 `type: evowm_transect` 分支，返回 `evowm.data.transect.TreeTransectDataset`；原 droid 分支改为先取出 type 再判断，行为不变 | 接入系统发育样带数据，写法仿上游的 DROID 分支 |
| 2026-10-15 | h_jepa/main_hjepa.py | 配置有 `evowm_units: true` 时改用 `evowm.models.units` 的 `create_world_model` 与 `units_forward`；`_hjepa_training_forward` 增加可选参数 `forward_fn`（默认仍为 `hjepa_forward`）。其他配置行为不变 | 按习性尺度分层（设计草案 `docs/design/第二层设计草案_2026-10-15.md`）：上层单位按习性事件分段、长度可变，上游按固定步长分块的 `encode_hierarchical` 用不了。各层仍是上游的 JEPA 对象，第 1 层的损失与上游逐位相同 |
| 2026-10-19 | h_jepa/models/jepa.py | `JEPA.predict`：对象上 `residual_predict` 为真时，在预测器输出上加输入的表示（预测变化量）；没有这个属性时行为不变 | 读数诊断 v1 显示预测器在习性没变时仍给出变化，上层预测不如照抄（设计 `docs/design/预测变化量_重训_2026-10-19.md`） |
| 2026-10-19 | h_jepa/main_hjepa.py | 配置有 `evowm_residual: true` 时，建好模型后调用 `evowm.models.residual.enable_residual`（各层打开 `residual_predict`，预测器投影头最后一个线性层置 0）。其他配置行为不变 | 同上 |

## 方法上的扩展（不改上游文件，代码在 `../src/evowm/`）

| 日期 | 扩展 | 与上游的不同 |
|---|---|---|
| 2026-10-15 | 按习性事件分块的上层（`evowm/data/units.py`、`evowm/models/units.py`） | 上游的上层按固定步长、固定窗口分块；这里的分块边界是叶序上运动方式（第 2 层）或栖境（第 3 层）改变的位置，单位长度可变。上层状态是单位成员下层表示的掩码平均再经上层编码器（上游是固定窗口的下层表示）；上层动作直接由单位的平均习性之差与成员间平均进化距离构成（上游把下层动作序列再编码一次）。理由见设计草案第 1、2 节；分层世界模型里按事件分块的先例：THICK（Gumbsch 等，ICLR 2024） |
| 2026-10-19 | 预测变化量（`evowm/models/residual.py`，配置 `recovery_v5_hab3_res`、`recovery_v5_l1_res`） | 上游预测器直接给出下一步的表示；这里给出"前一步的表示 + 变化量"，变化量一支最后一层初始化为 0，训练开始时等于照抄表示。结构、维数、损失与训练设置都不变 |
