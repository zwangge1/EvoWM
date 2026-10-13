# EvoWM：演化世界模型

把系统发育树上的演化时间当作时间轴，借 H-JEPA 的分级世界模型思路，学两件事：

- 前向：习性转变如何改变腿部形态；
- 逆向：从近缘种之间的形态差反推演化压力。

设计见 `docs/design/方案二_演化世界模型_设计草案_2026-10-13.md`。

本仓库建于 2026-10-13。前身 `Desktop\FFY\FFY`（STILT 工程线与生物侧旧分析）已封存，只读：不在其中新增或修改文件，需要的数据与结果已复制到本仓库。

## 目录

```
EvoWM/
├─ docs/
│  ├─ design/          当前设计
│  ├─ context/         数据盘点等背景记录
│  └─ prereg/          预注册（待写）
├─ data/
│  ├─ MANIFEST.md      每个文件的来源、大小、sha256（scripts/make_manifest.py 生成）
│  ├─ raw/             原始数据，只读
│  ├─ prior_results/   FFY 生物侧旧结果，只读，作基线与模拟先验
│  ├─ interim/         中间表，由 scripts/ 生成
│  └─ processed/       建模用表，由 scripts/ 生成
├─ src/evowm/
│  ├─ data/            物种主表、近缘对采样、树工具
│  ├─ sim/             树上 OU 模拟（恢复检验）
│  ├─ models/          编码器、预测器、SIGReg、逆动力学头、分级
│  └─ eval/            探针、类群留出、观测空间指标
├─ configs/
├─ scripts/            可执行入口；exploratory/ 只放选方向用的探索
└─ outputs/            每次运行一个子目录
```

## 数据注意

- Skelevision 的 `tarsus` 列是胫跗骨，`metatarsus` 列才是跗跖骨（对应 AVONET 的 `Tarsus.Length`）。
- Skelevision 物种名实为 BirdTree 分类（其 README 写 BirdLife V3，不准），2,057 种可直接挂 Hackett 树。
- Skelevision 物种均值表 `Skelevision_species_complete_v1.1.csv` 已沿树插补，不用于分析；从 `Skelevision_Only_Dataset_v1.csv` 自算物种均值。
- 觅食层与食性目前用 FFY 的派生表（`data/raw/eltontraits_derived/`）；原始 EltonTraits 1.0（Wilman et al. 2014）待取。
- 完整 1000 棵后验树压缩包未复制，仍在 `Desktop\FFY\FFY\data\birdtree\`。

## 口径

1. 先写预注册（`docs/prereg/`），再跑真实数据主分析；允许零结果。
2. 模拟恢复检验（真实树、已知参数、含习性无效的零模型）通过前，不跑真实数据主分析。
3. 结论在观测空间（mm、骨段比）计算；不同模型不在各自隐空间里比较。
4. 评估按类群整体留出，近缘对两端同在留出集。
5. 每个结果至少 2 个随机种子；文中每个数字能追溯到 `outputs/` 下的具体运行。
6. `scripts/exploratory/` 的结果只用于选方向，不进论文。

## 参考

- H-JEPA：https://github.com/kevinghst/H-JEPA（MIT），借用分级堆叠、SIGReg、逆动力学头。
- 旧 PGLS 与 R 脚本：`Desktop\FFY\FFY\src\stage6_surrogate\`。
