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
│  ├─ prereg/          预注册（每次检验跑之前写定）
│  └─ results/         入库的结果报告（只在本地从 outputs/ 复制进来再提交）
├─ data/
│  ├─ MANIFEST.md      每个文件的来源、大小、sha256（scripts/make_manifest.py 生成）
│  ├─ raw/             原始数据，只读
│  ├─ prior_results/   FFY 生物侧旧结果，只读，作基线与模拟先验
│  ├─ interim/         中间表，由 scripts/ 生成
│  └─ processed/       建模用表，由 scripts/ 生成
├─ hjepa/              H-JEPA 源码（上游 97054ec，MIT），模型在此基础上改；见 hjepa/UPSTREAM.md
├─ src/evowm/
│  ├─ data/            物种主表、系统发育样带数据集、树工具
│  ├─ sim/             树上 OU 模拟（恢复检验）
│  ├─ models/          对 hjepa 的鸟类专用扩展（尽量少）
│  └─ eval/            探针、类群留出、观测空间指标
├─ configs/hjepa/      H-JEPA 训练配置（hydra，用 --config-dir 加载，不改上游配置）
├─ scripts/            可执行入口；exploratory/ 只放选方向用的探索
└─ outputs/            运行产生的一切（检查点、日志、json、报告），git 忽略
```

## 运行

在 EvoWM 根目录：

1. 生成物种主表：`python scripts/build_species_table.py`，写到 `data/processed/species_traits.csv`。
2. 生成冒烟测试用的模拟表：`python scripts/sim_smoke_table.py`，不含任何真实形态数值。
3. 训练：`bash scripts/train_hjepa.sh evowm_l2 [hydra 覆盖项]`，配置在 `configs/hjepa/`，结果写到 `outputs/hjepa/ckpts/`。
   CPU 冒烟测试：`bash scripts/train_hjepa.sh evowm_l2 data.dataset.species_table=data/processed/sim/smoke/species_traits.csv data.dataset.n_trees=10 trainer.accelerator=cpu trainer.precision=32 trainer.max_epochs=1 +trainer.limit_train_batches=40 num_workers=2`

环境：`conda env create -f environment.yml`（Python 3.11、torch 2.7.1，与上游一致）。

## 三处分工（本地、服务器、云端）

代码和文档只有一个入口，生成物只往 `outputs/` 写，这样服务器上 `git pull` 不会冲突。

1. **本地（Windows，`Desktop\FFY\EvoWM`）**：唯一提交、推送的地方。Claude 改的文件同步到这里，由你提交推送。
2. **服务器（A100，`/mnt/zihanw/EvoWM`）**：只做 `git pull` 和运行，不在仓库里改文件、不提交。脚本的所有输出（检查点、日志、json、报告）都写进 `outputs/`。
3. **回传**：服务器 `outputs/<名字>/` 下的报告和 json 原样下载到本地 `outputs/<名字>/`，路径保持一致。要入库的报告由本地复制到 `docs/results/` 再提交。
4. **云端（Claude 的工作区）**：写代码、做不需要 GPU 的计算；通过桌面端读写本地文件夹，碰不到服务器。

服务器上 `git pull` 报 `untracked working tree files would be overwritten` 时，说明有文件被写进了仓库目录，把它挪到 `outputs/` 下再拉。

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

- H-JEPA：已原样导入 `hjepa/`（https://github.com/kevinghst/H-JEPA ，提交 97054ec，MIT），在其源码上改；改动逐条登记在 `hjepa/UPSTREAM.md`。
- 旧 PGLS 与 R 脚本：`Desktop\FFY\FFY\src\stage6_surrogate\`。
