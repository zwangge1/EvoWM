# 数据清单（2026-10-21，由 scripts/make_manifest.py 生成）

`data/raw/` 与 `data/prior_results/` 只读。新增数据先在脚本的 SOURCES 里登记来源，再重新生成本文件。

## raw/avonet

来源：AVONET 补充数据 1（Tobias et al. 2022, Ecology Letters 25:581）。复制自 Desktop\FFY\FFY\data\birdtree\

| 文件 | 大小 | sha256 |
|---|---|---|
| `AVONET_Supplementary_dataset1.xlsx` | 21.5 MB | `eb645e83dddb40f1654a3e8d721998dbca76eff540231b7b809267c1e96f8d3e` |

## raw/birdtree

来源：BirdTree.org Hackett/Ericson Stage2 后验树（Jetz et al. 2012, Nature 491:444）。*_100 为旧仓库抽取的 100 棵，hackett_tree1 为第 1 棵。复制自 Desktop\FFY\FFY\data\birdtree\；完整 1000 棵压缩包未复制，仍在原处

| 文件 | 大小 | sha256 |
|---|---|---|
| `ericson_100.nwk` | 46.4 MB | `7dfc42e80ced3eb37e54e613a42bb2a22b4cacfe5d5822bda5979106e2410084` |
| `hackett_100.nwk` | 46.4 MB | `72372cf2d78288ac4e45ec0e8374ebe5d625c82adc2b06465d841eab4f57ebff` |
| `hackett_tree1.nwk` | 463.9 KB | `b0b9fb0ed68e887650924d057d46905d5bbf5f4dca6050c729cfcb4367efc927` |

## raw/skelevision

来源：Skelevision 骨骼性状（Weeks et al., Dryad doi:10.5061/dryad.v41ns1s4c），版本 2025-11-05，2026-10-13 下载

| 文件 | 大小 | sha256 |
|---|---|---|
| `Complete_Trait_Dataset_v1.csv` | 2.1 MB | `216346a19aa2920bccaca3fb375922d1c9adfa597c146a6b6ec87c37bb44e9eb` |
| `README.md` | 10.6 KB | `d5481ab687ac1e41db1956293c00cfddea98175491a846e5ca25748fb7e6571f` |
| `Skelevision_Only_Dataset_v1.csv` | 1.3 MB | `115c818e254c077ca3ac1e28a99cac06b8150a7485bb36ebe985e16aa7b73933` |
| `Skelevision_species_complete_v1.1.csv` | 1.5 MB | `58cd6668e5dd7c0abf059fc64d998b421b3913e563f9055b664bb940216f2846` |

## raw/watanabe2017

来源：鸭科后肢骨段（Watanabe 2017, The Auk 134(3), doi:10.1642/AUK-17-23.1）。复制自 Desktop\FFY\FFY\data\skeletal\

| 文件 | 大小 | sha256 |
|---|---|---|
| `watanabe2017_anatidae.csv` | 5.4 KB | `40b17eb1cafd535724c53efac3e0146db9a8242ab0f4db609f0231c9e12d3519` |

## raw/eltontraits_derived

来源：旧仓库派生表：AVONET 并 EltonTraits 1.0（Wilman et al. 2014, Ecology 95:2027，经 R 包 traitdata）。复制自 Desktop\FFY\FFY\data\；原始 BirdFuncDat 待取

| 文件 | 大小 | sha256 |
|---|---|---|
| `avonet_hwi.csv` | 1.0 MB | `4df32f61f81c4dca96e6476ad330be78864d148c778743f3905a277e7bbedee4` |
| `avonet_joined_all.csv` | 856.0 KB | `57b3c9c5d46fd0ff285b6ae51bfe9a4a764b7a0d00c8be7f1b42577d91002976` |
| `forstrat_dom.csv` | 324.3 KB | `2ea7f485a3bd2c90d89a9a3c78f5dc2bacf57b06bd0ceca99496f15f14696647` |

## raw/ponti2025

来源：岛屿特有种与最近的大陆近缘种配对的形态（Ponti et al. 2025, Global Ecology and Biogeography 34(11)；Dryad doi:10.5061/dryad.1zcrjdg57），2026-10-18 下载，文件名保持原样。CSV 以分号分隔、Latin-1 编码、每行一件标本；trees_ex.nex 为加入灭绝种的 100 棵 Ericson 树，太大不入库

| 文件 | 大小 | sha256 |
|---|---|---|
| `README.md` | 3.5 KB | `3e4c7c8acd1854656b020732caea85c52970f0bc2e47fcd331d0094712d851f3` |
| `analyses_morpho.R` | 42.9 KB | `4140bfb3c84d3af088ab5b2c73d2744de0691e5b390cb97b25e8d18f8979c063` |
| `morpho_db.csv` | 1.7 MB | `79fe71e0f10febabb68ccedd57548f7cfb32096d9c363b340bdd717d74c2d16d` |
| `morpho_db_filtered.csv` | 808.6 KB | `13504121c078bb9e68d5e38e3b61754e02b7335b475acdbd8d96e2a6cb6bfe31` |
| `trees_ex.nex` | 33.7 MB | `ddb097d18c26244f95f4791700445de45c81450edc54c58622bc9d5aca04da3f` |

## raw/wright2016

来源：岛屿与大陆鸟类骨骼标本的逐件测量与飞行肌（Wright, Steadman & Witt 2016, PNAS 113(17):4765，doi:10.1073/pnas.1522931113；figshare doi:10.6084/m9.figshare.3123148），2026-10-18 下载。两个 xlsx 由 3123148.zip 解出，文件名保持原样；zip 不入库

| 文件 | 大小 | sha256 |
|---|---|---|
| `3123148.zip` | 685.8 KB | `28ecf201504512df87ea6149f8e7eba9d676c5a03922058655487046bf55a3f0` |
| `skeletal_data.xlsx` | 525.7 KB | `4290542ebc0a1570ea62286d6797bb896a8f6294898dab849194c94e976feefa` |
| `species_averages_flight_muscles.xlsx` | 159.7 KB | `db86a2287b598f7ca2baf8610235f251fc1084c9fcd8fcc515fbef5e2c1a5630` |

## raw/alerstam2007

来源：鸟类飞行速度与翼形测量（Alerstam et al. 2007, PLoS Biology 5(8):e197, doi:10.1371/journal.pbio.0050197），Protocol S1：逐种翼展、翅面积、体重等，PDF 表，2026-10-21 下载

| 文件 | 大小 | sha256 |
|---|---|---|
| `pbio.0050197.sd001.pdf` | 177.6 KB | `edd8521bf8d60774bed5677a99e53942790653f3426bb0071eef9f94dabe8c76` |

## raw/fox2026_wingspan

来源：翼展数据库：1,442 种实测翼展与 10,618 种按目回归预测的翼展（Fox, Brunton & Shephard 2026, bioRxiv doi:10.64898/2026.04.27.720532；figshare doi:10.6084/m9.figshare.32089083，CC BY 4.0），2026-10-21 下载

| 文件 | 大小 | sha256 |
|---|---|---|
| `AppendixS1_wingspan_data_all.xlsx` | 4.9 MB | `6b55747a5761c08356d92c7a0b524b4e586c28a25316b0a6c1de3738db1d8eda` |

## raw/serrano2017

来源：83 种现生鸟 188 只的体重、翼展与升力面积（Serrano et al. 2017, Paleobiology 43(1):144, doi:10.1017/pab.2016.24），马拉加大学仓库的作者接受稿 PDF，表在附录，2026-10-21 下载

| 文件 | 大小 | sha256 |
|---|---|---|
| `Serrano2017_Paleobiology_accepted.pdf` | 2.2 MB | `f2bd97422b6f750ec2a452b487c4867cc4578f55211fed937bad7c4d974dd298` |

## raw/tanaka2018

来源：现生鸟翅面积、翼展、体重与足迹（Tanaka 2018, Dryad doi:10.5061/dryad.4h1n1），online data package.xlsx 由 zip 解出，2026-10-21 下载

| 文件 | 大小 | sha256 |
|---|---|---|
| `doi_10_5061_dryad_4h1n1__v20180103.zip` | 23.2 KB | `25323808b741c904636cbc7f00eef702a669952f63e93442a8caab6dc1bd20f5` |
| `online+data+package.xlsx` | 23.1 KB | `8f6ea6e2195a72391bf7cf4a80ff404aa9a16688877520668a0409e071ee7a1d` |

## prior_results

来源：旧仓库生物侧结果。复制自 Desktop\FFY\FFY\outputs\phylo\ 与 outputs\pgls\

| 文件 | 大小 | sha256 |
|---|---|---|
| `README.md` | 0.8 KB | `114bab9f0d9de207bab62f4942485e4166503ce52f30c62ea133424149efbcea` |
| `S1_evo_models.csv` | 0.9 KB | `15df9c695ef3005f1cd938d5548537694f0521bb8d02a8721a50aefa86c2f5aa` |
| `S2_lability.csv` | 0.1 KB | `23f68088d7806ec923ca1b7166806a78e17c17f0ada4d372ae94e2ab13eb0752` |
| `S3_contrasts_tree1.csv` | 1.1 MB | `7b2343c01531ba968d1ce8f1bd7189c0c6e94069e2d2eb180bfd681798c4f998` |
| `S3_pic_signtest.csv` | 16.0 KB | `b1217c0da20c08cde259a772a5440b06b2cd620a190e7de381a7e8a58713cfdb` |
| `S4_redo40.csv` | 5.3 KB | `1bc7efc6c3527c54627c7e75e60eeb3ba62b1445404e5f824978aa41c7c0274e` |
| `S4_redo40_indep.csv` | 5.5 KB | `4739e1769e88b08cfeac25e9f3fa9ad943c8c9b3091e9d612bf7831d15ca0009` |
| `S4_shifts.csv` | 1.8 KB | `1c3fd1ff2f683df57c2dec46585800c5167e8a63d553bba83a2bd0c9ed82308f` |
| `S5_centerline.csv` | 0.2 KB | `5c8b811871de320d510c58a2a2d47ec7fe679ece2019929bbc7eef1472374551` |
| `S5_prior.csv` | 21.7 KB | `864ea6ab13ced1e6fc316932b337033c02a8344766857d8e2454ac64df874bdc` |
| `S6_clade_b.csv` | 10.4 KB | `d95c7a1298b40a55527fbdfdc8bd61631225efb9bac5728f51927d53fdb81358` |
| `S7_diet.csv` | 3.7 KB | `1361e9b8b39b5a0c003aec1fa8a4d06355670b9071f1a4cad655d555323697fd` |
| `T1_ou_covariate.csv` | 19.8 KB | `d01c217ce89e525c06894dc52b89e3d42370f43fce9c2d44f9920801381bbe37` |
| `T2_branchlen.csv` | 1.0 KB | `81a99f8b92347937d8af961b65678b68521167f7d4b12f138fbe95a63c09cbf3` |
| `e21_pair.csv` | 6.5 KB | `f2cd186cbe4f3e4d52905b61ca80e229ca806337e9d836f6f9f29427c7285065` |
| `falconidae.csv` | 7.1 KB | `0ba69deb0ac67b5e39c392bccdf5009d750cdf45454826cc52ed297c5f65d614` |
| `forstrat.csv` | 727.4 KB | `ee3b6f2c227caa5c8d938d3a74e73e223767bb7882f1fbbc384677e9d39c950b` |
| `pgls_p3_results.csv` | 22.9 KB | `5ea1ff0638424f4a935ab8fda6d342586b5f9a2f250c6b61a909f20af57bdbec` |
| `pgls_p4_results.csv` | 7.7 KB | `375b70f644d50599d84f92aed934a7587bf8bb03d8cf5ba9bdd52a421d2c31c5` |
| `within_属.csv` | 27.9 KB | `db3b9835c63efe6725f147572410b6de841180bd80a511acae7e964356e8649b` |
| `within_科.csv` | 6.5 KB | `2a35f8f4a383df2d536e7a837db395c6cc8a822262822c092e7687e956653e69` |
