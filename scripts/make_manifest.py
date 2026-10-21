"""生成 data/MANIFEST.md：data/raw 与 data/prior_results 下每个文件的来源、大小、sha256。

新增数据时：先在 SOURCES 里登记来源，再运行本脚本。
"""
import hashlib
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

SOURCES = {
    "raw/avonet": "AVONET 补充数据 1（Tobias et al. 2022, Ecology Letters 25:581）。复制自 Desktop\\FFY\\FFY\\data\\birdtree\\",
    "raw/birdtree": "BirdTree.org Hackett/Ericson Stage2 后验树（Jetz et al. 2012, Nature 491:444）。*_100 为旧仓库抽取的 100 棵，hackett_tree1 为第 1 棵。复制自 Desktop\\FFY\\FFY\\data\\birdtree\\；完整 1000 棵压缩包未复制，仍在原处",
    "raw/skelevision": "Skelevision 骨骼性状（Weeks et al., Dryad doi:10.5061/dryad.v41ns1s4c），版本 2025-11-05，2026-10-13 下载",
    "raw/watanabe2017": "鸭科后肢骨段（Watanabe 2017, The Auk 134(3), doi:10.1642/AUK-17-23.1）。复制自 Desktop\\FFY\\FFY\\data\\skeletal\\",
    "raw/eltontraits_derived": "旧仓库派生表：AVONET 并 EltonTraits 1.0（Wilman et al. 2014, Ecology 95:2027，经 R 包 traitdata）。复制自 Desktop\\FFY\\FFY\\data\\；原始 BirdFuncDat 待取",
    "raw/ponti2025": "岛屿特有种与最近的大陆近缘种配对的形态（Ponti et al. 2025, Global Ecology and Biogeography 34(11)；Dryad doi:10.5061/dryad.1zcrjdg57），2026-10-18 下载，文件名保持原样。CSV 以分号分隔、Latin-1 编码、每行一件标本；trees_ex.nex 为加入灭绝种的 100 棵 Ericson 树，太大不入库",
    "raw/wright2016": "岛屿与大陆鸟类骨骼标本的逐件测量与飞行肌（Wright, Steadman & Witt 2016, PNAS 113(17):4765，doi:10.1073/pnas.1522931113；figshare doi:10.6084/m9.figshare.3123148），2026-10-18 下载。两个 xlsx 由 3123148.zip 解出，文件名保持原样；zip 不入库",
    "raw/alerstam2007": "鸟类飞行速度与翼形测量（Alerstam et al. 2007, PLoS Biology 5(8):e197, doi:10.1371/journal.pbio.0050197），Protocol S1：逐种翼展、翅面积、体重等，PDF 表，2026-10-21 下载",
    "raw/fox2026_wingspan": "翼展数据库：1,442 种实测翼展与 10,618 种按目回归预测的翼展（Fox, Brunton & Shephard 2026, bioRxiv doi:10.64898/2026.04.27.720532；figshare doi:10.6084/m9.figshare.32089083，CC BY 4.0），2026-10-21 下载",
    "raw/serrano2017": "83 种现生鸟 188 只的体重、翼展与升力面积（Serrano et al. 2017, Paleobiology 43(1):144, doi:10.1017/pab.2016.24），马拉加大学仓库的作者接受稿 PDF，表在附录，2026-10-21 下载",
    "raw/tanaka2018": "现生鸟翅面积、翼展、体重与足迹（Tanaka 2018, Dryad doi:10.5061/dryad.4h1n1），online data package.xlsx 由 zip 解出，2026-10-21 下载",
    "prior_results": "旧仓库生物侧结果。复制自 Desktop\\FFY\\FFY\\outputs\\phylo\\ 与 outputs\\pgls\\",
}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def size(n):
    return f"{n / 1e6:.1f} MB" if n >= 1e6 else f"{n / 1e3:.1f} KB"


def main():
    for d in sorted((DATA / "raw").iterdir()):
        if d.is_dir() and f"raw/{d.name}" not in SOURCES:
            print(f"警告：data/raw/{d.name} 未在 SOURCES 登记来源")

    lines = [
        f"# 数据清单（{date.today()}，由 scripts/make_manifest.py 生成）",
        "",
        "`data/raw/` 与 `data/prior_results/` 只读。新增数据先在脚本的 SOURCES 里登记来源，再重新生成本文件。",
        "",
    ]
    for sub, src in SOURCES.items():
        d = DATA / sub
        files = sorted(p for p in d.rglob("*") if p.is_file() and p.name != ".gitkeep")
        lines += [f"## {sub}", "", f"来源：{src}", "", "| 文件 | 大小 | sha256 |", "|---|---|---|"]
        for p in files:
            lines.append(f"| `{p.relative_to(d).as_posix()}` | {size(p.stat().st_size)} | `{sha256(p)}` |")
        lines.append("")
    (DATA / "MANIFEST.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"写入 {DATA / 'MANIFEST.md'}")


if __name__ == "__main__":
    main()
