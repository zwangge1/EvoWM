"""从 100 棵后验树文件里取出预注册指定的几棵，各存成单树 Newick（data/processed/trees/，git 忽略）。

真实数据主分析 v1（docs/prereg/real_main_v1.md）：Hackett 第 21、41、61、81、100 棵（第 0 阶段训练也用这 5 棵，
模拟数据在第 1 棵上生成）；稳健性用 Ericson 第 21、41、61 棵。编号从 1 开始，与文件中的顺序一致。
在 EvoWM 根目录运行：python scripts/extract_trees.py
"""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PICK = {"hackett": [21, 41, 61, 81, 100], "ericson": [21, 41, 61]}


def main():
    out = ROOT / "data/processed/trees"
    out.mkdir(parents=True, exist_ok=True)
    lines = []
    for backbone, idxs in PICK.items():
        src = ROOT / f"data/raw/birdtree/{backbone}_100.nwk"
        if not src.exists():
            print(f"缺少 {src.relative_to(ROOT)}，跳过 {backbone}")
            continue
        parts = [p.strip() for p in src.read_text().split(";") if p.strip()]
        assert len(parts) == 100, f"{src.name} 应有 100 棵树，实际 {len(parts)}"
        for i in idxs:
            text = parts[i - 1] + ";\n"
            dst = out / f"{backbone}_t{i:03d}.nwk"
            dst.write_text(text)
            lines.append(f"{dst.name}\t{hashlib.md5(text.encode()).hexdigest()}")
    (out / "MANIFEST.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
