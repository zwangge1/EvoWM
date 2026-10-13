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
