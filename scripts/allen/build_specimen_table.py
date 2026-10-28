"""标本级主表：Skelevision 的 14,419 件标本，逐件接上博物馆记录（GBIF）与采集点温度（WorldClim 2.1）。

来源：
- 骨长：data/raw/skelevision/Skelevision_Only_Dataset_v1.csv（mm；tarsus 列是胫跗骨，metatarsus 是跗跖骨）。
- UMMZ 记录：data/raw/gbif_ummz_birds/0038434-*.zip 的 occurrence.txt（Darwin Core Archive），
  含 sex、lifeStage、preparations、dynamicProperties（野外测量的 JSON：weight、wing、tarsus、gonads、fat、skull_ossification）。
- FMNH 记录：data/raw/gbif_fmnh_birds/0038148-*.zip（GBIF 简单 CSV），只有年份、坐标、地点，没有性别。
- 关联键：ID 的 "UMMZ_107002" 对应 GBIF catalogNumber "107002"（去前导零后比较）。
- 温度：采集坐标处的 bio1、bio5、bio6；落在海里时取 0.25° 内最近的陆地像元（clim_dist 记回退像元数）。

派生列（设计文档 3.3 的纳入规则，这里只打标记不删行）：
- weight_g：从 dynamicProperties 的 weight 字段解析，默认克；oz、lb、kg 换算；≤ 0 置缺。
- skull_not_ossified：skull_ossification 字段写明未完全骨化（"not"、"unoss"、"incompl"、"partial"、"SIO"、"unpneum" 或百分数 < 100）。
- adult_ok：lifeStage 不属于 Immature、Juvenile、Chick、Nestling、Fledgling、Subadult，且 skull_not_ossified 不为真。
- lat_in_range：采集纬度在 AVONET 该种 Min/Max.Latitude 外扩 4.5°（约 500 km）之内；缺纬度或缺范围为 NaN。
  范围外的多是动物园（底特律动物园，42.47°N, -83.14°E）的圈养鸟和迁徙种的越冬地标本。
- captive_hint：locality 或 occurrenceRemarks 含 zoo、captive、aviary、cage、pet、dealer、import 等字样。
输出 data/processed/specimens.csv。在 EvoWM 根目录运行：python scripts/allen/build_specimen_table.py
"""
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from evowm.data.climate import WorldClim  # noqa: E402
from evowm.data.species_table import AVONET_XLSX, SKEL_ONLY, _tip  # noqa: E402

UMMZ_DWCA = "data/raw/gbif_ummz_birds/0038434-260928105237408.zip"
FMNH_CSV = "data/raw/gbif_fmnh_birds/0038148-260928105237408.zip"
BONES = ["femur", "tarsus", "metatarsus", "humerus", "ulna", "radius", "carpometacarpus",
         "second_digit", "cv.keel.1", "cv.furcula.1", "cv.skull.1", "sclerotic_ring"]
BONE_OUT = {"tarsus": "tibiotarsus", "metatarsus": "tarsometatarsus", "cv.keel.1": "keel",
            "cv.furcula.1": "furcula", "cv.skull.1": "skull", "second_digit": "digit2"}
GBIF_COLS = ["catalogNumber", "sex", "lifeStage", "reproductiveCondition", "preparations", "dynamicProperties",
             "year", "month", "decimalLatitude", "decimalLongitude", "coordinateUncertaintyInMeters",
             "countryCode", "stateProvince", "locality", "occurrenceRemarks"]
IMMATURE = {"Immature", "Juvenile", "Chick", "Nestling", "Fledgling", "Subadult"}
NUM = re.compile(r"(\d+(?:\.\d+)?)")


def read_gbif(zip_path: Path, member: str, cols):
    with zipfile.ZipFile(zip_path) as z:
        with z.open(member) as f:
            d = pd.read_csv(f, sep="\t", usecols=lambda c: c in cols, dtype=str, quoting=3, low_memory=False)
    for c in cols:
        if c not in d:
            d[c] = np.nan
    d["cat"] = d["catalogNumber"].str.strip().str.lstrip("0")
    return d


def parse_props(x):
    if not isinstance(x, str) or not x.startswith("{"):
        return {}
    try:
        return json.loads(x)
    except Exception:
        return {}


WEIGHT = re.compile(r"(\d+(?:\.\d+)?)\s*(kg|g|gm|gms|grams|gr|oz|lbs|lb)?\b")
UNIT = {"kg": 1000.0, "oz": 28.3495, "lb": 453.592, "lbs": 453.592}


def weight_g(v):
    """只认紧跟在数字后面的单位；备注里的 "(frozen)" 之类不算单位。无单位按克。"""
    if v is None:
        return np.nan
    m = WEIGHT.search(str(v).lower())
    if not m:
        return np.nan
    w = float(m.group(1)) * UNIT.get(m.group(2) or "g", 1.0)
    return w if w > 0 else np.nan


def first_num(v):
    if v is None:
        return np.nan
    m = NUM.search(str(v))
    return float(m.group(1)) if m else np.nan


def skull_not_ossified(v):
    if v is None:
        return np.nan
    s = str(v).lower()
    if any(k in s for k in ("not", "unoss", "incompl", "partial", "sio", "unpneum")):
        return True
    pct = re.search(r"(\d+)\s*%", s)
    if pct and int(pct.group(1)) < 100:
        return True
    return False


def main():
    sk = pd.read_csv(ROOT / SKEL_ONLY, dtype={"ID": str})
    out = pd.DataFrame({"ID": sk["ID"], "inst": sk["ID"].str.split("_").str[0],
                        "cat": sk["ID"].str.split("_").str[1].str.lstrip("0"),
                        "species": sk["species"], "tip": sk["species"].map(_tip)})
    for b in BONES:
        out[BONE_OUT.get(b, b)] = pd.to_numeric(sk[b], errors="coerce")

    u = read_gbif(ROOT / UMMZ_DWCA, "occurrence.txt", GBIF_COLS)
    u["inst"] = "UMMZ"
    with zipfile.ZipFile(ROOT / FMNH_CSV) as z:
        member = [n for n in z.namelist() if n.endswith(".csv")][0]
    f = read_gbif(ROOT / FMNH_CSV, member, GBIF_COLS)
    f["inst"] = "FMNH"
    g = pd.concat([u, f], ignore_index=True)
    g = g.drop_duplicates(["inst", "cat"])
    out = out.merge(g.drop(columns=["catalogNumber"]), on=["inst", "cat"], how="left", indicator="gbif_linked")
    out["gbif_linked"] = (out["gbif_linked"] == "both").astype(int)
    print(f"标本 {len(out)}，接上 GBIF 记录 {int(out['gbif_linked'].sum())}")

    for c in ["year", "month", "decimalLatitude", "decimalLongitude", "coordinateUncertaintyInMeters"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.rename(columns={"decimalLatitude": "lat", "decimalLongitude": "lon",
                              "coordinateUncertaintyInMeters": "coord_unc_m", "countryCode": "country"})

    props = [parse_props(x) for x in out["dynamicProperties"]]
    out["weight_raw"] = [p.get("weight") for p in props]
    out["weight_g"] = [weight_g(p.get("weight")) for p in props]
    out["wing_field"] = [first_num(p.get("wing")) for p in props]
    out["tarsus_field"] = [first_num(p.get("tarsus")) for p in props]
    out["gonads"] = [p.get("gonads") for p in props]
    out["fat"] = [p.get("fat") for p in props]
    out["skull_ossification"] = [p.get("skull_ossification") for p in props]
    out["skull_not_ossified"] = [skull_not_ossified(p.get("skull_ossification")) for p in props]
    out = out.drop(columns=["dynamicProperties"])

    out["adult_ok"] = ~out["lifeStage"].isin(IMMATURE) & (out["skull_not_ossified"] != True)  # noqa: E712
    text = (out["locality"].fillna("") + " | " + out["occurrenceRemarks"].fillna("")).str.lower()
    out["captive_hint"] = text.str.contains(r"zoo|captiv|aviary|aviar|cage|pet |pet$|pet shop|dealer|importe", regex=True).astype(int)

    av = pd.read_excel(ROOT / AVONET_XLSX, sheet_name="AVONET3_BirdTree")
    av["tip"] = av["Species3"].map(_tip)
    rng = av.drop_duplicates("tip").set_index("tip")[["Min.Latitude", "Max.Latitude"]].apply(pd.to_numeric, errors="coerce")
    lo = out["tip"].map(rng["Min.Latitude"]) - 4.5
    hi = out["tip"].map(rng["Max.Latitude"]) + 4.5
    out["lat_in_range"] = np.where(out["lat"].isna() | lo.isna(), np.nan, (out["lat"] >= lo) & (out["lat"] <= hi))

    wc = WorldClim(ROOT)
    vals, dist = wc.sample(out["lat"], out["lon"], fallback_deg=0.25)
    for k, v in vals.items():
        out[k] = v
    out["clim_dist"] = dist

    path = ROOT / "data/processed/specimens.csv"
    out.to_csv(path, index=False, float_format="%.4f")
    leg = out[["femur", "tibiotarsus", "tarsometatarsus"]].notna().all(axis=1)
    um = out["inst"] == "UMMZ"
    print(f"写入 {path}")
    print(f"UMMZ 三段腿骨齐全 {int((leg & um).sum())}：性别雌/雄 {int((leg & um & out.sex.isin(['Male', 'Female'])).sum())}，"
          f"成体标记 {int((leg & um & out.adult_ok).sum())}，有坐标 {int((leg & um & out.lat.notna()).sum())}，"
          f"有温度 {int((leg & um & out.bio1.notna()).sum())}（回退 {int((leg & um & (out.clim_dist > 0)).sum())}），"
          f"有体重 {int((leg & um & out.weight_g.notna()).sum())}，纬度在分布范围外 {int((leg & um & (out.lat_in_range == 0)).sum())}，"
          f"圈养字样 {int((leg & um & (out.captive_hint == 1)).sum())}")
    units = out["weight_raw"].dropna().str.lower().str.extract(WEIGHT.pattern)[1].value_counts(dropna=False)
    print("weight 单位计数", units.to_dict())


if __name__ == "__main__":
    main()
