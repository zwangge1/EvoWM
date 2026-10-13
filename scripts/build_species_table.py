"""生成 data/processed/species_traits.csv（物种主表）。在 EvoWM 根目录运行：python scripts/build_species_table.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evowm.data.species_table import main  # noqa: E402

if __name__ == "__main__":
    main(ROOT)
