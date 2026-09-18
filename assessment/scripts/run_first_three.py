from __future__ import annotations

import sys
from pathlib import Path

ASSESSMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ASSESSMENT_ROOT / "src"))

from alpha_trinity_assessment.first_three import main


if __name__ == "__main__":
    main()

