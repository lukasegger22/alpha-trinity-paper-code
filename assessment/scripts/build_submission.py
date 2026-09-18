import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alpha_trinity_assessment.submission import main

if __name__ == "__main__":
    main()
