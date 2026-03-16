import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from project_cli import main as cli_main


if __name__ == "__main__":
    cli_main(ROOT_DIR)
