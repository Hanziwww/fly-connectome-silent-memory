"""Self-contained four-level pilot."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
LIB=HERE/"lib"
sys.path.insert(0,str(LIB))
