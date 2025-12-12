import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent.parent

commands = [
    ["pip", "install", "-r", str(root / "requirements.txt")],
    ["python", "-m", "src.train_tfidf"],
    ["python", "-m", "src.train_transformer", "--report_to", "none"],
    ["python", "-m", "src.select_best"],
    ["python", "-m", "src.infer"],
]

for cmd in commands:
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=root)
