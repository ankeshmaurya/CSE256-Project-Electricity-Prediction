# src/main.py
import subprocess
import time
from pathlib import Path

def run_script(script_name):
    print(f"\n🌀 Running {script_name}...")
    start = time.time()
    result = subprocess.run(["python", f"src/{script_name}"], capture_output=True, text=True)
    
    elapsed = time.time() - start
    if result.returncode == 0:
        print(f"✅ {script_name} completed in {elapsed:.1f}s")
        print(result.stdout)
    else:
        print(f"❌ {script_name} failed after {elapsed:.1f}s")
        print(result.stderr)
    return result.returncode

def main():
    Path("../data").mkdir(exist_ok=True)
    Path("../models").mkdir(exist_ok=True)
    Path("../logs").mkdir(exist_ok=True)

    scripts = [
        "generate_data.py",
        "preprocess.py",
        "train.py",
        "predict.py",
        "anomaly.py"
    ]

    for script in scripts:
        if run_script(script) != 0:
            print(f"⚠️ Pipeline aborted due to failure in {script}")
            return

    print("\n✨ All stages completed successfully! ✨")

if __name__ == "__main__":
    print("Neural Network Pipeline")
    print("===================================")
    main()