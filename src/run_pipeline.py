import sys
import subprocess
import time
import logging
import os
from pathlib import Path
from datetime import datetime


CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent 
LOG_DIR = PROJECT_ROOT / "logs"
PYTHON_EXE = sys.executable 

SCRIPT_DATA      = CURRENT_SCRIPT_DIR / "ai_ls_allocation" / "data" / "download.py"
SCRIPT_FEATURES  = CURRENT_SCRIPT_DIR / "ai_ls_allocation" / "features" / "build.py"
SCRIPT_MACRO     = CURRENT_SCRIPT_DIR / "ai_ls_allocation" / "models" / "train_xgboost.py"
SCRIPT_ENGINE    = CURRENT_SCRIPT_DIR / "ai_ls_allocation" / "engine" / "bt_trinity.py"
SCRIPT_REPORT    = CURRENT_SCRIPT_DIR / "ai_ls_allocation" / "engine" / "tearsheet.py"
SCRIPT_EXECUTION = CURRENT_SCRIPT_DIR / "ai_ls_allocation" / "execution.py"

# Logging Setup
LOG_DIR.mkdir(exist_ok=True)
log_filename = LOG_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout) 
    ]
)

def run_step(script_path, step_name):
    if not script_path.exists():
        err_msg = f"❌ CRITICAL ERROR: Script not found at {script_path}"
        print(err_msg)
        logging.error(err_msg)
        sys.exit(1)

    header = f"\n{'='*60}\n🚀 STARTING STEP: {step_name}\n   Script: {script_path}\n{'='*60}"
    print(header)
    logging.info(header)
    
    start_time = time.time()
    
    # --- UMGEBUNGSVARIABLEN SETUP ---
    env_vars = os.environ.copy()
    
    env_vars["OMP_NUM_THREADS"] = "1"
    env_vars["KMP_DUPLICATE_LIB_OK"] = "True"
    env_vars["TF_CPP_MIN_LOG_LEVEL"] = "2"
    
    try:
        process = subprocess.Popen(
            [PYTHON_EXE, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, 
            text=True,
            bufsize=1,
            env=env_vars 
        )

        # read Live-Output 
        for line in process.stdout:
            print(line, end='') 

        
        process.wait()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, script_path)

        duration = time.time() - start_time
        success_msg = f"✅ STEP COMPLETED in {duration:.2f} seconds."
        print(success_msg)
        logging.info(success_msg)
        
    except subprocess.CalledProcessError:
        err_msg = f"\n❌ CRITICAL ERROR in {step_name}!"
        print(err_msg)
        logging.error(err_msg)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline stopped by user.")
        sys.exit(1)

def main():
    start_msg = f"--- 🏦 TRINITY INSTITUTIONAL PIPELINE START ---"
    print(start_msg)
    logging.info(start_msg)
    
    start_total = time.time()

    
    # 1. Daten Update 
    run_step(SCRIPT_DATA, "DATA INGESTION")
    
    # 2. Features 
    run_step(SCRIPT_FEATURES, "FEATURE ENGINEERING")
    
    # 3. Makro-Modell 
    run_step(SCRIPT_MACRO, "MACRO BRAIN TRAINING")
    
    # 4. Trinity Engine 
    run_step(SCRIPT_ENGINE, "TRINITY ENGINE & OPTIMIZATION")
    
    # 5. Tearsheet 
    run_step(SCRIPT_REPORT, "PERFORMANCE REPORTING")

    # 6. Execution 

    total_duration = time.time() - start_total
    finish_msg = f"\n{'='*60}\n🎉 PIPELINE FINISHED SUCCESSFULLY in {total_duration:.2f} seconds.\n   Check the log file at: {log_filename}\n{'='*60}"
    print(finish_msg)
    logging.info(finish_msg)

if __name__ == "__main__":
    main()