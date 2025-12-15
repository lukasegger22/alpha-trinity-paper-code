import sys
import subprocess
import time
import logging
import os  # <--- WICHTIG: os importieren
from pathlib import Path
from datetime import datetime

# --- KONFIGURATION ---
PROJECT_ROOT = Path(__file__).parent.parent 
SRC_DIR = Path(__file__).parent
LOG_DIR = PROJECT_ROOT / "logs"
PYTHON_EXE = sys.executable 

# Logging Setup
LOG_DIR.mkdir(exist_ok=True)
log_filename = LOG_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
    ]
)

def run_step(script_relative_path, step_name):
    script_path = SRC_DIR / script_relative_path
    
    header = f"\n{'='*60}\n🚀 STARTING STEP: {step_name}\n   Script: {script_path}\n{'='*60}"
    print(header)
    logging.info(header)
    
    start_time = time.time()
    
    # --- MAC OS DEADLOCK FIX ---
    # Wir erstellen eine Kopie der Umgebungsvariablen
    env_vars = os.environ.copy()
    # Wir zwingen den Sub-Prozess dazu, Single-Threaded zu laufen
    # Das verhindert den Konflikt zwischen XGBoost und TensorFlow
    env_vars["OMP_NUM_THREADS"] = "1"
    env_vars["KMP_DUPLICATE_LIB_OK"] = "True"
    env_vars["TF_CPP_MIN_LOG_LEVEL"] = "2"
    
    try:
        # Popen startet das Skript MIT den neuen Umgebungsvariablen (env=env_vars)
        process = subprocess.Popen(
            [PYTHON_EXE, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, 
            text=True,
            bufsize=1,
            env=env_vars # <--- HIER IST DER SCHLÜSSEL ZUM ERFOLG
        )

        for line in process.stdout:
            print(line, end='') 
            logging.info(line.strip())

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
    run_step("ai_ls_allocation/data/download.py", "DATA INGESTION")
    
    # 2. Features 
    run_step("ai_ls_allocation/features/build.py", "FEATURE ENGINEERING")
    
    # 3. Makro-Modell 
    run_step("ai_ls_allocation/models/train_xgboost.py", "MACRO BRAIN TRAINING")
    
    # 4. Trinity Engine 
    run_step("ai_ls_allocation/engine/bt_trinity.py", "TRINITY ENGINE & OPTIMIZATION")
    
    # 5. Tearsheet 
    run_step("ai_ls_allocation/engine/tearsheet.py", "PERFORMANCE REPORTING")

    # 6. Execution 
    run_step("ai_ls_allocation/execution.py", "ORDER GENERATION (OMS)")

    total_duration = time.time() - start_total
    finish_msg = f"\n{'='*60}\n🎉 PIPELINE FINISHED SUCCESSFULLY in {total_duration:.2f} seconds.\n   Check the log file at: {log_filename}\n{'='*60}"
    print(finish_msg)
    logging.info(finish_msg)

if __name__ == "__main__":
    main()