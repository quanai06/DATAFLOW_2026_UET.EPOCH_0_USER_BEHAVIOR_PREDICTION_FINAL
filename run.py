from __future__ import annotations

import importlib.util
import json
import signal
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
PRECOMPUTED_PARQUET = PROJECT_ROOT / "data" / "precomputed_orders.parquet"
PRECOMPUTED_CSV = PROJECT_ROOT / "data" / "precomputed_orders.csv"
PRECOMPUTED_SUMMARY = PROJECT_ROOT / "data" / "precomputed_dataset_summary.json"
BACKEND_HOST = "0.0.0.0"
BACKEND_PORT = 8000
FRONTEND_PORT = 8501


def _python_executable() -> str:
    if VENV_PYTHON.exists():
        return str(VENV_PYTHON)
    return sys.executable


def _ensure_dependencies() -> None:
    required_modules = {
        "fastapi": "Backend API",
        "uvicorn": "Backend server",
        "streamlit": "Frontend dashboard",
        "tensorflow": "Real model inference",
        "joblib": "Scaler loading",
        "requests": "Frontend API client",
        "plotly": "Frontend charts",
        "pandas": "Frontend batch table",
    }
    missing = [name for name in required_modules if not _module_available(name)]
    if not missing:
        return

    print("Thieu dependency de chay project:")
    for module_name in missing:
        print(f"  - {module_name}: {required_modules[module_name]}")
    print("\nHay cai dependencies truoc, vi du:")
    print("  pip install -r requirements.txt")
    raise SystemExit(1)


def _module_available(module_name: str) -> bool:
    python_exec = _python_executable()
    result = subprocess.run(
        [python_exec, "-c", f"import importlib.util; raise SystemExit(0 if importlib.util.find_spec('{module_name}') else 1)"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _start_process(command: list[str], name: str) -> subprocess.Popen[str]:
    print(f"[start] {name}: {' '.join(command)}")
    return subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        text=True,
    )


def _ensure_precomputed_data() -> None:
    needs_recompute = not (PRECOMPUTED_PARQUET.exists() or PRECOMPUTED_CSV.exists())
    if not needs_recompute and PRECOMPUTED_SUMMARY.exists():
        summary = json.loads(PRECOMPUTED_SUMMARY.read_text())
        processed_orders = int(summary.get("processed_orders", summary.get("total_orders", 0)))
        source_x_test_rows = int(summary.get("source_x_test_rows", 0))
        if source_x_test_rows and processed_orders < source_x_test_rows:
            needs_recompute = True

    if not needs_recompute:
        return

    python_exec = _python_executable()
    command = [python_exec, "precompute_x_test.py"]
    print("[setup] Dang tao/cap nhat precomputed file bang offline pipeline...")
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if result.returncode != 0:
        raise SystemExit("Khong tao duoc precomputed file. Dung khoi dong project.")


def main() -> None:
    _ensure_dependencies()
    _ensure_precomputed_data()
    python_exec = _python_executable()

    backend_command = [
        python_exec,
        "-m",
        "uvicorn",
        "src.app.backend.main:app",
        "--host",
        BACKEND_HOST,
        "--port",
        str(BACKEND_PORT),
        "--reload",
    ]
    frontend_command = [
        python_exec,
        "-m",
        "streamlit",
        "run",
        "src/app/frontend/app.py",
        "--server.port",
        str(FRONTEND_PORT),
        "--server.headless",
        "true",
    ]

    backend_process = _start_process(backend_command, "backend")
    time.sleep(2)
    frontend_process = _start_process(frontend_command, "frontend")

    processes = [
        ("backend", backend_process),
        ("frontend", frontend_process),
    ]

    print("\nProject dang chay:")
    print(f"  - Backend:  http://localhost:{BACKEND_PORT}/docs")
    print(f"  - Frontend: http://localhost:{FRONTEND_PORT}")
    print("Nhan Ctrl+C de dung ca 2 tien trinh.\n")

    try:
        while True:
            for name, process in processes:
                return_code = process.poll()
                if return_code is not None:
                    raise RuntimeError(f"{name} da dung voi exit code {return_code}.")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nDang dung project...")
    except RuntimeError as exc:
        print(f"\nLoi runtime: {exc}")
    finally:
        for _, process in processes:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
        for _, process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()


if __name__ == "__main__":
    main()
