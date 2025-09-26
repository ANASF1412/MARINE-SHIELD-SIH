# PowerShell script to troubleshoot/fix TensorFlow DLL issues on Windows
# Run in an elevated PowerShell if needed.

Write-Host "[1/6] Python version:" -ForegroundColor Cyan
python -V

Write-Host "[2/6] Uninstall existing TensorFlow/keras packages..." -ForegroundColor Cyan
pip uninstall -y tensorflow tensorflow-intel tensorflow-gpu keras || Write-Host "Uninstall step completed"

Write-Host "[3/6] Clear pip cache..." -ForegroundColor Cyan
pip cache purge || Write-Host "Cache purge done"

Write-Host "[4/6] Install TensorFlow 2.20.0 (CPU) fresh..." -ForegroundColor Cyan
pip install --no-cache-dir tensorflow==2.20.0

Write-Host "[5/6] Verify TensorFlow import..." -ForegroundColor Cyan
python - << 'PY'
try:
	import tensorflow as tf
	print("TensorFlow version:", tf.__version__)
	print("Devices:", tf.config.list_physical_devices())
except Exception as e:
	print("[ERROR] TensorFlow import failed:", e)
PY

Write-Host "[6/6] If import still fails, install VC++ Redistributable (x64) and reboot: https://aka.ms/vs/17/release/vc_redist.x64.exe" -ForegroundColor Yellow 