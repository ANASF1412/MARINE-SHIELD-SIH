## TensorFlow DLL Load Failure (Windows) - Troubleshooting Guide

Symptoms: ImportError: DLL load failed while importing _pywrap_tensorflow_internal

1) Verify versions
- Python: 3.9–3.12 recommended for TF 2.20
- TensorFlow: python -c "import tensorflow as tf; print(tf.__version__)"

2) Install Microsoft Visual C++ Redistributable (x64)
- Download and install: https://aka.ms/vs/17/release/vc_redist.x64.exe
- Reboot the machine afterwards

3) CPU vs GPU TensorFlow
- CPU: pip install tensorflow
- GPU: ensure compatible CUDA/cuDNN for your TF version and updated NVIDIA drivers
  - Compatibility: https://www.tensorflow.org/install/source#gpu
  - Verify CUDA: nvcc --version (if installed)
  - Verify cuDNN: check presence under C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA

4) Clean reinstall TensorFlow
- pip uninstall -y tensorflow tensorflow-intel tensorflow-gpu keras
- pip cache purge
- pip install --no-cache-dir tensorflow==2.20.0

5) Use a fresh virtual environment
- python -m venv .venv
- .venv\Scripts\activate
- python -m pip install --upgrade pip
- pip install -r requirements.txt

6) Check PATH conflicts
- Ensure no stray DLLs from old CUDA/cuDNN in PATH precedence
- Prefer launching from a clean venv shell

7) Additional references
- Errors guide: https://www.tensorflow.org/install/errors
- GPU setup: https://www.tensorflow.org/install/pip#windows-native

If issues persist, share the outputs of:
- python -V
- pip show tensorflow
- python -c "import tensorflow as tf; print(tf.config.list_physical_devices())" 