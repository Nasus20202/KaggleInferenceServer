"""Kaggle script: compile a static llama-server for Tesla T4 (sm_75).

The output in /kaggle/working/bin is attached to the server kernel, so each
session skips the ~25 min compile. Run it with the GPU T4 x2 accelerator
because the CUDA toolkit is only in the GPU image.
"""

import os
import shutil
import subprocess

# <params> (replaced by `kis build`)
# renovate: datasource=github-releases depName=ggml-org/llama.cpp
LLAMA_CPP_REF = "v0.6.0"
# </params>

SRC, OUT = "/tmp/llama.cpp", "/kaggle/working/bin"


def sh(cmd: str):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


sh(f"git clone --depth 1 --branch {LLAMA_CPP_REF} https://github.com/ggml-org/llama.cpp {SRC}")
# The linker needs libcuda; the Kaggle image keeps the driver copy outside the toolkit.
if not os.path.exists("/usr/local/cuda/lib64/libcuda.so"):
    sh("ln -sf /usr/local/nvidia/lib64/libcuda.so /usr/local/cuda/lib64/libcuda.so")

sh(
    f"cmake -S {SRC} -B {SRC}/build -DCMAKE_BUILD_TYPE=Release"
    " -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DGGML_NATIVE=OFF"
    " -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF"
    " -DCMAKE_EXE_LINKER_FLAGS=-L/usr/local/cuda/lib64/stubs"
)
sh(f"cmake --build {SRC}/build -j{os.cpu_count()} --target llama-server llama-bench")

os.makedirs(OUT, exist_ok=True)
for name in ("llama-server", "llama-bench"):
    shutil.copy2(f"{SRC}/build/bin/{name}", OUT)
sh(f"{OUT}/llama-server --version")
