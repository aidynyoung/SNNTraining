"""
C Kernel Bridge
===============
Python interface to C kernel via ctypes.

Provides:
- Fast LIF step (no Python overhead)
- Complete training step in C
- Deadline monitoring for real-time loops
"""

import torch
import ctypes
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import time


class CKernelBridge:
    """
    Bridge to compiled C kernel.
    
    Loads kernel.so/dll and provides Pythonic interface.
    """
    
    def __init__(self, kernel_path: Optional[str] = None):
        """
        Args:
            kernel_path: Path to compiled kernel shared library.
                        If None, attempts to find/compile it.
        """
        self.lib = None
        self._load_kernel(kernel_path)
        
    def _load_kernel(self, kernel_path: Optional[str]):
        """Load the C kernel library."""
        if kernel_path is None:
            # Try to find in package directory
            core_dir = Path(__file__).parent
            kernel_path = core_dir / "kernel.so"
        else:
            kernel_path = Path(kernel_path)
        
        if not kernel_path.exists():
            print(f"C kernel not found at {kernel_path}. Python fallback will be used.")
            print("To compile kernel: gcc -O3 -shared -fPIC -o kernel.so kernel.c -lm")
            return
        
        try:
            self.lib = ctypes.CDLL(str(kernel_path))
            print(f"Loaded C kernel from {kernel_path}")
        except OSError as e:
            print(f"Failed to load C kernel: {e}")
            print("Python fallback will be used.")
    
    def is_available(self) -> bool:
        """Check if C kernel is loaded and available."""
        return self.lib is not None
    
    def lif_step(
        self,
        current: torch.Tensor,
        v: torch.Tensor,
        spikes: torch.Tensor
    ) -> None:
        """
        C kernel LIF step (in-place).
        
        Args:
            current: Input current [n_neurons] (float32)
            v: Membrane potential [n_neurons] (in/out, float32)
            spikes: Spike buffer [n_neurons] (out, float32)
        """
        if self.lib is None:
            raise RuntimeError("C kernel not available")
        
        n = current.shape[0]
        
        # Set argument types
        self.lib.kernel_lif_step.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int
        ]
        
        # Call
        self.lib.kernel_lif_step(
            ctypes.cast(current.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(v.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(spikes.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            n
        )
    
    def training_step(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        W_in: torch.Tensor,
        W_rec: torch.Tensor,
        W_out: torch.Tensor,
        b_out: torch.Tensor,
        v: torch.Tensor,
        s_prev: torch.Tensor,
        e_fast: torch.Tensor,
        e_slow: torch.Tensor,
        lr_readout: float,
        lr_recurrent: float,
        tau_fast: float,
        tau_slow: float,
        alpha: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Complete training step in C.
        
        Returns (output, error) tensors.
        """
        if self.lib is None:
            raise RuntimeError("C kernel not available")
        
        input_size = x.shape[0]
        hidden_size = W_in.shape[0]
        output_size = W_out.shape[0]
        
        # Output buffers
        output = torch.zeros(output_size, dtype=torch.float32, device='cpu')
        error = torch.zeros(output_size, dtype=torch.float32, device='cpu')
        
        # Ensure all tensors are on CPU and float32
        def to_cpu_float(t):
            return t.detach().cpu().float().contiguous()
        
        x = to_cpu_float(x)
        target = to_cpu_float(target)
        W_in = to_cpu_float(W_in)
        W_rec = to_cpu_float(W_rec)
        W_out = to_cpu_float(W_out)
        b_out = to_cpu_float(b_out)
        v = to_cpu_float(v)
        s_prev = to_cpu_float(s_prev)
        e_fast = to_cpu_float(e_fast)
        e_slow = to_cpu_float(e_slow)
        
        # Set argument types
        self.lib.kernel_training_step.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # x
            ctypes.POINTER(ctypes.c_float),  # target
            ctypes.POINTER(ctypes.c_float),  # W_in
            ctypes.POINTER(ctypes.c_float),  # W_rec
            ctypes.POINTER(ctypes.c_float),  # W_out
            ctypes.POINTER(ctypes.c_float),  # b_out
            ctypes.POINTER(ctypes.c_float),  # v
            ctypes.POINTER(ctypes.c_float),  # s_prev
            ctypes.POINTER(ctypes.c_float),  # e_fast
            ctypes.POINTER(ctypes.c_float),  # e_slow
            ctypes.c_float,  # lr_readout
            ctypes.c_float,  # lr_recurrent
            ctypes.c_float,  # tau_fast
            ctypes.c_float,  # tau_slow
            ctypes.c_float,  # alpha
            ctypes.POINTER(ctypes.c_float),  # output
            ctypes.POINTER(ctypes.c_float),  # error
            ctypes.c_int,  # input_size
            ctypes.c_int,  # hidden_size
            ctypes.c_int,  # output_size
        ]
        
        # Call
        self.lib.kernel_training_step(
            ctypes.cast(x.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(target.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(W_in.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(W_rec.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(W_out.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(b_out.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(v.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(s_prev.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(e_fast.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(e_slow.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            lr_readout,
            lr_recurrent,
            tau_fast,
            tau_slow,
            alpha,
            ctypes.cast(output.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(error.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            input_size,
            hidden_size,
            output_size
        )
        
        return output, error


def compile_kernel(source_path: str = None, output_path: str = None) -> Path:
    """
    Compile C kernel to shared library.
    
    Args:
        source_path: Path to kernel.c (default: core/kernel.c)
        output_path: Path for output .so file (default: core/kernel.so)
        
    Returns:
        Path to compiled kernel
    """
    import subprocess
    
    if source_path is None:
        source_path = Path(__file__).parent / "kernel.c"
    else:
        source_path = Path(source_path)
    
    if output_path is None:
        output_path = source_path.with_suffix('.so')
    else:
        output_path = Path(output_path)
    
    # Detect platform
    import sys
    if sys.platform == 'darwin':
        # macOS
        cmd = [
            'gcc', '-O3', '-shared', '-fPIC',
            '-o', str(output_path),
            str(source_path),
            '-lm'
        ]
    elif sys.platform == 'win32':
        # Windows
        cmd = [
            'gcc', '-O3', '-shared',
            '-o', str(output_path.with_suffix('.dll')),
            str(source_path),
            '-lm'
        ]
        output_path = output_path.with_suffix('.dll')
    else:
        # Linux
        cmd = [
            'gcc', '-O3', '-shared', '-fPIC',
            '-o', str(output_path),
            str(source_path),
            '-lm'
        ]
    
    print(f"Compiling kernel: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Compilation failed:\n{result.stderr}")
        raise RuntimeError("Kernel compilation failed")
    
    print(f"Kernel compiled: {output_path}")
    return output_path


if __name__ == "__main__":
    # Test kernel bridge
    print("C Kernel Bridge Test")
    print("=" * 50)
    
    # Try to compile and load
    try:
        kernel_path = compile_kernel()
        bridge = CKerBridge(kernel_path)
        
        if bridge.is_available():
            print("C kernel loaded successfully!")
            
            # Test LIF step
            n = 128
            current = torch.randn(n).float()
            v = torch.zeros(n).float()
            spikes = torch.zeros(n).float()
            
            start = time.time()
            for _ in range(1000):
                bridge.lif_step(current, v, spikes)
            elapsed = time.time() - start
            
            print(f"1000 LIF steps in {elapsed*1000:.2f} ms ({elapsed:.4f} ms/step)")
        else:
            print("C kernel not available.")
    except Exception as e:
        print(f"Test failed: {e}")
