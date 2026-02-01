import os
import sys

def _get_library_path():
    """Find the PJRT library relative to this file or in site-packages."""
    path = os.path.dirname(__file__)
    lib_name = "libmlx_pjrt_plugin.dylib" 
    full_path = os.path.join(path, lib_name)
    if os.path.exists(full_path):
        return full_path
    
    # Try site-packages location
    try:
        import jax_mlx
        pkg_path = os.path.dirname(jax_mlx.__file__)
        full_path = os.path.join(pkg_path, lib_name)
        if os.path.exists(full_path):
            return full_path
    except ImportError:
        pass
    
    return None

def initialize():
    """Called by JAX to initialize this plugin."""
    from jax._src import xla_bridge
    
    lib_path = _get_library_path()
    if lib_path:
        try:
            xla_bridge.register_plugin("mlx", library_path=lib_path)
            
            # Register MLIR lowerings for linalg primitives
            # These reuse the CPU lowering rules which emit LAPACK FFI custom_calls
            # that our C++ backend already handles
            _register_linalg_lowerings()
            
        except Exception as e:
            print(f"[MLX-Plugin] Registration failed: {e}")
            raise
    else:
        print("[MLX-Plugin] Library not found!")
        raise RuntimeError("MLX PJRT library not found")


def _register_linalg_lowerings():
    """Register MLIR lowerings for linalg primitives on the mlx platform."""
    from jax._src.interpreters import mlir
    from jax._src.lax import linalg as lax_linalg
    from functools import partial
    
    # Register primitives that use "target_name_prefix" pattern (CPU/GPU lowering)
    prefix_lowerings = [
        ('eigh_p', '_eigh_cpu_gpu_lowering'),
        ('svd_p', '_svd_cpu_gpu_lowering'),
        ('lu_p', '_lu_cpu_gpu_lowering'),
        ('geqrf_p', '_geqrf_cpu_gpu_lowering'),  # Used by QR
        ('householder_product_p', '_householder_product_cpu_gpu_lowering'),  # Used by QR
    ]
    
    for prim_name, lower_name in prefix_lowerings:
        try:
            prim = getattr(lax_linalg, prim_name, None)
            lower_func = getattr(lax_linalg, lower_name, None)
            
            if prim is not None and lower_func is not None:
                lowering = partial(lower_func, target_name_prefix="cpu")
                mlir.register_lowering(prim, lowering, platform="mlx")
        except Exception as e:
            print(f"[MLX-Plugin] Warning: Could not register {prim_name}: {e}")
    
    # Register cholesky - uses direct lowering without target_name_prefix
    try:
        cholesky_p = getattr(lax_linalg, 'cholesky_p', None)
        cholesky_lower = getattr(lax_linalg, '_cholesky_cpu_lowering', None)
        if cholesky_p is not None and cholesky_lower is not None:
            mlir.register_lowering(cholesky_p, cholesky_lower, platform="mlx")
    except Exception as e:
        print(f"[MLX-Plugin] Warning: Could not register cholesky_p: {e}")
    
    # Register triangular_solve - uses direct lowering without target_name_prefix
    try:
        trs_p = getattr(lax_linalg, 'triangular_solve_p', None)
        trs_lower = getattr(lax_linalg, '_triangular_solve_cpu_lower', None)
        if trs_p is not None and trs_lower is not None:
            mlir.register_lowering(trs_p, trs_lower, platform="mlx")
    except Exception as e:
        print(f"[MLX-Plugin] Warning: Could not register triangular_solve_p: {e}")
    
    # Register QR - uses mlir.lower_fun wrapper
    try:
        qr_p = getattr(lax_linalg, 'qr_p', None)
        qr_lower = getattr(lax_linalg, '_qr_lowering', None)
        if qr_p is not None and qr_lower is not None:
            mlir.register_lowering(qr_p, mlir.lower_fun(qr_lower, multiple_results=True), platform="mlx")
    except Exception as e:
        print(f"[MLX-Plugin] Warning: Could not register qr_p: {e}")
    
    # Register eig (general eigenvalue) - uses direct lowering without target_name_prefix
    try:
        eig_p = getattr(lax_linalg, 'eig_p', None)
        eig_lower = getattr(lax_linalg, '_eig_cpu_lowering', None)
        if eig_p is not None and eig_lower is not None:
            mlir.register_lowering(eig_p, eig_lower, platform="mlx")
    except Exception as e:
        print(f"[MLX-Plugin] Warning: Could not register eig_p: {e}")
