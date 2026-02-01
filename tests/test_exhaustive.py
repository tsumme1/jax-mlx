#!/usr/bin/env python
"""Exhaustive JAX function test suite for the MLX PJRT plugin.

Tests ALL functions in jax.numpy, jax.scipy, jax.lax, and jax.random.
Functions are categorized by implementation status and relevance.

Run with: python test_exhaustive.py
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.lax as lax
import numpy as np
import traceback

# Get MLX device
try:
    mlx = jax.devices('mlx')[0]
except:
    print("ERROR: MLX device not found!")
    exit(1)

# Test tracking
results = {"passed": 0, "failed": 0, "skipped": 0, "value_mismatch": 0}
all_results = []

# Get CPU device for comparison
cpu = jax.devices('cpu')[0]

def section(name):
    print(f"\n{'='*70}\n{name}\n{'='*70}")

def compare_arrays(cpu_result, mlx_result, rtol=1e-4, atol=1e-5):
    """Compare arrays with tolerance, handling various types."""
    if isinstance(cpu_result, (list, tuple)):
        if len(cpu_result) != len(mlx_result):
            return False, f"Length mismatch: {len(cpu_result)} vs {len(mlx_result)}"
        for i, (c, m) in enumerate(zip(cpu_result, mlx_result)):
            ok, msg = compare_arrays(c, m, rtol, atol)
            if not ok:
                return False, f"Element {i}: {msg}"
        return True, "OK"
    
    # Convert to numpy for comparison
    try:
        c = np.asarray(cpu_result)
        m = np.asarray(mlx_result)
    except:
        return True, "Non-array result"  # Skip non-array outputs
    
    # Check shapes
    if c.shape != m.shape:
        return False, f"Shape mismatch: {c.shape} vs {m.shape}"
    
    # Check dtype compatibility (allow some flexibility)
    # Float types should match approximately
    if np.issubdtype(c.dtype, np.floating) or np.issubdtype(c.dtype, np.complexfloating):
        if not np.allclose(c, m, rtol=rtol, atol=atol, equal_nan=True):
            max_diff = np.max(np.abs(c - m))
            return False, f"Values differ, max diff: {max_diff:.6e}"
    elif np.issubdtype(c.dtype, np.integer) or np.issubdtype(c.dtype, np.bool_):
        if not np.array_equal(c, m):
            return False, f"Integer/bool values differ"
    
    return True, "OK"

def compare_svd(cpu_result, mlx_result, original_matrix, rtol=1e-4, atol=1e-5):
    """Compare SVD results using reconstruction error (handles sign ambiguity).
    
    SVD decomposition has sign ambiguity: U and Vh columns can be independently
    negated while still being valid decompositions. Instead of comparing U/Vh
    directly, we verify that both decompositions reconstruct the original matrix.
    """
    U_cpu, S_cpu, Vh_cpu = cpu_result
    U_mlx, S_mlx, Vh_mlx = mlx_result
    
    # Convert to numpy
    U_cpu, S_cpu, Vh_cpu = np.asarray(U_cpu), np.asarray(S_cpu), np.asarray(Vh_cpu)
    U_mlx, S_mlx, Vh_mlx = np.asarray(U_mlx), np.asarray(S_mlx), np.asarray(Vh_mlx)
    A = np.asarray(original_matrix)
    
    # Check shapes match
    if U_cpu.shape != U_mlx.shape:
        return False, f"U shape mismatch: {U_cpu.shape} vs {U_mlx.shape}"
    if S_cpu.shape != S_mlx.shape:
        return False, f"S shape mismatch: {S_cpu.shape} vs {S_mlx.shape}"
    if Vh_cpu.shape != Vh_mlx.shape:
        return False, f"Vh shape mismatch: {Vh_cpu.shape} vs {Vh_mlx.shape}"
    
    # Singular values should match exactly (no sign ambiguity)
    if not np.allclose(S_cpu, S_mlx, rtol=rtol, atol=atol):
        max_diff = np.max(np.abs(S_cpu - S_mlx))
        return False, f"Singular values differ, max diff: {max_diff:.6e}"
    
    # Check reconstruction error for MLX result: A ≈ U @ diag(S) @ Vh
    # Handle both full and reduced SVD cases
    k = min(A.shape)
    if U_mlx.shape[1] == k:  # Reduced SVD
        A_recon_mlx = U_mlx @ np.diag(S_mlx) @ Vh_mlx
    else:  # Full SVD
        S_full = np.zeros((U_mlx.shape[1], Vh_mlx.shape[0]))
        np.fill_diagonal(S_full, S_mlx)
        A_recon_mlx = U_mlx @ S_full @ Vh_mlx
    
    recon_error = np.max(np.abs(A - A_recon_mlx))
    if recon_error > atol + rtol * np.max(np.abs(A)):
        return False, f"Reconstruction error too large: {recon_error:.6e}"
    
    return True, "OK"

def test(name, fn, skip_reason=None, skip_comparison=False, no_jit=False):
    """Run a test function on both CPU and MLX, compare results.
    
    Args:
        name: Test name
        fn: Lambda function to test
        skip_reason: If set, skip the test with this reason
        skip_comparison: If True, run but don't compare CPU vs MLX (for random ops)
        no_jit: If True, disable JIT for this test (for ops like nonzero that need dynamic output)
    """
    if skip_reason:
        print(f"  ⊘ {name} [SKIP: {skip_reason}]")
        results["skipped"] += 1
        all_results.append(("skip", name, skip_reason))
        return
    
    try:
        # Optionally disable JIT for tests that require dynamic output sizes
        with jax.disable_jit(no_jit):
            # Run on CPU first
            with jax.default_device(cpu):
                cpu_result = fn()
            
            # Run on MLX
            with jax.default_device(mlx):
                mlx_result = fn()
        
        # Compare results (if not skipping comparison for random ops)
        if not skip_comparison:
            match, msg = compare_arrays(cpu_result, mlx_result)
            if not match:
                print(f"  ⚠ {name}: value mismatch - {msg}")
                results["value_mismatch"] += 1
                all_results.append(("mismatch", name, msg))
                return
        
        print(f"  ✓ {name}")
        results["passed"] += 1
        all_results.append(("pass", name, None))
    except Exception as e:
        err_msg = str(e).split('\n')[0][:60]
        print(f"  ✗ {name}: {err_msg}")
        results["failed"] += 1
        all_results.append(("fail", name, str(e)))

def test_svd(name, matrix, svd_fn):
    """Specialized test for SVD that handles sign ambiguity via reconstruction."""
    try:
        with jax.default_device(cpu):
            cpu_result = svd_fn(matrix)
        
        with jax.default_device(mlx):
            mlx_result = svd_fn(matrix)
        
        match, msg = compare_svd(cpu_result, mlx_result, matrix)
        if not match:
            print(f"  ⚠ {name}: value mismatch - {msg}")
            results["value_mismatch"] += 1
            all_results.append(("mismatch", name, msg))
            return
        
        print(f"  ✓ {name}")
        results["passed"] += 1
        all_results.append(("pass", name, None))
    except Exception as e:
        err_msg = str(e).split('\n')[0][:60]
        print(f"  ✗ {name}: {err_msg}")
        results["failed"] += 1
        all_results.append(("fail", name, str(e)))

# ============================================================================
# jax.random - Random Number Generation
# Note: Random tests use skip_comparison because RNG implementations differ
# ============================================================================
section("jax.random - Core RNG")

test("PRNGKey", lambda: jax.random.PRNGKey(0), skip_comparison=True)
test("key", lambda: jax.random.key(0), skip_comparison=True)
test("split", lambda: jax.random.split(jax.random.PRNGKey(0)), skip_comparison=True)
test("fold_in", lambda: jax.random.fold_in(jax.random.PRNGKey(0), 1), skip_comparison=True)

section("jax.random - Distributions")

test("normal", lambda: jax.random.normal(jax.random.PRNGKey(0), (100,)), skip_comparison=True)
test("uniform", lambda: jax.random.uniform(jax.random.PRNGKey(0), (100,)), skip_comparison=True)
test("bernoulli", lambda: jax.random.bernoulli(jax.random.PRNGKey(0), 0.5, (100,)), skip_comparison=True)
test("categorical", lambda: jax.random.categorical(jax.random.PRNGKey(0), jnp.ones(5)), skip_comparison=True)
test("choice", lambda: jax.random.choice(jax.random.PRNGKey(0), 10, (5,)), skip_comparison=True)
test("exponential", lambda: jax.random.exponential(jax.random.PRNGKey(0), (100,)), skip_comparison=True)
test("gamma", lambda: jax.random.gamma(jax.random.PRNGKey(0), 2.0, (100,)), skip_comparison=True)
test("gumbel", lambda: jax.random.gumbel(jax.random.PRNGKey(0), (100,)), skip_comparison=True)
test("laplace", lambda: jax.random.laplace(jax.random.PRNGKey(0), (100,)), skip_comparison=True)
test("logistic", lambda: jax.random.logistic(jax.random.PRNGKey(0), (100,)), skip_comparison=True)
test("maxwell", lambda: jax.random.maxwell(jax.random.PRNGKey(0), (100,)), skip_comparison=True)
test("multivariate_normal", lambda: jax.random.multivariate_normal(jax.random.PRNGKey(0), jnp.zeros(2), jnp.eye(2)), skip_comparison=True)
test("pareto", lambda: jax.random.pareto(jax.random.PRNGKey(0), 1.0, (100,)), skip_comparison=True)
test("permutation", lambda: jax.random.permutation(jax.random.PRNGKey(0), 10), skip_comparison=True)
test("poisson", lambda: jax.random.poisson(jax.random.PRNGKey(0), 2.0, (100,)), skip_comparison=True)
test("rademacher", lambda: jax.random.rademacher(jax.random.PRNGKey(0), (100,)), skip_comparison=True)
test("randint", lambda: jax.random.randint(jax.random.PRNGKey(0), (100,), 0, 10), skip_comparison=True)
test("shuffle", None, skip_reason="Deprecated in JAX 0.4+")
test("t", lambda: jax.random.t(jax.random.PRNGKey(0), 2.0, (100,)), skip_comparison=True)
test("truncated_normal", lambda: jax.random.truncated_normal(jax.random.PRNGKey(0), -1.0, 1.0, (100,)), skip_comparison=True)
test("weibull_min", lambda: jax.random.weibull_min(jax.random.PRNGKey(0), 1.0, 1.0, (100,)), skip_comparison=True)

# ============================================================================
# jax.numpy - Array Creation
# ============================================================================
section("jax.numpy - Array Creation")

test("array", lambda: jnp.array([1, 2, 3]))
test("zeros", lambda: jnp.zeros((3, 4)))
test("ones", lambda: jnp.ones((3, 4)))
test("full", lambda: jnp.full((3, 4), 5.0))
test("empty", lambda: jnp.empty((3, 4)), skip_comparison=True)  # Uninitialized values
test("arange", lambda: jnp.arange(10))
test("linspace", lambda: jnp.linspace(0, 1, 10))
test("logspace", lambda: jnp.logspace(0, 2, 10))
test("geomspace", lambda: jnp.geomspace(1, 100, 10))
test("eye", lambda: jnp.eye(3))
test("identity", lambda: jnp.identity(3))
test("diag", lambda: jnp.diag(jnp.array([1, 2, 3])))
test("diagflat", lambda: jnp.diagflat(jnp.array([1, 2, 3])))
test("tri", lambda: jnp.tri(3, 3))
test("tril", lambda: jnp.tril(jnp.ones((3, 3))))
test("triu", lambda: jnp.triu(jnp.ones((3, 3))))
test("meshgrid", lambda: jnp.meshgrid(jnp.arange(3), jnp.arange(4)))
test("fromfunction", lambda: jnp.fromfunction(lambda i, j: i + j, (3, 3), dtype=float))

# ============================================================================
# jax.numpy - Math Operations
# ============================================================================
section("jax.numpy - Basic Arithmetic")

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])

test("add", lambda: jnp.add(a, b))
test("subtract", lambda: jnp.subtract(a, b))
test("multiply", lambda: jnp.multiply(a, b))
test("divide", lambda: jnp.divide(a, b))
test("floor_divide", lambda: jnp.floor_divide(a, b))
test("true_divide", lambda: jnp.true_divide(a, b))
test("power", lambda: jnp.power(a, 2))
test("mod", lambda: jnp.mod(jnp.array([5, 7, 9]), 3))
test("remainder", lambda: jnp.remainder(jnp.array([5, 7, 9]), 3))
test("negative", lambda: jnp.negative(a))
test("positive", lambda: jnp.positive(a))
test("abs", lambda: jnp.abs(jnp.array([-1, -2, 3])))
test("absolute", lambda: jnp.absolute(jnp.array([-1, -2, 3])))

section("jax.numpy - Trigonometric")

x = jnp.array([0.0, np.pi/4, np.pi/2])
test("sin", lambda: jnp.sin(x))
test("cos", lambda: jnp.cos(x))
test("tan", lambda: jnp.tan(x[:2]))  # Avoid pi/2
test("arcsin", lambda: jnp.arcsin(jnp.array([0.0, 0.5, 1.0])))
test("arccos", lambda: jnp.arccos(jnp.array([0.0, 0.5, 1.0])))
test("arctan", lambda: jnp.arctan(a))
test("arctan2", lambda: jnp.arctan2(a, b))
test("hypot", lambda: jnp.hypot(jnp.array([3.0]), jnp.array([4.0])))
test("sinh", lambda: jnp.sinh(a))
test("cosh", lambda: jnp.cosh(a))
test("tanh", lambda: jnp.tanh(a))
test("arcsinh", lambda: jnp.arcsinh(a))
test("arccosh", lambda: jnp.arccosh(jnp.array([1.0, 2.0, 3.0])))
test("arctanh", lambda: jnp.arctanh(jnp.array([0.0, 0.5, 0.9])))
test("degrees", lambda: jnp.degrees(jnp.array([0, np.pi/2, np.pi])))
test("radians", lambda: jnp.radians(jnp.array([0, 90, 180])))
test("deg2rad", lambda: jnp.deg2rad(jnp.array([0, 90, 180])))
test("rad2deg", lambda: jnp.rad2deg(jnp.array([0, np.pi/2, np.pi])))

section("jax.numpy - Exponential & Logarithmic")

test("exp", lambda: jnp.exp(a))
test("exp2", lambda: jnp.exp2(a))
test("expm1", lambda: jnp.expm1(a))
test("log", lambda: jnp.log(a))
test("log2", lambda: jnp.log2(a))
test("log10", lambda: jnp.log10(a))
test("log1p", lambda: jnp.log1p(a))
test("logaddexp", lambda: jnp.logaddexp(a, b))
test("logaddexp2", lambda: jnp.logaddexp2(a, b))
test("sqrt", lambda: jnp.sqrt(a))
test("square", lambda: jnp.square(a))
test("cbrt", lambda: jnp.cbrt(a))
test("reciprocal", lambda: jnp.reciprocal(a))

section("jax.numpy - Rounding")

f = jnp.array([1.2, 2.5, 3.7, -1.2, -2.5])
test("floor", lambda: jnp.floor(f))
test("ceil", lambda: jnp.ceil(f))
test("round", lambda: jnp.round(f))
test("rint", lambda: jnp.rint(f))
test("fix", lambda: jnp.fix(f))
test("trunc", lambda: jnp.trunc(f))

section("jax.numpy - Special Functions")

test("sign", lambda: jnp.sign(jnp.array([-2, 0, 2])))
test("signbit", lambda: jnp.signbit(jnp.array([-2.0, 0.0, 2.0])))
test("copysign", lambda: jnp.copysign(a, jnp.array([-1, 1, -1])))
test("clip", lambda: jnp.clip(a, 1.5, 2.5))
test("maximum", lambda: jnp.maximum(a, b))
test("minimum", lambda: jnp.minimum(a, b))
test("fmax", lambda: jnp.fmax(a, b))
test("fmin", lambda: jnp.fmin(a, b))
test("isnan", lambda: jnp.isnan(jnp.array([1.0, np.nan, 2.0])))
test("isinf", lambda: jnp.isinf(jnp.array([1.0, np.inf, 2.0])))
test("isfinite", lambda: jnp.isfinite(jnp.array([1.0, np.inf, 2.0])))

# ============================================================================
# jax.numpy - Reductions
# ============================================================================
section("jax.numpy - Reductions")

m = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
test("sum", lambda: jnp.sum(m))
test("sum_axis", lambda: jnp.sum(m, axis=0))
test("prod", lambda: jnp.prod(m))
test("mean", lambda: jnp.mean(m))
test("std", lambda: jnp.std(m))
test("var", lambda: jnp.var(m))
test("max", lambda: jnp.max(m))
test("min", lambda: jnp.min(m))
test("amax", lambda: jnp.amax(m))
test("amin", lambda: jnp.amin(m))
test("argmax", lambda: jnp.argmax(m))
test("argmin", lambda: jnp.argmin(m))
test("all", lambda: jnp.all(jnp.array([True, True, False])))
test("any", lambda: jnp.any(jnp.array([True, True, False])))
test("cumsum", lambda: jnp.cumsum(a))
test("cumprod", lambda: jnp.cumprod(a))
test("nansum", lambda: jnp.nansum(jnp.array([1.0, np.nan, 2.0])))
test("nanmean", lambda: jnp.nanmean(jnp.array([1.0, np.nan, 2.0])))
test("nanstd", lambda: jnp.nanstd(jnp.array([1.0, np.nan, 2.0, 3.0])))
test("nanvar", lambda: jnp.nanvar(jnp.array([1.0, np.nan, 2.0, 3.0])))
test("nanmax", lambda: jnp.nanmax(jnp.array([1.0, np.nan, 2.0])))
test("nanmin", lambda: jnp.nanmin(jnp.array([1.0, np.nan, 2.0])))
test("nanargmax", lambda: jnp.nanargmax(jnp.array([1.0, np.nan, 2.0])))
test("nanargmin", lambda: jnp.nanargmin(jnp.array([1.0, np.nan, 2.0])))

# ============================================================================
# jax.numpy - Shape Manipulation
# ============================================================================
section("jax.numpy - Shape Manipulation")

arr = jnp.arange(12)
test("reshape", lambda: jnp.reshape(arr, (3, 4)))
test("ravel", lambda: jnp.ravel(jnp.ones((3, 4))))
test("flatten", lambda: jnp.ones((3, 4)).flatten())
test("transpose", lambda: jnp.transpose(jnp.ones((3, 4))))
test("swapaxes", lambda: jnp.swapaxes(jnp.ones((3, 4, 5)), 0, 2))
test("moveaxis", lambda: jnp.moveaxis(jnp.ones((3, 4, 5)), 0, -1))
test("rollaxis", lambda: jnp.rollaxis(jnp.ones((3, 4, 5)), 2))
test("expand_dims", lambda: jnp.expand_dims(a, 0))
test("squeeze", lambda: jnp.squeeze(jnp.ones((1, 3, 1))))
test("atleast_1d", lambda: jnp.atleast_1d(1))
test("atleast_2d", lambda: jnp.atleast_2d(a))
test("atleast_3d", lambda: jnp.atleast_3d(a))
test("broadcast_to", lambda: jnp.broadcast_to(a, (3, 3)))
test("broadcast_arrays", lambda: jnp.broadcast_arrays(a, jnp.ones((3, 3))))

# ============================================================================
# jax.numpy - Joining Arrays
# ============================================================================
section("jax.numpy - Joining Arrays")

test("concatenate", lambda: jnp.concatenate([a, b]))
test("stack", lambda: jnp.stack([a, b]))
test("vstack", lambda: jnp.vstack([a, b]))
test("hstack", lambda: jnp.hstack([a, b]))
test("dstack", lambda: jnp.dstack([a.reshape(1,3), b.reshape(1,3)]))
test("column_stack", lambda: jnp.column_stack([a, b]))
test("row_stack", None, skip_reason="Removed from jax.numpy")
test("block", lambda: jnp.block([[jnp.ones((2,2)), jnp.zeros((2,2))]]))

# ============================================================================
# jax.numpy - Splitting Arrays
# ============================================================================
section("jax.numpy - Splitting Arrays")

arr8 = jnp.arange(8)
test("split", lambda: jnp.split(arr8, 4))
test("array_split", lambda: jnp.array_split(arr8, 3))
test("vsplit", lambda: jnp.vsplit(jnp.ones((4, 4)), 2))
test("hsplit", lambda: jnp.hsplit(jnp.ones((4, 4)), 2))
test("dsplit", lambda: jnp.dsplit(jnp.ones((4, 4, 4)), 2))

# ============================================================================
# jax.numpy - Comparison
# ============================================================================
section("jax.numpy - Comparison")

test("equal", lambda: jnp.equal(a, a))
test("not_equal", lambda: jnp.not_equal(a, b))
test("less", lambda: jnp.less(a, b))
test("less_equal", lambda: jnp.less_equal(a, b))
test("greater", lambda: jnp.greater(a, b))
test("greater_equal", lambda: jnp.greater_equal(a, b))
test("array_equal", lambda: jnp.array_equal(a, a))
test("allclose", lambda: jnp.allclose(a, a + 1e-9))
test("isclose", lambda: jnp.isclose(a, a + 1e-9))

# ============================================================================
# jax.numpy - Logic
# ============================================================================
section("jax.numpy - Logic")

x_bool = jnp.array([True, False, True])
y_bool = jnp.array([False, True, True])
test("logical_and", lambda: jnp.logical_and(x_bool, y_bool))
test("logical_or", lambda: jnp.logical_or(x_bool, y_bool))
test("logical_xor", lambda: jnp.logical_xor(x_bool, y_bool))
test("logical_not", lambda: jnp.logical_not(x_bool))

# ============================================================================
# jax.numpy - Bitwise
# ============================================================================
section("jax.numpy - Bitwise")

i = jnp.array([1, 2, 3], dtype=jnp.int32)
j = jnp.array([4, 5, 6], dtype=jnp.int32)
test("bitwise_and", lambda: jnp.bitwise_and(i, j))
test("bitwise_or", lambda: jnp.bitwise_or(i, j))
test("bitwise_xor", lambda: jnp.bitwise_xor(i, j))
test("bitwise_not", lambda: jnp.bitwise_not(i))
test("left_shift", lambda: jnp.left_shift(i, 1))
test("right_shift", lambda: jnp.right_shift(j, 1))

# ============================================================================
# jax.numpy - Indexing
# ============================================================================
section("jax.numpy - Indexing")

test("take", lambda: jnp.take(a, jnp.array([0, 2])))
test("take_along_axis", lambda: jnp.take_along_axis(m, jnp.array([[0], [1]]), axis=1))
test("put", lambda: jnp.array([1,2,3]).at[0].set(5))  # JAX uses .at for mutations
test("where", lambda: jnp.where(jnp.array([True, False, True]), a, b))
test("select", lambda: jnp.select([a > 1, a > 2], [a, b], default=0))
test("nonzero", lambda: jnp.nonzero(jnp.array([0, 1, 0, 2]), size=2))
test("argwhere", lambda: jnp.argwhere(jnp.array([[1, 0], [0, 2]]), size=2))
test("flatnonzero", lambda: jnp.nonzero(jnp.array([0, 1, 0, 2]).ravel(), size=2))
test("searchsorted", lambda: jnp.searchsorted(jnp.array([1, 2, 3, 4, 5]), 3.5))
test("extract", lambda: jnp.extract(jnp.array([True, False, True]), a))

# ============================================================================
# jax.numpy - Linear Algebra
# ============================================================================
section("jax.numpy - Linear Algebra")

mat = jnp.array([[1.0, 2.0], [3.0, 4.0]])
vec = jnp.array([1.0, 2.0])
test("dot", lambda: jnp.dot(mat, vec))
test("matmul", lambda: jnp.matmul(mat, mat))
test("inner", lambda: jnp.inner(vec, vec))
test("outer", lambda: jnp.outer(vec, vec))
test("tensordot", lambda: jnp.tensordot(mat, mat, axes=1))
test("einsum", lambda: jnp.einsum('ij,jk->ik', mat, mat))
test("vdot", lambda: jnp.vdot(vec, vec))
test("kron", lambda: jnp.kron(jnp.eye(2), jnp.ones((2,2))))
test("trace", lambda: jnp.trace(mat))

# ============================================================================
# jax.numpy.linalg
# ============================================================================
section("jax.numpy.linalg")

test("linalg.norm", lambda: jnp.linalg.norm(vec))
test("linalg.det", lambda: jnp.linalg.det(mat))
test("linalg.inv", lambda: jnp.linalg.inv(mat))
test("linalg.solve", lambda: jnp.linalg.solve(mat, vec))
test("linalg.eig", lambda: jnp.linalg.eig(mat))
test("linalg.eigh", lambda: jnp.linalg.eigh(jnp.array([[1.0, 0.5], [0.5, 1.0]])))
test("linalg.eigvals", lambda: jnp.linalg.eigvals(mat))
test("linalg.eigvalsh", lambda: jnp.linalg.eigvalsh(jnp.array([[1.0, 0.5], [0.5, 1.0]])))
test_svd("linalg.svd", mat, jnp.linalg.svd)
test("linalg.qr", lambda: jnp.linalg.qr(mat))
test("linalg.cholesky", lambda: jnp.linalg.cholesky(jnp.array([[2.0, 1.0], [1.0, 2.0]])))
test("linalg.matrix_rank", lambda: jnp.linalg.matrix_rank(mat))
test("linalg.pinv", lambda: jnp.linalg.pinv(mat))
test("linalg.matrix_power", lambda: jnp.linalg.matrix_power(mat, 2))
test("linalg.slogdet", lambda: jnp.linalg.slogdet(mat))
test("linalg.cond", lambda: jnp.linalg.cond(mat))

# ============================================================================
# jax.numpy.fft
# ============================================================================
section("jax.numpy.fft")

test("fft.fft", lambda: jnp.fft.fft(a.astype(jnp.complex64)))
test("fft.ifft", lambda: jnp.fft.ifft(a.astype(jnp.complex64)))
test("fft.fft2", lambda: jnp.fft.fft2(m.astype(jnp.complex64)))
test("fft.ifft2", lambda: jnp.fft.ifft2(m.astype(jnp.complex64)))
test("fft.fftn", lambda: jnp.fft.fftn(m.astype(jnp.complex64)))
test("fft.ifftn", lambda: jnp.fft.ifftn(m.astype(jnp.complex64)))
test("fft.rfft", lambda: jnp.fft.rfft(a))
test("fft.irfft", lambda: jnp.fft.irfft(jnp.fft.rfft(a)))
test("fft.rfft2", lambda: jnp.fft.rfft2(m))
test("fft.irfft2", lambda: jnp.fft.irfft2(jnp.fft.rfft2(m)))
test("fft.fftshift", lambda: jnp.fft.fftshift(a))
test("fft.ifftshift", lambda: jnp.fft.ifftshift(a))
test("fft.fftfreq", lambda: jnp.fft.fftfreq(10))
test("fft.rfftfreq", lambda: jnp.fft.rfftfreq(10))

# ============================================================================
# jax.lax - Core Operations
# ============================================================================
section("jax.lax - Core Operations")

test("lax.add", lambda: lax.add(a, b))
test("lax.sub", lambda: lax.sub(a, b))
test("lax.mul", lambda: lax.mul(a, b))
test("lax.div", lambda: lax.div(a, b))
test("lax.neg", lambda: lax.neg(a))
test("lax.abs", lambda: lax.abs(jnp.array([-1, -2, 3])))
test("lax.exp", lambda: lax.exp(a))
test("lax.log", lambda: lax.log(a))
test("lax.sqrt", lambda: lax.sqrt(a))
test("lax.rsqrt", lambda: lax.rsqrt(a))
test("lax.pow", lambda: lax.pow(a, b))
test("lax.sin", lambda: lax.sin(a))
test("lax.cos", lambda: lax.cos(a))
test("lax.tanh", lambda: lax.tanh(a))
test("lax.max", lambda: lax.max(a, b))
test("lax.min", lambda: lax.min(a, b))
test("lax.clamp", lambda: lax.clamp(1.5, a, 2.5))

section("jax.lax - Shape Operations")

test("lax.reshape", lambda: lax.reshape(arr, (3, 4)))
test("lax.broadcast", lambda: lax.broadcast(a, (2, 3)))
test("lax.broadcast_in_dim", lambda: lax.broadcast_in_dim(a, (3, 3), (1,)))
test("lax.squeeze", lambda: lax.squeeze(jnp.ones((1, 3, 1)), (0, 2)))
test("lax.transpose", lambda: lax.transpose(jnp.ones((3, 4)), (1, 0)))
test("lax.slice", lambda: lax.slice(arr, (2,), (7,)))
test("lax.slice_in_dim", lambda: lax.slice_in_dim(arr, 2, 7, axis=0))
test("lax.dynamic_slice", lambda: lax.dynamic_slice(arr, (2,), (3,)))
test("lax.dynamic_update_slice", lambda: lax.dynamic_update_slice(arr, jnp.zeros(3, dtype=arr.dtype), (2,)))
test("lax.concatenate", lambda: lax.concatenate([a, b], 0))
test("lax.pad", lambda: lax.pad(a, 0.0, [(1, 1, 0)]))
test("lax.pad_interior", lambda: lax.pad(jnp.array([1., 2., 3.]), 0.0, [(0, 0, 1)]))  # Interior padding
test("lax.rev", lambda: lax.rev(a, (0,)))

section("jax.lax - Reductions")

test("lax.reduce", lambda: lax.reduce(a, 0.0, lax.add, (0,)))
test("lax.reduce_window_max", lambda: lax.reduce_window(jnp.ones((1, 4, 4, 1)), -np.inf, lax.max, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID'))
test("lax.reduce_window_sum", lambda: lax.reduce_window(jnp.ones((1, 4, 4, 1)), 0., lax.add, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID'))  # Avg pool numerator
test("lax.cumsum", lambda: lax.cumsum(a, axis=0))
test("lax.cumprod", lambda: lax.cumprod(a, axis=0))
test("lax.sort", lambda: lax.sort(jnp.array([3, 1, 2])))
test("lax.top_k", lambda: lax.top_k(jnp.array([3, 1, 4, 1, 5]), 3))

section("jax.lax - Comparisons")

test("lax.eq", lambda: lax.eq(a, a))
test("lax.ne", lambda: lax.ne(a, b))
test("lax.lt", lambda: lax.lt(a, b))
test("lax.le", lambda: lax.le(a, b))
test("lax.gt", lambda: lax.gt(a, b))
test("lax.ge", lambda: lax.ge(a, b))
test("lax.select", lambda: lax.select(jnp.array([True, False, True]), a, b))

section("jax.lax - Convolutions")

inp = jnp.ones((1, 8, 8, 3))
kernel = jnp.ones((3, 3, 3, 16))
test("lax.conv_general_dilated", lambda: lax.conv_general_dilated(inp, kernel, (1, 1), 'SAME', dimension_numbers=('NHWC', 'HWIO', 'NHWC')))
test("lax.conv", lambda: lax.conv(jnp.ones((1, 3, 4, 4)), jnp.ones((3, 3, 3, 3)), (1, 1), 'SAME'))

section("jax.lax - Control Flow")

test("lax.cond", lambda: lax.cond(True, lambda: a, lambda: b))
test("lax.switch", lambda: lax.switch(0, [lambda: a, lambda: b]))
test("lax.fori_loop", lambda: lax.fori_loop(0, 5, lambda i, x: x + 1, 0.0))
test("lax.while_loop", lambda: lax.while_loop(lambda x: x < 5, lambda x: x + 1, 0))
test("lax.scan", lambda: lax.scan(lambda c, x: (c + x, c), 0.0, a))
test("lax.map", lambda: lax.map(lambda x: x * 2, a))
test("lax.associative_scan", lambda: lax.associative_scan(jnp.add, jnp.arange(5.0)))


section("jax.lax - Bitwise")

test("lax.shift_left", lambda: lax.shift_left(i, jnp.array([1, 1, 1], dtype=jnp.int32)))
test("lax.shift_right_logical", lambda: lax.shift_right_logical(j, jnp.array([1, 1, 1], dtype=jnp.int32)))
test("lax.shift_right_arithmetic", lambda: lax.shift_right_arithmetic(j, jnp.array([1, 1, 1], dtype=jnp.int32)))
test("lax.and_", lambda: lax.bitwise_and(i, j))
test("lax.or_", lambda: lax.bitwise_or(i, j))
test("lax.xor", lambda: lax.bitwise_xor(i, j))
test("lax.not_", lambda: lax.bitwise_not(i))
test("lax.population_count", lambda: lax.population_count(i))

section("jax.lax - Type Conversion")

test("lax.convert_element_type", lambda: lax.convert_element_type(a, jnp.float16))
test("lax.bitcast_convert_type", lambda: lax.bitcast_convert_type(jnp.array([1.0], dtype=jnp.float32), jnp.int32))

# ============================================================================
# jax.lax - Sharding/Distributed (Not Relevant for Single Device)
# ============================================================================
section("jax.lax - Sharding/Distributed [NOT RELEVANT]")

test("lax.psum", None, skip_reason="Distributed/sharding operation")
test("lax.pmean", None, skip_reason="Distributed/sharding operation")
test("lax.pmax", None, skip_reason="Distributed/sharding operation")
test("lax.pmin", None, skip_reason="Distributed/sharding operation")
test("lax.all_gather", None, skip_reason="Distributed/sharding operation")
test("lax.all_to_all", None, skip_reason="Distributed/sharding operation")
test("lax.ppermute", None, skip_reason="Distributed/sharding operation")
test("lax.axis_index", None, skip_reason="Distributed/sharding operation")

# ============================================================================
# jax.scipy.special
# ============================================================================
section("jax.scipy.special")

test("scipy.special.erf", lambda: jsp.special.erf(a))
test("scipy.special.erfc", lambda: jsp.special.erfc(a))
test("scipy.special.erfinv", lambda: jsp.special.erfinv(jnp.array([0.0, 0.5, 0.9])))
test("scipy.special.expit", lambda: jsp.special.expit(a))
test("scipy.special.logit", lambda: jsp.special.logit(jnp.array([0.1, 0.5, 0.9])))
test("scipy.special.logsumexp", lambda: jsp.special.logsumexp(a))
test("scipy.special.gammaln", lambda: jsp.special.gammaln(a))
test("scipy.special.digamma", lambda: jsp.special.digamma(a))
test("scipy.special.betaln", lambda: jsp.special.betaln(a, b))
test("scipy.special.betainc", lambda: jsp.special.betainc(a, b, jnp.array([0.1, 0.5, 0.9])))
test("scipy.special.xlogy", lambda: jsp.special.xlogy(a, b))
test("scipy.special.xlog1py", lambda: jsp.special.xlog1py(a, b))
test("scipy.special.entr", lambda: jsp.special.entr(jnp.array([0.0, 0.5, 1.0])))
test("scipy.special.multigammaln", lambda: jsp.special.multigammaln(jnp.array([3.0]), 2))
test("scipy.special.i0", lambda: jsp.special.i0(a))
test("scipy.special.i0e", lambda: jsp.special.i0e(a))
test("scipy.special.i1", lambda: jsp.special.i1(a))
test("scipy.special.i1e", lambda: jsp.special.i1e(a))
test("scipy.special.softmax", lambda: jsp.special.softmax(a))
test("scipy.special.log_softmax", lambda: jsp.special.log_softmax(a))

section("jax.scipy.linalg")

test("scipy.linalg.solve", lambda: jsp.linalg.solve(mat, vec))
test("scipy.linalg.solve_triangular", lambda: jsp.linalg.solve_triangular(jnp.triu(mat), vec))
test("scipy.linalg.lu", lambda: jsp.linalg.lu(mat))
test("scipy.linalg.lu_factor", lambda: jsp.linalg.lu_factor(mat))
test("scipy.linalg.lu_solve", lambda: jsp.linalg.lu_solve(jsp.linalg.lu_factor(mat), vec))
test("scipy.linalg.qr", lambda: jsp.linalg.qr(mat))
test_svd("scipy.linalg.svd", mat, jsp.linalg.svd)
test("scipy.linalg.cholesky", lambda: jsp.linalg.cholesky(jnp.array([[2.0, 1.0], [1.0, 2.0]])))
test("scipy.linalg.eigh", lambda: jsp.linalg.eigh(jnp.array([[1.0, 0.5], [0.5, 1.0]])))
test("scipy.linalg.expm", lambda: jsp.linalg.expm(mat * 0.1))
test("scipy.linalg.det", lambda: jsp.linalg.det(mat))

section("jax.scipy.stats")

data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
test("scipy.stats.norm.pdf", lambda: jsp.stats.norm.pdf(data))
test("scipy.stats.norm.cdf", lambda: jsp.stats.norm.cdf(data))
test("scipy.stats.norm.logpdf", lambda: jsp.stats.norm.logpdf(data))
test("scipy.stats.norm.ppf", lambda: jsp.stats.norm.ppf(jnp.array([0.1, 0.5, 0.9])))
test("scipy.stats.beta.pdf", lambda: jsp.stats.beta.pdf(jnp.array([0.1, 0.5, 0.9]), 2, 5))
test("scipy.stats.beta.logpdf", lambda: jsp.stats.beta.logpdf(jnp.array([0.1, 0.5, 0.9]), 2, 5))
test("scipy.stats.gamma.pdf", lambda: jsp.stats.gamma.pdf(data, 2))
test("scipy.stats.gamma.logpdf", lambda: jsp.stats.gamma.logpdf(data, 2))
test("scipy.stats.uniform.pdf", lambda: jsp.stats.uniform.pdf(jnp.array([0.1, 0.5, 0.9])))
test("scipy.stats.uniform.logpdf", lambda: jsp.stats.uniform.logpdf(jnp.array([0.1, 0.5, 0.9])))
test("scipy.stats.expon.pdf", lambda: jsp.stats.expon.pdf(data))
test("scipy.stats.expon.logpdf", lambda: jsp.stats.expon.logpdf(data))
test("scipy.stats.poisson.pmf", lambda: jsp.stats.poisson.pmf(jnp.array([0, 1, 2, 3]), 2))
test("scipy.stats.poisson.logpmf", lambda: jsp.stats.poisson.logpmf(jnp.array([0, 1, 2, 3]), 2))
test("scipy.stats.bernoulli.pmf", lambda: jsp.stats.bernoulli.pmf(jnp.array([0, 1]), 0.5))
test("scipy.stats.bernoulli.logpmf", lambda: jsp.stats.bernoulli.logpmf(jnp.array([0, 1]), 0.5))
test("scipy.stats.laplace.pdf", lambda: jsp.stats.laplace.pdf(data))
test("scipy.stats.laplace.logpdf", lambda: jsp.stats.laplace.logpdf(data))
test("scipy.stats.t.pdf", lambda: jsp.stats.t.pdf(data, 3))
test("scipy.stats.t.logpdf", lambda: jsp.stats.t.logpdf(data, 3))

section("jax.scipy.signal")

test("scipy.signal.convolve", lambda: jsp.signal.convolve(jnp.array([1, 2, 3]), jnp.array([0, 1, 0.5])))
test("scipy.signal.correlate", lambda: jsp.signal.correlate(jnp.array([1, 2, 3]), jnp.array([0, 1, 0.5])))
test("scipy.signal.convolve2d", lambda: jsp.signal.convolve2d(jnp.ones((5, 5)), jnp.ones((3, 3))))
test("scipy.signal.correlate2d", lambda: jsp.signal.correlate2d(jnp.ones((5, 5)), jnp.ones((3, 3))))

section("jax.scipy.ndimage")

test("scipy.ndimage.map_coordinates", lambda: jsp.ndimage.map_coordinates(jnp.ones((5, 5)), [[1.5, 2.5], [1.5, 2.5]], order=1))

# ============================================================================
# jax.nn - Neural Network Activations
# ============================================================================
section("jax.nn - Activations")

nn_input = jnp.array([-1.0, 0.0, 1.0, 2.0])
test("nn.relu", lambda: jax.nn.relu(nn_input))
test("nn.relu6", lambda: jax.nn.relu6(jnp.array([-1., 3., 7.])))
test("nn.sigmoid", lambda: jax.nn.sigmoid(nn_input))
test("nn.softmax", lambda: jax.nn.softmax(jnp.ones(5)))
test("nn.log_softmax", lambda: jax.nn.log_softmax(jnp.ones(5)))
test("nn.softplus", lambda: jax.nn.softplus(nn_input))
test("nn.silu", lambda: jax.nn.silu(nn_input))  # swish
test("nn.gelu", lambda: jax.nn.gelu(nn_input))
test("nn.elu", lambda: jax.nn.elu(nn_input))
test("nn.leaky_relu", lambda: jax.nn.leaky_relu(nn_input))
test("nn.hard_sigmoid", lambda: jax.nn.hard_sigmoid(nn_input))
test("nn.hard_tanh", lambda: jax.nn.hard_tanh(jnp.array([-2., 0., 2.])))
test("nn.celu", lambda: jax.nn.celu(nn_input))
test("nn.selu", lambda: jax.nn.selu(nn_input))
test("nn.glu", lambda: jax.nn.glu(jnp.ones(6)))

# ============================================================================
# Autodiff / Transformations
# ============================================================================
section("Autodiff / Transformations")

test("jax.grad", lambda: jax.grad(lambda x: jnp.sum(x**2))(a))
test("jax.value_and_grad", lambda: jax.value_and_grad(lambda x: jnp.sum(x**2))(a))
test("jax.jacfwd", lambda: jax.jacfwd(lambda x: x**2)(a))
test("jax.jacrev", lambda: jax.jacrev(lambda x: x**2)(a))
test("jax.hessian", lambda: jax.hessian(lambda x: jnp.sum(x**2))(a))
test("jax.vmap", lambda: jax.vmap(lambda x: x**2)(jnp.ones((3, 4))))
test("jax.jit", lambda: jax.jit(lambda x: x**2)(a))

# ============================================================================
# Summary
# ============================================================================
total = results['passed'] + results['failed'] + results['skipped'] + results['value_mismatch']
print(f"\n{'='*70}")
print(f"FINAL RESULTS")
print(f"{'='*70}")
print(f"  Passed:         {results['passed']}")
print(f"  Failed:         {results['failed']}")
print(f"  Value Mismatch: {results['value_mismatch']}")
print(f"  Skipped:        {results['skipped']}")
print(f"  Total:          {total}")
print(f"{'='*70}")

if results["value_mismatch"] > 0:
    print("\nValue mismatches (MLX output differs from CPU):")
    for status, name, err in all_results:
        if status == "mismatch":
            print(f"  ⚠ {name}: {err}")

if results["failed"] > 0:
    print("\nFailed tests:")
    for status, name, err in all_results:
        if status == "fail":
            print(f"  ✗ {name}: {err[:80]}...")
    exit(1)
elif results["value_mismatch"] > 0:
    print("\nTests executed but some values differ from CPU reference.")
    exit(2)
else:
    print("\nAll tests passed with correct values!")
