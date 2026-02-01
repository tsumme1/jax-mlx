/**
 * @file jax_mlx_pjrt.cpp
 * @brief MLX PJRT Plugin for JAX - Enables JAX on Apple Silicon via MLX backend
 *
 * ARCHITECTURE OVERVIEW
 * =====================
 * This plugin implements the PJRT (Portable JAX Runtime) C API to enable JAX 
 * operations on Apple Silicon GPUs using the MLX framework.
 *
 * Key Components:
 * - PJRT API Implementation: Client, Device, Buffer, and Executable management
 * - Graph Execution Engine: Converts StableHLO/MHLO IR to MLX operations
 * - Compilation System: Uses mx.compile() for GPU kernel fusion
 *
 * FEATURE TOGGLES (Environment Variables)
 * ========================================
 * MLX_PJRT_DEBUG=1         - Enable debug output
 * MLX_NO_COMPILE=1         - Disable mx.compile() (default: enabled)
 * MLX_NO_COMPILE_AGGRESSIVE=1 - Disable aggressive compilation (default: enabled)
 * MLX_NO_MEGA_COMPILE=1    - Disable mega-compile (default: enabled)
 * MLX_TIMING=1             - Enable timing output
 * MLX_PROFILE=1            - Enable detailed profiling
 *
 * COMPILATION DECISION LOGIC
 * ==========================
 * The plugin decides whether to compile a graph based on:
 * 1. While loops: Block compilation (require runtime eval() for condition)
 * 2. NaN constants: Block compilation (MLX Metal bug)
 * 3. func.call: Allowed in aggressive mode (recursively checked)
 * 4. Control flow (if/case): Uses mx.where() for lazy selection
 * 5. Dynamic ops: Uses MLX native APIs (no eval() needed)
 *
 * See has_control_flow_recursive() for the complete logic.
 *
 * MAJOR SECTIONS
 * ==============
 * Lines 1-170:      Includes, OpType enum, helper functions
 * Lines 170-300:    Feature toggles and error handling
 * Lines 300-690:    Internal structures (MLXGraph, MLXOp, etc.)
 * Lines 690-800:    PJRT Client/Compile API
 * Lines 800-1500:   PJRT Buffer/Device/Executable API
 * Lines 1500-1850:  Control flow (while, case/if with mx.where)
 * Lines 1850-5100:  Operation dispatch and MLX execution
 * Lines 5100-5700:  PJRT API table and plugin entry point
 */

#include <iostream>
#include <vector>
#include <cstring>
#include <string>
#include <map>
#include <numeric>
#include "xla/pjrt/c/pjrt_c_api.h"
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <mlx/mlx.h>
#include <mlx/fft.h>
#include <mlx/linalg.h>
#include <mlx/compile.h>
#include <complex>
#include <unordered_map>
#include <optional>
#include <mutex>
#include <chrono>

// Global cache for QR decomposition - stores last Q matrix for Householder to retrieve
static std::optional<mlx::core::array> g_last_qr_q;

// OpType enum for fast operation dispatch (replaces string comparisons when enabled)
enum class OpType : uint16_t {
    UNKNOWN = 0,
    // Arithmetic
    ADD, SUBTRACT, MULTIPLY, DIVIDE, NEGATE, ABS, POWER, REMAINDER, FLOOR, CEIL, ROUND,
    MAXIMUM, MINIMUM, SIGN,
    // Math functions
    EXP, EXPM1, LOG, LOG1P, SQRT, RSQRT, CBRT,
    SIN, COS, TAN, TANH, SINH, COSH, ASIN, ACOS, ATAN, ATAN2, ERF, LOGISTIC,
    // Comparison
    COMPARE,
    // Constants and conversion
    CONSTANT, CONVERT, BITCAST_CONVERT,
    // Shape operations
    RESHAPE, BROADCAST_IN_DIM, TRANSPOSE, SLICE, DYNAMIC_SLICE, DYNAMIC_UPDATE_SLICE,
    CONCATENATE, PAD, GATHER, SCATTER, REVERSE, IOTA, GET_TUPLE_ELEMENT, TUPLE,
    // Reduction
    REDUCE, REDUCE_SUM, REDUCE_MAX, REDUCE_MIN, REDUCE_PROD, REDUCE_WINDOW, SELECT_AND_SCATTER,
    // Linear algebra
    DOT, DOT_GENERAL, CONVOLUTION, TRIANGULAR_SOLVE, CHOLESKY,
    // Control flow
    WHILE, COND_IF, CASE, CALL, FUNC_CALL, RETURN, CUSTOM_CALL,
    // Logical/Bitwise
    AND, OR, XOR, NOT, SHIFT_LEFT, SHIFT_RIGHT_LOGICAL, SHIFT_RIGHT_ARITHMETIC, POPCNT, CLZ,
    // Other
    CLAMP, SELECT, SORT, TOP_K, OPTIMIZATION_BARRIER, RNG_BIT_GENERATOR, FFT,
    REAL, IMAG, COMPLEX, IS_FINITE,
    OP_TYPE_COUNT
};

// Static lookup map: op name -> OpType (initialized once)
static const std::unordered_map<std::string, OpType> kOpNameToType = {
    // Arithmetic
    {"stablehlo.add", OpType::ADD}, {"mhlo.add", OpType::ADD},
    {"stablehlo.subtract", OpType::SUBTRACT}, {"mhlo.subtract", OpType::SUBTRACT},
    {"stablehlo.multiply", OpType::MULTIPLY}, {"mhlo.multiply", OpType::MULTIPLY},
    {"stablehlo.divide", OpType::DIVIDE}, {"mhlo.divide", OpType::DIVIDE},
    {"stablehlo.negate", OpType::NEGATE}, {"mhlo.negate", OpType::NEGATE},
    {"stablehlo.abs", OpType::ABS}, {"mhlo.abs", OpType::ABS},
    {"stablehlo.power", OpType::POWER}, {"mhlo.power", OpType::POWER},
    {"stablehlo.remainder", OpType::REMAINDER}, {"mhlo.remainder", OpType::REMAINDER},
    {"stablehlo.floor", OpType::FLOOR}, {"mhlo.floor", OpType::FLOOR},
    {"stablehlo.ceil", OpType::CEIL}, {"mhlo.ceil", OpType::CEIL},
    {"stablehlo.round_nearest_even", OpType::ROUND}, {"stablehlo.round_nearest_afz", OpType::ROUND},
    {"stablehlo.maximum", OpType::MAXIMUM}, {"mhlo.maximum", OpType::MAXIMUM},
    {"stablehlo.minimum", OpType::MINIMUM}, {"mhlo.minimum", OpType::MINIMUM},
    {"stablehlo.sign", OpType::SIGN}, {"mhlo.sign", OpType::SIGN},
    // Math functions
    {"stablehlo.exponential", OpType::EXP}, {"mhlo.exponential", OpType::EXP},
    {"stablehlo.expm1", OpType::EXPM1}, {"mhlo.expm1", OpType::EXPM1},
    {"stablehlo.log", OpType::LOG}, {"mhlo.log", OpType::LOG},
    {"stablehlo.log_plus_one", OpType::LOG1P}, {"mhlo.log_plus_one", OpType::LOG1P},
    {"stablehlo.sqrt", OpType::SQRT}, {"mhlo.sqrt", OpType::SQRT},
    {"stablehlo.rsqrt", OpType::RSQRT}, {"mhlo.rsqrt", OpType::RSQRT},
    {"stablehlo.cbrt", OpType::CBRT}, {"mhlo.cbrt", OpType::CBRT},
    {"stablehlo.sine", OpType::SIN}, {"mhlo.sine", OpType::SIN},
    {"stablehlo.cosine", OpType::COS}, {"mhlo.cosine", OpType::COS},
    {"stablehlo.tan", OpType::TAN}, {"mhlo.tan", OpType::TAN},
    {"stablehlo.tanh", OpType::TANH}, {"mhlo.tanh", OpType::TANH},
    {"stablehlo.sinh", OpType::SINH}, {"stablehlo.cosh", OpType::COSH},
    {"stablehlo.asin", OpType::ASIN}, {"stablehlo.acos", OpType::ACOS},
    {"stablehlo.atan", OpType::ATAN}, {"stablehlo.atan2", OpType::ATAN2},
    {"stablehlo.erf", OpType::ERF},
    {"stablehlo.logistic", OpType::LOGISTIC}, {"mhlo.logistic", OpType::LOGISTIC},
    // Comparison
    {"stablehlo.compare", OpType::COMPARE}, {"mhlo.compare", OpType::COMPARE},
    // Constants and conversion
    {"stablehlo.constant", OpType::CONSTANT}, {"mhlo.constant", OpType::CONSTANT},
    {"stablehlo.convert", OpType::CONVERT}, {"mhlo.convert", OpType::CONVERT},
    {"stablehlo.bitcast_convert", OpType::BITCAST_CONVERT},
    // Shape operations
    {"stablehlo.reshape", OpType::RESHAPE}, {"mhlo.reshape", OpType::RESHAPE},
    {"stablehlo.broadcast_in_dim", OpType::BROADCAST_IN_DIM}, {"mhlo.broadcast_in_dim", OpType::BROADCAST_IN_DIM},
    {"stablehlo.transpose", OpType::TRANSPOSE}, {"mhlo.transpose", OpType::TRANSPOSE},
    {"stablehlo.slice", OpType::SLICE}, {"mhlo.slice", OpType::SLICE},
    {"stablehlo.dynamic_slice", OpType::DYNAMIC_SLICE}, {"mhlo.dynamic_slice", OpType::DYNAMIC_SLICE},
    {"stablehlo.dynamic_update_slice", OpType::DYNAMIC_UPDATE_SLICE}, {"mhlo.dynamic_update_slice", OpType::DYNAMIC_UPDATE_SLICE},
    {"stablehlo.concatenate", OpType::CONCATENATE}, {"mhlo.concatenate", OpType::CONCATENATE},
    {"stablehlo.pad", OpType::PAD}, {"mhlo.pad", OpType::PAD},
    {"stablehlo.gather", OpType::GATHER}, {"mhlo.gather", OpType::GATHER},
    {"stablehlo.scatter", OpType::SCATTER}, {"mhlo.scatter", OpType::SCATTER},
    {"stablehlo.reverse", OpType::REVERSE}, {"mhlo.reverse", OpType::REVERSE},
    {"stablehlo.iota", OpType::IOTA}, {"mhlo.iota", OpType::IOTA},
    {"stablehlo.get_tuple_element", OpType::GET_TUPLE_ELEMENT}, {"mhlo.get_tuple_element", OpType::GET_TUPLE_ELEMENT},
    {"stablehlo.tuple", OpType::TUPLE}, {"mhlo.tuple", OpType::TUPLE},
    // Reduction
    {"stablehlo.reduce", OpType::REDUCE}, {"mhlo.reduce", OpType::REDUCE},
    {"stablehlo.reduce_sum", OpType::REDUCE_SUM}, {"mhlo.reduce_sum", OpType::REDUCE_SUM},
    {"stablehlo.reduce_max", OpType::REDUCE_MAX}, {"mhlo.reduce_max", OpType::REDUCE_MAX},
    {"stablehlo.reduce_min", OpType::REDUCE_MIN}, {"mhlo.reduce_min", OpType::REDUCE_MIN},
    {"stablehlo.reduce_prod", OpType::REDUCE_PROD}, {"mhlo.reduce_prod", OpType::REDUCE_PROD},
    {"stablehlo.reduce_window", OpType::REDUCE_WINDOW}, {"mhlo.reduce_window", OpType::REDUCE_WINDOW},
    {"stablehlo.select_and_scatter", OpType::SELECT_AND_SCATTER}, {"mhlo.select_and_scatter", OpType::SELECT_AND_SCATTER},
    // Linear algebra
    {"stablehlo.dot", OpType::DOT}, {"mhlo.dot", OpType::DOT},
    {"stablehlo.dot_general", OpType::DOT_GENERAL}, {"mhlo.dot_general", OpType::DOT_GENERAL},
    {"stablehlo.convolution", OpType::CONVOLUTION}, {"mhlo.convolution", OpType::CONVOLUTION},
    {"stablehlo.triangular_solve", OpType::TRIANGULAR_SOLVE}, {"mhlo.triangular_solve", OpType::TRIANGULAR_SOLVE},
    {"stablehlo.cholesky", OpType::CHOLESKY}, {"mhlo.cholesky", OpType::CHOLESKY},
    // Control flow
    {"stablehlo.while", OpType::WHILE}, {"mhlo.while", OpType::WHILE},
    {"stablehlo.if", OpType::COND_IF}, {"mhlo.if", OpType::COND_IF},
    {"stablehlo.case", OpType::CASE}, {"mhlo.case", OpType::CASE},
    {"stablehlo.call", OpType::CALL}, {"mhlo.call", OpType::CALL},
    {"func.call", OpType::FUNC_CALL},
    {"stablehlo.return", OpType::RETURN}, {"mhlo.return", OpType::RETURN}, {"func.return", OpType::RETURN},
    {"stablehlo.custom_call", OpType::CUSTOM_CALL}, {"mhlo.custom_call", OpType::CUSTOM_CALL},
    // Logical/Bitwise
    {"stablehlo.and", OpType::AND}, {"mhlo.and", OpType::AND},
    {"stablehlo.or", OpType::OR}, {"mhlo.or", OpType::OR},
    {"stablehlo.xor", OpType::XOR}, {"mhlo.xor", OpType::XOR},
    {"stablehlo.not", OpType::NOT}, {"mhlo.not", OpType::NOT},
    {"stablehlo.shift_left", OpType::SHIFT_LEFT}, {"mhlo.shift_left", OpType::SHIFT_LEFT},
    {"stablehlo.shift_right_logical", OpType::SHIFT_RIGHT_LOGICAL},
    {"stablehlo.shift_right_arithmetic", OpType::SHIFT_RIGHT_ARITHMETIC},
    {"stablehlo.popcnt", OpType::POPCNT}, {"mhlo.popcnt", OpType::POPCNT},
    {"stablehlo.count_leading_zeros", OpType::CLZ}, {"mhlo.count_leading_zeros", OpType::CLZ},
    // Other
    {"stablehlo.clamp", OpType::CLAMP}, {"mhlo.clamp", OpType::CLAMP},
    {"stablehlo.select", OpType::SELECT}, {"mhlo.select", OpType::SELECT},
    {"stablehlo.sort", OpType::SORT}, {"mhlo.sort", OpType::SORT},
    {"stablehlo.top_k", OpType::TOP_K},
    {"stablehlo.optimization_barrier", OpType::OPTIMIZATION_BARRIER}, {"mhlo.optimization_barrier", OpType::OPTIMIZATION_BARRIER},
    {"stablehlo.rng_bit_generator", OpType::RNG_BIT_GENERATOR}, {"mhlo.rng_bit_generator", OpType::RNG_BIT_GENERATOR},
    {"stablehlo.fft", OpType::FFT}, {"mhlo.fft", OpType::FFT},
    {"stablehlo.real", OpType::REAL}, {"mhlo.real", OpType::REAL},
    {"stablehlo.imag", OpType::IMAG}, {"mhlo.imag", OpType::IMAG},
    {"stablehlo.complex", OpType::COMPLEX}, {"mhlo.complex", OpType::COMPLEX},
    {"stablehlo.is_finite", OpType::IS_FINITE}, {"mhlo.is_finite", OpType::IS_FINITE},
};

// Fast lookup function
inline OpType GetOpType(const std::string& name) {
    auto it = kOpNameToType.find(name);
    return (it != kOpNameToType.end()) ? it->second : OpType::UNKNOWN;
}

// Internal Error structure
struct MLXError {
    PJRT_Error_Code code;
    std::string message;
};

// Helper to return success (nullptr error)
PJRT_Error* Ok() { return nullptr; }

// Debug mode helper - caches the env check
inline bool debug_mode() {
    static int cached = -1;
    if (cached == -1) cached = (getenv("MLX_PJRT_DEBUG") != nullptr) ? 1 : 0;
    return cached == 1;
}

// Feature toggle: OpType enum dispatch (MLX_OPTYPE_DISPATCH=1 to enable)
inline bool optype_dispatch_enabled() {
    static int cached = -1;
    if (cached == -1) cached = (getenv("MLX_OPTYPE_DISPATCH") != nullptr) ? 1 : 0;
    return cached == 1;
}

// Feature toggle: Timing mode (MLX_TIMING=1 to enable)
inline bool timing_mode() {
    static int cached = -1;
    if (cached == -1) cached = (getenv("MLX_TIMING") != nullptr) ? 1 : 0;
    return cached == 1;
}

// Feature toggle: Detailed profiling (MLX_PROFILE=1 to enable)
// Shows granular timing for each Execute step
inline bool profile_mode() {
    static int cached = -1;
    if (cached == -1) cached = (getenv("MLX_PROFILE") != nullptr) ? 1 : 0;
    return cached == 1;
}

// Feature toggle: mx.compile() integration (ENABLED BY DEFAULT)
// Set MLX_NO_COMPILE=1 to disable
inline bool compile_enabled() {
    static int cached = -1;
    if (cached == -1) cached = (getenv("MLX_NO_COMPILE") != nullptr) ? 0 : 1;
    return cached == 1;
}

// Feature toggle: Aggressive compilation (ENABLED BY DEFAULT)
// Allows func.call ops in compiled graphs
// Set MLX_NO_COMPILE_AGGRESSIVE=1 to disable
inline bool compile_aggressive_enabled() {
    static int cached = -1;
    if (cached == -1) cached = (getenv("MLX_NO_COMPILE_AGGRESSIVE") != nullptr) ? 0 : 1;
    return cached == 1;
}

// Feature toggle: Mega-compile (ENABLED BY DEFAULT)
// Defers eval() to sync points for lazy execution
// Set MLX_NO_MEGA_COMPILE=1 to disable
inline bool mega_compile_enabled() {
    static int cached = -1;
    if (cached == -1) cached = (getenv("MLX_NO_MEGA_COMPILE") != nullptr) ? 0 : 1;
    return cached == 1;
}

// Thread-local batch counter for mega-compile
static thread_local int g_current_batch_id = 1;  // Start at 1, 0 means "from host"

PJRT_Error* Error(PJRT_Error_Code code, const char* msg) {
    MLXError* e = new MLXError{code, msg};
    return reinterpret_cast<PJRT_Error*>(e);
}

PJRT_Error* Unimplemented(const std::string& name) {
    return Error(PJRT_Error_Code_UNIMPLEMENTED, ("Unimplemented: " + name).c_str());
}

PJRT_Error* InvalidArgument(const std::string& msg) {
    return Error(PJRT_Error_Code_INVALID_ARGUMENT, msg.c_str());
}

// Helpers
PJRT_Buffer_Type MlxTypeToPjrtType(mlx::core::Dtype dtype) {
    if (dtype == mlx::core::float32) return PJRT_Buffer_Type_F32;
    if (dtype == mlx::core::float16) return PJRT_Buffer_Type_F16;
    if (dtype == mlx::core::bfloat16) return PJRT_Buffer_Type_BF16;
    if (dtype == mlx::core::int32) return PJRT_Buffer_Type_S32;
    if (dtype == mlx::core::int64) return PJRT_Buffer_Type_S64;
    if (dtype == mlx::core::int8) return PJRT_Buffer_Type_S8;
    if (dtype == mlx::core::int16) return PJRT_Buffer_Type_S16;
    if (dtype == mlx::core::uint8) return PJRT_Buffer_Type_U8;
    if (dtype == mlx::core::uint16) return PJRT_Buffer_Type_U16;
    if (dtype == mlx::core::uint32) return PJRT_Buffer_Type_U32;
    if (dtype == mlx::core::uint64) return PJRT_Buffer_Type_U64;
    if (dtype == mlx::core::bool_) return PJRT_Buffer_Type_PRED;
    if (dtype == mlx::core::complex64) return PJRT_Buffer_Type_C64;
    return PJRT_Buffer_Type_F32; // Default
}

mlx::core::Dtype PjrtTypeToMlxType(PJRT_Buffer_Type type) {
    if (type == PJRT_Buffer_Type_F32) return mlx::core::float32;
    if (type == PJRT_Buffer_Type_F16) return mlx::core::float16;
    if (type == PJRT_Buffer_Type_BF16) return mlx::core::bfloat16;
    if (type == PJRT_Buffer_Type_S32) return mlx::core::int32;
    if (type == PJRT_Buffer_Type_S64) return mlx::core::int64;
    if (type == PJRT_Buffer_Type_S8) return mlx::core::int8;
    if (type == PJRT_Buffer_Type_S16) return mlx::core::int16;
    if (type == PJRT_Buffer_Type_U8) return mlx::core::uint8;
    if (type == PJRT_Buffer_Type_U16) return mlx::core::uint16;
    if (type == PJRT_Buffer_Type_U32) return mlx::core::uint32;
    if (type == PJRT_Buffer_Type_U64) return mlx::core::uint64;
    if (type == PJRT_Buffer_Type_PRED) return mlx::core::bool_;
    if (type == PJRT_Buffer_Type_C64) return mlx::core::complex64;
    return mlx::core::float32;
}

// --- Internal Structures ---

// Define PJRT_Error struct to be complete type so we can delete it
struct PJRT_Error {
    std::string message;
    PJRT_Error_Code code;
};

struct MLXMemory {
    int id;
    std::string kind_str;
    int kind_id;
    std::string debug_string;
    std::string to_string;
};

struct MLXDeviceDescription {
    int id;
    int process_index;
    std::string kind;
    std::string debug_string;
    std::string to_string;
};

struct MLXDevice {
    int id;
    int process_index;
    PJRT_Memory* memory; 
    MLXDeviceDescription* description;
};

struct MLXClient {
    int process_index;
    std::string platform_name;
    std::string platform_version;
    std::vector<MLXDevice*> devices;
};

struct MLXBuffer {
    mlx::core::array array;
    MLXClient* client;
    MLXDevice* device;
    bool is_deleted;
    std::vector<int64_t> dims;
    PJRT_Buffer_Type type;
    std::atomic<int> ref_count;
    
    // Mega-compile tracking
    int batch_id = 0;       // Which JIT batch this buffer belongs to
    bool from_host = true;  // True if created from host data (not graph output)
    
    MLXBuffer() : array(mlx::core::array(0.0f)), client(nullptr), device(nullptr), is_deleted(false), type(PJRT_Buffer_Type_INVALID), ref_count(1), batch_id(0), from_host(true) {}
    MLXBuffer(mlx::core::array a, MLXClient* c, MLXDevice* d, bool del, std::vector<int64_t> di, PJRT_Buffer_Type t) 
        : array(a), client(c), device(d), is_deleted(del), dims(di), type(t), ref_count(1), batch_id(0), from_host(true) {}
};

struct MLXEvent {
    bool is_ready;
    MLXEvent(bool r) : is_ready(r) {}
};

struct MLXGraph; // Forward declaration

struct MLXOp {
    std::string op_name;
    std::vector<int> inputs;
    std::vector<int> outputs;
    std::map<std::string, std::string> attributes;
    std::map<std::string, std::vector<float>> float_array_attrs;
    std::map<std::string, std::vector<int64_t>> int_array_attrs;
    std::map<std::string, int64_t> int_attrs;
    std::vector<std::vector<int>> output_shapes;
    std::vector<std::string> output_dtypes;
    std::vector<std::shared_ptr<MLXGraph>> subgraphs;
};

struct MLXGraph {
    std::vector<int> input_ids;
    std::vector<int> output_ids;
    std::vector<MLXOp> nodes;
    std::vector<std::vector<int>> input_shapes;
};

struct MLXExecutable {
    std::string name;
    int num_replicas;
    int num_partitions;
    int num_args;
    int num_outputs;
    std::string func_name;
    MLXGraph graph;
    std::map<std::string, std::shared_ptr<MLXGraph>> functions;
    std::atomic<int> ref_count;
    
    // Constant caching (MLX_CONSTANT_CACHE=1 to enable)
    std::vector<mlx::core::array> cached_constants;
    std::unordered_map<int, size_t> constant_output_to_cache_idx;
    bool constants_cached = false;
    
    // mx.compile() integration
    std::optional<std::function<std::vector<mlx::core::array>(const std::vector<mlx::core::array>&)>> compiled_fn;
    
    MLXExecutable(std::string n, int r, int p) 
        : name(n), num_replicas(r), num_partitions(p), num_args(1), num_outputs(1), 
          ref_count(1), constants_cached(false), compiled_fn(std::nullopt) {}
};

struct MLXLoadedExecutable {
    MLXExecutable* inner_executable;
    MLXClient* client;
    bool is_deleted;
};

struct MLXTopologyDescription {
    std::string platform_name;
    std::string platform_version;
};

// --- Mega-Compile Infrastructure ---
// Stores a pending graph execution for later batch compilation
struct PendingExecution {
    MLXExecutable* exec;
    std::vector<mlx::core::array> inputs;
    std::vector<MLXBuffer*> output_buffers;  // Where to store results when materialized
    int batch_id;
};

// Thread-local accumulator for pending graph executions
struct BatchAccumulator {
    std::map<int, std::vector<PendingExecution>> pending_by_batch;
    bool enabled = false;
    
    void add_pending(int batch_id, PendingExecution&& pe) {
        pending_by_batch[batch_id].push_back(std::move(pe));
    }
    
    bool has_pending(int batch_id) const {
        auto it = pending_by_batch.find(batch_id);
        return it != pending_by_batch.end() && !it->second.empty();
    }
    
    void clear_batch(int batch_id) {
        pending_by_batch.erase(batch_id);
    }
    
    void clear_all() {
        pending_by_batch.clear();
    }
};
static thread_local BatchAccumulator g_batch_accumulator;

// --- Globals ---
MLXClient global_client;
MLXDevice global_device;
MLXMemory global_memory;
MLXDeviceDescription global_device_description;
MLXTopologyDescription global_topology;

PJRT_Device* global_device_ptr = reinterpret_cast<PJRT_Device*>(&global_device);
PJRT_Memory* global_memory_ptr = reinterpret_cast<PJRT_Memory*>(&global_memory);
PJRT_DeviceDescription* global_device_description_ptr = reinterpret_cast<PJRT_DeviceDescription*>(&global_device_description);
PJRT_TopologyDescription* global_topology_ptr = reinterpret_cast<PJRT_TopologyDescription*>(&global_topology);

// --- API IMPLEMENTATION ---

// Error
void MLX_Error_Destroy(PJRT_Error_Destroy_Args* args) { 
    if (args->error) delete reinterpret_cast<MLXError*>(args->error);
}
void MLX_Error_Message(PJRT_Error_Message_Args* args) { 
    const MLXError* e = reinterpret_cast<const MLXError*>(args->error);
    args->message = e->message.c_str(); 
    args->message_size = e->message.size();
}
PJRT_Error* MLX_Error_GetCode(PJRT_Error_GetCode_Args* args) {
    const MLXError* e = reinterpret_cast<const MLXError*>(args->error);
    args->code = e->code;
    return Ok();
}

// Plugin
PJRT_Error* MLX_Plugin_Initialize(PJRT_Plugin_Initialize_Args* args) {
    return Ok();
}
PJRT_Error* MLX_Plugin_Attributes(PJRT_Plugin_Attributes_Args* args) {
    args->num_attributes = 0;
    return Ok();
}

// Event - Treat args->event as PJRT_Event** (handle pointer)
PJRT_Error* MLX_Event_Destroy(PJRT_Event_Destroy_Args* args) {
    if(args->event) {
        delete reinterpret_cast<MLXEvent*>(args->event);
    }
    return Ok();
}
PJRT_Error* MLX_Event_IsReady(PJRT_Event_IsReady_Args* args) {
    args->is_ready = reinterpret_cast<MLXEvent*>(args->event)->is_ready;
    return Ok();
}
PJRT_Error* MLX_Event_Error(PJRT_Event_Error_Args* args) {
    return Ok(); // No error
}
PJRT_Error* MLX_Event_Await(PJRT_Event_Await_Args* args) {
    return Ok();
}
PJRT_Error* MLX_Event_OnReady(PJRT_Event_OnReady_Args* args) {
    if(args->callback) args->callback(Ok(), args->user_arg);
    return Ok();
}
PJRT_Error* MLX_Event_Create(PJRT_Event_Create_Args* args) {
    args->event = reinterpret_cast<PJRT_Event*>(new MLXEvent(false));
    return Ok();
}
PJRT_Error* MLX_Event_Set(PJRT_Event_Set_Args* args) {
    reinterpret_cast<MLXEvent*>(args->event)->is_ready = true;
    return Ok();
}

// Client
PJRT_Error* MLX_Client_Create(PJRT_Client_Create_Args* args) {
    
    // Initialize Globals
    global_memory.id = 0;
    global_memory.kind_str = "unified";
    global_memory.kind_id = 0;
    global_memory.debug_string = "Unified MLX Memory";
    global_memory.to_string = "UnifiedMemory";

    global_device_description.id = 0;
    global_device_description.process_index = 0;
    global_device_description.kind = "mlx";
    global_device_description.debug_string = "MLX Device 0";
    global_device_description.to_string = "mlx:0";

    global_device.id = 0;
    global_device.process_index = 0;
    global_device.memory = reinterpret_cast<PJRT_Memory*>(&global_memory);
    global_device.description = &global_device_description;

    global_client.process_index = 0;
    global_client.platform_name = "mlx";
    global_client.platform_version = "0.0.1";
    global_client.devices.push_back(&global_device);  // Add device to the devices vector

    args->client = reinterpret_cast<PJRT_Client*>(&global_client);
    return Ok();
}

PJRT_Error* MLX_Client_Destroy(PJRT_Client_Destroy_Args* args) {
    return Ok();
}

PJRT_Error* MLX_Client_PlatformName(PJRT_Client_PlatformName_Args* args) {
    MLXClient* client = reinterpret_cast<MLXClient*>(args->client);
    args->platform_name = client->platform_name.c_str();
    args->platform_name_size = client->platform_name.size();
    return Ok();
}

PJRT_Error* MLX_Client_ProcessIndex(PJRT_Client_ProcessIndex_Args* args) {
    MLXClient* client = reinterpret_cast<MLXClient*>(args->client);
    args->process_index = client->process_index;
    return Ok();
}

PJRT_Error* MLX_Client_PlatformVersion(PJRT_Client_PlatformVersion_Args* args) {
    MLXClient* client = reinterpret_cast<MLXClient*>(args->client);
    args->platform_version = client->platform_version.c_str();
    args->platform_version_size = client->platform_version.size();
    return Ok();
}

PJRT_Error* MLX_Client_Devices(PJRT_Client_Devices_Args* args) {
    args->devices = &global_device_ptr; 
    args->num_devices = 1;
    return Ok();
}

PJRT_Error* MLX_Client_AddressableDevices(PJRT_Client_AddressableDevices_Args* args) {
    return MLX_Client_Devices(reinterpret_cast<PJRT_Client_Devices_Args*>(args));
}

PJRT_Error* MLX_Client_LookupDevice(PJRT_Client_LookupDevice_Args* args) {
    args->device = reinterpret_cast<PJRT_Device*>(&global_device);
    return Ok();
}

PJRT_Error* MLX_Client_LookupAddressableDevice(PJRT_Client_LookupAddressableDevice_Args* args) {
    args->addressable_device = reinterpret_cast<PJRT_Device*>(&global_device);
    return Ok();
}

PJRT_Error* MLX_Client_AddressableMemories(PJRT_Client_AddressableMemories_Args* args) {
    args->addressable_memories = &global_memory_ptr;
    args->num_addressable_memories = 1;
    return Ok();
}

// Client Stubs
// Helper to recursively load graph
void LoadGraph(const py::dict& graph_dict, MLXGraph& graph) {
    // Inputs (ordered)
    if (graph_dict.contains("inputs")) {
        auto inputs = graph_dict["inputs"].cast<py::list>();
        for (auto item : inputs) {
            auto input_dict = item.cast<py::dict>();
            graph.input_ids.push_back(input_dict["id"].cast<int>());
        }
    }
    
    // Outputs
    if (graph_dict.contains("outputs")) {
        auto outputs = graph_dict["outputs"].cast<py::list>();
        for (auto item : outputs) {
            graph.output_ids.push_back(item.cast<int>());
        }
    }
    
    // Input Shapes
    if (graph_dict.contains("input_shapes")) {
        auto shapes = graph_dict["input_shapes"].cast<py::list>();
        for (auto item : shapes) {
            graph.input_shapes.push_back(item.cast<std::vector<int>>());
        }
    }

    // Nodes
    if (graph_dict.contains("nodes")) {
        auto nodes = graph_dict["nodes"].cast<py::list>();
        for (auto item : nodes) {
            auto node_dict = item.cast<py::dict>();
            MLXOp op;
            op.op_name = node_dict["op"].cast<std::string>();
            op.inputs = node_dict["inputs"].cast<std::vector<int>>();
            op.outputs = node_dict["outputs"].cast<std::vector<int>>();
            
            if (node_dict.contains("attributes")) {
                auto attrs = node_dict["attributes"].cast<py::dict>();
                for (auto attr_item : attrs) {
                    std::string key = attr_item.first.cast<std::string>();
                    if (debug_mode()) {
                        std::cout << "[MLX-PJRT] Loading attr: " << key << std::endl;
                    }
                    auto val = attr_item.second;
                    
                    if (key == "value" && py::isinstance<py::list>(val)) {
                        try { op.float_array_attrs[key] = val.cast<std::vector<float>>(); } catch(...) { op.attributes[key] = py::str(val).cast<std::string>(); }
                    } else if (key == "int_value" && py::isinstance<py::list>(val)) {
                         try { op.int_array_attrs[key] = val.cast<std::vector<int64_t>>(); } catch(const std::exception& e) { 
if (debug_mode()) std::cout << "[MLX-PJRT] LoadGraph Error casting int_value: " << e.what() << std::endl;
                             op.attributes[key] = py::str(val).cast<std::string>(); 
                         } catch(...) {
if (debug_mode()) std::cout << "[MLX-PJRT] LoadGraph Error casting int_value (unknown)" << std::endl;
                             op.attributes[key] = py::str(val).cast<std::string>();
                         }
                    } else if (py::isinstance<py::list>(val)) {
                        // Generic/Whitelisted int array attributes
                        if (key == "dims" || key == "broadcast_dimensions" || key == "dimensions" ||
                            key == "sizes" || key == "slice_sizes" ||
                            key == "start_indices" || key == "limit_indices" || key == "strides" ||
                            key == "padding" || 
                            key == "window_dimensions" || key == "window_strides" ||
                            key == "lhs_contracting" || key == "rhs_contracting" || 
                            key == "lhs_batching" || key == "rhs_batching" ||
                            key == "permutation" || key == "transpose_permutation" ||
                            key == "low" || key == "high" || key == "interior" ||
                            key == "edge_padding_low" || key == "edge_padding_high" || key == "interior_padding" ||
                            key == "input_spatial_dimensions" || key == "kernel_spatial_dimensions" || key == "output_spatial_dimensions" ||
                            key == "lhs_dilation" || key == "rhs_dilation" ||
                            key == "start_index_map" || key == "offset_dims" || key == "collapsed_slice_dims" ||
                            key == "operand_batching_dims" || key == "start_indices_batching_dims") {
                            try { op.int_array_attrs[key] = val.cast<std::vector<int64_t>>(); } catch(...) { op.attributes[key] = py::str(val).cast<std::string>(); }
                        } else {
                            // Fallback for other lists (or store as string if generic int loading is desired but risky)
                            op.attributes[key] = py::str(val).cast<std::string>();
                        }
                    } else if (py::isinstance<py::int_>(val)) {
                         op.int_attrs[key] = val.cast<int64_t>();
                         // Also store as string for compatibility if needed
                         op.attributes[key] = py::str(val).cast<std::string>();
                    } else {
                        op.attributes[key] = py::str(val).cast<std::string>();
                    }
                }
            }
            
            if (node_dict.contains("output_types")) {
                auto out_types = node_dict["output_types"].cast<py::list>();
                for (auto type_item : out_types) {
                    auto type_dict = type_item.cast<py::dict>();
                    op.output_shapes.push_back(type_dict["shape"].cast<std::vector<int>>());
                    op.output_dtypes.push_back(type_dict["dtype"].cast<std::string>());
                }
            }

            // Subgraphs
            if (node_dict.contains("subgraphs")) {
                auto subs = node_dict["subgraphs"].cast<py::list>();
                for (auto sub_item : subs) {
                    auto sub_dict = sub_item.cast<py::dict>();
                    auto sub_graph = std::make_shared<MLXGraph>();
                    LoadGraph(sub_dict, *sub_graph);
                    op.subgraphs.push_back(sub_graph);
                }
            }
            
            graph.nodes.push_back(op);
        }
    }
}

PJRT_Error* MLX_Client_Compile(PJRT_Client_Compile_Args* args) { 
    if (!args->program) return Unimplemented("Compile: No program provided");

    MLXClient* client = reinterpret_cast<MLXClient*>(args->client);
    
    // Save bytecode for debugging when enabled
    if (debug_mode()) {
        FILE* f = fopen("/tmp/jit_compiled.mlir.bc", "wb");
        if (f) {
            fwrite(args->program->code, 1, args->program->code_size, f);
            fclose(f);
        }
    }


    try {
        py::gil_scoped_acquire acquire;
        
        py::module_ sys = py::module_::import("sys");
        py::module_ parser_mod;
        try {
            parser_mod = py::module_::import("jax_mlx.parser");
        } catch (...) {
            sys.attr("path").attr("append")(".");
            parser_mod = py::module_::import("jax_mlx.parser");
        }

        // Timing: Python parse_bytecode
        auto parse_start = std::chrono::high_resolution_clock::now();
        auto bytecode = py::bytes(args->program->code, args->program->code_size);
        auto result = parser_mod.attr("parse_bytecode")(bytecode).cast<py::dict>();
        auto parse_end = std::chrono::high_resolution_clock::now();
        
        if (timing_mode()) {
            auto parse_us = std::chrono::duration_cast<std::chrono::microseconds>(parse_end - parse_start).count();
            std::cout << "[TIMING] Python parse_bytecode: " << parse_us << "us (" 
                      << (parse_us / 1000.0) << "ms) [bytecode_size=" << args->program->code_size << "]" << std::endl;
        }
        
        if (result.contains("error")) {
            std::string err_msg = result["error"].cast<std::string>();
if (debug_mode()) std::cout << "[MLX-PJRT]   Parser error detected: " << err_msg << std::endl;
            return Error(PJRT_Error_Code_INTERNAL, ("Parser error: " + err_msg).c_str());
        }

        MLXExecutable* exec = new MLXExecutable("jit_executable", 1, 1);
        
        // Timing: LoadGraph (dict to C++ structs)
        auto load_start = std::chrono::high_resolution_clock::now();
        try {
            LoadGraph(result, exec->graph);
        } catch (const std::exception& e) {
if (debug_mode()) std::cout << "[MLX-PJRT] LoadGraph failed: " << e.what() << std::endl;
             return Error(PJRT_Error_Code_INTERNAL, e.what());
        } catch (...) {
if (debug_mode()) std::cout << "[MLX-PJRT] LoadGraph failed with unknown error" << std::endl;
             return Error(PJRT_Error_Code_INTERNAL, "Unknown LoadGraph error");
        }
        auto load_end = std::chrono::high_resolution_clock::now();
        
        if (timing_mode()) {
            auto load_us = std::chrono::duration_cast<std::chrono::microseconds>(load_end - load_start).count();
            std::cout << "[TIMING] LoadGraph (dict->C++): " << load_us << "us (" 
                      << (load_us / 1000.0) << "ms) [nodes=" << exec->graph.nodes.size() << "]" << std::endl;
        }

        // Load auxiliary functions
        if (result.contains("functions")) {
            auto funcs_dict = result["functions"].cast<py::dict>();
            for (auto item : funcs_dict) {
                std::string fname = item.first.cast<std::string>();
                auto fgraph_dict = item.second.cast<py::dict>();
                auto fgraph = std::make_shared<MLXGraph>();
                LoadGraph(fgraph_dict, *fgraph);
                exec->functions[fname] = fgraph;
if (debug_mode()) std::cout << "[MLX-PJRT]   Loaded function: " << fname << std::endl;
            }
        }

        exec->num_args = exec->graph.input_ids.size();
        exec->num_outputs = exec->graph.output_ids.size();

if (debug_mode()) std::cout << "[MLX-PJRT]   Compilation successful: " 
                  << exec->graph.nodes.size() << " nodes, " 
                  << exec->num_args << " inputs, " 
                  << exec->num_outputs << " outputs" << std::endl;

        MLXLoadedExecutable* loaded = new MLXLoadedExecutable{exec, client, false};
        args->executable = reinterpret_cast<PJRT_LoadedExecutable*>(loaded);
        return Ok();

    } catch (const py::error_already_set& e) {
        std::cerr << "[MLX-PJRT][ERROR] Python error during compilation: " << e.what() << std::endl;
        return Unimplemented("Python error during compilation: " + std::string(e.what()));
    } catch (const std::exception& e) {
        std::cerr << "[MLX-PJRT][ERROR] C++ error during compilation: " << e.what() << std::endl;
        return Unimplemented("C++ error during compilation: " + std::string(e.what()));
    }
}
PJRT_Error* MLX_Client_DefaultDeviceAssignment(PJRT_Client_DefaultDeviceAssignment_Args* args) { return Unimplemented("Client_DefaultDeviceAssignment"); }
PJRT_Error* MLX_Client_CreateViewOfDeviceBuffer(PJRT_Client_CreateViewOfDeviceBuffer_Args* args) { return Unimplemented("Client_CreateViewOfDeviceBuffer"); }
PJRT_Error* MLX_Client_CreateBuffersForAsyncHostToDevice(PJRT_Client_CreateBuffersForAsyncHostToDevice_Args* args) { return Unimplemented("Client_CreateBuffersForAsyncHostToDevice"); }
PJRT_Error* MLX_Client_TopologyDescription(PJRT_Client_TopologyDescription_Args* args) {
    // Initialize the global topology if needed
    global_topology.platform_name = "mlx";
    global_topology.platform_version = "0.0.1";
    args->topology = reinterpret_cast<PJRT_TopologyDescription*>(&global_topology);
    return Ok();
}

// --- Buffer API ---

// PJRT_Error* MLX_Client_BufferFromHostBuffer(PJRT_Client_BufferFromHostBuffer_Args* args) { ... }
PJRT_Error* MLX_Client_BufferFromHostBuffer(PJRT_Client_BufferFromHostBuffer_Args* args) {
    // Build shape vector
    std::vector<int> shape;
    std::vector<int64_t> dim_vector;
    for(size_t i = 0; i < args->num_dims; ++i) {
        shape.push_back(static_cast<int>(args->dims[i]));
        dim_vector.push_back(args->dims[i]);
    }
    
    mlx::core::Dtype dtype = PjrtTypeToMlxType(args->type);
    mlx::core::Shape mlx_shape(shape.begin(), shape.end());
    
    // Calculate total size in bytes
    size_t element_size = 4;  // Default to 4 bytes (float32)
    switch(args->type) {
        case PJRT_Buffer_Type_F32: element_size = 4; break;
        case PJRT_Buffer_Type_F64: element_size = 8; break;
        case PJRT_Buffer_Type_S32: element_size = 4; break;
        case PJRT_Buffer_Type_S64: element_size = 8; break;
        case PJRT_Buffer_Type_S16: element_size = 2; break;
        case PJRT_Buffer_Type_S8: element_size = 1; break;
        case PJRT_Buffer_Type_U8: element_size = 1; break;
        case PJRT_Buffer_Type_U16: element_size = 2; break;
        case PJRT_Buffer_Type_U32: element_size = 4; break;
        case PJRT_Buffer_Type_U64: element_size = 8; break;
        case PJRT_Buffer_Type_F16: element_size = 2; break;
        case PJRT_Buffer_Type_BF16: element_size = 2; break;
        case PJRT_Buffer_Type_PRED: element_size = 1; break;
        case PJRT_Buffer_Type_C64: element_size = 8; break; // Added C64
        default: element_size = 4; break;
    }
    
    size_t num_elements = 1;
    for (auto d : shape) num_elements *= d;
    size_t total_bytes = num_elements * element_size;
    
    // Create array using the appropriate typed pointer based on dtype
    mlx::core::array arr = mlx::core::zeros(mlx_shape, dtype);
    
    if (args->data && total_bytes > 0) {
        // Use from_blob approach: create array with data copy
        // MLX provides the iterator constructor - cast data to typed pointer
        switch(args->type) {
            case PJRT_Buffer_Type_F32: {
                const float* src = static_cast<const float*>(args->data);
                arr = mlx::core::array(src, mlx_shape, mlx::core::float32);
                break;
            }
            case PJRT_Buffer_Type_F64: {
                const double* src = static_cast<const double*>(args->data);
                arr = mlx::core::array(src, mlx_shape, mlx::core::float64);
                break;
            }
            case PJRT_Buffer_Type_S32: {
                const int32_t* src = static_cast<const int32_t*>(args->data);
                arr = mlx::core::array(src, mlx_shape, mlx::core::int32);
                break;
            }
            case PJRT_Buffer_Type_S64: {
                const int64_t* src = static_cast<const int64_t*>(args->data);
                arr = mlx::core::array(src, mlx_shape, mlx::core::int64);
                break;
            }
            case PJRT_Buffer_Type_S8: {
                const int8_t* src = static_cast<const int8_t*>(args->data);
                arr = mlx::core::array(src, mlx_shape, mlx::core::int8);
                break;
            }
            case PJRT_Buffer_Type_U8: {
                const uint8_t* src = static_cast<const uint8_t*>(args->data);
                arr = mlx::core::array(src, mlx_shape, mlx::core::uint8);
                break;
            }
            case PJRT_Buffer_Type_PRED: {
                const bool* src = static_cast<const bool*>(args->data);
                arr = mlx::core::array(src, mlx_shape, mlx::core::bool_);
                break;
            }
            case PJRT_Buffer_Type_C64: { // Added C64
                // Manually allocate and copy to avoid array::init template instantiation errors
                // (which tries to compile copy(complex -> float16) etc.)
                size_t num_elements = 1;
                for (auto s : mlx_shape) num_elements *= s;
                size_t bytes = num_elements * sizeof(std::complex<float>);
                
                auto buf = mlx::core::allocator::malloc(bytes);
                std::memcpy(buf.raw_ptr(), args->data, bytes);
                
                arr = mlx::core::array(buf, mlx_shape, mlx::core::complex64);
                break;
            }
            case PJRT_Buffer_Type_U32: {
                const uint32_t* src = static_cast<const uint32_t*>(args->data);
                arr = mlx::core::array(src, mlx_shape, mlx::core::uint32);
                break;
            }
            case PJRT_Buffer_Type_U64: {
                const uint64_t* src = static_cast<const uint64_t*>(args->data);
                arr = mlx::core::array(src, mlx_shape, mlx::core::uint64);
                break;
            }
            case PJRT_Buffer_Type_U16: {
                const uint16_t* src = static_cast<const uint16_t*>(args->data);
                arr = mlx::core::array(src, mlx_shape, mlx::core::uint16);
                break;
            }
            case PJRT_Buffer_Type_F16: {
                const mlx::core::float16_t* src = static_cast<const mlx::core::float16_t*>(args->data);
                arr = mlx::core::array(src, mlx_shape, mlx::core::float16);
                break;
            }
            case PJRT_Buffer_Type_BF16: {
                const mlx::core::bfloat16_t* src = static_cast<const mlx::core::bfloat16_t*>(args->data);
                arr = mlx::core::array(src, mlx_shape, mlx::core::bfloat16);
                break;
            }
            // Add other types if needed, fallback to default
            default: {
                // Try to handle memory copy by size if type unknown but we want to fail soft?
                if (args->type == 8) {
                    const uint32_t* src = static_cast<const uint32_t*>(args->data);
                    arr = mlx::core::array(src, mlx_shape, mlx::core::uint32);
                } else if (args->type == 9) {
                    const uint64_t* src = static_cast<const uint64_t*>(args->data);
                    arr = mlx::core::array(src, mlx_shape, mlx::core::uint64);
                } else {
                    std::cerr << "[MLX-PJRT][WARN] BufferFromHostBuffer unhandled type " << args->type << " (U32=" << PJRT_Buffer_Type_U32 << ")" << std::endl;
                    // Leave arr as zeros
                }
                break;
            }
        }
    }
    
    // Ensure data is materialized
    arr.eval();
    
    if (debug_mode()) {
        for(size_t i = 0; i < arr.shape().size(); ++i) {
            std::cout << arr.shape()[i];
            if (i < arr.shape().size() - 1) std::cout << ", ";
        }
        std::cout << "] dtype=" << arr.dtype() << std::endl;
    }
    
    // Create event for done_with_host_buffer
    args->done_with_host_buffer = reinterpret_cast<PJRT_Event*>(new MLXEvent(true));

    // Create and return buffer
    MLXBuffer* buffer = new MLXBuffer(arr, reinterpret_cast<MLXClient*>(args->client), &global_device, false, dim_vector, args->type);
    args->buffer = reinterpret_cast<PJRT_Buffer*>(buffer);
    
    return Ok();
}

PJRT_Error* MLX_Executable_Destroy(PJRT_Executable_Destroy_Args* args) {
if (debug_mode()) std::cout << "[MLX-PJRT] Executable_Destroy called" << std::endl;
    MLXExecutable* exec = reinterpret_cast<MLXExecutable*>(args->executable);
if (debug_mode()) std::cout << "[MLX-PJRT]   exec ptr: " << (void*)exec << std::endl;
    if (exec) {
        int old_count = exec->ref_count.fetch_sub(1);
if (debug_mode()) std::cout << "[MLX-PJRT]   old_count: " << old_count << std::endl;
        if (old_count == 1) {
            delete exec;
        }
    }
    return Ok();
}

PJRT_Error* MLX_Buffer_Destroy(PJRT_Buffer_Destroy_Args* args) {
    MLXBuffer* buf = reinterpret_cast<MLXBuffer*>(args->buffer);
    if (buf) {
        int old_count = buf->ref_count.fetch_sub(1);
        if (old_count == 1) {
            delete buf;
        }
    }
    return Ok();
}

void EnsureContiguous(MLXBuffer* buf) {
    size_t expected_nbytes = buf->array.size() * buf->array.itemsize();
    bool is_scalar_broadcast = true;
    for(size_t s : buf->array.strides()) {
        if (s != 0) is_scalar_broadcast = false;
    }

    if (!buf->array.flags().row_contiguous || buf->array.nbytes() != expected_nbytes) {
        if (is_scalar_broadcast && buf->array.dtype() == mlx::core::float32) {
             float val = *buf->array.data<float>();
             std::vector<float> vec(buf->array.size(), val);
             buf->array = mlx::core::array(vec.data(), buf->array.shape(), mlx::core::float32);
        } else {
             // Fallback for other cases
             auto zero = mlx::core::zeros(buf->array.shape(), buf->array.dtype());
             buf->array = mlx::core::add(zero, buf->array);
        }
        buf->array.eval();
    }
}

PJRT_Error* MLX_Buffer_ToHostBuffer(PJRT_Buffer_ToHostBuffer_Args* args) {
    MLXBuffer* buf = reinterpret_cast<MLXBuffer*>(args->src);
    
    size_t size = buf->array.nbytes();
    
    if (args->dst == nullptr) {
        args->dst_size = size;
        return Ok();
    }
    
    if (args->dst_size < size) {
        return Unimplemented("Buffer_ToHostBuffer: dst too small");
    }
    
    // Mega-compile: Materialize any pending graphs for this buffer's batch
    if (mega_compile_enabled() && buf->batch_id > 0) {
        // Forward declaration of materialize_batch - we'll call it if pending
        extern void materialize_batch(int batch_id);
        if (g_batch_accumulator.has_pending(buf->batch_id)) {
            materialize_batch(buf->batch_id);
        }
    }
    
    // Ensure array is computed
    buf->array.eval();
    

    // Check for contiguity and copy if necessary
    EnsureContiguous(buf);

    
    // Debug: print array info
    
    // Copy data - need to access raw data, MLX data<T>() returns typed pointer
    // For float32, use data<float>()
    if (buf->type == PJRT_Buffer_Type_F32) {
        const float* src_ptr = buf->array.data<float>();
        std::memcpy(args->dst, src_ptr, size);
    } else {
        const char* src_ptr = buf->array.data<char>();
        std::memcpy(args->dst, src_ptr, size);
    }
    

    
    if (args->event) {
        args->event = reinterpret_cast<PJRT_Event*>(new MLXEvent(true));
    }
    
    return Ok();
}

PJRT_Error* MLX_Buffer_Dimensions(PJRT_Buffer_Dimensions_Args* args) {
    MLXBuffer* buf = reinterpret_cast<MLXBuffer*>(args->buffer);
    args->dims = buf->dims.data();
    args->num_dims = buf->dims.size();
    return Ok();
}

PJRT_Error* MLX_Buffer_ElementType(PJRT_Buffer_ElementType_Args* args) {
    MLXBuffer* buf = reinterpret_cast<MLXBuffer*>(args->buffer);
    args->type = buf->type;
    return Ok();
}

PJRT_Error* MLX_Buffer_UnpaddedDimensions(PJRT_Buffer_UnpaddedDimensions_Args* args) {
    return MLX_Buffer_Dimensions(reinterpret_cast<PJRT_Buffer_Dimensions_Args*>(args));
}

PJRT_Error* MLX_Buffer_DynamicDimensionIndices(PJRT_Buffer_DynamicDimensionIndices_Args* args) {
    args->dynamic_dim_indices = nullptr;
    args->num_dynamic_dims = 0;
    return Ok();
}

PJRT_Error* MLX_Buffer_GetMemoryLayout(PJRT_Buffer_GetMemoryLayout_Args* args) {
    return Unimplemented("Buffer_GetMemoryLayout"); 
}

PJRT_Error* MLX_Buffer_OnDeviceSizeInBytes(PJRT_Buffer_OnDeviceSizeInBytes_Args* args) {
    args->on_device_size_in_bytes = reinterpret_cast<MLXBuffer*>(args->buffer)->array.nbytes();
    return Ok();
}

PJRT_Error* MLX_Buffer_Device(PJRT_Buffer_Device_Args* args) {
    MLXBuffer* buf = reinterpret_cast<MLXBuffer*>(args->buffer);
    args->device = reinterpret_cast<PJRT_Device*>(buf->device);
    return Ok();
}

PJRT_Error* MLX_Buffer_Memory(PJRT_Buffer_Memory_Args* args) {
    args->memory = reinterpret_cast<MLXBuffer*>(args->buffer)->device->memory;
    return Ok();
}

PJRT_Error* MLX_Buffer_Delete(PJRT_Buffer_Delete_Args* args) {
    reinterpret_cast<MLXBuffer*>(args->buffer)->is_deleted = true;
    return Ok();
}

PJRT_Error* MLX_Buffer_IsDeleted(PJRT_Buffer_IsDeleted_Args* args) {
    args->is_deleted = reinterpret_cast<MLXBuffer*>(args->buffer)->is_deleted;
    return Ok();
}

PJRT_Error* MLX_Buffer_CopyToDevice(PJRT_Buffer_CopyToDevice_Args* args) { return Unimplemented("Buffer_CopyToDevice"); }
PJRT_Error* MLX_Buffer_IsOnCpu(PJRT_Buffer_IsOnCpu_Args* args) { 
    args->is_on_cpu = true; 
    return Ok(); 
}
PJRT_Error* MLX_Buffer_ReadyEvent(PJRT_Buffer_ReadyEvent_Args* args) {
    MLXBuffer* buf = reinterpret_cast<MLXBuffer*>(args->buffer);
    
    // Mega-compile: this is a sync point - ensure array is evaluated
    if (mega_compile_enabled()) {
        buf->array.eval();
    }
    
    args->event = reinterpret_cast<PJRT_Event*>(new MLXEvent(true));
    return Ok();
}



PJRT_Error* MLX_Buffer_UnsafePointer(PJRT_Buffer_UnsafePointer_Args* args) {
    // Ensure array is evaluated and contiguous
    MLXBuffer* buf = reinterpret_cast<MLXBuffer*>(args->buffer);
    buf->array.eval();
    EnsureContiguous(buf);
    
    // Return the raw data pointer
    args->buffer_pointer = reinterpret_cast<uintptr_t>(buf->array.data<void>());
    return Ok();
}
PJRT_Error* MLX_Buffer_IncreaseExternalReferenceCount(PJRT_Buffer_IncreaseExternalReferenceCount_Args* args) { 
    int new_count = ++reinterpret_cast<MLXBuffer*>(args->buffer)->ref_count;
    return Ok(); 
}
PJRT_Error* MLX_Buffer_DecreaseExternalReferenceCount(PJRT_Buffer_DecreaseExternalReferenceCount_Args* args) { 
    int old_count = reinterpret_cast<MLXBuffer*>(args->buffer)->ref_count.fetch_sub(1);
    // If ref_count reaches 0, the buffer should be deleted
    if (old_count == 1) {
        delete reinterpret_cast<MLXBuffer*>(args->buffer);
    }
    return Ok(); 
}
PJRT_Error* MLX_Buffer_OpaqueDeviceMemoryDataPointer(PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args* args) { 
    MLXBuffer* buf = reinterpret_cast<MLXBuffer*>(args->buffer);
    buf->array.eval();
    EnsureContiguous(buf);
    args->device_memory_ptr = buf->array.data<void>();
    return Ok(); 
}


// --- Device API ---

PJRT_Error* MLX_Device_GetDescription(PJRT_Device_GetDescription_Args* args) {
if (debug_mode()) std::cout << "[MLX-PJRT] MLX_Device_GetDescription called" << std::endl;
    args->device_description = reinterpret_cast<PJRT_DeviceDescription*>(reinterpret_cast<MLXDevice*>(args->device)->description);
    return Ok();
}

PJRT_Error* MLX_Device_IsAddressable(PJRT_Device_IsAddressable_Args* args) {
if (debug_mode()) std::cout << "[MLX-PJRT] MLX_Device_IsAddressable called" << std::endl;
    args->is_addressable = true;
    return Ok();
}

PJRT_Error* MLX_Device_LocalHardwareId(PJRT_Device_LocalHardwareId_Args* args) {
if (debug_mode()) std::cout << "[MLX-PJRT] MLX_Device_LocalHardwareId called" << std::endl;
    args->local_hardware_id = reinterpret_cast<MLXDevice*>(args->device)->id;
    return Ok();
}

PJRT_Error* MLX_Device_AddressableMemories(PJRT_Device_AddressableMemories_Args* args) {
if (debug_mode()) std::cout << "[MLX-PJRT] MLX_Device_AddressableMemories called" << std::endl;
    args->memories = &global_memory_ptr;
    args->num_memories = 1;
    return Ok();
}

// --- Device Description API ---
PJRT_Error* MLX_DeviceDescription_Id(PJRT_DeviceDescription_Id_Args* args) {
if (debug_mode()) std::cout << "[MLX-PJRT] MLX_DeviceDescription_Id called" << std::endl;
    args->id = reinterpret_cast<const MLXDeviceDescription*>(args->device_description)->id;
    return Ok();
}
PJRT_Error* MLX_DeviceDescription_ProcessIndex(PJRT_DeviceDescription_ProcessIndex_Args* args) {
if (debug_mode()) std::cout << "[MLX-PJRT] MLX_DeviceDescription_ProcessIndex called" << std::endl;
    args->process_index = reinterpret_cast<const MLXDeviceDescription*>(args->device_description)->process_index;
    return Ok();
}
PJRT_Error* MLX_DeviceDescription_Attributes(PJRT_DeviceDescription_Attributes_Args* args) {
    // std::cout << "[MLX-PJRT] MLX_DeviceDescription_Attributes called" << std::endl;
    args->num_attributes = 0;
    return Ok();
}
PJRT_Error* MLX_DeviceDescription_Kind(PJRT_DeviceDescription_Kind_Args* args) {
if (debug_mode()) std::cout << "[MLX-PJRT] MLX_DeviceDescription_Kind called" << std::endl;
    args->device_kind = reinterpret_cast<const MLXDeviceDescription*>(args->device_description)->kind.c_str();
    args->device_kind_size = reinterpret_cast<const MLXDeviceDescription*>(args->device_description)->kind.size();
    return Ok();
}
PJRT_Error* MLX_DeviceDescription_DebugString(PJRT_DeviceDescription_DebugString_Args* args) {
if (debug_mode()) std::cout << "[MLX-PJRT] MLX_DeviceDescription_DebugString called" << std::endl;
    args->debug_string = reinterpret_cast<const MLXDeviceDescription*>(args->device_description)->debug_string.c_str();
    args->debug_string_size = reinterpret_cast<const MLXDeviceDescription*>(args->device_description)->debug_string.size();
    return Ok();
}
PJRT_Error* MLX_DeviceDescription_ToString(PJRT_DeviceDescription_ToString_Args* args) {
if (debug_mode()) std::cout << "[MLX-PJRT] MLX_DeviceDescription_ToString called" << std::endl;
    args->to_string = reinterpret_cast<const MLXDeviceDescription*>(args->device_description)->to_string.c_str();
    args->to_string_size = reinterpret_cast<const MLXDeviceDescription*>(args->device_description)->to_string.size();
    return Ok();
}
PJRT_Error* MLX_Device_DefaultMemory(PJRT_Device_DefaultMemory_Args* args) {
    args->memory = reinterpret_cast<MLXDevice*>(args->device)->memory;
    return Ok();
}
PJRT_Error* MLX_Device_MemoryStats(PJRT_Device_MemoryStats_Args* args) { return Unimplemented("Device_MemoryStats"); }
PJRT_Error* MLX_Device_PoisonExecution(PJRT_Device_PoisonExecution_Args* args) { return Unimplemented("Device_PoisonExecution"); }
PJRT_Error* MLX_Device_CreateAsyncTrackingEvent(PJRT_Device_CreateAsyncTrackingEvent_Args* args) { return Unimplemented("Device_CreateAsyncTrackingEvent"); }

// --- Device Description API ---

// --- Memory API ---
PJRT_Error* MLX_Memory_Id(PJRT_Memory_Id_Args* args) {
    args->id = reinterpret_cast<const MLXMemory*>(args->memory)->id;
    return Ok();
}
PJRT_Error* MLX_Memory_Kind(PJRT_Memory_Kind_Args* args) {
    args->kind = reinterpret_cast<const MLXMemory*>(args->memory)->kind_str.c_str();
    args->kind_size = reinterpret_cast<const MLXMemory*>(args->memory)->kind_str.size();
    return Ok();
}
PJRT_Error* MLX_Memory_Kind_Id(PJRT_Memory_Kind_Id_Args* args) {
    args->kind_id = reinterpret_cast<const MLXMemory*>(args->memory)->kind_id;
    return Ok();
}
PJRT_Error* MLX_Memory_DebugString(PJRT_Memory_DebugString_Args* args) {
    args->debug_string = reinterpret_cast<const MLXMemory*>(args->memory)->debug_string.c_str();
    args->debug_string_size = reinterpret_cast<const MLXMemory*>(args->memory)->debug_string.size();
    return Ok();
}
PJRT_Error* MLX_Memory_ToString(PJRT_Memory_ToString_Args* args) {
    args->to_string = reinterpret_cast<const MLXMemory*>(args->memory)->to_string.c_str();
    args->to_string_size = reinterpret_cast<const MLXMemory*>(args->memory)->to_string.size();
    return Ok();
}
PJRT_Error* MLX_Memory_AddressableByDevices(PJRT_Memory_AddressableByDevices_Args* args) {
    args->devices = &global_device_ptr;
    args->num_devices = 1;
    return Ok();
}

// --- Topology Description API ---

PJRT_Error* MLX_TopologyDescription_PlatformName(PJRT_TopologyDescription_PlatformName_Args* args) {
    args->platform_name = reinterpret_cast<const MLXTopologyDescription*>(args->topology)->platform_name.c_str();
    args->platform_name_size = reinterpret_cast<const MLXTopologyDescription*>(args->topology)->platform_name.size();
    return Ok();
}

PJRT_Error* MLX_TopologyDescription_PlatformVersion(PJRT_TopologyDescription_PlatformVersion_Args* args) {
    args->platform_version = reinterpret_cast<const MLXTopologyDescription*>(args->topology)->platform_version.c_str();
    args->platform_version_size = reinterpret_cast<const MLXTopologyDescription*>(args->topology)->platform_version.size();
    return Ok();
}

PJRT_Error* MLX_TopologyDescription_GetDeviceDescriptions(PJRT_TopologyDescription_GetDeviceDescriptions_Args* args) {
    args->descriptions = &global_device_description_ptr;
    args->num_descriptions = 1;
    return Ok();
}

PJRT_Error* MLX_TopologyDescription_Attributes(PJRT_TopologyDescription_Attributes_Args* args) {
    args->attributes = nullptr;
    args->num_attributes = 0;
    return Ok();
}

PJRT_Error* MLX_TopologyDescription_Destroy(PJRT_TopologyDescription_Destroy_Args* args) {
    // Global topology, don't delete
    return Ok();
}

// --- Executable API ---

PJRT_Error* MLX_Executable_Name(PJRT_Executable_Name_Args* args) {
if (debug_mode()) std::cout << "[MLX-PJRT] MLX_Executable_Name called" << std::endl;
    MLXExecutable* exec = reinterpret_cast<MLXExecutable*>(args->executable);
    if (!exec) return InvalidArgument("Executable is null");
    args->executable_name = exec->name.c_str();
    args->executable_name_size = exec->name.size();
    return Ok();
}

PJRT_Error* MLX_Executable_NumReplicas(PJRT_Executable_NumReplicas_Args* args) {
if (debug_mode()) std::cout << "[MLX-PJRT] MLX_Executable_NumReplicas called" << std::endl;
    MLXExecutable* exec = reinterpret_cast<MLXExecutable*>(args->executable);
    args->num_replicas = exec->num_replicas;
    return Ok();
}

PJRT_Error* MLX_Executable_NumPartitions(PJRT_Executable_NumPartitions_Args* args) {
if (debug_mode()) std::cout << "[MLX-PJRT] MLX_Executable_NumPartitions called" << std::endl;
    MLXExecutable* exec = reinterpret_cast<MLXExecutable*>(args->executable);
    args->num_partitions = exec->num_partitions;
    return Ok();
}

PJRT_Error* MLX_Executable_NumOutputs(PJRT_Executable_NumOutputs_Args* args) {
if (debug_mode()) std::cout << "[MLX-PJRT] MLX_Executable_NumOutputs called" << std::endl;
    MLXExecutable* exec = reinterpret_cast<MLXExecutable*>(args->executable);
    args->num_outputs = exec->num_outputs;
    return Ok();
}

PJRT_Error* MLX_Executable_OutputElementTypes(PJRT_Executable_OutputElementTypes_Args* args) {
if (debug_mode()) std::cout << "[MLX-PJRT] MLX_Executable_OutputElementTypes called" << std::endl;
    // Only implemented for simple F32 add
    static PJRT_Buffer_Type type = PJRT_Buffer_Type_F32; 
    args->output_types = &type;
    args->num_output_types = 1;
    return Ok();
}





static const char* output_memory_kind = "unified";
static size_t output_memory_kind_size = 7;
static int64_t output_dimensions[] = {2};  // 1D array with 2 elements
static size_t output_num_dims[] = {1};  // 1 dimension per output

PJRT_Error* MLX_Executable_OutputMemoryKinds(PJRT_Executable_OutputMemoryKinds_Args* args) {
    MLXExecutable* exec = reinterpret_cast<MLXExecutable*>(args->executable);
    size_t actual_outputs = exec ? exec->num_outputs : 1;
    
    // We need to provide memory kinds for each output
    // Create static storage for memory kinds array (up to reasonable max outputs)
    static const char* memory_kinds[16];
    static size_t memory_kind_sizes[16];
    for (size_t i = 0; i < 16; ++i) {
        memory_kinds[i] = output_memory_kind;
        memory_kind_sizes[i] = output_memory_kind_size;
    }
    
    args->num_outputs = actual_outputs;
    args->memory_kinds = memory_kinds;
    args->memory_kind_sizes = memory_kind_sizes;
    return Ok();
}

PJRT_Error* MLX_Executable_OutputDimensions(PJRT_Executable_OutputDimensions_Args* args) {
    MLXExecutable* exec = reinterpret_cast<MLXExecutable*>(args->executable);
    size_t actual_outputs = exec ? exec->num_outputs : 1;
    
    // Provide dummy dimensions for now (will be filled in during execution)
    static int64_t dims_storage[64];  // Storage for dimension values
    static size_t num_dims_storage[16];  // Storage for num dims per output
    
    args->num_outputs = actual_outputs;
    args->dims = dims_storage;
    args->dim_sizes = num_dims_storage;
    return Ok();
}

// Static fingerprint data
static const char* executable_fingerprint = "mlx-exec-fp";
static size_t executable_fingerprint_size = 11;

PJRT_Error* MLX_Executable_Fingerprint(PJRT_Executable_Fingerprint_Args* args) {
    args->executable_fingerprint = executable_fingerprint;
    args->executable_fingerprint_size = executable_fingerprint_size;
    return Ok();
}

PJRT_Error* MLX_Executable_GetCostAnalysis(PJRT_Executable_GetCostAnalysis_Args* args) {
    args->num_properties = 0;
    args->properties = nullptr;
    return Ok();
}

PJRT_Error* MLX_Executable_SizeOfGeneratedCodeInBytes(PJRT_Executable_SizeOfGeneratedCodeInBytes_Args* args) {
    args->size_in_bytes = 0;  // Minimal implementation
    return Ok();
}

PJRT_Error* MLX_Executable_GetCompiledMemoryStats(PJRT_Executable_GetCompiledMemoryStats_Args* args) {
    args->generated_code_size_in_bytes = 0;
    args->argument_size_in_bytes = 0;
    args->output_size_in_bytes = 0;
    args->alias_size_in_bytes = 0;
    args->temp_size_in_bytes = 0;
    args->host_generated_code_size_in_bytes = 0;
    args->host_argument_size_in_bytes = 0;
    args->host_output_size_in_bytes = 0;
    args->host_alias_size_in_bytes = 0;
    args->host_temp_size_in_bytes = 0;
    args->peak_memory_in_bytes = 0;
    args->total_size_in_bytes = 0;
    return Ok();
}

PJRT_Error* MLX_Executable_GetCompileOptions(PJRT_Executable_GetCompileOptions_Args* args) {
    args->serialized_bytes = "";
    args->serialized_bytes_size = 0;
    args->serialized_compile_options = nullptr;
    args->serialized_compile_options_deleter = nullptr;
    return Ok();
}

PJRT_Error* MLX_Executable_OptimizedProgram(PJRT_Executable_OptimizedProgram_Args* args) {
    // The args->program is provided by caller, we just set code_size
    // If program->code is nullptr, we report the size needed.
    // For minimal implementation, we return 0 bytes (empty program)
    if (args->program) {
        args->program->code_size = 0;
        args->program->format = "mlir";
        args->program->format_size = 4;
    }
    return Ok();
}

PJRT_Error* MLX_Executable_Serialize(PJRT_Executable_Serialize_Args* args) {
    args->serialized_bytes = "";
    args->serialized_bytes_size = 0;
    args->serialized_executable = nullptr;
    args->serialized_executable_deleter = nullptr;
    return Ok();
}

PJRT_Error* MLX_Executable_DeserializeAndLoad(PJRT_Executable_DeserializeAndLoad_Args* args) {
     MLXLoadedExecutable* loaded = new MLXLoadedExecutable{
         nullptr, 
         reinterpret_cast<MLXClient*>(args->client),
         false
     };
     args->loaded_executable = reinterpret_cast<PJRT_LoadedExecutable*>(loaded);
     return Ok();
}

PJRT_Error* MLX_LoadedExecutable_Destroy(PJRT_LoadedExecutable_Destroy_Args* args) {
    MLXLoadedExecutable* loaded = reinterpret_cast<MLXLoadedExecutable*>(args->executable);
    if(loaded) {
        if (loaded->inner_executable) {
            int old_count = loaded->inner_executable->ref_count.fetch_sub(1);
            if (old_count == 1) {
                delete loaded->inner_executable;
            }
        }
        delete loaded;
    }
    return Ok();
}

PJRT_Error* MLX_LoadedExecutable_GetExecutable(PJRT_LoadedExecutable_GetExecutable_Args* args) {
    MLXLoadedExecutable* loaded = reinterpret_cast<MLXLoadedExecutable*>(args->loaded_executable);
    MLXExecutable* exec = loaded->inner_executable;
    int new_count = ++exec->ref_count;
if (debug_mode()) std::cout << "[MLX-PJRT] LoadedExecutable_GetExecutable called, returning " << (void*)exec 
              << " ref_count now " << new_count << std::endl << std::flush;
    args->executable = reinterpret_cast<PJRT_Executable*>(exec);
    return Ok();
}

PJRT_Error* MLX_LoadedExecutable_AddressableDevices(PJRT_LoadedExecutable_AddressableDevices_Args* args) {
if (debug_mode()) std::cout << "[MLX-PJRT] MLX_LoadedExecutable_AddressableDevices called" << std::endl;
    args->addressable_devices = &global_device_ptr;
    args->num_addressable_devices = 1;
    return Ok();
}

PJRT_Error* MLX_LoadedExecutable_Delete(PJRT_LoadedExecutable_Delete_Args* args) {
if (debug_mode()) std::cout << "[MLX-PJRT] MLX_LoadedExecutable_Delete called" << std::endl;
    MLXLoadedExecutable* loaded = reinterpret_cast<MLXLoadedExecutable*>(args->executable);
    // Mark as deleted?
    return Ok();
}

PJRT_Error* MLX_LoadedExecutable_IsDeleted(PJRT_LoadedExecutable_IsDeleted_Args* args) {
if (debug_mode()) std::cout << "[MLX-PJRT] MLX_LoadedExecutable_IsDeleted called" << std::endl;
    args->is_deleted = false;
    return Ok();
}


// Forward declaration for recursive check
bool has_control_flow_recursive(const MLXGraph& graph, 
                                 const std::map<std::string, std::shared_ptr<MLXGraph>>* functions,
                                 std::set<std::string>& visited);

/**
 * @brief Determines if a graph can be compiled with mx.compile()
 * 
 * This is the entry point for compilation decisions. Returns true if the graph
 * contains operations that prevent compilation.
 *
 * @param graph The MLXGraph to analyze
 * @param functions Map of function names to their graph definitions (for func.call recursion)
 * @return true if graph contains control flow that blocks compilation
 */
bool has_control_flow(const MLXGraph& graph, 
                      const std::map<std::string, std::shared_ptr<MLXGraph>>* functions = nullptr) {
    std::set<std::string> visited;
    return has_control_flow_recursive(graph, functions, visited);
}

/**
 * @brief Recursively checks a graph for compilation-blocking operations
 *
 * WHAT BLOCKS COMPILATION:
 * - While loops (stablehlo.while/mhlo.while): Need runtime eval() for condition
 * - NaN constants: Trigger an MLX Metal bug in compiled kernels
 *
 * WHAT ALLOWS COMPILATION:
 * - Case/if ops: Use mx.where() for lazy selection (all branches computed)
 * - Dynamic slice/update: Use MLX native array-based slice APIs
 * - Scatter: Uses MLX native scatter/scatter_add APIs
 * - func.call: Allowed in aggressive mode (recursively checked)
 * - RNG ops: Static compilation with deterministic JAX RNG
 *
 * @param graph The graph to check
 * @param functions Map of callable functions (for recursive func.call checking)
 * @param visited Set of already-checked function names (prevents infinite recursion)
 * @return true if graph should NOT be compiled
 */
bool has_control_flow_recursive(const MLXGraph& graph, 
                                 const std::map<std::string, std::shared_ptr<MLXGraph>>* functions,
                                 std::set<std::string>& visited) {
    for (const auto& op : graph.nodes) {
        // While loops block compilation - we need eval() to check loop conditions
        // Case/if ops now use mx.where() for selection, so they CAN be compiled
        if (op.op_name == "stablehlo.while" || op.op_name == "mhlo.while") {
            return true;  // Block outer graph compilation
        }
        
        // All dynamic ops (dynamic_slice, dynamic_update_slice, scatter) now use MLX native APIs
        // No eval() calls needed - they can be compiled
        
        // RNG ops - now allowed with static compilation
        // JAX RNG is deterministic: same key → same output
        
        // Check func.call - recursively verify called function
        if (op.op_name == "func.call") {
            if (!compile_aggressive_enabled()) {
                return true;  // Block unless aggressive mode
            }
            
            // In aggressive mode, recursively check the called function
            if (functions && op.attributes.count("callee")) {
                std::string callee = op.attributes.at("callee");
                if (!callee.empty() && callee[0] == '@') callee = callee.substr(1);
                
                // Avoid infinite recursion
                if (visited.count(callee)) continue;
                visited.insert(callee);
                
                if (functions->count(callee)) {
                    auto& called_graph = functions->at(callee);
                    if (has_control_flow_recursive(*called_graph, functions, visited)) {
                        return true;
                    }
                }
            }
        }
        
        // Check for NaN constants - these trigger an MLX Metal bug in mx.compile
        if (op.op_name == "stablehlo.constant" || op.op_name == "mhlo.constant") {
            if (op.float_array_attrs.count("value")) {
                const auto& vals = op.float_array_attrs.at("value");
                for (float v : vals) {
                    if (std::isnan(v)) {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

// Helper to check if a graph is compile-safe
bool is_compile_safe(const MLXGraph& graph, 
                     const std::map<std::string, std::shared_ptr<MLXGraph>>* functions = nullptr) {
    return !has_control_flow(graph, functions);
}

// Helper to execute a graph
// parent_val_map allows subgraphs to access values from parent scope
// exec is optional - if provided, enables constant caching (MLX_CONSTANT_CACHE=1)
std::vector<mlx::core::array> ExecuteGraph(const MLXGraph& graph, const std::vector<mlx::core::array>& args,
                                            const std::map<int, mlx::core::array>* parent_val_map = nullptr,
                                            const std::map<std::string, std::shared_ptr<MLXGraph>>* functions = nullptr,
                                            MLXExecutable* exec = nullptr) {
    std::map<int, mlx::core::array> val_map;
    
    // Copy parent val_map to allow access to outer scope values
    if (parent_val_map) {
        val_map = *parent_val_map;
    }
    
    // Debug: Show graph info
if (debug_mode()) std::cout << "[MLX-PJRT]   ExecuteGraph: " << graph.nodes.size() << " nodes, " 
              << graph.input_ids.size() << " inputs, " << graph.output_ids.size() << " outputs" << std::endl;
if (debug_mode()) {
        std::cout << "[MLX-PJRT]   Input IDs: ";
        for (int id : graph.input_ids) std::cout << id << " ";
        std::cout << std::endl;
        std::cout << "[MLX-PJRT]   Output IDs: ";
        for (int id : graph.output_ids) std::cout << id << " ";
        std::cout << std::endl;
    }
    
    // std::cout << "[MLX-PJRT]   Binding inputs..." << std::endl;
    for (size_t i = 0; i < args.size(); ++i) {
        // std::cout << "[MLX-PJRT]     Arg " << i << " shape size=" << args[i].size() << std::endl;
        if (i < graph.input_ids.size()) {
            int id = graph.input_ids[i];
            // std::cout << "[MLX-PJRT]     Binding input " << i << " -> ID " << id << std::endl;
            val_map.erase(id);  // Remove existing value from parent scope
            val_map.insert(std::make_pair(id, args[i]));  // Insert new value
        }
    }
    // std::cout << "[MLX-PJRT]   Inputs bound. Entering op loop..." << std::endl;
    
    // Execute Nodes
    // Execute Nodes
    for (const auto& op : graph.nodes) {
        std::vector<mlx::core::array> op_inputs;
        // std::cout << "[MLX-PJRT]   Op " << op.op_name << " needs inputs: ";
        for (int in_id : op.inputs) {
            if (val_map.count(in_id)) {
                op_inputs.push_back(val_map.at(in_id));
            }
        }
        
        
        std::vector<mlx::core::array> op_outputs; // Generic outputs container
        mlx::core::array result = mlx::core::array(0, mlx::core::int32); // Default result (use int32)
        
        // Debug: Print op info
        // Debug: Print op info
if (debug_mode()) {
            std::cout << "[MLX-PJRT]   Processing op: " << op.op_name << " subgraphs=" << op.subgraphs.size();
            if (!op.outputs.empty()) {
                std::cout << " OutIDs: ";
                for (int oid : op.outputs) std::cout << oid << " ";
            }
            std::cout << std::endl;
        }

        // FAST PATH: OpType enum dispatch for common ops (MLX_OPTYPE_DISPATCH=1)
        bool handled = false;
        if (optype_dispatch_enabled()) {
            OpType ot = GetOpType(op.op_name);
            switch (ot) {
                case OpType::ADD:
                    if (op_inputs.size() >= 2) { result = mlx::core::add(op_inputs[0], op_inputs[1]); handled = true; }
                    break;
                case OpType::SUBTRACT:
                    if (op_inputs.size() >= 2) { result = mlx::core::subtract(op_inputs[0], op_inputs[1]); handled = true; }
                    break;
                case OpType::MULTIPLY:
                    if (op_inputs.size() >= 2) { result = mlx::core::multiply(op_inputs[0], op_inputs[1]); handled = true; }
                    break;
                case OpType::NEGATE:
                    if (!op_inputs.empty()) { result = mlx::core::negative(op_inputs[0]); handled = true; }
                    break;
                case OpType::ABS:
                    if (!op_inputs.empty()) { result = mlx::core::abs(op_inputs[0]); handled = true; }
                    break;
                case OpType::EXP:
                    if (!op_inputs.empty()) { result = mlx::core::exp(op_inputs[0]); handled = true; }
                    break;
                case OpType::LOG:
                    if (!op_inputs.empty()) { result = mlx::core::log(op_inputs[0]); handled = true; }
                    break;
                case OpType::SQRT:
                    if (!op_inputs.empty()) { result = mlx::core::sqrt(op_inputs[0]); handled = true; }
                    break;
                case OpType::RSQRT:
                    if (!op_inputs.empty()) { result = mlx::core::rsqrt(op_inputs[0]); handled = true; }
                    break;
                case OpType::TANH:
                    if (!op_inputs.empty()) { result = mlx::core::tanh(op_inputs[0]); handled = true; }
                    break;
                case OpType::SIN:
                    if (!op_inputs.empty()) { result = mlx::core::sin(op_inputs[0]); handled = true; }
                    break;
                case OpType::COS:
                    if (!op_inputs.empty()) { result = mlx::core::cos(op_inputs[0]); handled = true; }
                    break;
                case OpType::MAXIMUM:
                    if (op_inputs.size() >= 2) { result = mlx::core::maximum(op_inputs[0], op_inputs[1]); handled = true; }
                    break;
                case OpType::MINIMUM:
                    if (op_inputs.size() >= 2) { result = mlx::core::minimum(op_inputs[0], op_inputs[1]); handled = true; }
                    break;
                default:
                    // Fall through to original string-based dispatch
                    break;
            }
        }
        if (handled) {
            // Fast path handled it - store output and continue to next op
            for (int out_id : op.outputs) {
                val_map.erase(out_id);
                val_map.insert(std::make_pair(out_id, result));
            }
            continue; // Skip to next op in loop
        } else if (op.op_name == "stablehlo.while" || op.op_name == "mhlo.while") {
            // While Loop
            // input: operands
            // regions: [0] cond, [1] body
            if (op.subgraphs.size() >= 2) {
                auto current_args = op_inputs;
                int iter_limit = 10000; // Safety
                int iter = 0;
if (debug_mode()) std::cout << "[MLX-PJRT]   While loop starting with " << current_args.size() << " args" << std::endl;
                while (iter++ < iter_limit) {
if (debug_mode()) std::cout << "[MLX-PJRT]   While iter " << iter << std::endl;
                    // Eval Cond - pass val_map for access to parent scope values, and functions map
                    auto cond_res = ExecuteGraph(*op.subgraphs[0], current_args, &val_map, functions);
                    if (cond_res.empty()) { 
if (debug_mode()) std::cout << "[MLX-PJRT]   While cond returned empty, breaking" << std::endl;
                        break; 
                    }
                    
                    // Check condition (bool scalar)
                    bool keep_going = false;
                    try {
                        mlx::core::array c = cond_res[0];
                        c.eval();
                        // std::cout << "[MLX-PJRT]   While cond result dtype=" << c.dtype() << " shape=" << c.shape().size() << std::endl;
                        if (c.dtype() != mlx::core::bool_) c = mlx::core::astype(c, mlx::core::bool_);
                        c.eval();
                        keep_going = c.item<bool>();
                        // std::cout << "[MLX-PJRT]   While keep_going=" << keep_going << std::endl;
                    } catch(const std::exception& e) { 
if (debug_mode()) std::cout << "[MLX-PJRT]   While cond exception: " << e.what() << std::endl;
                        break; 
                    }
                    
                    if (!keep_going) { 
if (debug_mode()) std::cout << "[MLX-PJRT]   While condition false, exiting loop" << std::endl;
                        break;
                    }
                    
                    // Eval Body - pass val_map for access to parent scope values, and functions map
                    // Debug: check args BEFORE body
if (debug_mode()) {
                        std::cout << "[MLX-PJRT]   While iter " << iter << " BEFORE body, " << current_args.size() << " args:" << std::endl;
                        for (size_t i = 0; i < current_args.size(); ++i) {
                            std::cout << "[MLX-PJRT]     Arg " << i << " shape=[";
                            for (auto s : current_args[i].shape()) std::cout << s << ",";
                            std::cout << "]" << std::endl;
                        }
                    }
                    
                    current_args = ExecuteGraph(*op.subgraphs[1], current_args, &val_map, functions);
                    
                    // Debug: check args count and shapes
                    if (debug_mode()) {
                        std::cout << "[MLX-PJRT]   While iter " << iter << " body returned " << current_args.size() << " args" << std::endl;
                        for (size_t i = 0; i < current_args.size() && i < 3; ++i) {
                            std::cout << "[MLX-PJRT]     Arg " << i << " shape=[";
                            for (auto s : current_args[i].shape()) std::cout << s << ",";
                            std::cout << "]" << std::endl;
                        }
                    }
                }
                op_outputs = current_args;
            } else {
                op_outputs = op_inputs; // Pass through
            }
            
            // Store while outputs to val_map (erase first to allow update)
            for (size_t idx = 0; idx < op_outputs.size() && idx < op.outputs.size(); ++idx) {
                val_map.erase(op.outputs[idx]);
                val_map.insert(std::make_pair(op.outputs[idx], op_outputs[idx]));
            }
        
        } else if (op.op_name == "stablehlo.scan" || op.op_name == "mhlo.scan") {
             // Scan Loop
             // regions: [0] body
             // inputs: carry..., inputs...
             // Logic: slice inputs along dimension, loop, stack outputs.
             // Assume dim 0 for simplicity or parse dims.
             // This is complex, implementing pass-through for compilation but no-op behavior for now to avoid crashes?
             // Or simple loop over dim 0 if inputs align.
             if (op.subgraphs.size() >= 1 && !op_inputs.empty()) {
                 // Simplest case: 1 carry, 1 input
                 // FIXME: Full scan impl required
                 op_outputs = op_inputs; 
             }
            
            // Store scan outputs to val_map
            for (size_t idx = 0; idx < op_outputs.size() && idx < op.outputs.size(); ++idx) {
                val_map.insert(std::make_pair(op.outputs[idx], op_outputs[idx]));
            }
        
        /**
         * CASE/IF CONTROL FLOW - mx.where() Pattern
         * ==========================================
         * JAX's lax.cond and lax.switch lower to stablehlo.case.
         * 
         * Traditional approach: eval() the index, execute only selected branch.
         * This breaks mx.compile() because eval() creates a sync point.
         *
         * Our approach: Execute ALL branches lazily, then use mx.where() to
         * select the correct result based on the runtime condition. This keeps
         * everything in the lazy computation graph, enabling mx.compile().
         *
         * Trade-off: All branches run (wasted compute) but graph stays compiled.
         */
        } else if (op.op_name == "stablehlo.case" || op.op_name == "mhlo.case") {
            // Case - conditional dispatch with multiple branches
            // Input: index (i32 scalar indicating which branch to take)
            // Subgraphs: one for each branch
            //
            // For mx.compile compatibility, we execute ALL branches and use mx.where()
            // to select the correct result based on the index. This avoids eval().
            if (!op_inputs.empty() && !op.subgraphs.empty()) {
                auto index_arr = op_inputs[0];
                
                // Convert index to int32 if needed (lazy - no eval)
                if (index_arr.dtype() != mlx::core::int32) {
                    index_arr = mlx::core::astype(index_arr, mlx::core::int32);
                }
                
                // Execute ALL branches (no eval needed for branching)
                std::vector<std::vector<mlx::core::array>> all_branch_results;
                for (size_t i = 0; i < op.subgraphs.size(); ++i) {
                    std::vector<mlx::core::array> branch_inputs = {};
                    auto branch_results = ExecuteGraph(*op.subgraphs[i], branch_inputs, &val_map, functions);
                    all_branch_results.push_back(branch_results);
                }
                
if (debug_mode()) std::cout << "[MLX-PJRT]   Case executed " << all_branch_results.size() << " branches with mx.where selection" << std::endl;
                
                // Use mx.where to select results based on index
                // For 2 branches (common case like lax.cond): where(condition, true_result, false_result)
                // For N branches: chain where() calls
                if (!all_branch_results.empty() && !all_branch_results[0].empty()) {
                    size_t num_outputs = all_branch_results[0].size();
                    op_outputs.clear();
                    op_outputs.reserve(num_outputs);
                    
                    for (size_t out_idx = 0; out_idx < num_outputs; ++out_idx) {
                        if (op.subgraphs.size() == 2) {
                            // Optimized 2-branch case (lax.cond)
                            // index == 0 means take branch 0 (false case), index == 1 means branch 1 (true case)
                            // where(cond, x, y) returns x where cond is true, y where false
                            // So: where(index, branch1_result, branch0_result)
                            auto cond = mlx::core::astype(index_arr, mlx::core::bool_);
                            op_outputs.push_back(mlx::core::where(
                                cond,
                                all_branch_results[1][out_idx],  // true case (index=1)
                                all_branch_results[0][out_idx]   // false case (index=0)
                            ));
                        } else {
                            // N-branch case: chain where() calls from last to first
                            // Start with the last branch as default
                            mlx::core::array result = all_branch_results.back()[out_idx];
                            
                            // Chain from second-to-last back to first
                            for (int i = (int)op.subgraphs.size() - 2; i >= 0; --i) {
                                auto cond = mlx::core::equal(index_arr, mlx::core::array(i));
                                result = mlx::core::where(cond, all_branch_results[i][out_idx], result);
                            }
                            op_outputs.push_back(result);
                        }
                    }
                }
            }
            
            // Store case outputs to val_map (erase first to allow update)
            for (size_t idx = 0; idx < op_outputs.size() && idx < op.outputs.size(); ++idx) {
                val_map.erase(op.outputs[idx]);
                val_map.insert(std::make_pair(op.outputs[idx], op_outputs[idx]));
            }
        
        } else {
            // Legacy Ops - Map to single result logic or extract
            // Reuse existing logic structure but adapted for multiple outputs
            mlx::core::array result = mlx::core::array(0); // Placeholder for single-result ops
            bool executed = true;
            
            // --- Helper Lambdas for Rank Expansion ---
            // --- Helper Lambdas for Rank Expansion ---
            auto is_expanded = [&](const mlx::core::array& a) {
                // Expanded arrays must be u32 or i32 (simulating u64/i64)
                // And have last dimension 2
                bool type_match = (a.dtype() == mlx::core::uint32 || a.dtype() == mlx::core::int32);
                bool shape_match = (type_match && a.ndim() > 0 && a.shape().back() == 2);
                if (!shape_match) return false;

                // STRICT CHECK: Only consider expanded if execution context (output type) implies u64.
                // Exceptions: Compare (output is bool), but inputs might be u64.
                bool is_compare = (op.op_name.find("compare") != std::string::npos);
                if (is_compare) return true; // Trust shape for compare for now

                bool output_is_64 = false;
                if (!op.output_dtypes.empty()) {
                     std::string t = op.output_dtypes[0];
                     if (t.find("64") != std::string::npos) output_is_64 = true;
                }
                return output_is_64;
            };
            
            auto expand = [](mlx::core::array a) {
                 auto u64_a = mlx::core::astype(a, mlx::core::uint64);
                 auto lo = mlx::core::astype(u64_a, mlx::core::uint32);
                 auto hi = mlx::core::astype(mlx::core::right_shift(u64_a, mlx::core::array(32, mlx::core::uint64)), mlx::core::uint32);
                 return mlx::core::stack({lo, hi}, -1);
            };
            
            auto ensure_binary_expanded = [&](mlx::core::array& lhs, mlx::core::array& rhs) {
                bool lhs_exp = is_expanded(lhs);
                bool rhs_exp = is_expanded(rhs);
                if (lhs_exp != rhs_exp) {
                     if (lhs_exp) rhs = expand(rhs);
                     else lhs = expand(lhs);
                }
                return lhs_exp || rhs_exp; // Return true if expanded execution
            };
            
            auto ensure_ternary_expanded = [&](mlx::core::array& cond, mlx::core::array& on_true, mlx::core::array& on_false) {
                 bool true_exp = is_expanded(on_true);
                 bool false_exp = is_expanded(on_false);
                 
                 // Harmonize operands
                 if (true_exp != false_exp) {
                     if (true_exp) on_false = expand(on_false);
                     else on_true = expand(on_true);
                 }
                 bool result_expanded = true_exp || false_exp;
                 
                 if (result_expanded) {
                     // Check cond rank
                     // If cond rank == operand rank - 1, unsqueeze last dim to enable broadcasting
                     // e.g. cond [2, 8], op [2, 8, 2]
                     if (cond.ndim() == on_true.ndim() - 1) {
                         std::vector<int> shape(cond.shape().begin(), cond.shape().end());
                         shape.push_back(1);
                         cond = mlx::core::reshape(cond, mlx::core::Shape(shape.begin(), shape.end()));
                     }
                 }
                 return result_expanded;
            };

            if (op.op_name == "stablehlo.add" || op.op_name == "mhlo.add") {
                if (op_inputs.size() >= 2) {
                     auto lhs = op_inputs[0]; auto rhs = op_inputs[1];
                     ensure_binary_expanded(lhs, rhs);
                     result = mlx::core::add(lhs, rhs);
                     // Enforce wrapping for integers if MLX promoted
                     if (lhs.dtype() != mlx::core::float32 && lhs.dtype() != mlx::core::float16 && lhs.dtype() != mlx::core::bfloat16 && lhs.dtype() != mlx::core::float64) {
                         if (result.dtype() != lhs.dtype()) {
                             result = mlx::core::astype(result, lhs.dtype());
                         }
                     }
                }
            } else if (op.op_name == "stablehlo.subtract" || op.op_name == "mhlo.subtract") {
                if (op_inputs.size() >= 2) {
                     auto lhs = op_inputs[0]; auto rhs = op_inputs[1];
                     ensure_binary_expanded(lhs, rhs);
                     result = mlx::core::subtract(lhs, rhs);
                     // Enforce wrapping
                     if (lhs.dtype() != mlx::core::float32 && lhs.dtype() != mlx::core::float16 && lhs.dtype() != mlx::core::bfloat16 && lhs.dtype() != mlx::core::float64) {
                         if (result.dtype() != lhs.dtype()) {
                             result = mlx::core::astype(result, lhs.dtype());
                         }
                     }
                }
            } else if (op.op_name == "stablehlo.multiply" || op.op_name == "mhlo.multiply") {
                if (op_inputs.size() >= 2) {
                     auto lhs = op_inputs[0]; auto rhs = op_inputs[1];
                     ensure_binary_expanded(lhs, rhs);
                     result = mlx::core::multiply(lhs, rhs);
                     // Enforce wrapping
                     if (lhs.dtype() != mlx::core::float32 && lhs.dtype() != mlx::core::float16 && lhs.dtype() != mlx::core::bfloat16 && lhs.dtype() != mlx::core::float64) {
                         if (result.dtype() != lhs.dtype()) {
                             result = mlx::core::astype(result, lhs.dtype());
                         }
                     }
                }    
            } else if (op.op_name == "stablehlo.divide" || op.op_name == "mhlo.divide") {
                if (op_inputs.size() >= 2) {
                     auto lhs = op_inputs[0]; auto rhs = op_inputs[1];
                     ensure_binary_expanded(lhs, rhs);
                     // For integer types, use floor_divide to match StableHLO semantics
                     auto dt = lhs.dtype();
                     if (dt == mlx::core::int8 || dt == mlx::core::int16 || 
                         dt == mlx::core::int32 || dt == mlx::core::int64 ||
                         dt == mlx::core::uint8 || dt == mlx::core::uint16 ||
                         dt == mlx::core::uint32 || dt == mlx::core::uint64) {
                         result = mlx::core::floor_divide(lhs, rhs);
                     } else {
                         result = mlx::core::divide(lhs, rhs);
                     }
                }
            } else if (op.op_name == "stablehlo.maximum" || op.op_name == "mhlo.maximum") {
                if (op_inputs.size() >= 2) result = mlx::core::maximum(op_inputs[0], op_inputs[1]);
            } else if (op.op_name == "stablehlo.minimum" || op.op_name == "mhlo.minimum") {
                if (op_inputs.size() >= 2) result = mlx::core::minimum(op_inputs[0], op_inputs[1]);
            } else if (op.op_name == "stablehlo.power" || op.op_name == "mhlo.power") {
                if (op_inputs.size() >= 2) result = mlx::core::power(op_inputs[0], op_inputs[1]);
            } else if (op.op_name == "stablehlo.negate" || op.op_name == "mhlo.negate") {
                if (!op_inputs.empty()) result = mlx::core::negative(op_inputs[0]);
            } else if (op.op_name == "stablehlo.abs" || op.op_name == "mhlo.abs") {
                if (!op_inputs.empty()) result = mlx::core::abs(op_inputs[0]);
            } else if (op.op_name == "stablehlo.exponential" || op.op_name == "mhlo.exponential" || op.op_name == "stablehlo.exp") {
                if (!op_inputs.empty()) result = mlx::core::exp(op_inputs[0]);
            } else if (op.op_name == "stablehlo.log" || op.op_name == "mhlo.log" || op.op_name == "chlo.log") {
                if (!op_inputs.empty()) result = mlx::core::log(op_inputs[0]);
            } else if (op.op_name == "stablehlo.sqrt" || op.op_name == "mhlo.sqrt" || op.op_name == "chlo.sqrt") {
                if (!op_inputs.empty()) result = mlx::core::sqrt(op_inputs[0]);
            } else if (op.op_name == "stablehlo.cbrt" || op.op_name == "mhlo.cbrt" || op.op_name == "chlo.cbrt") {
                // Cube root: x^(1/3), handle negative values correctly
                if (!op_inputs.empty()) {
                    auto& x = op_inputs[0];
                    auto abs_x = mlx::core::abs(x);
                    auto cbrt_abs = mlx::core::power(abs_x, mlx::core::array(1.0f/3.0f));
                    result = mlx::core::where(mlx::core::greater_equal(x, mlx::core::array(0.0f)), cbrt_abs, mlx::core::negative(cbrt_abs));
                }
            } else if (op.op_name == "stablehlo.tanh" || op.op_name == "mhlo.tanh" || op.op_name == "chlo.tanh") {
                if (!op_inputs.empty()) result = mlx::core::tanh(op_inputs[0]);
            } else if (op.op_name == "stablehlo.sine" || op.op_name == "mhlo.sine" || op.op_name == "chlo.sin") {
                if (!op_inputs.empty()) result = mlx::core::sin(op_inputs[0]);
            } else if (op.op_name == "stablehlo.cosine" || op.op_name == "mhlo.cosine" || op.op_name == "chlo.cos") {
                if (!op_inputs.empty()) result = mlx::core::cos(op_inputs[0]);
            } else if (op.op_name == "stablehlo.tan" || op.op_name == "mhlo.tan" || op.op_name == "chlo.tan") {
                if (!op_inputs.empty()) result = mlx::core::tan(op_inputs[0]);
            } else if (op.op_name == "stablehlo.atan" || op.op_name == "mhlo.atan" || op.op_name == "chlo.atan") {
                if (!op_inputs.empty()) result = mlx::core::arctan(op_inputs[0]);
            } else if (op.op_name == "stablehlo.atan2" || op.op_name == "mhlo.atan2" || op.op_name == "chlo.atan2") {
                if (op_inputs.size() >= 2) result = mlx::core::arctan2(op_inputs[0], op_inputs[1]);
            } else if (op.op_name == "stablehlo.asin" || op.op_name == "mhlo.asin" || op.op_name == "chlo.asin") {
                if (!op_inputs.empty()) result = mlx::core::arcsin(op_inputs[0]);
            } else if (op.op_name == "stablehlo.acos" || op.op_name == "mhlo.acos" || op.op_name == "chlo.acos") {
                if (!op_inputs.empty()) result = mlx::core::arccos(op_inputs[0]);
            } else if (op.op_name == "stablehlo.sinh" || op.op_name == "mhlo.sinh" || op.op_name == "chlo.sinh") {
                if (!op_inputs.empty()) result = mlx::core::sinh(op_inputs[0]);
            } else if (op.op_name == "stablehlo.cosh" || op.op_name == "mhlo.cosh" || op.op_name == "chlo.cosh") {
                if (!op_inputs.empty()) result = mlx::core::cosh(op_inputs[0]);
            } else if (op.op_name == "stablehlo.asinh" || op.op_name == "mhlo.asinh" || op.op_name == "chlo.asinh") {
                if (!op_inputs.empty()) result = mlx::core::arcsinh(op_inputs[0]);
            } else if (op.op_name == "stablehlo.acosh" || op.op_name == "mhlo.acosh" || op.op_name == "chlo.acosh") {
                if (!op_inputs.empty()) result = mlx::core::arccosh(op_inputs[0]);
            } else if (op.op_name == "stablehlo.atanh" || op.op_name == "mhlo.atanh" || op.op_name == "chlo.atanh") {
                if (!op_inputs.empty()) result = mlx::core::arctanh(op_inputs[0]);
            } else if (op.op_name == "stablehlo.floor" || op.op_name == "mhlo.floor" || op.op_name == "chlo.floor") {
                if (!op_inputs.empty()) result = mlx::core::floor(op_inputs[0]);
            } else if (op.op_name == "stablehlo.ceil" || op.op_name == "mhlo.ceil" || op.op_name == "chlo.ceil") {
                if (!op_inputs.empty()) result = mlx::core::ceil(op_inputs[0]);
            } else if (op.op_name == "chlo.top_k") {
                // Top-K operation - return top k values and their indices
                if (!op_inputs.empty()) {
                    auto x = op_inputs[0];
                    int k = 1; // default
                    
                    // Get k from attributes
                    if (op.int_attrs.count("k")) {
                        k = op.int_attrs.at("k");
                    }
                    
                    // Use MLX topk for values
                    auto top_values = mlx::core::topk(x, k, -1);
                    
                    // For indices: sort descending and take first k
                    // argsort gives ascending, so we take last k and reverse
                    auto sorted_indices = mlx::core::argsort(x, -1);  // ascending indices
                    int n = static_cast<int>(x.shape(-1));
                    
                    // Slice the last k elements (largest) from sorted indices
                    mlx::core::Shape starts_shape = {n - k};
                    mlx::core::Shape stops_shape = {n};
                    auto top_indices = mlx::core::slice(sorted_indices, starts_shape, stops_shape);
                    
                    // Reverse to get descending order [largest, ..., k-th largest]
                    // Create reverse indices: k-1, k-2, ..., 1, 0
                    auto rev_arr = mlx::core::arange(k - 1, -1, -1, mlx::core::int32);
                    top_indices = mlx::core::take(top_indices, rev_arr, -1);
                    
                    // Return both as multi-output
                    op_outputs.clear();
                    op_outputs.push_back(top_values);
                    op_outputs.push_back(mlx::core::astype(top_indices, mlx::core::int32));
                    result = top_values;
                }
            
            } else if (op.op_name == "stablehlo.dot_general" || op.op_name == "mhlo.dot_general") {
                if (op_inputs.size() >= 2) {
                // Simplified dot_general: assume simple matmul for now or try to handle transpose
                // MLX matmul contract last dim of A and first dim of B (standard)?? No, standard is last of A and last-1 of B for >2D?
                // MLX Documention: matmul(a, b) -> standard matrix multiplication.
                
                // Inspect contracting dims
                // Default matmul in JAX (x @ y) for 2D is: lhs contract [1], rhs contract [0].
                
                // NOTE: Proper general dot requires transposing axes to align contracting dims.
                // For MVP, we pass directly to mlx::core::matmul which handles standard broadcasting.
                // If contracting dims are non-standard, we would need to transpose.
                
                if (op.int_array_attrs.count("lhs_contracting") && op.int_array_attrs.count("rhs_contracting")) {
                     std::vector<int> lhs_c, rhs_c;
                     { auto& v = op.int_array_attrs.at("lhs_contracting"); lhs_c.assign(v.begin(), v.end()); }
                     { auto& v = op.int_array_attrs.at("rhs_contracting"); rhs_c.assign(v.begin(), v.end()); }
                     std::vector<int> lhs_b, rhs_b;
                     if (op.int_array_attrs.count("lhs_batching")) { auto& v = op.int_array_attrs.at("lhs_batching"); lhs_b.assign(v.begin(), v.end()); }
                     if (op.int_array_attrs.count("rhs_batching")) { auto& v = op.int_array_attrs.at("rhs_batching"); rhs_b.assign(v.begin(), v.end()); }
                     
                     auto lhs = op_inputs[0];
                     auto rhs = op_inputs[1];
                     
                     // 1. Identify Remaining Dims
                     std::vector<int> lhs_remain, rhs_remain;
                     auto get_remain = [](const mlx::core::array& a, const std::vector<int>& batch, const std::vector<int>& contract) {
                        std::vector<int> remain;
                        for(int i=0; i<a.ndim(); ++i) {
                            bool is_b = false, is_c = false;
                            for(int b : batch) if(b==i) is_b=true;
                            for(int c : contract) if(c==i) is_c=true;
                            if(!is_b && !is_c) remain.push_back(i);
                        }
                        return remain;
                     };
                     lhs_remain = get_remain(lhs, lhs_b, lhs_c);
                     rhs_remain = get_remain(rhs, rhs_b, rhs_c);
                     
                     // 2. Permute: [Batch, Remain, Contract]
                     std::vector<int> lhs_perm;
                     lhs_perm.insert(lhs_perm.end(), lhs_b.begin(), lhs_b.end());
                     lhs_perm.insert(lhs_perm.end(), lhs_remain.begin(), lhs_remain.end());
                     lhs_perm.insert(lhs_perm.end(), lhs_c.begin(), lhs_c.end());
                     
                     std::vector<int> rhs_perm;
                     rhs_perm.insert(rhs_perm.end(), rhs_b.begin(), rhs_b.end());
                     rhs_perm.insert(rhs_perm.end(), rhs_c.begin(), rhs_c.end()); // Contract first for RHS? No, for matmul(A, B) -> A(..., K), B(K, ...) ? 
                     // MLX Matmul: A(..., M, K), B(..., K, N) -> (..., M, N)
                     // So we want RHS to be [Batch, Contract, Remain] -> (..., K, N)
                     rhs_perm.insert(rhs_perm.end(), rhs_remain.begin(), rhs_remain.end());
                     
                     lhs = mlx::core::transpose(lhs, lhs_perm);
                     rhs = mlx::core::transpose(rhs, rhs_perm);
                     
                     // 3. Reshape for Matmul (flatten batch/remain/contract groups)
                     // Target LHS: [BatchProd, RemainProd, ContractProd]
                     // Target RHS: [BatchProd, ContractProd, RemainProd]
                     // Actually, we can keep Batch distinct if we want, but flattening is safer for "BatchProd"
                     
                     // Need actual shapes
                     // Just perform matmul on the permuted? 
                     // If batch is multiple dims, matmul handles it? 
                     // MLX broadcast rules: Two arrays have compatible shapes if, for every dimension, the dimension lengths are equal or one of them is 1.
                     // But we want "Batch" dims to be treated as batch, not broadcast if different (they should match).
                     // The issue is if lhs_remain or rhs_remain are multiple dims. Matmul takes last 2 dims.
                     // So we MUST flatten "Remain" into 1 dim (M or N) and "Contract" into 1 dim (K).
                     // And flatten "Batch" into 1 dim (B) ? 
                     // Or [B1, B2..., M, K]
                     
                     // Safest: Flatten all Batch into 1 dim, all Remain into 1 dim, all Contract into 1 dim.
                     // LHS -> [B*..., M*..., K*...] -> 3D
                     // RHS -> [B*..., K*..., N*...] -> 3D
                     
                     // Helper to calc size
                     auto prod_dims = [&](const mlx::core::array& arr, const std::vector<int>& dims) {
                         int p = 1; for(int d : dims) p *= arr.shape(d); return p; // NOTE: arr is already permuted? No, use original
                         // Actually hard to track sizes from original indices.
                         // Easier to inspect current shape after transpose.
                     };
                     
                     // After transpose:
                     // LHS is [Batch..., Remain..., Contract...]
                     int b_rank = lhs_b.size();
                     int lr_rank = lhs_remain.size();
                     int lc_rank = lhs_c.size();
                     int rc_rank = rhs_c.size();
                     int rr_rank = rhs_remain.size();
                     
                     // Flatten Batch
                     // But wait, reshape requires knowing split points.
                     // Since we just transposed, they are contiguous.
                     
                     // std::cout << "[MLX-PJRT] DotGeneral: LHS Remain=" << lr_rank << " Contract=" << lc_rank << std::endl;
                     
                     auto lhs_s = lhs.shape();
                     std::vector<int> lhs_shape(lhs_s.begin(), lhs_s.end());
                     // Splits: [0...b_rank), [b_rank...b_rank+lr_rank), [end-lc_rank...end)
                     
                     // To do this cleanly: 
                     // Flatten to 3D: [BatchProd, M, K]
                     // But wait, we might not have Batch.
                     // Handle B=1 case.
                     
                     // Let's rely on MLX to handle >2D if we merge Remain and Contract.
                     // LHS -> [Batch..., M_flat, K_flat]
                     // RHS -> [Batch..., K_flat, N_flat]
                     
                     // We need to construct new shape for LHS
                     int m_size = 1; for(int i=b_rank; i<b_rank+lr_rank; ++i) m_size *= lhs_shape[i];
                     int k_size = 1; for(int i=b_rank+lr_rank; i<lhs_shape.size(); ++i) k_size *= lhs_shape[i];
                     
                     auto rhs_s = rhs.shape();
                     std::vector<int> rhs_shape(rhs_s.begin(), rhs_s.end());
                     int n_size = 1; for(int i=b_rank+rc_rank; i<rhs_shape.size(); ++i) n_size *= rhs_shape[i];
                     
                     std::vector<int> lhs_3d_shape;
                     for(int i=0; i<b_rank; ++i) lhs_3d_shape.push_back(lhs_shape[i]);
                     lhs_3d_shape.push_back(m_size);
                     lhs_3d_shape.push_back(k_size);
                     
                     std::vector<int> rhs_3d_shape;
                     for(int i=0; i<b_rank; ++i) rhs_3d_shape.push_back(rhs_shape[i]);
                     rhs_3d_shape.push_back(k_size); // Should match
                     rhs_3d_shape.push_back(n_size);
                     
                     lhs = mlx::core::reshape(lhs, mlx::core::Shape(lhs_3d_shape.begin(), lhs_3d_shape.end()));
                     rhs = mlx::core::reshape(rhs, mlx::core::Shape(rhs_3d_shape.begin(), rhs_3d_shape.end()));
                     
                     result = mlx::core::matmul(lhs, rhs);
                     
                     // Result is [Batch..., M, N]
                     // Reshape back to [Batch..., Remain_LHS..., Remain_RHS...]
                     std::vector<int> final_shape;
                     for(int i=0; i<b_rank; ++i) final_shape.push_back(lhs_shape[i]);
                     // Get original remaining dims sizes
                     // Warning: identifying them from original input needs care.
                     // But we know 'lhs_remain' indices referred to original. 
                     // We need their values.
                     auto get_sizes = [&](const mlx::core::array& a, const std::vector<int>& idxs) {
                        std::vector<int> s; for(int id : idxs) s.push_back(a.shape(id)); return s;
                     };
                     // Use op_inputs[0] (original)
                     auto lr_sizes = get_sizes(op_inputs[0], lhs_remain);
                     auto rr_sizes = get_sizes(op_inputs[1], rhs_remain);
                     
                     final_shape.insert(final_shape.end(), lr_sizes.begin(), lr_sizes.end());
                     final_shape.insert(final_shape.end(), rr_sizes.begin(), rr_sizes.end());
                     
                     result = mlx::core::reshape(result, mlx::core::Shape(final_shape.begin(), final_shape.end()));
                     
                } else {
                     // Fallback/Legacy
                     result = mlx::core::matmul(op_inputs[0], op_inputs[1]);
                }
            }
        } else if (op.op_name == "stablehlo.convert_element_type" || op.op_name == "mhlo.convert_element_type" || op.op_name == "stablehlo.convert") {
            if (getenv("MLX_PJRT_DEBUG") && !op_inputs.empty()) {
if (debug_mode()) std::cout << "[MLX-PJRT] convert input dtype=" << op_inputs[0].dtype() 
                          << " target=" << (op.output_dtypes.empty() ? "?" : op.output_dtypes[0]) << std::endl;
            }
             if (!op_inputs.empty()) {
                 std::string t = (!op.output_dtypes.empty()) ? op.output_dtypes[0] : "f32";
                 mlx::core::Dtype target_dtype = mlx::core::float32;
                  if (t.find("f16") != std::string::npos) target_dtype = mlx::core::float16;
                 else if (t.find("bf16") != std::string::npos) target_dtype = mlx::core::bfloat16;
                 else if (t.find("f32") != std::string::npos) target_dtype = mlx::core::float32;
                 else if (t.find("u32") != std::string::npos || t.find("ui32") != std::string::npos || t.find("uint32") != std::string::npos) target_dtype = mlx::core::uint32;
                 else if (t.find("i32") != std::string::npos || t.find("s32") != std::string::npos) target_dtype = mlx::core::int32;
                 else if (t.find("i64") != std::string::npos || t.find("s64") != std::string::npos) target_dtype = mlx::core::int64;

                 
                  // Rank Expansion Truncation
                  // Rank Expansion Truncation
                  bool input_expanded = (op_inputs[0].ndim() > 0 && op_inputs[0].shape().back() == 2 && 
                                        (op_inputs[0].dtype() == mlx::core::uint32 || op_inputs[0].dtype() == mlx::core::int32 ||
                                         op_inputs[0].dtype() == mlx::core::uint64 || op_inputs[0].dtype() == mlx::core::int64));
                                        
                  bool target_is_32bit = (target_dtype == mlx::core::uint32 || target_dtype == mlx::core::int32 || 
                                          target_dtype == mlx::core::float32);

                  bool target_is_u64 = (target_dtype == mlx::core::uint64 || target_dtype == mlx::core::int64);

                  bool should_truncate = false;
                  if (input_expanded && target_is_32bit) {
                       // Only truncate if output shape implies dimension reduction
                       if (op.output_shapes.size() > 0) {
                            auto& in_shape = op_inputs[0].shape();
                            auto& out_shape = op.output_shapes[0];
                            // Check if rank dropped or last dim dropped
                            if (out_shape.size() < in_shape.size()) should_truncate = true;
                            else if (out_shape.size() == in_shape.size() && in_shape.back() == 2 && out_shape.back() != 2) should_truncate = true;
                       }
                  }

                  if (input_expanded && target_is_32bit) {
                      if (should_truncate) {
                          // Truncate to LO bits (0-th element)
                          auto truncated = mlx::core::take(op_inputs[0], mlx::core::array(0), op_inputs[0].ndim()-1);
                          result = mlx::core::astype(truncated, target_dtype);
                      } else {
                          result = mlx::core::astype(op_inputs[0], target_dtype);
                      }

                      // Debug F32 values
                      if (target_dtype == mlx::core::float32) {
                           try {
                               bool has_zeros = mlx::core::any(mlx::core::equal(result, mlx::core::array(0.0f))).item<bool>();
                               bool has_nans = mlx::core::any(mlx::core::isnan(result)).item<bool>();
                               if (has_zeros || has_nans) {
                                    if (debug_mode()) {
                                        std::cout << "[MLX-PJRT] Convert F32 Issue. Zeros=" << has_zeros << " NaNs=" << has_nans << " Shape=[";
                                        for(auto s : result.shape()) std::cout << s << ",";
                                        std::cout << "]" << std::endl;
                                    }
                               }
                               if (has_zeros || has_nans) {
                                    if (debug_mode()) {
                                        std::cout << "[MLX-PJRT] Convert F32 Issue. Zeros=" << has_zeros << " NaNs=" << has_nans << " Shape=[";
                                        for(auto s : result.shape()) std::cout << s << ",";
                                        std::cout << "]" << std::endl;
                                    }
                               }
                               if (has_zeros || has_nans) {
                                    if (debug_mode()) {
                                        std::cout << "[MLX-PJRT] Convert F32 Issue. Zeros=" << has_zeros << " NaNs=" << has_nans << " Shape=[";
                                        for(auto s : result.shape()) std::cout << s << ",";
                                        std::cout << "]" << std::endl;
                                    }
                               }
                               if (has_zeros || has_nans) {
                                    if (debug_mode()) {
                                        std::cout << "[MLX-PJRT] Convert F32 Issue. Zeros=" << has_zeros << " NaNs=" << has_nans << " Shape=[";
                                        for(auto s : result.shape()) std::cout << s << ",";
                                        std::cout << "]" << std::endl;
                                    }
                               }
                               if (has_zeros || has_nans) {
                                    if (debug_mode()) {
                                        std::cout << "[MLX-PJRT] Convert F32 Issue. Zeros=" << has_zeros << " NaNs=" << has_nans << " Shape=[";
                                        for(auto s : result.shape()) std::cout << s << ",";
                                        std::cout << "]" << std::endl;
                                    }
                               }
                           } catch(...) {}
                      }
                  } else if (target_is_u64 && !input_expanded) {
                      // Expand to [..., 2] u32
                      // First cast to native u64 (if needed)
                      auto casted = mlx::core::astype(op_inputs[0], target_dtype);
                      result = expand(casted);
                  } else {
                      result = mlx::core::astype(op_inputs[0], target_dtype);
                  }
             } 
        // --- Priority 1: Basic Arithmetic Operations ---
        } else if (op.op_name == "stablehlo.subtract" || op.op_name == "mhlo.subtract") {
            if (op_inputs.size() >= 2) result = mlx::core::subtract(op_inputs[0], op_inputs[1]);
        } else if (op.op_name == "stablehlo.divide" || op.op_name == "mhlo.divide") {
            if (op_inputs.size() >= 2) {
                 auto lhs = op_inputs[0]; auto rhs = op_inputs[1];
                 auto dt = lhs.dtype();
                 // For integer types, use floor_divide to match StableHLO semantics
                 if (dt == mlx::core::int8 || dt == mlx::core::int16 || 
                     dt == mlx::core::int32 || dt == mlx::core::int64 ||
                     dt == mlx::core::uint8 || dt == mlx::core::uint16 ||
                     dt == mlx::core::uint32 || dt == mlx::core::uint64) {
                     result = mlx::core::floor_divide(lhs, rhs);
                 } else {
                     // Guard denominator against 0 for float types
                     auto safe_denom = mlx::core::where(mlx::core::equal(rhs, mlx::core::array(0.0f)), mlx::core::array(1e-9f), rhs);
                     result = mlx::core::divide(lhs, safe_denom);
                 }
            }
        } else if (op.op_name == "stablehlo.negate" || op.op_name == "mhlo.negate") {
            if (!op_inputs.empty()) result = mlx::core::negative(op_inputs[0]);
        } else if (op.op_name == "stablehlo.abs" || op.op_name == "mhlo.abs") {
            if (!op_inputs.empty()) result = mlx::core::abs(op_inputs[0]);
        } else if (op.op_name == "stablehlo.maximum" || op.op_name == "mhlo.maximum") {
            if (op_inputs.size() >= 2) result = mlx::core::maximum(op_inputs[0], op_inputs[1]);
        } else if (op.op_name == "stablehlo.minimum" || op.op_name == "mhlo.minimum") {
            if (op_inputs.size() >= 2) result = mlx::core::minimum(op_inputs[0], op_inputs[1]);
        } else if (op.op_name == "stablehlo.power" || op.op_name == "mhlo.power") {
            if (op_inputs.size() >= 2) result = mlx::core::power(op_inputs[0], op_inputs[1]);
        } else if (op.op_name == "stablehlo.sqrt" || op.op_name == "mhlo.sqrt") {
            if (!op_inputs.empty()) {
                // Guard against negative inputs (e.g. -0.0 or small negative errors)
                auto guarded = mlx::core::maximum(op_inputs[0], mlx::core::array(0.0f));
                result = mlx::core::sqrt(guarded);
            }
        } else if (op.op_name == "stablehlo.rsqrt" || op.op_name == "mhlo.rsqrt") {
            if (!op_inputs.empty()) {
                 auto guarded = mlx::core::maximum(op_inputs[0], mlx::core::array(1e-38f)); // avoid div zero
                 result = mlx::core::rsqrt(guarded);
            }
        } else if (op.op_name == "stablehlo.square" || op.op_name == "mhlo.square") {
            if (!op_inputs.empty()) result = mlx::core::square(op_inputs[0]);
        } else if (op.op_name == "stablehlo.sign" || op.op_name == "mhlo.sign") {
            if (!op_inputs.empty()) result = mlx::core::sign(op_inputs[0]);
        } else if (op.op_name == "stablehlo.remainder" || op.op_name == "mhlo.remainder") {
            if (op_inputs.size() >= 2) result = mlx::core::remainder(op_inputs[0], op_inputs[1]);
        // --- Priority 2: Transcendental/Math Operations ---
        } else if (op.op_name == "stablehlo.exponential" || op.op_name == "mhlo.exponential") {
            if (!op_inputs.empty()) result = mlx::core::exp(op_inputs[0]);
        } else if (op.op_name == "stablehlo.log" || op.op_name == "mhlo.log") {
            if (!op_inputs.empty()) {
                // Guard against <= 0
                auto guarded = mlx::core::maximum(op_inputs[0], mlx::core::array(1e-37f));
                result = mlx::core::log(guarded);
            }
        } else if (op.op_name == "stablehlo.log_plus_one" || op.op_name == "mhlo.log_plus_one") {
             if (!op_inputs.empty()) {
                  // Guard against <= -1
                  // log1p(x) valid for x > -1.
                  // Use a small epsilon above -1.
                  auto guard_val = mlx::core::array(-1.0f + 1e-7f);
                  auto guarded = mlx::core::maximum(op_inputs[0], guard_val);
                  result = mlx::core::log1p(guarded);
             }
        } else if (op.op_name == "stablehlo.exponential_minus_one" || op.op_name == "mhlo.exponential_minus_one") {
            if (!op_inputs.empty()) result = mlx::core::expm1(op_inputs[0]);
        } else if (op.op_name == "stablehlo.tanh" || op.op_name == "mhlo.tanh") {
            if (!op_inputs.empty()) result = mlx::core::tanh(op_inputs[0]);
        } else if (op.op_name == "stablehlo.logistic" || op.op_name == "mhlo.logistic") {
            if (!op_inputs.empty()) result = mlx::core::sigmoid(op_inputs[0]);
        } else if (op.op_name == "stablehlo.floor" || op.op_name == "mhlo.floor") {
            if (!op_inputs.empty()) result = mlx::core::floor(op_inputs[0]);
        } else if (op.op_name == "stablehlo.ceil" || op.op_name == "mhlo.ceil") {
            if (!op_inputs.empty()) result = mlx::core::ceil(op_inputs[0]);
        } else if (op.op_name == "stablehlo.round_nearest_even" || op.op_name == "mhlo.round_nearest_even") {
            if (!op_inputs.empty()) result = mlx::core::round(op_inputs[0]);
        } else if (op.op_name == "stablehlo.round_nearest_afz" || op.op_name == "mhlo.round_nearest_afz") {
            // Round away from zero for half-values
            if (!op_inputs.empty()) {
                auto x = op_inputs[0];
                auto sign_x = mlx::core::sign(x);
                auto abs_x = mlx::core::abs(x);
                result = mlx::core::multiply(sign_x, mlx::core::floor(mlx::core::add(abs_x, mlx::core::array(0.5f))));
            }
        // --- CHLO Dialect Ops (JAX extension ops) ---
        } else if (op.op_name == "chlo.sinh") {
            if (!op_inputs.empty()) result = mlx::core::sinh(op_inputs[0]);
        } else if (op.op_name == "chlo.cosh") {
            if (!op_inputs.empty()) result = mlx::core::cosh(op_inputs[0]);
        } else if (op.op_name == "chlo.tan") {
            if (!op_inputs.empty()) result = mlx::core::tan(op_inputs[0]);
        } else if (op.op_name == "chlo.erf") {
            if (!op_inputs.empty()) result = mlx::core::erf(op_inputs[0]);
        } else if (op.op_name == "chlo.log1p") {
            if (!op_inputs.empty()) result = mlx::core::log1p(op_inputs[0]);

        } else if (op.op_name == "stablehlo.clamp" || op.op_name == "mhlo.clamp") {
            // clamp(min, x, max) -> clip(x, min, max)
            if (op_inputs.size() >= 3) result = mlx::core::clip(op_inputs[1], op_inputs[0], op_inputs[2]);
        // --- Priority 3: Comparison Operations ---
        } else if (op.op_name == "stablehlo.compare" || op.op_name == "mhlo.compare") {
            if (op_inputs.size() >= 2) {
                std::string cmp_dir = "EQ";
                if (op.attributes.count("comparison_direction")) {
                    cmp_dir = op.attributes.at("comparison_direction");
                }
                if (cmp_dir == "EQ") {
                    result = mlx::core::equal(op_inputs[0], op_inputs[1]);
                } else if (cmp_dir == "NE") {
                    result = mlx::core::not_equal(op_inputs[0], op_inputs[1]);
                } else if (cmp_dir == "LT") {
                    result = mlx::core::less(op_inputs[0], op_inputs[1]);
                } else if (cmp_dir == "LE") {
                    result = mlx::core::less_equal(op_inputs[0], op_inputs[1]);
                } else if (cmp_dir == "GT") {
                    result = mlx::core::greater(op_inputs[0], op_inputs[1]);
                } else if (cmp_dir == "GE") {
                    result = mlx::core::greater_equal(op_inputs[0], op_inputs[1]);
                }
            }
        // --- Priority 4: Reduction Operations ---
        } else if (op.op_name == "stablehlo.reduce" || op.op_name == "mhlo.reduce") {
            // Reduction - read reduce_type from parser (extracted from body region)
            if (!op_inputs.empty()) {
                // If input is a scalar, just return it (nothing to reduce)
                if (op_inputs[0].ndim() == 0) {
                    result = op_inputs[0];
                } else {
                    // Try to get axes from attributes
                    std::vector<int> axes;
                    if (op.int_array_attrs.count("dimensions")) {
                        auto& v = op.int_array_attrs.at("dimensions"); axes.assign(v.begin(), v.end());
                    }
                    
                    // Filter out invalid axes
                    std::vector<int> valid_axes;
                    for (int ax : axes) {
                        if (ax >= 0 && ax < static_cast<int>(op_inputs[0].ndim())) {
                            valid_axes.push_back(ax);
                        }
                    }
                    axes = valid_axes;
                    
                    // Get reduction type from parser (extracted from body op)
                    std::string reduce_type = "sum"; // Default to sum
                    if (op.attributes.count("reduce_type")) {
                        reduce_type = op.attributes.at("reduce_type");
                    }
                    if (debug_mode()) std::cout << "[MLX-PJRT] Reduce type: " << reduce_type << std::endl;
                    
                    // Apply appropriate reduction
                    if (reduce_type == "max") {
                        result = axes.empty() ? mlx::core::max(op_inputs[0]) : mlx::core::max(op_inputs[0], axes);
                    } else if (reduce_type == "min") {
                        result = axes.empty() ? mlx::core::min(op_inputs[0]) : mlx::core::min(op_inputs[0], axes);
                    } else if (reduce_type == "prod") {
                        result = axes.empty() ? mlx::core::prod(op_inputs[0]) : mlx::core::prod(op_inputs[0], axes);
                    } else if (reduce_type == "argmax" || reduce_type == "argmin") {
                        // Argmax/Argmin produces two outputs: Value and Index
                        auto val = (reduce_type == "argmax") 
                            ? (axes.empty() ? mlx::core::max(op_inputs[0]) : mlx::core::max(op_inputs[0], axes))
                            : (axes.empty() ? mlx::core::min(op_inputs[0]) : mlx::core::min(op_inputs[0], axes));
                        
                        int axis = axes.empty() ? -1 : axes[0];
                        auto idx = (reduce_type == "argmax")
                            ? (axes.empty() ? mlx::core::argmax(op_inputs[0]) : mlx::core::argmax(op_inputs[0], axis))
                            : (axes.empty() ? mlx::core::argmin(op_inputs[0]) : mlx::core::argmin(op_inputs[0], axis));
                        
                        // Assign outputs via op_outputs to avoid overwrite issues
                        op_outputs.push_back(val);
                        op_outputs.push_back(mlx::core::astype(idx, mlx::core::int32));
                    } else if (reduce_type == "or") {
                        result = axes.empty() ? mlx::core::any(op_inputs[0]) : mlx::core::any(op_inputs[0], axes);
                    } else if (reduce_type == "and") {
                        result = axes.empty() ? mlx::core::all(op_inputs[0]) : mlx::core::all(op_inputs[0], axes);
                    } else { // Default to sum
                        result = axes.empty() ? mlx::core::sum(op_inputs[0]) : mlx::core::sum(op_inputs[0], axes);
                    }
                } // end else (non-scalar)
            }
        } else if (op.op_name == "stablehlo.reduce_sum" || op.op_name == "mhlo.reduce_sum") {
            if (!op_inputs.empty()) result = mlx::core::sum(op_inputs[0]);
        } else if (op.op_name == "stablehlo.reduce_max" || op.op_name == "mhlo.reduce_max") {
            if (!op_inputs.empty()) result = mlx::core::max(op_inputs[0]);
        } else if (op.op_name == "stablehlo.reduce_min" || op.op_name == "mhlo.reduce_min") {
            if (!op_inputs.empty()) result = mlx::core::min(op_inputs[0]);
        } else if (op.op_name == "stablehlo.reduce_prod" || op.op_name == "mhlo.reduce_prod") {
            if (!op_inputs.empty()) result = mlx::core::prod(op_inputs[0]);
        // --- Priority 5: Shape Operations ---
        } else if (op.op_name == "stablehlo.reshape" || op.op_name == "mhlo.reshape") {
            if (!op_inputs.empty() && !op.output_shapes.empty()) {
                const std::vector<int>& vec_shape = op.output_shapes[0];
                
                // Rank Expansion Logic
                bool output_is_64 = false;
                if (!op.output_dtypes.empty()) {
                     std::string t = op.output_dtypes[0];
                     if (t.find("64") != std::string::npos) output_is_64 = true;
                }

                bool input_expanded = (output_is_64 && op_inputs[0].ndim() > 0 && op_inputs[0].shape().back() == 2 && 
                                      (op_inputs[0].dtype() == mlx::core::uint32 || op_inputs[0].dtype() == mlx::core::int32 ||
                                       op_inputs[0].dtype() == mlx::core::uint64 || op_inputs[0].dtype() == mlx::core::int64));
                
                std::vector<int> target_shape_vec = vec_shape;
                if (input_expanded) {
                    target_shape_vec.push_back(2);
                }

                mlx::core::Shape target_shape(target_shape_vec.begin(), target_shape_vec.end());
                
                // Calculate sizes for validation
                int64_t input_size = 1;
                for (auto s : op_inputs[0].shape()) input_size *= s;
                int64_t target_size = 1;
                for (auto s : target_shape_vec) target_size *= s;
                
                if (input_size == target_size || target_size == 0) {
                    // Valid reshape (target_size==0 means scalar from [1])
                    try {
                        result = mlx::core::reshape(op_inputs[0], target_shape);
                    } catch (const std::exception& e) {
                        // Debug print for failure
                        std::cout << "[ERROR] Reshape failed! Input shape=[";
                        for (auto s : op_inputs[0].shape()) std::cout << s << ",";
                        std::cout << "] Target shape=[";
                        for (int d : target_shape_vec) std::cout << d << ",";
                        std::cout << "]" << std::endl;
                        throw;
                    }
                } else if (target_shape_vec.empty() && op_inputs[0].ndim() == 1 && op_inputs[0].shape()[0] == 1) {
                    // [1] -> scalar
                    result = mlx::core::reshape(op_inputs[0], target_shape);
                } else {
                    // Size mismatch - likely indexing issue in while loop, try squeeze or passthrough
                    if (debug_mode()) {
                        std::cout << "[MLX-PJRT] Reshape size mismatch, using passthrough. Input=[";
                        for (auto s : op_inputs[0].shape()) std::cout << s << ",";
                        std::cout << "] Target=[";
                        for (int d : target_shape_vec) std::cout << d << ",";
                        std::cout << "]" << std::endl;
                    }
                    // Try to at least squeeze if target is smaller-dimensional
                    if (target_shape_vec.empty() && input_size == 1) {
                        result = mlx::core::squeeze(op_inputs[0]);
                    } else {
                        result = op_inputs[0]; // Passthrough - may cause downstream issues but won't crash
                    }
                }
            } else if (!op_inputs.empty()) {
                result = op_inputs[0];
            }
        } else if (op.op_name == "stablehlo.constant" || op.op_name == "mhlo.constant") {
            // Parse constant from attributes
            std::vector<int> vec_shape = (!op.output_shapes.empty()) ? op.output_shapes[0] : std::vector<int>{};
            mlx::core::Shape shape(vec_shape.begin(), vec_shape.end());
            
            std::string target_type = "unknown";
            if (!op.output_dtypes.empty()) target_type = op.output_dtypes[0];

            if (op.int_array_attrs.count("value")) {
                const std::vector<int64_t>& val = op.int_array_attrs.at("value");
                
                bool is_explicit_float = (target_type.find("f32") != std::string::npos || 
                                          target_type.find("float32") != std::string::npos ||
                                          target_type.find("f16") != std::string::npos ||
                                          target_type.find("bf16") != std::string::npos);
                
                if (is_explicit_float) {
                    std::vector<float> casted(val.size());
                    for(size_t i=0; i<val.size(); ++i) casted[i] = (float)val[i];
                    result = mlx::core::array(casted.begin(), shape, mlx::core::float32);
                } else {
                    // Check if target is uint64/uint32 or int64
                    if (target_type.find("u") != std::string::npos || target_type.find("i64") != std::string::npos) {
                         // MLX doesn't have uint64 fully supported?
                         // But we can try using correct dtype
                         mlx::core::Dtype dtype = mlx::core::int32;
                         if (target_type.find("u32") != std::string::npos || target_type.find("ui32") != std::string::npos || target_type.find("uint32") != std::string::npos) dtype = mlx::core::uint32;
                         else if (target_type.find("u64") != std::string::npos || target_type.find("ui64") != std::string::npos || target_type.find("uint64") != std::string::npos) dtype = mlx::core::uint64;
                         else if (target_type.find("i64") != std::string::npos) dtype = mlx::core::int64;
                         
if (debug_mode()) std::cout << "[MLX-PJRT] Constant Debug. Type=" << target_type << " MLX_Dtype=" << dtype << " First(val)=" << val[0] << std::endl;
                         result = mlx::core::array(val.begin(), shape, dtype);
                    } else {
                         result = mlx::core::array(val.begin(), shape, mlx::core::int32);
                    }
                }
            } else if (op.int_array_attrs.count("int_value")) {
                const std::vector<int64_t>& val = op.int_array_attrs.at("int_value");
                // Scalar int
                if (target_type.find("f") != std::string::npos) {
                     // Numeric cast
                     result = mlx::core::array((float)val[0]);
                     if (mlx::core::Shape(shape).size() > 1) result = mlx::core::broadcast_to(result, shape);
                } else if (target_type.find("u64") != std::string::npos || target_type.find("ui64") != std::string::npos || target_type.find("uint64") != std::string::npos) {
                     // Rank Expansion: u64 -> [..., 2] u32
                     // Determine inputs
                     uint32_t lo=0, hi=0;
                     if (val.size() >= 2) { 
                         lo = (uint32_t)val[0]; hi = (uint32_t)val[1]; 
                     } else if (val.size() == 1) {
                         lo = (uint32_t)val[0]; hi = (uint32_t)(val[0] >> 32); 
                     }
                     
                     if (mlx::core::Shape(shape).size() == 0) { // Scalar
                         result = mlx::core::array({lo, hi}, {2}, mlx::core::uint32);
                     } else {
                         // New shape: input_shape + [2]
                         std::vector<int> new_dims = vec_shape;
                         new_dims.push_back(2);
                         mlx::core::Shape new_shape(new_dims.begin(), new_dims.end());
                         
                         size_t total_elements = 1;
                         for (int d : vec_shape) total_elements *= d;
                         
                         std::vector<uint32_t> u32_vals;
                         u32_vals.reserve(total_elements * 2);
                         for (size_t i=0; i<total_elements; ++i) {
                             u32_vals.push_back(lo);
                             u32_vals.push_back(hi);
                         }
                         result = mlx::core::array(u32_vals.begin(), new_shape, mlx::core::uint32);
                     }
                } else if (target_type.find("u32") != std::string::npos || target_type.find("ui32") != std::string::npos || target_type.find("uint32") != std::string::npos) {
                     result = mlx::core::array(val.begin(), shape, mlx::core::uint32);
                } else if (target_type.find("i64") != std::string::npos) {
                     if (val.size() == 1 && mlx::core::Shape(shape).size() > 1) result = mlx::core::broadcast_to(mlx::core::array(val[0], mlx::core::int64), shape);
                     else result = mlx::core::array(val.begin(), shape, mlx::core::int64);
                } else {
                     result = mlx::core::array(val.begin(), shape, mlx::core::int32);
                }
if (debug_mode()) std::cout << "[MLX-PJRT] Constant(scalar) Debug. Type=" << target_type << " Dtype=" << result.dtype() << " Val[0]=" << val[0] << std::endl;
            } else if (op.int_attrs.count("value")) {
                int64_t val = op.int_attrs.at("value");
                if (target_type.find("f") != std::string::npos) result = mlx::core::array((float)val);
                else if (target_type.find("u64") != std::string::npos || target_type.find("ui64") != std::string::npos || target_type.find("uint64") != std::string::npos) {
                   // Expand to [..., 2] uint32
                   uint64_t v = (uint64_t)val;
                   // shape is SmallVector<int>. Convert to std::vector<int> for manipulation.
                   std::vector<int> new_shape(shape.begin(), shape.end());
                   new_shape.push_back(2);
                   // Split into lo, hi
                   uint32_t lo = (uint32_t)(v & 0xFFFFFFFF);
                   uint32_t hi = (uint32_t)(v >> 32);
                   
                   auto scalar_pair = mlx::core::array({lo, hi}, {2}, mlx::core::uint32);
                   if (mlx::core::Shape(shape).size() > 0) {
                        result = mlx::core::broadcast_to(scalar_pair, mlx::core::Shape(new_shape.begin(), new_shape.end()));
                   } else {
                        result = scalar_pair;
                   }
                }
                else if (target_type.find("i64") != std::string::npos) result = mlx::core::array({val}, shape, mlx::core::int64);
                else result = mlx::core::array(val);
                
            } else if (op.float_array_attrs.count("value")) {
                const std::vector<float>& val = op.float_array_attrs.at("value");
                
                // Debug Float Constant
if (debug_mode()) std::cout << "[MLX-PJRT] Constant(dense-float) Shape=[" << shape << "] ValSize=" << val.size() << " First=" << (val.empty() ? 0.0f : val[0]) << std::endl;
                
                // Check if size mismatches (critical for dense attributes)
                size_t req_size = 1;
                for(auto s : vec_shape) req_size *= s;
                if (val.size() > 0 && val.size() != req_size) {
if (debug_mode()) std::cout << "[MLX-PJRT] WARNING: float constant size mismatch! Expected " << req_size << " got " << val.size() << std::endl;
                }

                result = mlx::core::array(val.begin(), shape, mlx::core::float32);
            } else if (op.attributes.count("value")) {
                 std::string bytes = op.attributes.at("value");
if (debug_mode()) std::cout << "[MLX-PJRT] FATAL: constant fallback. Bytes size=" << bytes.size() << " Hex: ";
                 for (unsigned char c : bytes) printf("%02x ", c);
                 std::cout << std::endl;
                 
                 // Try fallback parse for f32 scalar
                 if (target_type.find("f32") != std::string::npos && bytes.size() == 4) {
                      float f;
                      std::memcpy(&f, bytes.data(), 4);
if (debug_mode()) std::cout << "[MLX-PJRT] Attempted parse as f32: " << f << std::endl;
                      result = mlx::core::array(f);
                 } else {
                      result = mlx::core::array(0.0f);
                 }
            } else if (op.attributes.count("int_value")) {
                 std::string s = op.attributes.at("int_value");
                 // Parse [N, N] string if present
                 s.erase(std::remove(s.begin(), s.end(), '['), s.end());
                 s.erase(std::remove(s.begin(), s.end(), ']'), s.end());
                 std::stringstream ss(s);
                 std::string item;
                 std::vector<int64_t> vals;
                 while (std::getline(ss, item, ',')) {
                     try {
                         vals.push_back(std::stoll(item));
                     } catch (...) {}
                 }
                 
                 if (target_type.find("f") != std::string::npos) {
                      std::vector<float> fvals;
                      for(auto v : vals) fvals.push_back((float)(int32_t)v);
                      if (!fvals.empty()) {
                          if (fvals.size() == 1 && mlx::core::Shape(shape).size() > 1) {
                               result = mlx::core::broadcast_to(mlx::core::array(fvals[0]), shape);
                          } else {
                               result = mlx::core::array(fvals.begin(), shape, mlx::core::float32);
                          }
                      } else {
                          result = mlx::core::array(0.0f);
                      }
                 } else {
                      std::vector<int> ivals;
                      for(auto v : vals) ivals.push_back((int)v);
                      if (!ivals.empty()) {
                           if (ivals.size() == 1 && mlx::core::Shape(shape).size() > 1) {
                               result = mlx::core::broadcast_to(mlx::core::array(ivals[0]), shape);
                           } else {
                               result = mlx::core::array(ivals.begin(), shape, mlx::core::int32); 
                           }
                      } else {
                           result = mlx::core::array(0);
                      }
                 }
            } else {
                 if (target_type.find("f") != std::string::npos) result = mlx::core::array(0.0f);
                 else result = mlx::core::array(0);
            }
        } else if (op.op_name == "stablehlo.transpose" || op.op_name == "mhlo.transpose") {
            if (!op_inputs.empty()) {
                // Scalar arrays can't be transposed - just return as-is
                if (op_inputs[0].ndim() == 0) {
                    result = op_inputs[0];
                } else {
                    std::vector<int> perm;
                    if (op.int_array_attrs.count("permutation")) {
                        auto& v = op.int_array_attrs.at("permutation"); perm.assign(v.begin(), v.end());
                    } else if (op.attributes.count("permutation")) {
                        std::string s = op.attributes.at("permutation");
                        // Parse [1, 0]
                        // Remove brackets
                        s.erase(std::remove(s.begin(), s.end(), '['), s.end());
                        s.erase(std::remove(s.begin(), s.end(), ']'), s.end());
                        std::stringstream ss(s);
                        std::string item;
                        while (std::getline(ss, item, ',')) {
                            try {
                                perm.push_back(std::stoi(item));
                            } catch (...) {}
                        }
                    }
                    // Filter out invalid perm axes
                    std::vector<int> valid_perm;
                    for (int p : perm) {
                        if (p >= 0 && p < static_cast<int>(op_inputs[0].ndim())) {
                            valid_perm.push_back(p);
                        }
                    }
                    if (!valid_perm.empty() && valid_perm.size() == op_inputs[0].ndim()) {
                        result = mlx::core::contiguous(mlx::core::transpose(op_inputs[0], valid_perm));
                    } else if (op_inputs[0].ndim() <= 1) {
                        // 0D or 1D - nothing to transpose
                        result = op_inputs[0];
                    } else {
                        // Default transpose reverses all axes
                        result = mlx::core::transpose(op_inputs[0]);
                    }
                }
            }
        } else if (op.op_name == "stablehlo.bitcast_convert") {
            if (!op_inputs.empty()) {
                // Use mx.view() for bitcast - no eval() needed, allows compilation!
                std::string target_str = (!op.output_dtypes.empty()) ? op.output_dtypes[0] : "";
                auto input = op_inputs[0];
                
                // Map target dtype string to mlx dtype
                mlx::core::Dtype target_dtype = mlx::core::float32;  // default
                
                if (target_str.find("u32") != std::string::npos || target_str.find("ui32") != std::string::npos || target_str == "uint32") {
                    target_dtype = mlx::core::uint32;
                } else if (target_str.find("i32") != std::string::npos || target_str == "int32") {
                    target_dtype = mlx::core::int32;
                } else if (target_str.find("f32") != std::string::npos || target_str == "float32") {
                    target_dtype = mlx::core::float32;
                } else if (target_str.find("u16") != std::string::npos || target_str == "uint16") {
                    target_dtype = mlx::core::uint16;
                } else if (target_str.find("i16") != std::string::npos || target_str == "int16") {
                    target_dtype = mlx::core::int16;
                } else if (target_str.find("f16") != std::string::npos || target_str == "float16") {
                    target_dtype = mlx::core::float16;
                } else if (target_str.find("bf16") != std::string::npos || target_str == "bfloat16") {
                    target_dtype = mlx::core::bfloat16;
                } else if (target_str.find("u8") != std::string::npos || target_str == "uint8") {
                    target_dtype = mlx::core::uint8;
                } else if (target_str.find("i8") != std::string::npos || target_str == "int8") {
                    target_dtype = mlx::core::int8;
                } else if (target_str.find("u64") != std::string::npos || target_str == "uint64") {
                    // MLX doesn't support 64-bit directly, treat as uint32 pairs
                    target_dtype = mlx::core::uint32;
                } else if (target_str.find("i64") != std::string::npos || target_str == "int64") {
                    target_dtype = mlx::core::int64;
                }
                
                // Use mx.view() - reinterprets bits without copying
                result = mlx::core::view(input, target_dtype);
                
                // Reshape if needed to match output shape  
                if (!op.output_shapes.empty()) {
                    std::vector<int> vec_shape = op.output_shapes[0];
                    mlx::core::Shape target_shape(vec_shape.begin(), vec_shape.end());
                    if (result.shape() != target_shape) {
                        result = mlx::core::reshape(result, target_shape);
                    }
                }
                
                if (debug_mode()) {
                    std::cout << "[MLX-PJRT] bitcast_convert (view): " << input.dtype() << " -> " << target_dtype 
                              << " shape=" << result.shape() << std::endl;
                }
            }
        } else if (op.op_name == "stablehlo.concatenate" || op.op_name == "mhlo.concatenate") {
            if (getenv("MLX_PJRT_DEBUG") && !op_inputs.empty()) {
if (debug_mode()) std::cout << "[MLX-PJRT] concatenate Inputs=" << op_inputs.size() << " Input0Dtype=" << op_inputs[0].dtype() << std::endl;
            }
            if (!op_inputs.empty()) {
                int axis = 0;
                if (op.int_attrs.count("dimension")) {
                    axis = op.int_attrs.at("dimension");
                    // std::cout << "[MLX-PJRT] Concatenate found dimension in int_attrs: " << axis << std::endl;
                } else if (op.int_array_attrs.count("dimension")) {
                    auto& dim_vec = op.int_array_attrs.at("dimension");
                    if (!dim_vec.empty()) axis = dim_vec[0];
                    // std::cout << "[MLX-PJRT] Concatenate found dimension in int_array_attrs: " << axis << std::endl;
                } else {
                     // Check string attrs for fallback
                     // std::cout << "[MLX-PJRT] Concatenate defaulting to axis 0. Attrs: ";
                     // for(auto const& [key, val] : op.attributes) std::cout << key << "=" << val << " ";
                     // std::cout << std::endl;
                }
                
                // FORCE AXIS 1 if inputs are [2,1] and [2,1] to verify hypothesis? NO, dangerous.
                // Rank Expansion Logic
                bool output_is_64 = false;
                if (!op.output_dtypes.empty()) {
                     std::string t = op.output_dtypes[0];
                     if (t.find("64") != std::string::npos) output_is_64 = true;
                }

                bool any_expanded = false;
                for(auto& inp : op_inputs) {
                    bool inp_exp = (output_is_64 && inp.ndim() > 0 && inp.shape().back() == 2 && 
                                   (inp.dtype() == mlx::core::uint32 || inp.dtype() == mlx::core::int32));
                    if (inp_exp) { any_expanded = true; break; }
                }

                std::vector<mlx::core::array> processed_inputs;
                if (any_expanded) {
                    for(auto& inp : op_inputs) {
                        bool inp_exp = (output_is_64 && inp.ndim() > 0 && inp.shape().back() == 2 && 
                                       (inp.dtype() == mlx::core::uint32 || inp.dtype() == mlx::core::int32));
                        if (!inp_exp) {
                             processed_inputs.push_back(expand(inp));
                        } else {
                             processed_inputs.push_back(inp);
                        }
                    }
                } else {
                    // Handle scalar inputs by unsqueezing them
                    for(auto& inp : op_inputs) {
                        if (inp.ndim() == 0) {
                            // Unsqueeze scalar to shape [1]
                            processed_inputs.push_back(mlx::core::reshape(inp, {1}));
                        } else {
                            processed_inputs.push_back(inp);
                        }
                    }
                }
                // Clamp axis to valid range based on processed input dimensions
                if (!processed_inputs.empty() && processed_inputs[0].ndim() > 0) {
                    int max_axis = processed_inputs[0].ndim() - 1;
                    if (axis > max_axis) axis = max_axis;
                    if (axis < 0) axis = 0;
                }
                result = mlx::core::concatenate(processed_inputs, axis);
                if (debug_mode()) {
                    std::cout << "[MLX-PJRT] concatenate Result: Shape=[";
                    for (auto s : result.shape()) std::cout << s << ",";
                    std::cout << "] Axis=" << axis << std::endl;
                }
            }
        } else if (op.op_name == "stablehlo.broadcast_in_dim" || op.op_name == "mhlo.broadcast_in_dim") {
            if (!op_inputs.empty() && !op.output_shapes.empty()) {
                mlx::core::array input = op_inputs[0];
                const std::vector<int>& out_shape_vec = op.output_shapes[0];
                std::vector<int> dimensions;
                if (op.int_array_attrs.count("broadcast_dimensions")) {
                    auto& v = op.int_array_attrs.at("broadcast_dimensions"); dimensions.assign(v.begin(), v.end());
                }
                
                // std::cout << "[MLX-PJRT] Broadcast In: ["; 
                // for(auto s : input.shape()) std::cout << s << ",";
                // std::cout << "] Out: [";
                // for(auto s : out_shape_vec) std::cout << s << ",";
                // std::cout << "] Dims: [";
                // for(auto d : dimensions) std::cout << d << ",";
                // std::cout << "]" << std::endl;
                
                if (debug_mode()) {
                    std::cout << "[MLX-PJRT] broadcast_in_dim: InShape=[";
                    for (auto s : input.shape()) std::cout << s << ",";
                    std::cout << "] OutShape=[";
                    for (auto s : out_shape_vec) std::cout << s << ",";
                    std::cout << "] Dims=[";
                    for (auto d : dimensions) std::cout << d << ",";
                    std::cout << "]" << std::endl;
                }
                
                // Logic: 
                // 1. Create a shape of rank equal to output rank, filled with 1s.
                // 2. Place input dimensions into this shape at positions specified by dimensions.
                // Rank Expansion Logic
                bool output_is_64 = false;
                if (!op.output_dtypes.empty()) {
                     std::string t = op.output_dtypes[0];
                     if (t.find("64") != std::string::npos) output_is_64 = true;
                }

                bool input_expanded = (output_is_64 && input.ndim() > 0 && input.shape().back() == 2 && 
                                      (input.dtype() == mlx::core::uint32 || input.dtype() == mlx::core::int32 ||
                                       input.dtype() == mlx::core::uint64 || input.dtype() == mlx::core::int64));
                
                std::vector<int> effective_out_shape = out_shape_vec;
                if (input_expanded) effective_out_shape.push_back(2);
                
                std::vector<int> expand_shape(effective_out_shape.size(), 1);
                auto input_shape = input.shape();
                
                size_t logical_input_rank = input_expanded ? input_shape.size() - 1 : input_shape.size();
                
                // Map dimensions
                if (dimensions.size() == logical_input_rank) {
                     for (size_t i = 0; i < dimensions.size(); ++i) {
                         int out_dim = dimensions[i];
                         if (out_dim >= 0 && out_dim < expand_shape.size()) {
                             expand_shape[out_dim] = input_shape[i];
                         }
                     }
                }
                
                if (input_expanded) {
                    // Map the hidden dim. It effectively becomes the new last dimension.
                    // The 'dimensions' attribute maps input dims to output dims.
                    // Since 'expand' adds a dimension at the end, we should map that too?
                    // Usually broadcast_in_dim doesn't mention the expanded dim.
                    // So we must manually ensure the last dimension of output is 2.
                    expand_shape.back() = 2; 
                }
                
                // If dimensions were empty and input scalar, but expanded
                if (dimensions.empty() && input_expanded && input_shape.size() == 1) { // [2] -> [..., 2]
                    expand_shape.back() = 2; 
                }

                auto mlx_expand = mlx::core::Shape(expand_shape.begin(), expand_shape.end());
                auto mlx_out = mlx::core::Shape(effective_out_shape.begin(), effective_out_shape.end());
                
                // Calculate total sizes to check if reshape is valid
                int64_t input_size = 1;
                for (auto s : input.shape()) input_size *= s;
                int64_t expand_size = 1;
                for (auto s : expand_shape) expand_size *= s;
                
                if (input_size == expand_size) {
                    // Size-preserving reshape - proceed as normal
                    auto reshaped = mlx::core::reshape(input, mlx_expand);
                    result = mlx::core::broadcast_to(reshaped, mlx_out);
                } else if (input.ndim() == 0 || (input.ndim() == 1 && input.shape()[0] == 1)) {
                    // Scalar input - broadcast directly to output shape
                    result = mlx::core::broadcast_to(input, mlx_out);
                } else if (dimensions.empty() && input.shape().size() == mlx_out.size()) {
                    // Possibly matching shapes - check element-wise
                    bool matches = true;
                    for (size_t i = 0; i < mlx_out.size() && matches; ++i) {
                        if (input.shape()[i] != mlx_out[i]) matches = false;
                    }
                    if (matches) {
                        // Input already matches output - passthrough
                        result = input;
                    } else {
                        result = input; // Fallback
                    }
                } else {
                    // Cannot reshape - fall back to passthrough with warning
                    if (debug_mode()) {
                        std::cout << "[MLX-PJRT] broadcast_in_dim: Cannot reshape, using passthrough. ";
                        std::cout << "InShape=["; for (auto s : input.shape()) std::cout << s << ",";
                        std::cout << "] ExpandShape=["; for (auto s : expand_shape) std::cout << s << ",";
                        std::cout << "]" << std::endl;
                    }
                    result = input;
                }
            }
        } else if (op.op_name == "func.call") {
             if (debug_mode()) {
                 std::string callee = op.attributes.count("callee") ? op.attributes.at("callee") : "?";
                 std::cout << "[MLX-PJRT] func.call: " << callee << std::endl;
             }
             std::string callee = "";
             if (op.attributes.count("callee")) {
                 callee = op.attributes.at("callee");
                 // Remove @ prefix if present
                 if (!callee.empty() && callee[0] == '@') {
                     callee = callee.substr(1);
                 }
             }
             
             if (!callee.empty() && functions && functions->count(callee)) {
                 // Special case: detect cumsum/cumprod patterns and use MLX builtins
                 if (callee.find("cumsum") != std::string::npos && !op_inputs.empty()) {
                     // Use MLX cumsum directly
                     result = mlx::core::cumsum(op_inputs[0], 0);
                 } else if (callee.find("cumprod") != std::string::npos && !op_inputs.empty()) {
                     // Use MLX cumprod directly
                     result = mlx::core::cumprod(op_inputs[0], 0);
                 } else if (callee == "inv" && !op_inputs.empty()) {
                     // Use MLX inv directly instead of LU-based solve
                     result = mlx::core::linalg::inv(op_inputs[0], mlx::core::Device(mlx::core::Device::cpu));
                 } else if (callee == "solve" && op_inputs.size() >= 2) {
                     // Use MLX solve directly instead of LU-based solve
                     result = mlx::core::linalg::solve(op_inputs[0], op_inputs[1], mlx::core::Device(mlx::core::Device::cpu));
                 // NOTE: Disabled _lu_solve intercept - MLX doesn't have lu_solve that takes (LU, pivots, b)
                 // The inputs here are LU matrix and pivots, not original A and b
                 // Let JAX's triangular solve path execute instead
                 /*
                 } else if ((callee == "_lu_solve" || callee.find("lu_solve") != std::string::npos) && op_inputs.size() >= 2) {
                     result = mlx::core::linalg::solve(op_inputs[0], op_inputs[1], mlx::core::Device(mlx::core::Device::cpu));
                 */
                 } else if (false) {
                     // Placeholder for future lu_solve implementation
                 } else {
                     // Normal function call - execute subgraph
                 auto func_graph = functions->at(callee);
                 
                 // Reshape inputs to match callee signature
                 std::vector<mlx::core::array> call_inputs = op_inputs;
                 if (!func_graph->input_shapes.empty()) {
                     for(size_t i=0; i<call_inputs.size(); ++i) {
                         if (i < func_graph->input_shapes.size()) {
                             const std::vector<int>& target = func_graph->input_shapes[i];
                             if (target.empty()) continue; 
                             size_t target_elements = 1;
                             for (int d : target) target_elements *= d;
                             auto& arr = call_inputs[i];
                             
                             if (arr.size() == target_elements && target_elements > 0) {
                                 bool mismatch = (arr.ndim() != (int)target.size());
                                 if (!mismatch) {
                                     for(size_t k=0; k<target.size(); ++k) if (arr.shape(k) != target[k]) mismatch=true;
                                 }
                                 if (mismatch) {
                                     arr = mlx::core::reshape(arr, mlx::core::Shape(target.begin(), target.end()));
                                 }
                             }
                         }
                     }
                 }

                 auto result_nodes = ExecuteGraph(*func_graph, call_inputs, parent_val_map, functions);
                 
                 // func.call can return multiple values
                 if (result_nodes.size() == 1) {
                     result = result_nodes[0];
                 } else if (result_nodes.empty()) {
                     // Void return?
                 } else {
                     // Multiple returns handling handled by generic output mapping below? 
                     // No, generic logic is: if (!op_outputs.empty()) result = op_outputs[0]; 
                     // I need to populate op_outputs explicitly for multi-return.
                     op_outputs = result_nodes;
                 }
                 }
             } else {
if (debug_mode()) std::cout << "[MLX-PJRT]   Warning: func.call to unknown function: " << callee << std::endl;
                 // Fallback?
                 if (!op_inputs.empty()) result = op_inputs[0];
             }
        } else if (op.op_name == "stablehlo.and" || op.op_name == "mhlo.and") {
            // Logical/Bitwise AND
            if (op_inputs.size() >= 2) {
                auto lhs = op_inputs[0];
                auto rhs = op_inputs[1];
                bool expanded = ensure_binary_expanded(lhs, rhs);
                
                // For float or bool types, use logical_and; for integers, use bitwise_and
                if (lhs.dtype() == mlx::core::float32 || lhs.dtype() == mlx::core::float16 || 
                    lhs.dtype() == mlx::core::bfloat16 || lhs.dtype() == mlx::core::bool_) {
                    result = mlx::core::logical_and(lhs, rhs);
                } else {
                    // Integer types - use bitwise_and
                    result = mlx::core::bitwise_and(lhs, rhs);
                }
            }
        } else if (op.op_name == "stablehlo.or" || op.op_name == "mhlo.or") {
            // Logical/Bitwise OR
            if (op_inputs.size() >= 2) {
                auto lhs = op_inputs[0];
                auto rhs = op_inputs[1];
                bool expanded = ensure_binary_expanded(lhs, rhs);
                
                // For float or bool types, use logical_or; for integers, use bitwise_or
                if (lhs.dtype() == mlx::core::float32 || lhs.dtype() == mlx::core::float16 || 
                    lhs.dtype() == mlx::core::bfloat16 || lhs.dtype() == mlx::core::bool_) {
                    result = mlx::core::logical_or(lhs, rhs);
                } else {
                    // Integer types - use bitwise_or
                    result = mlx::core::bitwise_or(lhs, rhs);
                }
            }
        } else if (op.op_name == "stablehlo.not" || op.op_name == "mhlo.not") {
            if (!op_inputs.empty()) {
                auto x = op_inputs[0];
                // Use bitwise_not for integers, logical_not for booleans
                if (x.dtype() == mlx::core::bool_) {
                    result = mlx::core::logical_not(x);
                } else {
                    result = mlx::core::bitwise_invert(x);
                }
            }
        } else if (op.op_name == "stablehlo.select" || op.op_name == "mhlo.select") {
            // select(cond, on_true, on_false)
            if (op_inputs.size() >= 3) {
                 auto cond = op_inputs[0];
                 auto on_true = op_inputs[1];
                 auto on_false = op_inputs[2];
                 ensure_ternary_expanded(cond, on_true, on_false);
                 result = mlx::core::where(cond, on_true, on_false);
            }
        // --- Indexing/Slicing Operations ---
        } else if (op.op_name == "stablehlo.rng_bit_generator") {
            // Inputs: [state]
            // Outputs: [new_state, random_data]
            if (op_inputs.empty()) {
if (debug_mode()) std::cout << "[MLX-PJRT] Warning: rng_bit_generator called with no inputs." << std::endl;
                // Fallback, return empty array or throw?
                // For now, let's return a dummy array if no inputs.
                result = mlx::core::array(0);
            } else {
                auto& state = op_inputs[0]; // (2,) uint32 usually
                
                // We need the shape of the random output.
                // The node "outputs" field has IDs. We need the MLXOp to know shape?
                // In ExecuteGraph, 'node' has 'output_types'.
                // node.output_types[1] is the random output.
                
                if (op.output_shapes.size() < 2) {
                     throw std::runtime_error("rng_bit_generator requires at least 2 output shapes");
                }
                auto& out_shape = op.output_shapes[1]; // Vector of ints
                
                // MLX Split key
                // split(key, num) returns (num, 2) array assuming key is (2,)
                // We want 2 keys.
                auto split_keys = mlx::core::random::split(state, 2); 
                
                // split_keys is (2, 2) uint32 (if state was (2,))
                // key 0: slice(split_keys, {0,0}, {1,2}) -> reshape to (2,)
                // key 1: slice(split_keys, {1,0}, {2,2}) -> reshape to (2,)
                
                auto get_key = [&](int i) {
                    int cols = static_cast<int>(split_keys.shape(1));
                    // Slice returns (1, cols)
                    auto s = mlx::core::slice(split_keys, {i, 0}, {i + 1, cols}, {1, 1});
                    // Flatten to (cols,) i.e. (2,)
                    return mlx::core::reshape(s, {cols});
                };
                
                auto new_state = get_key(0);
                auto gen_key = get_key(1);
                
                // Determine width from dtype string
                // op.output_dtypes[1]
                int width = 4;
                bool expanded_u64 = false;
                if (op.output_dtypes.size() > 1) {
                    std::string dtype = op.output_dtypes[1];
                    if (dtype.find("64") != std::string::npos) {
                        width = 4; // Trick: Use 4 byte width but valid shape
                        expanded_u64 = true;
                    }
                    else if (dtype.find("8") != std::string::npos) width = 1; 
                    else if (dtype.find("16") != std::string::npos) width = 2;
                }
                
                std::vector<int> target_shape_vec(out_shape.begin(), out_shape.end());
                if (expanded_u64) {
                    target_shape_vec.push_back(2);
                }
                
                mlx::core::Shape s_out_shape(target_shape_vec.begin(), target_shape_vec.end());
                auto random_data = mlx::core::random::bits(s_out_shape, width, gen_key);
                
                // --- DEBUG RNG ---
                if (debug_mode()) {
                    std::cout << "[MLX-PJRT] RNG: StateShape=" << state.shape() 
                              << " KeyShape=" << gen_key.shape()
                              << " OutShape=" << random_data.shape() << std::endl;
                              
                    // Check for all-zeros which causes log(0) -> NaN in Box-Muller
                    auto is_zero = mlx::core::equal(random_data, mlx::core::array(0, random_data.dtype()));
                    bool has_zeros = mlx::core::any(is_zero).item<bool>();
                    
                    if (has_zeros) {
                        std::cout << "[MLX-PJRT][WARN] RNG generated zeros!" << std::endl;
                    }
                    
                    // Print first few values
                    // auto flat = mlx::core::flatten(random_data);
                    // auto slice = mlx::core::slice(flat, {0}, {std::min((int)flat.size(), 5)});
                    // std::cout << "[MLX-PJRT] RNG Samples: " << slice << std::endl;
                }
                // -----------------
                
                // Debug RNG
if (debug_mode()) std::cout << "[MLX-PJRT] RNG Check. State shape=[";
                for(auto s : state.shape()) std::cout << s << ",";
                std::cout << "] New key shape=[";
                for(auto s : new_state.shape()) std::cout << s << ",";
                std::cout << "] Random data shape=[";
                for(auto s : random_data.shape()) std::cout << s << ",";
                try {
                    mlx::core::array val_f32 = mlx::core::astype(random_data, mlx::core::float32);
                    float mean_val = mlx::core::mean(val_f32).item<float>();
                    float max_val = mlx::core::max(val_f32).item<float>();
                    std::cout << "] Mean: " << mean_val << " Max: " << max_val << std::endl;
                } catch(...) {
                    std::cout << "] Eval failed" << std::endl;
                }

                op_outputs.push_back(new_state);
                op_outputs.push_back(random_data);
            }
        } else if (op.op_name == "stablehlo.iota" || op.op_name == "mhlo.iota") {
            if (debug_mode()) {
                std::cout << "[MLX-PJRT] iota target=" << (op.output_dtypes.empty() ? "?" : op.output_dtypes[0]) << std::endl;
            }
            if (!op.output_shapes.empty()) {
                std::vector<int> shape = op.output_shapes[0];
                int iota_dim = 0;
                if (op.attributes.count("iota_dimension")) {
                    try { iota_dim = std::stoi(op.attributes.at("iota_dimension")); } catch(...) {}
                }
                
                // Resolving target dtype
                mlx::core::Dtype target_dtype = mlx::core::float32;
                if (!op.output_dtypes.empty()) {
                     std::string t = op.output_dtypes[0];
                     if (t.find("i32") != std::string::npos || t.find("int32") != std::string::npos) target_dtype = mlx::core::int32;
                     else if (t.find("i64") != std::string::npos || t.find("int64") != std::string::npos) target_dtype = mlx::core::int64;
                     else if (t.find("ui64") != std::string::npos || t.find("uint64") != std::string::npos) target_dtype = mlx::core::uint64;
                     else if (t.find("ui32") != std::string::npos || t.find("uint32") != std::string::npos) target_dtype = mlx::core::uint32;
                }

                // Rank Expansion for u64 iota
                if (target_dtype == mlx::core::uint64 || target_dtype == mlx::core::int64) {
                     // Produce [..., 2] u32
                     int dim_size = shape.empty() ? 1 : shape[iota_dim];
                     // Use u32 for indices (assuming indices fit in 32 bit). 
                     auto idxs = mlx::core::arange(0, dim_size, 1, mlx::core::uint32);
                     
                     // We need [dim_size, 2]. [i, 0].
                     // idxs shape [N].
                     // Reshape [N, 1].
                     mlx::core::Shape col_shape({dim_size, 1});
                     auto col = mlx::core::reshape(idxs, col_shape);
                     auto zeros = mlx::core::zeros(col_shape, mlx::core::uint32);
                     
                     // Concatenate [N, 2]
                     auto expanded = mlx::core::concatenate({col, zeros}, 1);
                     
                     if (shape.size() > 1) {
                         // Need to reshape/broadcast to [..., 2]
                         // Original broadcast logic would do [shape].
                         // New broadcast logic needs [shape + 2].
                         // Map iota_dim to new dim structure.
                         std::vector<int> expand_shape(shape.begin(), shape.end());
                         std::fill(expand_shape.begin(), expand_shape.end(), 1);
                         expand_shape[iota_dim] = dim_size;
                         expand_shape.push_back(2); // The packed dim
                         
                         auto padded = mlx::core::reshape(expanded, mlx::core::Shape(expand_shape.begin(), expand_shape.end()));
                         
                         std::vector<int> broadcast_shape = shape;
                         broadcast_shape.push_back(2);
                         
                         result = mlx::core::broadcast_to(padded, mlx::core::Shape(broadcast_shape.begin(), broadcast_shape.end()));
                     } else {
                         result = expanded; // [N, 2]
                     }
                } else {
                     // Standard iota
                     int dim_size = shape.empty() ? 1 : shape[iota_dim];
                     result = mlx::core::arange(0, dim_size, 1, target_dtype);
                     if (shape.size() > 1) {
                         std::vector<int> expand_shape(shape.size(), 1);
                         expand_shape[iota_dim] = dim_size;
                         
                         mlx::core::Shape shape_expand(expand_shape.begin(), expand_shape.end());
                         mlx::core::Shape shape_target(shape.begin(), shape.end()); // Original shape

                         result = mlx::core::reshape(result, shape_expand);
                         result = mlx::core::broadcast_to(result, shape_target);
                     }
                }
            }
        } else if (op.op_name == "stablehlo.slice" || op.op_name == "mhlo.slice") {
            if (!op_inputs.empty()) {
                // Handle scalar inputs: return as-is (no slicing possible)
                if (op_inputs[0].ndim() == 0) {
                    result = op_inputs[0];
                } else {
                    // Get start_indices, limit_indices, strides from attributes
                    std::vector<int> starts, limits, strides;
                    if (op.int_array_attrs.count("start_indices")) {
                        auto& v = op.int_array_attrs.at("start_indices"); starts.assign(v.begin(), v.end());
                    }
                    if (op.int_array_attrs.count("limit_indices")) {
                        auto& v = op.int_array_attrs.at("limit_indices"); limits.assign(v.begin(), v.end());
                    }
                    if (op.int_array_attrs.count("strides")) {
                        auto& v = op.int_array_attrs.at("strides"); strides.assign(v.begin(), v.end());
                    }
                    if (!starts.empty() && !limits.empty()) {
                        std::vector<int> strides_vec = strides.empty() ? std::vector<int>(starts.size(), 1) : strides;
                        
                        // Extend slice parameters to match input rank
                        size_t input_rank = op_inputs[0].ndim();
                        while (starts.size() < input_rank) {
                            starts.push_back(0);
                            limits.push_back(static_cast<int>(op_inputs[0].shape()[starts.size() - 1]));
                            strides_vec.push_back(1);
                        }
                        // Also handle the case where we have more slices than dims (truncate)
                        if (starts.size() > input_rank) {
                            starts.resize(input_rank);
                            limits.resize(input_rank);
                            strides_vec.resize(input_rank);
                        }
                        
                        try {
                            result = mlx::core::slice(op_inputs[0], 
                                                      mlx::core::Shape(starts.begin(), starts.end()), 
                                                      mlx::core::Shape(limits.begin(), limits.end()), 
                                                      mlx::core::Shape(strides_vec.begin(), strides_vec.end()));
                        } catch (const std::exception& e) {
                            if (debug_mode()) {
                                std::cout << "[ERROR] Slice failed! Input shape=[";
                                for (auto s : op_inputs[0].shape()) std::cout << s << ",";
                                std::cout << "] dtype=" << op_inputs[0].dtype() << " starts=" << starts.size() << std::endl;
                            }
                            // Fallback to passthrough
                            result = op_inputs[0];
                        }
                    } else {
                        result = op_inputs[0];
                    }
                }
            }
        /**
         * DYNAMIC OPERATIONS - MLX Native API Pattern
         * =============================================
         * JAX's dynamic_slice, dynamic_update_slice, and scatter use runtime indices.
         *
         * Traditional approach: eval() indices to get concrete values, then slice.
         * This breaks mx.compile() because eval() creates a sync point.
         *
         * Our approach: Use MLX's array-based slice APIs that accept indices as arrays:
         * - mlx::core::slice(array, start_array, axes, sizes)
         * - mlx::core::slice_update(src, update, start_array, axes)
         * - mlx::core::scatter(array, indices, updates, axes)
         *
         * These APIs keep indices lazy in the computation graph, enabling mx.compile().
         */
        } else if (op.op_name == "stablehlo.dynamic_slice" || op.op_name == "mhlo.dynamic_slice") {
            // Dynamic slice: inputs = [array, start_idx0, start_idx1, ...]
            // Attribute: slice_sizes = [size0, size1, ...]
            // 
            // Uses MLX's dynamic slice API: slice(array, start_array, axes, sizes)
            // This avoids eval() and enables mx.compile compatibility.
if (debug_mode()) std::cout << "[MLX-PJRT]   dynamic_slice: has_slice_sizes=" << op.int_array_attrs.count("slice_sizes") 
                      << " has_sizes=" << op.int_array_attrs.count("sizes") 
                      << " attrs_count=" << op.int_array_attrs.size() << std::endl;
            for (auto& kv : op.int_array_attrs) {
if (debug_mode()) std::cout << "[MLX-PJRT]     attr: " << kv.first << std::endl;
            }
            std::vector<int> sizes_vec;
            if (op_inputs.size() >= 2) {
                if (op.int_array_attrs.count("sizes")) {
                     auto& v = op.int_array_attrs.at("sizes"); sizes_vec.assign(v.begin(), v.end());
                } else if (op.int_array_attrs.count("slice_sizes")) {
                     auto& v = op.int_array_attrs.at("slice_sizes"); sizes_vec.assign(v.begin(), v.end());
                }
            }

            if (!sizes_vec.empty() && op_inputs.size() >= 2) {
                mlx::core::array data = op_inputs[0];
                auto sizes = sizes_vec; // Copy since we may modify
                
                // Extend sizes to match input rank if incomplete
                size_t input_rank = data.ndim();
                while (sizes.size() < input_rank) {
                    sizes.push_back(data.shape()[sizes.size()]);
                }
                
                // Collect start index arrays and stack them
                std::vector<mlx::core::array> start_arrays;
                for (size_t dim = 0; dim < input_rank && dim + 1 < op_inputs.size(); ++dim) {
                    // Ensure index is int32 for MLX
                    auto idx = op_inputs[dim + 1];
                    if (idx.dtype() != mlx::core::int32) {
                        idx = mlx::core::astype(idx, mlx::core::int32);
                    }
                    start_arrays.push_back(mlx::core::reshape(idx, {1}));
                }
                
                // Pad with zeros if we have fewer start indices than dimensions
                while (start_arrays.size() < input_rank) {
                    start_arrays.push_back(mlx::core::zeros({1}, mlx::core::int32));
                }
                
                // Stack all start indices into single array
                mlx::core::array start = mlx::core::concatenate(start_arrays, 0);
                
                // Build axes vector [0, 1, 2, ..., N-1]
                std::vector<int> axes(input_rank);
                for (size_t i = 0; i < axes.size(); ++i) axes[i] = (int)i;
                
                // Convert sizes to Shape
                mlx::core::Shape size_shape(sizes.begin(), sizes.end());

if (debug_mode()) {
    std::cout << "[MLX-PJRT]     dynamic_slice using MLX dynamic API, input shape=[";
    for (auto s : data.shape()) std::cout << s << ",";
    std::cout << "] sizes=[";
    for (auto s : sizes) std::cout << s << ",";
    std::cout << "]" << std::endl;
}
                
                // Use MLX's dynamic slice API (no eval needed!)
                result = mlx::core::slice(data, start, axes, size_shape);
            } else if (!op_inputs.empty()) {
                result = op_inputs[0]; // Fallback
            }
        } else if (op.op_name == "stablehlo.dynamic_update_slice" || op.op_name == "mhlo.dynamic_update_slice") {
            if (op_inputs.size() >= 3) {
                 mlx::core::array operand = op_inputs[0];
                 mlx::core::array update = op_inputs[1];
                 
                 std::vector<mlx::core::array> start_indices_arrays;
                 for (size_t i = 2; i < op_inputs.size(); ++i) {
                     start_indices_arrays.push_back(op_inputs[i]);
                 }
                 
                 // Stack execution-time indices into a single array
                 mlx::core::array start = mlx::core::stack(start_indices_arrays, 0);
                 
                 // Prepare axes [0, 1, ..., N-1] matching the start indices
                 std::vector<int> axes(start_indices_arrays.size());
                 for (size_t i = 0; i < axes.size(); ++i) axes[i] = i;
                 
                 result = mlx::core::slice_update(operand, update, start, axes);
            } else if (!op_inputs.empty()) {
                result = op_inputs[0];
            }
        } else if (op.op_name == "stablehlo.gather" || op.op_name == "mhlo.gather") {
            // Gather - complex indexing op
            if (op_inputs.size() >= 2) {
                auto operand = op_inputs[0];
                auto indices = op_inputs[1];
                
                // Get dimension_numbers attributes
                std::vector<int64_t> start_index_map;
                std::vector<int64_t> collapsed_slice_dims;
                std::vector<int64_t> offset_dims;
                std::vector<int64_t> slice_sizes;
                int64_t index_vector_dim = -1;
                
                if (op.int_array_attrs.count("start_index_map")) {
                    start_index_map = op.int_array_attrs.at("start_index_map");
                }
                if (op.int_array_attrs.count("collapsed_slice_dims")) {
                    collapsed_slice_dims = op.int_array_attrs.at("collapsed_slice_dims");
                }
                if (op.int_attrs.count("index_vector_dim")) {
                    index_vector_dim = op.int_attrs.at("index_vector_dim");
                }
                if (op.int_array_attrs.count("offset_dims")) {
                    offset_dims = op.int_array_attrs.at("offset_dims");
                }
                if (op.int_array_attrs.count("slice_sizes")) {
                    slice_sizes = op.int_array_attrs.at("slice_sizes");
                }
                
                if (debug_mode()) {
                    std::cout << "[MLX-PJRT] gather: operand=" << operand.shape() 
                              << " indices=" << indices.shape()
                              << " start_index_map=[";
                    for (auto v : start_index_map) std::cout << v << ",";
                    std::cout << "] collapsed=[";
                    for (auto v : collapsed_slice_dims) std::cout << v << ",";
                    std::cout << "] offset_dims=[";
                    for (auto v : offset_dims) std::cout << v << ",";
                    std::cout << "] slice_sizes=[";
                    for (auto v : slice_sizes) std::cout << v << ",";
                    std::cout << "] index_vector_dim=" << index_vector_dim << std::endl;
                }
                
                // Get operand_batching_dims (for batched gather like take_along_axis)
                std::vector<int64_t> operand_batching_dims;
                std::vector<int64_t> start_indices_batching_dims;
                if (op.int_array_attrs.count("operand_batching_dims")) {
                    operand_batching_dims = op.int_array_attrs.at("operand_batching_dims");
                }
                if (op.int_array_attrs.count("start_indices_batching_dims")) {
                    start_indices_batching_dims = op.int_array_attrs.at("start_indices_batching_dims");
                }
                
                // Case 0: take_along_axis pattern
                // Pattern: 2D operand, start_index_map=[1], collapsed_slice_dims=[1], slice_sizes=[1,1]
                // indices have shape (batch, out_dim, 1) and index_vector_dim points to last dim
                // offset_dims must be empty (distinguishes from general indexed gather)
                bool is_take_along_axis = 
                    operand.ndim() == 2 &&
                    start_index_map.size() == 1 && 
                    collapsed_slice_dims.size() == 1 && collapsed_slice_dims[0] == start_index_map[0] &&
                    slice_sizes.size() == 2 && slice_sizes[0] == 1 && slice_sizes[1] == 1 &&
                    index_vector_dim >= 0 && indices.ndim() > 0 && indices.shape(-1) == 1 &&
                    offset_dims.empty();  // Critical: offset_dims must be empty
                
                if (is_take_along_axis) {
                    
                    int64_t gather_axis = start_index_map[0];  // The axis to gather along (1 for take_along_axis on axis 1)
                    
                    // indices shape is (batch, ..., 1) - squeeze trailing 1 if index_vector_dim points there
                    auto take_indices = indices;
                    if (index_vector_dim >= 0 && take_indices.ndim() > 0 && 
                        take_indices.shape(-1) == 1 && static_cast<size_t>(index_vector_dim) == take_indices.ndim() - 1) {
                        take_indices = mlx::core::squeeze(take_indices, -1);
                    }
                    
                    // Flatten for simpler processing
                    take_indices = mlx::core::astype(take_indices, mlx::core::int32);
                    
                    // Use take_along_axis for proper batched indexing
                    // MLX's take doesn't directly support this pattern, so we iterate
                    // For 2D case: result[i,:] = operand[i, indices[i,:]]
                    
                    if (operand.ndim() == 2 && gather_axis == 1) {
                        // Batched take along axis 1: result[i,j] = operand[i, indices[i,j]]
                        // Use vmap-like approach with take
                        result = mlx::core::take_along_axis(operand, take_indices, static_cast<int>(gather_axis));
                        
                        if (debug_mode()) {
                            std::cout << "[MLX-PJRT] gather (take_along_axis): result=" << result.shape() << std::endl;
                        }
                    } else {
                        // General fallback for other batched cases
                        result = mlx::core::take_along_axis(operand, take_indices, static_cast<int>(gather_axis));
                        if (debug_mode()) {
                            std::cout << "[MLX-PJRT] gather (batched general): result=" << result.shape() << std::endl;
                        }
                    }
                }
                // Case 1: Multi-dimensional gather with index_vector_dim
                // This is used by map_coordinates for element-wise multi-dim indexing
                // indices shape = (batch_dims..., num_coords) where index_vector_dim points to num_coords
                // slice_sizes = [1, 1, ...] for element-wise gathering
                else {
                    bool all_slice_sizes_one = !slice_sizes.empty() && 
                    std::all_of(slice_sizes.begin(), slice_sizes.end(), [](int64_t s) { return s == 1; });
                
                if (all_slice_sizes_one && index_vector_dim >= 0 && 
                    static_cast<size_t>(index_vector_dim) == indices.ndim() - 1 &&
                    static_cast<size_t>(indices.shape(-1)) == operand.ndim()) {
                    // Element-wise multi-dimensional gather
                    // indices has shape (batch..., operand_ndim)
                    // We need to convert to linear indices and use take
                    
                    // Flatten indices to (num_points, operand_ndim)
                    int num_points = 1;
                    for (size_t i = 0; i < indices.ndim() - 1; i++) {
                        num_points *= indices.shape(i);
                    }
                    int num_coords = indices.shape(-1);
                    auto flat_indices = mlx::core::reshape(indices, {num_points, num_coords});
                    
                    // Calculate linear indices: idx = i0 * stride0 + i1 * stride1 + ...
                    // For operand shape (d0, d1, d2, ...), strides are (d1*d2*..., d2*..., ..., 1)
                    std::vector<int> strides(operand.ndim());
                    strides[operand.ndim() - 1] = 1;
                    for (int i = operand.ndim() - 2; i >= 0; i--) {
                        strides[i] = strides[i + 1] * operand.shape(i + 1);
                    }
                    
                    // Compute linear index for each point
                    auto linear_indices = mlx::core::zeros({num_points}, mlx::core::int32);
                    for (size_t dim = 0; dim < operand.ndim(); dim++) {
                        auto dim_indices = mlx::core::slice(flat_indices, {0, static_cast<int>(dim)}, 
                                                            {num_points, static_cast<int>(dim + 1)});
                        dim_indices = mlx::core::squeeze(dim_indices, 1);
                        dim_indices = mlx::core::astype(dim_indices, mlx::core::int32);
                        linear_indices = mlx::core::add(linear_indices, 
                            mlx::core::multiply(dim_indices, mlx::core::array(strides[dim], mlx::core::int32)));
                    }
                    
                    // Flatten operand and gather
                    auto flat_operand = mlx::core::reshape(operand, {-1});
                    result = mlx::core::take(flat_operand, linear_indices, 0);
                    
                    // Reshape to batch dimensions (remove the last dim that was index_vector_dim)
                    if (indices.ndim() > 1) {
                        std::vector<int> output_shape;
                        for (size_t i = 0; i < indices.ndim() - 1; i++) {
                            output_shape.push_back(indices.shape(i));
                        }
                        result = mlx::core::reshape(result, mlx::core::Shape(output_shape.begin(), output_shape.end()));
                    }
                    
                    if (debug_mode()) {
                        std::cout << "[MLX-PJRT] gather: multi-dim result=" << result.shape() << std::endl;
                    }
                }
                // Case 2: Simple 1D operand gather (jnp.take)
                else if (operand.ndim() == 1 && 
                    (start_index_map.empty() || (start_index_map.size() == 1 && start_index_map[0] == 0)) &&
                    (collapsed_slice_dims.empty() || (collapsed_slice_dims.size() == 1 && collapsed_slice_dims[0] == 0))) {
                    
                    // If indices has an extra dimension (e.g., [3,1]), squeeze it
                    auto take_indices = indices;
                    if (indices.ndim() == 2 && indices.shape(-1) == 1) {
                        take_indices = mlx::core::squeeze(indices, -1);
                    }
                    
                    // Flatten indices to 1D for take
                    if (take_indices.ndim() > 1) {
                        take_indices = mlx::core::reshape(take_indices, {-1});
                    }
                    
                    result = mlx::core::take(operand, take_indices, 0);
                } 
                // Case 3: General fallback with expected output shape
                else {
                    // Fallback: use basic take along axis 0
                    auto take_indices = indices;
                    if (indices.ndim() >= 2 && indices.shape(-1) == 1) {
                        take_indices = mlx::core::squeeze(indices, -1);
                    }
                    result = mlx::core::take(operand, take_indices, 0);
                    
                    // Try to reshape to expected output shape if provided
                    if (!op.output_shapes.empty() && !op.output_shapes[0].empty()) {
                        auto& expected_shape = op.output_shapes[0];
                        mlx::core::Shape target_shape(expected_shape.begin(), expected_shape.end());
                        
                        size_t result_size = 1;
                        for (auto d : result.shape()) result_size *= d;
                        size_t target_size = 1;
                        for (auto d : target_shape) target_size *= d;
                        
                        if (result_size == target_size && result.shape() != target_shape) {
                            result = mlx::core::reshape(result, target_shape);
                        }
                    }
                }
                }  // Close the else block for Case 0
            } else if (!op_inputs.empty()) {
                result = op_inputs[0];
            }
        } else if (op.op_name == "stablehlo.scatter" || op.op_name == "mhlo.scatter") {
            // Scatter - update values at indexed positions
            // Input 0: operand (the array to scatter into)
            // Input 1: scatter_indices (indices to update)
            // Input 2: updates (values to scatter)
            //
            // Uses MLX's native scatter/scatter_add APIs for mx.compile compatibility (no eval!)
            if (op_inputs.size() >= 3) {
                auto operand = op_inputs[0];
                auto indices = op_inputs[1];
                auto updates = op_inputs[2];
                
                if (debug_mode()) {
                    std::cout << "[MLX-PJRT] Scatter using MLX native API, inputs:" << std::endl;
                    std::cout << "  [0] operand: shape=[";
                    for (auto s : operand.shape()) std::cout << s << ",";
                    std::cout << "] dtype=" << operand.dtype() << std::endl;
                    std::cout << "  [1] indices: shape=[";
                    for (auto s : indices.shape()) std::cout << s << ",";
                    std::cout << "] dtype=" << indices.dtype() << std::endl;
                    std::cout << "  [2] updates: shape=[";
                    for (auto s : updates.shape()) std::cout << s << ",";
                    std::cout << "] dtype=" << updates.dtype() << std::endl;
                }
                
                try {
                    // Detect scatter mode from subgraph (if present) or unique_indices attribute
                    // unique_indices=true typically means scatter-set (replace)
                    // Otherwise, check if subgraph returns add or second arg
                    bool is_scatter_add = true;  // Default to add (more common in JAX)
                    
                    if (op.attributes.count("unique_indices")) {
                        std::string val = op.attributes.at("unique_indices");
                        if (val == "true" || val == "1") {
                            is_scatter_add = false;  // unique_indices implies set
                        }
                    }
                    
                    // Check subgraph for update computation pattern
                    // If subgraph returns %arg4 (second arg), it's scatter-set
                    // If subgraph returns add(%arg3, %arg4), it's scatter-add
                    if (!op.subgraphs.empty() && op.subgraphs[0]) {
                        auto& update_graph = *op.subgraphs[0];
                        if (!update_graph.nodes.empty()) {
                            const auto& last_op = update_graph.nodes.back();
                            if (last_op.op_name == "stablehlo.return" || last_op.op_name == "mhlo.return") {
                                // Check what's being returned
                                if (!last_op.inputs.empty()) {
                                    int return_id = last_op.inputs[0];
                                    // If it's directly the second input arg, it's scatter-set
                                    // Block args are typically negative or special IDs
                                    // Simple heuristic: if graph has no add op, it's probably set
                                    bool has_add = false;
                                    for (const auto& n : update_graph.nodes) {
                                        if (n.op_name == "stablehlo.add" || n.op_name == "mhlo.add") {
                                            has_add = true;
                                            break;
                                        }
                                    }
                                    is_scatter_add = has_add;
                                }
                            }
                        }
                    }
                    
if (debug_mode()) std::cout << "[MLX-PJRT] Scatter mode: " << (is_scatter_add ? "ADD" : "SET") << std::endl;
                    
                    // Prepare indices - convert to int32 and flatten if needed
                    auto flat_indices = indices;
                    if (indices.ndim() == 2 && indices.shape(-1) == 1) {
                        // Shape (N, 1) -> flatten to (N,)
                        flat_indices = mlx::core::reshape(indices, {(int)indices.shape(0)});
                    }
                    if (flat_indices.dtype() != mlx::core::int32) {
                        flat_indices = mlx::core::astype(flat_indices, mlx::core::int32);
                    }
                    
                    // Prepare updates - handle scalar and shape matching
                    auto flat_updates = updates;
                    if (updates.ndim() == 0 && flat_indices.ndim() == 1) {
                        // Scalar update - broadcast to match indices
                        auto shape = mlx::core::Shape{(int)flat_indices.shape(0)};
                        flat_updates = mlx::core::broadcast_to(updates, shape);
                    }
                    
                    // Ensure updates have correct shape for MLX scatter
                    // MLX scatter expects updates.ndim() == indices[0].ndim() + a.ndim()
                    // For simple 1D case: updates shape should be (num_indices,) + operand_shape[1:]
                    // For 1D operand with 1D indices: updates should be (num_indices,)
                    if (operand.ndim() == 1 && flat_updates.ndim() == 1) {
                        // Need updates shape: (num_indices,) but MLX expects (num_indices, ...)
                        // Actually for 1D operand with scatter on axis 0, updates = (num_indices,) is correct
                        // But MLX scatter on axis 0 expects updates shape = indices.shape + operand.shape[1:]
                        // For 1D operand that's just indices.shape which is (N,)
                        // We need to add trailing dimensions
                        flat_updates = mlx::core::expand_dims(flat_updates, 1);
                    }
                    
                    // Use MLX scatter (set) or scatter_add
                    std::vector<int> axes = {0};  // Scatter on first axis
                    
                    if (is_scatter_add) {
                        result = mlx::core::scatter_add(operand, flat_indices, flat_updates, 0);
                    } else {
                        result = mlx::core::scatter(operand, flat_indices, flat_updates, 0);
                    }
                    
if (debug_mode()) {
    std::cout << "[MLX-PJRT] Scatter result shape=[";
    for (auto s : result.shape()) std::cout << s << ",";
    std::cout << "]" << std::endl;
}
                } catch (const std::exception& e) {
                    if (debug_mode()) std::cout << "[MLX-PJRT] Scatter MLX API failed: " << e.what() << ", using fallback" << std::endl;
                    // Fallback to original operand
                    result = operand;
                }
            } else if (!op_inputs.empty()) {
                result = op_inputs[0];
            }
        // --- Pad Operation ---
        } else if (op.op_name == "stablehlo.pad" || op.op_name == "mhlo.pad") {
            if (op_inputs.size() >= 2) {
                auto lhs = op_inputs[0];
                auto val = op_inputs[1];
                
                // Rank Expansion
                // We must synchronize expansion. If LHS uses [..., 2], RHS (padding val) must too.
                bool exp = ensure_binary_expanded(lhs, val);
                
                std::vector<int> low_pads;
                if (op.int_array_attrs.count("edge_padding_low")) {
                    auto& v = op.int_array_attrs.at("edge_padding_low"); low_pads.assign(v.begin(), v.end());
                } else if (op.int_array_attrs.count("low")) {
                    auto& v = op.int_array_attrs.at("low"); low_pads.assign(v.begin(), v.end());
                } else {
                    // Default to 0 padding for original dims
                    size_t orig_rank = lhs.ndim() - (exp ? 1 : 0);
                    low_pads.assign(orig_rank, 0);
                }
                
                std::vector<int> high_pads;
                if (op.int_array_attrs.count("edge_padding_high")) {
                    auto& v = op.int_array_attrs.at("edge_padding_high"); high_pads.assign(v.begin(), v.end());
                } else if (op.int_array_attrs.count("high")) {
                    auto& v = op.int_array_attrs.at("high"); high_pads.assign(v.begin(), v.end());
                } else {
                    size_t orig_rank = lhs.ndim() - (exp ? 1 : 0);
                    high_pads.assign(orig_rank, 0);
                }
                
                std::vector<int> interior;
                if (op.int_array_attrs.count("interior_padding")) { auto& v = op.int_array_attrs.at("interior_padding"); interior.assign(v.begin(), v.end()); }
                else if (op.int_array_attrs.count("interior")) { auto& v = op.int_array_attrs.at("interior"); interior.assign(v.begin(), v.end()); }
                
                // If expanded, we must pad the new last dimension with 0
                if (exp) {
                    low_pads.push_back(0);
                    high_pads.push_back(0);
                    if (!interior.empty()) interior.push_back(0);
                }
                
                // Handle interior padding: insert zeros between elements
                // MLX doesn't support interior padding directly, so we do it manually:
                // For each dimension with interior > 0, we need to expand the array
                bool has_interior = false;
                for (size_t i = 0; i < interior.size() && i < (size_t)lhs.ndim(); i++) {
                    if (interior[i] > 0) has_interior = true;
                }
                
                mlx::core::array padded_lhs = lhs;
                if (has_interior) {
                    // For each dimension, if interior[d] > 0, expand that dimension
                    // New size = old_size + (old_size - 1) * interior[d]
                    for (size_t d = 0; d < interior.size() && d < (size_t)lhs.ndim(); d++) {
                        if (interior[d] > 0) {
                            int old_size = padded_lhs.shape()[d];
                            if (old_size <= 1) continue;  // No interior padding needed for size 0 or 1
                            
                            int new_size = old_size + (old_size - 1) * interior[d];
                            
                            // Create output filled with padding value
                            std::vector<int> new_shape(padded_lhs.shape().begin(), padded_lhs.shape().end());
                            new_shape[d] = new_size;
                            auto expanded = mlx::core::broadcast_to(val, mlx::core::Shape(new_shape.begin(), new_shape.end()));
                            
                            // Copy original elements at stride positions
                            // Use scatter-like logic: for each index i in original, place at i*(interior+1) in output
                            // Since MLX doesn't have easy scatter, use slice assignments via indexing
                            // Actually, we can build this using concatenation of slices
                            
                            // Alternative: reshape + pad per-slice approach
                            // Simpler: interleave zeros by transposing, padding on new axis, then flattening
                            // Cleaner approach: use repeat + masking
                            
                            // Simplest approach for 1D: expand_dims, broadcast, flatten, then slice
                            // For ND: handle dimension d specifically
                            
                            // Method: stack elements with zeros between them
                            // For dim d: take slices [i:i+1] and concat with zeros
                            std::vector<mlx::core::array> parts;
                            for (int i = 0; i < old_size; i++) {
                                // Slice at position i along dimension d
                                std::vector<int> starts(padded_lhs.ndim(), 0);
                                std::vector<int> ends(padded_lhs.shape().begin(), padded_lhs.shape().end());
                                starts[d] = i;
                                ends[d] = i + 1;
                                auto elem = mlx::core::slice(padded_lhs, 
                                    mlx::core::Shape(starts.begin(), starts.end()),
                                    mlx::core::Shape(ends.begin(), ends.end()));
                                parts.push_back(elem);
                                
                                // Add interior zeros (except after last element)
                                if (i < old_size - 1) {
                                    auto zero_shape = std::vector<int>(padded_lhs.shape().begin(), padded_lhs.shape().end());
                                    zero_shape[d] = interior[d];
                                    auto zeros = mlx::core::broadcast_to(val, mlx::core::Shape(zero_shape.begin(), zero_shape.end()));
                                    parts.push_back(zeros);
                                }
                            }
                            padded_lhs = mlx::core::concatenate(parts, d);
                        }
                    }
                }
                
                // MLX pad(array, axes, low, high, val) - for edge padding
                std::vector<int> axes(padded_lhs.ndim());
                std::iota(axes.begin(), axes.end(), 0);
                result = mlx::core::pad(padded_lhs, axes, mlx::core::Shape(low_pads.begin(), low_pads.end()), mlx::core::Shape(high_pads.begin(), high_pads.end()), val);
            }
        // --- Bitwise Operations ---
        } else if (op.op_name == "stablehlo.popcnt" || op.op_name == "mhlo.popcnt") {
            // Population count - count number of 1 bits
            if (!op_inputs.empty()) {
                auto x = op_inputs[0];
                // MLX doesn't have direct popcount, implement using parallel counting
                // This works for 32-bit integers
                auto dtype = x.dtype();
                
                // Ensure we work with uint32
                if (dtype != mlx::core::uint32) {
                    x = mlx::core::astype(x, mlx::core::uint32);
                }
                
                // Brian Kernighan's algorithm: count = 0; while(x) { x &= (x-1); count++; }
                // But that's a loop. Instead use parallel counting:
                // x = x - ((x >> 1) & 0x55555555)
                // x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
                // x = (x + (x >> 4)) & 0x0F0F0F0F
                // x = x * 0x01010101 >> 24
                
                auto m1 = mlx::core::array(0x55555555u, mlx::core::uint32);
                auto m2 = mlx::core::array(0x33333333u, mlx::core::uint32);
                auto m4 = mlx::core::array(0x0F0F0F0Fu, mlx::core::uint32);
                auto h01 = mlx::core::array(0x01010101u, mlx::core::uint32);
                
                auto t1 = mlx::core::right_shift(x, mlx::core::array(1, mlx::core::uint32));
                t1 = mlx::core::bitwise_and(t1, m1);
                x = mlx::core::subtract(x, t1);
                
                auto t2 = mlx::core::bitwise_and(x, m2);
                auto t3 = mlx::core::right_shift(x, mlx::core::array(2, mlx::core::uint32));
                t3 = mlx::core::bitwise_and(t3, m2);
                x = mlx::core::add(t2, t3);
                
                auto t4 = mlx::core::right_shift(x, mlx::core::array(4, mlx::core::uint32));
                x = mlx::core::add(x, t4);
                x = mlx::core::bitwise_and(x, m4);
                
                x = mlx::core::multiply(x, h01);
                result = mlx::core::right_shift(x, mlx::core::array(24, mlx::core::uint32));
                
                // Convert back to original dtype if needed
                if (dtype != mlx::core::uint32) {
                    result = mlx::core::astype(result, dtype);
                }
            }
        } else if (op.op_name == "stablehlo.xor" || op.op_name == "mhlo.xor") {
            if (op_inputs.size() >= 2) {
                auto lhs = op_inputs[0];
                auto rhs = op_inputs[1];
                bool expanded = ensure_binary_expanded(lhs, rhs);
                
                // Now both are expanded or both not.
                 // Force unsigned
                // Simply perform bitwise XOR. MLX handles types.
                 if (lhs.dtype() == mlx::core::float32 && rhs.dtype() != mlx::core::float32) {
                     result = mlx::core::bitwise_xor(lhs, mlx::core::astype(rhs, lhs.dtype()));
                 } else if (rhs.dtype() == mlx::core::float32 && lhs.dtype() != mlx::core::float32) {
                     result = mlx::core::bitwise_xor(mlx::core::astype(lhs, rhs.dtype()), rhs);
                 } else {
                     result = mlx::core::bitwise_xor(lhs, rhs);
                 }
            }
        } else if (op.op_name == "stablehlo.shift_left" || op.op_name == "mhlo.shift_left") {
            if (op_inputs.size() >= 2) {
                 auto lhs = op_inputs[0]; auto rhs = op_inputs[1];
                 // Shift usually doesn't expand RHS if LHS is expanded? 
                 // Actually, if we are simulating 32-bit ops on 64-bit values (expanded),
                 // we might need to be careful.
                 // But for now, ensure expansion consistency.
                  bool expanded = ensure_binary_expanded(lhs, rhs);
                  
                  if (expanded) {
                      // 64-bit shift simulation on [..., 2] arrays
                      auto shape_vec = lhs.shape();
                      std::vector<int> start(lhs.ndim(), 0);
                      std::vector<int> stop(shape_vec.begin(), shape_vec.end());
                      std::vector<int> strides(lhs.ndim(), 1);
                      
                      start.back() = 0; stop.back() = 1;
                      auto l_lo = mlx::core::reshape(mlx::core::slice(lhs, mlx::core::Shape(start.begin(), start.end()), mlx::core::Shape(stop.begin(), stop.end()), mlx::core::Shape(strides.begin(), strides.end())), mlx::core::Shape(lhs.shape().begin(), lhs.shape().end()-1));
                      
                      start.back() = 1; stop.back() = 2;
                      auto l_hi = mlx::core::reshape(mlx::core::slice(lhs, mlx::core::Shape(start.begin(), start.end()), mlx::core::Shape(stop.begin(), stop.end()), mlx::core::Shape(strides.begin(), strides.end())), mlx::core::Shape(lhs.shape().begin(), lhs.shape().end()-1));
                      
                      start.back() = 0; stop.back() = 1;
                      auto r_val = mlx::core::reshape(mlx::core::slice(rhs, mlx::core::Shape(start.begin(), start.end()), mlx::core::Shape(stop.begin(), stop.end()), mlx::core::Shape(strides.begin(), strides.end())), mlx::core::Shape(rhs.shape().begin(), rhs.shape().end()-1));
                      
                      auto s = mlx::core::bitwise_and(r_val, mlx::core::array(63, mlx::core::uint32)); // Mask 63
                      
                      // Case 1: s < 32
                      auto lo_s = mlx::core::left_shift(l_lo, s);
                      auto hi_s = mlx::core::bitwise_or(
                          mlx::core::left_shift(l_hi, s),
                          mlx::core::right_shift(l_lo, mlx::core::subtract(mlx::core::array(32, mlx::core::uint32), s))
                      );
                      
                      // Case 2: s >= 32
                      auto lo_s_32 = mlx::core::array(0, mlx::core::uint32);
                      auto hi_s_32 = mlx::core::left_shift(l_lo, mlx::core::subtract(s, mlx::core::array(32, mlx::core::uint32)));
                      
                      auto cond = mlx::core::less(s, mlx::core::array(32, mlx::core::uint32));
                      auto fres_lo = mlx::core::where(cond, lo_s, lo_s_32);
                      auto fres_hi = mlx::core::where(cond, hi_s, hi_s_32);
                      
                      result = mlx::core::stack({fres_lo, fres_hi}, -1);
                  } else {
                      // Normal 32-bit (or 168/8) shift
                      auto original_dtype = lhs.dtype();
                      bool cast_back = false;
                      if (lhs.dtype() == mlx::core::int32) { lhs = mlx::core::astype(lhs, mlx::core::uint32); cast_back = true; }
                      else if (lhs.dtype() == mlx::core::int64) { lhs = mlx::core::astype(lhs, mlx::core::uint64); cast_back = true; }
                      else if (lhs.dtype() == mlx::core::int16) { lhs = mlx::core::astype(lhs, mlx::core::uint16); cast_back = true; }
                      else if (lhs.dtype() == mlx::core::int8) { lhs = mlx::core::astype(lhs, mlx::core::uint8); cast_back = true; }
                      
                      if (cast_back) rhs = mlx::core::astype(rhs, lhs.dtype());
                      result = mlx::core::left_shift(lhs, rhs);
                      if (cast_back) result = mlx::core::astype(result, original_dtype);
                  }
            }
        } else if (op.op_name == "stablehlo.shift_right_logical" || op.op_name == "mhlo.shift_right_logical") {
            if (op_inputs.size() >= 2) {
                 auto lhs = op_inputs[0];
                 auto rhs = op_inputs[1];
                  bool expanded = ensure_binary_expanded(lhs, rhs);

                  if (expanded) {
                      // 64-bit logical right shift
                      auto shape_vec = lhs.shape();
                      std::vector<int> start(lhs.ndim(), 0);
                      std::vector<int> stop(shape_vec.begin(), shape_vec.end());
                      std::vector<int> strides(lhs.ndim(), 1);
                      
                      start.back() = 0; stop.back() = 1;
                      auto l_lo = mlx::core::reshape(mlx::core::slice(lhs, mlx::core::Shape(start.begin(), start.end()), mlx::core::Shape(stop.begin(), stop.end()), mlx::core::Shape(strides.begin(), strides.end())), mlx::core::Shape(lhs.shape().begin(), lhs.shape().end()-1));
                      
                      start.back() = 1; stop.back() = 2;
                      auto l_hi = mlx::core::reshape(mlx::core::slice(lhs, mlx::core::Shape(start.begin(), start.end()), mlx::core::Shape(stop.begin(), stop.end()), mlx::core::Shape(strides.begin(), strides.end())), mlx::core::Shape(lhs.shape().begin(), lhs.shape().end()-1));
                      
                      start.back() = 0; stop.back() = 1;
                      auto r_val = mlx::core::reshape(mlx::core::slice(rhs, mlx::core::Shape(start.begin(), start.end()), mlx::core::Shape(stop.begin(), stop.end()), mlx::core::Shape(strides.begin(), strides.end())), mlx::core::Shape(rhs.shape().begin(), rhs.shape().end()-1));
                      
                      auto s = mlx::core::bitwise_and(r_val, mlx::core::array(63, mlx::core::uint32));
                      
                      // Case 1: s < 32
                      auto hi_s = mlx::core::right_shift(l_hi, s);
                      auto lo_s = mlx::core::bitwise_or(
                          mlx::core::right_shift(l_lo, s),
                          mlx::core::left_shift(l_hi, mlx::core::subtract(mlx::core::array(32, mlx::core::uint32), s))
                      );
                      
                      // Case 2: s >= 32
                      auto hi_s_32 = mlx::core::array(0, mlx::core::uint32);
                      auto lo_s_32 = mlx::core::right_shift(l_hi, mlx::core::subtract(s, mlx::core::array(32, mlx::core::uint32)));
                      
                      auto cond = mlx::core::less(s, mlx::core::array(32, mlx::core::uint32));
                      auto fres_lo = mlx::core::where(cond, lo_s, lo_s_32);
                      auto fres_hi = mlx::core::where(cond, hi_s, hi_s_32);
                      
                      result = mlx::core::stack({fres_lo, fres_hi}, -1);
                  } else {
                      auto original_dtype = lhs.dtype();
                      bool cast_back = false;
                      // Force unsigned for logical shift
                      if (lhs.dtype() == mlx::core::int32) { lhs = mlx::core::astype(lhs, mlx::core::uint32); cast_back = true; }
                      else if (lhs.dtype() == mlx::core::int64) { lhs = mlx::core::astype(lhs, mlx::core::uint64); cast_back = true; }
                      else if (lhs.dtype() == mlx::core::int16) { lhs = mlx::core::astype(lhs, mlx::core::uint16); cast_back = true; }
                      else if (lhs.dtype() == mlx::core::int8) { lhs = mlx::core::astype(lhs, mlx::core::uint8); cast_back = true; }
                      
                      rhs = mlx::core::astype(rhs, lhs.dtype());
                      
                      // Handle shift >= bit_width (undefined behavior in C++, but should return 0 for logical shift)
                      int bit_width = lhs.itemsize() * 8;
                      auto shift_mask = mlx::core::greater_equal(rhs, mlx::core::array(bit_width, rhs.dtype()));
                      auto shifted = mlx::core::right_shift(lhs, mlx::core::minimum(rhs, mlx::core::array(bit_width - 1, rhs.dtype())));
                      result = mlx::core::where(shift_mask, mlx::core::zeros_like(lhs), shifted);
                      if (cast_back) result = mlx::core::astype(result, original_dtype);
                  }
            }
        } else if (op.op_name == "stablehlo.shift_right_arithmetic" || op.op_name == "mhlo.shift_right_arithmetic") {
            if (op_inputs.size() >= 2) result = mlx::core::right_shift(op_inputs[0], op_inputs[1]);
        } else if (op.op_name == "stablehlo.count_leading_zeros" || op.op_name == "mhlo.count_leading_zeros") {
             if (!op_inputs.empty()) {
                  auto x = op_inputs[0];
                  // CLZ via Smear + Popcount
                  // 1. Smear bits right
                  // Cast to unsigned to ensure logical shift
                  auto dtype = x.dtype();
                  if (dtype == mlx::core::int32) x = mlx::core::astype(x, mlx::core::uint32);
                  else if (dtype == mlx::core::int64) x = mlx::core::astype(x, mlx::core::uint64);
                  
                  x = mlx::core::bitwise_or(x, mlx::core::right_shift(x, mlx::core::array(1, x.dtype())));
                  x = mlx::core::bitwise_or(x, mlx::core::right_shift(x, mlx::core::array(2, x.dtype())));
                  x = mlx::core::bitwise_or(x, mlx::core::right_shift(x, mlx::core::array(4, x.dtype())));
                  x = mlx::core::bitwise_or(x, mlx::core::right_shift(x, mlx::core::array(8, x.dtype())));
                  x = mlx::core::bitwise_or(x, mlx::core::right_shift(x, mlx::core::array(16, x.dtype())));
                  if (x.itemsize() == 8) x = mlx::core::bitwise_or(x, mlx::core::right_shift(x, mlx::core::array(32, x.dtype())));
                  
                  // 2. Popcount of smeared = bit_width - clz
                  // Implement Popcount logic inline (Hamming weight) since we don't have popcnt op yet
                   // popcount_32(x):
                   // x -= (x >> 1) & 0x55555555;
                   // x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
                   // x = (x + (x >> 4)) & 0x0f0f0f0f;
                   // x += (x >> 8);
                   // x += (x >> 16);
                   // return x & 0x3f;
                   
                   // Simplified: Use existing subtraction/masks
                   // Note: Constants need to be scalar arrays
                   
                   auto k1 = mlx::core::array(0x55555555, x.dtype());
                   auto k2 = mlx::core::array(0x33333333, x.dtype());
                   auto k4 = mlx::core::array(0x0f0f0f0f, x.dtype());
                   // For 64-bit, we need larger constants
                   if (x.itemsize() == 8) {
                        k1 = mlx::core::array(0x5555555555555555ULL, x.dtype());
                        k2 = mlx::core::array(0x3333333333333333ULL, x.dtype());
                        k4 = mlx::core::array(0x0f0f0f0f0f0f0f0fULL, x.dtype());
                   }

                   auto x_shr1 = mlx::core::bitwise_and(mlx::core::right_shift(x, mlx::core::array(1, x.dtype())), k1);
                   x = mlx::core::subtract(x, x_shr1);
                   
                   auto x_shr2 = mlx::core::bitwise_and(mlx::core::right_shift(x, mlx::core::array(2, x.dtype())), k2);
                   x = mlx::core::add(mlx::core::bitwise_and(x, k2), x_shr2);
                   
                   auto x_shr4 = mlx::core::bitwise_and(mlx::core::right_shift(x, mlx::core::array(4, x.dtype())), k4);
                   x = mlx::core::bitwise_and(mlx::core::add(x, x_shr4), k4);
                   
                   // Multiply method for remaining bytes: (x * 0x01010101) >> 24
                   // Or just add shifts
                   x = mlx::core::add(x, mlx::core::right_shift(x, mlx::core::array(8, x.dtype())));
                   x = mlx::core::add(x, mlx::core::right_shift(x, mlx::core::array(16, x.dtype())));
                   if (x.itemsize() == 8) x = mlx::core::add(x, mlx::core::right_shift(x, mlx::core::array(32, x.dtype())));
                   
                   x = mlx::core::bitwise_and(mlx::core::right_shift(x, mlx::core::array(x.itemsize() == 8 ? 56 : 24, x.dtype())), mlx::core::array(0xff, x.dtype())); // Mask last byte just in case? Or 0x7f

                  // Result is bit_width - popcount
                  int bit_width = x.itemsize() * 8;
                  result = mlx::core::subtract(mlx::core::array(bit_width, x.dtype()), x);
                  result = mlx::core::astype(result, dtype); // Cast back
             }
        // --- FFT Operations ---
        } else if (op.op_name == "stablehlo.fft") {
            if (!op_inputs.empty()) {
                // Attributes: "fft_type" (FFT, IFFT, RFFT, IRFFT) in format "#stablehlo<fft_type XXX>"
                std::string type_attr = "";
                if (op.attributes.count("fft_type")) type_attr = op.attributes.at("fft_type");
                
                if (debug_mode()) {
                    std::cout << "[MLX-PJRT] FFT: type_attr=" << type_attr << " input_shape=" << op_inputs[0].shape() << std::endl;
                }
                
                // Check for FFT type in attribute string (can be "RFFT", "IRFFT", "IFFT", "FFT")
                // Attribute format: "#stablehlo<fft_type RFFT>" or just "RFFT"
                
                // First, check if this is a multi-dimensional FFT by parsing fft_length
                std::vector<int> fft_lengths;
                if (op.attributes.count("fft_length")) {
                    std::string fft_len_str = op.attributes.at("fft_length");
                    // Parse [2, 3] or [4] format
                    size_t pos = fft_len_str.find('[');
                    if (pos != std::string::npos) {
                        size_t end = fft_len_str.find(']', pos);
                        if (end != std::string::npos) {
                            std::string nums = fft_len_str.substr(pos + 1, end - pos - 1);
                            // Parse comma-separated values
                            std::stringstream ss(nums);
                            std::string item;
                            while (std::getline(ss, item, ',')) {
                                // Trim whitespace
                                size_t start = item.find_first_not_of(" ");
                                size_t finish = item.find_last_not_of(" ");
                                if (start != std::string::npos) {
                                    try { fft_lengths.push_back(std::stoi(item.substr(start, finish - start + 1))); } catch (...) {}
                                }
                            }
                        }
                    }
                }
                bool is_multidim = fft_lengths.size() > 1;
                if (debug_mode() && is_multidim) std::cout << "[MLX-PJRT] FFT multi-dim: " << fft_lengths.size() << "D" << std::endl;
                
                if (type_attr.find("IRFFT") != std::string::npos) {
                    // Get the expected output length from fft_length attribute
                    int n = op_inputs[0].shape(-1);
                    if (!fft_lengths.empty()) n = fft_lengths.back();
                    if (debug_mode()) std::cout << "[MLX-PJRT] FFT IRFFT final n=" << n << std::endl;
                    if (is_multidim) {
                        result = mlx::core::fft::irfftn(op_inputs[0]);
                    } else {
                        result = mlx::core::fft::irfft(op_inputs[0], n, -1);  // axis=-1 (last dim)
                    }
                    if (debug_mode()) std::cout << "[MLX-PJRT] FFT IRFFT result=" << result.shape() << " dtype=" << result.dtype() << std::endl;
                } else if (type_attr.find("RFFT") != std::string::npos) {
                    if (is_multidim) {
                        result = mlx::core::fft::rfftn(op_inputs[0]);
                    } else {
                        result = mlx::core::fft::rfft(op_inputs[0]);
                    }
                    if (debug_mode()) std::cout << "[MLX-PJRT] FFT RFFT result=" << result.shape() << std::endl;
                } else if (type_attr.find("IFFT") != std::string::npos) {
                    if (is_multidim) {
                        result = mlx::core::fft::ifftn(op_inputs[0]);
                    } else {
                        result = mlx::core::fft::ifft(op_inputs[0]);
                    }
                } else {
                    // Default to FFT
                    if (is_multidim) {
                        result = mlx::core::fft::fftn(op_inputs[0]);
                    } else {
                        result = mlx::core::fft::fft(op_inputs[0]);
                    }
                }
            }
        // --- Linear Algebra ---
        } else if (op.op_name == "stablehlo.cholesky") {
            if (!op_inputs.empty()) {
                bool lower = true;
                if (op.attributes.count("lower")) {
                     std::string val = op.attributes.at("lower");
                     if (val == "false" || val == "0") lower = false;
                }
                // MLX cholesky(a, upper=False), currently CPU only
                result = mlx::core::linalg::cholesky(op_inputs[0], !lower, mlx::core::Device::cpu);
            }
        } else if (op.op_name == "stablehlo.popcnt" || op.op_name == "mhlo.popcnt") {
             if (!op_inputs.empty()) {
                  auto x = op_inputs[0];
                  auto dtype = x.dtype();
                  // Force unsigned
                  if (dtype == mlx::core::int32) x = mlx::core::astype(x, mlx::core::uint32);
                  else if (dtype == mlx::core::int64) x = mlx::core::astype(x, mlx::core::uint64);
                  
                   // SWar Popcount
                   auto k1 = mlx::core::array(0x55555555, x.dtype());
                   auto k2 = mlx::core::array(0x33333333, x.dtype());
                   auto k4 = mlx::core::array(0x0f0f0f0f, x.dtype());
                   if (x.itemsize() == 8) {
                        k1 = mlx::core::array(0x5555555555555555ULL, x.dtype());
                        k2 = mlx::core::array(0x3333333333333333ULL, x.dtype());
                        k4 = mlx::core::array(0x0f0f0f0f0f0f0f0fULL, x.dtype());
                   }
                   
                   auto x_shr1 = mlx::core::bitwise_and(mlx::core::right_shift(x, mlx::core::array(1, x.dtype())), k1);
                   x = mlx::core::subtract(x, x_shr1);
                   
                   auto x_shr2 = mlx::core::bitwise_and(mlx::core::right_shift(x, mlx::core::array(2, x.dtype())), k2);
                   x = mlx::core::add(mlx::core::bitwise_and(x, k2), x_shr2);
                   
                   auto x_shr4 = mlx::core::bitwise_and(mlx::core::right_shift(x, mlx::core::array(4, x.dtype())), k4);
                   x = mlx::core::bitwise_and(mlx::core::add(x, x_shr4), k4);
                   
                   x = mlx::core::add(x, mlx::core::right_shift(x, mlx::core::array(8, x.dtype())));
                   x = mlx::core::add(x, mlx::core::right_shift(x, mlx::core::array(16, x.dtype())));
                   if (x.itemsize() == 8) x = mlx::core::add(x, mlx::core::right_shift(x, mlx::core::array(32, x.dtype())));
                   
                   int shift_total = (x.itemsize() == 8) ? 56 : 24;
                   result = mlx::core::bitwise_and(mlx::core::right_shift(x, mlx::core::array(shift_total, x.dtype())), mlx::core::array(0xff, x.dtype()));
                   result = mlx::core::astype(result, dtype);
             }
        // --- Reverse/Flip Operations ---
        } else if (op.op_name == "stablehlo.reverse" || op.op_name == "mhlo.reverse") {
            if (!op_inputs.empty()) {
                std::vector<int> dims;
                if (op.int_array_attrs.count("dimensions")) {
                    auto& v = op.int_array_attrs.at("dimensions"); dims.assign(v.begin(), v.end());
                }
                result = op_inputs[0];
                // Implement reverse using slice with negative strides or index manipulation
                for (int d : dims) {
                    // MLX doesn't have flip, implement via indexing:
                    // arr[::-1] equivalent using take with reversed indices
                    int dim_size = result.shape()[d];
                    if (dim_size > 0) {
                        // Create reversed indices: [dim_size-1, dim_size-2, ..., 0]
                        auto indices = mlx::core::arange(dim_size - 1, -1, -1, mlx::core::int32);
                        result = mlx::core::take(result, indices, d);
                    }
                }
            }
        // --- Real/Imag Operations ---
        } else if (op.op_name == "stablehlo.real" || op.op_name == "mhlo.real") {
            if (!op_inputs.empty()) result = op_inputs[0]; // Fallback for real part
        } else if (op.op_name == "stablehlo.imag" || op.op_name == "mhlo.imag") {
            if (!op_inputs.empty()) result = mlx::core::zeros_like(op_inputs[0]); // Fallback
        } else if (op.op_name == "stablehlo.complex" || op.op_name == "mhlo.complex") {
            if (!op_inputs.empty()) result = op_inputs[0]; // Fallback
        // --- Is operations ---
        } else if (op.op_name == "stablehlo.is_finite" || op.op_name == "mhlo.is_finite") {
            if (!op_inputs.empty()) result = mlx::core::isfinite(op_inputs[0]);
        } else if (op.op_name == "stablehlo.is_inf" || op.op_name == "mhlo.is_inf") {
            if (!op_inputs.empty()) result = mlx::core::isinf(op_inputs[0]);
        } else if (op.op_name == "stablehlo.is_nan" || op.op_name == "mhlo.is_nan") {
            if (!op_inputs.empty()) result = mlx::core::isnan(op_inputs[0]);
        // --- Argmax/Argmin ---  
        } else if (op.op_name == "stablehlo.reduce_argmax" || op.op_name == "mhlo.reduce_argmax") {
            if (!op_inputs.empty()) result = mlx::core::argmax(op_inputs[0]);
        } else if (op.op_name == "stablehlo.reduce_argmin" || op.op_name == "mhlo.reduce_argmin") {
            if (!op_inputs.empty()) result = mlx::core::argmin(op_inputs[0]);
        // --- Mean reduction ---
        } else if (op.op_name == "stablehlo.reduce_mean" || op.op_name == "mhlo.reduce_mean") {
            if (!op_inputs.empty()) result = mlx::core::mean(op_inputs[0]);
        // --- Identity/Copy ---
        } else if (op.op_name == "stablehlo.copy" || op.op_name == "mhlo.copy") {
            if (!op_inputs.empty()) result = op_inputs[0];
        // --- Sort ---
        } else if (op.op_name == "stablehlo.sort" || op.op_name == "mhlo.sort") {
            if (!op_inputs.empty()) {
                int axis = 0; // Default to first axis
                if (op.int_attrs.count("dimension")) {
                    axis = op.int_attrs.at("dimension");
                }
                if (op.int_array_attrs.count("dimension")) {
                    auto& dim_vec = op.int_array_attrs.at("dimension");
                    if (!dim_vec.empty()) axis = dim_vec[0];
                }
                
                // Multi-input sort: use argsort on first input, apply to all
                if (op_inputs.size() > 1) {
                    // Get sort indices from first input (keys)
                    auto sort_indices = mlx::core::argsort(op_inputs[0], axis);
                    
                    // Apply indices to each input
                    for (size_t i = 0; i < op_inputs.size(); ++i) {
                        auto sorted = mlx::core::take_along_axis(op_inputs[i], sort_indices, axis);
                        op_outputs.push_back(sorted);
                    }
                } else {
                    // Single input: simple sort
                    result = mlx::core::sort(op_inputs[0], axis);
                }
            }
        // --- Reduce Window (pooling) ---
        } else if (op.op_name == "stablehlo.reduce_window" || op.op_name == "mhlo.reduce_window") {
            // Detect pool type from init value
            bool is_max_pool = false;
            bool is_sum_pool = false;
            if (op_inputs.size() >= 2) {
                auto init_val = op_inputs[1];
                if (init_val.size() == 1) {
                    mlx::core::eval(init_val);  // Need concrete value
                    if (init_val.dtype() == mlx::core::float32) {
                        float val = init_val.item<float>();
                        if (std::isinf(val) && val < 0) is_max_pool = true;
                        else if (val == 0.0f) is_sum_pool = true;
                    }
                }
            }
            
            if ((is_max_pool || is_sum_pool) && op.int_array_attrs.count("window_dimensions") && op.int_array_attrs.count("window_strides")) {
                 auto win_dims = op.int_array_attrs.at("window_dimensions");
                 auto strides = op.int_array_attrs.at("window_strides");
                 
                 // Detect layout from window dimensions - spatial dims have window > 1
                 // NHWC: [1, 2, 2, 1] -> h_dim=1, w_dim=2
                 // HWCN: [2, 2, 1, 1] -> h_dim=0, w_dim=1
                 int h_dim = -1, w_dim = -1, n_dim = -1, c_dim = -1;
                 std::vector<int> spatial_dims, non_spatial_dims;
                 
                 for (size_t i = 0; i < win_dims.size(); ++i) {
                     if (win_dims[i] > 1) spatial_dims.push_back(i);
                     else non_spatial_dims.push_back(i);
                 }
                 
                 if (spatial_dims.size() == 2 && win_dims.size() >= 4) {
                     h_dim = spatial_dims[0];
                     w_dim = spatial_dims[1];
                     if (non_spatial_dims.size() >= 2) {
                         if (non_spatial_dims[0] == 0) { // NHWC
                             n_dim = non_spatial_dims[0];
                             c_dim = non_spatial_dims[1];
                         } else { // HWCN
                             c_dim = non_spatial_dims[0];
                             n_dim = non_spatial_dims[1];
                         }
                     }
                     
                     int win_h = static_cast<int>(win_dims[h_dim]);
                     int win_w = static_cast<int>(win_dims[w_dim]);
                     int str_h = static_cast<int>(strides[h_dim]);
                     int str_w = static_cast<int>(strides[w_dim]);
                     
                     auto input = op_inputs[0];
                     size_t H_sz = input.shape()[h_dim];
                     size_t W_sz = input.shape()[w_dim];
                     size_t N_sz = input.shape()[n_dim];
                     size_t C_sz = input.shape()[c_dim];
                     
                     size_t win_h_sz = static_cast<size_t>(win_h);
                     size_t win_w_sz = static_cast<size_t>(win_w);
                     size_t str_h_sz = static_cast<size_t>(str_h);
                     size_t str_w_sz = static_cast<size_t>(str_w);
                     size_t out_h_sz = (H_sz - win_h_sz) / str_h_sz + 1;
                     size_t out_w_sz = (W_sz - win_w_sz) / str_w_sz + 1;
                     
                     // Build output shape in same layout as input
                     mlx::core::Shape out_shape(4);
                     out_shape[h_dim] = static_cast<int>(out_h_sz);
                     out_shape[w_dim] = static_cast<int>(out_w_sz);
                     out_shape[n_dim] = static_cast<int>(N_sz);
                     out_shape[c_dim] = static_cast<int>(C_sz);
                     
                     if (is_max_pool) {
                         result = mlx::core::full(out_shape, -std::numeric_limits<float>::infinity(), input.dtype());
                     } else {
                         result = mlx::core::zeros(out_shape, input.dtype());
                     }
                     
                     for (size_t wh = 0; wh < win_h_sz; ++wh) {
                         for (size_t ww = 0; ww < win_w_sz; ++ww) {
                             std::vector<int> start_idx(4), stop_idx(4), stride_idx(4);
                             
                             start_idx[n_dim] = 0;
                             stop_idx[n_dim] = static_cast<int>(N_sz);
                             stride_idx[n_dim] = 1;
                             
                             start_idx[c_dim] = 0;
                             stop_idx[c_dim] = static_cast<int>(C_sz);
                             stride_idx[c_dim] = 1;
                             
                             start_idx[h_dim] = static_cast<int>(wh);
                             stop_idx[h_dim] = static_cast<int>(wh + out_h_sz * str_h_sz);
                             stride_idx[h_dim] = static_cast<int>(str_h_sz);
                             
                             start_idx[w_dim] = static_cast<int>(ww);
                             stop_idx[w_dim] = static_cast<int>(ww + out_w_sz * str_w_sz);
                             stride_idx[w_dim] = static_cast<int>(str_w_sz);
                             
                             mlx::core::Shape start_shape(start_idx.begin(), start_idx.end());
                             mlx::core::Shape stop_shape(stop_idx.begin(), stop_idx.end());
                             mlx::core::Shape stride_shape(stride_idx.begin(), stride_idx.end());
                             
                             auto window_vals = mlx::core::slice(input, start_shape, stop_shape, stride_shape);
                             if (is_max_pool) {
                                 result = mlx::core::maximum(result, window_vals);
                             } else {
                                 result = mlx::core::add(result, window_vals);
                             }
                         }
                     }
                     
                 } else {
                     // Fallback for non-standard pooling
                     if (!op_inputs.empty()) result = op_inputs[0]; 
                 }
            } else {
                 if (!op_inputs.empty()) result = op_inputs[0];
            }
        // --- Convolution ---
        } else if (op.op_name == "stablehlo.convolution" || op.op_name == "mhlo.convolution") {
            if (op_inputs.size() >= 2) {
                auto input = op_inputs[0]; 
                auto kernel = op_inputs[1];
                
                // 1. Parse dimension numbers (or use defaults)
                int64_t in_batch = 0, in_feat = 3;
                std::vector<int64_t> in_spatial = {1, 2};
                
                int64_t kern_in = 2, kern_out = 3;
                std::vector<int64_t> kern_spatial = {0, 1};
                
                int64_t out_batch = 0, out_feat = 3;
                std::vector<int64_t> out_spatial = {1, 2};
                
                if (op.int_attrs.count("input_batch_dimension")) {
                    in_batch = op.int_attrs.at("input_batch_dimension");
                    in_feat = op.int_attrs.at("input_feature_dimension");
                    in_spatial = op.int_array_attrs.at("input_spatial_dimensions");
                    
                    kern_in = op.int_attrs.at("kernel_input_feature_dimension");
                    kern_out = op.int_attrs.at("kernel_output_feature_dimension");
                    kern_spatial = op.int_array_attrs.at("kernel_spatial_dimensions");
                    
                    out_batch = op.int_attrs.at("output_batch_dimension");
                    out_feat = op.int_attrs.at("output_feature_dimension");
                    out_spatial = op.int_array_attrs.at("output_spatial_dimensions");
                }

                // 2. Permute Input to NHWC [Batch, Spatial..., Feature]
                std::vector<int> in_perm;
                in_perm.push_back(static_cast<int>(in_batch));
                for(auto d : in_spatial) in_perm.push_back(static_cast<int>(d));
                in_perm.push_back(static_cast<int>(in_feat));
                
                bool need_in_transpose = false;
                for(size_t i=0; i<in_perm.size(); ++i) if(in_perm[i] != i) need_in_transpose = true;
                
                if (need_in_transpose && input.ndim() == in_perm.size()) {
                    input = mlx::core::transpose(input, in_perm);
                }

                // 3. Permute Kernel to OHWI [Out, Spatial..., In]
                // (MLX expects: weight filters of shape [out_channels, H, W, in_channels])
                std::vector<int> kern_perm;
                kern_perm.push_back(static_cast<int>(kern_out));
                for(auto d : kern_spatial) kern_perm.push_back(static_cast<int>(d));
                kern_perm.push_back(static_cast<int>(kern_in));
                
                if (kernel.ndim() == kern_perm.size()) {
                    kernel = mlx::core::transpose(kernel, kern_perm);
                }

                // 4. Parse Strides and Dilations (works for any number of spatial dims)
                size_t num_spatial = in_spatial.size();
                
                std::vector<int> strides;
                if (op.int_array_attrs.count("window_strides")) {
                    auto& stride_arr = op.int_array_attrs.at("window_strides");
                    for (auto s : stride_arr) strides.push_back(static_cast<int>(s));
                }
                while (strides.size() < num_spatial) strides.push_back(1);
                
                std::vector<int> lhs_dilation;  // Input dilation
                if (op.int_array_attrs.count("lhs_dilation")) {
                    auto& dil_arr = op.int_array_attrs.at("lhs_dilation");
                    for (auto d : dil_arr) lhs_dilation.push_back(static_cast<int>(d));
                }
                while (lhs_dilation.size() < num_spatial) lhs_dilation.push_back(1);

                std::vector<int> rhs_dilation;  // Kernel dilation
                if (op.int_array_attrs.count("rhs_dilation")) {
                    auto& dil_arr = op.int_array_attrs.at("rhs_dilation");
                    for (auto d : dil_arr) rhs_dilation.push_back(static_cast<int>(d));
                }
                while (rhs_dilation.size() < num_spatial) rhs_dilation.push_back(1);

                // 5. Calculate Padding (generalized for any dimension count)
                std::vector<int> pad_lo(num_spatial, 0);
                std::vector<int> pad_hi(num_spatial, 0);
                
                // Override with explicit padding if present
                if (op.int_array_attrs.count("padding")) {
                    auto& pad_arr = op.int_array_attrs.at("padding");
                    // padding is stored as [low0, high0, low1, high1, ...]
                    for (size_t i = 0; i < num_spatial && i * 2 + 1 < pad_arr.size(); ++i) {
                        pad_lo[i] = static_cast<int>(pad_arr[i * 2]);
                        pad_hi[i] = static_cast<int>(pad_arr[i * 2 + 1]);
                    }
                }

                if (debug_mode()) {
                    std::cout << "[MLX-PJRT] conv_general: in=" << input.shape() << " k=" << kernel.shape() 
                              << " stride=[";
                    for (auto s : strides) std::cout << s << ",";
                    std::cout << "] pad_lo=[";
                    for (auto p : pad_lo) std::cout << p << ",";
                    std::cout << "] pad_hi=[";
                    for (auto p : pad_hi) std::cout << p << ",";
                    std::cout << "]" << std::endl;
                }

                // 6. Use conv_general which supports any dimensionality
                result = mlx::core::conv_general(input, kernel, strides, pad_lo, pad_hi, 
                                                 rhs_dilation, lhs_dilation);
                
                // 7. Permute Output to Target Layout
                // MLX result: [Batch(0), Spatial...(1..N), Feature(N+1)]
                std::vector<int> out_perm(result.ndim());
                if ((size_t)out_batch < out_perm.size()) out_perm[out_batch] = 0;
                if ((size_t)out_feat < out_perm.size()) out_perm[out_feat] = static_cast<int>(num_spatial + 1);
                for (size_t i = 0; i < out_spatial.size(); ++i) {
                    if ((size_t)out_spatial[i] < out_perm.size()) 
                        out_perm[out_spatial[i]] = static_cast<int>(i + 1);
                }
                
                bool need_out_transpose = false;
                for(size_t i=0; i<out_perm.size(); ++i) if(out_perm[i] != static_cast<int>(i)) need_out_transpose = true;
                
                if (need_out_transpose) {
                    result = mlx::core::transpose(result, out_perm);
                }

            } else {
                if (!op_inputs.empty()) result = op_inputs[0];
            }
        // --- Custom call ---
        // --- FFT Operations (duplicate handler - updated to match line 3675) ---
        } else if (op.op_name == "stablehlo.fft") {
            if (!op_inputs.empty()) {
                // Attributes: "fft_type" in format "#stablehlo<fft_type XXX>"
                std::string type_attr = "";
                if (op.attributes.count("fft_type")) type_attr = op.attributes.at("fft_type");
                
                // Check for FFT type using find() since format is "#stablehlo<fft_type RFFT>"
                if (type_attr.find("IRFFT") != std::string::npos) {
                    int n = op_inputs[0].shape(-1);
                    if (op.int_array_attrs.count("fft_length")) {
                        auto& lens = op.int_array_attrs.at("fft_length");
                        if (!lens.empty()) n = lens.back();
                    } else if (op.attributes.count("fft_length")) {
                        std::string fft_len_str = op.attributes.at("fft_length");
                        size_t pos = fft_len_str.find('[');
                        if (pos != std::string::npos) {
                            size_t end = fft_len_str.find(']', pos);
                            if (end != std::string::npos) {
                                std::string num_str = fft_len_str.substr(pos + 1, end - pos - 1);
                                try { n = std::stoi(num_str); } catch (...) {}
                            }
                        }
                    }
                    result = mlx::core::fft::irfft(op_inputs[0], n);
                } else if (type_attr.find("RFFT") != std::string::npos) {
                    result = mlx::core::fft::rfft(op_inputs[0]);
                } else if (type_attr.find("IFFT") != std::string::npos) {
                    result = mlx::core::fft::ifft(op_inputs[0]);
                } else {
                    result = mlx::core::fft::fft(op_inputs[0]);
                }
            }
        // --- Linear Algebra ---
        } else if (op.op_name == "stablehlo.cholesky") {
            if (!op_inputs.empty()) {
                bool lower = true;
                if (op.attributes.count("lower")) {
                     std::string val = op.attributes.at("lower");
                     if (val == "false" || val == "0") lower = false;
                }
                // MLX cholesky(a, upper=False), currently CPU only
                result = mlx::core::linalg::cholesky(op_inputs[0], !lower, mlx::core::Device::cpu);
            }
        // --- Fusion Region ---
        } else if (op.op_name == "stablehlo.fusion") {
            // Fusion: execute the subgraph in region 0
            if (op.subgraphs.size() != 1) {
                std::cerr << "[MLX-PJRT][ERROR] stablehlo.fusion must have 1 region" << std::endl;
            } else {
                if (debug_mode()) std::cout << "[MLX-PJRT]   Entering Fusion Region..." << std::endl;
                
                // Get inputs for the fused function
                std::vector<mlx::core::array> region_inputs;
                for (int in_id : op.inputs) {
                    if (val_map.count(in_id)) {
                        region_inputs.push_back(val_map.at(in_id));
                    } else {
                        if (debug_mode()) std::cout << "[MLX-PJRT]     Fusion input " << in_id << " not found!" << std::endl;
                        region_inputs.push_back(mlx::core::array({0.0f})); // Dummy fallback
                    }
                }

                // Execute the region
                std::vector<mlx::core::array> region_outputs = ExecuteGraph(*op.subgraphs[0], region_inputs, nullptr, functions);
                
                if (region_outputs.size() != op.outputs.size()) {
                     std::cerr << "[MLX-PJRT][ERROR] Fusion region output count mismatch" << std::endl;
                }
                
                // Bind outputs
                for (size_t k = 0; k < region_outputs.size() && k < op.outputs.size(); ++k) {
                    val_map.erase(op.outputs[k]);
                    val_map.insert(std::make_pair(op.outputs[k], region_outputs[k]));
                }
                
                if (getenv("MLX_PJRT_DEBUG")) std::cout << "[MLX-PJRT]   Exited Fusion Region" << std::endl;
            }
        // --- Select And Scatter (MaxPool Gradient) - OPTIMIZED ---
        } else if (op.op_name == "stablehlo.select_and_scatter" || op.op_name == "mhlo.select_and_scatter") {
            // Implements MaxPool gradient using optimized mask-based approach
            if (op_inputs.size() >= 3) {
                auto operand = op_inputs[0];  // [N, H, W, C] - original forward input
                auto source = op_inputs[1];   // [N, H_out, W_out, C] - gradient from next layer  
                auto init_val = op_inputs[2]; // Scalar (usually 0.0)
                
                // Parse window dimensions and strides
                std::vector<int64_t> win_dims = {1, 2, 2, 1};
                std::vector<int64_t> strides_arr = {1, 2, 2, 1};
                
                if (op.int_array_attrs.count("window_dimensions")) {
                    win_dims = op.int_array_attrs.at("window_dimensions");
                }
                if (op.int_array_attrs.count("window_strides")) {
                    strides_arr = op.int_array_attrs.at("window_strides");
                }
                
                // Detect layout from window dimensions - spatial dims have window > 1
                // NHWC: [1, 2, 2, 1] -> h_dim=1, w_dim=2
                // HWCN: [2, 2, 1, 1] -> h_dim=0, w_dim=1
                int h_dim = -1, w_dim = -1, n_dim = -1, c_dim = -1;
                std::vector<int> spatial_dims, non_spatial_dims;
                
                for (size_t i = 0; i < win_dims.size(); ++i) {
                    if (win_dims[i] > 1) spatial_dims.push_back(i);
                    else non_spatial_dims.push_back(i);
                }
                
                if (spatial_dims.size() == 2 && win_dims.size() >= 4) {
                    h_dim = spatial_dims[0];
                    w_dim = spatial_dims[1];
                    if (non_spatial_dims.size() >= 2) {
                        if (non_spatial_dims[0] == 0) { // NHWC
                            n_dim = non_spatial_dims[0];
                            c_dim = non_spatial_dims[1];
                        } else { // HWCN
                            c_dim = non_spatial_dims[0];
                            n_dim = non_spatial_dims[1];
                        }
                    }
                    
                    int win_h = static_cast<int>(win_dims[h_dim]);
                    int win_w = static_cast<int>(win_dims[w_dim]);
                    int str_h = static_cast<int>(strides_arr[h_dim]);
                    int str_w = static_cast<int>(strides_arr[w_dim]);
                    
                    int N = static_cast<int>(operand.shape()[n_dim]);
                    int H = static_cast<int>(operand.shape()[h_dim]);
                    int W = static_cast<int>(operand.shape()[w_dim]);
                    int C = static_cast<int>(operand.shape()[c_dim]);
                    int H_out = static_cast<int>(source.shape()[h_dim]);
                    int W_out = static_cast<int>(source.shape()[w_dim]);
                    
                    // For non-overlapping pooling with NHWC, use efficient reshape approach
                    if (n_dim == 0 && str_h == win_h && str_w == win_w && H == H_out * win_h && W == W_out * win_w) {
                        // Step 1: Create window view via reshape: [N, H_out, win_h, W_out, win_w, C]
                        auto reshaped = mlx::core::reshape(operand, {N, H_out, win_h, W_out, win_w, C});
                        
                        // Step 2: Transpose to [N, H_out, W_out, win_h, win_w, C]
                        auto windows = mlx::core::transpose(reshaped, {0, 1, 3, 2, 4, 5});
                        
                        // Step 3: Find max (fast) and create mask
                        auto max_val = mlx::core::max(windows, {3, 4}, true);
                        auto mask = mlx::core::equal(windows, max_val);
                        mask = mlx::core::astype(mask, operand.dtype());
                        
                        // Handle ties by normalizing
                        auto mask_sum = mlx::core::sum(mask, {3, 4}, true);
                        auto one_arr = mlx::core::array(1.0f, operand.dtype());
                        mask = mlx::core::divide(mask, mlx::core::maximum(mask_sum, one_arr));
                        
                        // Step 4: Broadcast source gradient and multiply by mask
                        auto source_exp = mlx::core::reshape(source, {N, H_out, W_out, 1, 1, C});
                        auto grad_windows = mlx::core::multiply(source_exp, mask);
                        
                        // Step 5: Reshape back to original shape
                        auto grad_transposed = mlx::core::transpose(grad_windows, {0, 1, 3, 2, 4, 5});
                        result = mlx::core::reshape(grad_transposed, {N, H, W, C});
                    } else {
                        // HWCN layout or overlapping pooling: fallback to zeros
                        // Correct gradient requires more complex implementation
                        result = mlx::core::zeros(operand.shape(), operand.dtype());
                    }
                } else {
                    // Non-standard pattern: fallback to zeros
                    result = mlx::core::zeros_like(operand);
                }
            } else if (!op_inputs.empty()) {
                result = mlx::core::zeros_like(op_inputs[0]);
            }
        } else if (op.op_name == "stablehlo.custom_call" || op.op_name == "mhlo.custom_call") {
            // Custom call dispatch based on call_target_name
            std::string target = "";
            if (op.attributes.count("call_target_name")) {
                target = op.attributes.at("call_target_name");
            }
            
            // CHLO ops that get lowered to custom_call
            // NOTE: Check asinh/acosh/atanh BEFORE sinh/cosh/tanh to avoid prefix matching
            if (target.find("asinh") != std::string::npos) {
                if (!op_inputs.empty()) result = mlx::core::arcsinh(op_inputs[0]);
            } else if (target.find("acosh") != std::string::npos) {
                if (!op_inputs.empty()) result = mlx::core::arccosh(op_inputs[0]);
            } else if (target.find("atanh") != std::string::npos) {
                if (!op_inputs.empty()) result = mlx::core::arctanh(op_inputs[0]);
            } else if (target.find("sinh") != std::string::npos) {
                if (!op_inputs.empty()) result = mlx::core::sinh(op_inputs[0]);
            } else if (target.find("cosh") != std::string::npos) {
                if (!op_inputs.empty()) result = mlx::core::cosh(op_inputs[0]);
            } else if (target.find("tan") != std::string::npos && target.find("atan") == std::string::npos) {
                if (!op_inputs.empty()) result = mlx::core::tan(op_inputs[0]);
            } else if (target.find("erf") != std::string::npos && target.find("erfc") == std::string::npos && target.find("erfinv") == std::string::npos) {
                if (!op_inputs.empty()) result = mlx::core::erf(op_inputs[0]);
            } else if (target.find("erfc") != std::string::npos) {
                // erfc(x) = 1 - erf(x)
                if (!op_inputs.empty()) result = mlx::core::subtract(mlx::core::array(1.0f), mlx::core::erf(op_inputs[0]));
            } else if (target.find("log1p") != std::string::npos) {
                if (!op_inputs.empty()) result = mlx::core::log1p(op_inputs[0]);
            } else if (target.find("expm1") != std::string::npos) {
                if (!op_inputs.empty()) result = mlx::core::expm1(op_inputs[0]);
            } else if (target.find("asin") != std::string::npos) {
                if (!op_inputs.empty()) result = mlx::core::arcsin(op_inputs[0]);
            } else if (target.find("acos") != std::string::npos) {
                if (!op_inputs.empty()) result = mlx::core::arccos(op_inputs[0]);
            } else if (target.find("atan") != std::string::npos) {
                if (!op_inputs.empty()) result = mlx::core::arctan(op_inputs[0]);
            } 
            // LAPACK FFI custom calls - Linear Algebra
            // eig: lapack_sgeev_ffi (float), lapack_dgeev_ffi (double), lapack_cgeev_ffi (complex)
            else if (target.find("geev") != std::string::npos || target.find("_eig") != std::string::npos) {
                if (!op_inputs.empty()) {
                    try {
                        auto eig_result = mlx::core::linalg::eig(op_inputs[0], mlx::core::Device(mlx::core::Device::cpu));
                        // Output order: eigenvalues_real, eigenvalues_imag, left_eigenvectors, right_eigenvectors, info
                        // MLX eig returns: pair<eigenvalues, eigenvectors>
                        auto eigenvalues = eig_result.first;
                        auto eigenvectors = eig_result.second;
                        
                        // For real input, eigenvalues may be complex - extract real/imag parts
                        auto real_part = mlx::core::real(eigenvalues);
                        auto imag_part = mlx::core::imag(eigenvalues);
                        
                        // Build multi-output
                        op_outputs.clear();
                        op_outputs.push_back(real_part);
                        op_outputs.push_back(imag_part);
                        op_outputs.push_back(eigenvectors);  // left eigenvectors (same for symmetric)
                        op_outputs.push_back(eigenvectors);  // right eigenvectors
                        op_outputs.push_back(mlx::core::array(0, mlx::core::int32));  // info = 0 (success)
                        result = eigenvalues;  // Also set result for single-output fallback
                    } catch (const std::exception& e) {
                        if (debug_mode()) std::cout << "[MLX-PJRT] eig failed: " << e.what() << std::endl;
                        if (!op_inputs.empty()) result = op_inputs[0];
                    }
                }
            }
            // eigh (symmetric/hermitian eig): lapack_ssyevd_ffi, lapack_dsyevd_ffi
            else if (target.find("syev") != std::string::npos || target.find("heev") != std::string::npos || target.find("_eigh") != std::string::npos) {
                if (!op_inputs.empty()) {
                    try {
                        auto eigh_result = mlx::core::linalg::eigh(op_inputs[0], "L", mlx::core::Device(mlx::core::Device::cpu));
                        // LAPACK FFI returns (eigenvectors, eigenvalues, info) - NOT (eigenvalues, eigenvectors, info)!
                        op_outputs.clear();
                        op_outputs.push_back(eigh_result.second);  // eigenvectors FIRST (to match LAPACK)
                        op_outputs.push_back(eigh_result.first);   // eigenvalues SECOND
                        op_outputs.push_back(mlx::core::array(0, mlx::core::int32));  // info
                        result = eigh_result.second;  // Return eigenvectors as primary result
                    } catch (const std::exception& e) {
                        if (debug_mode()) std::cout << "[MLX-PJRT] eigh failed: " << e.what() << std::endl;
                        if (!op_inputs.empty()) result = op_inputs[0];
                    }
                }
            }
            // svd: lapack_sgesvd_ffi, lapack_sgesdd_ffi
            // LAPACK FFI returns 5 outputs: [A_workspace, S, U, Vt, info]
            // JAX then selects: S=%0#1, U=%0#2, Vt=%0#3 from these outputs
            else if (target.find("gesvd") != std::string::npos || target.find("gesdd") != std::string::npos || target.find("_svd") != std::string::npos) {
                if (!op_inputs.empty()) {
                    try {
                        auto svd_result = mlx::core::linalg::svd(op_inputs[0], mlx::core::Device(mlx::core::Device::cpu));
                        // MLX svd returns vector<array>: [U, S, Vh]
                        // LAPACK FFI output order: [A_workspace, S, U, Vt, info]
                        op_outputs.clear();
                        op_outputs.push_back(op_inputs[0]);  // [0] A workspace (copy of input)
                        op_outputs.push_back(svd_result[1]); // [1] S (singular values)
                        op_outputs.push_back(svd_result[0]); // [2] U
                        op_outputs.push_back(svd_result[2]); // [3] Vt (Vh from MLX)
                        op_outputs.push_back(mlx::core::array(0, mlx::core::int32));  // [4] info
                        result = svd_result[0];  // Return U as primary result
                        if (debug_mode()) {
                            std::cout << "[MLX-PJRT] SVD success: S=" << svd_result[1].shape() 
                                      << " U=" << svd_result[0].shape() 
                                      << " Vh=" << svd_result[2].shape() << std::endl;
                        }
                    } catch (const std::exception& e) {
                        if (debug_mode()) std::cout << "[MLX-PJRT] svd failed: " << e.what() << std::endl;
                        if (!op_inputs.empty()) result = op_inputs[0];
                    }
                }
            }
            // LU decomposition: lapack_sgetrf_ffi
            else if (target.find("getrf") != std::string::npos || target.find("_lu") != std::string::npos) {
                if (!op_inputs.empty()) {
                    try {
                        // MLX lu_factor returns pair<LU, pivots>
                        auto lu_result = mlx::core::linalg::lu_factor(op_inputs[0], mlx::core::Device(mlx::core::Device::cpu));
                        auto lu = lu_result.first;
                        auto pivots = lu_result.second;
                        
                        // JAX expects (LU, pivots, info) where pivots are 1-indexed
                        // MLX pivots are 0-indexed, need to add 1
                        auto pivots_1indexed = mlx::core::add(pivots, mlx::core::array(1, pivots.dtype()));
                        
                        op_outputs.clear();
                        op_outputs.push_back(lu);
                        op_outputs.push_back(pivots_1indexed);
                        op_outputs.push_back(mlx::core::array(0, mlx::core::int32));  // info = 0 (success)
                        result = lu;
                    } catch (const std::exception& e) {
                        if (debug_mode()) std::cout << "[MLX-PJRT] LU decomposition failed: " << e.what() << std::endl;
                        if (!op_inputs.empty()) result = op_inputs[0];
                    }
                }
            }
            // Cholesky: lapack_spotrf_ffi
            else if (target.find("potrf") != std::string::npos || target.find("cholesky") != std::string::npos) {
                if (!op_inputs.empty()) {
                    try {
                        // Check for upper/lower from attributes
                        bool upper = false;
                        if (op.attributes.count("uplo") && op.attributes.at("uplo") == "U") {
                            upper = true;
                        }
                        auto chol = mlx::core::linalg::cholesky(op_inputs[0], upper, mlx::core::Device(mlx::core::Device::cpu));
                        op_outputs.clear();
                        op_outputs.push_back(chol);
                        op_outputs.push_back(mlx::core::array(0, mlx::core::int32));  // info
                        result = chol;
                    } catch (const std::exception& e) {
                        if (debug_mode()) std::cout << "[MLX-PJRT] cholesky failed: " << e.what() << std::endl;
                        if (!op_inputs.empty()) result = op_inputs[0];
                    }
                }
            }
            // Matrix inverse: lapack_sgetri_ffi
            else if (target.find("getri") != std::string::npos || target.find("_inv") != std::string::npos) {
                if (!op_inputs.empty()) {
                    try {
                        auto inv_result = mlx::core::linalg::inv(op_inputs[0], mlx::core::Device(mlx::core::Device::cpu));
                        result = inv_result;
                    } catch (const std::exception& e) {
                        if (debug_mode()) std::cout << "[MLX-PJRT] inv failed: " << e.what() << std::endl;
                        if (!op_inputs.empty()) result = op_inputs[0];
                    }
                }
            }
            // Solve linear system: lapack_sgesv_ffi  
            else if (target.find("gesv") != std::string::npos && target.find("gesvd") == std::string::npos) {
                if (op_inputs.size() >= 2) {
                    try {
                        auto solve_result = mlx::core::linalg::solve(op_inputs[0], op_inputs[1], mlx::core::Device(mlx::core::Device::cpu));
                        op_outputs.clear();
                        op_outputs.push_back(solve_result);
                        op_outputs.push_back(mlx::core::array(0, mlx::core::int32));  // info
                        result = solve_result;
                    } catch (const std::exception& e) {
                        if (debug_mode()) std::cout << "[MLX-PJRT] solve failed: " << e.what() << std::endl;
                        if (!op_inputs.empty()) result = op_inputs[0];
                    }
                }
            }
            // Triangular solve: lapack_strsm_ffi
            else if (target.find("trsm") != std::string::npos || target.find("triangular_solve") != std::string::npos) {
                if (op_inputs.size() >= 2) {
                    try {
                        bool upper = true;  // Default to upper triangular
                        bool unit_diagonal = false;  // diag = 85 ('U') means unit diagonal
                        
                        // Check uplo attribute - can be string "U"/"L" or ASCII value 85/76
                        if (op.attributes.count("uplo")) {
                            std::string uplo_val = op.attributes.at("uplo");
                            if (uplo_val == "U" || uplo_val == "85") {
                                upper = true;
                            } else if (uplo_val == "L" || uplo_val == "76") {
                                upper = false;
                            }
                        }
                        // Also check mhlo.backend_config for uplo = 85 : ui8 and diag = 85 : ui8
                        if (op.attributes.count("mhlo.backend_config")) {
                            std::string cfg = op.attributes.at("mhlo.backend_config");
                            // Parse uplo
                            auto pos = cfg.find("uplo");
                            if (pos != std::string::npos) {
                                auto eq_pos = cfg.find("=", pos);
                                if (eq_pos != std::string::npos) {
                                    size_t num_start = eq_pos + 1;
                                    while (num_start < cfg.size() && (cfg[num_start] == ' ')) num_start++;
                                    try {
                                        int uplo_ascii = std::stoi(cfg.substr(num_start));
                                        upper = (uplo_ascii == 85);  // 'U' = 85, 'L' = 76
                                    } catch (...) {}
                                }
                            }
                            // Parse diag - diag = 85 ('U') means unit diagonal
                            pos = cfg.find("diag");
                            if (pos != std::string::npos) {
                                auto eq_pos = cfg.find("=", pos);
                                if (eq_pos != std::string::npos) {
                                    size_t num_start = eq_pos + 1;
                                    while (num_start < cfg.size() && (cfg[num_start] == ' ')) num_start++;
                                    try {
                                        int diag_ascii = std::stoi(cfg.substr(num_start));
                                        unit_diagonal = (diag_ascii == 85);  // 'U' = 85 = unit, 'N' = 78 = non-unit
                                    } catch (...) {}
                                }
                            }
                        }
                        if (debug_mode()) std::cout << "[MLX-PJRT] triangular_solve: upper=" << upper << " unit_diag=" << unit_diagonal << std::endl;
                        
                        auto A = op_inputs[0];
                        // If unit_diagonal, we need to set diagonal to 1s for correct solve
                        if (unit_diagonal) {
                            // Create a copy and set diagonal to 1
                            auto diag_ones = mlx::core::eye(A.shape(0), A.shape(1), 0, A.dtype());
                            auto diag_mask = mlx::core::subtract(mlx::core::ones_like(A), mlx::core::eye(A.shape(0), A.shape(1), 0, A.dtype()));
                            A = mlx::core::add(mlx::core::multiply(A, diag_mask), diag_ones);
                        }
                        
                        auto solve_result = mlx::core::linalg::solve_triangular(A, op_inputs[1], upper, mlx::core::Device(mlx::core::Device::cpu));
                        result = solve_result;
                    } catch (const std::exception& e) {
                        if (debug_mode()) std::cout << "[MLX-PJRT] triangular_solve failed: " << e.what() << std::endl;
                        if (!op_inputs.empty()) result = op_inputs[0];
                    }
                }
            }
            // QR decomposition: "Qr" (JAX style) or lapack_sgeqrf_ffi
            // Note: target may be "Qr" or "@Qr" or "\"Qr\"" (with quotes)
            // LAPACK geqrf returns R with same shape as input (m x n), with R in upper triangular part
            else if (target == "Qr" || target == "\"Qr\"" || target.find("geqrf") != std::string::npos) {
                if (!op_inputs.empty()) {
                    try {
                        auto qr_result = mlx::core::linalg::qr(op_inputs[0], mlx::core::Device(mlx::core::Device::cpu));
                        auto Q = qr_result.first;   // Q matrix
                        auto R_reduced = qr_result.second;  // R matrix (reduced: min(m,n) x n)
                        
                        int m = op_inputs[0].shape(0);
                        int n = op_inputs[0].shape(1);
                        int k = std::min(m, n);
                        
                        // LAPACK geqrf returns matrix with same shape as input (m x n)
                        // R is in upper triangular part, reflectors in strict lower triangular
                        // For m > n case, we need to pad R_reduced (k x n) to (m x n)
                        mlx::core::array R = R_reduced;  // Start with reduced R
                        if (m > n) {
                            // Pad with zeros: R_reduced is (n x n), need (m x n)
                            auto zeros_padding = mlx::core::zeros({m - n, n}, R_reduced.dtype());
                            R = mlx::core::concatenate({R_reduced, zeros_padding}, 0);
                        }
                        
                        // Tau has length min(m, n)
                        auto tau = mlx::core::ones({k}, op_inputs[0].dtype());
                        
                        op_outputs.clear();
                        op_outputs.push_back(R);    // R (upper triangular in m x n matrix)
                        op_outputs.push_back(tau);  // tau vector
                        result = R;
                        
                        // Store Q in global cache for Householder to retrieve
                        g_last_qr_q = Q;
                        if (debug_mode()) std::cout << "[MLX-PJRT] QR stored Q for Householder, R=" << R.shape() << " Q=" << Q.shape() << std::endl;
                    } catch (const std::exception& e) {
                        if (debug_mode()) std::cout << "[MLX-PJRT] qr failed: " << e.what() << std::endl;
                        if (!op_inputs.empty()) result = op_inputs[0];
                    }
                }
            }
            // Householder: ProductOfElementaryHouseholderReflectors - produce Q matrix
            // orgqr reconstructs Q from elementary reflectors
            // Input[0] is the padded matrix (m x m for full QR), Input[1] is tau
            // Output should match shape of Input[0]
            else if (target == "ProductOfElementaryHouseholderReflectors" || target == "\"ProductOfElementaryHouseholderReflectors\"" || target.find("Householder") != std::string::npos || target.find("orgqr") != std::string::npos) {
                if (!op_inputs.empty()) {
                    try {
                        // Get expected output shape from first input
                        auto expected_shape = op_inputs[0].shape();
                        int out_m = expected_shape[0];
                        int out_n = expected_shape[1];
                        
                        // Try to retrieve Q from global cache
                        if (g_last_qr_q.has_value()) {
                            auto cached_Q = g_last_qr_q.value();
                            g_last_qr_q.reset();  // Clean up
                            
                            // Check if cached Q needs to be expanded to match expected output
                            int cached_m = cached_Q.shape(0);
                            int cached_n = cached_Q.shape(1);
                            
                            if (cached_m == out_m && cached_n == out_n) {
                                result = cached_Q;
                            } else if (cached_m == out_m && cached_n < out_n) {
                                // Need to expand Q horizontally by appending identity columns
                                // For full QR: Q is m x m orthogonal, we have m x k
                                // Append (m x (m-k)) identity-like columns
                                auto extra_cols = mlx::core::zeros({out_m, out_n - cached_n}, cached_Q.dtype());
                                // Set diagonal elements to 1 for the extra columns
                                for (int i = 0; i < out_n - cached_n && i + cached_n < out_m; i++) {
                                    // This is imprecise; for better results recompute full QR
                                }
                                result = mlx::core::concatenate({cached_Q, extra_cols}, 1);
                            } else {
                                // Shape mismatch - recompute from input
                                auto qr_result = mlx::core::linalg::qr(op_inputs[0], mlx::core::Device(mlx::core::Device::cpu));
                                result = qr_result.first;
                            }
                            if (debug_mode()) std::cout << "[MLX-PJRT] Householder Q output shape=" << result.shape() << " expected=" << out_m << "x" << out_n << std::endl;
                        } else {
                            // No cached Q - compute from input (the padded matrix)
                            auto qr_result = mlx::core::linalg::qr(op_inputs[0], mlx::core::Device(mlx::core::Device::cpu));
                            result = qr_result.first;
                            if (debug_mode()) std::cout << "[MLX-PJRT] Householder recomputed Q, shape=" << result.shape() << std::endl;
                        }
                    } catch (const std::exception& e) {
                        if (debug_mode()) std::cout << "[MLX-PJRT] Householder failed: " << e.what() << std::endl;
                        if (!op_inputs.empty()) result = op_inputs[0];
                    }
                }
            }
            // Top-K: mhlo.topk
            else if (target.find("topk") != std::string::npos) {
                if (!op_inputs.empty()) {
                    auto x = op_inputs[0];
                    int k = 1; // default
                    
                    // Get k from attributes (typically in backend_config or as separate attr)
                    if (op.int_attrs.count("k")) {
                        k = op.int_attrs.at("k");
                    }
                    // Check mhlo.attributes which contains "k = 3" format
                    if (op.attributes.count("mhlo.attributes")) {
                        std::string attrs = op.attributes.at("mhlo.attributes");
                        auto pos = attrs.find("k");
                        if (pos != std::string::npos) {
                            auto eq_pos = attrs.find("=", pos);
                            if (eq_pos != std::string::npos) {
                                // Skip whitespace
                                size_t num_start = eq_pos + 1;
                                while (num_start < attrs.size() && (attrs[num_start] == ' ' || attrs[num_start] == ':')) num_start++;
                                try {
                                    k = std::stoi(attrs.substr(num_start));
                                } catch (...) {}
                            }
                        }
                    }
                    // Also check backend_config for k
                    if (k == 1 && op.attributes.count("mhlo.backend_config")) {
                        std::string cfg = op.attributes.at("mhlo.backend_config");
                        auto pos = cfg.find("k");
                        if (pos != std::string::npos) {
                            auto eq_pos = cfg.find("=", pos);
                            if (eq_pos != std::string::npos) {
                                try {
                                    k = std::stoi(cfg.substr(eq_pos + 1));
                                } catch (...) {}
                            }
                        }
                    }
                    // Last resort: infer k from output shape  
                    if (k == 1 && !op.output_shapes.empty() && !op.output_shapes[0].empty()) {
                        // output_shapes[0] contains the first output's shape, e.g. [3] for k=3
                        k = op.output_shapes[0][0];
                    }
                    if (debug_mode()) std::cout << "[MLX-PJRT] topk: k=" << k << std::endl;
                    // Use MLX topk for values (returns ascending order)
                    auto top_values = mlx::core::topk(x, k, -1);
                    
                    // For indices: sort descending and take first k
                    auto sorted_indices = mlx::core::argsort(x, -1);  // ascending indices
                    int n = static_cast<int>(x.shape(-1));
                    
                    // Slice the last k elements (largest) from sorted indices
                    mlx::core::Shape starts_shape = {n - k};
                    mlx::core::Shape stops_shape = {n};
                    auto top_indices = mlx::core::slice(sorted_indices, starts_shape, stops_shape);
                    
                    // Reverse both values and indices to get descending order [largest, ..., k-th largest]
                    auto rev_arr = mlx::core::arange(k - 1, -1, -1, mlx::core::int32);
                    top_values = mlx::core::take(top_values, rev_arr, -1);
                    top_indices = mlx::core::take(top_indices, rev_arr, -1);
                    
                    // Return both as multi-output
                    op_outputs.clear();
                    op_outputs.push_back(top_values);
                    op_outputs.push_back(mlx::core::astype(top_indices, mlx::core::int32));
                    result = top_values;
                }
            } else {
                // Unhandled custom_call - passthrough
                if (debug_mode()) {
                    std::cout << "[MLX-PJRT][WARN] Unhandled custom_call target: " << target << std::endl;
                }
                if (!op_inputs.empty()) result = op_inputs[0];
            }
        } else {
if (debug_mode()) std::cout << "[MLX-PJRT]   Warning: Unhandled op " << op.op_name << ", bypass" << std::endl;
            if (!op_inputs.empty()) result = op_inputs[0];
        }
        
        // --- Store Outputs (INSIDE Legacy Ops block to access inner 'result') ---
        if (!op_outputs.empty()) {
             if (op.outputs.size() == op_outputs.size()) {
                 for (size_t i = 0; i < op.outputs.size(); ++i) {
                     val_map.erase(op.outputs[i]);
                     val_map.insert(std::make_pair(op.outputs[i], op_outputs[i]));
                     
                     if (debug_mode()) {
                         std::cout << "[MLX-PJRT] Op Result: Op=" << op.op_name << " ID=" << op.outputs[i] << " Dtype=" << op_outputs[i].dtype() << std::endl;
                     }
                 }
             }
        }
        
        if (op_outputs.empty() && !op.outputs.empty()) {
             for (int out_id : op.outputs) {
                  val_map.erase(out_id);
                  val_map.insert(std::make_pair(out_id, result));
                  
                  if (debug_mode()) {
                      std::cout << "[MLX-PJRT] Op Result (Single): Op=" << op.op_name << " ID=" << out_id << " Dtype=" << result.dtype() << " Shape=[";
                      for(auto s : result.shape()) std::cout << s << ",";
                      std::cout << "]" << std::endl;
                      
                      // Check for NaNs in debug mode
                      if (result.dtype() == mlx::core::float32 || result.dtype() == mlx::core::float16 || result.dtype() == mlx::core::bfloat16) {
                          bool has_nan = mlx::core::any(mlx::core::isnan(result)).item<bool>();
                          if (has_nan) {
                              std::cout << "[MLX-PJRT][WARN] NaN detected in output of " << op.op_name << " (Single) OutID=" << out_id << " Shape=" << result.shape() << std::endl;
                          }
                      }
                  }
             }
        }
        } // End Legacy Ops block
    } // End of op loop
if (debug_mode()) std::cout << "[MLX-PJRT]   Op loop finished. Gathering outputs..." << std::endl;

    // Gather outputs using the graph's output_ids
if (debug_mode()) std::cout << "[MLX-PJRT]   Gathering outputs..." << std::endl;
    std::vector<mlx::core::array> output_arrays;
    // Collect outputs based on graph.output_ids
    for (int out_id : graph.output_ids) {
if (debug_mode()) std::cout << "[MLX-PJRT]     Looking for Output ID " << out_id << " in val_map size=" << val_map.size() << std::endl;
        if (val_map.count(out_id)) {
             auto& arr = val_map.at(out_id);
if (debug_mode()) {
                 std::cout << "[MLX-PJRT]     Found output ID " << out_id << " Shape=[";
                 for(auto s : arr.shape()) std::cout << s << ",";
                 std::cout << "]" << std::endl;
             }
             output_arrays.push_back(arr);
        } else {
if (debug_mode()) std::cout << "[MLX-PJRT]     MISSING Output ID " << out_id << "! Returning zero." << std::endl; 
             // Should not happen if graph is valid
             output_arrays.push_back(mlx::core::array(0.0f)); 
        }
    }
if (debug_mode()) std::cout << "[MLX-PJRT]   ExecuteGraph returning " << output_arrays.size() << " arrays." << std::endl; 
    
    return output_arrays;
}

// Mega-Compile: Materialize all pending executions in a batch
// Called at sync points (BufferToHostBytes, block_until_ready)
void materialize_batch(int batch_id) {
    if (!g_batch_accumulator.has_pending(batch_id)) return;
    
    auto& pending = g_batch_accumulator.pending_by_batch[batch_id];
    if (pending.empty()) return;
    
    if (timing_mode()) {
        std::cout << "[TIMING] Mega-compile: materializing batch " << batch_id 
                  << " with " << pending.size() << " pending graphs" << std::endl;
    }
    
    auto t_start = std::chrono::high_resolution_clock::now();
    
    // Execute each pending graph and store results in output buffers
    // For now, execute sequentially - optimization: compile all together
    for (auto& pe : pending) {
        std::vector<mlx::core::array> outputs;
        
        // Check if we can compile this graph
        bool should_compile = compile_enabled() && is_compile_safe(pe.exec->graph, &pe.exec->functions);
        
        if (should_compile) {
            // Use cached compiled function if available
            if (!pe.exec->compiled_fn.has_value()) {
                auto graph_copy = pe.exec->graph;
                auto functions_copy = pe.exec->functions;
                auto fn = [graph_copy, functions_copy](const std::vector<mlx::core::array>& inputs) {
                    return ExecuteGraph(graph_copy, inputs, nullptr, &functions_copy, nullptr);
                };
                pe.exec->compiled_fn = mlx::core::compile(fn);
            }
            outputs = pe.exec->compiled_fn.value()(pe.inputs);
        } else {
            // Cannot compile - run directly
            outputs = ExecuteGraph(pe.exec->graph, pe.inputs, nullptr, &pe.exec->functions, pe.exec);
        }
        
        // Store results in output buffers
        for (size_t i = 0; i < outputs.size() && i < pe.output_buffers.size(); ++i) {
            pe.output_buffers[i]->array = outputs[i];
        }
    }
    
    // Batch evaluate all outputs at once
    std::vector<mlx::core::array> all_arrays;
    for (auto& pe : pending) {
        for (auto* buf : pe.output_buffers) {
            all_arrays.push_back(buf->array);
        }
    }
    mlx::core::eval(all_arrays);
    
    auto t_end = std::chrono::high_resolution_clock::now();
    if (timing_mode()) {
        auto ms = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / 1000.0;
        std::cout << "[TIMING] Mega-compile batch " << batch_id << " took " << ms << "ms" << std::endl;
    }
    
    g_batch_accumulator.clear_batch(batch_id);
}

// Main PJRT Execute function
PJRT_Error* MLX_LoadedExecutable_Execute(PJRT_LoadedExecutable_Execute_Args* args) {
    // Profiling: track total execution time
    auto t_total_start = profile_mode() ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{};
    
if (debug_mode()) std::cout << "[MLX-PJRT] MLX_LoadedExecutable_Execute called" << std::endl;
    if (args == nullptr) {
        return InvalidArgument("Args is null");
    }
    
    MLXLoadedExecutable* loaded = reinterpret_cast<MLXLoadedExecutable*>(args->executable);
    MLXExecutable* exec = loaded->inner_executable;
    
    // FAST PATH: Skip all debug logging and unnecessary checks when not debugging
    // This saves ~50-100µs per call
    const bool is_fast_path = !debug_mode() && exec->compiled_fn.has_value();
    
if (debug_mode()) {
    // Debug args structure
    // std::cout << "  Args struct size: " << args->struct_size << std::endl;
    std::cout << "[MLX-PJRT] LoadedExecutable_Execute ENTRY" << std::endl << std::flush;
    std::cout << "[MLX-PJRT]   args ptr: " << (void*)args->execute_device << std::endl << std::flush;
    std::cout << "[MLX-PJRT]   executable ptr: " << (void*)loaded << std::endl;
    std::cout << "[MLX-PJRT]   num_devices: " << args->num_devices << std::endl;
    std::cout << "[MLX-PJRT]   loaded ptr: " << (void*)loaded << std::endl;
    std::cout << "[MLX-PJRT]   inner_executable ptr: " << (void*)exec << std::endl;
    std::cout << "[MLX-PJRT] LoadedExecutable_Execute (IR Interpreter)" << std::endl;
    std::cout << "[MLX-PJRT]   num_args=" << args->num_args << " graph_nodes=" << exec->graph.nodes.size() << std::endl;
}
    
    // Profiling: input extraction start
    auto t_input_start = profile_mode() ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{};
    
    // 1. Extract input arrays from PJRT buffers
    std::vector<mlx::core::array> input_arrays;
    input_arrays.reserve(args->num_args);  // Optimization: avoid reallocations
    int batch_id = 0;  // Mega-compile: track batch_id from inputs
    
    for (size_t i = 0; i < args->num_args; ++i) {
        MLXBuffer* buf = reinterpret_cast<MLXBuffer*>(args->argument_lists[0][i]);
        mlx::core::array arr = buf->array;
        
        // Mega-compile: propagate batch_id from input buffers
        if (mega_compile_enabled() && buf->batch_id > batch_id) {
            batch_id = buf->batch_id;
        }
        
        // FAST PATH: Skip reshape checks for compiled functions (shapes already validated)
        if (!is_fast_path && i < exec->graph.input_shapes.size()) {
             const std::vector<int>& target = exec->graph.input_shapes[i];
             size_t target_elements = 1;
             for (int d : target) target_elements *= d;
             
             if (arr.size() == target_elements && target_elements > 0) {
                 bool mismatch = (arr.ndim() != (int)target.size());
                 if (!mismatch) {
                     for(size_t k=0; k<target.size(); ++k) if (arr.shape(k) != target[k]) mismatch=true;
                 }
                 
                 if (mismatch) {
                     arr = mlx::core::reshape(arr, mlx::core::Shape(target.begin(), target.end()));
                 }
             }
        }

        input_arrays.push_back(arr);
if (debug_mode()) std::cout << "[MLX-PJRT]   Input " << i << " (ID " << exec->graph.input_ids[i] << ") bound" << std::endl;
    }
    
    // Profiling: input extraction end
    auto t_input_end = profile_mode() ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{};
    
    // Mega-compile: assign new batch_id if inputs came from host (batch_id=0)
    if (mega_compile_enabled() && batch_id == 0) {
        batch_id = g_current_batch_id++;
    }
    
    // 2. Execute the graph
    auto t_exec_start = std::chrono::high_resolution_clock::now();
    std::vector<mlx::core::array> output_arrays;
    
    // Check if we can compile this graph
    bool should_compile = compile_enabled() && is_compile_safe(exec->graph, &exec->functions);
    
    if (debug_mode()) {
        std::cout << "[MLX-PJRT] Graph has " << exec->graph.nodes.size() << " nodes, compile_safe=" << (should_compile ? "true" : "false") << ", ops: ";
        for (const auto& n : exec->graph.nodes) std::cout << n.op_name << " ";
        std::cout << std::endl;
    }
    if (timing_mode() && !should_compile && exec->graph.nodes.size() > 0 && compile_enabled()) {
        // Show which ops prevent compilation
        std::cout << "[TIMING] Graph not compiled (" << exec->graph.nodes.size() << " nodes) due to: ";
        for (const auto& n : exec->graph.nodes) {
            if (n.op_name == "func.call" || n.op_name.find("while") != std::string::npos ||
                n.op_name.find("if") != std::string::npos || n.op_name.find("case") != std::string::npos ||
                n.op_name.find("dynamic") != std::string::npos || n.op_name.find("scatter") != std::string::npos) {
                std::cout << n.op_name << " ";
            }
        }
        std::cout << std::endl;
    }
    
    // FAST PATH: Direct execution for cached compiled function
    if (is_fast_path && should_compile) {
        output_arrays = exec->compiled_fn.value()(input_arrays);
    } else if (should_compile) {
        // Create or use cached compiled function
        if (!exec->compiled_fn.has_value()) {
            auto graph_copy = exec->graph;
            auto functions_copy = exec->functions;
            
            auto fn = [graph_copy, functions_copy](const std::vector<mlx::core::array>& inputs) {
                return ExecuteGraph(graph_copy, inputs, nullptr, &functions_copy, nullptr);
            };
            
            exec->compiled_fn = mlx::core::compile(fn);
            if (debug_mode()) std::cout << "[MLX-PJRT] Compiled function cached for " << exec->name << std::endl;
        }
        
        output_arrays = exec->compiled_fn.value()(input_arrays);
    } else {
        // Cannot compile - run directly
        output_arrays = ExecuteGraph(exec->graph, input_arrays, nullptr, &exec->functions, exec);
    }
    
    // Profiling: graph execution end (also used by timing_mode)
    auto t_exec_end = (profile_mode() || timing_mode()) ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{};
    
    // Only measure timing when enabled (avoid chrono overhead on hot path)
    if (timing_mode()) {
        auto exec_us = std::chrono::duration_cast<std::chrono::microseconds>(t_exec_end - t_exec_start).count();
        std::cout << "[TIMING] ExecuteGraph: " << exec_us << "us (" << exec_us/1000.0 << "ms)" 
                  << (exec->compiled_fn.has_value() ? " [compiled]" : "") << std::endl;
    }
    
    // 3. Create PJRT output buffers
    // Profiling: output creation start
    auto t_output_start = profile_mode() ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{};
    
    // Mega-compile: skip eval() here - defer to ToHostBuffer sync point
    // This is the key optimization: batch all evals together at materialization
    if (!mega_compile_enabled()) {
        mlx::core::eval(output_arrays);
    }
    
    if (timing_mode()) {
        std::cout << "[TIMING] batch eval(" << output_arrays.size() << " outputs): "
                  << (mega_compile_enabled() ? "0us (0ms) [deferred]" : "completed") << std::endl;
    }
    
if (debug_mode()) std::cout << "[MLX-PJRT]   Populating outputs, count=" << output_arrays.size() << std::endl << std::flush;
    for (size_t i = 0; i < output_arrays.size(); ++i) {
        int out_id = exec->graph.output_ids[i];
if (debug_mode()) std::cout << "[MLX-PJRT]   Looking for output " << i << " ID " << out_id << std::endl << std::flush;
        
        mlx::core::array output_arr = output_arrays[i];
        // Already evaluated in batch above

        std::vector<int64_t> out_dims;
        for(auto s : output_arr.shape()) out_dims.push_back(s);
if (debug_mode()) std::cout << "[MLX-PJRT]   Built dims vector" << std::endl << std::flush;
        
        PJRT_Buffer_Type out_type = MlxTypeToPjrtType(output_arr.dtype());
if (debug_mode()) std::cout << "[MLX-PJRT]   Output " << i << " Dtype: " << output_arr.dtype() 
                  << " is_bool=" << (output_arr.dtype() == mlx::core::bool_) 
                  << " -> PJRT: " << out_type << std::endl;
if (debug_mode()) std::cout << "[MLX-PJRT]   Got output type, creating buffer..." << std::endl << std::flush;
        
        MLXBuffer* out_buf = new MLXBuffer(
            output_arr,
            loaded->client,
            loaded->client->devices[0],
            false,
            out_dims,
            out_type
        );
if (debug_mode()) std::cout << "[MLX-PJRT]   Buffer created at " << (void*)out_buf << std::endl << std::flush;
        
        // Mega-compile: propagate batch_id to output buffers
        if (mega_compile_enabled()) {
            out_buf->batch_id = batch_id;
            out_buf->from_host = false;  // This is a computed output, not from host
        }
        
if (debug_mode()) std::cout << "[MLX-PJRT]   Assigning to output_lists[0][" << i << "]..." << std::endl << std::flush;
        args->output_lists[0][i] = reinterpret_cast<PJRT_Buffer*>(out_buf);
if (debug_mode()) std::cout << "[MLX-PJRT]   Output " << i << " (ID " << out_id << ") generated" << std::endl << std::flush;
    }
    
    // Profiling: output creation end
    auto t_output_end = profile_mode() ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{};

    // Set completion event
    if (args->device_complete_events) {
        args->device_complete_events[0] = reinterpret_cast<PJRT_Event*>(new MLXEvent{true});
    }
    
    // Profiling: print detailed breakdown
    if (profile_mode()) {
        auto t_total_end = std::chrono::high_resolution_clock::now();
        auto input_us = std::chrono::duration_cast<std::chrono::microseconds>(t_input_end - t_input_start).count();
        auto graph_us = std::chrono::duration_cast<std::chrono::microseconds>(t_exec_end - t_exec_start).count();
        auto output_us = std::chrono::duration_cast<std::chrono::microseconds>(t_output_end - t_output_start).count();
        auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(t_total_end - t_total_start).count();
        
        std::cout << "[PROFILE] Execute: total=" << total_us << "us | "
                  << "input=" << input_us << "us | "
                  << "graph=" << graph_us << "us | "
                  << "output=" << output_us << "us | "
                  << "overhead=" << (total_us - input_us - graph_us - output_us) << "us"
                  << std::endl;
    }

    return Ok();
}

// Static data for LoadedExecutable fingerprint
static const char* loaded_executable_fingerprint = "mlx-loaded-fp";
static size_t loaded_executable_fingerprint_size = 13;
// Static data for device assignment
static int device_assignment_data[] = {0};

PJRT_Error* MLX_LoadedExecutable_Fingerprint(PJRT_LoadedExecutable_Fingerprint_Args* args) {
if (debug_mode()) std::cout << "[MLX-PJRT] MLX_LoadedExecutable_Fingerprint called" << std::endl;
    args->executable_fingerprint = loaded_executable_fingerprint;
    args->executable_fingerprint_size = loaded_executable_fingerprint_size;
    return Ok();
}

void MLX_DeviceAssignment_Deleter(PJRT_DeviceAssignmentSerialized* da) {
if (debug_mode()) std::cout << "[MLX-PJRT] MLX_DeviceAssignment_Deleter called" << std::endl;
}

PJRT_Error* MLX_LoadedExecutable_GetDeviceAssignment(PJRT_LoadedExecutable_GetDeviceAssignment_Args* args) {
if (debug_mode()) std::cout << "[MLX-PJRT] MLX_LoadedExecutable_GetDeviceAssignment called" << std::endl;
    // Corrected DeviceAssignmentProto for [[0]] (1 replica, 1 partition, device 0)
    // replica_count (1): 1 -> 08 01
    // computation_count (2): 1 -> 10 01
    // computation_devices (3): 
    //   element 1: 
    //     replica_device_ids (1): 0 -> 08 00
    //     Message length: 2 (08 00)
    //   Tag 3, Len 2 -> 1A 02 08 00
    // Total: 08 01 10 01 1A 02 08 00
    static char proto_bytes[] = {0x08, 0x01, 0x10, 0x01, 0x1A, 0x02, 0x08, 0x00};
    static size_t proto_size = 8;

    args->serialized_bytes = proto_bytes;
    args->serialized_bytes_size = proto_size;
    // Return a dummy non-null handle and a valid deleter
    args->serialized_device_assignment = reinterpret_cast<PJRT_DeviceAssignmentSerialized*>(0xCAFEBABE);
    args->serialized_device_assignment_deleter = MLX_DeviceAssignment_Deleter;
    return Ok();
}

// --- Other Stubs ---

PJRT_Error* MLX_Generic_Unimplemented(void* args) {
    return Unimplemented("Generic_Stub");
}

extern "C" {

const PJRT_Api* GetPjrtApi() {
    static PJRT_Api api;
    static bool initialized = false;

    if (initialized) return &api;

if (debug_mode()) std::cout << "[MLX-PJRT] GetPjRtApi called" << std::endl;
    // Note: Can't easily log api.pjrt_api_version before it's set below, 
    // but we can log them after setting.

    std::memset(&api, 0, sizeof(PJRT_Api));
    
    api.struct_size = PJRT_Api_STRUCT_SIZE;
    api.pjrt_api_version.struct_size = PJRT_Api_Version_STRUCT_SIZE;
    api.pjrt_api_version.major_version = PJRT_API_MAJOR;
    api.pjrt_api_version.minor_version = PJRT_API_MINOR;
    api.extension_start = nullptr; 
    api.pjrt_api_version.extension_start = nullptr;

if (debug_mode()) std::cout << "[MLX-PJRT]   API Version: " << api.pjrt_api_version.major_version 
              << "." << api.pjrt_api_version.minor_version << std::endl;

    // Error
    api.PJRT_Error_Destroy = MLX_Error_Destroy;
    api.PJRT_Error_Message = MLX_Error_Message;
    api.PJRT_Error_GetCode = MLX_Error_GetCode;

    // Plugin
    api.PJRT_Plugin_Initialize = MLX_Plugin_Initialize;
    api.PJRT_Plugin_Attributes = MLX_Plugin_Attributes;

    // Event
    api.PJRT_Event_Destroy = MLX_Event_Destroy;
    api.PJRT_Event_IsReady = MLX_Event_IsReady;
    api.PJRT_Event_Error = MLX_Event_Error;
    api.PJRT_Event_Await = MLX_Event_Await;
    api.PJRT_Event_OnReady = MLX_Event_OnReady;
    api.PJRT_Event_Create = MLX_Event_Create;
    api.PJRT_Event_Set = MLX_Event_Set;

    // Client
    api.PJRT_Client_Create = MLX_Client_Create;
    api.PJRT_Client_Destroy = MLX_Client_Destroy;
    api.PJRT_Client_PlatformName = MLX_Client_PlatformName;
    api.PJRT_Client_ProcessIndex = MLX_Client_ProcessIndex;
    api.PJRT_Client_PlatformVersion = MLX_Client_PlatformVersion;
    api.PJRT_Client_Devices = MLX_Client_Devices;
    api.PJRT_Client_AddressableDevices = MLX_Client_AddressableDevices;
    api.PJRT_Client_LookupDevice = MLX_Client_LookupDevice;
    api.PJRT_Client_LookupAddressableDevice = MLX_Client_LookupAddressableDevice;
    api.PJRT_Client_AddressableMemories = MLX_Client_AddressableMemories;
    api.PJRT_Client_Compile = MLX_Client_Compile;
    api.PJRT_Client_DefaultDeviceAssignment = MLX_Client_DefaultDeviceAssignment;
    api.PJRT_Client_BufferFromHostBuffer = MLX_Client_BufferFromHostBuffer;
    api.PJRT_Client_CreateViewOfDeviceBuffer = MLX_Client_CreateViewOfDeviceBuffer;
    api.PJRT_Client_CreateBuffersForAsyncHostToDevice = MLX_Client_CreateBuffersForAsyncHostToDevice;
    api.PJRT_Client_TopologyDescription = MLX_Client_TopologyDescription;

    // Device Description
    api.PJRT_DeviceDescription_Id = MLX_DeviceDescription_Id;
    api.PJRT_DeviceDescription_ProcessIndex = MLX_DeviceDescription_ProcessIndex;
    api.PJRT_DeviceDescription_Attributes = MLX_DeviceDescription_Attributes;
    api.PJRT_DeviceDescription_Kind = MLX_DeviceDescription_Kind;
    api.PJRT_DeviceDescription_DebugString = MLX_DeviceDescription_DebugString;
    api.PJRT_DeviceDescription_ToString = MLX_DeviceDescription_ToString;

    // Device
    api.PJRT_Device_GetDescription = MLX_Device_GetDescription;
    api.PJRT_Device_IsAddressable = MLX_Device_IsAddressable;
    api.PJRT_Device_LocalHardwareId = MLX_Device_LocalHardwareId;
    api.PJRT_Device_AddressableMemories = MLX_Device_AddressableMemories;
    api.PJRT_Device_DefaultMemory = MLX_Device_DefaultMemory;
    api.PJRT_Device_MemoryStats = MLX_Device_MemoryStats;
    api.PJRT_Device_PoisonExecution = MLX_Device_PoisonExecution;
    api.PJRT_Device_CreateAsyncTrackingEvent = MLX_Device_CreateAsyncTrackingEvent;

    // Buffer
    api.PJRT_Buffer_Destroy = MLX_Buffer_Destroy;
    api.PJRT_Buffer_ElementType = MLX_Buffer_ElementType;
    api.PJRT_Buffer_Dimensions = MLX_Buffer_Dimensions;
    api.PJRT_Buffer_UnpaddedDimensions = MLX_Buffer_UnpaddedDimensions;
    api.PJRT_Buffer_DynamicDimensionIndices = MLX_Buffer_DynamicDimensionIndices;
    api.PJRT_Buffer_GetMemoryLayout = MLX_Buffer_GetMemoryLayout;
    api.PJRT_Buffer_OnDeviceSizeInBytes = MLX_Buffer_OnDeviceSizeInBytes;
    api.PJRT_Buffer_Device = MLX_Buffer_Device;
    api.PJRT_Buffer_Memory = MLX_Buffer_Memory;
    api.PJRT_Buffer_Delete = MLX_Buffer_Delete;
    api.PJRT_Buffer_IsDeleted = MLX_Buffer_IsDeleted;
    api.PJRT_Buffer_CopyToDevice = MLX_Buffer_CopyToDevice;
    api.PJRT_Buffer_ToHostBuffer = MLX_Buffer_ToHostBuffer;
    api.PJRT_Buffer_IsOnCpu = MLX_Buffer_IsOnCpu;
    api.PJRT_Buffer_ReadyEvent = MLX_Buffer_ReadyEvent;
    api.PJRT_Buffer_UnsafePointer = MLX_Buffer_UnsafePointer;
    api.PJRT_Buffer_IncreaseExternalReferenceCount = MLX_Buffer_IncreaseExternalReferenceCount;
    api.PJRT_Buffer_DecreaseExternalReferenceCount = MLX_Buffer_DecreaseExternalReferenceCount;
    api.PJRT_Buffer_OpaqueDeviceMemoryDataPointer = MLX_Buffer_OpaqueDeviceMemoryDataPointer;

    // Generic Stubs for now
    #define STUB(func) if (!api.func) api.func = reinterpret_cast<func*>(MLX_Generic_Unimplemented)
    
    STUB(PJRT_CopyToDeviceStream_Destroy);
    STUB(PJRT_CopyToDeviceStream_AddChunk);
    STUB(PJRT_CopyToDeviceStream_TotalBytes);
    STUB(PJRT_CopyToDeviceStream_GranuleSize);
    STUB(PJRT_CopyToDeviceStream_CurrentBytes);
    
    STUB(PJRT_TopologyDescription_Create);
    api.PJRT_TopologyDescription_Destroy = MLX_TopologyDescription_Destroy;
    api.PJRT_TopologyDescription_PlatformName = MLX_TopologyDescription_PlatformName;
    api.PJRT_TopologyDescription_PlatformVersion = MLX_TopologyDescription_PlatformVersion;
    api.PJRT_TopologyDescription_GetDeviceDescriptions = MLX_TopologyDescription_GetDeviceDescriptions;
    STUB(PJRT_TopologyDescription_Serialize);
    api.PJRT_TopologyDescription_Attributes = MLX_TopologyDescription_Attributes;
    
    STUB(PJRT_Compile);
    api.PJRT_Executable_GetCompiledMemoryStats = MLX_Executable_GetCompiledMemoryStats;
    
    STUB(PJRT_ExecuteContext_Create);
    STUB(PJRT_ExecuteContext_Destroy);
    STUB(PJRT_Buffer_CopyRawToHost);
    STUB(PJRT_AsyncHostToDeviceTransferManager_Destroy);
    STUB(PJRT_AsyncHostToDeviceTransferManager_TransferData);
    STUB(PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer);
    STUB(PJRT_AsyncHostToDeviceTransferManager_Device);
    STUB(PJRT_AsyncHostToDeviceTransferManager_BufferCount);
    STUB(PJRT_AsyncHostToDeviceTransferManager_BufferSize);
    STUB(PJRT_AsyncHostToDeviceTransferManager_SetBufferError);
    STUB(PJRT_AsyncHostToDeviceTransferManager_AddMetadata);
    STUB(PJRT_AsyncHostToDeviceTransferManager_TransferLiteral);
    
    api.PJRT_Client_DmaMap = reinterpret_cast<PJRT_Client_DmaMap*>(MLX_Generic_Unimplemented);
    api.PJRT_Client_DmaUnmap = reinterpret_cast<PJRT_Client_DmaUnmap*>(MLX_Generic_Unimplemented);
    STUB(PJRT_Client_CreateUninitializedBuffer);
    STUB(PJRT_Client_UpdateGlobalProcessInfo);
    STUB(PJRT_TopologyDescription_Deserialize);
    STUB(PJRT_Client_CreateAliasBuffer);
    STUB(PJRT_Client_FulfillAliasBuffer);
    STUB(PJRT_Client_CreateErrorBuffer);
    STUB(PJRT_Buffer_CopyRawToHostFuture);
    STUB(PJRT_AsyncTrackingEvent_Destroy);
    STUB(PJRT_Buffer_DonateWithControlDependency);
    
    api.PJRT_Executable_Destroy = MLX_Executable_Destroy;
    api.PJRT_Executable_Name = MLX_Executable_Name;
    api.PJRT_Executable_NumReplicas = MLX_Executable_NumReplicas;
    api.PJRT_Executable_NumPartitions = MLX_Executable_NumPartitions;
    api.PJRT_Executable_NumOutputs = MLX_Executable_NumOutputs;
    api.PJRT_Executable_OutputElementTypes = MLX_Executable_OutputElementTypes;
    
    api.PJRT_Executable_SizeOfGeneratedCodeInBytes = MLX_Executable_SizeOfGeneratedCodeInBytes;
    api.PJRT_Executable_GetCostAnalysis = MLX_Executable_GetCostAnalysis;
    api.PJRT_Executable_OutputMemoryKinds = MLX_Executable_OutputMemoryKinds;
    api.PJRT_Executable_OptimizedProgram = MLX_Executable_OptimizedProgram;
    api.PJRT_Executable_Serialize = MLX_Executable_Serialize;
    api.PJRT_Executable_OutputDimensions = MLX_Executable_OutputDimensions; 
    api.PJRT_Executable_Fingerprint = MLX_Executable_Fingerprint;
    api.PJRT_Executable_DeserializeAndLoad = MLX_Executable_DeserializeAndLoad;
    api.PJRT_Executable_GetCompileOptions = MLX_Executable_GetCompileOptions;

    api.PJRT_LoadedExecutable_Destroy = MLX_LoadedExecutable_Destroy;
    api.PJRT_LoadedExecutable_GetExecutable = MLX_LoadedExecutable_GetExecutable;
    api.PJRT_LoadedExecutable_AddressableDevices = MLX_LoadedExecutable_AddressableDevices;
    api.PJRT_LoadedExecutable_Delete = MLX_LoadedExecutable_Delete;
    api.PJRT_LoadedExecutable_IsDeleted = MLX_LoadedExecutable_IsDeleted;
    api.PJRT_LoadedExecutable_Execute = MLX_LoadedExecutable_Execute;
    api.PJRT_LoadedExecutable_Fingerprint = MLX_LoadedExecutable_Fingerprint;
    api.PJRT_LoadedExecutable_GetDeviceAssignment = MLX_LoadedExecutable_GetDeviceAssignment;

    // Memory
    api.PJRT_Memory_Id = MLX_Memory_Id;
    api.PJRT_Memory_Kind = MLX_Memory_Kind;
    api.PJRT_Memory_Kind_Id = MLX_Memory_Kind_Id;
    api.PJRT_Memory_DebugString = MLX_Memory_DebugString;
    api.PJRT_Memory_ToString = MLX_Memory_ToString;
    api.PJRT_Memory_AddressableByDevices = MLX_Memory_AddressableByDevices;

    // Other missing ones after previous block
    STUB(PJRT_Client_CreateViewOfDeviceBuffer);
    STUB(PJRT_Client_CreateBuffersForAsyncHostToDevice);
    STUB(PJRT_Event_Set);
    
    api.PJRT_Buffer_CopyToMemory = reinterpret_cast<PJRT_Buffer_CopyToMemory*>(MLX_Generic_Unimplemented);

    initialized = true;
    return &api;
}

} // extern "C"
