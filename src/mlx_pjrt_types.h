// mlx_pjrt_types.h - Type definitions for JAX-MLX PJRT plugin
// This header contains all type definitions, enums, and structs used by the plugin.

#ifndef MLX_PJRT_TYPES_H
#define MLX_PJRT_TYPES_H

#include <iostream>
#include <vector>
#include <cstring>
#include <string>
#include <map>
#include <numeric>
#include <unordered_map>
#include <optional>
#include <atomic>
#include <memory>

#include "xla/pjrt/c/pjrt_c_api.h"
#include <mlx/mlx.h>

// =============================================================================
// OpType Enum - Fast operation dispatch (MLX_OPTYPE_DISPATCH=1)
// =============================================================================

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

// Op name to OpType lookup map
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

// =============================================================================
// Error Handling
// =============================================================================

struct MLXError {
    PJRT_Error_Code code;
    std::string message;
};

inline PJRT_Error* Ok() { return nullptr; }

inline bool debug_mode() {
    static int cached = -1;
    if (cached == -1) cached = (getenv("MLX_PJRT_DEBUG") != nullptr) ? 1 : 0;
    return cached == 1;
}

inline bool constant_cache_enabled() {
    static int cached = -1;
    if (cached == -1) cached = (getenv("MLX_CONSTANT_CACHE") != nullptr) ? 1 : 0;
    return cached == 1;
}

inline bool optype_dispatch_enabled() {
    static int cached = -1;
    if (cached == -1) cached = (getenv("MLX_OPTYPE_DISPATCH") != nullptr) ? 1 : 0;
    return cached == 1;
}

inline PJRT_Error* Error(PJRT_Error_Code code, const char* msg) {
    MLXError* e = new MLXError{code, msg};
    return reinterpret_cast<PJRT_Error*>(e);
}

inline PJRT_Error* Unimplemented(const std::string& name) {
    return Error(PJRT_Error_Code_UNIMPLEMENTED, (name + " not implemented").c_str());
}

inline PJRT_Error* InvalidArgument(const std::string& msg) {
    return Error(PJRT_Error_Code_INVALID_ARGUMENT, msg.c_str());
}

// =============================================================================
// Type Conversion Helpers
// =============================================================================

inline PJRT_Buffer_Type MlxTypeToPjrtType(mlx::core::Dtype dtype) {
    switch (dtype) {
        case mlx::core::bool_: return PJRT_Buffer_Type_PRED;
        case mlx::core::int8: return PJRT_Buffer_Type_S8;
        case mlx::core::int16: return PJRT_Buffer_Type_S16;
        case mlx::core::int32: return PJRT_Buffer_Type_S32;
        case mlx::core::int64: return PJRT_Buffer_Type_S64;
        case mlx::core::uint8: return PJRT_Buffer_Type_U8;
        case mlx::core::uint16: return PJRT_Buffer_Type_U16;
        case mlx::core::uint32: return PJRT_Buffer_Type_U32;
        case mlx::core::uint64: return PJRT_Buffer_Type_U64;
        case mlx::core::float16: return PJRT_Buffer_Type_F16;
        case mlx::core::bfloat16: return PJRT_Buffer_Type_BF16;
        case mlx::core::float32: return PJRT_Buffer_Type_F32;
        case mlx::core::complex64: return PJRT_Buffer_Type_C64;
        default: return PJRT_Buffer_Type_INVALID;
    }
}

inline mlx::core::Dtype PjrtTypeToMlxType(PJRT_Buffer_Type type) {
    switch (type) {
        case PJRT_Buffer_Type_PRED: return mlx::core::bool_;
        case PJRT_Buffer_Type_S8: return mlx::core::int8;
        case PJRT_Buffer_Type_S16: return mlx::core::int16;
        case PJRT_Buffer_Type_S32: return mlx::core::int32;
        case PJRT_Buffer_Type_S64: return mlx::core::int64;
        case PJRT_Buffer_Type_U8: return mlx::core::uint8;
        case PJRT_Buffer_Type_U16: return mlx::core::uint16;
        case PJRT_Buffer_Type_U32: return mlx::core::uint32;
        case PJRT_Buffer_Type_U64: return mlx::core::uint64;
        case PJRT_Buffer_Type_F16: return mlx::core::float16;
        case PJRT_Buffer_Type_BF16: return mlx::core::bfloat16;
        case PJRT_Buffer_Type_F32: return mlx::core::float32;
        case PJRT_Buffer_Type_C64: return mlx::core::complex64;
        default: return mlx::core::float32;
    }
}

// =============================================================================
// Internal Structures
// =============================================================================

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
    
    MLXBuffer() : array(mlx::core::array(0.0f)), client(nullptr), device(nullptr), is_deleted(false), type(PJRT_Buffer_Type_INVALID), ref_count(1) {}
    MLXBuffer(mlx::core::array a, MLXClient* c, MLXDevice* d, bool del, std::vector<int64_t> di, PJRT_Buffer_Type t) 
        : array(a), client(c), device(d), is_deleted(del), dims(di), type(t), ref_count(1) {}
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
    
    MLXExecutable(std::string n, int r, int p) 
        : name(n), num_replicas(r), num_partitions(p), num_args(1), num_outputs(1), 
          ref_count(1), constants_cached(false) {}
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

#endif // MLX_PJRT_TYPES_H
