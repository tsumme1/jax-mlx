// mlx_mlir_parser.h - Native C++ MLIR text parser for standalone mode
// Replaces Python parser.py when MLX_STANDALONE is defined
// 
// NOTE: This header must be included AFTER the type definitions (MLXGraph, MLXOp)
// are already defined, since it uses those types but doesn't re-define them.

#ifndef MLX_MLIR_PARSER_H
#define MLX_MLIR_PARSER_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <unordered_map>
#include <sstream>
#include <regex>
#include <algorithm>
#include <cctype>
#include <iostream>
#include <cstdlib>

// Forward declarations - types are defined in jax_mlx_pjrt.cpp
// This header expects MLXGraph and MLXOp to be already defined when it's included

namespace mlx_parser {

// Check if data is a portable artifact (bytecode) format
inline bool IsPortableArtifact(const char* data, size_t size) {
    // ML\xefR is the magic header for portable artifacts
    return size >= 4 && 
           data[0] == 'M' && data[1] == 'L' && 
           static_cast<unsigned char>(data[2]) == 0xef && data[3] == 'R';
}

// Lightweight MLIR text parser for StableHLO
class MLIRParser {
public:
    MLIRParser() : next_id_(0) {}
    
    bool parse(const std::string& text, MLXGraph& graph) {
        std::map<std::string, std::shared_ptr<MLXGraph>> dummy_functions;
        return parseAll(text, graph, dummy_functions);
    }
    
    // Parse all functions - main goes into graph, others go into functions map
    bool parseAll(const std::string& text, MLXGraph& graph, 
                  std::map<std::string, std::shared_ptr<MLXGraph>>& functions) {
        lines_.clear();
        std::istringstream stream(text);
        std::string line;
        while (std::getline(stream, line)) {
            lines_.push_back(line);
        }
        
        // Find all functions
        std::vector<std::pair<size_t, std::string>> func_locations;
        for (size_t i = 0; i < lines_.size(); ++i) {
            if (lines_[i].find("func.func") != std::string::npos) {
                // Extract function name
                std::string func_name;
                size_t at_pos = lines_[i].find('@');
                if (at_pos != std::string::npos) {
                    size_t name_end = lines_[i].find_first_of("( ", at_pos + 1);
                    if (name_end != std::string::npos) {
                        func_name = lines_[i].substr(at_pos + 1, name_end - at_pos - 1);
                    }
                }
                if (!func_name.empty()) {
                    func_locations.push_back({i, func_name});
                }
            }
        }
        
        if (func_locations.empty()) {
            std::cerr << "[MLX-PARSER] No functions found in MLIR" << std::endl;
            return false;
        }
        
        // Parse each function
        bool found_main = false;
        for (const auto& [line_num, func_name] : func_locations) {
            // Reset ssa_map for each function to avoid ID collisions
            ssa_map_.clear();
            int saved_next_id = next_id_;  // Save current ID counter
            
            if (func_name == "main" || 
                lines_[line_num].find("public @") != std::string::npos) {
                // This is the main function
                if (parseFunction(line_num, graph)) {
                    found_main = true;
                }
            } else {
                // This is a private function
                auto func_graph = std::make_shared<MLXGraph>();
                if (parseFunction(line_num, *func_graph)) {
                    functions[func_name] = func_graph;
                    if (debug_enabled()) {
                        std::cerr << "[MLX-PARSER] Parsed function: @" << func_name 
                                  << " (" << func_graph->nodes.size() << " nodes)" << std::endl;
                    }
                }
            }
        }
        
        if (!found_main) {
            std::cerr << "[MLX-PARSER] No main function found in MLIR" << std::endl;
            return false;
        }
        
        if (debug_enabled() && !functions.empty()) {
            std::cerr << "[MLX-PARSER] Parsed " << functions.size() << " additional functions" << std::endl;
        }
        
        return true;
    }
    
private:
    static bool debug_enabled() {
        static bool checked = false;
        static bool enabled = false;
        if (!checked) {
            enabled = std::getenv("MLX_PJRT_DEBUG") != nullptr;
            checked = true;
        }
        return enabled;
    }
    
    std::vector<std::string> lines_;
    int next_id_;
    std::unordered_map<std::string, int> ssa_map_;  // Maps %name to id
    
    bool parseFunction(size_t start_line, MLXGraph& graph) {
        // Parse function signature to get input arguments
        std::string& sig_line = lines_[start_line];
        
        // Extract arguments between ( and )
        size_t arg_start = sig_line.find('(');
        size_t arg_end = sig_line.find(')');
        if (arg_start != std::string::npos && arg_end != std::string::npos) {
            std::string args = sig_line.substr(arg_start + 1, arg_end - arg_start - 1);
            parseArguments(args, graph);
        }
        
        // Parse operations in the function body
        int brace_count = 0;
        bool in_body = false;
        std::string current_op_lines;
        
        for (size_t i = start_line; i < lines_.size(); ++i) {
            const std::string& line = lines_[i];
            
            // Track braces to know when we're in the function body
            for (char c : line) {
                if (c == '{') { brace_count++; in_body = true; }
                if (c == '}') brace_count--;
            }
            
            if (!in_body) continue;
            
            // Handle end of function
            if (brace_count == 0) {
                if (!current_op_lines.empty()) {
                    parseOperation(current_op_lines, graph);
                    current_op_lines.clear();
                }
                break;
            }
            
            // Skip empty lines and comments
            std::string trimmed = trim(line);
            if (trimmed.empty() || trimmed[0] == '/') continue;
            
            // Handle func.return - explicit terminator
            if (trimmed.find("func.return") == 0 || trimmed.find("return") == 0) {
                if (!current_op_lines.empty()) {
                    parseOperation(current_op_lines, graph);
                    current_op_lines.clear();
                }
                parseReturn(trimmed, graph);
                continue;
            }

            // Check if this line starts a NEW operation assignment: %name = ...
            // Must have % and =
            size_t eq_pos = trimmed.find('=');
            bool starts_new_op = false;
            
            if (eq_pos != std::string::npos) {
                std::string lhs = trimmed.substr(0, eq_pos);
                if (lhs.find('%') != std::string::npos) {
                    // Check if it's an argument decl (skip) or op assignment
                    // Argument decl: %arg0: tensor... (has colon, tensor type)
                    // Op assignment: %0 = ... (no type info before =)
                    // But wait, multi-output op: %0:2 = ...
                    // Check if LHS has "tensor" keyword -> argument
                    if (lhs.find("tensor") == std::string::npos) {
                        starts_new_op = true;
                    }
                }
            }
            
            if (starts_new_op) {
                // If we were building an op, parse it now
                if (!current_op_lines.empty()) {
                    parseOperation(current_op_lines, graph);
                    current_op_lines.clear();
                }
                // Start new op
                current_op_lines = trimmed;
            } else {
                // Continuation line (attributes etc.)
                // Only append if we are currently building an op
                if (!current_op_lines.empty()) {
                    current_op_lines += " " + trimmed;
                }
            }
        }
        
        return true;
    }
    
    void parseArguments(const std::string& args, MLXGraph& graph) {
        // Parse: %arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>
        std::regex arg_regex(R"(%(\w+)\s*:\s*tensor<([^>]+)>)");
        std::sregex_iterator iter(args.begin(), args.end(), arg_regex);
        std::sregex_iterator end;
        
        while (iter != end) {
            std::smatch match = *iter;
            std::string name = match[1].str();
            std::string shape_dtype = match[2].str();
            
            int id = next_id_++;
            ssa_map_["%" + name] = id;
            graph.input_ids.push_back(id);
            
            // Parse shape
            std::vector<int> shape = parseShape(shape_dtype);
            graph.input_shapes.push_back(shape);
            
            ++iter;
        }
    }
    
    void parseOperation(const std::string& line, MLXGraph& graph) {
        MLXOp op;
        
        // Parse: %0 = stablehlo.add %arg0, %arg1 : tensor<2x3xf32>
        // Or: %result:2 = stablehlo.split %input ... (multiple results)
        
        size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) return;
        
        // Parse outputs (left of =)
        std::string outputs_str = trim(line.substr(0, eq_pos));
        std::vector<std::string> output_names;
        
        // Check for multiple outputs (%result:2)
        size_t output_colon_pos = outputs_str.find(':');
        if (output_colon_pos != std::string::npos && 
            outputs_str.find("tensor") == std::string::npos) {
            // Multi-output case: %17:2 creates %17#0, %17#1
            std::string base_name = outputs_str.substr(0, output_colon_pos);
            std::string count_str = outputs_str.substr(output_colon_pos + 1);
            try {
                int num_outputs = std::stoi(count_str);
                // Assign IDs to each output
                for (int i = 0; i < num_outputs; ++i) {
                    int id = next_id_++;
                    std::string indexed_name = base_name + "#" + std::to_string(i);
                    ssa_map_[indexed_name] = id;
                    op.outputs.push_back(id);
                }
            } catch (...) {
                // Parse error, treat as single output
                int id = next_id_++;
                ssa_map_[outputs_str] = id;
                op.outputs.push_back(id);
            }
        } else {
            // Single output case
            int id = next_id_++;
            ssa_map_[outputs_str] = id;
            op.outputs.push_back(id);
        }
        
        // Parse RHS: op_name operands : type
        std::string rhs = trim(line.substr(eq_pos + 1));
        
        // Extract operation name - find first space or '('
        size_t space_pos = rhs.find(' ');
        size_t paren_pos = rhs.find('(');
        size_t name_end = std::string::npos;
        
        if (space_pos != std::string::npos && paren_pos != std::string::npos) {
            name_end = std::min(space_pos, paren_pos);
        } else if (space_pos != std::string::npos) {
            name_end = space_pos;
        } else if (paren_pos != std::string::npos) {
            name_end = paren_pos;
        }
        
        if (name_end != std::string::npos) {
            op.op_name = rhs.substr(0, name_end);
        } else {
            op.op_name = rhs;
        }
        
        // Strip quotes if present (e.g. "stablehlo.reduce_window" -> stablehlo.reduce_window)
        if (op.op_name.length() >= 2 && op.op_name.front() == '"' && op.op_name.back() == '"') {
            op.op_name = op.op_name.substr(1, op.op_name.length() - 2);
        }
        
        // Special handling for call operations: call @function_name(args)
        if (op.op_name == "call") {
            size_t at_pos = rhs.find('@');
            size_t call_paren = rhs.find('(', at_pos);
            if (at_pos != std::string::npos && call_paren != std::string::npos) {
                std::string func_name = rhs.substr(at_pos + 1, call_paren - at_pos - 1);
                op.op_name = "func.call";  // Match what executor expects
                op.attributes["callee"] = func_name;
                // Note: name_end needs adjustment for operand parsing
                name_end = call_paren;
            }
        }
        
        // Special handling for sdy.sharding_constraint - treat as passthrough
        if (op.op_name.find("sdy.") == 0) {
            op.op_name = "sdy_passthrough";  // We'll pass through the input unchanged
        }
        
        // Special handling for custom_call: stablehlo.custom_call @target(args)
        if (op.op_name == "stablehlo.custom_call" || op.op_name == "mhlo.custom_call") {
            size_t at_pos = rhs.find('@');
            if (at_pos != std::string::npos) {
                // Find end of target name (either '(' or space)
                size_t target_end = rhs.find_first_of("( ", at_pos + 1);
                if (target_end == std::string::npos) target_end = rhs.length();
                std::string target_name = rhs.substr(at_pos + 1, target_end - at_pos - 1);
                op.attributes["call_target_name"] = target_name;
            }
        }
        
        // For iota ops, parse inline "dim = N" attribute
        // Format: stablehlo.iota dim = 0 : tensor<3x3xui64>
        if (op.op_name == "stablehlo.iota" || op.op_name == "mhlo.iota") {
            std::regex dim_regex(R"(dim\s*=\s*(\d+))");
            std::smatch match;
            if (std::regex_search(rhs, match, dim_regex)) {
                op.attributes["iota_dimension"] = match[1].str();
            }
        }
        
        // For compare ops, parse inline comparison direction
        // Format: stablehlo.compare  LT, %arg0, %1,  FLOAT : ...
        if (op.op_name == "stablehlo.compare" || op.op_name == "mhlo.compare") {
            // Look for the comparison direction (LT, LE, GT, GE, EQ, NE)
            std::regex cmp_regex(R"(\b(LT|LE|GT|GE|EQ|NE)\b)");
            std::smatch match;
            if (std::regex_search(rhs, match, cmp_regex)) {
                op.attributes["comparison_direction"] = match[1].str();
            }
        }
        
        // For convolution ops, parse dim_numbers and window attributes
        // Format: stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], ...}
        if (op.op_name == "stablehlo.convolution" || op.op_name == "mhlo.convolution") {
            // Parse dim_numbers: [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]
            // Input format: b = batch, f = feature, 0,1 = spatial dims
            // This is complex to parse, so we'll use defaults that match NHWC x HWIO -> NHWC
            // Default values (already set in executor): input [b,0,1,f], kernel [0,1,i,o], output [b,0,1,f]
            
            // Parse window attributes: stride, pad, lhs_dilate, rhs_dilate
            size_t window_pos = rhs.find("window = {");
            if (window_pos != std::string::npos) {
                size_t window_end = rhs.find("}", window_pos);
                if (window_end != std::string::npos) {
                    std::string window_str = rhs.substr(window_pos, window_end - window_pos + 1);
                    
                    // Parse stride = [1, 1]
                    std::regex stride_regex(R"(stride\s*=\s*\[([\d,\s]+)\])");
                    std::smatch stride_match;
                    if (std::regex_search(window_str, stride_match, stride_regex)) {
                        std::string stride_str = stride_match[1].str();
                        std::vector<int64_t> strides;
                        std::regex num_regex(R"(\d+)");
                        auto nums_begin = std::sregex_iterator(stride_str.begin(), stride_str.end(), num_regex);
                        auto nums_end = std::sregex_iterator();
                        for (std::sregex_iterator i = nums_begin; i != nums_end; ++i) {
                            strides.push_back(std::stoll(i->str()));
                        }
                        op.int_array_attrs["stride"] = strides;
                    }
                    
                    // Parse pad = [[0, 0], [0, 0]]
                    std::regex pad_regex(R"(pad\s*=\s*\[\s*\[([^\]]*)\]\s*,\s*\[([^\]]*)\]\s*\])");
                    std::smatch pad_match;
                    if (std::regex_search(window_str, pad_match, pad_regex)) {
                        std::vector<int64_t> padding;
                        std::regex num_regex(R"(\d+)");
                        // First dimension
                        std::string pad0_str = pad_match[1].str();
                        auto it0 = std::sregex_iterator(pad0_str.begin(), pad0_str.end(), num_regex);
                        auto end = std::sregex_iterator();
                        while (it0 != end) {
                            padding.push_back(std::stoll(it0->str()));
                            ++it0;
                        }
                        // Second dimension
                        std::string pad1_str = pad_match[2].str();
                        auto it1 = std::sregex_iterator(pad1_str.begin(), pad1_str.end(), num_regex);
                        while (it1 != end) {
                            padding.push_back(std::stoll(it1->str()));
                            ++it1;
                        }
                        if (!padding.empty()) op.int_array_attrs["padding"] = padding;
                    }
                    
                    // Parse lhs_dilate = [1, 1]
                    std::regex lhs_dil_regex(R"(lhs_dilate\s*=\s*\[([\d,\s]+)\])");
                    std::smatch lhs_dil_match;
                    if (std::regex_search(window_str, lhs_dil_match, lhs_dil_regex)) {
                        std::string dil_str = lhs_dil_match[1].str();
                        std::vector<int64_t> dilation;
                        std::regex num_regex(R"(\d+)");
                        auto it = std::sregex_iterator(dil_str.begin(), dil_str.end(), num_regex);
                        auto end = std::sregex_iterator();
                        while (it != end) {
                            dilation.push_back(std::stoll(it->str()));
                            ++it;
                        }
                        if (!dilation.empty()) op.int_array_attrs["lhs_dilation"] = dilation;
                    }
                    
                    // Parse rhs_dilate = [1, 1]
                    std::regex rhs_dil_regex(R"(rhs_dilate\s*=\s*\[([\d,\s]+)\])");
                    std::smatch rhs_dil_match;
                    if (std::regex_search(window_str, rhs_dil_match, rhs_dil_regex)) {
                        std::string dil_str = rhs_dil_match[1].str();
                        std::vector<int64_t> dilation;
                        std::regex num_regex(R"(\d+)");
                        auto it = std::sregex_iterator(dil_str.begin(), dil_str.end(), num_regex);
                        auto end = std::sregex_iterator();
                        while (it != end) {
                            dilation.push_back(std::stoll(it->str()));
                            ++it;
                        }
                        if (!dilation.empty()) op.int_array_attrs["rhs_dilation"] = dilation;
                    }
                }
            }
        }
        
        // For dot_general ops
        if (op.op_name == "stablehlo.dot_general" || op.op_name == "mhlo.dot_general") {
             // Parse dot_dimension_numbers = {lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]}
             // Also batching dimensions.
             
             size_t dims_start = rhs.find("dot_dimension_numbers = {");
             if (dims_start != std::string::npos) {
                 size_t dims_end = rhs.find("}", dims_start);
                 if (dims_end != std::string::npos) {
                     std::string dims_block = rhs.substr(dims_start, dims_end - dims_start);
                     
                     // Helper regex to extract arrays
                     auto parse_dims = [&](const std::string& key, const std::string& attr_name) {
                         std::string pattern = key + R"(\s*=\s*\[([\d,\s]*)\])";
                         std::regex r(pattern);
                         std::smatch m;
                         if (std::regex_search(dims_block, m, r)) {
                             std::string nums = m[1].str();
                             std::vector<int64_t> d;
                             if (!nums.empty()) {
                                 std::regex num_regex(R"(\d+)");
                                 auto b = std::sregex_iterator(nums.begin(), nums.end(), num_regex);
                                 auto e = std::sregex_iterator();
                                 for (auto i = b; i != e; ++i) d.push_back(std::stoll(i->str()));
                             }
                             op.int_array_attrs[attr_name] = d;
                         }
                     };
                     
                     parse_dims("lhs_contracting_dimensions", "lhs_contracting");
                     parse_dims("rhs_contracting_dimensions", "rhs_contracting");
                     parse_dims("lhs_batching_dimensions", "lhs_batching");
                     parse_dims("rhs_batching_dimensions", "rhs_batching");
                 }
             }
             
             // Parse shorthand: contracting_dims = [0] x [0]
             std::regex contract_regex(R"(contracting_dims\s*=\s*\[([\d,\s]*)\]\s*x\s*\[([\d,\s]*)\])");
             std::smatch contract_match;
             if (std::regex_search(rhs, contract_match, contract_regex)) {
                 auto parse_list = [&](std::string nums) {
                     std::vector<int64_t> d;
                     std::regex num_regex(R"(\d+)");
                     auto b = std::sregex_iterator(nums.begin(), nums.end(), num_regex);
                     auto e = std::sregex_iterator();
                     for (auto i = b; i != e; ++i) d.push_back(std::stoll(i->str()));
                     return d;
                 };
                 op.int_array_attrs["lhs_contracting"] = parse_list(contract_match[1].str());
                 op.int_array_attrs["rhs_contracting"] = parse_list(contract_match[2].str());
             }

             // Parse shorthand: batching_dims = [0] x [0]
             std::regex batch_regex(R"(batching_dims\s*=\s*\[([\d,\s]*)\]\s*x\s*\[([\d,\s]*)\])");
             std::smatch batch_match;
             if (std::regex_search(rhs, batch_match, batch_regex)) {
                 auto parse_list = [&](std::string nums) {
                     std::vector<int64_t> d;
                     std::regex num_regex(R"(\d+)");
                     auto b = std::sregex_iterator(nums.begin(), nums.end(), num_regex);
                     auto e = std::sregex_iterator();
                     for (auto i = b; i != e; ++i) d.push_back(std::stoll(i->str()));
                     return d;
                 };
                 op.int_array_attrs["lhs_batching"] = parse_list(batch_match[1].str());
                 op.int_array_attrs["rhs_batching"] = parse_list(batch_match[2].str());
             }
        }

        // For reduce ops
        if (op.op_name == "stablehlo.reduce" || op.op_name == "mhlo.reduce") {
             // Parse dimensions = [1]
             std::regex dims_regex(R"(dimensions\s*=\s*\[([\d,\s]+)\])");
             std::smatch dims_match;
             if (std::regex_search(rhs, dims_match, dims_regex)) {
                 std::string nums = dims_match[1].str();
                 std::vector<int64_t> dims;
                 std::regex num_regex(R"(\d+)");
                 auto nums_begin = std::sregex_iterator(nums.begin(), nums.end(), num_regex);
                 auto nums_end = std::sregex_iterator();
                 for (std::sregex_iterator i = nums_begin; i != nums_end; ++i) {
                     dims.push_back(std::stoll(i->str()));
                 }
                 op.int_array_attrs["dimensions"] = dims;
             }
        }

        // For broadcast_in_dim
        if (op.op_name == "stablehlo.broadcast_in_dim" || op.op_name == "mhlo.broadcast_in_dim") {
            // Parse dims = [0, 1]
            std::regex dims_regex(R"(dims\s*=\s*\[([\d,\s]+)\])");
            std::smatch dims_match;
            if (std::regex_search(rhs, dims_match, dims_regex)) {
                std::string nums = dims_match[1].str();
                std::vector<int64_t> dims;
                std::regex num_regex(R"(\d+)");
                auto nums_begin = std::sregex_iterator(nums.begin(), nums.end(), num_regex);
                auto nums_end = std::sregex_iterator();
                for (std::sregex_iterator i = nums_begin; i != nums_end; ++i) {
                    dims.push_back(std::stoll(i->str()));
                }
                op.int_array_attrs["broadcast_dimensions"] = dims;
            }
        }

        // For reduce_window ops (generic syntax)
        if (op.op_name == "stablehlo.reduce_window" || op.op_name == "mhlo.reduce_window") {
            // Check for generic attributes block <{...}>
            size_t attr_start = rhs.find("<{");
            size_t attr_end = rhs.find("}>");
            if (attr_start != std::string::npos && attr_end != std::string::npos) {
                 std::string attr_block = rhs.substr(attr_start, attr_end - attr_start);
                 
                 // Parse window_dimensions = array<i64: 1, 2, 2, 1>
                 std::regex dim_regex(R"(window_dimensions\s*=\s*array<i64:\s*([\d,\s]+)>)");
                 std::smatch dim_match;
                 if (std::regex_search(attr_block, dim_match, dim_regex)) {
                     std::string nums = dim_match[1].str();
                     std::vector<int64_t> dims;
                     std::regex num_regex(R"(\d+)");
                     auto nums_begin = std::sregex_iterator(nums.begin(), nums.end(), num_regex);
                     auto nums_end = std::sregex_iterator();
                     for (std::sregex_iterator i = nums_begin; i != nums_end; ++i) {
                         dims.push_back(std::stoll(i->str()));
                     }
                     op.int_array_attrs["window_dimensions"] = dims;
                 }
                 
                 // Parse window_strides = array<i64: 1, 2, 2, 1>
                 std::regex stride_regex(R"(window_strides\s*=\s*array<i64:\s*([\d,\s]+)>)");
                 std::smatch stride_match;
                 if (std::regex_search(attr_block, stride_match, stride_regex)) {
                     std::string nums = stride_match[1].str();
                     std::vector<int64_t> strides;
                     std::regex num_regex(R"(\d+)");
                     auto nums_begin = std::sregex_iterator(nums.begin(), nums.end(), num_regex);
                     auto nums_end = std::sregex_iterator();
                     for (std::sregex_iterator i = nums_begin; i != nums_end; ++i) {
                         strides.push_back(std::stoll(i->str()));
                     }
                     op.int_array_attrs["window_strides"] = strides;
                 }
            }
        }
        
        // For reduce ops, extract the reduce function from "applies stablehlo.add"
        if (op.op_name == "stablehlo.reduce" || op.op_name == "mhlo.reduce") {
            size_t applies_pos = rhs.find("applies ");
            if (applies_pos != std::string::npos) {
                size_t fn_start = applies_pos + 8;  // After "applies "
                size_t fn_end = rhs.find(' ', fn_start);
                if (fn_end == std::string::npos) fn_end = rhs.length();
                std::string reduce_fn = rhs.substr(fn_start, fn_end - fn_start);
                // Map reduce function to reduce_type
                if (reduce_fn.find("add") != std::string::npos) {
                    op.attributes["reduce_type"] = "sum";
                } else if (reduce_fn.find("maximum") != std::string::npos || reduce_fn.find("max") != std::string::npos) {
                    op.attributes["reduce_type"] = "max";
                } else if (reduce_fn.find("minimum") != std::string::npos || reduce_fn.find("min") != std::string::npos) {
                    op.attributes["reduce_type"] = "min";
                } else if (reduce_fn.find("multiply") != std::string::npos || reduce_fn.find("mul") != std::string::npos) {
                    op.attributes["reduce_type"] = "prod";
                } else if (reduce_fn.find("or") != std::string::npos) {
                    op.attributes["reduce_type"] = "or";
                } else if (reduce_fn.find("and") != std::string::npos) {
                    op.attributes["reduce_type"] = "and";
                } else {
                    op.attributes["reduce_type"] = "sum";  // Default
                }
            }
            
            // Extract dimensions from "dimensions = [0]" or "across dimensions = [0]"
            size_t dim_pos = rhs.find("dimensions = [");
            if (dim_pos != std::string::npos) {
                size_t bracket_start = rhs.find('[', dim_pos);
                size_t bracket_end = rhs.find(']', bracket_start);
                if (bracket_start != std::string::npos && bracket_end != std::string::npos) {
                    std::string dim_str = rhs.substr(bracket_start + 1, bracket_end - bracket_start - 1);
                    std::vector<int64_t> dims;
                    std::regex num_regex(R"(-?\d+)");
                    std::sregex_iterator dim_iter(dim_str.begin(), dim_str.end(), num_regex);
                    std::sregex_iterator dim_end;
                    while (dim_iter != dim_end) {
                        dims.push_back(std::stoll((*dim_iter)[0].str()));
                        ++dim_iter;
                    }
                    op.int_array_attrs["dimensions"] = dims;
                }
            }
        }
        
        // For slice ops, parse inline [start:limit, ...] syntax
        // Format: stablehlo.slice %arg0 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
        //         stablehlo.slice %arg0 [0:2, 1:3] : (tensor<4x5xf32>) -> tensor<2x2xf32>
        if (op.op_name == "stablehlo.slice" || op.op_name == "mhlo.slice") {
            size_t bracket_start = rhs.find('[');
            size_t bracket_end = rhs.find(']');
            if (bracket_start != std::string::npos && bracket_end != std::string::npos && bracket_end > bracket_start) {
                std::string slice_spec = rhs.substr(bracket_start + 1, bracket_end - bracket_start - 1);
                // Parse each dimension's start:limit[:stride]
                std::vector<int64_t> starts, limits, strides;
                
                // Split by comma
                size_t pos = 0;
                while (pos < slice_spec.length()) {
                    // Find next comma or end
                    size_t next_comma = slice_spec.find(',', pos);
                    std::string dim_spec = (next_comma == std::string::npos) 
                        ? slice_spec.substr(pos) 
                        : slice_spec.substr(pos, next_comma - pos);
                    
                    // Parse start:limit[:stride]
                    size_t colon1 = dim_spec.find(':');
                    if (colon1 != std::string::npos) {
                        std::string start_str = trim(dim_spec.substr(0, colon1));
                        std::string rest = dim_spec.substr(colon1 + 1);
                        
                        size_t colon2 = rest.find(':');
                        std::string limit_str = (colon2 == std::string::npos) 
                            ? trim(rest) 
                            : trim(rest.substr(0, colon2));
                        std::string stride_str = (colon2 == std::string::npos) 
                            ? "1" 
                            : trim(rest.substr(colon2 + 1));
                        
                        try {
                            starts.push_back(std::stoll(start_str));
                            limits.push_back(std::stoll(limit_str));
                            strides.push_back(stride_str.empty() ? 1 : std::stoll(stride_str));
                        } catch (...) {
                            // Parse error, skip
                        }
                    }
                    
                    if (next_comma == std::string::npos) break;
                    pos = next_comma + 1;
                }
                
                if (!starts.empty()) {
                    op.int_array_attrs["start_indices"] = starts;
                    op.int_array_attrs["limit_indices"] = limits;
                    op.int_array_attrs["strides"] = strides;
                }
            }
        }
        
        // For constant ops, parse dense<...> values
        // Format: stablehlo.constant dense<32> : tensor<i32>
        //         stablehlo.constant dense<[1, 2, 3]> : tensor<3xi32>
        //         stablehlo.constant dense<1.0> : tensor<f32>
        if (op.op_name == "stablehlo.constant" || op.op_name == "mhlo.constant") {
            size_t dense_pos = rhs.find("dense<");
            if (dense_pos != std::string::npos) {
                size_t value_start = dense_pos + 6;  // After "dense<"
                size_t value_end = rhs.find('>', value_start);
                if (value_end != std::string::npos) {
                    std::string value_str = rhs.substr(value_start, value_end - value_start);
                    
                    // Check if it's a float (contains '.' or 'e')
                    bool is_float = (value_str.find('.') != std::string::npos || 
                                     value_str.find('e') != std::string::npos ||
                                     value_str.find('E') != std::string::npos);
                    
                    if (is_float) {
                        // Parse as float array
                        std::vector<float> float_vals;
                        std::regex float_regex(R"(-?[\d.]+(?:[eE][+-]?\d+)?)");
                        std::sregex_iterator fiter(value_str.begin(), value_str.end(), float_regex);
                        std::sregex_iterator fend;
                        while (fiter != fend) {
                            try {
                                float_vals.push_back(std::stof((*fiter)[0].str()));
                            } catch (...) {}
                            ++fiter;
                        }
                        if (!float_vals.empty()) {
                            op.float_array_attrs["value"] = float_vals;
                        }
                    } else {
                        // Parse as int array
                        std::vector<int64_t> int_vals;
                        std::regex int_regex(R"(-?\d+)");
                        std::sregex_iterator iiter(value_str.begin(), value_str.end(), int_regex);
                        std::sregex_iterator iend;
                        while (iiter != iend) {
                            try {
                                int_vals.push_back(std::stoll((*iiter)[0].str()));
                            } catch (...) {}
                            ++iiter;
                        }
                        if (!int_vals.empty()) {
                            op.int_array_attrs["value"] = int_vals;
                        }
                    }
                }
            }
        }
        
        // Extract operands and type info
        size_t colon_pos = rhs.rfind(':');
        std::string operands_str;
        std::string type_str;
        
        if (colon_pos != std::string::npos && name_end != std::string::npos) {
            operands_str = rhs.substr(name_end + 1, colon_pos - name_end - 1);
            type_str = trim(rhs.substr(colon_pos + 1));
        } else if (name_end != std::string::npos) {
            operands_str = rhs.substr(name_end + 1);
        }
        
        // Parse operands (space or comma separated %names)
        std::regex operand_regex(R"(%[\w#]+)");
        std::sregex_iterator iter(operands_str.begin(), operands_str.end(), operand_regex);
        std::sregex_iterator end;
        
        while (iter != end) {
            std::string operand = (*iter)[0].str();
            if (ssa_map_.count(operand)) {
                op.inputs.push_back(ssa_map_[operand]);
            }
            ++iter;
        }
        
        // Parse output type
        if (!type_str.empty()) {
            // Handle arrow type format: (input) -> output
            // We want the OUTPUT type (after ->)
            std::string output_type_str = type_str;
            size_t arrow_pos = type_str.find("->");
            if (arrow_pos != std::string::npos) {
                output_type_str = trim(type_str.substr(arrow_pos + 2));
            }
            
            // Handle tuple types: (tensor<...>, tensor<...>) for multiple outputs
            // Also handle simple type: tensor<...>
            std::regex tensor_regex(R"(tensor<([^>]+)>)");
            std::sregex_iterator type_iter(output_type_str.begin(), output_type_str.end(), tensor_regex);
            std::sregex_iterator type_end;
            
            while (type_iter != type_end) {
                std::string shape_dtype = (*type_iter)[1].str();
                op.output_shapes.push_back(parseShape(shape_dtype));
                op.output_dtypes.push_back(parseDtype(shape_dtype));
                ++type_iter;
            }
        }
        
        // Parse attributes in {...}
        size_t attr_start = rhs.find('{');
        size_t attr_end = rhs.rfind('}');
        if (attr_start != std::string::npos && attr_end != std::string::npos) {
            std::string attrs = rhs.substr(attr_start + 1, attr_end - attr_start - 1);
            parseAttributes(attrs, op);
        }
        
        graph.nodes.push_back(op);
    }
    
    void parseReturn(const std::string& line, MLXGraph& graph) {
        // Parse: func.return %0 : tensor<2x3xf32>
        std::regex ret_regex(R"(%[\w#]+)");
        std::sregex_iterator iter(line.begin(), line.end(), ret_regex);
        std::sregex_iterator end;
        
        while (iter != end) {
            std::string name = (*iter)[0].str();
            if (ssa_map_.count(name)) {
                graph.output_ids.push_back(ssa_map_[name]);
            }
            ++iter;
        }
    }
    
    std::vector<int> parseShape(const std::string& shape_dtype) {
        std::vector<int> shape;
        // Parse "2x3xf32" -> [2, 3]
        std::regex dim_regex(R"((\d+)x)");
        std::sregex_iterator iter(shape_dtype.begin(), shape_dtype.end(), dim_regex);
        std::sregex_iterator end;
        
        while (iter != end) {
            shape.push_back(std::stoi((*iter)[1].str()));
            ++iter;
        }
        
        // Handle scalar case (no dimensions, just dtype like "f32")
        if (shape.empty() && !shape_dtype.empty()) {
            // Check if it's purely a dtype (no dimensions)
            if (shape_dtype.find('x') == std::string::npos) {
                // Scalar - empty shape is correct
            }
        }
        
        return shape;
    }
    
    std::string parseDtype(const std::string& shape_dtype) {
        // Extract dtype from end of "2x3xf32" or just "f32"
        size_t last_x = shape_dtype.rfind('x');
        if (last_x != std::string::npos) {
            return shape_dtype.substr(last_x + 1);
        }
        return shape_dtype;  // Pure dtype like "f32"
    }
    
    void parseAttributes(const std::string& attrs, MLXOp& op) {
        // Parse: key = value, key2 = value2
        // Simple regex-based parsing
        std::regex attr_regex(R"((\w+)\s*=\s*([^,}]+))");
        std::sregex_iterator iter(attrs.begin(), attrs.end(), attr_regex);
        std::sregex_iterator end;
        
        while (iter != end) {
            std::string key = (*iter)[1].str();
            std::string value = trim((*iter)[2].str());
            
            // Try to parse as integer
            try {
                size_t pos;
                int64_t int_val = std::stoll(value, &pos);
                if (pos == value.length()) {
                    op.int_attrs[key] = int_val;
                }
            } catch (...) {}
            
            // Try to parse as int array [1, 2, 3] or dense<[1, 2]>
            if (value[0] == '[' || value.find("dense<") != std::string::npos) {
                std::vector<int64_t> arr;
                std::regex num_regex(R"(-?\d+)");
                std::sregex_iterator num_iter(value.begin(), value.end(), num_regex);
                std::sregex_iterator num_end;
                while (num_iter != num_end) {
                    arr.push_back(std::stoll((*num_iter)[0].str()));
                    ++num_iter;
                }
                if (!arr.empty()) {
                    op.int_array_attrs[key] = arr;
                }
            }
            
            // Store as string attr always
            op.attributes[key] = value;
            
            ++iter;
        }
    }
    
    static std::string trim(const std::string& s) {
        size_t start = s.find_first_not_of(" \t\n\r");
        if (start == std::string::npos) return "";
        size_t end = s.find_last_not_of(" \t\n\r");
        return s.substr(start, end - start + 1);
    }
};

// Main entry point
inline bool ParseMLIRText(const std::string& text, MLXGraph& graph) {
    MLIRParser parser;
    return parser.parse(text, graph);
}

// Extended entry point that also parses private functions
inline bool ParseMLIRText(const std::string& text, MLXGraph& graph, 
                          std::map<std::string, std::shared_ptr<MLXGraph>>& functions) {
    MLIRParser parser;
    return parser.parseAll(text, graph, functions);
}

} // namespace mlx_parser

#endif // MLX_MLIR_PARSER_H
