import jax
import os

# Debug flag - avoid print overhead in hot path
_DEBUG = os.environ.get("MLX_PJRT_DEBUG") is not None

try:
    from jax._src.lib.mlir import ir
    from jax._src.lib.mlir.dialects import stablehlo, func, mhlo
    if _DEBUG: print("[MLX-Parser] Using jax._src.lib.mlir", flush=True)
except ImportError:
    from jaxlib.mlir import ir
    from jaxlib.mlir.dialects import stablehlo, func, mhlo
    if _DEBUG: print("[MLX-Parser] Using jaxlib.mlir", flush=True)
import io

# Try to use JAX's internal context factory
try:
    from jax._src.interpreters.mlir import make_ir_context
    if _DEBUG: print("[MLX-Parser] Imported make_ir_context from jax._src.interpreters.mlir", flush=True)
except ImportError:
    make_ir_context = None
    if _DEBUG: print("[MLX-Parser] Could not import make_ir_context", flush=True)

# Import MLX for optimized gradient helpers
try:
    import mlx.core as mx
    _MLX_AVAILABLE = True
    if _DEBUG: print("[MLX-Parser] MLX core imported for gradient helpers", flush=True)
except ImportError:
    _MLX_AVAILABLE = False
    if _DEBUG: print("[MLX-Parser] MLX core not available", flush=True)

if _DEBUG: print("[MLX-Parser] Module loaded", flush=True)

# Cache for compiled gradient functions
_grad_fn_cache = {}

def compute_conv_weight_grad(input_arr, upstream_grad, stride, padding, kernel_shape):
    """
    Compute weight gradient using MLX's optimized grad function.
    This is ~10x faster than computing it via conv_general with swapped dimensions.
    
    Args:
        input_arr: The original input to the forward conv (NHWC format)
        upstream_grad: The gradient from upstream (NHWC format)
        stride: (stride_h, stride_w)
        padding: (pad_h, pad_w)
        kernel_shape: Target kernel shape (out_ch, kH, kW, in_ch) in OHWI format
    
    Returns:
        Weight gradient in OHWI format
    """
    if not _MLX_AVAILABLE:
        return None
    
    out_ch, kh, kw, in_ch = kernel_shape
    sh, sw = stride
    ph, pw = padding
    
    # Create cache key
    cache_key = (input_arr.shape, upstream_grad.shape, tuple(stride), tuple(padding), tuple(kernel_shape))
    
    if cache_key not in _grad_fn_cache:
        # Create a dummy kernel for the gradient function
        dummy_kernel = mx.zeros((out_ch, kh, kw, in_ch))
        
        # Create grad function using mx.grad
        def conv_fn(kernel):
            y = mx.conv2d(input_arr, kernel, stride=(sh, sw), padding=(ph, pw))
            # Weight the output by upstream gradient and sum
            return mx.sum(y * upstream_grad)
        
        _grad_fn_cache[cache_key] = mx.grad(conv_fn)
    
    grad_fn = _grad_fn_cache[cache_key]
    
    # Compute gradient
    dummy_kernel = mx.zeros((out_ch, kh, kw, in_ch), dtype=input_arr.dtype)
    wgrad = grad_fn(dummy_kernel)
    
    return wgrad


# Cache for MaxPool layers
_max_pool_cache = {}

def max_pool_2d(input_arr, kernel_h, kernel_w, stride_h, stride_w, pad_h=0, pad_w=0):
    """
    Apply 2D max pooling using MLX's native implementation.
    
    Args:
        input_arr: Input array in NHWC format
        kernel_h, kernel_w: Pooling window size
        stride_h, stride_w: Stride for pooling
        pad_h, pad_w: Padding (symmetric)
    
    Returns:
        Max pooled output in NHWC format
    """
    if not _MLX_AVAILABLE:
        return None
    
    import mlx.nn as nn
    
    # Create cache key - pool layers are stateless so just cache by params
    cache_key = (kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w)
    
    if cache_key not in _max_pool_cache:
        pool_layer = nn.MaxPool2d(
            kernel_size=(kernel_h, kernel_w),
            stride=(stride_h, stride_w),
            padding=(pad_h, pad_w)
        )
        _max_pool_cache[cache_key] = pool_layer
    
    pool = _max_pool_cache[cache_key]
    return pool(input_arr)


def parse_bytecode(bytecode):
    if _DEBUG: print(f"[MLX-Parser] parse_bytecode called with {len(bytecode)} bytes", flush=True)
    if _DEBUG and len(bytecode) > 16:
        print(f"[MLX-Parser]   Header: {bytecode[:16].hex()}", flush=True)
        print(f"[MLX-Parser]   Header (str): {bytecode[:16]}", flush=True)
    
    ctx = None
    if make_ir_context:
        try:
            ctx = make_ir_context()
            if _DEBUG: print("[MLX-Parser] Created context via make_ir_context", flush=True)
        except Exception as e:
            if _DEBUG: print(f"[MLX-Parser] make_ir_context failed: {e}", flush=True)
            
    if ctx is None:
        ctx = ir.Context()
        # Fallback registration
        try:
            stablehlo.register_dialect(ctx)
        except: pass
        try:
            mhlo.register_dialect(ctx)
        except: pass
        ctx.allow_unregistered_dialects = True

    with ctx:
        ctx.allow_unregistered_dialects = True
        try:
            # Check for JAX portable artifact header
            if bytecode.startswith(b'ML\xefR'):
                try:
                    from jaxlib import xla_client
                    bytecode_text = xla_client._xla.mlir.deserialize_portable_artifact(bytecode)
                    if _DEBUG: print("[MLX-Parser]   Deserialized portable artifact successful", flush=True)
                    # Debug: print while body contents (disabled - was causing issues)
                    # if "stablehlo.while" in bytecode_text:
                    #     lines = bytecode_text.split('\n')
                    #     in_body = False
                    #     for line in lines:
                    #         if "} do {" in line:
                    #             in_body = True
                    #         if in_body:
                    #             print(f"[MLX-Parser] BODY: {line}")
                    #             if "stablehlo.return" in line:
                    #                 break
                    module_op = ir.Module.parse(bytecode_text, context=ctx)
                except Exception as e:
                    if _DEBUG: print(f"[MLX-Parser]   Portable deserialization failed: {e}, trying raw parse", flush=True)
                    module_op = ir.Module.parse(bytecode, context=ctx)
            else:
                module_op = ir.Module.parse(bytecode, context=ctx)
        except Exception as e:
            return {"error": f"Failed to parse bytecode: {str(e)}"}

        # Maps
        value_to_id = {}
        next_id = 0

        def get_id(val):
            nonlocal next_id
            if val not in value_to_id:
                value_to_id[val] = next_id
                next_id += 1
            return value_to_id[val]

        def get_type_info(t):
            t_str = str(t)
            shape = []
            dtype = "unknown"
            try:
                from jaxlib.mlir.ir import RankedTensorType
                if isinstance(t, RankedTensorType):
                    shape = t.shape
                    dtype = str(t.element_type)
            except: pass
            return {"shape": shape, "dtype": dtype, "full": t_str}

        # Collect functions
        funcs = {}
        all_ops = []
        # Fix: Iterate module_op.body
        try:
            for op in module_op.body.operations:
                if op.operation.name == "builtin.module":
                    for region in op.regions:
                        for block in region.blocks:
                            all_ops.extend(block.operations)
                else:
                    all_ops.append(op)
        except:
             # Fallback
             all_ops = module_op.body.operations

        for op in all_ops:
            if op.operation.name == "func.func":
                try:
                    name = op.operation.attributes["sym_name"].value
                    funcs[name] = op
                except: pass

        # Recursive Parser
        def parse_block(block, operand_ids_map=None):
            local_graph = {"nodes": [], "inputs": [], "outputs": []}
            
            # Map arguments
            if operand_ids_map:
                # For subgraphs: Generate FRESH IDs for block args to avoid collision
                # with CSE'd parent-scope values. Store the parent_id in input info for runtime.
                for i, arg in enumerate(block.arguments):
                    # Generate a fresh unique ID for this block arg
                    fresh_id = get_id(arg)  # This will create a new unique ID
                    if i < len(operand_ids_map):
                        # Record that this input maps from parent operand
                        local_graph["inputs"].append({
                            "id": fresh_id,
                            "parent_id": operand_ids_map[i],  # Store parent mapping for reference
                            "index": i,
                            "type": get_type_info(arg.type)
                        })
                    else:
                        local_graph["inputs"].append({
                            "id": fresh_id,
                            "index": i,
                            "type": get_type_info(arg.type)
                        })
            else:
                for i, arg in enumerate(block.arguments):
                    arg_id = get_id(arg)
                    local_graph["inputs"].append({
                        "id": arg_id,
                        "index": i,
                        "type": get_type_info(arg.type)
                    })
            
            for inner_op in block.operations:
                op_name = inner_op.operation.name
                
                # Terminator
                if op_name in ["func.return", "stablehlo.return", "mhlo.return"]:
                    local_graph["outputs"] = [get_id(operand) for operand in inner_op.operands]
                    continue
                
                # Inlining logic (Only if called directly, usually flattened by JAX)
                if op_name == "func.call":
                    # We skip inlining in this sub-graph parser model for now (except for main?)
                    # If main logic calls parse_block on main func, it works.
                    # If func calls another func, treating it as node is better BUT we need implementation.
                    # For JIT, loops use regions.
                    pass
                
                # DEBUG: trace constant parsing
                if _DEBUG and "constant" in op_name and operand_ids_map:
                    print(f"[MLX-Parser] Subgraph constant: {op_name}")
                    for res in inner_op.results:
                        print(f"[MLX-Parser]   result id={get_id(res)}")

                node = {
                    "op": op_name,
                    "inputs": [get_id(operand) for operand in inner_op.operands],
                    "outputs": [get_id(res) for res in inner_op.results],
                    "output_types": [get_type_info(res.type) for res in inner_op.results],
                    "attributes": {},
                    "subgraphs": []
                }

                # Attributes logic follows

                # Re-implement loop with safe extraction
                # attrs.items() can fail on complex DenseElementsAttr types
                
                op_attrs = inner_op.operation.attributes
                use_items = False
                try:
                    iterator = op_attrs.items() if hasattr(op_attrs, "items") else op_attrs
                    use_items = hasattr(op_attrs, "items")
                except TypeError:
                    # Complex element types cause items() to fail
                    # Fall back to iterating NamedAttributes
                    iterator = op_attrs
                    use_items = False
                
                for item in iterator:
                    try:
                        if use_items: # item is (key, value)
                            attr_name, attr_val = item
                        elif isinstance(item, str): # item is key
                            attr_name = item
                            attr_val = op_attrs[attr_name] 
                        else: # item is NamedAttribute
                             attr_name = item.name
                             attr_val = item.attr
                    except TypeError as attr_err:
                        # Skip attributes that can't be accessed due to complex type issues
                        if _DEBUG: print(f"[MLX-Parser] Skipping attr due to type error: {attr_err}")
                        continue

                    type_str = str(type(attr_val))
                    
                    # Optimized dense attr handling logic
                    if "dot_dimension_numbers" in attr_name:
                         try:
                             dot_nums = stablehlo.DotDimensionNumbers(attr_val)
                             node["attributes"]["lhs_batching"] = [int(x) for x in dot_nums.lhs_batching_dimensions]
                             node["attributes"]["rhs_batching"] = [int(x) for x in dot_nums.rhs_batching_dimensions]
                             node["attributes"]["lhs_contracting"] = [int(x) for x in dot_nums.lhs_contracting_dimensions]
                             node["attributes"]["rhs_contracting"] = [int(x) for x in dot_nums.rhs_contracting_dimensions]
                         except Exception as e:
                             if _DEBUG: print(f"[MLX-Parser] Failed to parse dot_dims: {e}")

                    # Only try to parse convolution dimension numbers for convolution ops
                    is_conv_op = "conv" in op_name.lower()
                    if "convolution_dimension_numbers" in attr_name or (attr_name == "dimension_numbers" and is_conv_op):
                         try:
                             conv_nums = stablehlo.ConvDimensionNumbers(attr_val)
                             node["attributes"]["input_batch_dimension"] = int(conv_nums.input_batch_dimension)
                             node["attributes"]["input_feature_dimension"] = int(conv_nums.input_feature_dimension)
                             node["attributes"]["input_spatial_dimensions"] = [int(x) for x in conv_nums.input_spatial_dimensions]
                             
                             node["attributes"]["kernel_input_feature_dimension"] = int(conv_nums.kernel_input_feature_dimension)
                             node["attributes"]["kernel_output_feature_dimension"] = int(conv_nums.kernel_output_feature_dimension)
                             node["attributes"]["kernel_spatial_dimensions"] = [int(x) for x in conv_nums.kernel_spatial_dimensions]
                             
                             node["attributes"]["output_batch_dimension"] = int(conv_nums.output_batch_dimension)
                             node["attributes"]["output_feature_dimension"] = int(conv_nums.output_feature_dimension)
                             node["attributes"]["output_spatial_dimensions"] = [int(x) for x in conv_nums.output_spatial_dimensions]
                         except Exception as e:
                             if _DEBUG: print(f"[MLX-Parser] Failed to parse conv_dims: {e}")
                    
                    # Parse gather dimension numbers for gather ops
                    if attr_name == "dimension_numbers" and "gather" in op_name.lower():
                         try:
                             gather_nums = stablehlo.GatherDimensionNumbers(attr_val)
                             node["attributes"]["offset_dims"] = [int(x) for x in gather_nums.offset_dims]
                             node["attributes"]["collapsed_slice_dims"] = [int(x) for x in gather_nums.collapsed_slice_dims]
                             node["attributes"]["start_index_map"] = [int(x) for x in gather_nums.start_index_map]
                             node["attributes"]["index_vector_dim"] = int(gather_nums.index_vector_dim)
                         except Exception as e:
                             if _DEBUG: print(f"[MLX-Parser] Failed to parse gather_dims: {e}")

                    # Handle pad low/high/interior attributes (DenseI64ArrayAttr)
                    if attr_name in ("low", "high", "interior") or "edge_padding" in attr_name or "interior_padding" in attr_name:
                         try:
                             vals = [int(x) for x in attr_val]
                             node["attributes"][attr_name] = vals
                         except Exception as e:
                             if _DEBUG: print(f"[MLX-Parser] Failed to parse pad attr {attr_name}: {e}")

                    if "Dense" in type_str and "ElementsAttr" in type_str:
                         try:
                             vals = []
                             # First try iterating - this will fail for complex types
                             try:
                                 for x in attr_val:
                                     if hasattr(x, "item"): vals.append(x.item())
                                     else: vals.append(x)
                                 # Success - process as normal
                                 if attr_name == "value":
                                     if "Int" in type_str:
                                         node["attributes"]["int_value"] = vals
                                         node["attributes"]["value_type"] = "dense_int"
                                     else:
                                         node["attributes"]["value"] = vals
                                         node["attributes"]["value_type"] = "dense"
                                 elif attr_name == "broadcast_dimensions":
                                     node["attributes"]["dims"] = [int(x) for x in vals]
                                 else:
                                     node["attributes"][attr_name] = vals
                             except TypeError as iter_err:
                                 # Iteration failed - likely complex type
                                 # Parse from string representation
                                 attr_str = str(attr_val)
                                 if "complex" in attr_str.lower():
                                     import re
                                     # Extract pairs like (0x7FC00000,0x7FC00000) or (1.0,2.0)
                                     pairs = re.findall(r'\(([-\d.xA-Fa-feE+]+),\s*([-\d.xA-Fa-feE+]+)\)', attr_str)
                                     complex_vals = []
                                     for real_str, imag_str in pairs:
                                         try:
                                             if 'x' in real_str.lower():
                                                 real = float('nan')
                                             else:
                                                 real = float(real_str)
                                             if 'x' in imag_str.lower():
                                                 imag = float('nan')
                                             else:
                                                 imag = float(imag_str)
                                             complex_vals.append((real, imag))
                                         except:
                                             complex_vals.append((float('nan'), float('nan')))
                                     
                                     if attr_name == "value":
                                         node["attributes"]["value"] = complex_vals
                                         node["attributes"]["value_type"] = "complex"
                                     else:
                                         node["attributes"][attr_name] = complex_vals
                                     if _DEBUG: print(f"[MLX-Parser] Parsed complex constant: {len(complex_vals)} values")
                                 else:
                                     # Non-complex iteration error, just use string
                                     raise iter_err
                         except Exception as e:
                             if _DEBUG: print(f"[MLX-Parser] Dense attr parse error: {e}")
                             node["attributes"][attr_name] = str(attr_val)
                    elif "ArrayAttr" in type_str:
                         try:
                             vals = []
                             for x in attr_val:
                                 if hasattr(x, "value"): vals.append(x.value)
                                 else: vals.append(x)
                             node["attributes"][attr_name] = vals
                         except:
                             node["attributes"][attr_name] = str(attr_val)
                    elif "IntegerAttr" in type_str:
                         try:
                             node["attributes"][attr_name] = int(attr_val.value)
                         except:
                             node["attributes"][attr_name] = str(attr_val)
                    else:
                        # Fallback
                        if attr_name == "comparison_direction":
                             s = str(attr_val)
                             clean = s.split(' ')[-1].replace('>', '').replace('"', '')
                             node["attributes"][attr_name] = clean
                        else:
                             node["attributes"][attr_name] = str(attr_val)

                # Region Parsing (Recursion)
                if op_name in ["stablehlo.while", "mhlo.while"]:
                    if len(inner_op.regions) >= 2:
                        # Region 0: Cond, Region 1: Body
                        # Pass the op's input IDs so subgraph block args map correctly
                        op_input_ids = [get_id(operand) for operand in inner_op.operands]
                        node["subgraphs"].append(parse_block(inner_op.regions[0].blocks[0], op_input_ids))
                        node["subgraphs"].append(parse_block(inner_op.regions[1].blocks[0], op_input_ids))
                
                elif op_name in ["stablehlo.scan", "mhlo.scan"]:
                    if len(inner_op.regions) >= 1:
                        node["subgraphs"].append(parse_block(inner_op.regions[0].blocks[0]))
                
                elif op_name in ["stablehlo.case", "mhlo.case"]:
                    # Case has multiple regions - one per branch
                    for region in inner_op.regions:
                        node["subgraphs"].append(parse_block(region.blocks[0]))

                # Reduce Handlers (Argmax/Argmin)
                if op_name in ["stablehlo.reduce", "mhlo.reduce"]:
                     try:
                         # Still populate reduce_type for C++ optimization
                         # Also could populate subgraph if we want generic reduce
                         node["subgraphs"].append(parse_block(inner_op.regions[0].blocks[0]))
                         
                         # Legacy attribute detection for C++ quick path
                         region = inner_op.regions[0]
                         block = region.blocks[0]
                         has_gt = False
                         has_lt = False
                         for op in block.operations:
                             op_name_local = op.operation.name
                             if "compare" in op_name_local:
                                 # Use dict-style access for OpAttributeMap
                                 if "comparison_direction" in op.attributes:
                                     val = str(op.attributes["comparison_direction"])
                                     if "GT" in val or "GE" in val: has_gt = True
                                     if "LT" in val or "LE" in val: has_lt = True
                             
                             if "add" in op_name_local: node["attributes"]["reduce_type"] = "sum"
                             if "max" in op_name_local and "argmax" not in op_name_local: node["attributes"]["reduce_type"] = "max"
                             if "min" in op_name_local and "argmin" not in op_name_local: node["attributes"]["reduce_type"] = "min"
                             if "mul" in op_name_local: node["attributes"]["reduce_type"] = "prod"
                             if "or" in op_name_local and "xor" not in op_name_local: node["attributes"]["reduce_type"] = "or"
                             if "and" in op_name_local: node["attributes"]["reduce_type"] = "and"
                         
                         if has_gt:
                             node["attributes"]["reduce_type"] = "argmax"
                         elif has_lt:
                             node["attributes"]["reduce_type"] = "argmin"
                     except: pass
                
                # Reduce Window Handlers
                elif op_name in ["stablehlo.reduce_window", "mhlo.reduce_window"]:
                    if len(inner_op.regions) >= 1:
                        node["subgraphs"].append(parse_block(inner_op.regions[0].blocks[0]))
                
                # Select And Scatter (MaxPool Gradient)
                elif op_name in ["stablehlo.select_and_scatter", "mhlo.select_and_scatter"]:
                    # Region 0: select (comparator), Region 1: scatter (combiner)
                    if len(inner_op.regions) >= 2:
                        node["subgraphs"].append(parse_block(inner_op.regions[0].blocks[0]))
                        node["subgraphs"].append(parse_block(inner_op.regions[1].blocks[0]))

                local_graph["nodes"].append(node)
            return local_graph

        # Execution
        main_func = None
        # Try to find main
        if "main" in funcs: main_func = funcs["main"]
        elif funcs: main_func = list(funcs.values())[0]
        else:
             # Look in module_op explicitly
             for op in module_op.body.operations:
                 if op.operation.name == "func.func":
                     main_func = op
                     break
        
        result_graph = {}
        
        if main_func:
            result_graph = parse_block(main_func.regions[0].blocks[0])
            # Extract input signature
            try:
                input_shapes = []
                # main_func is OpView. type property gives FunctionType
                ftype = main_func.type
                for t in ftype.inputs:
                    info = get_type_info(t)
                    input_shapes.append(info.get("shape", []))
                result_graph["input_shapes"] = input_shapes
            except Exception as e:
                if _DEBUG: print(f"[MLX-Parser] Failed to extract input signature: {e}", flush=True)
        else:
            return {"error": "No main function found"}

        # Process other functions
        result_graph["functions"] = {}
        for name, func_op in funcs.items():
            if func_op == main_func: continue
            # Parse function body
            result_graph["functions"][name] = parse_block(func_op.regions[0].blocks[0])
            
            # Extract signature
            try:
                f_input_shapes = []
                ftype = func_op.type
                for t in ftype.inputs:
                    info = get_type_info(t)
                    f_input_shapes.append(info.get("shape", []))
                result_graph["functions"][name]["input_shapes"] = f_input_shapes
            except: pass
        return result_graph

if __name__ == "__main__":
    import jax.numpy as jnp
    @jax.jit
    def f(x, y):
        return x * y + 1.0
    
    lowered = f.lower(jnp.array([1.0]), jnp.array([1.0]))
    mlir_module = lowered.compiler_ir()
    buf = io.BytesIO()
    mlir_module.operation.write_bytecode(buf)
    bytecode = buf.getvalue()
    
    result = parse_bytecode(bytecode)
    import json
    print(json.dumps(result, indent=2))
