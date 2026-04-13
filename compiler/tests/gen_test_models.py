#!/usr/bin/env python3
"""
Generate test ONNX models for xtpu-import validation (P5-2).

Creates small INT8 models that fit the xTPU simulator's constraints:
  - 4×4 matmul tiles (matching ComputeEngine MATMUL)
  - INT8 weights/activations, INT32 accumulation
  - Small enough for LocalMemory (64KB per buffer)
"""

import os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def make_single_matmul():
    """Single 4×4 INT8 MatMul: Y = X @ W."""
    X = helper.make_tensor_value_info("X", TensorProto.INT8, [1, 4, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.INT32, [1, 4, 4])

    W_data = np.eye(4, dtype=np.int8)
    W_init = numpy_helper.from_array(W_data.reshape(1, 4, 4), name="W")

    matmul = helper.make_node("MatMul", ["X", "W"], ["Y"], name="matmul0")

    graph = helper.make_graph([matmul], "single_matmul",
                              [X], [Y], [W_init])
    model = helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", 13)])
    model.ir_version = 7
    onnx.checker.check_model(model)

    path = os.path.join(OUTPUT_DIR, "single_matmul_i8.onnx")
    onnx.save(model, path)
    print(f"Created {path}")
    return path


def make_two_layer_mlp():
    """Two-layer MLP: Y = ReLU(X @ W1 + B1) @ W2 + B2.

    Shapes (batch=1):
      X:  [1, 4, 4] (INT8)
      W1: [1, 4, 4] (INT8)  → MatMul → [1, 4, 4] (INT32)
      B1: [4] (INT32)        → Add    → [1, 4, 4] (INT32)
                              → ReLU   → [1, 4, 4] (INT32)
      W2: [1, 4, 4] (INT32)  → MatMul → [1, 4, 4] (INT32)
      B2: [4] (INT32)        → Add    → [1, 4, 4] (INT32)
    """
    X = helper.make_tensor_value_info("X", TensorProto.INT8, [1, 4, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.INT32, [1, 4, 4])

    # Layer 1 weights
    w1 = np.array([[1, 0, 0, 0],
                    [0, 2, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]], dtype=np.int8).reshape(1, 4, 4)
    W1_init = numpy_helper.from_array(w1, name="W1")

    b1 = np.array([1, 1, 1, 1], dtype=np.int32)
    B1_init = numpy_helper.from_array(b1, name="B1")

    # Layer 2 weights
    w2 = np.eye(4, dtype=np.int32).reshape(1, 4, 4)
    W2_init = numpy_helper.from_array(w2, name="W2")

    b2 = np.array([0, 0, 0, 0], dtype=np.int32)
    B2_init = numpy_helper.from_array(b2, name="B2")

    # Graph nodes
    matmul1 = helper.make_node("MatMul", ["X", "W1"], ["mm1"],
                               name="matmul1")
    # Reshape B1 for broadcasting (tosa.add needs compatible shapes)
    reshape1_shape = numpy_helper.from_array(
        np.array([1, 1, 4], dtype=np.int64), name="reshape1_shape")
    reshape1 = helper.make_node("Reshape", ["B1", "reshape1_shape"],
                                ["B1_3d"], name="reshape1")
    add1 = helper.make_node("Add", ["mm1", "B1_3d"], ["add1"],
                            name="add1")
    relu1 = helper.make_node("Relu", ["add1"], ["relu1"], name="relu1")

    matmul2 = helper.make_node("MatMul", ["relu1", "W2"], ["mm2"],
                               name="matmul2")
    reshape2_shape = numpy_helper.from_array(
        np.array([1, 1, 4], dtype=np.int64), name="reshape2_shape")
    reshape2 = helper.make_node("Reshape", ["B2", "reshape2_shape"],
                                ["B2_3d"], name="reshape2")
    add2 = helper.make_node("Add", ["mm2", "B2_3d"], ["Y"], name="add2")

    graph = helper.make_graph(
        [matmul1, reshape1, add1, relu1, matmul2, reshape2, add2],
        "two_layer_mlp",
        [X], [Y],
        [W1_init, B1_init, W2_init, B2_init, reshape1_shape, reshape2_shape])
    model = helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", 13)])
    model.ir_version = 7
    onnx.checker.check_model(model)

    path = os.path.join(OUTPUT_DIR, "two_layer_mlp_i8.onnx")
    onnx.save(model, path)
    print(f"Created {path}")
    return path


def make_gemm_mlp():
    """Single-layer MLP using Gemm: Y = X @ W^T + B.

    Gemm op is common in ONNX exports from PyTorch Linear layers.
    Shapes: X[4,4] @ W^T[4,4] + B[4] → Y[4,4] (all INT8/INT32)
    """
    X = helper.make_tensor_value_info("X", TensorProto.INT8, [4, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.INT32, [4, 4])

    w = np.eye(4, dtype=np.int8)
    W_init = numpy_helper.from_array(w, name="W")

    b = np.array([1, 2, 3, 4], dtype=np.int32)
    B_init = numpy_helper.from_array(b, name="B")

    gemm = helper.make_node("Gemm", ["X", "W", "B"], ["Y"],
                            name="gemm0", transB=1)

    graph = helper.make_graph([gemm], "gemm_mlp",
                              [X], [Y], [W_init, B_init])
    model = helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", 13)])
    model.ir_version = 7
    onnx.checker.check_model(model)

    path = os.path.join(OUTPUT_DIR, "gemm_mlp_i8.onnx")
    onnx.save(model, path)
    print(f"Created {path}")
    return path


if __name__ == "__main__":
    make_single_matmul()
    make_two_layer_mlp()
    make_gemm_mlp()
    print("\nAll test models generated.")
