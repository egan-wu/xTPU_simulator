#!/usr/bin/env python3
"""
Generate test ONNX models for xTPU compiler pipeline validation.

Creates small INT8 models that fit the xTPU simulator's constraints:
  - 4x4 matmul tiles (matching ComputeEngine MATMUL)
  - INT8 weights/activations, INT32 accumulation
  - Small enough for LocalMemory (64KB per buffer)

Models are inspired by mainstream architectures but scaled down to 4x4.
"""

import os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(graph, name):
    """Wrap graph in model, validate, save."""
    model = helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", 13)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    path = os.path.join(OUTPUT_DIR, f"{name}.onnx")
    onnx.save(model, path)
    print(f"  Created {path}")
    return path


def _const_i8(name, data):
    return numpy_helper.from_array(np.array(data, dtype=np.int8), name=name)

def _const_i32(name, data):
    return numpy_helper.from_array(np.array(data, dtype=np.int32), name=name)

def _const_i64(name, data):
    return numpy_helper.from_array(np.array(data, dtype=np.int64), name=name)

def _rand_i8(name, shape):
    return numpy_helper.from_array(
        np.random.randint(-3, 4, size=shape, dtype=np.int8), name=name)


# ===========================================================================
# Original 3 models (P5-2 baseline)
# ===========================================================================

def make_single_matmul():
    """Single 4x4 INT8 MatMul: Y = X @ W."""
    X = helper.make_tensor_value_info("X", TensorProto.INT8, [1, 4, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.INT32, [1, 4, 4])
    W_init = _const_i8("W", np.eye(4).reshape(1, 4, 4))

    graph = helper.make_graph(
        [helper.make_node("MatMul", ["X", "W"], ["Y"], name="matmul0")],
        "single_matmul", [X], [Y], [W_init])
    return _make_model(graph, "single_matmul_i8")


def make_two_layer_mlp():
    """Two-layer MLP: Y = ReLU(X @ W1 + B1) @ W2 + B2.

    Architecture: Classic feedforward neural network (BERT FFN, GPT MLP).
    """
    X = helper.make_tensor_value_info("X", TensorProto.INT8, [1, 4, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.INT32, [1, 4, 4])

    w1 = np.array([[1,0,0,0],[0,2,0,0],[0,0,1,0],[0,0,0,1]],
                   dtype=np.int8).reshape(1,4,4)
    inits = [
        numpy_helper.from_array(w1, name="W1"),
        _const_i32("B1", [1,1,1,1]),
        _const_i32("W2", np.eye(4, dtype=np.int32).reshape(1,4,4)),
        _const_i32("B2", [0,0,0,0]),
        _const_i64("rs1", [1,1,4]),
        _const_i64("rs2", [1,1,4]),
    ]
    nodes = [
        helper.make_node("MatMul", ["X","W1"], ["mm1"], name="matmul1"),
        helper.make_node("Reshape", ["B1","rs1"], ["B1_3d"], name="reshape1"),
        helper.make_node("Add", ["mm1","B1_3d"], ["add1"], name="add1"),
        helper.make_node("Relu", ["add1"], ["relu1"], name="relu1"),
        helper.make_node("MatMul", ["relu1","W2"], ["mm2"], name="matmul2"),
        helper.make_node("Reshape", ["B2","rs2"], ["B2_3d"], name="reshape2"),
        helper.make_node("Add", ["mm2","B2_3d"], ["Y"], name="add2"),
    ]
    graph = helper.make_graph(nodes, "two_layer_mlp", [X], [Y], inits)
    return _make_model(graph, "two_layer_mlp_i8")


def make_gemm_mlp():
    """Single-layer MLP using Gemm: Y = X @ W^T + B.

    Architecture: PyTorch nn.Linear export pattern.
    """
    X = helper.make_tensor_value_info("X", TensorProto.INT8, [4, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.INT32, [4, 4])

    inits = [
        _const_i8("W", np.eye(4)),
        _const_i32("B", [1,2,3,4]),
    ]
    nodes = [
        helper.make_node("Gemm", ["X","W","B"], ["Y"], name="gemm0", transB=1),
    ]
    graph = helper.make_graph(nodes, "gemm_mlp", [X], [Y], inits)
    return _make_model(graph, "gemm_mlp_i8")


# ===========================================================================
# New models inspired by mainstream architectures
# ===========================================================================

def make_transformer_attention():
    """Simplified Transformer Self-Attention (single-head).

    Architecture: Transformer (Vaswani et al. 2017) / BERT / GPT
    Flow: X → Q,K,V projections → Attention = Q @ K^T → Attn @ V → Output proj

    Scaled down to 4x4:
      X:  [1, 4, 4] — 4 tokens, dim=4
      Wq: [1, 4, 4] — query projection
      Wk: [1, 4, 4] — key projection
      Wv: [1, 4, 4] — value projection
      Wo: [1, 4, 4] — output projection
      Q = X @ Wq  →  Attn = Q @ K^T  →  Ctx = Attn @ V  →  Y = Ctx @ Wo
    """
    X  = helper.make_tensor_value_info("X", TensorProto.INT8, [1, 4, 4])
    Y  = helper.make_tensor_value_info("Y", TensorProto.INT32, [1, 4, 4])

    # Projection matrices (small random values to avoid overflow)
    np.random.seed(42)
    inits = [
        _rand_i8("Wq", (1, 4, 4)),
        _rand_i8("Wk", (1, 4, 4)),
        _rand_i8("Wv", (1, 4, 4)),
        _const_i32("Wo", np.eye(4, dtype=np.int32).reshape(1, 4, 4)),
    ]

    nodes = [
        # Q = X @ Wq
        helper.make_node("MatMul", ["X", "Wq"], ["Q"], name="proj_q"),
        # K = X @ Wk
        helper.make_node("MatMul", ["X", "Wk"], ["K"], name="proj_k"),
        # V = X @ Wv
        helper.make_node("MatMul", ["X", "Wv"], ["V"], name="proj_v"),
        # K^T (transpose last two dims)
        helper.make_node("Transpose", ["K"], ["Kt"], name="transpose_k",
                         perm=[0, 2, 1]),
        # Attn = Q @ K^T  (attention scores, skip softmax for INT8)
        helper.make_node("MatMul", ["Q", "Kt"], ["Attn"], name="attn_scores"),
        # Ctx = Attn @ V  (context)
        helper.make_node("MatMul", ["Attn", "V"], ["Ctx"], name="attn_ctx"),
        # Y = Ctx @ Wo  (output projection)
        helper.make_node("MatMul", ["Ctx", "Wo"], ["Y"], name="proj_out"),
    ]

    graph = helper.make_graph(nodes, "transformer_attention", [X], [Y], inits)
    return _make_model(graph, "transformer_attention_i8")


def make_residual_block():
    """Residual Block (skip connection).

    Architecture: ResNet (He et al. 2015)
    Flow: Y = ReLU(X @ W1 + B1) @ W2 + X  (residual add)

    The skip connection is the defining feature of ResNet.
    """
    X = helper.make_tensor_value_info("X", TensorProto.INT8, [1, 4, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.INT32, [1, 4, 4])

    np.random.seed(123)
    inits = [
        _rand_i8("W1", (1, 4, 4)),
        _const_i32("B1", [1, 1, 1, 1]),
        _const_i8("W2", np.eye(4, dtype=np.int8).reshape(1, 4, 4)),
        _const_i64("rs1", [1, 1, 4]),
    ]

    nodes = [
        # Branch: X @ W1 + B1 → ReLU → @ W2
        helper.make_node("MatMul", ["X", "W1"], ["mm1"], name="res_mm1"),
        helper.make_node("Reshape", ["B1", "rs1"], ["B1_3d"], name="res_reshape"),
        helper.make_node("Add", ["mm1", "B1_3d"], ["add1"], name="res_add1"),
        helper.make_node("Relu", ["add1"], ["relu1"], name="res_relu"),
        helper.make_node("MatMul", ["relu1", "W2"], ["mm2"], name="res_mm2"),
        # Residual: mm2 + X (skip connection)
        helper.make_node("Add", ["mm2", "X"], ["Y"], name="res_skip"),
    ]

    graph = helper.make_graph(nodes, "residual_block", [X], [Y], inits)
    return _make_model(graph, "residual_block_i8")


def make_mlp_mixer_block():
    """MLP-Mixer style block.

    Architecture: MLP-Mixer (Tolstikhin et al. 2021)
    Flow: Token mixing → Channel mixing
      Token mix: X^T @ Wt → transpose back
      Channel mix: result @ Wc

    Key idea: alternating matmuls along different dimensions via transpose.
    """
    X = helper.make_tensor_value_info("X", TensorProto.INT8, [1, 4, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.INT32, [1, 4, 4])

    np.random.seed(77)
    inits = [
        _rand_i8("Wt", (1, 4, 4)),  # token-mixing weights
        _const_i8("Wc", np.eye(4, dtype=np.int8).reshape(1, 4, 4)),  # channel-mixing
    ]

    nodes = [
        # Token mixing: transpose → matmul → transpose back
        helper.make_node("Transpose", ["X"], ["Xt"], name="mix_t1",
                         perm=[0, 2, 1]),
        helper.make_node("MatMul", ["Xt", "Wt"], ["token_mixed"], name="mix_mm1"),
        helper.make_node("Transpose", ["token_mixed"], ["tm_back"],
                         name="mix_t2", perm=[0, 2, 1]),
        # Channel mixing: matmul
        helper.make_node("MatMul", ["tm_back", "Wc"], ["Y"], name="mix_mm2"),
    ]

    graph = helper.make_graph(nodes, "mlp_mixer_block", [X], [Y], inits)
    return _make_model(graph, "mlp_mixer_block_i8")


def make_gpt_block():
    """Simplified GPT-2 Transformer Block.

    Architecture: GPT-2 (Radford et al. 2019)
    Flow: Attn → Residual Add → FFN → Residual Add

    Combines self-attention with feedforward, both with residual connections.
    This is the fundamental building block of all GPT models.
    """
    X = helper.make_tensor_value_info("X", TensorProto.INT8, [1, 4, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.INT32, [1, 4, 4])

    np.random.seed(200)
    inits = [
        # Attention weights (combined QKV for simplicity)
        _rand_i8("Wqkv", (1, 4, 4)),
        _const_i8("Wv", np.eye(4, dtype=np.int8).reshape(1, 4, 4)),
        # FFN weights
        _rand_i8("Wff1", (1, 4, 4)),
        _const_i32("Bff1", [1, 1, 1, 1]),
        _const_i8("Wff2", np.eye(4, dtype=np.int8).reshape(1, 4, 4)),
        _const_i64("rs1", [1, 1, 4]),
    ]

    nodes = [
        # Self-attention (simplified: single matmul as proxy)
        helper.make_node("MatMul", ["X", "Wqkv"], ["attn_out"], name="gpt_attn"),
        # Attention + Transpose → matmul with V (simplified)
        helper.make_node("Transpose", ["attn_out"], ["attn_t"],
                         name="gpt_attn_t", perm=[0, 2, 1]),
        helper.make_node("MatMul", ["attn_t", "Wv"], ["attn_proj"],
                         name="gpt_attn_proj"),
        # Residual add #1: attn_proj + X
        helper.make_node("Add", ["attn_proj", "X"], ["res1"],
                         name="gpt_res1"),
        # FFN: Linear → ReLU → Linear
        helper.make_node("MatMul", ["res1", "Wff1"], ["ff1"],
                         name="gpt_ff1"),
        helper.make_node("Reshape", ["Bff1", "rs1"], ["Bff1_3d"],
                         name="gpt_reshape"),
        helper.make_node("Add", ["ff1", "Bff1_3d"], ["ff1_bias"],
                         name="gpt_ff1_add"),
        helper.make_node("Relu", ["ff1_bias"], ["ff1_relu"],
                         name="gpt_relu"),
        helper.make_node("MatMul", ["ff1_relu", "Wff2"], ["ff2"],
                         name="gpt_ff2"),
        # Residual add #2: ff2 + res1
        helper.make_node("Add", ["ff2", "res1"], ["Y"],
                         name="gpt_res2"),
    ]

    graph = helper.make_graph(nodes, "gpt_block", [X], [Y], inits)
    return _make_model(graph, "gpt_block_i8")


def make_multi_head_attention():
    """Multi-Head Attention with 2 heads (head_dim=2).

    Architecture: Multi-Head Attention (Transformer)
    Uses Gemm for projections, split via reshape, concat via reshape.

    2 heads × head_dim=2 = model_dim=4
    Q, K, V projections → reshape to heads → per-head attention → concat
    """
    X = helper.make_tensor_value_info("X", TensorProto.INT8, [4, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.INT32, [4, 4])

    np.random.seed(99)
    inits = [
        _const_i8("Wq", np.eye(4)),
        _const_i32("Bq", [0, 0, 0, 0]),
        _const_i8("Wk", np.eye(4)),
        _const_i32("Bk", [0, 0, 0, 0]),
        _const_i8("Wv", np.eye(4)),
        _const_i32("Bv", [0, 0, 0, 0]),
        _const_i8("Wo", np.eye(4)),
        _const_i32("Bo", [0, 0, 0, 0]),
    ]

    nodes = [
        # Projections via Gemm (nn.Linear pattern)
        helper.make_node("Gemm", ["X", "Wq", "Bq"], ["Q"],
                         name="mha_proj_q", transB=1),
        helper.make_node("Gemm", ["X", "Wk", "Bk"], ["K"],
                         name="mha_proj_k", transB=1),
        helper.make_node("Gemm", ["X", "Wv", "Bv"], ["V"],
                         name="mha_proj_v", transB=1),
        # K^T for attention scores
        helper.make_node("Transpose", ["K"], ["Kt"], name="mha_kt",
                         perm=[1, 0]),
        # Attn = Q @ K^T
        helper.make_node("MatMul", ["Q", "Kt"], ["Attn"], name="mha_attn"),
        # Ctx = Attn @ V
        helper.make_node("MatMul", ["Attn", "V"], ["Ctx"], name="mha_ctx"),
        # Output projection
        helper.make_node("Gemm", ["Ctx", "Wo", "Bo"], ["Y"],
                         name="mha_proj_out", transB=1),
    ]

    graph = helper.make_graph(nodes, "multi_head_attention", [X], [Y], inits)
    return _make_model(graph, "multi_head_attention_i8")


def make_bottleneck_block():
    """Bottleneck Block.

    Architecture: ResNet-50/101/152 bottleneck (He et al. 2015)
    Flow: 1x1 reduce → 3x3 conv (simulated as matmul) → 1x1 expand + skip

    We simulate the 1x1 convolutions as matmul (they're equivalent for 1x1).
    The "3x3 conv" middle layer is also a matmul (our hardware limit).

    Pattern: reduce(4→4) → process(4→4) → expand(4→4) + residual
    """
    X = helper.make_tensor_value_info("X", TensorProto.INT8, [1, 4, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.INT32, [1, 4, 4])

    np.random.seed(55)
    inits = [
        # 1x1 "reduce" (pointwise)
        _rand_i8("W_reduce", (1, 4, 4)),
        # "3x3 conv" simulated as matmul
        _rand_i8("W_mid", (1, 4, 4)),
        # 1x1 "expand" (pointwise)
        _const_i8("W_expand", np.eye(4, dtype=np.int8).reshape(1, 4, 4)),
    ]

    nodes = [
        helper.make_node("MatMul", ["X", "W_reduce"], ["reduced"],
                         name="bn_reduce"),
        helper.make_node("Relu", ["reduced"], ["reduced_relu"],
                         name="bn_relu1"),
        helper.make_node("MatMul", ["reduced_relu", "W_mid"], ["mid"],
                         name="bn_mid"),
        helper.make_node("Relu", ["mid"], ["mid_relu"],
                         name="bn_relu2"),
        helper.make_node("MatMul", ["mid_relu", "W_expand"], ["expanded"],
                         name="bn_expand"),
        # Residual skip
        helper.make_node("Add", ["expanded", "X"], ["Y"],
                         name="bn_skip"),
    ]

    graph = helper.make_graph(nodes, "bottleneck_block", [X], [Y], inits)
    return _make_model(graph, "bottleneck_block_i8")


def make_encoder_decoder():
    """Encoder-Decoder Attention.

    Architecture: Original Transformer / T5 / BART
    Flow: Encoder processes source, Decoder cross-attends to encoder output.

    Enc: X_enc @ We → enc_out
    Dec: X_dec @ Wd → dec_q, cross-attend with enc_out
    """
    X_enc = helper.make_tensor_value_info("X_enc", TensorProto.INT8, [1, 4, 4])
    X_dec = helper.make_tensor_value_info("X_dec", TensorProto.INT8, [1, 4, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.INT32, [1, 4, 4])

    np.random.seed(88)
    inits = [
        _rand_i8("We", (1, 4, 4)),   # encoder projection
        _rand_i8("Wd", (1, 4, 4)),   # decoder query projection
        _const_i8("Wo", np.eye(4, dtype=np.int8).reshape(1, 4, 4)),
    ]

    nodes = [
        # Encoder
        helper.make_node("MatMul", ["X_enc", "We"], ["enc_kv"],
                         name="enc_proj"),
        # Decoder query
        helper.make_node("MatMul", ["X_dec", "Wd"], ["dec_q"],
                         name="dec_proj"),
        # Cross-attention: dec_q @ enc_kv^T
        helper.make_node("Transpose", ["enc_kv"], ["enc_kt"],
                         name="enc_transpose", perm=[0, 2, 1]),
        helper.make_node("MatMul", ["dec_q", "enc_kt"], ["cross_attn"],
                         name="cross_attn"),
        # Attention @ encoder values
        helper.make_node("MatMul", ["cross_attn", "enc_kv"], ["ctx"],
                         name="cross_ctx"),
        # Output projection
        helper.make_node("MatMul", ["ctx", "Wo"], ["Y"],
                         name="out_proj"),
    ]

    graph = helper.make_graph(nodes, "encoder_decoder", [X_enc, X_dec], [Y], inits)
    return _make_model(graph, "encoder_decoder_i8")


def make_deep_mlp():
    """4-Layer Deep MLP.

    Architecture: Deep feedforward network / DNN classifier
    Flow: X → Linear+ReLU → Linear+ReLU → Linear+ReLU → Linear

    Tests deep sequential chains (common in recommendation systems, NLP classifiers).
    """
    X = helper.make_tensor_value_info("X", TensorProto.INT8, [1, 4, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.INT32, [1, 4, 4])

    np.random.seed(333)
    inits = [
        _rand_i8("W1", (1, 4, 4)),
        _const_i32("B1", [1, 0, 1, 0]),
        _rand_i8("W2", (1, 4, 4)),
        _const_i32("B2", [0, 1, 0, 1]),
        _rand_i8("W3", (1, 4, 4)),
        _const_i32("B3", [1, 1, 0, 0]),
        _const_i8("W4", np.eye(4, dtype=np.int8).reshape(1, 4, 4)),
        _const_i64("rs", [1, 1, 4]),
    ]

    nodes = []
    prev = "X"
    for i, (w, b) in enumerate([(f"W{j}", f"B{j}") for j in range(1, 4)], 1):
        mm_out = f"mm{i}"
        nodes.append(helper.make_node("MatMul", [prev, w], [mm_out],
                                      name=f"deep_mm{i}"))
        b_3d = f"{b}_3d"
        nodes.append(helper.make_node("Reshape", [b, "rs"], [b_3d],
                                      name=f"deep_rs{i}"))
        add_out = f"add{i}"
        nodes.append(helper.make_node("Add", [mm_out, b_3d], [add_out],
                                      name=f"deep_add{i}"))
        relu_out = f"relu{i}"
        nodes.append(helper.make_node("Relu", [add_out], [relu_out],
                                      name=f"deep_relu{i}"))
        prev = relu_out

    # Final layer (no ReLU)
    nodes.append(helper.make_node("MatMul", [prev, "W4"], ["Y"],
                                  name="deep_mm4"))

    graph = helper.make_graph(nodes, "deep_mlp", [X], [Y], inits)
    return _make_model(graph, "deep_mlp_i8")


def make_dual_path_network():
    """Dual-Path Network (DPN).

    Architecture: DPN (Chen et al. 2017) / two-branch networks
    Two parallel branches process input, results are added.
    Tests parallel data flow (common in Inception, DPN, multi-branch nets).
    """
    X = helper.make_tensor_value_info("X", TensorProto.INT8, [1, 4, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.INT32, [1, 4, 4])

    np.random.seed(444)
    inits = [
        _rand_i8("Wa", (1, 4, 4)),   # branch A
        _rand_i8("Wb", (1, 4, 4)),   # branch B
        _const_i8("Wa2", np.eye(4, dtype=np.int8).reshape(1, 4, 4)),
        _const_i8("Wb2", np.eye(4, dtype=np.int8).reshape(1, 4, 4)),
    ]

    nodes = [
        # Branch A: X @ Wa → ReLU → @ Wa2
        helper.make_node("MatMul", ["X", "Wa"], ["a1"], name="dpn_a1"),
        helper.make_node("Relu", ["a1"], ["a1_relu"], name="dpn_a_relu"),
        helper.make_node("MatMul", ["a1_relu", "Wa2"], ["a2"], name="dpn_a2"),
        # Branch B: X @ Wb → ReLU → @ Wb2
        helper.make_node("MatMul", ["X", "Wb"], ["b1"], name="dpn_b1"),
        helper.make_node("Relu", ["b1"], ["b1_relu"], name="dpn_b_relu"),
        helper.make_node("MatMul", ["b1_relu", "Wb2"], ["b2"], name="dpn_b2"),
        # Merge: A + B
        helper.make_node("Add", ["a2", "b2"], ["Y"], name="dpn_merge"),
    ]

    graph = helper.make_graph(nodes, "dual_path_network", [X], [Y], inits)
    return _make_model(graph, "dual_path_network_i8")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("=== Generating test ONNX models ===\n")

    print("[Baseline models (P5-2)]")
    make_single_matmul()
    make_two_layer_mlp()
    make_gemm_mlp()

    print("\n[Mainstream architecture models]")
    make_transformer_attention()     # Transformer / BERT / GPT attention
    make_residual_block()            # ResNet skip connection
    make_mlp_mixer_block()           # MLP-Mixer token/channel mixing
    make_gpt_block()                 # GPT-2 transformer block
    make_multi_head_attention()      # Multi-Head Attention with Gemm
    make_bottleneck_block()          # ResNet-50 bottleneck
    make_encoder_decoder()           # T5/BART encoder-decoder
    make_deep_mlp()                  # 4-layer DNN
    make_dual_path_network()         # DPN two-branch

    print("\nAll test models generated.")
