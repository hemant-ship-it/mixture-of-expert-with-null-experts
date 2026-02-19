

# ONNX, PyTorch, LLMs, and MoE:

## What Actually Works in 2026 (and Why Your Export Failed)

> *If you’ve ever tried exporting a modern transformer or MoE model to ONNX and felt like the tooling was fighting you — it wasn’t just you. You ran into a real architectural boundary.*

This post explains:

* what ONNX actually is (not the marketing version),
* how PyTorch models are saved for **training vs inference**,
* how ONNX export really works under the hood,
* why ONNX breaks for modern **MoE architectures**,
* and **which models should never be exported to ONNX**.

This is written from the perspective of someone building real models, not toy demos.

---

## 1. What ONNX Actually Is (and What It Is Not)

**ONNX (Open Neural Network Exchange)** is:

* a **static computation graph format**
* consisting of:

  * tensors
  * operators (matmul, add, softmax, etc.)
  * fixed graph topology

ONNX is **not**:

* a Python runtime
* a dynamic execution engine
* a replacement for PyTorch’s control flow
* a “model checkpoint” like `.pt`

Think of ONNX as:

> “A frozen mathematical circuit that only knows how to move tensors through ops.”

No Python.
No `if`.
No loops based on runtime data.

---

## 2. PyTorch Models: Training vs Inference Artifacts

### Training checkpoints 

During training, people usually save **everything**:

```python
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "step": step,
    "config": cfg,
}, checkpoint)
```

This is correct for **resuming training**.

But it contains:

* optimizer moments (often 2–3× model size)
* Python objects (`Config`)
* training-only metadata

This is **not** an inference artifact.

---

### Inference artifacts (what deployment wants)

For inference, you only need:

* model weights
* model graph
* tokenizer

That’s it.

In PyTorch inference:

```python
torch.save(model.state_dict(), "model_weights.pt")
```

In ONNX inference:

* weights + graph are merged into **one `.onnx` file**

---

## 3. How ONNX Export Actually Works (Important)

ONNX export does **not** serialize your model code.

Instead:

> **PyTorch runs your model once and records the tensor operations that happened.**

That’s why `torch.onnx.export` needs a **dummy input**.

### Dummy input is NOT:

* your max context length
* a constraint on the model
* related to tokenizer

It is just a **probe** so PyTorch can see:

* embeddings
* attention
* MLPs
* output layers

With `dynamic_axes`, ONNX is told:

> “This dimension can vary later.”

---

## 4. Why ONNX Needs Static Control Flow

ONNX graphs must be valid for **all inputs**.

That means:

* both branches of computation must exist in the graph
* decisions must be tensor-based, not Python-based

This is fine for dense transformers:

```text
input → embedding → attention → mlp → logits
```

No branching based on data.

---

## 5. Why Modern LLMs Stress ONNX

Modern LLMs introduce:

* causal masking
* rotary embeddings (RoPE)
* dynamic shapes
* sometimes KV caches

These are **still mostly tensor-based**, so ONNX can usually handle them.

That’s why:

* dense GPT-style models
* BERT
* ViTs
  often export successfully.

---

## 6. Why MoE Breaks ONNX (This Is the Core Issue)

A **Mixture-of-Experts (MoE)** model adds:

* routing networks
* top-k expert selection
* sparse execution
* conditional expert execution

Typical MoE code contains things like:

```python
if not mask.any():
    return zeros
```

or

```python
if tokens_for_expert == 0:
    continue
```

This looks harmless in PyTorch.

But under the hood it does:

```python
mask.any().item() → Python bool
```

That single `.item()` is fatal for ONNX.

---

## 7. Why That `.item()` Kills Export

ONNX cannot represent:

* Python `if`
* Python `for` depending on tensor values
* tensor → Python boolean conversion

Because ONNX must decide **graph structure ahead of time**.

But your code says:

> “Decide the execution path based on runtime data.”

ONNX cannot do that.

So the exporter throws errors like:

```
GuardOnDataDependentSymNode
Converting a tensor to a Python bool
```

This is not a bug.
This is a **fundamental mismatch**.

---

## 8. Why the “New” PyTorch Exporter Makes It Worse

New PyTorch versions try to be clever.

They use:

* `torch.export`
* symbolic shape reasoning
* guard generation

This works for compiler research.
It fails badly for:

* MoE
* dynamic routing
* sparse execution

That’s why even dense transformers sometimes need the **legacy exporter**.

And for MoE, **symbolic export is basically impossible** today.

---

## 9. Can MoE Be Exported to ONNX at All?

### In theory: yes

### In practice: rarely worth it

To make MoE ONNX-compatible, you must:

* remove all Python control flow
* replace `if` with tensor masks
* compute *all* experts every time
* gate outputs with masks

That destroys:

* sparsity
* performance
* simplicity

At that point, you’ve basically turned MoE back into a dense model.

---

## 10. What People Actually Deploy in 2026

### PyTorch (most common)

* eager execution
* supports MoE naturally
* simple deployment
* works on CPU/GPU

This is what:

* Hugging Face Spaces
* research demos
* startups
  use.

---

### ONNX (selective use)

Used when:

* model is dense
* control flow is static
* no data-dependent routing
* portability matters more than flexibility

---

### High-end production (big teams)

* TensorRT / TensorRT-LLM
* XLA / JAX
* TVM
* custom C++ runtimes

These require **significant engineering effort**.

---

## 11. Models That Are NOT Suitable for ONNX

You should **avoid ONNX** if your model has:

❌ Data-dependent Python control flow
❌ MoE routing
❌ Conditional expert execution
❌ Dynamic loops over tensor values
❌ `.item()` used for decisions
❌ Sparse computation graphs

Examples:

* MoE Transformers
* Adaptive computation models
* Conditional routing networks
* Any model that *skips work* based on data

---

## 12. Models That Are Good Fits for ONNX

ONNX works well for:

✅ Dense transformers
✅ CNNs
✅ Vision transformers
✅ Audio models
✅ Static graph NLP models

Basically:

> “Same computation for every input.”



> **Modern MoE architectures violate ONNX’s core assumption: static computation graphs.**

That’s it.


> ONNX requires a static graph — every op, every branch, every expert must be known at export time. Your MoE fundamentally decides at runtime which experts to activate based on input data. These two things are incompatible.

> Baked trace (what just happened)
Only the experts activated by dummy input exist in the graph. Completely broken for real use.

1. Just use PyTorch directly
Run your model as-is in Python. Simplest, no conversion needed. Fine for learning/research.
2. TorchScript (torch.jit)
Converts your Python model into a format that can run without Python. Useful if you want to deploy in C++ or mobile. Still PyTorch ecosystem.
3. ONNX
A universal format. Convert once, run anywhere — on phones, browsers, different frameworks. But as you just saw, struggles with dynamic models like MoE.
4. GGUF (llama.cpp)
A format specifically for running LLMs efficiently on CPUs, even your laptop. Used by apps like Ollama, LM Studio. Requires converting weights.
5. TorchServe / vLLM / Triton
Servers that take your model and expose it as an API (POST /generate → response). Like turning your model into a mini ChatGPT API.