# Avalon across machines (feature level L6) - design

Status: proposal, nothing implemented beyond the two fork branches below.

## What Avalon already has

With two GPUs in one process, Avalon runs a two-stage pipeline:

- the last stage's `llama_context` owns `stage_first`, a second `llama_context`
  over layers `[stage_il0, stage_il1)` on the other device
  (`llama_context_params.stage_il0/stage_il1/stage_device`);
- the memory of the pair is `llama_memory_stage_pair`, so every sequence
  operation (`seq_rm`, `seq_cp`, `seq_keep`, checkpoints) reaches both KV caches;
- a batch is cut into ubatches the same way on both sides: `stage_begin`
  (the batch and its breaks), then `stage_step` writes one ubatch of the first
  stage's output rows into a buffer the last stage reads as `cparams.stage_in`
  (`qwen4exp`: the wide `n_embd*hc` residual);
- the prompt is pipelined: the first stage works on ubatch k+1 while the last
  finishes ubatch k.

On `pom` the fork also carries, from the POM's llama.cpp fork:

- `feature/external-tensor-data`: a header-only virtual slice loads the layers
  from the whole model's files (the node's `<SLUG>_VIRTUAL_SLICES=1`);
- `feature/hidden-state-input`: a batch of explicit row width (the wide
  residual) is accepted as input, and a continuation hop does not broadcast it
  again.

## Proposal: a remote first stage

Make `stage_first` able to live in another process, behind the same interface:

1. **Engine (fork, neutral branch `feature/remote-stage`).**
   - `llama-stage` tool (`tools/stage/`): loads a model (or a slice) with
     `stage_il0/stage_il1`, listens on a loopback TCP port, and serves three
     framed requests: `BEGIN` (batch tokens, positions, seq ids, breaks),
     `STEP` (returns one ubatch of output rows as f32) and `MEM` (a memory
     operation: `seq_rm`, `seq_cp`, `seq_keep`, `seq_add`, checkpoint save and
     restore). Frames: little-endian `u32` length, a kind byte, payload.
   - `llama_context`: `stage_remote` (host:port and a token) in the context
     params. When set, the context builds a `llama_stage_remote` in place of
     `stage_first`: `stage_begin`/`stage_step` send `BEGIN`/`STEP` and copy the
     returned rows into `stage_buf_in`; `llama_memory_stage_pair` forwards the
     memory operations as `MEM` frames. Nothing else in the context changes, so
     `llama-server` (slots, prompt cache, MTP, sampling, tool calls) runs
     unchanged on the last stage.
   - `llama-server --stage-remote host:port` passes it through; the token comes
     from the environment, never the command line.
2. **Plugin.**
   - `STAGES = true`, `pom_settings` gains `prefill_max_inflight_chunks` and
     `hidden_state_dtype`.
   - First stage (`link: {}`): start `llama-stage` on a loopback port, make a
     secret, and answer `engine.link` with `{port, token}`. The POM node carries
     the frames between machines in the deployment's `hidden_state_dtype` and
     hands them back as f32 (`engine-contract.md`, stages).
   - Last stage (`link: {token, prev_port, max_inflight}`): start
     `llama-server --stage-remote 127.0.0.1:<prev_port>` with the token in its
     environment; it is the only stage that takes requests.
   - The model file is the node's slice, real or virtual (header only), so no
     `stage_ranges` and no range cutting in the plugin.
3. **Middle stages** (three or more machines): `llama-stage` with both a
   listening side and a `--stage-remote` upstream. Later; two stages first.

## Open questions

- The POM link carries frames of kind `2` (tensor) and `1` (raw). `STEP`
  answers are tensors; `BEGIN` and `MEM` are raw. Confirm the node's link
  passes raw frames both ways in order (`plugin_stage_link.rs`).
- The MTP draft (`--spec-type draft-mtp`) reads the target's last hidden rows
  on the last stage only; it should not need the first stage. Verify.
- The expert cache and the RAM arena size themselves from the whole model;
  with a slice they must size from the stage's own experts.
- Latency: generation sends one `STEP` per token per stage; the in-process
  pipeline hides it, the network will not. Measure before tuning.

## Order of work

1. `llama-stage` serving `BEGIN/STEP/MEM` on one machine, with a test that a
   two-process run equals the two-GPU in-process run (greedy).
2. `stage_remote` in `llama_context` and `--stage-remote` in `llama-server`.
3. Plugin: `engine.link`, the two launches, `STAGES`.
4. Two machines through the POM link; then middle stages.
