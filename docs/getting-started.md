# Avalon · Qwen3.8-Flash-Next

This plugin runs **one model only: Qwen3.8-Flash-Next**, on
[Avalon](https://github.com/freriadoc/Avalon), a llama.cpp fork tuned for it
(experts in RAM, a GPU expert cache, MTP speculative decoding). Its GGUF files
use types and a layout stock llama.cpp does not have, and Avalon gives other
models none of its speed: deploy any other model with llama.cpp.

## 1. Download the model

On every machine that will serve it, download the Avalon GGUF from Hugging
Face, [freriadoc/Qwen3.8-Flash-Next-Avalon-GGUF](https://huggingface.co/freriadoc/Qwen3.8-Flash-Next-Avalon-GGUF)
(~89 GiB: 33 parts, the `MTP/` draft head and `moe-cache-ranking.bin`), with
the Hugging Face plugin or by hand:

```bash
hf download freriadoc/Qwen3.8-Flash-Next-Avalon-GGUF --local-dir Qwen3.8-Flash-Next-Avalon-GGUF
```

Keep the folder as it is: the plugin finds the MTP head in its `MTP/` folder.

## 2. Deploy it

In **Deploy**, pick the model's first part
(`...-IQ3KS-00001-of-00033.gguf`) and choose **Avalon** as the inference
service. One machine runs the whole model; several split its 48 layers in
ranges.

Avalon's own settings for best speed: batch and ubatch 9536, one session,
flash attention on, KV q8_0 / q4_0.

## Cards

NVIDIA (CUDA, Turing to Blackwell) and AMD (ROCm, RDNA3 and RDNA4). The first
load on a machine downloads the runtime its cards need.
