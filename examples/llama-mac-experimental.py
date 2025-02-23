import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from moba import register_moba, MoBAConfig

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--moba-chunk-size", type=int, default=4096)
    parser.add_argument("--moba-topk", type=int, default=12)
    parser.add_argument(
        "--attn",
        default="moba",
        help="choose attention backend",
        choices=["flash_attention_2", "moba", "moba_naive"],
    )
    args = parser.parse_args()

    register_moba(MoBAConfig(args.moba_chunk_size, args.moba_topk))

    # **Fix: Prevent Offloading Issues**
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.float32 if device == "mps" else torch.float16,  # ✅ Use float32 for MPS
        device_map=device,  # ✅ Ensures full model stays in memory
        offload_state_dict=False,  # ✅ Prevents layer offloading to disk
    )

    if torch.backends.mps.is_available():
        model = model.to(torch.float32)  # ✅ Prevent MPS precision errors

    tknz = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # **Fix: Set padding token**
    tknz.pad_token = tknz.eos_token

    prompt = "how are you?"

    # **Fix: Ensure correct tokenization**
    input_tokens = tknz(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )

    input_ids = input_tokens["input_ids"].to(model.device)
    attention_mask = input_tokens["attention_mask"].to(model.device)

    # **Fix: Ensure correct input shape**
    if len(input_ids.shape) == 1:
        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)

    # **Fix: Adjust sampling for more coherent text**
    tokens = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=64,
        do_sample=True,
        temperature=0.5,  # ✅ Reduce randomness
        top_k=40,  # ✅ Reduce top-k to limit low-quality tokens
        top_p=0.8,  # ✅ Lower probability mass for nucleus sampling
        repetition_penalty=1.2,  # ✅ Stronger penalty for repeated words
    )

    # **Fix: Decode output properly**
    decoded_output = tknz.decode(tokens[0], skip_special_tokens=True)
    print(decoded_output.strip())  # ✅ Ensure clean final output
