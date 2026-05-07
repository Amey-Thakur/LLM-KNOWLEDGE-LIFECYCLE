import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.nn.functional as F
import numpy as np

def calculate_dsync():
    print("Loading GPT-2 model...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()

    # Define the queries
    prompt_no_rag = "Question: Is Vioxx safe to prescribe? Answer: Vioxx is considered"
    
    context = "Context: In September 2004, Merck voluntarily withdrew Vioxx after trials revealed increased cardiovascular risks.\n"
    prompt_rag = context + prompt_no_rag

    with torch.no_grad():
        # Get logits for No RAG
        inputs_no_rag = tokenizer(prompt_no_rag, return_tensors='pt')
        outputs_no_rag = model(**inputs_no_rag)
        logits_no_rag = outputs_no_rag.logits[0, -1, :]
        probs_no_rag = F.softmax(logits_no_rag, dim=-1)

        # Get logits for RAG
        inputs_rag = tokenizer(prompt_rag, return_tensors='pt')
        outputs_rag = model(**inputs_rag)
        logits_rag = outputs_rag.logits[0, -1, :]
        probs_rag = F.softmax(logits_rag, dim=-1)

        # Calculate KL Divergence D_KL(P_RAG || P_No_RAG)
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        probs_no_rag_clamped = torch.clamp(probs_no_rag, min=epsilon)
        probs_rag_clamped = torch.clamp(probs_rag, min=epsilon)

        kl_div = torch.sum(probs_rag_clamped * torch.log(probs_rag_clamped / probs_no_rag_clamped)).item()

        # Let's also look at the top predictions to understand the divergence
        top_k = 5
        print("\n--- Top Predictions (No RAG) ---")
        top_probs_no, top_indices_no = torch.topk(probs_no_rag, top_k)
        for p, idx in zip(top_probs_no, top_indices_no):
            print(f"{tokenizer.decode(idx)}: {p.item():.4f}")

        print("\n--- Top Predictions (With RAG) ---")
        top_probs_rag, top_indices_rag = torch.topk(probs_rag, top_k)
        for p, idx in zip(top_probs_rag, top_indices_rag):
            print(f"{tokenizer.decode(idx)}: {p.item():.4f}")

        print(f"\n=========================================")
        print(f"Calculated Lifecycle Desynchronisation (D_sync): {kl_div:.4f}")
        print(f"=========================================")

if __name__ == "__main__":
    calculate_dsync()
