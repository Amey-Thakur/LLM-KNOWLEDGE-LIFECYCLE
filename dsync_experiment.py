import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.nn.functional as F
import math

def calculate_true_dsync():
    print("Loading GPT-2 model...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()

    prompt_no_rag = "Question: Is Vioxx safe to prescribe? Answer: Vioxx is considered"
    context = "Context: In September 2004, Merck voluntarily withdrew Vioxx after trials revealed increased cardiovascular risks.\n"
    prompt_rag = context + prompt_no_rag

    with torch.no_grad():
        inputs_rag = tokenizer(prompt_rag, return_tensors='pt')
        outputs_rag = model(**inputs_rag)
        logits_rag = outputs_rag.logits[0, -1, :]
        probs_rag = F.softmax(logits_rag, dim=-1)

        # Look at the probability of the target token " withdrawn"
        target_token = " withdrawn"
        target_token_id = tokenizer.encode(target_token)[0]
        p_actual_target = probs_rag[target_token_id].item()

        # D_sync is D_KL(P_target || P_actual)
        # P_target(withdrawn) = 1.0
        # D_KL = 1.0 * log(1.0 / p_actual_target)
        
        # Guard against zero probability
        p_actual_target = max(p_actual_target, 1e-10)
        dsync = math.log(1.0 / p_actual_target)

        print("\n--- Actual Output Distribution (P_actual) with RAG ---")
        top_k = 5
        top_probs_rag, top_indices_rag = torch.topk(probs_rag, top_k)
        for p, idx in zip(top_probs_rag, top_indices_rag):
            print(f"'{tokenizer.decode(idx)}': {p.item():.4f}")

        print(f"'{target_token}': {p_actual_target:.6f}")

        print(f"\n=========================================")
        print(f"Target Distribution (P_target) ['withdrawn']: 1.000")
        print(f"Actual Distribution (P_actual) ['withdrawn']: {p_actual_target:.6f}")
        print(f"Calculated True Lifecycle Desynchronisation (D_sync): {dsync:.4f}")
        print(f"=========================================")

if __name__ == "__main__":
    calculate_true_dsync()
