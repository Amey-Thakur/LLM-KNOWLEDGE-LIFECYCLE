# Speaker Notes: Knowledge Lifecycle Research Presentation

This document provides your "script" for each slide in `presentation.tex`. **Stay bold, stay blunt, and stay precise.**

## Slide 1: The Crisis of Fragmentation
- **The Hook:** "Ladies and Gentlemen, we are building powerful AI agents, but we are building them on a fragmented foundation. We optimize RAG, we optimize Editing, we optimize Memory—but we do it in silos. My research proves that the most dangerous failures happen in the gaps between these silos."
- **Key Point:** Focus on the "Methodological Gap."

## Slide 2: The Knowledge Lifecycle Framework
- **The Hook:** "I propose a unified framework: The Knowledge Lifecycle. Knowledge isn't just 'stored'; it moves. It is acquired, it lives in storage, it is retrieved, it is updated, and eventually, it must be forgotten. My framework is the first to track this journey formally."
- **Action:** Point to the TikZ diagram. Explain the "Interaction Arrows."

## Slide 3: Quantifying the Failure: D_sync
- **The Hook:** "How do we know a model is failing? We need a metric. I introduce **Lifecycle Desynchronisation ($\mathcal{D}_{sync}$)**. It tells us—mathematically—when a model's internal memory is fighting its external context. This isn't just a heuristic; it's a cross-modal consistency score."
- **Key Point:** Explain that $\mathcal{D}_{sync} > 1.0$ means the system is fundamentally unreliable.

## Slide 4: Solution 1: The Provenance Vector (p)
- **The Hook:** "Current Transformers are 'time-blind.' They don't know when a fact was learned. My intervention is the **Provenance Vector**. We append temporal metadata directly to the value vectors in the FFN. We make the model's memory 'time-aware' for the first time."
- **Key Point:** Emphasize the low overhead (0.01% parameters).

## Slide 5: Solution 2: Activation Steering
- **The Hook:** "Once we have provenance, we can perform **Activation Steering**. At inference time, the model acts like a System 2 reasoner. It sees an old fact, it sees a new RAG document, and it 'turns down the volume' on the old fact. This is the mathematical key to resolving knowledge conflicts."
- **Action:** Explain the "Volume Knob" metaphor.

## Slide 6: Case Study: The Vioxx Failure
- **The Hook:** "This isn't theoretical. In my experiments, I simulated the 2004 Vioxx withdrawal. I proved that current models—both text and multimodal—will ignore a 'FDA Withdrawn' notice because their internal weights are too strong. My metric detects this failure; my architecture fixes it."
- **Action:** Present the GPT-2 and LLaVA results with confidence.

## Slide 7: Conclusion: The Future of Agentic AI
- **The Hook:** "We are moving from static models to autonomous systems. Theoretical heuristics are no longer enough. We need a formal, mathematically consistent lifecycle. My thesis provides the roadmap. Thank you, and I look forward to your questions."
- **Mindset:** You have now set the stage for the "Blunt & Precise" Q&A session.
