# viXra submission brief - Knowledge Lifecycle

One copy block per form field, in the order the form presents them.
Generated from the manuscripts, so nothing here is retyped and the brief
cannot drift from the paper.

This paper is dated 12 November 2026 and is submitted separately from the five SAIR papers.

## Which archive

viXra's front page states it plainly: "viXra.org only accepts scholarly
articles written without AI assistance. Please go to ai.viXra.org to submit new
scholarly article written with AI assistance". The form repeats it and the
moderators enforce it, stamping November 2026 listings with that redirect.
These manuscripts were drafted with substantial AI assistance, so on the
archive's own rule they go to **ai.viXra.org**
(`fs23.formsite.com/viXra/ai_submit`), whose only extra requirement is a list
of scientific references in the PDF. Every paper here has a bibliography, and
the largest PDF is 0.37 MB against a 5 MB cap.

## Fields that are the same every time

- **PDF submission method:** Upload.
- **Your e-mail address:** the same address for all five. Replacements and
  withdrawals are only accepted from the address that made the original
  submission, and ai.viXra.org allows four replacements, then none.
- **Author(s):** `Amey Thakur`. There is no affiliation field.
- **Other instructions:** blank.
- Upload the **PDF only**. viXra requires one self-contained PDF and rejects
  extra files, so never the `arxiv-submission.tar.gz`.
- **One category, final at submission.** There is no cross-listing; the
  moderators may move a paper they judge misfiled.
- **No DOI.** You get `viXra:YYMM.NNNN` and the abstract page URL is the
  citable reference.

---


## Paper VI - Knowledge Lifecycle

**PDF to upload:** `papers/main.pdf` (0.17 MB, 8 pages). Source `LLM-KNOWLEDGE-LIFECYCLE/preprint/main.tex`, was bound for arXiv `cs.CL`.

**Science Category** - select from the dropdown, one only:

```
Computational Science - Artificial Intelligence
```

**Title of Paper**

```
Probability Is Not an Answer: Rank-Level Measurement of Knowledge Conflict in Small Language Models
```

**Author(s)**

```
Amey Thakur
```

**Abstract** - 252 words, limit 400. Plain text; paragraph breaks are discarded by viXra, so paste as one block.

```
A retrieval-augmented system succeeds only when the document it supplies beats what the model already believes. We measure which one wins, at the level of the token the model would actually emit, over \ forward passes: \ open models from 124M to 1.5B parameters, 24 counterfactual probes, three phrasings, with and without a contradicting document. Restricting to the \ cases where the model demonstrably held the fact beforehand, the document is adopted % of the time and the model keeps its own answer %. Which model defers is not predicted by size. Holding the fact rises monotonically with parameter count, from % to %, but adoption does not: TinyLlama-1.1B holds the fact % of the time and still adopts the document least of the six, at %, while Qwen2.5-0.5B, a third its size, adopts %. Wording moves the result nearly as much as the model does: over all probes the rate spreads by a median of \ percentage points across three paraphrases, and restricted to conflicts, where the spread measures deference rather than recall, it still reaches . We also show why this has to be read at the rank: the in-context answer carries more than 0.10 of the probability mass in \ conditions and is still not what the model says in \ of them, %. That gap is not hypothetical. It is how an earlier version of this work reached a published conclusion that the measurement did not support, which we correct here. All code, probes and measurements are public.
```

**Number of pages**

```
8
```

**Comment**

```
8 pages. Code and data: https://github.com/Amey-Thakur/LLM-KNOWLEDGE-LIFECYCLE
```

**Other instructions** - leave blank.

---

## Note on the two manuscripts here

`preprint/` is the short empirical paper and the one to submit. `paper/` is the
longer journal manuscript, which still carries the Provenance Vector and the
five-stage framework in full; it has no date field and is not part of this
submission.
