# viXra submission brief - Knowledge Lifecycle

Generated from the manuscripts on 11 November 2026. Title, abstract, figure
and table counts come from the `.tex`; page counts from the built PDF. Nothing
here is retyped, so the brief cannot drift from the paper.

## Read this first: which archive

viXra's front page states it plainly: "viXra.org only accepts scholarly
articles written without AI assistance. Please go to ai.viXra.org to submit new
scholarly article written with AI assistance". The submission form at
vixra.org repeats it, and the moderators enforce it in public: several November
2026 listings carry the admin note "Please submit article written with AI
assistance to ai.viXra.org".

These manuscripts were drafted with substantial AI assistance. On the archive's
own rule that routes them to **ai.viXra.org**, not vixra.org:

| | vixra.org | ai.viXra.org |
| :--- | :--- | :--- |
| Form | fs23.formsite.com/viXra/form2 | fs23.formsite.com/viXra/ai_submit |
| PDF must contain | title, authors, abstract, content | the same **plus a list of scientific references** |
| Size | about 10 MB | keep under 5 MB |
| Replacements | more allowed with good reason | **four**, then no more |
| Removal | removal form exists | withdrawal only, no removal |
| Problems | vixra.org@gmail.com | aibox@vixra.org |

Every paper here carries a bibliography, so the references requirement is met,
and the largest PDF is 0.35 MB, so the 5 MB cap is not close. Neither form has
a checkbox for AI assistance; the obligation sits with the author.

The decision is yours, and it is worth making deliberately: submitting
AI-assisted work to vixra.org contradicts a rule the site states on its front
page and its form, and the moderators visibly act on it.

## Rules that apply to every submission

- **One PDF, nothing else.** "The submitted work must be completely contained
  in one PDF document. It cannot include any extra files such as figures." The
  `arxiv-submission.tar.gz` in each repository is useless here; upload the
  compiled PDF.
- **One category.** "viXra does not have a cross-listing feature and will
  reject multiple submissions." The admins may move a paper they judge
  misfiled.
- **Author name.** Real name, Latin letters, first name or initials before the
  family name, no titles, not all capitals. No pseudonyms or single-word names.
  Claim an affiliation only if a page on the institution's own website proves
  it; otherwise give none.
- **Abstract.** Under 400 words. Paragraph breaks and most formatting are
  lost. Basic HTML is allowed for subscripts, italics, line breaks and special
  characters, but not links, images, bold, fonts, sizes or colours. Avoid
  complicated equations.
- **Title.** Entered close to Title Case; viXra normalises it.
- **Comment.** Brief technicalities only, such as the page and figure counts
  below. Left blank, viXra substitutes the page count.
- **No DOI.** viXra assigns an identifier of the form `viXra:YYMM.NNNN` and the
  abstract page URL serves as the fixed reference. Nothing on the site says it
  mints DOIs; the DOIs visible on some listings are author-supplied Zenodo
  ones in the comment field.
- **Timing.** Normally online within 48 hours. On ai.viXra.org, no confirmation
  within 48 hours means the submission was not accepted. Do not resubmit in the
  meantime.
- Do not upload a publisher's typeset version, and do not embed executables.


## Paper VI - Knowledge Lifecycle

PDF: `LLM-KNOWLEDGE-LIFECYCLE-preprint.pdf` (0.17 MB, 8 pages). Source: `LLM-KNOWLEDGE-LIFECYCLE/preprint/main.tex`. Was bound for arXiv `cs.CL`.

| Form field | What to enter |
| :--- | :--- |
| Your e-mail address | the address you will use for every replacement of this paper |
| PDF submission method | **Upload** |
| PDF document | `LLM-KNOWLEDGE-LIFECYCLE-preprint.pdf` |
| Science Category | **Computational Science - Artificial Intelligence** |
| Title of Paper | Probability Is Not an Answer: Rank-Level Measurement of Knowledge Conflict in Small Language Models |
| Author(s) | Amey Thakur |
| Number of pages | 8 |
| Comment | 8 pages, 0 figures, 0 tables |
| Other instructions | leave blank |

**Abstract** (paste as plain text; 252 words, limit is 400)

```
A retrieval-augmented system succeeds only when the document it supplies beats what the model already believes. We measure which one wins, at the level of the token the model would actually emit, over \ forward passes: \ open models from 124M to 1.5B parameters, 24 counterfactual probes, three phrasings, with and without a contradicting document. Restricting to the \ cases where the model demonstrably held the fact beforehand, the document is adopted % of the time and the model keeps its own answer %. Which model defers is not predicted by size. Holding the fact rises monotonically with parameter count, from % to %, but adoption does not: TinyLlama-1.1B holds the fact % of the time and still adopts the document least of the six, at %, while Qwen2.5-0.5B, a third its size, adopts %. Wording moves the result nearly as much as the model does: over all probes the rate spreads by a median of \ percentage points across three paraphrases, and restricted to conflicts, where the spread measures deference rather than recall, it still reaches . We also show why this has to be read at the rank: the in-context answer carries more than 0.10 of the probability mass in \ conditions and is still not what the model says in \ of them, %. That gap is not hypothetical. It is how an earlier version of this work reached a published conclusion that the measurement did not support, which we correct here. All code, probes and measurements are public.
```

## Note on the two manuscripts in this repository

`preprint/` is the short empirical paper and the one to submit: it is the
version that was bound for arXiv, and its date is 12 November 2026.
`paper/` is the longer journal manuscript, which still carries the Provenance
Vector and the five-stage framework in full. It has no date field and is not
part of this submission.
