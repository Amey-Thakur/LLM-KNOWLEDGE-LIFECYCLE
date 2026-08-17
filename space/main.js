// =============================================================================
// File        : main.js
// Project     : The Knowledge Lifecycle of Large Language Models
// Purpose     : Browser-side measurement engine. Loads GPT-2, runs the two
//               prompt conditions, and renders the narrative, the paired
//               distributions, the gauges, and the stage diagnosis.
// Tech Stack  : JavaScript (ES modules), transformers.js 3.4.0,
//               ONNX Runtime Web (WASM)
// Authors     : Amey Thakur (https://github.com/Amey-Thakur)
//               Sarvesh Talele (https://github.com/sarveshtalele)
// Repository  : https://github.com/Amey-Thakur/LLM-KNOWLEDGE-LIFECYCLE
// Release Date: August 16, 2026
// License     : CC BY 4.0
// =============================================================================

// The entire computation runs locally: the model is downloaded once from the
// Hugging Face Hub, and every forward pass happens in this tab. Deterministic:
// identical inputs give identical numbers on every run.

import { AutoTokenizer, AutoModelForCausalLM } from
  "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.4.0";

const PRESETS = {
  vioxx: {
    query: "Question: Is Vioxx safe to prescribe? Answer: Vioxx is considered",
    context: "Context: In September 2004, Merck voluntarily withdrew Vioxx after trials revealed increased cardiovascular risks.",
    answer: " withdrawn",
    // Stated as the direction of the change rather than as the winning token.
    // At full precision "safe" is the top answer; under the 8-bit weights this
    // page runs, a function word takes first place. What holds in both builds
    // is that the corrective document pushes "safe" up and leaves "withdrawn"
    // nowhere.
    reading: "The headline case. The withdrawal notice is in the prompt, and the model's confidence in “safe” goes up rather than down: 37.53% to 42.58% at full precision, where the paper measures 12.05 nats, far past the 9.2 failure threshold.",
  },
  monarch: {
    query: "Question: Who is the current British monarch? Answer: The current British monarch is",
    context: "Context: Queen Elizabeth II died in September 2022. Charles III acceded to the throne and is the reigning King of the United Kingdom.",
    answer: " Charles",
    reading: "A subtler failure. The document does raise the correct answer, but its strongest effect is boosting “ Queen”. Naming a fact, even to correct it, reinforces the old association.",
  },
  twitter: {
    query: "Question: What is the social network Twitter called today? Answer: Twitter is now called",
    context: "Context: In July 2023, Twitter was rebranded as X under Elon Musk's ownership.",
    answer: " X",
    reading: "Here the document moves the model hard and lifts the correct answer by orders of magnitude. It still answers “Twitter”. Influence without resolution.",
  },
};

const $ = (id) => document.getElementById(id);
let tokenizer = null;
let model = null;
let last = null;   // logits and metadata of the most recent measurement

// ---------- tabs ----------
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    $(btn.dataset.tab).classList.add("active");
  });
});

// ---------- presets ----------
function applyPreset(key) {
  const p = PRESETS[key];
  $("query").value = p.query;
  $("context").value = p.context;
  $("answer").value = p.answer;
  $("preset-reading").textContent = p.reading;
  document.querySelectorAll(".preset").forEach((b) =>
    b.classList.toggle("active", b.dataset.preset === key));
}
document.querySelectorAll(".preset").forEach((btn) =>
  btn.addEventListener("click", () => applyPreset(btn.dataset.preset)));
applyPreset("vioxx");

// ---------- model loading (on first Measure, so one button drives everything) ----------
async function ensureModel() {
  if (model) return;
  $("load-bar-wrap").hidden = false;
  const status = $("load-status");
  const seen = {};
  const progress = (info) => {
    if (info.status === "progress" && info.total) {
      seen[info.file] = info.loaded / info.total;
      const vals = Object.values(seen);
      const pct = (vals.reduce((a, b) => a + b, 0) / vals.length) * 100;
      $("load-bar").style.width = pct.toFixed(1) + "%";
      status.textContent = "downloading GPT-2, " + pct.toFixed(0) + "% of 128 MB, one time only";
    }
  };
  status.textContent = "downloading GPT-2 (128 MB, one time only)";
  tokenizer = await AutoTokenizer.from_pretrained("Xenova/gpt2", { progress_callback: progress });
  // The Xenova/gpt2 repo uses legacy file naming: dtype "q8" maps to the
  // "_quantized" suffix, and the merged decoder is the 128 MB build.
  model = await AutoModelForCausalLM.from_pretrained("Xenova/gpt2", {
    model_file_name: "decoder_model_merged",
    dtype: "q8",
    progress_callback: progress,
  });
  status.textContent = "model ready: GPT-2 base, 8-bit quantized, running locally";
  $("load-bar-wrap").hidden = true;
}

// ---------- measurement core ----------

// Regime boundaries in nats. SYNC is ln 2 exactly, the point at which the
// correct answer holds half the probability mass; PERCENT and FAIL are ln 100
// and ln 10000 to the precision they are quoted at.
const SYNC = Math.LN2, PERCENT = 4.6, FAIL = 9.2;
async function lastLogits(text) {
  const inputs = await tokenizer(text);
  const { logits } = await model(inputs);
  const [, T, V] = logits.dims;
  return Float32Array.from(logits.data.slice((T - 1) * V, T * V));
}

function softmax(row, temperature = 1.0) {
  const V = row.length;
  let max = -Infinity;
  for (let i = 0; i < V; i++) if (row[i] > max) max = row[i];
  let sum = 0;
  const probs = new Float64Array(V);
  for (let i = 0; i < V; i++) { probs[i] = Math.exp((row[i] - max) / temperature); sum += probs[i]; }
  for (let i = 0; i < V; i++) probs[i] /= sum;
  return probs;
}

/**
 * Indices of the k largest probabilities, highest first.
 *
 * One pass over the vocabulary, maintaining a short sorted list. Repeated scans
 * with an "already taken" set cost k passes over 50,257 entries per call.
 */
function topK(probs, k) {
  const best = [];   // indices, kept sorted by descending probability
  for (let i = 0; i < probs.length; i++) {
    const p = probs[i];
    if (best.length === k && p <= probs[best[k - 1]]) continue;
    let at = best.length;
    while (at > 0 && probs[best[at - 1]] < p) at--;
    best.splice(at, 0, i);
    if (best.length > k) best.pop();
  }
  return best;
}

// ---------- rendering ----------
const fmtPct = (p) => p >= 0.0001 ? (p * 100).toFixed(2) + "%" : "<0.01%";
const fmtTok = (t) => JSON.stringify(t).slice(1, -1);

function renderBars(pPlain, pCtx, target) {
  // Union of both top-5 sets, plus the correct answer, ordered by with-context
  // probability. Linear scale on purpose: a correct answer you cannot see IS
  // the finding.
  const ids = [...new Set([...topK(pCtx, 5), ...topK(pPlain, 5), target])];
  ids.sort((a, b) => pCtx[b] - pCtx[a]);
  const maxP = Math.max(pCtx[ids[0]], pPlain[ids[0]], 1e-9);
  const rows = ids.map((i) => {
    const tok = fmtTok(tokenizer.decode([i]));
    const cls = i === target ? "bar-row hit" : "bar-row";
    const w0 = Math.max(0.4, (pPlain[i] / maxP) * 100);
    const w1 = Math.max(0.4, (pCtx[i] / maxP) * 100);
    return `<div class="${cls}">
      <span class="bar-tok">${tok}</span>
      <span class="bar-pair">
        <span class="bar b-plain" style="width:${w0.toFixed(1)}%"></span><em>${fmtPct(pPlain[i])}</em>
        <span class="bar b-ctx" style="width:${w1.toFixed(1)}%"></span><em>${fmtPct(pCtx[i])}</em>
      </span>
    </div>`;
  });
  $("bars").innerHTML =
    '<div class="bar-row bar-head"><span class="bar-tok"></span><span class="bar-pair"><em>without document</em><em>with document</em></span></div>'
    + rows.join("");
}

/**
 * Plain-language account of one measurement, written from the numbers.
 *
 * The observations are additive rather than exclusive: one run can be both
 * "the document helped" and "the document strengthened the wrong answer", so
 * they accumulate instead of branching.
 */
function narrative(m) {
  const answerWon = m.topCtx === m.answerPiece;
  const odds = Math.round(1 / m.pt1);
  const s = [
    `With the corrective document in its prompt, the model's most likely continuation is “${m.topCtx}” at ${fmtPct(m.pTopCtx)}.`,
    `The correct answer “${m.answerPiece.trim()}” receives ${fmtPct(m.pt1)}: about one chance in ${odds.toLocaleString()}.`,
  ];

  if (m.ictx < 0.05) {
    s.push(`The document barely registered: it moved the model's whole distribution by ${m.ictx.toFixed(3)} nats.`);
  }
  if (m.pt1 > m.pt0 * 1.5) {
    s.push(`It did help the correct answer, multiplying its probability by ${(m.pt1 / m.pt0).toFixed(1)}.`);
  }
  if (!answerWon && m.pTopCtx > m.pTopPlainSameTok) {
    s.push(`It also strengthened the leading wrong answer, raising “${m.topCtx}” from ${fmtPct(m.pTopPlainSameTok)} to ${fmtPct(m.pTopCtx)}: mentioning a fact, even to correct it, reinforces the association.`);
  }

  s.push(answerWon
    ? "The document won here: the model's top answer is the correct one. This is what synchronization looks like."
    : "The parametric memory still controls the answer.");
  return s.join(" ");
}

function renderStagebar(m) {
  const r = $("sb-retrieve"), u = $("sb-update"), note = $("stagenote");
  $("stagebar").classList.add("diagnosed");
  document.querySelectorAll(".stagebar .s1, .stagebar .s2, .stagebar .s5")
    .forEach((el) => el.classList.add("dimstage"));
  r.textContent = "Retrieve ✓";
  if (m.dsync > 9.2) {
    u.textContent = "Update ✗";
    note.textContent = "Retrieve succeeded: the document is in the context. Update failed: the conflict was resolved in favor of stale memory.";
  } else if (m.dsync > SYNC) {
    u.textContent = "Update △";
    note.textContent = "Retrieve succeeded. Update is partial: the document shifted the model, but the stale memory still leads.";
  } else {
    u.textContent = "Update ✓";
    note.textContent = "Retrieve succeeded and Update resolved the conflict: the context controls the answer.";
  }
}

function renderTemperature() {
  if (!last) return;
  const T = parseFloat($("temp").value);
  $("t-value").textContent = T.toFixed(1);
  const p = softmax(last.rowCtx, T);
  const pAns = Math.max(p[last.target], 1e-12);
  const top = topK(p, 1)[0];
  const draws = Math.round(1 / pAns);
  $("temp-readout").textContent =
    `At temperature ${T.toFixed(1)}, sampling one answer with the document present: ` +
    `the correct answer comes up about once in ${draws.toLocaleString()} draws; ` +
    `the most likely token is “${fmtTok(tokenizer.decode([top]))}” at ${fmtPct(p[top])}. ` +
    (T > 1.0 ? "Higher temperature flattens the distribution but cannot rescue an answer this far down."
             : "Lower temperature sharpens the model's existing preference.");
}
$("temp").addEventListener("input", renderTemperature);

// ---------- the one-button flow ----------
$("run").addEventListener("click", async () => {
  const runBtn = $("run");
  runBtn.disabled = true;
  try {
    runBtn.textContent = model ? "measuring..." : "loading model...";
    await ensureModel();
    runBtn.textContent = "measuring...";

    const query = $("query").value.trim();
    const context = $("context").value.trim();
    const repeat = parseInt($("repeat").value, 10);
    let answer = $("answer").value;
    if (!answer.startsWith(" ")) answer = " " + answer.trim();
    if (!query || !context || !answer.trim()) {
      $("load-status").textContent = "enter a query, a corrective document, and the correct continuation";
      return;
    }

    const doc = Array(repeat).fill(context).join("\n");
    const rowPlain = await lastLogits(query);
    const rowCtx = await lastLogits(doc + "\n" + query);
    const pPlain = softmax(rowPlain);
    const pCtx = softmax(rowCtx);

    const ids = tokenizer.encode(answer);
    const pieces = ids.map((i) => tokenizer.decode([i]));
    const target = ids[0];

    const pt0 = Math.max(pPlain[target], 1e-12);
    const pt1 = Math.max(pCtx[target], 1e-12);
    const dsync = -Math.log(pt1);
    let ictx = 0;
    for (let i = 0; i < pCtx.length; i++) {
      if (pCtx[i] > 0 && pPlain[i] > 0) ictx += pCtx[i] * Math.log(pCtx[i] / pPlain[i]);
    }

    const topPlainId = topK(pPlain, 1)[0];
    const topCtxId = topK(pCtx, 1)[0];

    let chipText, chipClass, detail;
    if (dsync > FAIL) {
      if (ictx < 0.05) {
        chipText = "resolution failure"; chipClass = "chip-fail";
        detail = "context ignored; no realistic decoding recovers the correct answer";
      } else {
        chipText = "drift, context losing"; chipClass = "chip-fail";
        detail = "context influential but losing; no realistic decoding recovers the correct answer";
      }
    } else if (dsync > PERCENT) {
      chipText = "severe drift"; chipClass = "chip-drift";
      detail = "correct answer below 1% probability";
    } else if (dsync > SYNC) {
      chipText = "drift"; chipClass = "chip-drift";
      detail = "correct answer no longer holds most of the probability mass";
    } else {
      chipText = "synchronized"; chipClass = "chip-ok";
      detail = "correct answer holds at least half the probability mass";
    }

    // Gauge markers follow the drawn zone boundaries: the three thresholds sit
    // at 5% / 33% / 66% of the track, and the scale is capped at 14 nats.
    const dPos = dsync <= SYNC ? (dsync / SYNC) * 5
      : dsync <= PERCENT ? 5 + ((dsync - SYNC) / (PERCENT - SYNC)) * 28
      : dsync <= FAIL ? 33 + ((dsync - PERCENT) / (FAIL - PERCENT)) * 33
      : Math.min(100, 66 + ((dsync - FAIL) / 4.8) * 34);
    const iPos = ictx <= 0.05 ? (ictx / 0.05) * 10
      : ictx <= 0.5 ? 10 + ((ictx - 0.05) / 0.45) * 40
      : Math.min(100, 50 + ((ictx - 0.5) / 1.5) * 50);

    const m = {
      pt0, pt1, dsync, ictx,
      answerPiece: fmtTok(pieces[0]),
      topPlain: fmtTok(tokenizer.decode([topPlainId])),
      topCtx: fmtTok(tokenizer.decode([topCtxId])),
      pTopCtx: pCtx[topCtxId],
      pTopPlainSameTok: pPlain[topCtxId],
    };

    const chip = $("verdict-chip");
    chip.textContent = chipText;
    chip.className = "chip " + chipClass;
    $("r-verdict-detail").textContent = detail + (repeat > 1 ? ` (document repeated ${repeat}x)` : "");
    $("narrative").textContent = narrative(m);
    renderBars(pPlain, pCtx, target);
    $("g-marker").style.left = dPos.toFixed(1) + "%";
    $("g-ictx").style.left = iPos.toFixed(1) + "%";
    $("r-dsync").textContent = dsync.toFixed(2) + " nats";
    $("r-ictx").textContent = ictx.toFixed(3) + " nats";
    $("r-p0").textContent = pt0.toExponential(2);
    $("r-p1").textContent = pt1.toExponential(2);
    $("r-tok").textContent = `${JSON.stringify(pieces)} (${ids.length} piece(s); first piece measured)`;
    renderStagebar({ dsync });

    last = { rowCtx, target };
    renderTemperature();

    $("results").hidden = false;
  } catch (e) {
    $("load-status").textContent = "error: " + e.message;
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = "Measure";
  }
});
