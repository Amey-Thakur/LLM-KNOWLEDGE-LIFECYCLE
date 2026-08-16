// LLM Knowledge Lifecycle: browser-side measurement on GPT-2.
// The entire computation runs locally: the model is downloaded once from the
// Hugging Face Hub, and every forward pass happens in this tab.

import { AutoTokenizer, AutoModelForCausalLM } from
  "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.4.0";

const PRESETS = {
  vioxx: {
    query: "Question: Is Vioxx safe to prescribe? Answer: Vioxx is considered",
    context: "Context: In September 2004, Merck voluntarily withdrew Vioxx after trials revealed increased cardiovascular risks.",
    answer: " withdrawn",
    reading: "The paper's headline case. The withdrawal notice is in the prompt, yet the correct answer stays near one in a hundred thousand. Expect D_sync deep past the 9.2-nat failure threshold; the paper's fp32 run measures 12.05 nats with I_ctx = 0.033.",
  },
  monarch: {
    query: "Question: Who is the current British monarch? Answer: The current British monarch is",
    context: "Context: Queen Elizabeth II died in September 2022. Charles III acceded to the throne and is the reigning King of the United Kingdom.",
    answer: " Charles",
    reading: "A subtler failure. The context raises P(Charles), but its strongest effect is boosting “ Queen”: merely mentioning the late monarch reinforces the stale association. Correct information can strengthen the wrong answer.",
  },
  twitter: {
    query: "Question: What is the social network Twitter called today? Answer: Twitter is now called",
    context: "Context: In July 2023, Twitter was rebranded as X under Elon Musk's ownership.",
    answer: " X",
    reading: "The context moves the distribution hard (the paper's fp32 run: I_ctx = 0.9 nats) and lifts the correct answer by orders of magnitude. The model still answers “Twitter”. Influence without resolution.",
  },
};

const $ = (id) => document.getElementById(id);
let tokenizer = null;
let model = null;

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

// ---------- model loading ----------
$("load-btn").addEventListener("click", async () => {
  $("load-btn").disabled = true;
  $("load-bar-wrap").hidden = false;
  const status = $("load-status");
  const seen = {};
  const progress = (info) => {
    if (info.status === "progress" && info.total) {
      seen[info.file] = info.loaded / info.total;
      const vals = Object.values(seen);
      const pct = (vals.reduce((a, b) => a + b, 0) / vals.length) * 100;
      $("load-bar").style.width = pct.toFixed(1) + "%";
      status.textContent = "downloading " + pct.toFixed(0) + "%";
    }
  };
  try {
    status.textContent = "downloading";
    tokenizer = await AutoTokenizer.from_pretrained("Xenova/gpt2", { progress_callback: progress });
    // The Xenova/gpt2 repo uses legacy file naming: dtype "q8" maps to the
    // "_quantized" suffix, and the merged decoder is the 128 MB build.
    model = await AutoModelForCausalLM.from_pretrained("Xenova/gpt2", {
      model_file_name: "decoder_model_merged",
      dtype: "q8",
      progress_callback: progress,
    });
    status.textContent = "ready (GPT-2 base, 8-bit quantized, running locally)";
    $("load-bar").style.width = "100%";
    $("run").disabled = false;
  } catch (e) {
    status.textContent = "failed to load: " + e.message;
    $("load-btn").disabled = false;
  }
});

// ---------- measurement ----------
async function nextTokenDistribution(text) {
  const inputs = await tokenizer(text);
  const { logits } = await model(inputs);
  const [, T, V] = logits.dims;
  const row = logits.data.slice((T - 1) * V, T * V);
  // stable softmax
  let max = -Infinity;
  for (let i = 0; i < V; i++) if (row[i] > max) max = row[i];
  let sum = 0;
  const probs = new Float64Array(V);
  for (let i = 0; i < V; i++) { probs[i] = Math.exp(row[i] - max); sum += probs[i]; }
  for (let i = 0; i < V; i++) probs[i] /= sum;
  return probs;
}

function topK(probs, k) {
  const idx = [];
  const taken = new Set();
  for (let n = 0; n < k; n++) {
    let best = -1, bp = -1;
    for (let i = 0; i < probs.length; i++) {
      if (!taken.has(i) && probs[i] > bp) { bp = probs[i]; best = i; }
    }
    taken.add(best);
    idx.push(best);
  }
  return idx;
}

function renderTop(tableId, probs, targetId) {
  const rows = topK(probs, 8).map((i) => {
    const tok = tokenizer.decode([i]);
    const hit = i === targetId ? ' class="hit"' : "";
    return `<tr${hit}><td>${JSON.stringify(tok).slice(1, -1)}</td><td>${(probs[i] * 100).toFixed(4)}%</td></tr>`;
  });
  $(tableId).innerHTML = "<tr><th>token</th><th>probability</th></tr>" + rows.join("");
}

$("run").addEventListener("click", async () => {
  const runBtn = $("run");
  runBtn.disabled = true;
  runBtn.textContent = "measuring...";
  try {
    const query = $("query").value.trim();
    const context = $("context").value.trim();
    let answer = $("answer").value;
    if (!answer.startsWith(" ")) answer = " " + answer.trim();

    const pPlain = await nextTokenDistribution(query);
    const pCtx = await nextTokenDistribution(context + "\n" + query);

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

    let chipText, chipClass, detail;
    if (dsync > 9.2) {
      if (ictx < 0.05) {
        chipText = "resolution failure"; chipClass = "chip-fail";
        detail = "context ignored; no realistic decoding recovers the correct answer";
      } else {
        chipText = "drift, context losing"; chipClass = "chip-fail";
        detail = "context influential but losing; no realistic decoding recovers the correct answer";
      }
    } else if (dsync > 4.6) {
      chipText = "severe drift"; chipClass = "chip-drift";
      detail = "correct answer below 1% probability";
    } else if (dsync > 0.7) {
      chipText = "drift"; chipClass = "chip-drift";
      detail = "correct answer no longer holds most of the probability mass";
    } else {
      chipText = "synchronized"; chipClass = "chip-ok";
      detail = "correct answer holds at least half the probability mass";
    }

    // Marker positions follow the drawn zone boundaries: D_sync thresholds
    // 0.7 / 4.6 / 9.2 sit at 5% / 33% / 66%, scale capped at 14 nats.
    const dPos = dsync <= 0.7 ? (dsync / 0.7) * 5
      : dsync <= 4.6 ? 5 + ((dsync - 0.7) / 3.9) * 28
      : dsync <= 9.2 ? 33 + ((dsync - 4.6) / 4.6) * 33
      : Math.min(100, 66 + ((dsync - 9.2) / 4.8) * 34);
    // I_ctx thresholds 0.05 / 0.5 sit at 10% / 50%, scale capped at 2 nats.
    const iPos = ictx <= 0.05 ? (ictx / 0.05) * 10
      : ictx <= 0.5 ? 10 + ((ictx - 0.05) / 0.45) * 40
      : Math.min(100, 50 + ((ictx - 0.5) / 1.5) * 50);

    const chip = $("verdict-chip");
    chip.textContent = chipText;
    chip.className = "chip " + chipClass;
    $("r-verdict-detail").textContent = detail;
    $("g-marker").style.left = dPos.toFixed(1) + "%";
    $("g-ictx").style.left = iPos.toFixed(1) + "%";

    $("r-tok").textContent = `${JSON.stringify(pieces)} (${ids.length} piece(s); first piece measured)`;
    $("r-p0").textContent = pt0.toExponential(2);
    $("r-p1").textContent = pt1.toExponential(2);
    $("r-dsync").textContent = dsync.toFixed(2) + " nats";
    $("r-ictx").textContent = ictx.toFixed(3) + " nats";
    renderTop("t-plain", pPlain, target);
    renderTop("t-ctx", pCtx, target);
    $("results").hidden = false;
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = "Measure";
  }
});
