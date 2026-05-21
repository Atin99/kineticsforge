var activeBmsRun = 0;
var bmsPresetFromDiagnostics = null;
var kfAssistantHistory = [];
var KF_HISTORY_KEY = "kineticsforge.runHistory.v1";
var kfRunHistory = loadRunHistory();
window.__kfAssistantHistory = kfAssistantHistory;
window.__kfRunHistory = kfRunHistory;

function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }
function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
function relu(x) { return Math.max(0, x); }
function expClamp(x, lo, hi) { return Math.exp(clamp(x, lo, hi)); }
function fmt(x, d) { return Number.isFinite(x) ? x.toFixed(d == null ? 2 : d) : "--"; }
function gaussian() {
  var u = 1 - Math.random();
  var v = 1 - Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// Seeded PRNG: mulberry32 — deterministic 32-bit PRNG for reproducible BMS simulation
function mulberry32(seed) {
  var state = seed | 0;
  return function () {
    state = (state + 0x6D2B79F5) | 0;
    var t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function seededGaussian(rng) {
  var u = 1 - rng();
  var v = 1 - rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, function (c) {
    return ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c];
  });
}
function num(id, fallback) {
  var el = document.getElementById(id);
  if (!el) return fallback;
  var v = parseFloat(el.value);
  return Number.isFinite(v) ? v : fallback;
}
function setHtml(id, html) {
  var el = document.getElementById(id);
  if (el) el.innerHTML = html;
}
function showToast(message, kind) {
  var dock = document.getElementById("toast-dock");
  if (!dock) {
    dock = document.createElement("div");
    dock.id = "toast-dock";
    dock.className = "toast-dock";
    document.body.appendChild(dock);
  }
  var note = document.createElement("div");
  note.className = "toast " + (kind || "info");
  note.textContent = message;
  dock.appendChild(note);
  setTimeout(function () { note.classList.add("show"); }, 20);
  setTimeout(function () {
    note.classList.remove("show");
    setTimeout(function () { note.remove(); }, 260);
  }, 4200);
}
function setReadouts(id, items) {
  var el = document.getElementById(id);
  if (!el) return;
  el.innerHTML = items.map(function (it) {
    var val = it.html ? it.v : escapeHtml(it.v);
    return '<div class="readout"><div class="k">' + escapeHtml(it.k) + '</div><div class="v">' + val + '</div></div>';
  }).join("");
}
function downloadCSV(rows, filename) {
  if (!rows || !rows.length) return;
  var keys = Object.keys(rows[0]);
  var body = [keys.join(",")].concat(rows.map(function (row) {
    return keys.map(function (k) {
      var v = row[k] == null ? "" : String(row[k]);
      return /[",\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v;
    }).join(",");
  })).join("\n");
  var blob = new Blob([body], { type: "text/csv;charset=utf-8" });
  var url = URL.createObjectURL(blob);
  var a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function downloadJSON(payload, filename) {
  if (!payload) return;
  var blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json;charset=utf-8" });
  var url = URL.createObjectURL(blob);
  var a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function downloadText(text, filename, type) {
  var blob = new Blob([String(text || "")], { type: type || "text/plain;charset=utf-8" });
  var url = URL.createObjectURL(blob);
  var a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function loadRunHistory() {
  try {
    var raw = localStorage.getItem(KF_HISTORY_KEY);
    var parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed.slice(0, 40) : [];
  } catch (err) {
    return [];
  }
}

function saveRunHistory() {
  try {
    localStorage.setItem(KF_HISTORY_KEY, JSON.stringify(kfRunHistory.slice(0, 40)));
  } catch (err) {
    // Local storage can be unavailable in locked-down browsers. History is optional.
  }
}

function runKindLabel(kind) {
  return ({
    diagnostics: "Diagnostics",
    upload: "Upload",
    upload_compare: "Upload A/B",
    upload_batch: "Batch Upload",
    bms: "BMS",
    bms_sweep: "BMS Sweep",
    materials: "Materials",
    recycling: "Recycling"
  })[kind] || kind;
}

function compactNumber(x, d) {
  if (!Number.isFinite(Number(x))) return "--";
  var n = Number(x);
  if (Math.abs(n) >= 1000000) return (n / 1000000).toFixed(1) + "M";
  if (Math.abs(n) >= 1000) return (n / 1000).toFixed(1) + "k";
  return n.toFixed(d == null ? 2 : d);
}

function parseCompactNumber(text) {
  var raw = String(text || "").trim().toUpperCase();
  var mul = raw.indexOf("M") >= 0 ? 1000000 : raw.indexOf("K") >= 0 ? 1000 : 1;
  var n = parseFloat(raw.replace(/[^0-9.\-]/g, ""));
  return Number.isFinite(n) ? n * mul : 0;
}

function setNavHealth(state, label) {
  var wrap = document.getElementById("nav-status");
  var text = document.getElementById("nav-status-text");
  if (!wrap || !text) return;
  wrap.classList.remove("health-ok", "health-warn", "health-bad");
  wrap.classList.add(state === "ok" ? "health-ok" : state === "warn" ? "health-warn" : "health-bad");
  text.textContent = label;
}

function updateNavHealth() {
  setNavHealth("warn", "CHECKING");
  var ctrl = typeof AbortController !== "undefined" ? new AbortController() : null;
  var timer = setTimeout(function () {
    if (ctrl) ctrl.abort();
  }, 4500);
  fetch("/health", { cache: "no-store", signal: ctrl ? ctrl.signal : undefined })
    .then(function (res) {
      if (!res.ok) throw new Error("health " + res.status);
      return res.json();
    })
    .then(function (data) {
      var ok = data && (data.status === "ok" || data.status === "operational");
      setNavHealth(ok ? "ok" : "warn", ok ? "READY" : "WAKING");
    })
    .catch(function (err) {
      if (err && err.name === "AbortError") {
        setNavHealth("warn", "DEGRADED");
        return;
      }
      setNavHealth("bad", "OFFLINE");
    })
    .finally(function () {
      clearTimeout(timer);
    });
}

function mean(arr) {
  if (!arr || !arr.length) return 0;
  var s = 0;
  for (var i = 0; i < arr.length; i++) s += Number(arr[i]) || 0;
  return s / arr.length;
}

function stddev(arr) {
  if (!arr || arr.length < 2) return 0;
  var m = mean(arr);
  var s2 = 0;
  for (var i = 0; i < arr.length; i++) {
    var d = (Number(arr[i]) || 0) - m;
    s2 += d * d;
  }
  return Math.sqrt(s2 / arr.length);
}

function formatConfidence(confidence) {
  if (!Number.isFinite(Number(confidence))) return "--";
  return Math.round(clamp(Number(confidence), 0, 1) * 100) + "%";
}

function recordRun(kind, summary) {
  var payload = Object.assign({
    id: Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 7),
    kind: kind,
    label: runKindLabel(kind),
    at: new Date().toISOString()
  }, summary || {});
  kfRunHistory.unshift(payload);
  kfRunHistory = kfRunHistory.slice(0, 40);
  window.__kfRunHistory = kfRunHistory;
  saveRunHistory();
  renderDecisionConsole();
}

function latestRunLabel() {
  if (!kfRunHistory.length) return "None";
  var r = kfRunHistory[0];
  return r.label + " " + new Date(r.at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function severityRank(level) {
  return ({ critical: 0, warn: 1, ok: 2, info: 3 })[level] == null ? 4 : ({ critical: 0, warn: 1, ok: 2, info: 3 })[level];
}

function severityTag(level) {
  if (level === "critical") return '<span class="tag critical">Critical</span>';
  if (level === "warn") return '<span class="tag warn">Watch</span>';
  if (level === "ok") return '<span class="tag ok">Ready</span>';
  return '<span class="tag">Info</span>';
}

var HARDPOINT_MAP = {
  M1_CathodeUDE: {
    hardpoint: "Na-ion ODE terms (SEI/P2/JT/desolv/BV) + bounded residual gate",
    fallback: "Physics mirror only; checkpoint probe disabled."
  },
  M2_SOH: {
    hardpoint: "Capacity-window trend and operating-condition projection",
    fallback: "Capacity passthrough/rule estimate."
  },
  M3_CycleLife: {
    hardpoint: "Early-cycle class boundaries from fade and CE",
    fallback: "Rules bucket (<500, 500-1000, 1000-1500, >1500)."
  },
  M4_FadeRate: {
    hardpoint: "Linear early-cycle fade slope",
    fallback: "Rule slope from extracted cycle summaries."
  },
  M5_BMS_TGN: {
    hardpoint: "Thermal graph ODE + EIS-derived risk + temporal neighbor memory",
    fallback: "Physics-forward label-gate proxy."
  },
  M6_RUL: {
    hardpoint: "Cycle-to-80% projection",
    fallback: "Rule extrapolation from observed fade."
  },
  M7_Anomaly: {
    hardpoint: "Residual/variance anomaly proxy",
    fallback: "Feature-mask plus signal-shape heuristic."
  },
  M8_Joint_SOH_RUL: {
    hardpoint: "Joint fusion of SOH, RUL, and fade",
    fallback: "Rule fusion from M2/M4/M6 estimates."
  },
  M9_KneeDetect: {
    hardpoint: "Capacity curvature/knee detector",
    fallback: "Second-derivative knee proxy."
  },
  M10_ChemRank: {
    hardpoint: "Chemistry rank score from dQ/dV + feature priors",
    fallback: "Rule-based chemistry ranking."
  },
  M11_ElectrolyteHealth: {
    hardpoint: "EIS/plating compatibility head",
    fallback: "Tier-1 heuristic (degradation + plating risk)."
  },
  M12_Replenishability: {
    hardpoint: "Recovery probability and expected gain",
    fallback: "Preview heuristic (research only)."
  },
  M13_ChemIdentifier: {
    hardpoint: "Chem family from dQ/dV and voltage windows",
    fallback: "Rule classification with confidence cap."
  },
  M14_FormationProtocol: {
    hardpoint: "Formation life/robustness/SEI quality head",
    fallback: "Preview rule output; validate experimentally."
  }
};

function addDecision(items, item) {
  items.push(Object.assign({
    severity: "info",
    source: "Workbench",
    owner: "Operator",
    evidence: "--",
    action: "Run a panel to create evidence.",
    next: "No experiment ticket yet.",
    confidence: null
  }, item || {}));
}

function topLoss(out) {
  if (!out || !out.cap) return null;
  return [
    ["SEI/desolvation", out.sei[out.sei.length - 1]],
    ["P2-O2", out.p2[out.p2.length - 1]],
    ["Jahn-Teller", out.jt[out.jt.length - 1]],
    ["rate stress", out.rate[out.rate.length - 1]],
    ["residual", out.residual[out.residual.length - 1]]
  ].sort(function (a, b) { return b[1] - a[1]; })[0];
}

function buildDecisionItems() {
  var items = [];
  var diag = window.__kfDiag;
  if (diag && diag.cap && diag.cap.length) {
    var eol = diag.cap[diag.cap.length - 1];
    var fade = (1 - eol) * 100;
    var top = topLoss(diag);
    var diagConf = computeDiagConfidence(diag);
    var conf = diagConf.confidence;
    var residual = diag.residual[diag.residual.length - 1];
    var totalLoss = diag.sei[diag.sei.length - 1] + diag.p2[diag.p2.length - 1] + diag.jt[diag.jt.length - 1] + diag.rate[diag.rate.length - 1] + residual;
    var residualFrac = totalLoss > 1e-9 ? residual / totalLoss : 0;
    var sev = eol < 0.80 ? "critical" : (fade > 12 || residualFrac > 0.30 ? "warn" : "ok");
    var action = "Use this as the baseline degradation case for the next comparison.";
    var next = "Cycle 3 replicate cells at " + num("temp-slider", 45).toFixed(0) + " C and " + num("crate-slider", 1).toFixed(1) + "C; compare measured fade to the exported curve.";
    var cycleCap = parseInt(document.getElementById("cycles-slider").value, 10) || 500;
    if (eol < 0.80) {
      action = "[D-GATE-01] Halt this operating point and rerun at <=0.7C and <=35 C before cycle " + Math.min(350, Math.round(cycleCap * 0.7)) + ".";
      next = "Run one lower-stress replicate immediately and verify whether the dominant term changes.";
    } else if (residualFrac > 0.30) {
      action = "[D-CAL-02] Residual channel is too large; recalibrate against measured cycle-capacity rows before claiming mechanism attribution.";
      next = "Add at least 7 measured cycle-capacity points and rerun calibration; require RMSE <= 0.02.";
    } else if (top[0] === "SEI/desolvation" && conf >= 0.70) {
      action = "[D-SEI-03] Prioritize electrolyte/additive screening and lower-temperature charge protocol.";
      next = "Run additive A/B at fixed chemistry and compare CE drift + capacity fade over first 200 cycles.";
    } else if (top[0] === "P2-O2" && conf >= 0.70) {
      action = "[D-P2-04] Reduce upper cutoff voltage and/or increase Al/Ti stabilization before full cycling.";
      next = "Test upper-V reduction by 0.10 V and compare P2 contribution at cycle 300.";
    } else if (top[0] === "rate stress" && conf >= 0.65) {
      action = "[D-RATE-05] Reduce C-rate immediately; current profile is power-stress limited.";
      next = "Rerun at 0.7C and 1.0C to map rate sensitivity with the same composition.";
    }
    addDecision(items, {
      severity: sev,
      source: "Diagnostics",
      owner: "Cell R&D",
      evidence: "EOL " + eol.toFixed(3) + ", fade " + fade.toFixed(1) + "%, dominant " + top[0],
      action: action,
      next: next,
      confidence: conf,
      confidence_detail: diagConf.detail
    });
  }

  var bms = window.__kfBms;
  if (bms && bms.frames && bms.frames.length) {
    var frame = bms.frames[bms.frames.length - 1];
    var tmax = Math.max.apply(null, frame.temps);
    var risk = Number(frame.maxRisk);
    if (!Number.isFinite(risk)) risk = 0;
    var gate = bms.threshold || 0.42;
    var bmsConf = computeBmsConfidence(bms, frame);
    var bmsSev = risk >= gate ? "critical" : risk >= gate * 0.80 ? "warn" : "ok";
    var bmsAction = risk >= gate
      ? "[B-ALERT-01] Cool or isolate C" + frame.maxCell + " first; inspect Rct/RSEI rise before assuming pack-wide failure."
      : "Keep monitoring; this run stays below the configured action gate.";
    if (risk >= gate && Number.isFinite(frame.rcts[frame.maxCell]) && Number.isFinite(frame.rseis[frame.maxCell])) {
      var eisSum = frame.rcts[frame.maxCell] + frame.rseis[frame.maxCell];
      if (eisSum > 0.060) {
        bmsAction = "[B-EIS-02] C" + frame.maxCell + " crossed risk gate with high impedance drift; isolate cell and trigger impedance check.";
      }
    }
    addDecision(items, {
      severity: bmsSev,
      source: "BMS",
      owner: "Pack Safety",
      evidence: "C" + frame.maxCell + " risk " + risk.toFixed(3) + " / gate " + gate.toFixed(2) + ", Tmax " + tmax.toFixed(1) + " C, seed " + (bms.seed == null ? "--" : bms.seed),
      action: bmsAction,
      next: "Repeat the run across 5 seeds and one lower threshold to estimate false-negative sensitivity before pack policy changes.",
      confidence: bmsConf.confidence,
      confidence_detail: bmsConf.detail
    });
  }
  var sweep = window.__kfBmsSweep;
  if (sweep && Number.isFinite(sweep.meanRisk) && Number.isFinite(sweep.stdRisk) && Number.isFinite(sweep.alertRate)) {
    var sameScenario = !bms || !bms.scenarioKey || !sweep.scenarioKey || bms.scenarioKey === sweep.scenarioKey;
    if (sameScenario) {
      var sweepSev = sweep.alertRate >= 0.7 ? "warn" : sweep.alertRate <= 0.15 ? "ok" : "info";
      var sweepConf = clamp(0.74 - sweep.stdRisk * 1.8 - Math.abs(sweep.alertRate - 0.5) * 0.10, 0.26, 0.86);
      addDecision(items, {
        severity: sweepSev,
        source: "BMS Sweep",
        owner: "Pack Safety",
        evidence: sweep.count + " seeds, alert rate " + (sweep.alertRate * 100).toFixed(1) + "%, mean maxRisk " + sweep.meanRisk.toFixed(3) + " +/- " + sweep.stdRisk.toFixed(3) + ", detect " + (sweep.meanDetectionTime == null ? "--" : sweep.meanDetectionTime.toFixed(0) + "s") + ", fault-hit " + (sweep.faultHitRate == null ? "--" : (sweep.faultHitRate * 100).toFixed(0) + "%"),
        action: sweep.alertRate >= 0.7
          ? "This configuration trips frequently across seeds; tighten cooling or reduce thermal stress before policy rollout."
          : sweep.alertRate <= 0.15
            ? "This configuration is robust across sampled seeds; keep it as a baseline while verifying with measured telemetry."
            : "Seed sensitivity is moderate; keep this setup in watch mode and validate against additional pack logs.",
        next: "Run one physical telemetry capture with the same threshold and compare observed alert frequency to this sweep.",
        confidence: sweepConf,
        confidence_detail: "Seed sweep variability check; includes time-to-detection and non-fault alert share " + ((Number(sweep.nonFaultAlertRate) || 0) * 100).toFixed(0) + "%."
      });
    }
  }

  var mat = window.__kfMaterials;
  if (mat && Number.isFinite(mat.score)) {
    var matConf = computeMaterialsConfidence(mat);
    var synthReady = mat.stability > 0.72 && mat.fade500 < 0.16 && mat.chargeRisk < 0.28 && mat.oxygenRisk < 0.45;
    var matSev = synthReady ? "ok" : (mat.oxygenRisk > 0.55 || mat.chargeRisk > 0.45 || mat.fade500 > 0.24 ? "critical" : "warn");
    var blockers = [];
    if (mat.oxygenRisk > 0.45) blockers.push("oxygen risk");
    if (mat.chargeRisk > 0.28) blockers.push("charge balance");
    if (mat.fade500 > 0.16) blockers.push("fade@500");
    if (mat.stability <= 0.72) blockers.push("stability");
    addDecision(items, {
      severity: matSev,
      source: "Materials",
      owner: "Synthesis",
      evidence: "score " + mat.score.toFixed(3) + ", stability " + mat.stability.toFixed(2) + ", fade@500 " + (mat.fade500 * 100).toFixed(1) + "%",
      action: synthReady ? "Move this composition into a small synthesis queue." : "Keep in simulation queue until " + blockers.slice(0, 2).join(" and ") + " improve.",
      next: synthReady
        ? "Prepare a coin-cell batch with XRD phase check before cycling."
        : "Run the same composition with Al/Ti toggles and lower upper-voltage stress; export the best candidate.",
      confidence: matConf.confidence,
      confidence_detail: matConf.detail
    });
  }

  var rec = window.__kfRecycling;
  if (rec && Number.isFinite(rec.totalRecovered)) {
    var recConf = computeRecyclingConfidence(rec);
    var mn = rec.targets && rec.targets.length ? rec.targets[0] : 0;
    var recSev = rec.marginProxy < 0 ? "critical" : (mn < 0.82 || rec.purity < 0.86 ? "warn" : "ok");
    addDecision(items, {
      severity: recSev,
      source: "Recycling",
      owner: "Process",
      evidence: "recovered " + rec.totalRecovered.toFixed(1) + " kg, Mn " + (mn * 100).toFixed(1) + "%, purity " + (rec.purity * 100).toFixed(1) + "%, margin INR " + compactNumber(rec.marginProxy, 0),
      action: recSev === "ok" ? "Recipe clears the current pilot gate." : "Do not pilot as-is; improve recovery, purity, or cost first.",
      next: recSev === "ok"
        ? "Run a bench leach with duplicate assays for Mn/Fe/Na and compare against the exported 90% interval."
        : "Sweep acid concentration, time, and particle size before committing black mass.",
      confidence: recConf.confidence,
      confidence_detail: recConf.detail
    });
  }

  var byod = window.__kfByod;
  if (byod && byod.predictions) {
    var uploadConf = computeUploadConfidence(byod);
    var pred = byod.predictions || {};
    var warnings = Array.isArray(byod.warnings) ? byod.warnings : [];
    var soh = Number(pred.soh);
    var c = Number(pred.confidence);
    var uploadSev = warnings.length || c < 0.55 ? "warn" : (soh < 0.85 ? "critical" : "ok");
    addDecision(items, {
      severity: uploadSev,
      source: "Upload",
      owner: "Data",
      evidence: (byod.filename || "upload") + ": SOH " + (Number.isFinite(soh) ? (soh * 100).toFixed(1) + "%" : "--") + ", confidence " + (Number.isFinite(c) ? (c * 100).toFixed(0) + "%" : "--") + ", warnings " + warnings.length,
      action: warnings.length ? "Fix parser/data warnings before making a cell claim." : "Use the extracted features as the measured-data anchor.",
      next: "Export JSON, attach the original cycler file, and rerun after adding temperature and impedance columns if available.",
      confidence: uploadConf.confidence,
      confidence_detail: uploadConf.detail
    });
  }

  if (!items.length) {
    addDecision(items, {
      severity: "info",
      source: "Workbench",
      owner: "Operator",
      evidence: "No current panel outputs.",
      action: "Run Diagnostics, BMS, Materials, Recycling, or Upload to populate the action queue.",
      next: "Start with Diagnostics for cell chemistry or BMS for pack safety.",
      confidence: null
    });
  }
  items.sort(function (a, b) { return severityRank(a.severity) - severityRank(b.severity); });
  window.__kfDecisionItems = items;
  return items;
}

function renderDecisionConsole() {
  var body = document.getElementById("decision-action-body");
  if (!body) return;
  var items = buildDecisionItems();
  var open = items.filter(function (x) { return x.severity === "critical" || x.severity === "warn"; }).length;
  var critical = items.filter(function (x) { return x.severity === "critical"; }).length;
  var confs = items.map(function (x) { return Number(x.confidence); }).filter(Number.isFinite);
  var avgConf = confs.length ? confs.reduce(function (a, b) { return a + b; }, 0) / confs.length : null;
  var set = function (id, value) {
    var el = document.getElementById(id);
    if (el) el.textContent = value;
  };
  set("decision-open", String(open));
  set("decision-critical", String(critical));
  set("decision-confidence", avgConf == null ? "--" : Math.round(avgConf * 100) + "%");
  set("decision-last-run", latestRunLabel());
  body.innerHTML = items.map(function (it) {
    var confDetail = it.confidence_detail ? '<div class="decision-conf-note">' + escapeHtml(it.confidence_detail) + "</div>" : "";
    return "<tr>"
      + "<td>" + severityTag(it.severity) + "</td>"
      + "<td>" + escapeHtml(it.source) + "</td>"
      + "<td>" + escapeHtml(it.evidence) + "</td>"
      + "<td>" + escapeHtml(formatConfidence(it.confidence)) + confDetail + "</td>"
      + "<td>" + escapeHtml(it.action) + "</td>"
      + "<td>" + escapeHtml(it.owner) + "</td>"
      + "</tr>";
  }).join("");

  var ticketBox = document.getElementById("decision-ticket-list");
  if (ticketBox) {
    ticketBox.innerHTML = items.slice(0, 6).map(function (it) {
      return '<div class="experiment-card ' + escapeHtml(it.severity) + '">'
        + '<div class="experiment-head"><span>' + escapeHtml(it.source) + '</span>' + severityTag(it.severity) + '</div>'
        + '<div class="experiment-action">' + escapeHtml(it.next) + '</div>'
        + '<div class="experiment-meta">Owner: ' + escapeHtml(it.owner) + ' | Confidence: ' + escapeHtml(formatConfidence(it.confidence)) + ' | Evidence: ' + escapeHtml(it.evidence) + '</div>'
        + '</div>';
    }).join("");
  }

  var historyBody = document.getElementById("decision-history-body");
  if (historyBody) {
    historyBody.innerHTML = kfRunHistory.slice(0, 10).map(function (r) {
      return "<tr>"
        + "<td>" + escapeHtml(new Date(r.at).toLocaleString()) + "</td>"
        + "<td>" + escapeHtml(r.label || runKindLabel(r.kind)) + "</td>"
        + "<td>" + escapeHtml(r.summary || "--") + "</td>"
        + "<td>" + escapeHtml(r.key_metric || "--") + "</td>"
        + "</tr>";
    }).join("") || '<tr><td colspan="4">Run a panel to start local history.</td></tr>';
  }
}

function decisionMemoMarkdown() {
  var items = buildDecisionItems();
  var lines = [
    "# KineticsForge Decision Memo",
    "",
    "Generated: " + new Date().toISOString(),
    "Claim level: simulation-backed unless an uploaded measured dataset is referenced.",
    "",
    "## Action Queue"
  ];
  items.forEach(function (it) {
    lines.push("- [" + it.severity.toUpperCase() + "] " + it.source + " | " + it.action + " | Evidence: " + it.evidence + " | Owner: " + it.owner);
  });
  lines.push("", "## Experiment Tickets");
  items.forEach(function (it, idx) {
    lines.push((idx + 1) + ". " + it.next);
  });
  lines.push("", "## Recent Runs");
  (kfRunHistory.slice(0, 10)).forEach(function (r) {
    lines.push("- " + r.at + " | " + (r.label || runKindLabel(r.kind)) + " | " + (r.summary || "--") + " | " + (r.key_metric || "--"));
  });
  return lines.join("\n");
}

function exportDecisionMemoMarkdown() {
  downloadText(decisionMemoMarkdown(), "kineticsforge_decision_memo.md", "text/markdown;charset=utf-8");
}

function exportDecisionMemoJSON() {
  downloadJSON({
    format: "kineticsforge_decision_memo_v1",
    generated_at: new Date().toISOString(),
    claim_level: "simulation-backed unless current upload data is referenced",
    actions: buildDecisionItems(),
    run_history: kfRunHistory.slice(0, 20)
  }, "kineticsforge_decision_memo.json");
}

function exportExperimentTicketsCSV() {
  var rows = buildDecisionItems().map(function (it, idx) {
    return {
      ticket_id: "KF-" + String(idx + 1).padStart(3, "0"),
      severity: it.severity,
      source: it.source,
      owner: it.owner,
      evidence: it.evidence,
      action: it.action,
      next_experiment: it.next,
      confidence: it.confidence == null ? "" : it.confidence
    };
  });
  downloadCSV(rows, "kineticsforge_experiment_tickets.csv");
}

function valueOf(id) {
  var el = document.getElementById(id);
  if (!el) return null;
  return el.type === "checkbox" ? !!el.checked : el.value;
}

function setControlValue(id, value) {
  var el = document.getElementById(id);
  if (!el || value == null) return;
  if (el.type === "checkbox") el.checked = !!value;
  else el.value = value;
}

function sessionSnapshot() {
  var ids = [
    "temp-slider", "crate-slider", "cycles-slider", "diag-na", "diag-mn", "diag-fe", "diag-dop", "diag-dopant-type",
    "diag-k-sei", "diag-ea-sei", "diag-p2-k", "diag-ea-p2", "diag-p2-soc", "diag-jt-scale", "diag-jt-coupling", "diag-bv-scale", "diag-stress-exp", "diag-residual-scale",
    "sw-p2o2", "sw-jt", "sw-sei", "sw-neural",
    "pack-slider", "dur-slider", "bms-topology", "bms-format", "bms-ambient", "bms-cth", "bms-kedge", "bms-cool", "bms-load", "bms-rct-gate", "bms-risk-thresh", "bms-loss-ratio", "bms-seed", "sw-fault", "sw-eis", "sw-asym",
    "na-slider", "mn-slider", "fe-slider", "sw-al", "sw-ti", "mat-w-cap", "mat-w-stab", "mat-w-fade", "mat-w-cost", "mat-upper-v", "mat-ehull-slope", "mat-charge-penalty", "mat-defect-penalty",
    "bm-slider", "acid-slider", "leach-slider", "rec-time", "rec-particle", "rec-acid-order", "rec-ea-mn", "rec-acid-cost", "rec-energy-cost", "rec-processing-cost", "rec-metal-price", "sw-mc", "sw-bayes",
    "range-soh", "range-energy", "range-pack-kg", "range-eff", "range-target", "range-motor-kw", "climate-region", "climate-days", "climate-temp-offset", "climate-charge-stress", "climate-rh-bias"
  ];
  var controls = {};
  ids.forEach(function (id) {
    var val = valueOf(id);
    if (val != null) controls[id] = val;
  });
  return {
    format: "kineticsforge_session_v1",
    generated_at: new Date().toISOString(),
    controls: controls,
    run_history: kfRunHistory.slice(0, 40),
    decisions: buildDecisionItems()
  };
}

function saveSessionJSON() {
  downloadJSON(sessionSnapshot(), "kineticsforge_session.json");
}

function loadSessionJSONFile(event) {
  var file = event && event.target && event.target.files ? event.target.files[0] : null;
  if (!file) return;
  var reader = new FileReader();
  reader.onload = function () {
    try {
      var data = JSON.parse(String(reader.result || "{}"));
      var controls = data.controls || {};
      Object.keys(controls).forEach(function (id) { setControlValue(id, controls[id]); });
      if (Array.isArray(data.run_history)) {
        kfRunHistory = data.run_history.slice(0, 40);
        window.__kfRunHistory = kfRunHistory;
        saveRunHistory();
      }
      updateDiag();
      updateBMS();
      updateMat();
      renderDecisionConsole();
      showToast("Session controls restored from JSON.", "ok");
    } catch (err) {
      showToast("Could not load session JSON.", "warn");
    } finally {
      if (event.target) event.target.value = "";
    }
  };
  reader.readAsText(file);
}

function clearRunHistory() {
  kfRunHistory = [];
  window.__kfRunHistory = kfRunHistory;
  saveRunHistory();
  renderDecisionConsole();
  showToast("Local run history cleared.", "ok");
}

// Diffusion-limited SEI: single coefficient replaces old dual-term (constant + sqrt) model.
// Ploehn-Ramadass parabolic kinetics: dQ_SEI/dN ~ 1/(2*sqrt(N)), no hidden Nf parameter.
var SEI_GROWTH_COEFF = 0.048;
var SEI_REF_EA = 0.56;
var SEI_REF_TEMP = 318.15;
var GLOBAL_DEGRADATION_SCALE = 0.052;
var JT_LOSS_COEFF = 6.5e-3;
var DESOLV_LOSS_COEFF = 2.5e-4;
var BV_RATE_LOSS_COEFF = 1.2e-4;
var RESIDUAL_LOSS_COEFF = 1.0e-5;
// Legacy fixed P2-O2 branch weighting is absorbed into the default P2 rate knob.
// so the user can override it by adjusting a single knob without a hidden multiplier.
var RESIDUAL_MAX_FRACTION = 0.15; // residual can't exceed 15% of total explicit loss
var RECYCLING_MC_SAMPLES = 200;

// Navigation
function navigate(p) {
  document.querySelectorAll(".section").forEach(function (s) { s.classList.remove("active"); });
  document.querySelectorAll(".nav-links a").forEach(function (a) { a.classList.remove("active"); });
  var s = document.getElementById("sec-" + p);
  if (s) {
    s.classList.add("active");
    s.style.animation = "none";
    s.offsetHeight;
    s.style.animation = "";
  }
  var l = document.getElementById("nav-" + p);
  if (l) l.classList.add("active");
  if (p === "decisions") renderDecisionConsole();
  window.scrollTo(0, 0);
}

function animC(el, t, d) {
  if (!el) return;
  d = d || 1200;
  var s = performance.now();
  var startVal = parseCompactNumber(el.textContent);
  var format = t > 9999 ? function (v) { return (v / 1e6).toFixed(1) + "M"; } : function (v) { return Math.round(v).toString(); };
  requestAnimationFrame(function step(n) {
    var p = Math.min((n - s) / d, 1);
    el.textContent = format(startVal + (t - startVal) * (1 - Math.pow(1 - p, 3)));
    if (p < 1) requestAnimationFrame(step);
  });
}

// Canvas primitives
function makeCanvas(id) {
  var c = document.getElementById(id);
  if (!c) return null;
  var cv = c.querySelector("canvas");
  if (!cv) {
    cv = document.createElement("canvas");
    c.appendChild(cv);
  }
  var r = c.getBoundingClientRect();
  var w = Math.max(220, Math.floor(r.width - 32));
  var h = Math.max(120, Math.floor(r.height - 32));
  cv.width = w;
  cv.height = h;
  return cv;
}

function drawGrid(ctx, p, pw, ph) {
  ctx.strokeStyle = "rgba(255,26,26,0.07)";
  ctx.lineWidth = 1;
  for (var i = 0; i <= 4; i++) {
    var y = p.t + ph * i / 4;
    ctx.beginPath();
    ctx.moveTo(p.l, y);
    ctx.lineTo(p.l + pw, y);
    ctx.stroke();
  }
}

function drawMultiLine(cv, series, opts) {
  if (!cv || !series.length) return;
  opts = opts || {};
  var ctx = cv.getContext("2d");
  var W = cv.width, H = cv.height;
  var p = { t: 28, r: 18, b: 34, l: 52 };
  var pw = W - p.l - p.r, ph = H - p.t - p.b;
  var vals = [];
  series.forEach(function (s) { vals = vals.concat(s.values.filter(Number.isFinite)); });
  if (!vals.length) return;
  var yMn = opts.yMin != null ? opts.yMin : Math.min.apply(null, vals);
  var yMx = opts.yMax != null ? opts.yMax : Math.max.apply(null, vals);
  if (Math.abs(yMx - yMn) < 1e-9) { yMx += 0.1; yMn -= 0.1; }
  ctx.clearRect(0, 0, W, H);
  drawGrid(ctx, p, pw, ph);
  if (opts.title) {
    ctx.fillStyle = "#777";
    ctx.font = "10px JetBrains Mono, monospace";
    ctx.fillText(opts.title, p.l + 6, 15);
  }
  if (opts.band && opts.band.lo && opts.band.hi && opts.band.lo.length === opts.band.hi.length && opts.band.lo.length > 1) {
    ctx.beginPath();
    opts.band.hi.forEach(function (v, i) {
      var x = p.l + i / (opts.band.hi.length - 1) * pw;
      var y = p.t + ph - (v - yMn) / (yMx - yMn) * ph;
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    for (var bi = opts.band.lo.length - 1; bi >= 0; bi--) {
      var bv = opts.band.lo[bi];
      var bx = p.l + bi / (opts.band.lo.length - 1) * pw;
      var by = p.t + ph - (bv - yMn) / (yMx - yMn) * ph;
      ctx.lineTo(bx, by);
    }
    ctx.closePath();
    ctx.fillStyle = opts.band.color || "rgba(255,26,26,0.12)";
    ctx.fill();
  }
  series.forEach(function (s) {
    var values = s.values;
    if (values.length < 2) return;
    ctx.beginPath();
    for (var i = 0; i < values.length; i++) {
      var x = p.l + i / (values.length - 1) * pw;
      var y = p.t + ph - (values[i] - yMn) / (yMx - yMn) * ph;
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.strokeStyle = s.color || "#ff1a1a";
    ctx.lineWidth = s.width || 2;
    ctx.shadowColor = s.color || "#ff1a1a";
    ctx.shadowBlur = s.glow ? 8 : 0;
    ctx.stroke();
    ctx.shadowBlur = 0;
  });
  ctx.fillStyle = "#5f5f5f";
  ctx.font = "10px JetBrains Mono, monospace";
  for (var j = 0; j <= 4; j++) {
    var v = yMn + (yMx - yMn) * (1 - j / 4);
    ctx.fillText(fmt(v, opts.yDigits == null ? 2 : opts.yDigits), 2, p.t + ph * j / 4 + 4);
  }
  if (opts.xMax != null) {
    ctx.fillText("0", p.l, H - 6);
    ctx.fillText(String(opts.xMax), p.l + pw - 32, H - 6);
  }
  if (opts.legend) {
    var lx = W - 150, ly = 14;
    series.forEach(function (s, i) {
      ctx.fillStyle = s.color || "#ff1a1a";
      ctx.fillRect(lx, ly + i * 13 - 7, 14, 2);
      ctx.fillStyle = "#777";
      ctx.fillText(s.name || "", lx + 20, ly + i * 13 - 3);
    });
  }
  if (opts.points && opts.points.length) {
    ctx.fillStyle = opts.pointColor || "#ffffff";
    opts.points.forEach(function (pt) {
      if (!Number.isFinite(pt.x) || !Number.isFinite(pt.y)) return;
      var x = p.l + clamp(pt.x / Math.max(1, opts.xMax || pt.x), 0, 1) * pw;
      var y = p.t + ph - (pt.y - yMn) / (yMx - yMn) * ph;
      ctx.beginPath();
      ctx.arc(x, y, 3.2, 0, Math.PI * 2);
      ctx.fill();
    });
  }
}

function drawLine(cv, vals, opts) {
  drawMultiLine(cv, [{ name: opts && opts.name, values: vals, color: opts && opts.color || "#ff1a1a", glow: true }], opts);
}

function drawScatter(cv, pts, opts) {
  if (!cv || !pts.length) return;
  opts = opts || {};
  var ctx = cv.getContext("2d");
  var W = cv.width, H = cv.height;
  var p = { t: 28, r: 18, b: 34, l: 52 };
  var pw = W - p.l - p.r, ph = H - p.t - p.b;
  var xs = pts.map(function (p) { return p.x; });
  var ys = pts.map(function (p) { return p.y; });
  var xMn = Math.min.apply(null, xs) - 6, xMx = Math.max.apply(null, xs) + 6;
  var yMn = Math.max(0, Math.min.apply(null, ys) - 0.05), yMx = Math.min(1.15, Math.max.apply(null, ys) + 0.05);
  ctx.clearRect(0, 0, W, H);
  drawGrid(ctx, p, pw, ph);
  if (opts.title) {
    ctx.fillStyle = "#777";
    ctx.font = "10px JetBrains Mono, monospace";
    ctx.fillText(opts.title, p.l + 6, 15);
  }
  pts.forEach(function (pt) {
    var x = p.l + (pt.x - xMn) / (xMx - xMn) * pw;
    var y = p.t + ph - (pt.y - yMn) / (yMx - yMn) * ph;
    var r = pt.selected ? 7 : pt.front ? 4.5 : 3.2;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = pt.selected ? "rgba(255,26,26,1)" : pt.front ? "rgba(255,92,92,0.86)" : "rgba(255,120,120,0.35)";
    ctx.shadowColor = "#ff1a1a";
    ctx.shadowBlur = pt.selected ? 12 : 0;
    ctx.fill();
    ctx.shadowBlur = 0;
  });
  ctx.fillStyle = "#606060";
  ctx.font = "10px JetBrains Mono, monospace";
  ctx.fillText("Capacity (mAh/g)", p.l + pw / 2 - 42, H - 6);
  ctx.fillText("Stability", 2, p.t + ph / 2);
}

// Na-ion degradation physics mirror from core/phase_transition.py.
function degradationKnobs() {
  return {
    seiScale: num("diag-k-sei", 1.0),
    seiEa: num("diag-ea-sei", 0.56),
    p2Rate: num("diag-p2-k", 0.00182),
    p2Ea: num("diag-ea-p2", 0.22),
    p2Soc: num("diag-p2-soc", 0.78),
    jtScale: num("diag-jt-scale", 1.0),
    jtCoupling: num("diag-jt-coupling", 0.35),
    bvScale: num("diag-bv-scale", 1.0),
    stressExp: num("diag-stress-exp", 0.55),
    residualScale: num("diag-residual-scale", 1.0)
  };
}
function diagnosticComposition() {
  var dopTypeEl = document.getElementById("diag-dopant-type");
  return {
    Na: clamp(num("diag-na", 1.02), 0.60, 1.20),
    Mn: clamp(num("diag-mn", 0.52), 0.05, 0.95),
    Fe: clamp(num("diag-fe", 0.43), 0.05, 0.95),
    dopant_frac: clamp(num("diag-dop", 0.05), 0, 0.25),
    dopant_type: dopTypeEl ? dopTypeEl.value : "Al"
  };
}
function setDiagnosticComposition(comp) {
  [["diag-na", comp.Na], ["diag-mn", comp.Mn], ["diag-fe", comp.Fe], ["diag-dop", comp.dopant_frac == null ? 0.05 : comp.dopant_frac]].forEach(function (item) {
    var el = document.getElementById(item[0]);
    if (el && Number.isFinite(item[1])) el.value = Number(item[1]).toFixed(item[0] === "diag-dop" ? 3 : 2);
  });
  var typeEl = document.getElementById("diag-dopant-type");
  if (typeEl && comp.dopant_type) typeEl.value = comp.dopant_type;
  updateDiag();
}

function diagnosticDopantFactors(type) {
  if (type === "Ti") {
    return { label: "Ti", jtSuppression: 1.05, p2Shift: 0.14, p2Suppression: 0.32, barrierDrop: 0.075 };
  }
  if (type === "generic") {
    return { label: "generic", jtSuppression: 0.70, p2Shift: 0.18, p2Suppression: 0.42, barrierDrop: 0.050 };
  }
  return { label: "Al", jtSuppression: 0.55, p2Shift: 0.26, p2Suppression: 0.58, barrierDrop: 0.035 };
}

function naIonTerms(state, comp, T, cfg) {
  cfg = cfg || degradationKnobs();
  var kB = 8.617e-5;
  var soc = clamp(state.soc, 0, 1);
  var mn = clamp(comp.Mn, 0, 1.5);
  var fe = clamp(comp.Fe, 0, 1.5);
  var dop = clamp(comp.dopant_frac || 0, 0, 0.25);
  var dopant = diagnosticDopantFactors(comp.dopant_type);
  var jt = clamp(cfg.jtScale * mn * clamp(1.15 - soc, 0, 1) * expClamp((T - 298.15) * 0.018, -4, 4) * Math.exp(-0.45 * fe - dopant.jtSuppression * dop), 0, 4);
  var socCrit = clamp(cfg.p2Soc - 0.09 * mn + 0.06 * fe + dopant.p2Shift * dop, 0.55, 0.95);
  var p2Gate = sigmoid((soc - socCrit) / 0.045);
  var p2Ref = Math.exp(-cfg.p2Ea / (kB * SEI_REF_TEMP));
  var p2Arrhenius = Math.exp(-cfg.p2Ea / (kB * T)) / Math.max(p2Ref, 1e-30);
  var p2o2Rate = clamp(cfg.p2Rate * p2Gate * p2Arrhenius * (1 + cfg.jtCoupling * jt) * Math.exp(-dopant.p2Suppression * dop), 0, 0.08);
  // Na+ desolvation barrier: 0.4-0.6 eV (Jian et al., Komaba et al.) — not 0.18 eV (SEI migration)
  var barrier = 0.50 + 0.025 * mn - 0.014 * fe - dopant.barrierDrop * dop;
  var desolv = clamp(Math.exp(clamp(barrier / (kB * T + 1e-10), -2, 4)) * (1 + 0.25 * relu(soc - 0.85)), 0.2, 30);
  var beta = clamp(0.48 - 0.035 * Math.log1p(desolv) + 0.025 * clamp(soc - 0.5, -0.5, 0.5), 0.25, 0.75);
  // Unified SEI: Arrhenius factor normalized to reference T=318.15K
  var seiRef = Math.exp(-cfg.seiEa / (kB * SEI_REF_TEMP));
  var seiArrhenius = cfg.seiScale * Math.exp(-cfg.seiEa / (kB * T)) / Math.max(seiRef, 1e-30);
  return { jt: jt, p2o2Rate: p2o2Rate, desolv: desolv, beta: beta, seiRate: seiArrhenius, socCrit: socCrit, dopant: dopant.label };
}

function simulateDegradation(options) {
  options = options || {};
  var T = (options.temperatureC != null ? options.temperatureC : parseFloat(document.getElementById("temp-slider").value)) + 273.15;
  var cRate = options.cRate != null ? options.cRate : parseFloat(document.getElementById("crate-slider").value);
  var nCycles = options.cycles != null ? options.cycles : parseInt(document.getElementById("cycles-slider").value, 10);
  var enableP2 = document.getElementById("sw-p2o2").checked;
  var enableJt = document.getElementById("sw-jt").checked;
  var enableSei = document.getElementById("sw-sei").checked;
  var enableNeural = document.getElementById("sw-neural").checked;
  var cfg = Object.assign({}, degradationKnobs(), options.cfg || {});
  var comp = options.comp || diagnosticComposition();
  var Q = 1.0, V = 3.34;
  var cap = [Q], voltage = [V], p2Cum = [0], jtCum = [0], seiCum = [0], rateCum = [0], resCum = [0];
  var socSeries = [0.78], p2RateSeries = [0], jtSeries = [0], desolvSeries = [0], dominant = ["SEI"];
  var p2 = 0, jt = 0, sei = 0, rate = 0, res = 0, knee = -1;
  var stress = 0.6 + Math.pow(cRate, cfg.stressExp);
  var maxSubsteps = 1;
  for (var i = 1; i <= nCycles; i++) {
    var substeps = Math.max(1, Math.min(4, Math.round(1 + 0.9 * Math.max(0, cRate - 0.9) + 0.7 * Math.max(0, (T - SEI_REF_TEMP) / 20))));
    maxSubsteps = Math.max(maxSubsteps, substeps);
    var cycleScale = GLOBAL_DEGRADATION_SCALE * stress;
    var dtScale = cycleScale / substeps;
    var cycleP2 = 0, cycleJt = 0, cycleSei = 0, cycleRate = 0, cycleRes = 0;
    var cycleSoc = 0.78;
    var cycleP2Rate = 0, cycleJtRaw = 0, cycleDesolvLog = 0;
    for (var s = 0; s < substeps; s++) {
      var n0 = (i - 1) + s / substeps;
      var n1 = (i - 1) + (s + 1) / substeps;
      var nMid = (n0 + n1) * 0.5;
      var sohWindow = clamp(Q, 0.50, 1.0);
      var socBase = 0.78 + 0.04 * Math.min(1, cRate / 2.4);
      var usableSoc = 0.62 + 0.38 * sohWindow;
      var soc = clamp(0.55 + (socBase - 0.55) * usableSoc + 0.022 * Math.sin(nMid * 0.17) * usableSoc, 0.55, 0.98);
      var terms = naIonTerms({ Q: Q, V: V, soc: soc }, comp, T, cfg);
      var sqrtIncrement = Math.sqrt(Math.max(n1, 1e-9)) - Math.sqrt(Math.max(n0, 0));
      var seiLoss = enableSei ? Q * terms.seiRate * SEI_GROWTH_COEFF * sqrtIncrement * cycleScale : 0;
      var p2Loss = enableP2 ? Q * terms.p2o2Rate * dtScale : 0;
      var jtLoss = enableJt ? Q * JT_LOSS_COEFF * terms.jt * dtScale : 0;
      var desolvLoss = Q * DESOLV_LOSS_COEFF * Math.log1p(terms.desolv) * dtScale;
      var exchangeProxy = clamp(0.34 + 0.18 * comp.Fe - 0.08 * Math.log1p(terms.desolv) + 0.04 * (1 - terms.beta), 0.08, 0.9);
      var eta = Math.asinh(cRate / (2 * exchangeProxy));
      var rateStress = 1 + 0.20 * Math.pow(Math.max(0, cRate - 1.5), 2);
      var rateLoss = Q * cfg.bvScale * BV_RATE_LOSS_COEFF * eta * eta * rateStress * dtScale;
      var explicitLoss = seiLoss + p2Loss + jtLoss + desolvLoss + rateLoss;
      var progress = n1 / Math.max(1, nCycles);
      var rawResidualLoss = enableNeural ? Q * cfg.residualScale * RESIDUAL_LOSS_COEFF * sigmoid((progress - 0.62) / 0.16) * (0.8 + 0.35 * cRate) / substeps : 0;
      var residualLoss = Math.min(rawResidualLoss, explicitLoss * RESIDUAL_MAX_FRACTION);
      Q = clamp(Q - (explicitLoss + residualLoss), 0.25, 1.02);
      cycleP2 += p2Loss;
      cycleJt += jtLoss;
      cycleSei += seiLoss + desolvLoss;
      cycleRate += rateLoss;
      cycleRes += residualLoss;
      cycleSoc = soc;
      cycleP2Rate = terms.p2o2Rate;
      cycleJtRaw = terms.jt;
      cycleDesolvLog = Math.log1p(terms.desolv);
    }
    p2 += cycleP2;
    jt += cycleJt;
    sei += cycleSei;
    rate += cycleRate;
    res += cycleRes;
    var vDegradation = p2 * 0.15 + jt * 0.08 + sei * 0.05 + rate * 0.04;
    V = clamp(3.34 - vDegradation, 2.4, 3.5);
    cap.push(Q); voltage.push(V); p2Cum.push(p2 * 100); jtCum.push(jt * 100); seiCum.push(sei * 100); rateCum.push(rate * 100); resCum.push(res * 100);
    socSeries.push(cycleSoc); p2RateSeries.push(cycleP2Rate); jtSeries.push(cycleJtRaw); desolvSeries.push(cycleDesolvLog);
    var termsNow = [
      ["P2-O2", cycleP2],
      ["JT", cycleJt],
      ["SEI", cycleSei],
      ["Rate", cycleRate],
      ["Residual", cycleRes]
    ].sort(function (a, b) { return b[1] - a[1]; });
    dominant.push(termsNow[0][0]);
    if (knee < 0 && i > 10) {
      var d2 = cap[i] - 2 * cap[i - 1] + cap[i - 2];
      if (d2 < -1.6e-5) knee = i;
    }
  }
  return { cap: cap, voltage: voltage, p2: p2Cum, jt: jtCum, sei: seiCum, rate: rateCum, residual: resCum, knee: knee, nCycles: nCycles, soc: socSeries, p2Rate: p2RateSeries, jtRaw: jtSeries, desolv: desolvSeries, dominant: dominant, comp: comp, cfg: cfg, integrator: { mode: "adaptive_substep_euler", maxSubsteps: maxSubsteps } };
}

function updateDiag() {
  document.getElementById("temp-val").textContent = document.getElementById("temp-slider").value + " C";
  document.getElementById("crate-val").textContent = parseFloat(document.getElementById("crate-slider").value).toFixed(1) + "C";
  document.getElementById("cycles-val").textContent = document.getElementById("cycles-slider").value;
  ["diag-na", "diag-mn", "diag-fe", "diag-dop"].forEach(function (id) {
    var el = document.getElementById(id);
    var val = document.getElementById(id + "-val");
    if (el && val) val.textContent = parseFloat(el.value).toFixed(id === "diag-dop" ? 3 : 2);
  });
}

function runDiagnostics() {
  var out = simulateDegradation();
  out.band = uncertaintyBand(out);
  var cv = makeCanvas("diag-chart");
  var vcv = makeCanvas("diag-voltage-chart");
  var mech = makeCanvas("diag-mech-chart");
  var surf = makeCanvas("diag-phase-chart");
  var frame = 0, total = 72, step = Math.max(2, Math.floor(out.cap.length / total));
  function anim() {
    frame++;
    var n = Math.min(frame * step, out.cap.length);
    drawMultiLine(cv, [{ name: "capacity", values: out.cap.slice(0, n), color: "#ff1a1a", glow: true }], { yMin: 0.65, yMax: 1.02, xMax: out.nCycles, title: "Capacity fade with SEI sensitivity range", color: "#ff1a1a", yDigits: 3, band: n > 4 ? { lo: out.band.lo.slice(0, n), hi: out.band.hi.slice(0, n), color: "rgba(255,26,26,0.12)" } : null });
    drawLine(vcv, out.voltage.slice(0, n), { yMin: 3.20, yMax: 3.36, xMax: out.nCycles, title: "Average discharge voltage from accumulated loss terms", color: "#ff9f1a", yDigits: 3 });
    drawMultiLine(mech, [
      { name: "SEI+desolv", values: out.sei.slice(0, n), color: "#ff9f1a" },
      { name: "P2-O2", values: out.p2.slice(0, n), color: "#ff1a1a", glow: true },
      { name: "JT", values: out.jt.slice(0, n), color: "#d946ef" },
      { name: "rate", values: out.rate.slice(0, n), color: "#22c55e" },
      { name: "residual", values: out.residual.slice(0, n), color: "#38bdf8" }
    ], { yMin: 0, xMax: out.nCycles, title: "Cumulative loss contribution (%)", legend: true, yDigits: 1 });
    if (n < out.cap.length) requestAnimationFrame(anim);
  }
  anim();
  drawDegradationSurface(surf, out);
  var eol = out.cap[out.cap.length - 1];
  var r80 = out.cap.findIndex(function (v) { return v < 0.8; });
  window.__kfDiag = out;
  document.getElementById("diag-eol").textContent = eol.toFixed(3);
  document.getElementById("diag-fade").textContent = ((1 - eol) * 100).toFixed(1) + "%";
  document.getElementById("diag-knee").textContent = out.knee > 0 ? out.knee : "N/A";
  document.getElementById("diag-rul").textContent = r80 > 0 ? r80 : ">" + out.nCycles;
  var totals = [
    ["P2-O2 phase transition", out.p2[out.p2.length - 1]],
    ["Jahn-Teller distortion", out.jt[out.jt.length - 1]],
    ["SEI + Na desolvation", out.sei[out.sei.length - 1]],
    ["Butler-Volmer rate stress", out.rate[out.rate.length - 1]],
    ["bounded residual", out.residual[out.residual.length - 1]]
  ].sort(function (a, b) { return b[1] - a[1]; });
  var recommendation = totals[0][0].indexOf("P2") >= 0
    ? "Lower upper cutoff voltage or add Al/Ti stabilization before spending lab cycles."
    : totals[0][0].indexOf("SEI") >= 0
      ? "Prioritize electrolyte/additive screening and lower-temperature cycling validation."
      : totals[0][0].indexOf("Jahn") >= 0
        ? "Reduce Mn3+ fraction or test Ti/Fe compensation."
        : totals[0][0].indexOf("Butler") >= 0
          ? "Reduce C-rate or improve interfacial kinetics before deeper cycling."
          : "Residual is large: calibrate the residual term against holdout cycles before making a claim.";
  setHtml("diag-decision", "<strong>Dominant mechanism:</strong> " + totals[0][0] + " (" + totals[0][1].toFixed(2) + "% loss contribution). <strong>Next experiment:</strong> " + recommendation);
  drawMechanismPie(out);
  var integ = out.integrator || {};
  setHtml("diag-map-note", "State map readout: dominant=" + totals[0][0] + ", EOL=" + eol.toFixed(3) + ", fade=" + ((1 - eol) * 100).toFixed(1) + "%, RUL80=" + (r80 > 0 ? r80 : ">" + out.nCycles) + ", integrator=" + (integ.mode || "explicit") + " (max sub-steps " + (integ.maxSubsteps || 1) + ").");
  var cal = document.getElementById("diag-cal-result");
  if (cal && !cal.textContent.trim()) cal.textContent = "Paste cycle,capacity rows and run calibration to fit SEI/P2/JT/stress coefficients.";
  var diagConf = computeDiagConfidence(out);
  renderConfidence("diag-confidence", diagConf.confidence, diagConf.detail);
  recordRun("diagnostics", {
    summary: "Dominant " + totals[0][0] + ", fade " + ((1 - eol) * 100).toFixed(1) + "% over " + out.nCycles + " cycles",
    key_metric: "EOL " + eol.toFixed(3),
    confidence: diagConf.confidence
  });
}

function uncertaintyBand(base) {
  var lowCfg = Object.assign({}, base.cfg, {
    seiScale: base.cfg.seiScale * 0.85,
    p2Rate: base.cfg.p2Rate * 0.85,
    jtScale: base.cfg.jtScale * 0.85,
    residualScale: base.cfg.residualScale * 0.85
  });
  var highCfg = Object.assign({}, base.cfg, {
    seiScale: base.cfg.seiScale * 1.15,
    p2Rate: base.cfg.p2Rate * 1.15,
    jtScale: base.cfg.jtScale * 1.15,
    residualScale: base.cfg.residualScale * 1.15
  });
  var common = {
    temperatureC: parseFloat(document.getElementById("temp-slider").value),
    cRate: parseFloat(document.getElementById("crate-slider").value),
    cycles: base.nCycles,
    comp: base.comp
  };
  var low = simulateDegradation(Object.assign({}, common, { cfg: lowCfg })).cap;
  var high = simulateDegradation(Object.assign({}, common, { cfg: highCfg })).cap;
  return {
    lo: low.map(function (v, i) { return Math.min(v, high[i]); }),
    hi: low.map(function (v, i) { return Math.max(v, high[i]); })
  };
}

function parseCalibrationData() {
  var el = document.getElementById("diag-cal-data");
  if (!el) return [];
  return el.value.split(/\n+/).map(function (line) {
    var parts = line.trim().split(/[,\s]+/).map(parseFloat);
    if (parts.length < 2 || !Number.isFinite(parts[0]) || !Number.isFinite(parts[1])) return null;
    var capacity = parts[1] > 1.5 ? parts[1] / 100 : parts[1];
    return { cycle: Math.max(0, Math.round(parts[0])), capacity: clamp(capacity, 0.25, 1.05) };
  }).filter(Boolean).sort(function (a, b) { return a.cycle - b.cycle; });
}

function calibrationError(curve, data) {
  var sse = 0;
  data.forEach(function (pt) {
    var idx = clamp(pt.cycle, 0, curve.length - 1);
    var err = curve[idx] - pt.capacity;
    sse += err * err;
  });
  return sse / Math.max(1, data.length);
}

function calibrateDiagnostics() {
  var data = parseCalibrationData();
  var result = document.getElementById("diag-cal-result");
  if (data.length < 5) {
    if (result) result.textContent = "Need at least 5 rows for four fitted parameters: cycle, capacity_fraction_or_percent.";
    return;
  }
  var baseCfg = degradationKnobs();
  var maxCycle = Math.max(parseInt(document.getElementById("cycles-slider").value, 10), data[data.length - 1].cycle);
  var common = {
    temperatureC: parseFloat(document.getElementById("temp-slider").value),
    cRate: parseFloat(document.getElementById("crate-slider").value),
    cycles: maxCycle,
    comp: diagnosticComposition()
  };
  var before = simulateDegradation(Object.assign({}, common, { cfg: baseCfg }));
  var best = { cfg: baseCfg, err: calibrationError(before.cap, data), curve: before.cap };
  [0.55, 0.75, 1.0, 1.30, 1.65].forEach(function (seiM) {
    [0.55, 0.80, 1.0, 1.30, 1.70].forEach(function (p2M) {
      [0.55, 0.85, 1.0, 1.35, 1.80].forEach(function (jtM) {
        [0.92, 1.0, 1.08].forEach(function (stressM) {
          var cfg = Object.assign({}, baseCfg, {
            seiScale: clamp(baseCfg.seiScale * seiM, 0.05, 10),
            p2Rate: clamp(baseCfg.p2Rate * p2M, 0.0002, 0.025),
            jtScale: clamp(baseCfg.jtScale * jtM, 0.05, 4),
            stressExp: clamp(baseCfg.stressExp * stressM, 0.25, 3)
          });
          var out = simulateDegradation(Object.assign({}, common, { cfg: cfg }));
          var err = calibrationError(out.cap, data);
          if (err < best.err) best = { cfg: cfg, err: err, curve: out.cap };
        });
      });
    });
  });
  document.getElementById("diag-k-sei").value = best.cfg.seiScale.toFixed(3);
  document.getElementById("diag-p2-k").value = best.cfg.p2Rate.toFixed(5);
  document.getElementById("diag-jt-scale").value = best.cfg.jtScale.toFixed(3);
  document.getElementById("diag-stress-exp").value = best.cfg.stressExp.toFixed(3);
  var mean = data.reduce(function (a, b) { return a + b.capacity; }, 0) / data.length;
  var sst = data.reduce(function (a, b) { return a + Math.pow(b.capacity - mean, 2); }, 0);
  var r2 = sst > 1e-12 ? 1 - best.err * data.length / sst : 1;
  var rmse = Math.sqrt(best.err);
  drawMultiLine(makeCanvas("diag-cal-chart"), [
    { name: "before", values: before.cap, color: "#777" },
    { name: "calibrated", values: best.curve, color: "#ff1a1a", glow: true }
  ], { yMin: 0.65, yMax: 1.02, xMax: maxCycle, title: "Calibration fit: measured points vs simulated curves", legend: true, yDigits: 3, points: data.map(function (d) { return { x: d.cycle, y: d.capacity }; }), pointColor: "#ffffff" });
  if (result) {
    var fitFlag = rmse > 0.03 ? " Poor fit warning: add more measured points or revisit assumptions." : "";
    result.textContent = "Method: bounded grid-search MSE (375 candidates). Fitted SEI scale=" + best.cfg.seiScale.toFixed(2) + ", P2 rate=" + best.cfg.p2Rate.toFixed(4) + ", JT scale=" + best.cfg.jtScale.toFixed(2) + ", stress exp=" + best.cfg.stressExp.toFixed(3) + ". RMSE=" + rmse.toFixed(4) + ", R2=" + r2.toFixed(3) + "." + fitFlag;
  }
  runDiagnostics();
}

function exportDiagnosticsCSV() {
  var out = window.__kfDiag || simulateDegradation();
  var rows = out.cap.map(function (cap, i) {
    return { cycle: i, capacity: cap, voltage: out.voltage[i], sei_pct: out.sei[i], p2_pct: out.p2[i], jt_pct: out.jt[i], rate_pct: out.rate[i], residual_pct: out.residual[i] };
  });
  downloadCSV(rows, "kineticsforge_diagnostics.csv");
}

function exportBmsCSV() {
  var sim = window.__kfBms || {};
  var rows = [];
  (sim.frames || []).forEach(function (frame) {
    frame.risks.forEach(function (risk, i) {
      rows.push({ time_s: frame.t, cell: "C" + i, risk: risk, temp_C: frame.temps[i], dTdt_K_s: frame.slopes[i], rct_ohm: frame.rcts[i], rsei_ohm: frame.rseis[i] });
    });
  });
  downloadCSV(rows, "kineticsforge_bms.csv");
}

function exportMaterialsCSV() {
  var mat = window.__kfMaterials || {};
  var comp = mat.comp || {};
  downloadCSV([{ Na: comp.Na, Mn: comp.Mn, Fe: comp.Fe, Al: !!comp.al, Ti: !!comp.ti, capacity_mAh_g: mat.Q0, voltage_V: mat.avgVoltage, stability: mat.stability, phase_state: mat.phaseState, phase_stability: mat.phaseStability, p2_o2_risk: mat.p2Risk, fade500: mat.fade500, cycle_life: mat.cycleLife, score: mat.score, oxygen_risk: mat.oxygenRisk, charge_risk: mat.chargeRisk, site_error: mat.siteError, confidence: mat.confidence }], "kineticsforge_materials.csv");
}

function exportRecyclingCSV() {
  var rec = window.__kfRecycling || {};
  var rows = (rec.elements || []).map(function (el, i) {
    return { element: el.n, wt_fraction: el.wt, recovery: rec.targets ? rec.targets[i] : "", recovered_kg: rec.mass ? rec.mass * el.wt * rec.targets[i] : "" };
  });
  rows.push({ element: "TOTAL", wt_fraction: "", recovery: "", recovered_kg: rec.totalRecovered || "" });
  downloadCSV(rows, "kineticsforge_recycling.csv");
}

// ── Mechanism Attribution Donut Chart ────────────────────────────────────
function drawMechanismPie(out) {
  var cv = document.getElementById("diag-pie-canvas");
  var legend = document.getElementById("diag-pie-legend");
  if (!cv || !legend) return;
  var ctx = cv.getContext("2d");
  var W = cv.width, H = cv.height;
  var cx = W / 2, cy = H / 2, R = Math.min(cx, cy) - 8, r = R * 0.52;
  var totalFade = (1 - out.cap[out.cap.length - 1]);
  if (totalFade < 1e-6) { legend.innerHTML = "No significant fade detected."; return; }
  var slices = [
    { name: "SEI + Desolvation", value: out.sei[out.sei.length - 1], color: "#ff9f1a" },
    { name: "P2-O2 Transition", value: out.p2[out.p2.length - 1], color: "#ff1a1a" },
    { name: "Jahn-Teller", value: out.jt[out.jt.length - 1], color: "#d946ef" },
    { name: "Rate Polarization", value: out.rate[out.rate.length - 1], color: "#22c55e" },
    { name: "Residual", value: out.residual[out.residual.length - 1], color: "#38bdf8" }
  ];
  var sum = slices.reduce(function (a, s) { return a + s.value; }, 0);
  if (sum < 1e-6) return;
  ctx.clearRect(0, 0, W, H);
  var start = -Math.PI / 2;
  slices.forEach(function (s) {
    var frac = s.value / sum;
    var sweep = frac * Math.PI * 2;
    ctx.beginPath();
    ctx.arc(cx, cy, R, start, start + sweep);
    ctx.arc(cx, cy, r, start + sweep, start, true);
    ctx.closePath();
    ctx.fillStyle = s.color;
    ctx.globalAlpha = 0.88;
    ctx.fill();
    ctx.globalAlpha = 1.0;
    s.pct = (frac * 100).toFixed(1);
    start += sweep;
  });
  // Center label
  ctx.fillStyle = "#ddd";
  ctx.font = "bold 15px JetBrains Mono, monospace";
  ctx.textAlign = "center";
  ctx.fillText((totalFade * 100).toFixed(1) + "%", cx, cy - 3);
  ctx.font = "10px JetBrains Mono, monospace";
  ctx.fillStyle = "#888";
  ctx.fillText("total fade", cx, cy + 13);
  ctx.textAlign = "start";
  // Legend
  legend.innerHTML = slices.map(function (s) {
    return '<div style="display:flex;align-items:center;gap:0.5rem">'
      + '<span style="display:inline-block;width:12px;height:12px;border-radius:2px;background:' + s.color + '"></span>'
      + '<span>' + s.name + ': <strong>' + s.pct + '%</strong></span></div>';
  }).join("");
}

// ── JSON Exports (all panels) ───────────────────────────────────────────
function exportDiagnosticsJSON() {
  var out = window.__kfDiag || simulateDegradation();
  var payload = {
    format: "kineticsforge_diagnostics_v1",
    generated_at: new Date().toISOString(),
    parameters: {
      temperature_C: parseFloat(document.getElementById("temp-slider").value),
      c_rate: parseFloat(document.getElementById("crate-slider").value),
      cycles: parseInt(document.getElementById("cycles-slider").value, 10),
      composition: out.comp,
      knobs: out.cfg
    },
    results: {
      eol_capacity: out.cap[out.cap.length - 1],
      total_fade_pct: ((1 - out.cap[out.cap.length - 1]) * 100),
      knee_cycle: out.knee > 0 ? out.knee : null,
      rul_80: out.cap.findIndex(function (v) { return v < 0.8; }),
      mechanism_pct: {
        sei_desolv: out.sei[out.sei.length - 1],
        p2o2: out.p2[out.p2.length - 1],
        jt: out.jt[out.jt.length - 1],
        rate: out.rate[out.rate.length - 1],
        residual: out.residual[out.residual.length - 1]
      }
    },
    curves: {
      capacity: out.cap,
      voltage: out.voltage
    }
  };
  downloadJSON(payload, "kineticsforge_diagnostics.json");
}

function exportBmsJSON() {
  var sim = window.__kfBms || {};
  var payload = {
    format: "kineticsforge_bms_v1",
    generated_at: new Date().toISOString(),
    parameters: {
      n_cells: sim.n,
      duration_s: sim.duration,
      seed: sim.seed,
      fault_cell: sim.faultCell,
      topology: sim.topology,
      loss_ratio: sim.lossRatio
    },
    alerts: sim.alerts || [],
    max_risk: sim.maxRisk,
    frames: (sim.frames || []).map(function (f) {
      return { t: f.t, temps: f.temps, risks: f.risks, rcts: f.rcts, rseis: f.rseis, slopes: f.slopes };
    })
  };
  downloadJSON(payload, "kineticsforge_bms.json");
}

function exportMaterialsJSON() {
  var mat = window.__kfMaterials || {};
  var payload = {
    format: "kineticsforge_materials_v1",
    generated_at: new Date().toISOString(),
    composition: mat.comp,
    predicted: {
      capacity_mAh_g: mat.Q0,
      voltage_V: mat.avgVoltage,
      stability: mat.stability,
      fade_500: mat.fade500,
      score: mat.score,
      oxygen_risk: mat.oxygenRisk,
      charge_balance_risk: mat.chargeRisk,
      energy_density: mat.energyDensity,
      cost_usd_kwh: mat.costKwh
    },
    physics: {
      phase_state: mat.phaseState,
      phase_stability: mat.phaseStability,
      p2_o2_risk: mat.p2Risk,
      mn_oxidation_state: mat.mnOx,
      mn3_fraction: mat.mn3Fraction,
      site_balance_error: mat.siteError,
      confidence: mat.confidence,
      evidence: mat.evidence,
      mechanisms: mat.mechanisms,
      retention_curve: mat.retentionCurve
    },
    candidates: mat.candidates || []
  };
  downloadJSON(payload, "kineticsforge_materials.json");
}

function exportRecyclingJSON() {
  var rec = window.__kfRecycling || {};
  var payload = {
    format: "kineticsforge_recycling_v1",
    generated_at: new Date().toISOString(),
    parameters: {
      mass_kg: rec.mass,
      acid_molarity: rec.acid,
      temperature_C: rec.tempC,
      leach_time_min: rec.leachMin,
      particle_um: rec.particleUm,
      economics_inr: rec.economics || {}
    },
    recoveries: {},
    total_recovered_kg: rec.totalRecovered,
    uncertainty_interval: { p05_kg: rec.lo, p95_kg: rec.hi },
    purity_proxy: rec.purity,
    margin_proxy_inr: rec.margin,
    cost_estimate_inr: rec.cost
  };
  (rec.elements || []).forEach(function (el, i) {
    payload.recoveries[el.n] = {
      wt_fraction: el.wt,
      recovery_rate: rec.targets ? rec.targets[i] : null,
      recovered_kg: rec.mass && rec.targets ? rec.mass * el.wt * rec.targets[i] : null
    };
  });
  downloadJSON(payload, "kineticsforge_recycling.json");
}

// ── Formation Efficiency Scorer (BYOD uploads) ─────────────────────────
function renderFormationScore(data) {
  var container = document.getElementById("byod-formation");
  if (!container) return;
  var pred = (data.predictions || {});
  var m14 = ((pred.model_outputs || {}).M14_FormationProtocol || {});
  var m11 = ((pred.model_outputs || {}).M11_ElectrolyteHealth || {});
  var features = data.features || {};
  var ce = features.early_coulombic_efficiency;
  var seiQ = m14.sei_quality;
  var lifeIdx = m14.life_index;
  var robust = m14.robustness_index;
  var proto = m14.suggested_protocol || {};
  if (seiQ == null && ce == null) { container.innerHTML = ""; return; }
  var ceStr = ce != null ? (ce * 100).toFixed(1) + "%" : "--";
  var optimal = "97%+";
  var gap = ce != null ? Math.max(0, 0.97 - ce) * 100 : null;
  var gapStr = gap != null && gap > 0.1 ? gap.toFixed(1) + "% below optimal" : "near optimal";
  function gauge(val, label) {
    if (val == null) return "";
    var pct = Math.round(clamp(val, 0, 1) * 100);
    var color = pct >= 75 ? "#22c55e" : pct >= 50 ? "#ff9f1a" : "#ff1a1a";
    return '<div style="margin:0.5rem 0"><div style="font-size:0.72rem;color:#888;margin-bottom:3px">' + label + '</div>'
      + '<div style="background:#1a1a1a;border-radius:4px;height:10px;overflow:hidden"><div style="height:100%;border-radius:4px;width:' + pct + '%;background:' + color + ';transition:width 0.6s ease"></div></div>'
      + '<div style="font-size:0.7rem;color:#aaa;margin-top:2px">' + pct + '%</div></div>';
  }
  var html = '<div class="panel-title"><span class="indicator"></span> Formation Efficiency <span class="tag warn" style="margin-left:0.5rem">Research Preview</span></div>';
  html += '<div style="display:flex;gap:1.5rem;flex-wrap:wrap;align-items:flex-start">';
  html += '<div style="flex:1;min-width:180px">';
  html += gauge(seiQ, "SEI Quality");
  html += gauge(lifeIdx, "Lifetime Index");
  html += gauge(robust, "Robustness");
  html += '</div>';
  html += '<div style="flex:1;min-width:200px;font-size:0.78rem;color:#bbb;line-height:1.7">';
  html += '<div>Formation CE: <strong style="color:#fff">' + ceStr + '</strong> (optimal: ' + optimal + ')</div>';
  html += '<div>Status: <strong style="color:' + (gap != null && gap > 3 ? "#ff9f1a" : "#22c55e") + '">' + gapStr + '</strong></div>';
  if (proto.formation_c_rate != null) {
    html += '<div style="margin-top:0.5rem;padding:0.5rem;background:rgba(255,255,255,0.03);border-radius:6px">';
    html += '<div style="font-size:0.68rem;color:#666;margin-bottom:3px">SUGGESTED PROTOCOL</div>';
    html += '<div>Formation C-rate: <strong>' + proto.formation_c_rate + 'C</strong></div>';
    html += '<div>Rest time: <strong>' + proto.rest_time_hours + 'h</strong></div>';
    html += '</div>';
  }
  html += '</div></div>';
  html += '<div class="plot-note" style="margin-top:0.6rem">M14 FormationProtocol is a preview output; validate against formation experiments before using it as a production recipe.</div>';
  container.innerHTML = html;
}

function fmtFeatureValue(v) {
  if (v == null || !Number.isFinite(Number(v))) return "--";
  var n = Number(v);
  if (Math.abs(n) >= 1000) return n.toFixed(0);
  if (Math.abs(n) >= 10) return n.toFixed(2);
  if (Math.abs(n) >= 1) return n.toFixed(4);
  return n.toPrecision(4);
}

function drawDqdv(cv, pts, opts) {
  if (!cv || !pts || pts.length < 2) return;
  opts = opts || {};
  var ctx = cv.getContext("2d");
  var W = cv.width, H = cv.height;
  var p = { t: 28, r: 18, b: 34, l: 52 };
  var pw = W - p.l - p.r, ph = H - p.t - p.b;
  var xs = pts.map(function (x) { return Number(x.voltage); }).filter(Number.isFinite);
  var ys = pts.map(function (x) { return Number(x.dqdv); }).filter(Number.isFinite);
  if (!xs.length || !ys.length) return;
  var xMn = Math.min.apply(null, xs), xMx = Math.max.apply(null, xs);
  var yMn = Math.min.apply(null, ys), yMx = Math.max.apply(null, ys);
  if (Math.abs(yMx - yMn) < 1e-9) { yMx += 1; yMn -= 1; }
  ctx.clearRect(0, 0, W, H);
  drawGrid(ctx, p, pw, ph);
  // dQ/dV line
  ctx.strokeStyle = "#ff1a1a";
  ctx.lineWidth = 2;
  ctx.shadowColor = "#ff1a1a";
  ctx.shadowBlur = 8;
  ctx.beginPath();
  pts.forEach(function (pt, i) {
    var x = p.l + (pt.voltage - xMn) / Math.max(1e-9, xMx - xMn) * pw;
    var y = p.t + ph - (pt.dqdv - yMn) / (yMx - yMn) * ph;
    if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
  });
  ctx.stroke();
  ctx.shadowBlur = 0;
  // d²Q/dV² overlay (second derivative fingerprint)
  if (pts.length > 6) {
    ctx.strokeStyle = "rgba(56,189,248,0.5)";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    var d2 = [];
    for (var k = 1; k < pts.length - 1; k++) {
      var dv1 = pts[k].voltage - pts[k - 1].voltage;
      var dv2 = pts[k + 1].voltage - pts[k].voltage;
      if (Math.abs(dv1) < 1e-9 || Math.abs(dv2) < 1e-9) continue;
      d2.push({ voltage: pts[k].voltage, val: (pts[k + 1].dqdv - 2 * pts[k].dqdv + pts[k - 1].dqdv) / ((dv1 + dv2) / 2) });
    }
    if (d2.length > 2) {
      var d2Vals = d2.map(function (x) { return x.val; });
      var d2Mn = Math.min.apply(null, d2Vals), d2Mx = Math.max.apply(null, d2Vals);
      if (Math.abs(d2Mx - d2Mn) > 1e-9) {
        ctx.beginPath();
        d2.forEach(function (pt, i) {
          var x = p.l + (pt.voltage - xMn) / Math.max(1e-9, xMx - xMn) * pw;
          var y = p.t + ph - (pt.val - d2Mn) / (d2Mx - d2Mn) * ph;
          if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
        });
        ctx.stroke();
      }
    }
    ctx.setLineDash([]);
  }
  // Peak annotations
  var peaks = opts.peaks || [];
  var peakLabels = [
    { range: [2.0, 2.8], label: "Na ordering" },
    { range: [2.8, 3.35], label: "Fe³⁺/²⁺" },
    { range: [3.35, 3.65], label: "Fe⁴⁺/³⁺" },
    { range: [3.65, 4.0], label: "Mn³⁺/⁴⁺" },
    { range: [4.0, 4.6], label: "O²⁻ redox" }
  ];
  peaks.forEach(function (peakV, idx) {
    var x = p.l + (peakV - xMn) / Math.max(1e-9, xMx - xMn) * pw;
    // Find y at peak
    var closestPt = pts.reduce(function (best, pt) {
      return Math.abs(pt.voltage - peakV) < Math.abs(best.voltage - peakV) ? pt : best;
    });
    var y = p.t + ph - (closestPt.dqdv - yMn) / (yMx - yMn) * ph;
    // Draw marker
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#ff1a1a";
    ctx.lineWidth = 1.5;
    ctx.stroke();
    // Label
    var label = "Peak " + (idx + 1);
    peakLabels.forEach(function (pl) {
      if (peakV >= pl.range[0] && peakV < pl.range[1]) label = pl.label;
    });
    ctx.fillStyle = "#ddd";
    ctx.font = "bold 9px JetBrains Mono, monospace";
    var yOff = idx % 2 === 0 ? -14 : -26;
    ctx.fillText(label, x - 15, y + yOff);
    ctx.fillStyle = "#888";
    ctx.font = "8px JetBrains Mono, monospace";
    ctx.fillText(peakV.toFixed(2) + "V", x - 12, y + yOff + 10);
  });
  ctx.fillStyle = "#777";
  ctx.font = "10px JetBrains Mono, monospace";
  ctx.fillText("dQ/dV vs voltage  (dashed: d²Q/dV²)", p.l + 6, 15);
  ctx.fillText(xMn.toFixed(2) + "V", p.l, H - 6);
  ctx.fillText(xMx.toFixed(2) + "V", p.l + pw - 44, H - 6);
}

function renderBYOD(data) {
  window.__kfByod = data;
  window.__kfByodCompare = null;
  window.__kfByodBatch = null;
  var schemaBody = document.getElementById("byod-schema-body");
  var featureBody = document.getElementById("byod-feature-body");
  var warnings = document.getElementById("byod-warnings");
  var schema = data.schema || {};
  var mapping = schema.mapping || {};
  var confidence = schema.confidence || {};
  if (schemaBody) {
    var keys = Object.keys(mapping);
    schemaBody.innerHTML = keys.length ? keys.map(function (k) {
      return "<tr><td>" + escapeHtml(k) + "</td><td>" + escapeHtml(mapping[k]) + "</td><td>" + fmtFeatureValue(confidence[k]) + "</td></tr>";
    }).join("") : '<tr><td colspan="3">No confident mappings found.</td></tr>';
  }
  if (featureBody) {
    var names = data.feature_names || [];
    var feats = data.features || {};
    var avail = data.feature_availability || {};
    featureBody.innerHTML = names.map(function (name) {
      if (!avail[name]) return "";
      return "<tr><td>" + escapeHtml(name) + "</td><td>" + escapeHtml(fmtFeatureValue(feats[name])) + '</td><td><span class="tag ok">present</span></td></tr>';
    }).join("") || '<tr><td colspan="3">No tier-1 features were available.</td></tr>';
  }
  var pred = data.predictions || {};
  var present = (data.feature_mask || []).filter(function (x) { return !!x; }).length;
  setHtml("byod-soh", pred.soh != null ? (pred.soh * 100).toFixed(1) + "%" : "--");
  setHtml("byod-confidence", pred.confidence != null ? (pred.confidence * 100).toFixed(0) + "%" : "--");
  setHtml("byod-cycle80", pred.cycle_80_estimate != null ? String(pred.cycle_80_estimate) : "--");
  setHtml("byod-features", present + "/" + ((data.feature_names || []).length || 27));
  var dqdv = data.dqdv || {};
  drawDqdv(makeCanvas("byod-dqdv-chart"), dqdv.points || [], { peaks: (dqdv.peaks || []).map(Number) });
  setHtml("byod-dqdv-note", dqdv.peaks && dqdv.peaks.length ? "Detected peak voltages: " + dqdv.peaks.map(function (v) { return Number(v).toFixed(3) + "V"; }).join(", ") : "No reliable dQ/dV peaks detected. Upload discharge voltage-capacity traces for this panel.");
  var models = pred.model_outputs || {};
  var modelRows = Object.keys(models).map(function (id) {
    var m = models[id];
    var desc = Object.keys(m).filter(function (k) { return k !== "source" && k !== "checkpoint_source" && k !== "suggested_protocol" && k !== "research_preview"; }).map(function (k) {
      return k + "=" + (typeof m[k] === "number" ? fmtFeatureValue(m[k]) : m[k]);
    }).join(", ");
    var preview = !!m.research_preview || id === "M12_Replenishability" || id === "M14_FormationProtocol";
    var labelGate = id === "M5_BMS_TGN";
    var hasCheckpoint = m.checkpoint_source === "trained_forward" && !preview && !labelGate;
    var tag = (preview || labelGate) ? "warn" : (hasCheckpoint ? "ok" : "ok");
    var label = preview ? "Preview (Not Validated)" : labelGate ? "Label Gate" : (hasCheckpoint ? "trained forward" : (m.source || "derived"));
    var displayDesc = preview ? "Preview only. " + desc : desc;
    return '<div class="switch-row"><div class="switch-label"><span class="name">' + escapeHtml(id) + '</span><span class="desc">' + escapeHtml(displayDesc) + '</span></div><span class="tag ' + tag + '">' + escapeHtml(label) + "</span></div>";
  }).join("");
  setHtml("byod-model-output", modelRows || '<div class="switch-row"><div class="switch-label"><span class="name">No outputs</span><span class="desc">Feature extraction did not produce enough inputs.</span></div><span class="tag warn">Low data</span></div>');
  var chem = models.M13_ChemIdentifier ? models.M13_ChemIdentifier.predicted_family : "unknown";
  var m11 = models.M11_ElectrolyteHealth || {};
  setHtml("byod-decision", "<strong>Upload result:</strong> SOH " + (pred.soh != null ? (pred.soh * 100).toFixed(1) + "%" : "--") + ", chemistry " + escapeHtml(chem) + ", sodium plating risk " + fmtFeatureValue(m11.checkpoint_sodium_plating_probability != null ? m11.checkpoint_sodium_plating_probability : m11.sodium_plating_probability) + ", inference " + escapeHtml(pred.inference_mode || "rules_only") + ". <strong>Use:</strong> export the arrays and validate low-confidence or research-preview outputs before decisions.");
  var uploadConf = computeUploadConfidence(data);
  renderConfidence("byod-confidence-detail", uploadConf.confidence, uploadConf.detail);
  if (warnings) {
    var list = data.warnings && data.warnings.length ? data.warnings : ["No warnings from parser."];
    warnings.innerHTML = list.map(function (w) { return '<div class="' + (w.indexOf("low") >= 0 || w.indexOf("not") >= 0 ? "warn" : "info") + '">' + escapeHtml(w) + "</div>"; }).join("");
  }
  renderFormationScore(data);
  recordRun("upload", {
    summary: (data.filename || "upload") + ", SOH " + (pred.soh != null ? (pred.soh * 100).toFixed(1) + "%" : "--") + ", confidence " + (pred.confidence != null ? (pred.confidence * 100).toFixed(0) + "%" : "--"),
    key_metric: present + " features",
    confidence: uploadConf.confidence
  });
}

function analyzeBYOD() {
  var input = document.getElementById("byod-file");
  if (!input || !input.files || !input.files.length) {
    showToast("Choose a cycler CSV/TXT/XLSX file first.", "warn");
    return;
  }
  var file = input.files[0];
  var status = document.getElementById("byod-status");
  if (status) status.textContent = "Uploading and extracting features from " + file.name + "...";
  var form = new FormData();
  form.append("file", file);
  fetch("/api/byod/analyze", { method: "POST", body: form })
    .then(function (res) {
      if (!res.ok) {
        return res.json().catch(function () { return {}; }).then(function (body) {
          throw new Error(body.detail || ("Upload analysis failed with HTTP " + res.status));
        });
      }
      return res.json();
    })
    .then(function (data) {
      renderBYOD(data);
      if (status) status.textContent = "Analyzed " + data.rows_read + " rows from " + data.filename + ". Session " + data.session_id + " expires automatically.";
      showToast("Upload analyzed. Features and dQ/dV are ready.", "ok");
    })
    .catch(function (err) {
      if (status) status.textContent = err.message;
      showToast(err.message, "error");
      console.warn(err);
    });
}

function compareBYOD() {
  var a = document.getElementById("byod-file");
  var b = document.getElementById("byod-file-b");
  if (!a || !b || !a.files.length || !b.files.length) {
    showToast("Choose two cycler files for A/B comparison.", "warn");
    return;
  }
  var status = document.getElementById("byod-status");
  if (status) status.textContent = "Comparing " + a.files[0].name + " against " + b.files[0].name + "...";
  var form = new FormData();
  form.append("file_a", a.files[0]);
  form.append("file_b", b.files[0]);
  fetch("/api/byod/compare", { method: "POST", body: form })
    .then(function (res) {
      if (!res.ok) {
        return res.json().catch(function () { return {}; }).then(function (body) {
          throw new Error(body.detail || ("Compare failed with HTTP " + res.status));
        });
      }
      return res.json();
    })
    .then(function (data) {
      window.__kfByodCompare = data;
      window.__kfByod = null;
      window.__kfByodBatch = null;
      setHtml("byod-soh", "--");
      setHtml("byod-confidence", "--");
      setHtml("byod-cycle80", "--");
      setHtml("byod-features", "--");
      var schemaBody = document.getElementById("byod-schema-body");
      var featureBody = document.getElementById("byod-feature-body");
      if (schemaBody) schemaBody.innerHTML = '<tr><td colspan="3">Schema table is available for single-file analysis.</td></tr>';
      if (featureBody) featureBody.innerHTML = '<tr><td colspan="3">Feature table is available for single-file analysis.</td></tr>';
      setHtml("byod-model-output", '<div class="switch-row"><div class="switch-label"><span class="name">Compare mode</span><span class="desc">Model card details are shown for single-file analysis only.</span></div><span class="tag">A/B</span></div>');
      setHtml("byod-dqdv-note", "dQ/dV fingerprint is displayed for single-file analysis.");
      renderConfidence("byod-confidence-detail", null, "");
      var c = data.comparison || {};
      var delta = c.delta || {};
      setHtml("byod-decision", "<strong>A/B result:</strong> " + escapeHtml(c.decision || "no decision") + ". SOH delta " + fmtFeatureValue(delta.soh) + ", fade delta " + fmtFeatureValue(delta.fade_fraction_per_cycle) + ". <strong>Use:</strong> compare only matched protocols.");
      recordRun("upload_compare", {
        summary: c.decision || "Compared matched uploads",
        key_metric: "SOH delta " + fmtFeatureValue(delta.soh),
        confidence: null
      });
      if (status) status.textContent = "Compared two uploads. Sessions: " + (data.file_a_session_id || "--") + " / " + (data.file_b_session_id || "--");
      showToast("A/B comparison ready.", "ok");
    })
    .catch(function (err) {
      if (status) status.textContent = err.message;
      showToast(err.message, "error");
      console.warn(err);
    });
}

function analyzeBYODBatch() {
  var input = document.getElementById("byod-batch-file");
  if (!input || !input.files || !input.files.length) {
    showToast("Choose a ZIP of cycler files first.", "warn");
    return;
  }
  var status = document.getElementById("byod-status");
  if (status) status.textContent = "Analyzing batch ZIP " + input.files[0].name + "...";
  var form = new FormData();
  form.append("file", input.files[0]);
  fetch("/api/byod/batch", { method: "POST", body: form })
    .then(function (res) {
      if (!res.ok) {
        return res.json().catch(function () { return {}; }).then(function (body) {
          throw new Error(body.detail || ("Batch analysis failed with HTTP " + res.status));
        });
      }
      return res.json();
    })
    .then(function (data) {
      window.__kfByodBatch = data;
      window.__kfByod = null;
      window.__kfByodCompare = null;
      setHtml("byod-soh", "--");
      setHtml("byod-confidence", "--");
      setHtml("byod-cycle80", "--");
      setHtml("byod-features", "--");
      var schemaBody = document.getElementById("byod-schema-body");
      var featureBody = document.getElementById("byod-feature-body");
      if (schemaBody) schemaBody.innerHTML = '<tr><td colspan="3">Schema table is available for single-file analysis.</td></tr>';
      if (featureBody) featureBody.innerHTML = '<tr><td colspan="3">Feature table is available for single-file analysis.</td></tr>';
      setHtml("byod-model-output", '<div class="switch-row"><div class="switch-label"><span class="name">Batch mode</span><span class="desc">Per-file model cards are available inside exported batch summaries.</span></div><span class="tag">ZIP</span></div>');
      setHtml("byod-dqdv-note", "dQ/dV fingerprint is displayed for single-file analysis.");
      renderConfidence("byod-confidence-detail", null, "");
      var stats = data.stats || {};
      setHtml("byod-decision", "<strong>Batch result:</strong> " + escapeHtml(data.decision || "batch analyzed") + ". Cells " + escapeHtml(String(data.files_analyzed || 0)) + ", SOH mean " + fmtFeatureValue(stats.soh_mean) + ", SOH std " + fmtFeatureValue(stats.soh_std) + ", outliers " + escapeHtml(String((data.outliers || []).length)) + ".");
      recordRun("upload_batch", {
        summary: (data.files_analyzed || 0) + " files, outliers " + ((data.outliers || []).length),
        key_metric: "SOH mean " + fmtFeatureValue(stats.soh_mean),
        confidence: null
      });
      var warnings = document.getElementById("byod-warnings");
      if (warnings) {
        warnings.innerHTML = (data.outliers || []).slice(0, 12).map(function (o) {
          return '<div class="warn">' + escapeHtml(o.filename || "cell") + ": " + escapeHtml((o.reasons || []).join(", ")) + "</div>";
        }).join("") || '<div class="info">No batch outliers flagged.</div>';
      }
      if (status) status.textContent = "Batch analyzed. Session " + (data.batch_session_id || "--") + " expires automatically.";
      showToast("Batch report ready.", "ok");
    })
    .catch(function (err) {
      if (status) status.textContent = err.message;
      showToast(err.message, "error");
      console.warn(err);
    });
}

function exportBYODCSV() {
  var data = window.__kfByod;
  if (window.__kfByodBatch && !data) {
    downloadCSV((window.__kfByodBatch.summaries || []).map(function (r) {
      return { kind: "batch_cell", session_id: r.session_id, filename: r.filename, soh: r.soh, confidence: r.confidence, cycle_80_estimate: r.cycle_80_estimate, fade_fraction_per_cycle: r.fade_fraction_per_cycle, anomaly_score: r.anomaly_score };
    }), "kineticsforge_byod_batch.csv");
    return;
  }
  if (window.__kfByodCompare && !data) {
    var comp = window.__kfByodCompare.comparison || {};
    downloadCSV((comp.top_feature_deltas || []).map(function (r) {
      return { kind: "compare_delta", feature: r.feature, a: r.a, b: r.b, delta: r.delta, relative_delta: r.relative_delta };
    }), "kineticsforge_byod_compare.csv");
    return;
  }
  if (!data) {
    showToast("Run an upload analysis before exporting.", "warn");
    return;
  }
  var rows = [];
  (data.cycle_summary || []).forEach(function (r) {
    rows.push({ kind: "cycle", name: r.cycle, discharge_capacity_ah: r.discharge_capacity_ah, charge_capacity_ah: r.charge_capacity_ah, ce: r.ce });
  });
  Object.keys(data.features || {}).forEach(function (k) {
    rows.push({ kind: "feature", name: k, discharge_capacity_ah: data.features[k], charge_capacity_ah: "", ce: "" });
  });
  downloadCSV(rows, "kineticsforge_byod_analysis.csv");
}

function exportBYODJSON() {
  var batch = window.__kfByodBatch;
  var compare = window.__kfByodCompare;
  var data = window.__kfByod;
  var sid = data && data.session_id ? data.session_id : (batch && batch.batch_session_id ? batch.batch_session_id : null);
  if (sid) {
    fetch("/api/byod/session/" + encodeURIComponent(sid) + "/export-json")
      .then(function (res) {
        if (!res.ok) throw new Error("JSON export failed with HTTP " + res.status);
        return res.json();
      })
      .then(function (payload) {
        downloadJSON(payload, batch ? "kineticsforge_byod_batch.json" : "kineticsforge_byod_session.json");
      })
      .catch(function (err) {
        showToast(err.message, "error");
        console.warn(err);
      });
    return;
  }
  if (compare) {
    downloadJSON(compare, "kineticsforge_byod_compare.json");
    return;
  }
  showToast("Run an upload analysis before exporting.", "warn");
}

function testMaterialInDiagnostics() {
  var mat = window.__kfMaterials || {};
  var norm = mat.normalized || normalizeMaterialComposition(materialSelectionFromControls(false));
  var dopants = norm.dopants || norm;
  var dopFrac = Number(dopants.Al || 0) + Number(dopants.Ti || 0);
  var comp = {
    Na: Number(norm.Na != null ? norm.Na : parseFloat(document.getElementById("na-slider").value)),
    Mn: Number(norm.Mn != null ? norm.Mn : parseFloat(document.getElementById("mn-slider").value)),
    Fe: Number(norm.Fe != null ? norm.Fe : parseFloat(document.getElementById("fe-slider").value)),
    dopant_frac: dopFrac,
    dopant_type: Number(dopants.Ti || 0) > Number(dopants.Al || 0) ? "Ti" : (dopFrac > 0 ? "Al" : "generic")
  };
  setDiagnosticComposition(comp);
  navigate("diagnostics");
  runDiagnostics();
}

function loadDiagnosticsInMaterials() {
  var comp = diagnosticComposition();
  var dop = clamp(Number(comp.dopant_frac) || 0, 0, 0.18);
  var mn = clamp(Number(comp.Mn), 0.1, 0.9);
  var fe = clamp(Number(comp.Fe), 0.1, 0.9);
  var naEl = document.getElementById("na-slider");
  var mnEl = document.getElementById("mn-slider");
  var feEl = document.getElementById("fe-slider");
  var alEl = document.getElementById("sw-al");
  var tiEl = document.getElementById("sw-ti");
  if (naEl) naEl.value = clamp(Number(comp.Na), 0.6, 1.2).toFixed(2);
  if (mnEl) mnEl.value = mn.toFixed(2);
  if (feEl) feEl.value = fe.toFixed(2);
  if (alEl) alEl.checked = dop > 0 && comp.dopant_type !== "Ti";
  if (tiEl) tiEl.checked = dop > 0 && comp.dopant_type === "Ti";
  updateMat();
  showToast("Diagnostics composition loaded into Materials with dopant species preserved.", "ok");
}

function pushDiagnosticsToBms() {
  var diag = window.__kfDiag || simulateDegradation();
  var eol = Number(diag.cap && diag.cap.length ? diag.cap[diag.cap.length - 1] : 0.86);
  var fade = clamp(1 - eol, 0.02, 0.55);
  var loadEl = document.getElementById("bms-load");
  var gateEl = document.getElementById("bms-rct-gate");
  var threshEl = document.getElementById("bms-risk-thresh");
  var faultEl = document.getElementById("sw-fault");
  if (loadEl) loadEl.value = clamp(0.95 + fade * 2.1, 0.8, 2.2).toFixed(2);
  if (gateEl) gateEl.value = clamp(0.039 + fade * 0.030, 0.032, 0.090).toFixed(3);
  if (threshEl) threshEl.value = clamp(0.44 - fade * 0.18, 0.20, 0.55).toFixed(2);
  if (faultEl) faultEl.checked = true;
  bmsPresetFromDiagnostics = {
    cell: 0,
    r0Multiplier: clamp(1 + fade * 1.8, 1.0, 2.2),
    seiOffset: clamp(fade * 0.025, 0, 0.08)
  };
  updateBMS();
  navigate("bms");
  showToast("Diagnostics stress profile pushed to BMS (cell C0 seeded as degraded baseline).", "ok");
}

function sendBmsToDecisionQueue() {
  navigate("decisions");
  renderDecisionConsole();
  showToast("Decision queue refreshed with current BMS evidence.", "ok");
}

function drawDegradationSurface(cv, out) {
  if (!cv) return;
  var ctx = cv.getContext("2d"), W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  var x0 = 48, y0 = 28, plotW = W - 88, plotH = H - 62;
  var n = Math.min(96, out.cap.length);
  var step = Math.max(1, Math.floor(out.cap.length / n));
  var colors = { "P2-O2": "#ff1a1a", "JT": "#d946ef", "SEI": "#ff9f1a", "Rate": "#22c55e", "Residual": "#38bdf8" };
  ctx.font = "10px JetBrains Mono, monospace";
  ctx.fillStyle = "#858585";
  ctx.fillText("cycle ->   y=SOC window   color=dominant loss   brightness=stress", 12, 15);
  ctx.strokeStyle = "rgba(255,26,26,0.12)";
  ctx.strokeRect(x0, y0, plotW, plotH);
  for (var gy = 0; gy <= 4; gy++) {
    var yy = y0 + gy * plotH / 4;
    ctx.strokeStyle = "rgba(255,26,26,0.06)";
    ctx.beginPath();
    ctx.moveTo(x0, yy);
    ctx.lineTo(x0 + plotW, yy);
    ctx.stroke();
  }
  for (var i = 0; i < n; i++) {
    var idx = Math.min(out.cap.length - 1, i * step);
    var soc = out.soc[idx] || 0.8;
    var intensity = clamp((out.p2Rate[idx] || 0) * 150 + (out.jtRaw[idx] || 0) * 0.16 + (out.desolv[idx] || 0) * 0.10, 0.04, 1);
    var x = x0 + (i / Math.max(1, n - 1)) * plotW;
    var y = y0 + plotH - ((soc - 0.55) / 0.43) * plotH;
    var band = 4 + intensity * 13;
    var w = Math.max(3, plotW / n * 0.9);
    ctx.fillStyle = colors[out.dominant[idx]] || "#ff1a1a";
    ctx.globalAlpha = 0.22 + intensity * 0.72;
    ctx.fillRect(x - w / 2, y - band / 2, w, band);
    ctx.globalAlpha = 0.14 + intensity * 0.22;
    ctx.fillRect(x - w / 2, y - band * 1.7, w, band * 3.4);
  }
  ctx.globalAlpha = 1;
  ctx.strokeStyle = "rgba(232,232,232,0.55)";
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  for (var j = 0; j < n; j++) {
    var idc = Math.min(out.cap.length - 1, j * step);
    var cx = x0 + (j / Math.max(1, n - 1)) * plotW;
    var cy = y0 + plotH - clamp((out.cap[idc] - 0.65) / 0.37, 0, 1) * plotH;
    if (j === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
  }
  ctx.stroke();
  ctx.fillStyle = "#606060";
  ctx.fillText("SOC high", 2, y0 + 8);
  ctx.fillText("SOC low", 4, y0 + plotH);
  ctx.fillText("capacity trace", W - 118, y0 + plotH + 18);
  var lx = x0, ly = y0 + plotH + 16;
  [["SEI", "#ff9f1a"], ["P2-O2", "#ff1a1a"], ["JT", "#d946ef"], ["Rate", "#22c55e"], ["Residual", "#38bdf8"]].forEach(function (item, k) {
    ctx.fillStyle = item[1];
    ctx.fillRect(lx + k * 62, ly - 8, 10, 4);
    ctx.fillStyle = "#777";
    ctx.fillText(item[0], lx + 14 + k * 62, ly - 4);
  });
}

// BMS thermal graph simulation
function bmsTopologyInfo(n, mode) {
  mode = mode || (document.getElementById("bms-topology") ? document.getElementById("bms-topology").value : "4s2p");
  var parallel = mode === "2s4p" ? 4 : mode === "4s2p" ? 2 : 1;
  if (mode === "grid") parallel = 1;
  parallel = Math.max(1, Math.min(parallel, n));
  var series = Math.max(1, Math.ceil(n / parallel));
  return {
    mode: mode,
    parallel: parallel,
    series: series,
    label: mode === "series" ? n + "S1P series string" : mode === "2s4p" ? "2S4P current-sharing module" : mode === "4s2p" ? "4S2P current-sharing module" : "thermal grid only"
  };
}

function topologyPairSummary(n, mode) {
  if (mode === "4s2p" || mode === "2s4p") {
    var info = bmsTopologyInfo(n, mode);
    var groups = [];
    for (var col = 0; col < info.series; col++) {
      var members = [];
      for (var row = 0; row < info.parallel; row++) {
        var idx = col + row * info.series;
        if (idx < n) members.push("C" + idx);
      }
      if (members.length > 1) groups.push(members);
    }
    if (groups.length) {
      var rendered = groups.slice(0, 4).map(function (g) { return g.join("||"); }).join(", ");
      return "Parallel groups: " + rendered + (groups.length > 4 ? ", ..." : "") + ".";
    }
  }
  if (mode === "series") return "No parallel sharing; all cells carry the same string current.";
  return "Grid mode links immediate geometric neighbors only.";
}

function buildTopology(n, mode) {
  var info = bmsTopologyInfo(n, mode);
  var cols = info.mode === "grid" ? (n <= 8 ? n : Math.ceil(Math.sqrt(n * 1.4))) : info.series;
  var rows = info.mode === "grid" ? Math.ceil(n / cols) : info.parallel;
  var pos = [];
  var edges = [];
  var nbr = Array.from({ length: n }, function () { return []; });
  for (var i = 0; i < n; i++) pos.push({ x: i % cols, y: Math.floor(i / cols), group: i % cols });
  for (var a = 0; a < n; a++) {
    for (var b = a + 1; b < n; b++) {
      var dx = Math.abs(pos[a].x - pos[b].x), dy = Math.abs(pos[a].y - pos[b].y);
      if ((dx === 1 && dy === 0) || (dx === 0 && dy === 1)) {
        edges.push([a, b]);
        nbr[a].push(b);
        nbr[b].push(a);
      }
    }
  }
  return { cols: cols, rows: rows, pos: pos, edges: edges, neighbors: nbr, info: info };
}

function bmsKnobs(useAsym) {
  var baseThreshold = num("bms-risk-thresh", 0.42);
  var lossRatio = Math.max(1, num("bms-loss-ratio", 6.0));
  var asymDrop = clamp(Math.log(lossRatio) / Math.log(2) * 0.035, 0, 0.18);
  return {
    cth: Math.max(10, num("bms-cth", 95)),
    kedge: Math.max(0, num("bms-kedge", 0.18)),
    cooling: Math.max(0, num("bms-cool", 0.045)),
    load: Math.max(0.05, num("bms-load", 1.0)),
    rctGate: Math.max(0.001, num("bms-rct-gate", 0.043)),
    lossRatio: lossRatio,
    threshold: useAsym ? clamp(baseThreshold - asymDrop, 0.12, 0.95) : Math.max(0.48, baseThreshold),
    ambient: num("bms-ambient", 45) + 273.15,
    topologyMode: document.getElementById("bms-topology") ? document.getElementById("bms-topology").value : "4s2p"
  };
}

function bmsScenarioKey(n, duration, fault, eis, asym, ambient, knobs) {
  knobs = knobs || bmsKnobs(asym);
  var preset = bmsPresetFromDiagnostics || {};
  return [
    n,
    duration,
    fault ? 1 : 0,
    eis ? 1 : 0,
    asym ? 1 : 0,
    Number(ambient).toFixed(2),
    Number(knobs.cth).toFixed(2),
    Number(knobs.kedge).toFixed(4),
    Number(knobs.cooling).toFixed(4),
    Number(knobs.load).toFixed(4),
    Number(knobs.rctGate).toFixed(4),
    Number(knobs.threshold).toFixed(4),
    Number(knobs.lossRatio).toFixed(2),
    knobs.topologyMode || "4s2p",
    Number(preset.cell != null ? preset.cell : -1),
    Number(preset.r0Multiplier || 1).toFixed(3),
    Number(preset.seiOffset || 0).toFixed(4)
  ].join("|");
}

function bmsCurrentShare(topology, cells, idx) {
  var info = topology.info || {};
  if (!info.parallel || info.parallel <= 1 || info.mode === "grid") return 1.0;
  var group = topology.pos[idx].group;
  var peers = [];
  for (var i = 0; i < topology.pos.length; i++) {
    if (topology.pos[i].group === group) peers.push(i);
  }
  var inv = peers.map(function (p) {
    var c = cells[p];
    return 1 / Math.max(0.004, c.r0 + 0.18 * c.sei);
  });
  var invSum = inv.reduce(function (a, b) { return a + b; }, 0);
  var own = inv[peers.indexOf(idx)] || 0;
  return clamp((own / Math.max(invSum, 1e-9)) * peers.length, 0.35, 2.4);
}

function simulateBmsPhysics(n, duration, injectFault, useEis, useAsym, seed, preset) {
  var cfg = bmsKnobs(useAsym);
  var topology = buildTopology(n, cfg.topologyMode);
  var steps = clamp(Math.round(duration), 60, 240);
  var dt = duration / steps;
  var ambient = cfg.ambient;
  // Use seeded PRNG for reproducible results
  var rng = mulberry32(seed != null ? seed : 42);
  var cells = [];
  var faultCell = injectFault ? Math.floor(rng() * n) : -1;
  if (preset && Number.isFinite(preset.cell)) faultCell = clamp(Math.round(preset.cell), 0, n - 1);
  for (var i = 0; i < n; i++) {
    var r0Mul = (preset && i === faultCell) ? clamp(Number(preset.r0Multiplier) || 1, 1, 3) : 1;
    var seiOffset = (preset && i === faultCell) ? clamp(Number(preset.seiOffset) || 0, 0, 0.10) : 0;
    cells.push({
      T: ambient + seededGaussian(rng) * 0.25,
      r0: 0.033 * (1 + seededGaussian(rng) * 0.025) * r0Mul,
      sei: 0.010 + rng() * 0.002 + seiOffset,
      risk: 0,
      rawHist: []
    });
  }
  var frames = [];
  var alerts = [];
  var threshold = clamp(cfg.threshold, 0.05, 0.95);
  var failureTime = duration * 0.84;
  for (var s = 0; s <= steps; s++) {
    var t = s * dt;
    var prevT = cells.map(function (c) { return c.T; });
    var prevRisk = cells.map(function (c) { return c.risk || 0; });
    var raw = [];
    for (var c = 0; c < n; c++) {
      var cell = cells[c];
      var isFault = c === faultCell;
      var faultDrive = isFault ? Math.pow(sigmoid((t - duration * 0.46) / Math.max(3, duration * 0.07)), 2) : 0;
      var arrh = Math.exp(-0.28 / (8.617e-5) * (1 / cell.T - 1 / ambient));
      cell.sei += dt * (1.0e-6 * arrh + faultDrive * 7.0e-5);
      var rInt = cell.r0 + 0.18 * cell.sei + faultDrive * 0.020;
      var currentShare = bmsCurrentShare(topology, cells, c);
      var qOhm = cfg.load * (34 * rInt * currentShare * currentShare + faultDrive * 14.0);
      var coupling = 0;
      topology.neighbors[c].forEach(function (j) { coupling += cfg.kedge * (prevT[j] - prevT[c]); });
      var dTdt = (qOhm + coupling - cfg.cooling * (prevT[c] - ambient)) / cfg.cth;
      cell.T = clamp(prevT[c] + dt * dTdt, 290, 390);
      var rSei = 0.006 + 0.080 * cell.sei + faultDrive * 0.010;
      var rCt = 0.028 * Math.exp(1800 * (1 / cell.T - 1 / ambient)) * (1 + 3.5 * cell.sei + faultDrive * 1.6);
      cell.dTdt = dTdt;
      cell.rSei = rSei;
      cell.rCt = rCt;
      var tempScore = sigmoid((cell.T - 333.15) / 4.5);
      var slopeScore = sigmoid((dTdt * 60 - 1.2) / 0.7);
      var eisScore = useEis ? sigmoid((rCt + rSei - cfg.rctGate) / 0.009) : 0.25 * tempScore;
      // Use prevT for neighbor score to avoid cell-ordering artifacts
      var neighborTemp = 0;
      var neighborRiskPrev = 0;
      topology.neighbors[c].forEach(function (j) { neighborTemp += sigmoid((prevT[j] - 273.15 - 60) / 5.0); });
      topology.neighbors[c].forEach(function (j) { neighborRiskPrev += prevRisk[j] || 0; });
      neighborTemp = topology.neighbors[c].length ? neighborTemp / topology.neighbors[c].length : 0;
      neighborRiskPrev = topology.neighbors[c].length ? neighborRiskPrev / topology.neighbors[c].length : 0;
      raw[c] = clamp(0.34 * tempScore + 0.21 * slopeScore + 0.27 * eisScore + 0.10 * neighborTemp + 0.08 * neighborRiskPrev, 0, 1);
      cell.rawHist.push(raw[c]);
      var h = cell.rawHist;
      var lookback = 0.40 * _histBack(h, 30) + 0.28 * _histBack(h, 60) + 0.20 * _histBack(h, 120) + 0.12 * _histBack(h, 240);
      cell.risk = clamp(0.78 * cell.risk + 0.22 * lookback, 0, 1);
    }
    var risks = cells.map(function (c) { return c.risk; });
    var temps = cells.map(function (c) { return c.T - 273.15; });
    var slopes = cells.map(function (c) { return c.dTdt || 0; });
    var rcts = cells.map(function (c) { return c.rCt || 0; });
    var rseis = cells.map(function (c) { return c.rSei || 0; });
    var maxRisk = Math.max.apply(null, risks);
    var maxCell = risks.indexOf(maxRisk);
    if (s % Math.max(4, Math.floor(steps / 16)) === 0 || maxRisk > threshold) {
      var line = maxRisk > threshold
        ? { kind: "warn", text: "t=" + Math.round(t) + "s ALERT C" + maxCell + " risk=" + maxRisk.toFixed(3) + " lead=" + Math.max(0, Math.round(failureTime - t)) + "s" }
        : { kind: "info", text: "t=" + Math.round(t) + "s nominal max=" + maxRisk.toFixed(3) + " Tmax=" + Math.max.apply(null, temps).toFixed(1) + " C" };
      if (!alerts.length || alerts[alerts.length - 1].text !== line.text) alerts.push(line);
    }
    frames.push({ t: t, risks: risks.slice(), temps: temps.slice(), slopes: slopes.slice(), rcts: rcts.slice(), rseis: rseis.slice(), maxRisk: maxRisk, maxCell: maxCell });
  }
  var finalFrame = frames[frames.length - 1] || {};
  return { topology: topology, frames: frames, alerts: alerts, faultCell: faultCell, threshold: threshold, lossRatio: cfg.lossRatio, seed: seed != null ? seed : 42, maxRisk: finalFrame.maxRisk, maxCell: finalFrame.maxCell };
}

// Lookback window average helper (extracted from inner function to avoid closure-in-loop)
function _histBack(h, w) {
  var from = Math.max(0, h.length - w);
  var sum = 0;
  for (var k = from; k < h.length; k++) sum += h[k];
  return sum / Math.max(1, h.length - from);
}

function drawBmsThermal(cv, sim, frame) {
  if (!cv || !sim || !frame) return;
  var ctx = cv.getContext("2d");
  var W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  var top = sim.topology;
  var pad = 24;
  var cell = Math.min((W - pad * 2) / top.cols, (H - pad * 2) / top.rows) * 0.72;
  var x0 = (W - (top.cols - 1) * cell * 1.45 - cell) / 2;
  var y0 = (H - (top.rows - 1) * cell * 1.45 - cell) / 2 + 6;
  function xy(i) {
    return { x: x0 + top.pos[i].x * cell * 1.45, y: y0 + top.pos[i].y * cell * 1.45 };
  }
  top.edges.forEach(function (e) {
    var a = xy(e[0]), b = xy(e[1]);
    var grad = Math.abs(frame.temps[e[0]] - frame.temps[e[1]]);
    ctx.strokeStyle = "rgba(255,26,26," + clamp(0.12 + grad / 30, 0.12, 0.65) + ")";
    ctx.lineWidth = 1 + clamp(grad / 8, 0, 2);
    ctx.beginPath();
    ctx.moveTo(a.x + cell / 2, a.y + cell / 2);
    ctx.lineTo(b.x + cell / 2, b.y + cell / 2);
    ctx.stroke();
  });
  frame.temps.forEach(function (temp, i) {
    var p = xy(i);
    var hot = clamp((temp - 35) / 35, 0, 1);
    var risk = frame.risks[i];
    var z = 7 + hot * 24 + risk * 20;
    var main = "rgba(" + Math.round(80 + 175 * hot) + "," + Math.round(20 + 50 * (1 - hot)) + ",0," + (0.25 + 0.65 * Math.max(hot, risk)) + ")";
    ctx.strokeStyle = i === sim.faultCell ? "#ffb020" : "rgba(255,26,26," + (0.2 + risk * 0.7) + ")";
    ctx.lineWidth = i === sim.faultCell ? 2 : 1;
    ctx.shadowColor = "#ff1a1a";
    ctx.shadowBlur = risk > sim.threshold ? 12 : 0;
    ctx.fillStyle = "rgba(60,8,0,0.55)";
    ctx.beginPath();
    ctx.moveTo(p.x, p.y + cell);
    ctx.lineTo(p.x + cell, p.y + cell);
    ctx.lineTo(p.x + cell, p.y + cell - z);
    ctx.lineTo(p.x, p.y + cell - z);
    ctx.closePath();
    ctx.fill();
    ctx.fillStyle = main;
    ctx.beginPath();
    ctx.moveTo(p.x, p.y + cell - z);
    ctx.lineTo(p.x + cell, p.y + cell - z);
    ctx.lineTo(p.x + cell * 0.82, p.y + cell - z - 10);
    ctx.lineTo(p.x + cell * 0.18, p.y + cell - z - 10);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.shadowBlur = 0;
    ctx.fillStyle = "#f5f5f5";
    ctx.font = "10px JetBrains Mono, monospace";
    ctx.fillText("C" + i, p.x + 6, p.y + cell - z - 13);
    ctx.fillStyle = "#d0d0d0";
    ctx.fillText(temp.toFixed(1) + "C", p.x + 6, p.y + cell - 8);
  });
  ctx.fillStyle = "#777";
  ctx.font = "10px JetBrains Mono, monospace";
  ctx.fillText("Thermal coupling: Cth dT/dt = q + sum(kij(Tj-Ti)) - h(T-Ta)", 12, 14);
}

function applyBmsFormatDefaults() {
  var fmtEl = document.getElementById("bms-format");
  var cthEl = document.getElementById("bms-cth");
  var kEl = document.getElementById("bms-kedge");
  if (!fmtEl || !cthEl) return;
  var fmt = fmtEl.value;
  if (fmt === "18650") {
    cthEl.value = "65";
    if (kEl) kEl.value = "0.10";
  } else if (fmt === "prismatic") {
    cthEl.value = "1200";
    if (kEl) kEl.value = "0.36";
  } else {
    cthEl.value = "95";
    if (kEl) kEl.value = "0.18";
  }
  updateBMS();
}

function updateBMS() {
  document.getElementById("pack-val").textContent = document.getElementById("pack-slider").value + " cells";
  document.getElementById("dur-val").textContent = document.getElementById("dur-slider").value + "s";
  var amb = document.getElementById("bms-ambient-val");
  if (amb) amb.textContent = num("bms-ambient", 45).toFixed(0) + " C";
  var topoNote = document.getElementById("bms-topology-note");
  if (topoNote) {
    var n = parseInt(document.getElementById("pack-slider").value, 10) || 8;
    var info = bmsTopologyInfo(n);
    var presetNote = bmsPresetFromDiagnostics ? " Diagnostics preset active on C" + bmsPresetFromDiagnostics.cell + "." : "";
    topoNote.textContent = info.label + ": " + topologyPairSummary(n, info.mode) + " Thermal neighbors use previous-timestep heat/risk (t-dt) to avoid algebraic circularity. Asymmetric gate uses FN/FP cost " + Math.max(1, num("bms-loss-ratio", 6.0)).toFixed(1) + "x." + presetNote;
  }
}

function runBMS() {
  var runId = ++activeBmsRun;
  var n = parseInt(document.getElementById("pack-slider").value, 10);
  var dur = parseInt(document.getElementById("dur-slider").value, 10);
  var fault = document.getElementById("sw-fault").checked;
  var eis = document.getElementById("sw-eis").checked;
  var asym = document.getElementById("sw-asym").checked;
  var seedEl = document.getElementById("bms-seed");
  var seed = seedEl ? parseInt(seedEl.value, 10) : 42;
  if (!Number.isFinite(seed)) seed = 42;
  var sim = simulateBmsPhysics(n, dur, fault, eis, asym, seed, bmsPresetFromDiagnostics);
  var ambientC = num("bms-ambient", 45);
  var scenarioKey = bmsScenarioKey(n, dur, fault, eis, asym, ambientC);
  window.__kfBms = {
    n: n,
    duration: dur,
    threshold: sim.threshold,
    faultCell: sim.faultCell,
    frames: sim.frames,
    alerts: sim.alerts,
    ambient_C: ambientC,
    seed: seed,
    maxRisk: sim.maxRisk,
    topology: sim.topology.info,
    lossRatio: sim.lossRatio,
    scenarioKey: scenarioKey,
    preset: bmsPresetFromDiagnostics
  };
  var sweepNote = document.getElementById("bms-sweep-note");
  if (sweepNote) {
    if (window.__kfBmsSweep && window.__kfBmsSweep.scenarioKey === scenarioKey) {
      sweepNote.textContent = "Sweep " + window.__kfBmsSweep.count + " seeds: alert rate " + Math.round(window.__kfBmsSweep.alertRate * 100) + "%, mean max-risk " + window.__kfBmsSweep.meanRisk.toFixed(3) + " +/- " + window.__kfBmsSweep.stdRisk.toFixed(3) + ", mean detect " + (window.__kfBmsSweep.meanDetectionTime == null ? "--" : window.__kfBmsSweep.meanDetectionTime.toFixed(0) + "s") + ", fault-hit " + (window.__kfBmsSweep.faultHitRate == null ? "--" : Math.round(window.__kfBmsSweep.faultHitRate * 100) + "%") + ".";
    } else {
      sweepNote.textContent = "Run a seed sweep to quantify how sensitive this setup is to stochastic initialization.";
    }
  }
  var grid = document.getElementById("bms-grid");
  grid.innerHTML = "";
  grid.style.gridTemplateColumns = "repeat(" + Math.min(n, 8) + ", minmax(54px, 1fr))";
  for (var i = 0; i < n; i++) {
    var d = document.createElement("div");
    d.className = "cell-tile";
    d.dataset.cell = "C" + i;
    d.dataset.fault = i === sim.faultCell ? "1" : "0";
    d.innerHTML = '<div class="cell-name">C' + i + '</div><div class="cell-risk">0.00</div><div class="cell-temp">-- C</div>';
    grid.appendChild(d);
  }
  var log = document.getElementById("bms-log");
  var presetTxt = bmsPresetFromDiagnostics ? " diagnostics-preset=C" + bmsPresetFromDiagnostics.cell + " r0x" + Number(bmsPresetFromDiagnostics.r0Multiplier || 1).toFixed(2) : "";
  log.innerHTML = '<div class="cmd">$ bms --thermal-ode --cells=' + n + " --topology=" + escapeHtml(sim.topology.info.label) + " --fault=" + (sim.faultCell >= 0 ? "C" + sim.faultCell : "none") + ' --seed=' + seed + '</div><div class="info">Pack graph built with ' + sim.topology.edges.length + " thermal edges. Ambient=" + ambientC.toFixed(0) + " C EIS=" + (eis ? "on" : "off") + " threshold=" + sim.threshold.toFixed(2) + " FN/FP=" + sim.lossRatio.toFixed(1) + " seed=" + seed + presetTxt + "</div>";
  var cv = makeCanvas("bms-thermal-chart");
  var trendCv = makeCanvas("bms-trend-chart");
  var idx = 0;
  function paint() {
    if (runId !== activeBmsRun) return;
    idx = Math.min(idx + 2, sim.frames.length - 1);
    var frame = sim.frames[idx];
    Array.from(grid.children).forEach(function (el, i) {
      var r = frame.risks[i];
      var hot = clamp((frame.temps[i] - 35) / 35, 0, 1);
      el.style.background = "rgba(" + Math.round(60 + 195 * Math.max(r, hot)) + ",8,0," + (0.18 + 0.65 * Math.max(r, hot)) + ")";
      el.style.borderColor = "rgba(255,26,26," + (0.15 + r * 0.75) + ")";
      el.style.boxShadow = r > sim.threshold ? "0 0 24px rgba(255,26,26,0.45)" : "none";
      el.dataset.risk = r.toFixed(4);
      el.dataset.temp = frame.temps[i].toFixed(2);
      el.dataset.slope = frame.slopes[i].toFixed(5);
      el.dataset.rct = frame.rcts[i].toFixed(5);
      el.dataset.rsei = frame.rseis[i].toFixed(5);
      el.dataset.hot = hot > 0.35 ? "1" : "0";
      el.querySelector(".cell-risk").textContent = r.toFixed(2);
      el.querySelector(".cell-temp").textContent = frame.temps[i].toFixed(1) + " C";
    });
    drawBmsThermal(cv, sim, frame);
    var upto = sim.frames.slice(0, idx + 1);
    drawMultiLine(trendCv, [
      { name: "max risk", values: upto.map(function (f) { return f.maxRisk; }), color: "#ff1a1a", glow: true },
      { name: "Tmax/100", values: upto.map(function (f) { return Math.max.apply(null, f.temps) / 100; }), color: "#ff9f1a" }
    ], { yMin: 0, yMax: 1, xMax: dur, title: "Risk and pack temperature trend", legend: true, yDigits: 2 });
    if (idx < sim.frames.length - 1) requestAnimationFrame(paint);
    else {
      sim.alerts.forEach(function (a) { log.innerHTML += '<div class="' + a.kind + '">' + escapeHtml(a.text) + "</div>"; });
      log.innerHTML += '<div class="ok">Complete. fault=' + (sim.faultCell >= 0 ? "C" + sim.faultCell : "none") + " maxRisk=" + frame.maxRisk.toFixed(3) + "</div>";
      log.scrollTop = log.scrollHeight;
      var action = frame.maxRisk > sim.threshold
        ? "Cool or isolate C" + frame.maxCell + " first; inspect impedance rise before pack-level thermal spread."
        : "No cell crossed the action threshold. Continue monitoring; lower the threshold if false negatives matter more than false positives.";
      window.__kfBms.maxRisk = frame.maxRisk;
      window.__kfBms.maxCell = frame.maxCell;
      window.__kfBms.finalTmax = Math.max.apply(null, frame.temps);
      var bmsConf = computeBmsConfidence(window.__kfBms, frame);
      setHtml("bms-decision", "<strong>BMS output:</strong> C" + frame.maxCell + " has max risk " + frame.maxRisk.toFixed(3) + ". <strong>Action:</strong> " + action);
      setReadouts("bms-readout", [
        { k: "Highest cell", v: "C" + frame.maxCell },
        { k: "Risk / gate", v: frame.maxRisk.toFixed(3) + " / " + sim.threshold.toFixed(2) },
        { k: "Tmax", v: Math.max.apply(null, frame.temps).toFixed(1) + " C" },
        { k: "Rct + RSEI", v: (frame.rcts[frame.maxCell] + frame.rseis[frame.maxCell]).toFixed(3) + " ohm" },
        { k: "Topology", v: sim.topology.info.label },
        { k: "FN/FP cost", v: sim.lossRatio.toFixed(1) + "x" }
      ]);
      renderConfidence("bms-confidence", bmsConf.confidence, bmsConf.detail);
      recordRun("bms", {
        summary: "C" + frame.maxCell + " risk " + frame.maxRisk.toFixed(3) + " / gate " + sim.threshold.toFixed(2) + ", seed " + seed,
        key_metric: Math.max.apply(null, frame.temps).toFixed(1) + " C Tmax",
        confidence: bmsConf.confidence
      });
    }
  }
  requestAnimationFrame(paint);
}

function runBmsSweep() {
  var n = parseInt(document.getElementById("pack-slider").value, 10);
  var dur = parseInt(document.getElementById("dur-slider").value, 10);
  var fault = document.getElementById("sw-fault").checked;
  var eis = document.getElementById("sw-eis").checked;
  var asym = document.getElementById("sw-asym").checked;
  var ambientC = num("bms-ambient", 45);
  var seedEl = document.getElementById("bms-seed");
  var sweepEl = document.getElementById("bms-sweep-count");
  var seedBase = seedEl ? parseInt(seedEl.value, 10) : 42;
  if (!Number.isFinite(seedBase)) seedBase = 42;
  var sweepCount = sweepEl ? parseInt(sweepEl.value, 10) : 16;
  if (!Number.isFinite(sweepCount)) sweepCount = 16;
  sweepCount = Math.max(3, Math.min(80, sweepCount));
  var scenarioKey = bmsScenarioKey(n, dur, fault, eis, asym, ambientC);

  var maxRisks = [];
  var tmaxs = [];
  var maxCells = {};
  var alerts = 0;
  var detectionTimes = [];
  var correctFaultHits = 0;
  var nonFaultAlerts = 0;
  for (var i = 0; i < sweepCount; i++) {
    var sim = simulateBmsPhysics(n, dur, fault, eis, asym, seedBase + i, bmsPresetFromDiagnostics);
    var frame = sim.frames[sim.frames.length - 1];
    if (!frame) continue;
    var maxRisk = Number(frame.maxRisk);
    if (!Number.isFinite(maxRisk)) continue;
    maxRisks.push(maxRisk);
    tmaxs.push(Math.max.apply(null, frame.temps));
    maxCells[frame.maxCell] = (maxCells[frame.maxCell] || 0) + 1;
    var firstAlert = (sim.frames || []).find(function (f) { return f.maxRisk > sim.threshold; });
    if (firstAlert) {
      alerts += 1;
      detectionTimes.push(firstAlert.t);
      if (sim.faultCell >= 0 && firstAlert.maxCell === sim.faultCell) correctFaultHits += 1;
      if (sim.faultCell < 0 || firstAlert.maxCell !== sim.faultCell) nonFaultAlerts += 1;
    }
  }
  if (!maxRisks.length) {
    showToast("Sweep failed to produce any frames.", "warn");
    return;
  }
  var dominantCell = Object.keys(maxCells).map(function (k) {
    return { cell: Number(k), hits: maxCells[k] };
  }).sort(function (a, b) { return b.hits - a.hits; })[0];
  var meanRisk = mean(maxRisks);
  var stdRisk = stddev(maxRisks);
  var alertRate = alerts / maxRisks.length;
  var meanTmax = mean(tmaxs);
  var meanDetect = detectionTimes.length ? mean(detectionTimes) : null;
  var faultHitRate = alerts ? correctFaultHits / alerts : null;
  var nonFaultAlertRate = alerts ? nonFaultAlerts / alerts : 0;
  window.__kfBmsSweep = {
    scenarioKey: scenarioKey,
    count: maxRisks.length,
    startSeed: seedBase,
    endSeed: seedBase + maxRisks.length - 1,
    meanRisk: meanRisk,
    stdRisk: stdRisk,
    alertRate: alertRate,
    meanTmax: meanTmax,
    meanDetectionTime: meanDetect,
    faultHitRate: faultHitRate,
    nonFaultAlertRate: nonFaultAlertRate,
    dominantCell: dominantCell ? dominantCell.cell : null
  };

  var note = document.getElementById("bms-sweep-note");
  if (note) {
    note.textContent = "Sweep " + maxRisks.length + " seeds: alert rate " + Math.round(alertRate * 100) + "%, mean max-risk " + meanRisk.toFixed(3) + " +/- " + stdRisk.toFixed(3) + ", mean detect " + (meanDetect == null ? "--" : meanDetect.toFixed(0) + "s") + ", fault-hit " + (faultHitRate == null ? "--" : Math.round(faultHitRate * 100) + "%") + ", non-fault alerts " + Math.round(nonFaultAlertRate * 100) + "%.";
  }
  var log = document.getElementById("bms-log");
  if (log) {
    log.innerHTML += '<div class="info">Sweep seeds ' + seedBase + "-" + (seedBase + maxRisks.length - 1) + ": alert rate " + (alertRate * 100).toFixed(1) + "%, mean maxRisk " + meanRisk.toFixed(3) + ", detect " + (meanDetect == null ? "--" : meanDetect.toFixed(0) + "s") + ", fault-hit " + (faultHitRate == null ? "--" : (faultHitRate * 100).toFixed(0) + "%") + ", non-fault alerts " + (nonFaultAlertRate * 100).toFixed(0) + "%.</div>";
    log.scrollTop = log.scrollHeight;
  }

  if (window.__kfBms && window.__kfBms.scenarioKey === scenarioKey && window.__kfBms.frames && window.__kfBms.frames.length) {
    var last = window.__kfBms.frames[window.__kfBms.frames.length - 1];
    var bmsConf = computeBmsConfidence(window.__kfBms, last);
    renderConfidence("bms-confidence", bmsConf.confidence, bmsConf.detail);
  }
  var sweepConf = clamp(0.74 - stdRisk * 1.8 - Math.abs(alertRate - 0.5) * 0.10, 0.26, 0.86);
  recordRun("bms_sweep", {
    summary: maxRisks.length + " seeds, alert rate " + (alertRate * 100).toFixed(1) + "%, fault-hit " + (faultHitRate == null ? "--" : (faultHitRate * 100).toFixed(0) + "%"),
    key_metric: "mean detect " + (meanDetect == null ? "--" : meanDetect.toFixed(0) + "s"),
    confidence: sweepConf
  });
}

// Materials screening mirror from core/materials_physics.py.
var MATERIAL_DOPANTS = {
  Al: { frac: 0.04, valence: 3.0, phase: 0.34, jt: 0.16, p2: 0.42, rate: 0.04, mass: 26.9815, cost: 2.7 },
  Ti: { frac: 0.03, valence: 4.0, phase: 0.28, jt: 0.44, p2: 0.30, rate: 0.08, mass: 47.867, cost: 11.0 }
};
var MATERIAL_EVIDENCE_ANCHORS = [
  { formula: "Na0.67Fe0.3Mn0.7O2", Na: 0.67, Mn: 0.70, Fe: 0.30, retention: 0.826, cycles: 300, cRate: 5 },
  { formula: "Na0.75Co0.125Cu0.125Fe0.125Ni0.125Mn0.5O2", Na: 0.75, Mn: 0.50, Fe: 0.125, capacity: 70, retention: 0.96, cycles: 500, cRate: 1 },
  { formula: "Na0.76Mn0.5Ni0.3Fe0.1Mg0.1O2", Na: 0.76, Mn: 0.50, Fe: 0.10, retention: 0.80, cycles: 700 }
];
function materialKnobs() {
  return {
    wCap: num("mat-w-cap", 0.32),
    wStab: num("mat-w-stab", 0.32),
    wFade: num("mat-w-fade", 0.22),
    wCost: num("mat-w-cost", 0.14),
    upperV: num("mat-upper-v", 4.10),
    ehullSlope: Math.max(1, num("mat-ehull-slope", 20)),
    chargePenalty: Math.max(0, num("mat-charge-penalty", 0.10)),
    defectPenalty: Math.max(0, num("mat-defect-penalty", 0.06))
  };
}
function materialDopants(comp) {
  var out = {};
  if (comp.al || comp.al_doped) out.Al = MATERIAL_DOPANTS.Al.frac;
  if (comp.ti || comp.ti_doped) out.Ti = MATERIAL_DOPANTS.Ti.frac;
  return out;
}
function normalizeMaterialComposition(comp) {
  var dop = materialDopants(comp);
  var na = clamp(Number(comp.Na != null ? comp.Na : comp.na) || 1.0, 0.45, 1.30);
  var mnRaw = clamp(Number(comp.Mn != null ? comp.Mn : comp.mn) || 0.5, 0, 1.4);
  var feRaw = clamp(Number(comp.Fe != null ? comp.Fe : comp.fe) || 0.5, 0, 1.4);
  var dopRaw = Object.keys(dop).reduce(function (a, k) { return a + dop[k]; }, 0);
  var tmTotal = Math.max(1e-9, mnRaw + feRaw + dopRaw);
  var siteDop = {};
  Object.keys(dop).forEach(function (k) { siteDop[k] = dop[k] / tmTotal; });
  return {
    Na: na,
    Mn: mnRaw / tmTotal,
    Fe: feRaw / tmTotal,
    dopants: siteDop,
    dopantTotal: dopRaw / tmTotal,
    tmTotalRaw: tmTotal,
    siteError: tmTotal - 1,
    siteErrorAbs: Math.abs(tmTotal - 1),
    al: !!(comp.al || comp.al_doped),
    ti: !!(comp.ti || comp.ti_doped)
  };
}

function materialSelectionFromControls(normalized) {
  var raw = {
    Na: parseFloat(document.getElementById("na-slider").value),
    Mn: parseFloat(document.getElementById("mn-slider").value),
    Fe: parseFloat(document.getElementById("fe-slider").value),
    al: document.getElementById("sw-al").checked,
    ti: document.getElementById("sw-ti").checked
  };
  if (!normalized) return raw;
  var norm = normalizeMaterialComposition(raw);
  return {
    Na: raw.Na,
    Mn: norm.Mn,
    Fe: norm.Fe,
    al: raw.al,
    ti: raw.ti,
    rawMn: raw.Mn,
    rawFe: raw.Fe,
    normalized: norm
  };
}

function updateMaterialSiteNote(raw) {
  var note = document.getElementById("mat-site-note");
  if (!note) return;
  raw = raw || materialSelectionFromControls(false);
  var norm = normalizeMaterialComposition(raw);
  var dopText = Object.keys(norm.dopants || {}).map(function (k) {
    return k + "=" + (norm.dopants[k] * 100).toFixed(1) + "%";
  }).join(", ") || "no dopant";
  var rawSum = Number(norm.tmTotalRaw || 0);
  var warn = Math.abs(rawSum - 1) > 0.02;
  note.innerHTML = (warn ? '<span class="tag warn">normalized</span> ' : '<span class="tag ok">site balanced</span> ')
    + "TM site raw sum " + rawSum.toFixed(3) + "; scoring uses Mn=" + norm.Mn.toFixed(3)
    + ", Fe=" + norm.Fe.toFixed(3) + ", " + dopText + ".";
}
function dopantStrength(norm, key) {
  return clamp(Object.keys(norm.dopants).reduce(function (sum, el) {
    var meta = MATERIAL_DOPANTS[el] || {};
    return sum + (norm.dopants[el] / Math.max(1e-6, meta.frac || 0.04)) * (meta[key] || 0);
  }, 0), 0, 1);
}
function nearestMaterialEvidence(norm) {
  var best = null;
  MATERIAL_EVIDENCE_ANCHORS.forEach(function (a) {
    var d = Math.sqrt(
      1.6 * Math.pow(norm.Na - a.Na, 2) +
      1.2 * Math.pow(norm.Mn - a.Mn, 2) +
      1.0 * Math.pow(norm.Fe - a.Fe, 2)
    );
    if (!best || d < best.distance) best = Object.assign({ distance: d }, a);
  });
  return best || { distance: 1, formula: "none" };
}
function simulateMaterialRetention(fadeTerms, cycleLimit) {
  var beta = 0.72;
  var k = Object.keys(fadeTerms).reduce(function (a, n) { return a + Math.max(0, fadeTerms[n]); }, 0);
  k = Math.max(k, 1e-7);
  var p2Frac = (fadeTerms.p2_o2 || 0) / k;
  var jtFrac = (fadeTerms.jahn_teller || 0) / k;
  var cycles = Math.max(1000, Math.min(2200, Math.round(cycleLimit || 1200)));
  var knee = clamp((0.58 - 0.18 * p2Frac - 0.10 * jtFrac) * cycles, 80, cycles * 0.92);
  var xs = [], ys = [], lo = [], hi = [];
  var step = Math.max(1, Math.floor(cycles / 240));
  for (var n = 0; n <= cycles; n += step) {
    var accel = n > knee ? 1 + (p2Frac * 0.75 + jtFrac * 0.35) * Math.pow((n - knee) / Math.max(1, cycles - knee), 1.35) : 1;
    var cap = clamp(1 - (1 - Math.exp(-k * Math.pow(n, beta) * accel)), 0.42, 1.01);
    var band = clamp(0.012 + 0.055 * Math.sqrt(n / cycles), 0, 0.12);
    xs.push(n); ys.push(cap); lo.push(clamp(cap - band, 0.35, 1.01)); hi.push(clamp(cap + band, 0.35, 1.04));
  }
  var eol = null;
  for (var i = 0; i < ys.length; i++) { if (ys[i] <= 0.8) { eol = xs[i]; break; } }
  return { cycles: xs, retention: ys, lo: lo, hi: hi, knee_cycle: Math.round(knee), eol80_cycle: eol };
}
function scoreComposition(comp, T, cfg) {
  T = T || 318.15;
  cfg = cfg || materialKnobs();
  var norm = normalizeMaterialComposition(comp);
  var na = norm.Na, mn = norm.Mn, fe = norm.Fe;
  var phaseStabilization = dopantStrength(norm, "phase");
  var jtSuppression = dopantStrength(norm, "jt");
  var p2Suppression = dopantStrength(norm, "p2");
  var dopCharge = Object.keys(norm.dopants).reduce(function (sum, el) { return sum + norm.dopants[el] * (MATERIAL_DOPANTS[el].valence || 3); }, 0);
  var mnOxRaw = (4.0 - na - 3.0 * fe - dopCharge) / Math.max(mn, 1e-6);
  var mnOxError = Math.max(0, 3.0 - mnOxRaw, mnOxRaw - 4.0);
  var mnOx = clamp(mnOxRaw, 3.0, 4.0);
  var mn3 = clamp(4.0 - mnOx, 0, 1);
  var chargeRisk = clamp(0.65 * mnOxError + 0.90 * norm.siteErrorAbs, 0, 1);
  var naMobility = clamp(1 - 1.15 * Math.abs(na - 0.88) + 0.08 * sigmoid((na - 0.65) / 0.08), 0.22, 1);
  var feGate = sigmoid((cfg.upperV - 4.02) / 0.10);
  var mnUtil = clamp(0.70 + 0.17 * sigmoid((cfg.upperV - 3.72) / 0.12) - 0.20 * chargeRisk - 0.12 * norm.siteErrorAbs, 0.15, 0.94);
  var feUtil = clamp(0.14 + 0.58 * feGate - 0.10 * chargeRisk, 0.05, 0.72);
  var oxygenE = clamp((cfg.upperV - 4.18) * 0.75, 0, 0.16) * mn * clamp(1 - na, 0, 0.45);
  var mnE = mn * mn3 * mnUtil;
  var feE = fe * feUtil;
  var eFormula = Math.min(clamp(na - 0.22 - 0.08 * norm.siteErrorAbs, 0, 0.92), mnE + feE + oxygenE);
  var mass = na * 22.9898 + mn * 54.938 + fe * 55.845 + 2 * 15.999;
  Object.keys(norm.dopants).forEach(function (el) { mass += norm.dopants[el] * MATERIAL_DOPANTS[el].mass; });
  var utilization = clamp(0.94 * naMobility - 0.20 * norm.siteErrorAbs - 0.10 * chargeRisk, 0.35, 0.98);
  var q0 = 26801 * eFormula / Math.max(1e-6, mass) * utilization;
  var denom = Math.max(1e-9, mnE + feE + oxygenE);
  var avgVoltage = ((mnE * (3.50 + 0.24 * (1 - mn3) + 0.05 * (cfg.upperV - 4.0))) + (feE * (3.18 + 0.08 * feGate)) + oxygenE * 4.15) / denom + 0.06 * (0.85 - na);
  avgVoltage = clamp(avgVoltage, 2.60, Math.min(cfg.upperV, 4.35));
  var p2Crit = 4.04 + 0.12 * fe + 0.18 * phaseStabilization - 0.13 * mn3 - 0.08 * Math.max(0, 0.78 - na);
  var naP2Weight = 0.35 + (1 - 0.35) * sigmoid((0.86 - na) / 0.12);
  var p2Risk = clamp(sigmoid((cfg.upperV - p2Crit) / 0.075) * naP2Weight * (1 - 0.52 * p2Suppression), 0, 1);
  var jtRisk = clamp(mn * mn3 * (1 - 0.55 * jtSuppression) * (1 + 0.18 * sigmoid((cfg.upperV - 4.05) / 0.10)), 0, 1);
  var oxygenRisk = clamp(0.12 + 0.54 * p2Risk + 0.34 * clamp(cfg.upperV - 4.10, 0, 0.35) / 0.35 + 0.30 * Math.max(0, 0.76 - na) + 0.18 * mn * (1 - mn3) - 0.20 * phaseStabilization, 0, 1);
  var mixingRisk = clamp(0.10 + 0.28 * Math.abs(mn - fe) + 0.34 * norm.siteErrorAbs + 0.16 * Math.max(0, 0.72 - na) - 0.10 * phaseStabilization, 0, 1);
  var moistureRisk = clamp(0.16 + 0.62 * Math.abs(na - 0.92) + 0.18 * norm.siteErrorAbs, 0, 1);
  var defectScore = clamp(1 - (0.26 * oxygenRisk + 0.22 * mixingRisk + 0.18 * moistureRisk + 0.24 * jtRisk + 0.22 * chargeRisk), 0, 1);
  var ehull = clamp(0.020 + 0.080 * Math.max(0, 0.78 - na) + 0.050 * norm.siteErrorAbs + 0.038 * oxygenRisk + 0.030 * mixingRisk + 0.018 * chargeRisk - 0.022 * phaseStabilization, 0, 0.26);
  var phaseStab = 1 / (1 + expClamp(cfg.ehullSlope * (ehull - 0.055), -40, 40));
  var thermalOnset = 238 - 42 * oxygenRisk - 24 * jtRisk + 24 * phaseStabilization + 10 * fe;
  var thermalAbuse = clamp((thermalOnset - 170) / 125, 0, 1);
  var rateCap = clamp(0.45 + 0.42 * naMobility + 0.12 * fe + 0.12 * dopantStrength(norm, "rate") - 0.16 * chargeRisk, 0.05, 1);
  var arrh = expClamp(-0.42 / 8.617e-5 * (1 / T - 1 / 318.15), -3, 3);
  var fadeTerms = {
    sei: 2.7e-4 * arrh * (0.75 + 0.50 * moistureRisk),
    p2_o2: 7.5e-4 * p2Risk * (0.75 + 0.35 * sigmoid((cfg.upperV - 4.05) / 0.08)),
    jahn_teller: 5.8e-4 * jtRisk,
    oxygen: 4.2e-4 * oxygenRisk,
    rate: 2.6e-4 * (1 - rateCap),
    charge_site: 4.8e-4 * chargeRisk
  };
  var fadeK = Object.keys(fadeTerms).reduce(function (a, k) { return a + fadeTerms[k]; }, 0);
  var fade500 = clamp(1 - Math.exp(-fadeK * Math.pow(500, 0.72)), 0.002, 0.68);
  var cycleLife = clamp(Math.pow(-Math.log(0.80) / Math.max(fadeK, 1e-7), 1 / 0.72), 50, 5000);
  var energyDensity = q0 * avgVoltage;
  var costKg = 2.5 + (na * 22.9898 / mass) * 3.1 + (mn * 54.938 / mass) * 2.4 + (fe * 55.845 / mass) * 0.45;
  Object.keys(norm.dopants).forEach(function (el) { costKg += (norm.dopants[el] * MATERIAL_DOPANTS[el].mass / mass) * MATERIAL_DOPANTS[el].cost; });
  var costKwh = costKg / Math.max(energyDensity / 1000, 0.01);
  var stability = clamp(0.24 * (1 - fade500) + 0.22 * phaseStab + 0.18 * defectScore + 0.14 * thermalAbuse + 0.12 * rateCap + 0.10 * (1 - chargeRisk), 0, 1);
  var evidence = nearestMaterialEvidence(norm);
  var confidence = clamp(0.84 - 0.22 * Math.min(evidence.distance, 1.5) - 0.28 * chargeRisk - 0.22 * norm.siteErrorAbs - 0.12 * oxygenRisk + 0.05, 0.18, 0.92);
  var score = cfg.wCap * clamp(q0 / 180, 0, 1.25) + cfg.wStab * stability + cfg.wFade * (1 - fade500) + cfg.wCost * clamp(1 - costKwh / 220, 0, 1) - cfg.chargePenalty * chargeRisk - cfg.defectPenalty * (1 - defectScore);
  var phaseState = p2Risk > 0.62 ? "P2->O2 transition risk" : chargeRisk > 0.45 ? "charge-compensated defect phase" : phaseStab < 0.38 ? "mixed/impurity phase risk" : jtRisk > 0.38 ? "JT-distorted layered phase" : "P2 layered phase";
  var curve = simulateMaterialRetention(fadeTerms, Math.max(1000, Math.min(2200, cycleLife * 1.25)));
  return {
    Q0: q0, Q500: q0 * (1 - fade500), fade500: fade500, cycleLife: cycleLife,
    avgVoltage: avgVoltage, stability: stability, jtIndex: jtRisk, energyDensity: energyDensity,
    costKwh: costKwh, score: score, oxygenRisk: oxygenRisk, chargeRisk: chargeRisk,
    phaseStability: phaseStab, phaseState: phaseState, ehull: ehull, p2Risk: p2Risk,
    mnOx: mnOx, mn3Fraction: mn3, siteError: norm.siteError, normalized: norm,
    latticeSpacing: clamp(5.48 + 0.22 * na - 0.16 * p2Risk - 0.08 * norm.siteErrorAbs + 0.04 * phaseStabilization, 5.25, 5.82),
    thermalOnset: thermalOnset, rateCapability: rateCap, defectScore: defectScore,
    mixingRisk: mixingRisk, moistureRisk: moistureRisk, confidence: confidence, evidence: evidence,
    mechanisms: fadeTerms, retentionCurve: curve,
    radar: [
      { label: "Capacity", value: clamp(q0 / 180, 0, 1) },
      { label: "Phase", value: phaseStab },
      { label: "Fade", value: 1 - fade500 },
      { label: "Cost", value: clamp(1 - costKwh / 220, 0, 1) },
      { label: "O2 Safe", value: 1 - oxygenRisk },
      { label: "Charge", value: 1 - chargeRisk }
    ]
  };
}

function apiMaterialToProp(pred) {
  pred = pred || {};
  var curve = pred.retention_curve || {};
  return {
    Q0: Number(pred.capacity),
    Q500: Number(pred.capacity_500),
    fade500: Number(pred.fade_500),
    cycleLife: Number(pred.cycle_life),
    avgVoltage: Number(pred.voltage),
    stability: Number(pred.stability),
    jtIndex: Number(pred.jt_index),
    energyDensity: Number(pred.energy_density),
    costKwh: Number(pred.cost_usd_kwh),
    score: Number(pred.score),
    oxygenRisk: Number(pred.oxygen_risk),
    chargeRisk: Number(pred.charge_balance_risk),
    phaseStability: Number(pred.phase_stability),
    phaseState: pred.phase_state || "screened phase",
    ehull: Number(pred.ehull_ev_atom),
    p2Risk: Number(pred.p2_o2_risk),
    mnOx: Number(pred.mn_oxidation_state),
    mn3Fraction: Number(pred.mn3_fraction),
    siteError: Number(pred.site_balance_error),
    normalized: pred.normalized_composition,
    latticeSpacing: Number(pred.lattice_spacing_A),
    thermalOnset: Number(pred.thermal_onset_C),
    rateCapability: Number(pred.rate_capability),
    defectScore: Number(pred.defect_score),
    mixingRisk: Number(pred.tm_mixing_risk),
    moistureRisk: Number(pred.moisture_risk),
    confidence: Number(pred.confidence),
    evidence: pred.evidence,
    mechanisms: pred.mechanisms,
    retentionCurve: { cycles: curve.cycles || [], retention: curve.retention || [], lo: curve.lo || [], hi: curve.hi || [], knee_cycle: curve.knee_cycle, eol80_cycle: curve.eol80_cycle },
    radar: pred.radar || []
  };
}

function apiCompositionToUi(comp) {
  comp = comp || {};
  return {
    Na: Number(comp.Na != null ? comp.Na : comp.na),
    Mn: Number(comp.Mn != null ? comp.Mn : comp.mn),
    Fe: Number(comp.Fe != null ? comp.Fe : comp.fe),
    al: !!(comp.al || comp.al_doped),
    ti: !!(comp.ti || comp.ti_doped)
  };
}

function generateCandidates(selected, cfg) {
  var pts = [];
  for (var na = 0.62; na <= 1.10; na += 0.04) {
    for (var mn = 0.20; mn <= 0.80; mn += 0.04) {
      ["none", "al", "ti", "al_ti"].forEach(function (d) {
        var al = d === "al" || d === "al_ti";
        var ti = d === "ti" || d === "al_ti";
        var dop = (al ? MATERIAL_DOPANTS.Al.frac : 0) + (ti ? MATERIAL_DOPANTS.Ti.frac : 0);
        var fe = clamp(1.0 - mn - dop, 0.06, 0.82);
        var comp = { Na: na, Mn: mn, Fe: fe, al: al, ti: ti };
        var prop = scoreComposition(comp, 318.15, cfg);
        pts.push({ comp: comp, prop: prop });
      });
    }
  }
  return pts;
}

function paretoMark(items) {
  items.forEach(function (it) { it.front = true; });
  for (var i = 0; i < items.length; i++) {
    for (var j = 0; j < items.length; j++) {
      if (i === j) continue;
      var a = items[j].prop, b = items[i].prop;
      var ge = a.Q0 >= b.Q0 && -a.fade500 >= -b.fade500 && a.cycleLife >= b.cycleLife && -a.costKwh >= -b.costKwh;
      var gt = a.Q0 > b.Q0 || -a.fade500 > -b.fade500 || a.cycleLife > b.cycleLife || -a.costKwh > -b.costKwh;
      if (ge && gt) { items[i].front = false; break; }
    }
  }
}

function drawAtom(ctx, x, y, type) {
  ctx.beginPath();
  var r = 5;
  var color = "#ff1a1a";
  if (type === "Mn") { color = "#ff5c5c"; r = 6; }
  else if (type === "Fe") { color = "#ec4899"; r = 6; }
  else if (type === "Al") { color = "#3b82f6"; r = 5.5; }
  else if (type === "Ti") { color = "#10b981"; r = 5.5; }
  else if (type === "Na") { color = "#eab308"; r = 4; }

  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 1;
  ctx.stroke();

  // Glow
  if (type === "Al" || type === "Ti") {
    ctx.shadowColor = color;
    ctx.shadowBlur = 8;
    ctx.beginPath();
    ctx.arc(x, y, r - 1, 0, Math.PI * 2);
    ctx.stroke();
    ctx.shadowBlur = 0;
  }
}

function drawLegendAtom(ctx, x, y, type, label) {
  drawAtom(ctx, x, y, type);
  ctx.fillStyle = "#aaa";
  ctx.font = "9px JetBrains Mono, monospace";
  ctx.fillText(label, x + 10, y + 3);
}

function drawLatticeSimulation(selected) {
  var cv = document.getElementById("lattice-canvas");
  if (!cv) return;
  var ctx = cv.getContext("2d");
  var W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);

  var props = selected.prop || scoreComposition(selected, 318.15, materialKnobs());
  var norm = props.normalized && props.normalized.Na != null ? props.normalized : normalizeMaterialComposition(selected);
  var na = Number(norm.Na != null ? norm.Na : selected.Na);
  var mn = Number(norm.Mn != null ? norm.Mn : selected.Mn);
  var dopants = norm.dopants || { Al: Number(norm.Al || 0), Ti: Number(norm.Ti || 0) };
  var al = !!(selected.al || selected.al_doped || dopants.Al);
  var ti = !!(selected.ti || selected.ti_doped || dopants.Ti);
  var jtDist = clamp(Number(props.jtIndex) || 0, 0, 1);
  var collapse = clamp((Number(props.p2Risk) || 0) * 0.72 + Math.max(0, 0.72 - na) * 0.35, 0, 1);

  // Background grid
  ctx.strokeStyle = "rgba(255,255,255,0.03)";
  ctx.lineWidth = 1;
  for (var i = 0; i < W; i += 20) {
    ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, H); ctx.stroke();
  }

  var cols = 8;
  var dx = W / (cols + 1);
  function siteType(c, layerOffset) {
    var alFrac = Number(dopants.Al || 0);
    var tiFrac = Number(dopants.Ti || 0);
    var r = (((c + 1) * 37 + layerOffset * 19) % 100) / 100;
    if (al && r < alFrac) return "Al";
    if (ti && r < alFrac + tiFrac) return "Ti";
    return r < alFrac + tiFrac + mn ? "Mn" : "Fe";
  }

  // Draw bonds (Background bonds first)
  ctx.strokeStyle = "rgba(255,255,255,0.12)";
  ctx.lineWidth = 1.5;
  for (var c = 0; c < cols; c++) {
    var x = dx * (c + 1);
    var yTop = 35 + collapse * 15;
    var yBot = 115 - collapse * 15;
    
    // Lattice distortion offset, scaled from the charge-balance JT index.
    var jitterTop = (Math.sin(c * 2) * jtDist * 8);
    var jitterBot = (Math.cos(c * 2) * jtDist * 8);

    ctx.beginPath();
    ctx.moveTo(x, yTop + jitterTop);
    ctx.lineTo(x, yBot + jitterBot);
    ctx.stroke();

    // Horizontal top bonds
    if (c < cols - 1) {
      var nextX = dx * (c + 2);
      var nextJitterTop = (Math.sin((c + 1) * 2) * jtDist * 8);
      ctx.beginPath();
      ctx.moveTo(x, yTop + jitterTop);
      ctx.lineTo(nextX, yTop + nextJitterTop);
      ctx.stroke();

      var nextJitterBot = (Math.cos((c + 1) * 2) * jtDist * 8);
      ctx.beginPath();
      ctx.moveTo(x, yBot + jitterBot);
      ctx.lineTo(nextX, yBot + nextJitterBot);
      ctx.stroke();
    }
  }

  // Draw Atoms
  for (var c = 0; c < cols; c++) {
    var x = dx * (c + 1);
    var yTop = 35 + collapse * 15;
    var yBot = 115 - collapse * 15;
    var jitterTop = (Math.sin(c * 2) * jtDist * 8);
    var jitterBot = (Math.cos(c * 2) * jtDist * 8);

    var topType = siteType(c, 0);
    var botType = siteType(c, 1);

    drawAtom(ctx, x, yTop + jitterTop, topType);
    drawAtom(ctx, x, yBot + jitterBot, botType);

    // Sodium Ions in the gallery
    var showNa = ((c + 0.5) / cols) < clamp(na, 0, 1.20) / 1.20;
    if (showNa) {
      var naY = (yTop + yBot) / 2 + (Math.sin(c * 5) * 2);
      drawAtom(ctx, x + dx/2, naY, "Na");
    }
  }

  // Draw legend
  ctx.font = "9px JetBrains Mono, monospace";
  var legX = 12;
  var legY = H - 12;
  drawLegendAtom(ctx, legX, legY, "Mn", "Mn");
  drawLegendAtom(ctx, legX + 50, legY, "Fe", "Fe");
  drawLegendAtom(ctx, legX + 100, legY, "Na", "Na (gallery)");
  if (al) drawLegendAtom(ctx, legX + 200, legY, "Al", "Al (pillar)");
  if (ti) drawLegendAtom(ctx, legX + 280, legY, "Ti", "Ti (dopant)");

  var phaseName = props.phaseState || "P2 layered phase";
  var phaseColor = "#22c55e";
  if ((props.p2Risk || 0) > 0.62 || collapse > 0.42) {
    phaseColor = "#ff1a1a";
  } else if ((props.chargeRisk || 0) > 0.35 || jtDist > 0.30) {
    phaseColor = "#ff9f1a";
  }

  setReadouts("lattice-readout", [
    { k: "Lattice Spacing", v: fmt(props.latticeSpacing || (5.62 - collapse * 0.45), 2) + " A" },
    { k: "Structure Phase", v: '<span style="color:' + phaseColor + ';font-weight:700">' + phaseName + '</span>', html: true },
    { k: "Mn valence", v: fmt(props.mnOx, 2) + "+ / Mn3 " + fmt((props.mn3Fraction || 0) * 100, 0) + "%" },
    { k: "Site balance", v: fmt((props.siteError || 0) * 100, 1) + "% vs TM=1" }
  ]);
}

// ── Radar Chart for Composition Quality ─────────────────────────────────
function drawRadarChart(props) {
  var cv = document.getElementById("radar-chart");
  if (!cv) return;
  var ctx = cv.getContext("2d");
  var W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  var cx = W / 2, cy = H / 2 + 5;
  var R = Math.min(cx, cy) - 35;
  var axes = [
    { label: "Capacity", value: clamp((props.Q0 - 80) / 130, 0, 1) },
    { label: "Stability", value: clamp(props.stability, 0, 1) },
    { label: "Fade Resist", value: clamp(1 - props.fade500, 0, 1) },
    { label: "Cost Effic.", value: clamp(1 - props.costKwh / 200, 0, 1) },
    { label: "O₂ Safety", value: clamp(1 - props.oxygenRisk, 0, 1) },
    { label: "Charge Bal.", value: clamp(1 - props.chargeRisk, 0, 1) }
  ];
  if (Array.isArray(props.radar) && props.radar.length) {
    axes = props.radar.map(function (r) {
      return { label: String(r.label || ""), value: clamp(Number(r.value), 0, 1) };
    });
  }
  var n = axes.length;
  var angleStep = (2 * Math.PI) / n;
  // Draw grid rings
  [0.25, 0.5, 0.75, 1.0].forEach(function (frac) {
    ctx.beginPath();
    for (var i = 0; i <= n; i++) {
      var a = -Math.PI / 2 + i * angleStep;
      var x = cx + Math.cos(a) * R * frac;
      var y = cy + Math.sin(a) * R * frac;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.strokeStyle = "rgba(255,255,255," + (frac === 1 ? 0.12 : 0.06) + ")";
    ctx.lineWidth = 1;
    ctx.stroke();
  });
  // Draw axis lines + labels
  ctx.font = "9px Inter, sans-serif";
  ctx.textAlign = "center";
  axes.forEach(function (ax, i) {
    var a = -Math.PI / 2 + i * angleStep;
    var ex = cx + Math.cos(a) * R;
    var ey = cy + Math.sin(a) * R;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(ex, ey);
    ctx.strokeStyle = "rgba(255,255,255,0.08)";
    ctx.stroke();
    var lx = cx + Math.cos(a) * (R + 22);
    var ly = cy + Math.sin(a) * (R + 22);
    ctx.fillStyle = "#888";
    ctx.fillText(ax.label, lx, ly + 3);
  });
  // Draw filled polygon
  ctx.beginPath();
  axes.forEach(function (ax, i) {
    var a = -Math.PI / 2 + i * angleStep;
    var r = R * ax.value;
    var x = cx + Math.cos(a) * r;
    var y = cy + Math.sin(a) * r;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.closePath();
  ctx.fillStyle = "rgba(255,26,26,0.18)";
  ctx.fill();
  ctx.strokeStyle = "#ff1a1a";
  ctx.lineWidth = 2;
  ctx.shadowColor = "#ff1a1a";
  ctx.shadowBlur = 8;
  ctx.stroke();
  ctx.shadowBlur = 0;
  // Draw value dots
  axes.forEach(function (ax, i) {
    var a = -Math.PI / 2 + i * angleStep;
    var r = R * ax.value;
    ctx.beginPath();
    ctx.arc(cx + Math.cos(a) * r, cy + Math.sin(a) * r, 3, 0, Math.PI * 2);
    ctx.fillStyle = "#fff";
    ctx.fill();
  });
}

// ── Cycle Life Degradation Curve ────────────────────────────────────────
function drawCycleLifeCurve(props) {
  var cv = makeCanvas("cycle-life-chart");
  if (!cv) return;
  var curve = props.retentionCurve || props.retention_curve;
  if (curve && Array.isArray(curve.retention) && curve.retention.length > 2) {
    var xMaxCurve = Array.isArray(curve.cycles) && curve.cycles.length ? curve.cycles[curve.cycles.length - 1] : curve.retention.length - 1;
    drawMultiLine(cv, [
      { name: "Capacity", values: curve.retention, color: "#22c55e", glow: true }
    ], {
      yMin: 0.5, yMax: 1.05, title: "Capacity retention from mechanism-resolved fade model",
      xMax: xMaxCurve, yDigits: 2, legend: false,
      band: curve.lo && curve.hi ? { lo: curve.lo, hi: curve.hi, color: "rgba(34,197,94,0.10)" } : null
    });
    var ctx0 = cv.getContext("2d");
    var W0 = cv.width, H0 = cv.height;
    var pad0 = { t: 28, r: 18, b: 34, l: 52 };
    var pw0 = W0 - pad0.l - pad0.r, ph0 = H0 - pad0.t - pad0.b;
    var y80a = pad0.t + ph0 - (0.8 - 0.5) / (1.05 - 0.5) * ph0;
    ctx0.setLineDash([4, 4]);
    ctx0.beginPath(); ctx0.moveTo(pad0.l, y80a); ctx0.lineTo(pad0.l + pw0, y80a);
    ctx0.strokeStyle = "rgba(255,26,26,0.5)"; ctx0.stroke(); ctx0.setLineDash([]);
    ctx0.fillStyle = "#ff1a1a"; ctx0.font = "9px JetBrains Mono, monospace"; ctx0.fillText("80% EOL", pad0.l + pw0 - 50, y80a - 4);
    if (curve.knee_cycle != null && xMaxCurve > 0) {
      var kx0 = pad0.l + clamp(curve.knee_cycle / xMaxCurve, 0, 1) * pw0;
      ctx0.setLineDash([3, 3]); ctx0.beginPath(); ctx0.moveTo(kx0, pad0.t); ctx0.lineTo(kx0, pad0.t + ph0);
      ctx0.strokeStyle = "rgba(234,179,8,0.4)"; ctx0.stroke(); ctx0.setLineDash([]);
      ctx0.fillStyle = "#eab308"; ctx0.fillText("knee", kx0 + 4, pad0.t + 12);
    }
    var finalCap = curve.retention[curve.retention.length - 1];
    var mechanisms = props.mechanisms || {};
    var mech = Object.keys(mechanisms).map(function (k) { return [k, Number(mechanisms[k]) || 0]; }).sort(function (a, b) { return b[1] - a[1]; });
    setReadouts("cycle-life-readout", [
      { k: "Knee Point", v: curve.knee_cycle != null ? curve.knee_cycle + " cycles" : "not reached" },
      { k: "EOL (80%)", v: curve.eol80_cycle != null ? curve.eol80_cycle + " cycles" : ">" + xMaxCurve + " cycles" },
      { k: "Fade@500", v: (100 * props.fade500).toFixed(1) + "%" },
      { k: "Driver", v: mech.length ? mech[0][0].replace(/_/g, " ") : "mixed" }
    ]);
    return;
  }
  var stab = clamp(props.stability || 0.5, 0, 1);
  var fade = clamp(props.fade500 || 0.15, 0, 1);
  var jtIdx = clamp(props.jtIndex || 0.5, 0, 1);
  // Simulate capacity retention over 1000 cycles
  var pts = [];
  var cap = 1.0;
  var baseRate = 0.00005 + fade * 0.0003;
  var kneePoint = Math.round(400 + stab * 600 - jtIdx * 200);
  kneePoint = clamp(kneePoint, 200, 900);
  for (var c = 0; c <= 1000; c += 5) {
    var rate = baseRate;
    if (c > kneePoint) {
      rate += (c - kneePoint) * 0.0000008 * (1 + fade * 2);
    }
    cap = Math.max(0.5, cap - rate);
    pts.push(cap);
  }
  // Draw using drawMultiLine
  var eolIdx = pts.findIndex(function (v) { return v < 0.8; });
  var eolCycle = eolIdx >= 0 ? eolIdx * 5 : 1000;
  drawMultiLine(cv, [
    { name: "Capacity", values: pts, color: "#22c55e", glow: true }
  ], {
    yMin: 0.5, yMax: 1.05, title: "Capacity retention vs cycles (simulated)",
    xMax: 1000, yDigits: 2, legend: false
  });
  // Draw 80% EOL line
  var ctx = cv.getContext("2d");
  var W = cv.width, H = cv.height;
  var pad = { t: 28, r: 18, b: 34, l: 52 };
  var pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
  var y80 = pad.t + ph - (0.8 - 0.5) / (1.05 - 0.5) * ph;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(pad.l, y80);
  ctx.lineTo(pad.l + pw, y80);
  ctx.strokeStyle = "rgba(255,26,26,0.5)";
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#ff1a1a";
  ctx.font = "9px JetBrains Mono, monospace";
  ctx.fillText("80% EOL", pad.l + pw - 50, y80 - 4);
  // Draw knee point marker
  if (kneePoint < 1000) {
    var kx = pad.l + (kneePoint / 1000) * pw;
    ctx.beginPath();
    ctx.moveTo(kx, pad.t);
    ctx.lineTo(kx, pad.t + ph);
    ctx.strokeStyle = "rgba(234,179,8,0.4)";
    ctx.setLineDash([3, 3]);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#eab308";
    ctx.fillText("knee", kx + 4, pad.t + 12);
  }
  setReadouts("cycle-life-readout", [
    { k: "Knee Point", v: kneePoint + " cycles" },
    { k: "EOL (80%)", v: eolCycle >= 1000 ? ">1000 cycles" : eolCycle + " cycles" },
    { k: "Fade@500", v: (100 * fade).toFixed(1) + "%" },
    { k: "Final Cap", v: (pts[pts.length - 1] * 100).toFixed(1) + "%" }
  ]);
}

function updateMat() {
  var selected = materialSelectionFromControls(false);
  document.getElementById("na-val").textContent = selected.Na.toFixed(2);
  document.getElementById("mn-val").textContent = selected.Mn.toFixed(2);
  document.getElementById("fe-val").textContent = selected.Fe.toFixed(2);
  updateMaterialSiteNote(selected);
  drawLatticeSimulation(selected);
}

async function runScreening() {
  var cfg = materialKnobs();
  var selected = materialSelectionFromControls(true);
  setHtml("mat-decision", "<strong>Screening:</strong> computing charge balance, phase stability, fade mechanisms, and local evidence distance...");
  var selectedProp = null;
  var items = null;
  var apiSource = "browser fallback";
  try {
    var res = await fetch("/api/screen/cathode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        na: selected.Na,
        mn: selected.Mn,
        fe: selected.Fe,
        al_doped: selected.al,
        ti_doped: selected.ti,
        temperature_K: 318.15,
        upper_voltage: cfg.upperV,
        ehull_slope: cfg.ehullSlope,
        w_capacity: cfg.wCap,
        w_stability: cfg.wStab,
        w_fade: cfg.wFade,
        w_cost: cfg.wCost,
        charge_penalty: cfg.chargePenalty,
        defect_penalty: cfg.defectPenalty
      })
    });
    if (!res.ok) throw new Error("screen endpoint returned " + res.status);
    var data = await res.json();
    selectedProp = apiMaterialToProp(data.predicted || {});
    items = (data.candidates || []).map(function (it) {
      return { comp: apiCompositionToUi(it.composition || {}), prop: apiMaterialToProp(it.properties || {}) };
    });
    apiSource = "API physics";
  } catch (err) {
    selectedProp = scoreComposition(selected, 318.15, cfg);
    items = generateCandidates(selected, cfg);
    showToast("Materials API unavailable; using browser physics fallback.", "info");
    console.warn(err);
  }
  items.push({ comp: selected, prop: selectedProp, selected: true });
  paretoMark(items);
  drawLatticeSimulation(Object.assign({}, selected, { prop: selectedProp }));
  drawRadarChart(selectedProp);
  drawCycleLifeCurve(selectedProp);
  document.getElementById("mat-cap").textContent = selectedProp.Q0.toFixed(0);
  document.getElementById("mat-volt").textContent = selectedProp.avgVoltage.toFixed(2) + "V";
  document.getElementById("mat-stab").textContent = selectedProp.stability.toFixed(2);
  document.getElementById("mat-jt").textContent = selectedProp.jtIndex.toFixed(2);
  selectedProp.comp = selected;
  selectedProp.rawComp = { Mn: selected.rawMn, Fe: selected.rawFe };
  selectedProp.candidates = items.slice(0, 48).map(function (it) {
    return { composition: it.comp, score: it.prop.score, capacity: it.prop.Q0, stability: it.prop.stability, fade500: it.prop.fade500 };
  });
  selectedProp.apiSource = apiSource;
  window.__kfMaterials = selectedProp;
  var cv = makeCanvas("mat-chart");
  var pts = items.map(function (it) {
    return { x: it.prop.Q0, y: it.prop.stability, front: it.front, selected: !!it.selected };
  });
  drawScatter(cv, pts, { title: "Computed Pareto front: capacity vs stability" });
  drawCompositionLandscape(makeCanvas("mat-landscape-chart"), items, selected);
  var synth = selectedProp.stability > 0.72 && selectedProp.fade500 < 0.16 && selectedProp.chargeRisk < 0.28 && selectedProp.oxygenRisk < 0.46 && Math.abs(selectedProp.siteError || 0) < 0.12;
  var normalizedNote = selected.normalized && Math.abs(selected.normalized.siteError || 0) > 0.02
    ? " Raw TM sliders were normalized before scoring."
    : "";
  var advice = synth
    ? "Good candidate for a small coin-cell synthesis queue, with XRD phase check before cycling."
    : "Keep in simulation queue; correct site balance, charge state, or phase risk before spending lab synthesis effort.";
  var evidenceText = selectedProp.evidence && selectedProp.evidence.nearest_formula ? ", nearest evidence " + escapeHtml(selectedProp.evidence.nearest_formula) : "";
  setHtml("mat-decision", "<strong>Screening output:</strong> score " + selectedProp.score.toFixed(3) + ", phase " + escapeHtml(selectedProp.phaseState || "screened") + ", fade500 " + (100 * selectedProp.fade500).toFixed(1) + "%, confidence " + fmt(selectedProp.confidence, 2) + evidenceText + "." + normalizedNote + " <strong>Decision:</strong> " + advice + ' <button class="btn btn-ghost" style="margin-left:0.75rem" onclick="testMaterialInDiagnostics()">Test in Diagnostics &rarr;</button>');
  setReadouts("mat-risk-readout", [
    { k: "Objective", v: selectedProp.score.toFixed(3) },
    { k: "Phase risk", v: fmt(selectedProp.p2Risk, 2) },
    { k: "Charge risk", v: fmt(selectedProp.chargeRisk, 2) },
    { k: "Defect score", v: fmt(selectedProp.defectScore, 2) },
    { k: "Site error", v: fmt((selectedProp.siteError || 0) * 100, 1) + "%" },
    { k: "Cost proxy", v: "$" + selectedProp.costKwh.toFixed(0) + "/kWh" }
  ]);
  var matConf = computeMaterialsConfidence(selectedProp);
  renderConfidence("mat-confidence", matConf.confidence, matConf.detail);
  recordRun("materials", {
    summary: "score " + selectedProp.score.toFixed(3) + ", stability " + selectedProp.stability.toFixed(2) + ", fade@500 " + (100 * selectedProp.fade500).toFixed(1) + "%",
    key_metric: selectedProp.Q0.toFixed(0) + " mAh/g",
    confidence: matConf.confidence
  });
}

function drawCompositionLandscape(cv, items, selected) {
  if (!cv || !items.length) return;
  var ctx = cv.getContext("2d"), W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  ctx.font = "10px JetBrains Mono, monospace";
  ctx.fillStyle = "#777";
  ctx.fillText("Composition score landscape: x=Mn, y=Na, height/color=objective", 12, 15);
  var grid = items.filter(function (it) { return it.comp.al === selected.al && it.comp.ti === selected.ti; });
  var minS = Math.min.apply(null, grid.map(function (g) { return g.prop.score; }));
  var maxS = Math.max.apply(null, grid.map(function (g) { return g.prop.score; }));
  var x0 = W * 0.18, y0 = H * 0.78, sx = W * 0.48, sy = H * 0.34;
  grid.sort(function (a, b) { return (a.comp.Na + a.comp.Mn) - (b.comp.Na + b.comp.Mn); });
  
  var idx = 0;
  var batchSize = 500;
  function drawBatch() {
    var end = Math.min(idx + batchSize, grid.length);
    for (; idx < end; idx++) {
      var it = grid[idx];
      var mn = (it.comp.Mn - 0.2) / 0.62;
      var na = (it.comp.Na - 0.84) / 0.28;
      var s = (it.prop.score - minS) / Math.max(1e-6, maxS - minS);
      var x = x0 + mn * sx + na * 34;
      var y = y0 - na * sy - s * 52;
      var color = "rgba(255," + Math.round(50 + 120 * s) + "," + Math.round(26 + 80 * (1 - s)) + "," + (0.38 + 0.45 * s) + ")";
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x + 16, y - 7);
      ctx.lineTo(x + 32, y);
      ctx.lineTo(x + 16, y + 7);
      ctx.closePath();
      ctx.fill();
    }
    
    if (idx < grid.length) {
      setTimeout(drawBatch, 0);
    } else {
      var selectedItem = { comp: selected, prop: scoreComposition(selected, 318.15, materialKnobs()) };
      var smn = (selectedItem.comp.Mn - 0.2) / 0.62;
      var sna = (selectedItem.comp.Na - 0.84) / 0.28;
      var ss = (selectedItem.prop.score - minS) / Math.max(1e-6, maxS - minS);
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x0 + smn * sx + sna * 34 + 16, y0 - sna * sy - ss * 52, 8, 0, Math.PI * 2);
      ctx.stroke();
    }
  }
  drawBatch();
}

// Recycling: shrinking-core leaching with Bayesian recovery priors.
function betaMean(a, b) { return a / (a + b); }
function shrinkingCoreConversion(k, tMin) {
  // Reaction-controlled: X = 1 - (1 - kt)^3. Clamp final conversion, not intermediate.
  var kt = k * tMin;
  if (kt >= 1.0) return 0.995;
  return clamp(1 - Math.pow(Math.max(1 - kt, 0), 3), 0, 0.995);
}
function diffusionCoreConversion(k, tMin) {
  // Product-layer diffusion control: 1 - 2X/3 - (1-X)^(2/3) = kt.
  var target = Math.max(0, k * tMin);
  var lo = 0, hi = 0.995;
  for (var i = 0; i < 36; i++) {
    var mid = (lo + hi) / 2;
    var lhs = 1 - 2 * mid / 3 - Math.pow(Math.max(1 - mid, 0), 2 / 3);
    if (lhs < target) lo = mid; else hi = mid;
  }
  return clamp((lo + hi) / 2, 0, 0.995);
}
function leachingTransportState(el, acid, tempC) {
  var R = 8.314;
  var T = tempC + 273.15;
  var tempFactor = Math.exp(-el.Ea / R * (1 / T - 1 / 353.15));
  var kSurface = el.k0 * Math.pow(Math.max(acid, 0.05), el.order) * tempFactor;
  var particleFactor = Math.pow(50 / Math.max(el.particle, 2), 0.35);
  var k = kSurface * particleFactor;
  // Dynamic diffusion criterion via Thiele-like modulus (phi > ~0.3 => diffusion influence)
  var rpM = Math.max(1e-7, el.particle * 0.5e-6);
  var dEff = 4.8e-13 * Math.exp(0.016 * (tempC - 80)) * Math.pow(Math.max(acid, 0.3) / 2.0, 0.20);
  dEff = Math.max(dEff, 1e-16);
  var phi = rpM * Math.sqrt(Math.max(kSurface, 1e-12) / dEff);
  var diffusionWeight = clamp((phi - 0.30) / 0.90, 0, 0.85);
  var rpCrit = 0.30 / Math.sqrt(Math.max(kSurface, 1e-12) / dEff);
  var particleCritUm = rpCrit * 2e6;
  return { k: k, phi: phi, diffusionWeight: diffusionWeight, particleCritUm: particleCritUm };
}
function recoveryForElement(el, acid, tempC, tMin, bayes) {
  var tr = leachingTransportState(el, acid, tempC);
  var surfaceX = shrinkingCoreConversion(tr.k, tMin);
  var diffusionX = diffusionCoreConversion(tr.k * 0.62, tMin);
  var x = surfaceX * (1 - tr.diffusionWeight) + diffusionX * tr.diffusionWeight;
  if (!bayes || !el.prior) return x;
  return clamp(0.75 * x + 0.25 * betaMean(el.prior[0], el.prior[1]), 0, 0.995);
}

function updateRecycling() {
  document.getElementById("bm-val").textContent = document.getElementById("bm-slider").value;
  document.getElementById("acid-val").textContent = parseFloat(document.getElementById("acid-slider").value).toFixed(1);
  document.getElementById("leach-val").textContent = document.getElementById("leach-slider").value + " C";
}

function recyclingKnobs() {
  return {
    time: Math.max(5, num("rec-time", 120)),
    particle: Math.max(2, num("rec-particle", 50)),
    acidOrder: Math.max(0.05, num("rec-acid-order", 0.95)),
    eaMn: Math.max(5000, num("rec-ea-mn", 27000)),
    acidCost: Math.max(0, num("rec-acid-cost", 8.5)),
    energyCost: Math.max(0, num("rec-energy-cost", 8.0)),
    processingCost: Math.max(0, num("rec-processing-cost", 150)),
    metalPrice: Math.max(1, num("rec-metal-price", 620))
  };
}

function recyclingScenarioSeed(mass, acid, temp, cfg) {
  var text = [mass, acid, temp, cfg.time, cfg.particle, cfg.acidOrder, cfg.eaMn].map(function (x) {
    return Number(x).toFixed(4);
  }).join("|");
  var h = 2166136261;
  for (var i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function runRecycling() {
  var mass = parseFloat(document.getElementById("bm-slider").value);
  var acid = parseFloat(document.getElementById("acid-slider").value);
  var temp = parseFloat(document.getElementById("leach-slider").value);
  var mc = document.getElementById("sw-mc").checked;
  var bay = document.getElementById("sw-bayes").checked;
  var cfg = recyclingKnobs();
  var tFinal = cfg.time;
  var elements = [
    { n: "Mn", wt: 0.22, k0: 0.0038, Ea: cfg.eaMn, order: cfg.acidOrder, particle: cfg.particle, prior: [8.8, 1.2] },
    { n: "Fe", wt: 0.11, k0: 0.0029, Ea: 30000, order: Math.max(0.1, cfg.acidOrder - 0.10), particle: cfg.particle * 1.1, prior: [7.2, 2.8] },
    { n: "Na", wt: 0.05, k0: 0.0062, Ea: 19000, order: Math.max(0.1, cfg.acidOrder - 0.35), particle: cfg.particle * 0.9, prior: [6.5, 3.5] },
    { n: "Al", wt: 0.04, k0: 0.0011, Ea: 36000, order: cfg.acidOrder + 0.10, particle: cfg.particle * 1.3 },
    { n: "Cu", wt: 0.015, k0: 0.0007, Ea: 34000, order: Math.max(0.1, cfg.acidOrder - 0.15), particle: cfg.particle * 1.4 }
  ];
  var bars = document.getElementById("recycle-bars");
  bars.innerHTML = "";
  var targets = elements.map(function (el) { return recoveryForElement(el, acid, temp, tFinal, bay); });
  var mnTransport = leachingTransportState(elements[0], acid, temp);
  elements.forEach(function (el, i) {
    var row = document.createElement("div");
    row.className = "recovery-row";
    row.innerHTML = '<div class="bar-head"><span>' + el.n + '</span><span id="rv-' + el.n + '">0.0%</span></div><div class="progress-bar"><div class="fill" id="rb-' + el.n + '" style="width:0%"></div></div>';
    bars.appendChild(row);
    setTimeout(function () {
      document.getElementById("rb-" + el.n).style.width = (targets[i] * 100).toFixed(1) + "%";
      document.getElementById("rv-" + el.n).textContent = (targets[i] * 100).toFixed(1) + "%";
    }, 80 + i * 50);
  });
  var timeline = [];
  for (var t = 0; t <= tFinal; t += 4) {
    timeline.push({
      t: t,
      Mn: recoveryForElement(elements[0], acid, temp, t, bay) * 100,
      Fe: recoveryForElement(elements[1], acid, temp, t, bay) * 100,
      Na: recoveryForElement(elements[2], acid, temp, t, bay) * 100
    });
  }
  drawMultiLine(makeCanvas("recycle-chart"), [
    { name: "Mn", values: timeline.map(function (x) { return x.Mn; }), color: "#ff1a1a", glow: true },
    { name: "Fe", values: timeline.map(function (x) { return x.Fe; }), color: "#ff9f1a" },
    { name: "Na", values: timeline.map(function (x) { return x.Na; }), color: "#38bdf8" }
  ], { yMin: 0, yMax: 100, xMax: tFinal, title: "Shrinking-core conversion over leach time", legend: true, yDigits: 0 });

  var totalRecovered = 0;
  targets.forEach(function (r, i) { totalRecovered += mass * elements[i].wt * r; });
  var mcTotals = [];
  if (mc) {
    var rng = mulberry32(recyclingScenarioSeed(mass, acid, temp, cfg));
    for (var s = 0; s < RECYCLING_MC_SAMPLES; s++) {
      var tot = 0;
      elements.forEach(function (el, i) {
        var feedNoise = clamp(1 + seededGaussian(rng) * 0.08, 0.75, 1.25);
        var assayNoise = clamp(1 + seededGaussian(rng) * 0.025, 0.92, 1.08);
        var particleNoise = clamp(1 + seededGaussian(rng) * 0.18, 0.55, 1.65);
        var elSample = Object.assign({}, el, { particle: Math.max(2, el.particle * particleNoise) });
        var sampleRecovery = recoveryForElement(elSample, acid, temp, tFinal, bay);
        tot += mass * el.wt * feedNoise * sampleRecovery * assayNoise;
      });
      mcTotals.push(tot);
    }
    mcTotals.sort(function (a, b) { return a - b; });
  }
  var lo = mc ? mcTotals[Math.floor(mcTotals.length * 0.05)] : totalRecovered;
  var hi = mc ? mcTotals[Math.floor(mcTotals.length * 0.95)] : totalRecovered;
  var acidKg = acid * 0.098 * mass;
  var heatKwh = Math.max(0, temp - 25) * mass * 0.00116;
  var cost = acidKg * cfg.acidCost + heatKwh * cfg.energyCost + mass * cfg.processingCost;
  var impurityPenalty = clamp(targets[3] * 0.28 + targets[4] * 0.36, 0, 0.8);
  var productPurity = clamp(0.94 - impurityPenalty * 0.18 + (targets[0] + targets[1] + targets[2]) * 0.012, 0.70, 0.98);
  var log = document.getElementById("recycle-log");
  log.innerHTML = '<div class="cmd">$ recycling --shrinking-core --mass=' + mass + "kg --acid=" + acid + "M --temp=" + temp + 'C</div>';
  log.innerHTML += '<div class="info">ODE: 1 - (1-X)^(1/3) = k(C_acid,T,Rp)t, with Arrhenius temperature scaling and Thiele-based diffusion blending.</div>';
  if (mc) log.innerHTML += '<div class="info">Monte Carlo: ' + RECYCLING_MC_SAMPLES + ' feedstock, assay, and particle-size samples with deterministic scenario seed.</div>';
  if (mnTransport.phi > 0.30) {
    log.innerHTML += '<div class="warn">Diffusion-influenced regime: phi=' + mnTransport.phi.toFixed(2) + " (>0.30). Equivalent transition size at this acid/temp is ~" + mnTransport.particleCritUm.toFixed(1) + " um.</div>";
  } else {
    log.innerHTML += '<div class="info">Surface-control regime: phi=' + mnTransport.phi.toFixed(2) + " (<=0.30). Equivalent transition size ~" + mnTransport.particleCritUm.toFixed(1) + " um.</div>";
  }
  if (bay) log.innerHTML += '<div class="info">Bayesian priors: Mn Beta(8.8,1.2), Fe Beta(7.2,2.8), Na Beta(6.5,3.5).</div>';
  log.innerHTML += '<div class="ok">Recovered metals: ' + totalRecovered.toFixed(1) + "kg, 90% interval " + lo.toFixed(1) + "-" + hi.toFixed(1) + "kg</div>";
  log.innerHTML += '<div class="info">Process cost estimate: INR ' + cost.toFixed(0) + " per batch; product purity proxy " + (productPurity * 100).toFixed(1) + "%</div>";
  var marginProxy = totalRecovered * cfg.metalPrice * productPurity - cost;
  var decision = marginProxy > 0 && targets[0] > 0.82 && productPurity > 0.86
    ? "Run this recipe as a pilot batch; Mn recovery and economics are inside the current gate."
    : "Do not run as-is; adjust acid/time/particle size or improve impurity control before pilot scale.";
  window.__kfRecycling = { totalRecovered: totalRecovered, lo: lo, hi: hi, purity: productPurity, cost: cost, marginProxy: marginProxy, elements: elements, targets: targets, mass: mass, acid: acid, tempC: temp, leachMin: cfg.time, particleUm: cfg.particle, transport: { phi: mnTransport.phi, particleCritUm: mnTransport.particleCritUm, diffusionWeight: mnTransport.diffusionWeight }, economics: { acidCost: cfg.acidCost, energyCost: cfg.energyCost, processingCost: cfg.processingCost, metalPrice: cfg.metalPrice } };
  setHtml("recycle-decision", "<strong>Recipe output:</strong> recovered " + totalRecovered.toFixed(1) + " kg, Mn " + (targets[0] * 100).toFixed(1) + "%, purity proxy " + (productPurity * 100).toFixed(1) + "%, cost INR " + cost.toFixed(0) + ". <strong>Decision:</strong> " + decision);
  setReadouts("recycle-readout", [
    { k: "Recovered", v: totalRecovered.toFixed(1) + " kg" },
    { k: "90% interval", v: lo.toFixed(1) + "-" + hi.toFixed(1) + " kg" },
    { k: "Purity proxy", v: (productPurity * 100).toFixed(1) + "%" },
    { k: "Margin proxy", v: "INR " + marginProxy.toFixed(0) },
    { k: "Particle regime", v: mnTransport.phi > 0.30 ? "mixed diffusion (phi>0.3)" : "surface control (phi<=0.3)" },
    { k: "Metal price", v: "INR " + cfg.metalPrice.toFixed(0) + "/kg" }
  ]);
  var recConf = computeRecyclingConfidence(window.__kfRecycling);
  renderConfidence("recycle-confidence", recConf.confidence, recConf.detail);
  recordRun("recycling", {
    summary: "recovered " + totalRecovered.toFixed(1) + " kg, purity " + (productPurity * 100).toFixed(1) + "%, margin INR " + compactNumber(marginProxy, 0),
    key_metric: "Mn " + (targets[0] * 100).toFixed(1) + "%",
    confidence: recConf.confidence
  });
}

function setAssistantOpen(open) {
  var dock = document.getElementById("assistant-dock");
  var input = document.getElementById("assistant-input");
  if (!dock) return;
  dock.classList.toggle("open", !!open);
  if (open && input) setTimeout(function () { input.focus(); }, 120);
}

// Simple markdown renderer for chatbot responses
function renderMd(text) {
  if (!text) return "";
  var html = escapeHtml(text);
  // Bold: **text** or __text__
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/__(.+?)__/g, '<strong>$1</strong>');
  // Italic fallback without lookbehind (works on older WebKit/Safari engines)
  html = html.replace(/(^|[\s(>])\*([^*\n][^*\n]*?)\*(?=[\s).,!?;:]|$)/gm, '$1<em>$2</em>');
  // Inline code: `text`
  html = html.replace(/`([^`]+)`/g, '<code style="background:rgba(255,26,26,0.12);padding:1px 4px;border-radius:3px;font-size:0.9em">$1</code>');
  // Bullet points: lines starting with - or *
  html = html.replace(/^(\s*)[\-\*]\s+(.+)/gm, '$1<span style="color:#ff5c5c">•</span> $2');
  // Numbered lists: lines starting with 1. 2. etc
  html = html.replace(/^(\s*)\d+\.\s+(.+)/gm, '$1<span style="color:#ff5c5c">▸</span> $2');
  // Newlines
  html = html.replace(/\n/g, '<br>');
  return html;
}

function assistantMessage(kind, text, meta) {
  var body = document.getElementById("assistant-body");
  if (!body) return null;
  var msg = document.createElement("div");
  msg.className = "assistant-msg " + kind;
  // Use markdown rendering for bot messages, plain for user messages
  if (kind.indexOf("user") >= 0) {
    msg.innerHTML = escapeHtml(text || "").replace(/\n/g, "<br>");
  } else {
    msg.innerHTML = renderMd(text || "");
  }
  if (meta) {
    var small = document.createElement("div");
    small.className = "assistant-meta";
    small.textContent = meta;
    msg.appendChild(small);
  }
  body.appendChild(msg);
  body.scrollTop = body.scrollHeight;
  return msg;
}

function updateAssistantFoot(data) {
  var badge = document.getElementById("assistant-source-badge");
  var modelEl = document.getElementById("assistant-model-label");
  var memEl = document.getElementById("assistant-memory-label");
  if (!badge || !modelEl || !memEl) return;
  if (!data) {
    badge.innerHTML = '<span class="source-badge local">offline</span>';
    modelEl.textContent = "no connection";
    memEl.textContent = "memory off";
    return;
  }
  var isCloud = data.source === "openrouter";
  badge.innerHTML = '<span class="source-badge ' + (isCloud ? "cloud" : "local") + '">' + (isCloud ? "cloud" : "local") + '</span>';
  modelEl.textContent = data.model && data.model !== "none" ? data.model : (isCloud ? "OpenRouter" : "rule engine");
  memEl.textContent = data.memory && data.memory !== "off" ? "context on" : "memory off";
  if (data.setup_required) memEl.textContent += " · needs API key";
}

function rememberAssistantTurn(role, text) {
  var clean = String(text || "").trim();
  if (!clean) return;
  kfAssistantHistory.push({ role: role, text: clean.slice(0, 900), at: Date.now() });
  if (kfAssistantHistory.length > 12) {
    kfAssistantHistory.splice(0, kfAssistantHistory.length - 12);
  }
  window.__kfAssistantHistory = kfAssistantHistory;
}

function textOf(id) {
  var el = document.getElementById(id);
  return el ? el.textContent.trim() : "";
}

function activeSection() {
  var active = document.querySelector(".section.active");
  return active ? active.id.replace(/^sec-/, "") : "general";
}

function collectAssistantState() {
  var cellDetails = Array.from(document.querySelectorAll("#bms-grid .cell-tile")).map(function (el) {
    var nameEl = el.querySelector(".cell-name");
    var riskEl = el.querySelector(".cell-risk");
    var tempEl = el.querySelector(".cell-temp");
    return {
      cell: el.dataset.cell || (nameEl ? nameEl.textContent.trim() : ""),
      risk: parseFloat(el.dataset.risk || (riskEl ? riskEl.textContent : "")),
      temp_C: parseFloat(el.dataset.temp || (tempEl ? tempEl.textContent : "")),
      fault: el.dataset.fault === "1",
      hot: el.dataset.hot === "1"
    };
  }).filter(function (c) {
    return c.cell && Number.isFinite(c.risk);
  });
  var risks = cellDetails.map(function (c) {
    return c.risk;
  }).filter(Number.isFinite);
  var bmsMeta = window.__kfBms || {};
  var diag = window.__kfDiag || {};
  var mat = window.__kfMaterials || {};
  var rec = window.__kfRecycling || {};
  var byod = window.__kfByod || {};
  var byodPred = byod.predictions || {};
  var byodModels = byodPred.model_outputs || {};
  var decisions = buildDecisionItems();
  return {
    section: activeSection(),
    diagnostics: {
      temperature_C: parseFloat(document.getElementById("temp-slider").value),
      c_rate: parseFloat(document.getElementById("crate-slider").value),
      cycles: parseInt(document.getElementById("cycles-slider").value, 10),
      na: num("diag-na", 1.02),
      mn: num("diag-mn", 0.52),
      fe: num("diag-fe", 0.43),
      dopant_frac: num("diag-dop", 0.05),
      dopant_type: (document.getElementById("diag-dopant-type") || {}).value || "Al",
      eol_capacity: textOf("diag-eol"),
      fade: textOf("diag-fade"),
      voltage_end: diag.voltage ? diag.voltage[diag.voltage.length - 1] : null,
      knee: textOf("diag-knee"),
      rul80: textOf("diag-rul"),
      sei_loss: diag.sei ? diag.sei[diag.sei.length - 1] : null,
      p2_loss: diag.p2 ? diag.p2[diag.p2.length - 1] : null,
      jt_loss: diag.jt ? diag.jt[diag.jt.length - 1] : null,
      rate_loss: diag.rate ? diag.rate[diag.rate.length - 1] : null,
      residual_loss: diag.residual ? diag.residual[diag.residual.length - 1] : null,
      decision: textOf("diag-decision")
    },
    bms: {
      cells: parseInt(document.getElementById("pack-slider").value, 10),
      duration_seconds: parseInt(document.getElementById("dur-slider").value, 10),
      ambient_C: num("bms-ambient", 45),
      seed: bmsMeta.seed || num("bms-seed", 42),
      inject_fault: document.getElementById("sw-fault").checked,
      max_risk: risks.length ? Math.max.apply(null, risks) : null,
      threshold: Number.isFinite(bmsMeta.threshold) ? bmsMeta.threshold : num("bms-risk-thresh", 0.42),
      topology: bmsMeta.topology || bmsTopologyInfo(parseInt(document.getElementById("pack-slider").value, 10) || 8),
      loss_ratio: bmsMeta.lossRatio || num("bms-loss-ratio", 6.0),
      fault_cell: Number.isFinite(bmsMeta.faultCell) && bmsMeta.faultCell >= 0 ? "C" + bmsMeta.faultCell : "none",
      cell_details: cellDetails,
      decision: textOf("bms-decision")
    },
    materials: {
      na: parseFloat(document.getElementById("na-slider").value),
      mn: parseFloat(document.getElementById("mn-slider").value),
      fe: parseFloat(document.getElementById("fe-slider").value),
      capacity: textOf("mat-cap"),
      voltage: textOf("mat-volt"),
      stability: textOf("mat-stab"),
      jt_index: textOf("mat-jt"),
      fade500_pct: mat.fade500 ? mat.fade500 * 100 : null,
      oxygen_risk: mat.oxygenRisk,
      charge_risk: mat.chargeRisk,
      cost_kwh: mat.costKwh,
      score: mat.score,
      decision: textOf("mat-decision")
    },
    recycling: {
      mass_kg: parseFloat(document.getElementById("bm-slider").value),
      acid_molarity: parseFloat(document.getElementById("acid-slider").value),
      temperature_C: parseFloat(document.getElementById("leach-slider").value),
      recovered_kg: rec.totalRecovered,
      interval_kg: Number.isFinite(rec.lo) ? rec.lo.toFixed(1) + "-" + rec.hi.toFixed(1) : "",
      purity_proxy: rec.purity,
      margin_proxy_inr: rec.marginProxy,
      decision: textOf("recycle-decision")
    },
    upload: {
      filename: byod.filename,
      rows_read: byod.rows_read,
      schema_score: byod.schema ? byod.schema.score : null,
      features_present: byod.feature_mask ? byod.feature_mask.filter(function (x) { return !!x; }).length : 0,
      soh: byodPred.soh,
      confidence: byodPred.confidence,
      cycle_80: byodPred.cycle_80_estimate,
      inference_mode: byodPred.inference_mode,
      checkpoint_status: byod.checkpoint_inference ? byod.checkpoint_inference.status : null,
      model_outputs: {
        M1_CathodeUDE: byodModels.M1_CathodeUDE || null,
        M11_ElectrolyteHealth: byodModels.M11_ElectrolyteHealth || null,
        M12_Replenishability: byodModels.M12_Replenishability || null,
        M13_ChemIdentifier: byodModels.M13_ChemIdentifier || null,
        M14_FormationProtocol: byodModels.M14_FormationProtocol || null
      },
      warnings: byod.warnings || []
    },
    decisions: {
      open_actions: decisions.filter(function (x) { return x.severity === "critical" || x.severity === "warn"; }).length,
      critical_actions: decisions.filter(function (x) { return x.severity === "critical"; }).length,
      latest_run: latestRunLabel(),
      items: decisions.slice(0, 6).map(function (x) {
        return {
          severity: x.severity,
          source: x.source,
          owner: x.owner,
          evidence: x.evidence,
          action: x.action,
          next: x.next,
          confidence: x.confidence
        };
      })
    },
    conversation: (window.__kfAssistantHistory || []).slice(-8)
  };
}

function askAssistant(question) {
  var clean = String(question || "").trim();
  if (!clean) return;
  setAssistantOpen(true);
  assistantMessage("user", clean);
  var pending = assistantMessage("bot loading", "Thinking...");
  var form = document.getElementById("assistant-form");
  var btn = form ? form.querySelector("button") : null;
  if (btn) btn.disabled = true;
  var assistantState = collectAssistantState();
  rememberAssistantTurn("user", clean);

  fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question: clean, section: activeSection(), state: assistantState })
  })
    .then(function (res) {
      if (!res.ok) throw new Error("Assistant endpoint returned " + res.status);
      return res.json();
    })
    .then(function (data) {
      var answer = data.answer || "I could not form an answer from the available project context.";
      if (pending) {
        pending.className = "assistant-msg bot";
        pending.innerHTML = renderMd(answer);
        if (data.source === "openrouter" && data.context && data.context.length) {
          var meta = document.createElement("div");
          meta.className = "assistant-meta";
          meta.textContent = "Context: " + data.context.slice(0, 3).join(", ");
          pending.appendChild(meta);
        }
      }
      rememberAssistantTurn("assistant", answer);
      updateAssistantFoot(data);
      if (data.warning) assistantMessage("bot note", data.warning);
    })
    .catch(function (err) {
      if (pending) {
        pending.className = "assistant-msg bot";
        pending.textContent = "The assistant endpoint is offline. Run the FastAPI server and try again.";
      }
      updateAssistantFoot(null);
      showToast("Assistant request failed: " + err.message, "error");
      console.warn(err);
    })
    .finally(function () {
      if (btn) btn.disabled = false;
      var input = document.getElementById("assistant-input");
      if (input) input.focus();
    });
}

function initAssistant() {
  var launch = document.getElementById("assistant-launch");
  var close = document.getElementById("assistant-close");
  var form = document.getElementById("assistant-form");
  var input = document.getElementById("assistant-input");
  if (launch) launch.addEventListener("click", function () {
    var dock = document.getElementById("assistant-dock");
    setAssistantOpen(!(dock && dock.classList.contains("open")));
  });
  if (close) close.addEventListener("click", function () { setAssistantOpen(false); });
  document.querySelectorAll(".assistant-prompts button").forEach(function (btn) {
    btn.addEventListener("click", function () {
      askAssistant(btn.getAttribute("data-q"));
    });
  });
  if (form) {
    form.addEventListener("submit", function (event) {
      event.preventDefault();
      var q = input ? input.value : "";
      if (input) input.value = "";
      askAssistant(q);
    });
  }
}

function initArchitecture() {
  var cards = [
    { t: "Universal Differential Equations", d: "Physics ODE terms for SEI, P2-O2, Jahn-Teller coupling, Na desolvation, and rate stress are explicit; residuals are bounded correction terms." },
    { t: "Pack Thermal Graph", d: "Pack monitoring uses topology-aware thermal coupling, EIS drift, and multi-scale lookback windows." },
    { t: "qNEHVI-style Bayesian Screening", d: "Composition candidates are scored on capacity, fade, life, and cost, then filtered by a noisy hypervolume-improvement proxy; full BoTorch qNEHVI is reserved for the GPU training path." },
    { t: "Bayesian Recycling Loop", d: "Recovery priors are beta distributions and process conversion follows shrinking-core leaching kinetics." },
    { t: "Uncertainty Propagation", d: "Composition, cell, pack, and recycling predictions carry uncertainty bounds instead of single unqualified numbers." },
    { t: "Evidence Registry", d: "Prediction claims are tied to local datasets, validation gates, and model provenance." },
    { t: "Na-ion Phase Physics", d: "P2-O2 structural transition, Mn3+ Jahn-Teller distortion, and Na+ desolvation are first-class physics terms." },
    { t: "EIS Feature Extraction", d: "Randles-circuit features R_ct, R_SEI, and Warburg coefficient feed the pack risk model." },
    { t: "Regional Climate Model", d: "Operating conditions can be conditioned on local hot-weather temperature and humidity profiles." },
    { t: "BYOD Feature Masking", d: "Uploads produce a feature vector plus availability mask so missing EIS or early-cycle fields are visible instead of silently zero-filled." }
  ];
  var grid = document.getElementById("arch-grid");
  if (grid && !grid.children.length) {
    cards.forEach(function (c) {
      var d = document.createElement("div");
      d.className = "panel";
      d.innerHTML = '<div class="panel-title"><span class="indicator"></span> ' + c.t + '</div><div class="panel-desc">' + c.d + "</div>";
      grid.appendChild(d);
    });
  }
  var models = [
    ["Cathode UDE", "UDE physics plus residual", "Runnable", "371K cycles", "~150K"],
    ["SOH Estimator", "MLP regression", "Runnable", "371K cycles", "~45K"],
    ["Cycle Life", "Classifier", "Runnable", "371K cycles", "~35K"],
    ["Fade Rate", "MLP regression", "Runnable", "371K cycles", "~20K"],
    ["BMS Pack Graph", "Thermal graph risk", "Label Gate", "pack ODE cases", "~80K"],
    ["RUL Predictor", "MLP regression", "Runnable", "371K cycles", "~50K"],
    ["Anomaly AE", "Autoencoder", "Runnable", "371K cycles", "~30K"],
    ["Joint SOH+RUL", "Multi-task MLP", "Runnable", "371K cycles", "~120K"],
    ["Knee Detector", "Conv1D + FC", "Runnable", "371K cycles", "~60K"],
    ["Chem Ranker", "Embedding MLP", "Runnable", "371K cycles", "~15K"],
    ["Electrolyte Health", "EIS diagnostic head", "Runnable", "NASA EIS", "8.8K"],
    ["Replenishability", "Recovery preview", "Research Preview", "capacity windows", "9.6K"],
    ["Chem Identifier", "Early-cycle classifier", "Runnable", "dQ/dV + features", "80.6K"],
    ["Formation Protocol", "Formation quality", "Research Preview", "first cycles", "19.7K"]
  ];
  var tb = document.querySelector("#model-table tbody");
  if (tb && !tb.children.length) {
    models.forEach(function (m) {
      var tr = document.createElement("tr");
      var tagClass = (m[2] === "Label Gate" || m[2] === "Research Preview") ? "warn" : "ok";
      tr.innerHTML = '<td style="font-weight:600">' + m[0] + "</td><td>" + m[1] + '</td><td><span class="tag ' + tagClass + '">' + m[2] + "</span></td><td>" + m[3] + '</td><td style="font-family:JetBrains Mono;font-size:11px">' + m[4] + "</td>";
      tb.appendChild(tr);
    });
  }
  renderHardpointTable();
}

function modelStatusLabel(status, checkpointPresent) {
  status = String(status || "");
  if (/preview/i.test(status)) return "Preview";
  if (/label[_ ]?gate/i.test(status)) return "Label Gate";
  if (!checkpointPresent) return "Simulation";
  if (/ml[_ ]?surrogate/i.test(status)) return "ML Surrogate";
  if (status === "trained_checkpoint") return "Trained";
  return String(status || "Unknown").replace(/_/g, " ");
}

function modelStatusClass(label) {
  if (/simulation/i.test(label)) return "sim";
  return /missing|preview|gate/i.test(label) ? "warn" : "ok";
}

function renderModelRegistryFromApi(data) {
  var tb = document.querySelector("#model-table tbody");
  if (!tb || !data || !Array.isArray(data.models)) return;
  tb.innerHTML = "";
  data.models.forEach(function (m) {
    var tr = document.createElement("tr");
    var modelId = String(m.id || m.name || m.model || "").trim();
    var checkpointPresent = m.checkpoint_present != null
      ? !!m.checkpoint_present
      : !!(m.checkpoint_file || m.checkpoint);
    var label = modelStatusLabel(m.status || (checkpointPresent ? "trained_checkpoint" : "simulation"), checkpointPresent);
    var checkpoint = checkpointPresent
      ? (m.checkpoint_source_zip || m.checkpoint_file || m.checkpoint || "checkpoint present")
      : "not found";
    tr.innerHTML = '<td style="font-weight:600">' + escapeHtml(modelId) +
      "</td><td>" + escapeHtml(m.type || m.version || "") +
      '</td><td><span class="tag ' + modelStatusClass(label) + '">' + escapeHtml(label) +
      "</span></td><td>" + escapeHtml(checkpoint) +
      '</td><td style="font-family:JetBrains Mono;font-size:11px">' + escapeHtml(String(m.params || m.param_count || "--")) +
      "</td>";
    tb.appendChild(tr);
  });
  renderHardpointTable(data.models);
}

function renderHardpointTable(models) {
  var body = document.getElementById("hardpoint-table-body");
  if (!body) return;
  var orderedIds = [];
  if (Array.isArray(models)) {
    models.forEach(function (m) {
      var id = String(m.id || m.name || m.model || "").trim();
      if (id && HARDPOINT_MAP[id] && orderedIds.indexOf(id) < 0) orderedIds.push(id);
    });
  }
  if (!orderedIds.length) orderedIds = Object.keys(HARDPOINT_MAP);
  body.innerHTML = orderedIds.map(function (id) {
    var meta = HARDPOINT_MAP[id] || {};
    var fromApi = Array.isArray(models) ? models.find(function (m) { return String(m.id || m.name || m.model || "").trim() === id; }) : null;
    var fallback = fromApi && (fromApi.without_checkpoint || fromApi.fallback_mode) ? (fromApi.without_checkpoint || fromApi.fallback_mode) : (meta.fallback || "rules-only fallback");
    return "<tr><td>" + escapeHtml(id) + "</td><td>" + escapeHtml(meta.hardpoint || "--") + "</td><td>" + escapeHtml(fallback) + "</td></tr>";
  }).join("");
}

function renderCheckpointProvenance(data) {
  var body = document.getElementById("checkpoint-provenance-body");
  if (!body) return;
  var models = Array.isArray(data && data.models) ? data.models : [];
  var sourceCounts = {};
  models.forEach(function (m) {
    var src = m.checkpoint_source_zip || (m.checkpoint_present ? "loose checkpoint" : "missing");
    sourceCounts[src] = (sourceCounts[src] || 0) + 1;
  });
  var sources = Array.isArray(data && data.source_zips) ? data.source_zips : [];
  var rows = sources.map(function (src) {
    var name = src.name || "unknown";
    return "<tr><td>" + escapeHtml(name) + "</td><td>" + escapeHtml(String(src.artifacts || sourceCounts[name] || 0)) +
      "</td><td>" + escapeHtml(data.checkpoint_manifest && data.checkpoint_manifest.generated_at ? data.checkpoint_manifest.generated_at : "not generated") + "</td></tr>";
  });
  if (!rows.length && models.length) {
    rows = Object.keys(sourceCounts).sort().map(function (name) {
      return "<tr><td>" + escapeHtml(name) + "</td><td>" + escapeHtml(String(sourceCounts[name])) + "</td><td>runtime scan</td></tr>";
    });
  }
  body.innerHTML = rows.length ? rows.join("") : '<tr><td colspan="3">No checkpoint manifest found. Run scripts/extract_checkpoints.py.</td></tr>';
}

function fetchModelRegistry() {
  fetch("/api/models")
    .then(function (res) {
      if (!res.ok) throw new Error("model registry HTTP " + res.status);
      return res.json();
    })
    .then(function (data) {
      renderModelRegistryFromApi(data);
      renderHardpointTable(data.models);
      renderCheckpointProvenance(data);
    })
    .catch(function (err) {
      console.warn("Model registry fetch failed", err);
      renderHardpointTable();
    });
}

function initAPIEndpoints() {
  var eps = [
    { m: "POST", p: "/api/predict/degradation", d: "Na-ion UDE capacity fade prediction with mechanism contributions." },
    { m: "POST", p: "/api/simulate/bms", d: "Thermal graph pack simulation with EIS-informed cell risk." },
    { m: "POST", p: "/api/optimize/recycling", d: "Shrinking-core leaching plus Bayesian recovery priors." },
    { m: "POST", p: "/api/screen/cathode", d: "Composition-property scoring with Pareto candidates." },
    { m: "POST", p: "/api/byod/analyze", d: "Upload cycler data, fingerprint schema, extract tier-1 features, dQ/dV, M1-M14 rules, and checkpoint outputs when available." },
    { m: "POST", p: "/api/byod/analyze-full", d: "Strict upload analysis that requires trained PyTorch checkpoint inference." },
    { m: "POST", p: "/api/byod/compare", d: "Compare two cycler files for A/B cell, protocol, or formation studies." },
    { m: "POST", p: "/api/byod/batch", d: "Analyze a ZIP of cell files and return batch variance plus outlier flags." },
    { m: "POST", p: "/api/byod/webhook/cycle", d: "Cycle-level cycler integration hook returning continue, investigate, or stop." },
    { m: "GET", p: "/api/byod/session/{session_id}", d: "Return a canonical JSON view of an in-memory BYOD session." },
    { m: "GET", p: "/api/byod/session/{session_id}/export-json", d: "Export canonical BYOD JSON for downstream notebooks, QC systems, or archives." },
    { m: "GET", p: "/api/byod/session/{session_id}/export", d: "Export cycle summaries, extracted features, and M1-M14 readouts from an in-memory BYOD session." },
    { m: "POST", p: "/api/chat", d: "OpenRouter assistant with deterministic fallback and browser-supplied recent-turn context." },
    { m: "POST", p: "/predict/lifetime", d: "Compatibility alias for degradation prediction." },
    { m: "POST", p: "/alert/bms", d: "Compatibility alias for BMS alert output." },
    { m: "POST", p: "/optimize/recycling", d: "Compatibility alias for recycling optimization." },
    { m: "POST", p: "/cathode/screen", d: "Compatibility alias for cathode screening." },
    { m: "GET", p: "/api/models", d: "Honest M1-M14 registry with checkpoint presence, status gates, and parameter-count basis." },
    { m: "GET", p: "/health", d: "Server status and timestamp. No auth required." }
  ];
  var ct = document.getElementById("api-endpoints");
  if (!ct || ct.children.length) return;
  eps.forEach(function (ep) {
    var d = document.createElement("div");
    d.className = "switch-row";
    d.style.cursor = "pointer";
    d.onclick = function () { showAPI(ep); };
    d.innerHTML = '<div class="switch-label"><span class="name"><span class="tag" style="margin-right:6px">' + ep.m + "</span> " + ep.p + '</span><span class="desc">' + ep.d + "</span></div>";
    ct.appendChild(d);
  });
}

function showAPI(ep) {
  var examples = {
    "/api/predict/degradation": '{\n  "result": {\n    "capacity_end": 0.912,\n    "fade_pct": 0.088,\n    "mechanisms": {"p2o2": 0.041, "jt": 0.008, "sei_desolv": 0.024}\n  },\n  "provenance": {"model": "Na-ion UDE physics mirror"}\n}',
    "/api/simulate/bms": '{\n  "cells": 8,\n  "topology": {"label": "4S2P current-sharing module"},\n  "loss_ratio": 6.0,\n  "thermal_equation": "Cth dT/dt = q + sum(kij(Tj-Ti)) - h(T-Ta)",\n  "max_risk": 0.64,\n  "alerts": [{"t": 74, "cell": 3, "risk": 0.51}]\n}',
    "/api/optimize/recycling": '{\n  "recoveries": {"Mn": {"recovery_rate": 0.91}, "Fe": {"recovery_rate": 0.78}, "Na": {"recovery_rate": 0.82}},\n  "kinetics": "shrinking-core leaching with mixed diffusion correction below 25 um",\n  "economics": {"metal_price_inr_kg": 620},\n  "uncertainty": {"basis": "Monte Carlo feedstock, assay, and particle size"}\n}',
    "/api/byod/analyze": '{\n  "session_id": "uuid",\n  "schema": {"format": "neware", "score": 0.82},\n  "features": {"early_coulombic_efficiency": 0.992},\n  "feature_mask": [1,1,1],\n  "predictions": {"soh": 0.94, "confidence": 0.71, "inference_mode": "checkpoint_plus_rules"}\n}',
    "/api/byod/analyze-full": '{\n  "session_id": "uuid",\n  "checkpoint_inference": {"status": "ok"},\n  "predictions": {"inference_mode": "checkpoint_plus_rules"}\n}',
    "/api/byod/compare": '{\n  "comparison": {"delta": {"soh": 0.012}, "decision": "file_b looks healthier on SOH"},\n  "file_a_session_id": "uuid",\n  "file_b_session_id": "uuid"\n}',
    "/api/byod/batch": '{\n  "files_analyzed": 12,\n  "stats": {"soh_mean": 0.941, "soh_std": 0.018},\n  "outliers": [{"filename": "cell_07.csv", "reasons": ["SOH outside 2 sigma"]}]\n}',
    "/api/byod/webhook/cycle": '{\n  "status": "accepted",\n  "result": {"cell_id": "A17", "cycle": 42, "recommendation": "continue", "soh": 0.943}\n}',
    "/api/byod/session/{session_id}": '{\n  "format": "kineticsforge_canonical_v1",\n  "features": {"capacity_fade_rate_ah_per_cycle": -0.00014},\n  "predictions": {"inference_mode": "checkpoint_plus_rules"}\n}',
    "/api/byod/session/{session_id}/export-json": '{\n  "format": "kineticsforge_canonical_v1",\n  "cycle_summary": [],\n  "checkpoint_inference": {"status": "ok"}\n}',
    "/api/models": '{\n  "total": 14,\n  "checkpoint_manifest": {"files": 57, "generated_at": "..."},\n  "models": [{"id": "M11_ElectrolyteHealth", "checkpoint_present": true, "hardpoint": "EIS/plating diagnostic head", "without_checkpoint": "tier-1 heuristic fallback", "checkpoint_source_zip": "results (25).zip"}]\n}',
    "/api/chat": '{\n  "answer": "C6 is highlighted because its risk crossed the action threshold...",\n  "source": "openrouter",\n  "memory": "browser_recent_turns"\n}'
  };
  var r = examples[ep.p] || '{"status":"ok","endpoint":"' + ep.p + '"}';
  var c = document.getElementById("api-response");
  c.innerHTML = '<div class="cmd">$ curl -X ' + ep.m + ' http://localhost:8000' + ep.p + '</div>\n<div class="ok">' + escapeHtml(r).replace(/\n/g, "<br>").replace(/ /g, "&nbsp;") + "</div>";
}

// ── PDF Report Generator ────────────────────────────────────────────────
function generateReport() {
  var active = document.querySelector(".nav-links a.active");
  var panelName = active ? active.textContent.trim() : "Diagnostics";
  var now = new Date();
  var dateStr = now.toISOString().slice(0, 10) + " " + now.toTimeString().slice(0, 5);
  // Collect all visible canvases in the active section
  var sections = document.querySelectorAll(".section");
  var activeSection = null;
  sections.forEach(function (s) {
    if (s.style.display !== "none" && s.offsetParent !== null) activeSection = s;
  });
  if (!activeSection) {
    // Fallback: find by panel name
    var sectionMap = { "DIAGNOSTICS": "sec-diagnostics", "UPLOAD": "sec-upload", "BMS": "sec-bms", "MATERIALS": "sec-materials", "RECYCLING": "sec-recycling" };
    activeSection = document.getElementById(sectionMap[panelName.toUpperCase()] || "sec-diagnostics");
  }
  // Build a report in a new window
  var win = window.open("", "_blank", "width=800,height=1100");
  if (!win) { showToast("Pop-up blocked. Allow pop-ups for report generation.", "warn"); return; }
  var html = '<!DOCTYPE html><html><head><title>KineticsForge Report - ' + panelName + '</title>';
  html += '<style>';
  html += 'body{font-family:Inter,system-ui,sans-serif;color:#222;margin:2rem;line-height:1.6;}';
  html += 'h1{font-size:1.4rem;border-bottom:2px solid #cc0000;padding-bottom:0.5rem;margin-bottom:0.3rem;}';
  html += '.meta{color:#666;font-size:0.78rem;margin-bottom:1.5rem;}';
  html += '.metric{display:inline-block;text-align:center;margin:0.5rem 1rem 0.5rem 0;padding:0.7rem 1.2rem;border:1px solid #ddd;border-radius:6px;min-width:100px;}';
  html += '.metric .val{font-size:1.3rem;font-weight:700;color:#cc0000;font-family:monospace;}';
  html += '.metric .lbl{font-size:0.65rem;color:#888;text-transform:uppercase;margin-top:2px;}';
  html += '.decision{border-left:3px solid #cc0000;background:#fff5f5;padding:0.8rem;margin:1rem 0;font-size:0.82rem;}';
  html += 'img{max-width:100%;margin:0.8rem 0;border:1px solid #eee;border-radius:4px;}';
  html += '.footer{margin-top:2rem;padding-top:1rem;border-top:1px solid #ddd;font-size:0.68rem;color:#999;}';
  html += '.confidence-bar{height:8px;background:#eee;border-radius:4px;margin:4px 0 8px;overflow:hidden;}';
  html += '.confidence-fill{height:100%;border-radius:4px;}';
  html += '@media print{body{margin:1cm;}@page{margin:1.5cm;}}';
  html += '</style></head><body>';
  html += '<h1>KineticsForge — ' + escapeHtml(panelName) + ' Report</h1>';
  html += '<div class="meta">Generated: ' + dateStr + ' | Platform: KineticsForge v1.0 | Mode: simulation-backed</div>';
  // Metrics
  if (activeSection) {
    var stats = activeSection.querySelectorAll(".stat");
    if (stats.length) {
      stats.forEach(function (s) {
        var val = s.querySelector(".val");
        var lbl = s.querySelector(".lbl");
        if (val && lbl) {
          html += '<div class="metric"><div class="val">' + escapeHtml(val.textContent) + '</div><div class="lbl">' + escapeHtml(lbl.textContent) + '</div></div>';
        }
      });
      html += '<br>';
    }
    // Readout grids
    var readouts = activeSection.querySelectorAll(".readout");
    if (readouts.length) {
      readouts.forEach(function (r) {
        var k = r.querySelector(".k");
        var v = r.querySelector(".v");
        if (k && v) html += '<div class="metric"><div class="val">' + escapeHtml(v.textContent) + '</div><div class="lbl">' + escapeHtml(k.textContent) + '</div></div>';
      });
      html += '<br>';
    }
    // Decision box
    var decision = activeSection.querySelector(".decision-box");
    if (decision && decision.textContent.trim()) {
      html += '<div class="decision">' + decision.innerHTML + '</div>';
    }
    // Canvases → inline images
    var canvases = activeSection.querySelectorAll("canvas");
    canvases.forEach(function (cv) {
      try {
        var dataUrl = cv.toDataURL("image/png");
        if (dataUrl && dataUrl.length > 100) {
          html += '<img src="' + dataUrl + '" alt="Chart">';
        }
      } catch (e) { /* tainted canvas, skip */ }
    });
  }
  html += '<div class="footer">KineticsForge — Simulation-backed, real-data indexed, validation-gated.<br>All predictions carry uncertainty bounds and provenance metadata. Research-preview outputs are not certification.</div>';
  html += '</body></html>';
  win.document.write(html);
  win.document.close();
  setTimeout(function () { win.print(); }, 600);
}

// ── Confidence/Uncertainty Display ──────────────────────────────────────
function renderConfidence(containerId, confidence, detail) {
  var el = document.getElementById(containerId);
  if (!el) return;
  if (confidence == null || !Number.isFinite(confidence)) {
    el.innerHTML = "";
    return;
  }
  var pct = Math.round(clamp(confidence, 0, 1) * 100);
  var color = pct >= 75 ? "#22c55e" : pct >= 50 ? "#ff9f1a" : "#ff1a1a";
  var label = pct >= 75 ? "High confidence" : pct >= 50 ? "Moderate confidence" : "Low confidence";
  var html = '<div style="margin-top:0.5rem;padding:0.5rem 0.65rem;background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);border-radius:6px">';
  html += '<div style="display:flex;justify-content:space-between;align-items:center;font-size:0.7rem">';
  html += '<span style="color:#888">Model Confidence</span>';
  html += '<span style="color:' + color + ';font-weight:700;font-family:JetBrains Mono,monospace">' + pct + '% — ' + label + '</span>';
  html += '</div>';
  html += '<div style="height:6px;background:#1a1a1a;border-radius:3px;margin-top:5px;overflow:hidden">';
  html += '<div style="height:100%;width:' + pct + '%;background:' + color + ';border-radius:3px;transition:width 0.6s ease"></div>';
  html += '</div>';
  if (detail) {
    html += '<div style="font-size:0.64rem;color:#666;margin-top:4px">' + escapeHtml(detail) + '</div>';
  }
  html += '</div>';
  el.innerHTML = html;
}

function outsidePenalty(value, lo, hi, maxPenalty) {
  if (!Number.isFinite(value)) return maxPenalty * 0.5;
  if (value >= lo && value <= hi) return 0;
  var dist = value < lo ? (lo - value) : (value - hi);
  var span = Math.max(hi - lo, 1e-9);
  return clamp((dist / span) * maxPenalty, 0, maxPenalty);
}

function computeDiagConfidence(out) {
  if (!out || !out.cap || !out.cap.length) {
    return { confidence: 0.25, detail: "No diagnostic curve is available yet." };
  }
  var sei = Number(out.sei && out.sei[out.sei.length - 1]) || 0;
  var p2 = Number(out.p2 && out.p2[out.p2.length - 1]) || 0;
  var jt = Number(out.jt && out.jt[out.jt.length - 1]) || 0;
  var rate = Number(out.rate && out.rate[out.rate.length - 1]) || 0;
  var residual = Number(out.residual && out.residual[out.residual.length - 1]) || 0;
  var total = Math.max(1e-9, sei + p2 + jt + rate + residual);
  var residualFrac = clamp(residual / total, 0, 1);
  var topShare = Math.max(sei, p2, jt, rate, residual) / total;
  var fade = 1 - (Number(out.cap[out.cap.length - 1]) || 1);
  var bandWidth = 0.06;
  if (out.band && out.band.lo && out.band.hi && out.band.lo.length && out.band.hi.length) {
    var lo = Number(out.band.lo[out.band.lo.length - 1]);
    var hi = Number(out.band.hi[out.band.hi.length - 1]);
    if (Number.isFinite(lo) && Number.isFinite(hi)) bandWidth = Math.max(0, hi - lo);
  }
  var tempC = num("temp-slider", 45);
  var cRate = num("crate-slider", 1.0);
  var inputPenalty = outsidePenalty(tempC, 20, 55, 0.12) + outsidePenalty(cRate, 0.2, 2.0, 0.10);
  var conf = 0.78;
  conf -= residualFrac * 0.40;
  conf -= clamp((bandWidth - 0.02) / 0.20, 0, 0.22);
  conf -= fade < 0.02 ? 0.08 : 0;
  conf -= inputPenalty;
  conf += clamp((topShare - 0.22) * 0.14, 0, 0.10);
  conf = clamp(conf, 0.18, 0.90);
  var detail = "Residual share " + Math.round(residualFrac * 100) + "%, end-band width " + (bandWidth * 100).toFixed(1) + "%, stress penalty " + Math.round(inputPenalty * 100) + " pts.";
  return { confidence: conf, detail: detail };
}

function computeBmsConfidence(bms, frame) {
  if (!bms || !frame || !Array.isArray(frame.risks) || !frame.risks.length) {
    return { confidence: 0.25, detail: "No BMS frame is available yet." };
  }
  var gate = Number(bms.threshold);
  if (!Number.isFinite(gate)) gate = 0.42;
  var maxRisk = Number(frame.maxRisk);
  if (!Number.isFinite(maxRisk)) maxRisk = Math.max.apply(null, frame.risks);
  var distance = Math.abs(maxRisk - gate);
  var distanceScore = clamp(distance / 0.22, 0, 1);
  var spreadScore = clamp(stddev(frame.risks) / 0.16, 0, 1);
  var tempSpread = clamp(stddev(frame.temps || []) / 8.0, 0, 1);
  var slopePeak = Math.max.apply(null, (frame.slopes || [0]).map(function (x) { return Math.abs(Number(x) || 0); }));
  var slopeScore = clamp(slopePeak / 0.06, 0, 1);
  var ambient = Number(bms.ambient_C);
  var ambientPenalty = outsidePenalty(ambient, 20, 55, 0.12);
  var nearGatePenalty = distance < 0.035 ? 0.10 : 0;
  var conf = 0.42 + 0.24 * distanceScore + 0.13 * spreadScore + 0.10 * tempSpread + 0.08 * slopeScore - ambientPenalty - nearGatePenalty;
  var sweep = window.__kfBmsSweep;
  var sweepNote = "";
  if (sweep && sweep.scenarioKey && bms.scenarioKey && sweep.scenarioKey === bms.scenarioKey) {
    var sweepStd = Number(sweep.stdRisk);
    var alertRate = Number(sweep.alertRate);
    if (Number.isFinite(sweepStd)) conf -= clamp(sweepStd / 0.16, 0, 0.12);
    if (Number.isFinite(alertRate)) conf -= clamp(Math.abs(alertRate - 0.5) * 0.16, 0, 0.08);
    sweepNote = " Sweep alert rate " + (Number.isFinite(alertRate) ? (alertRate * 100).toFixed(0) : "--") + "%, sigma " + (Number.isFinite(sweepStd) ? sweepStd.toFixed(3) : "--") + ".";
  }
  conf = clamp(conf, 0.20, 0.88);
  var detail = "|risk-gate|=" + distance.toFixed(3) + ", pack risk std=" + stddev(frame.risks).toFixed(3) + ", Tmax spread " + stddev(frame.temps || []).toFixed(1) + " C." + sweepNote;
  return { confidence: conf, detail: detail };
}

function computeMaterialsConfidence(mat) {
  if (!mat) return { confidence: 0.25, detail: "No composition has been scored yet." };
  var stab = clamp(Number(mat.stability) || 0, 0, 1);
  var fade500 = clamp(Number(mat.fade500) || 0.5, 0, 1);
  var oxygenRisk = clamp(Number(mat.oxygenRisk) || 0.5, 0, 1);
  var chargeRisk = clamp(Number(mat.chargeRisk) || 0.5, 0, 1);
  var q0 = Number(mat.Q0) || 0;
  var v = Number(mat.avgVoltage) || 0;
  var comp = mat.comp || {};
  var compPenalty = outsidePenalty(Number(comp.Na), 0.85, 1.12, 0.10)
    + outsidePenalty(Number(comp.Mn), 0.18, 0.82, 0.08)
    + outsidePenalty(Number(comp.Fe), 0.12, 0.82, 0.08);
  var propertyPenalty = outsidePenalty(q0, 95, 210, 0.10) + outsidePenalty(v, 2.8, 4.3, 0.08);
  var quality = 0.36 * stab + 0.20 * (1 - fade500) + 0.22 * (1 - oxygenRisk) + 0.22 * (1 - chargeRisk);
  var modelConf = Number.isFinite(Number(mat.confidence)) ? Number(mat.confidence) : null;
  var conf = modelConf != null ? modelConf : (0.30 + 0.52 * quality - compPenalty - propertyPenalty);
  conf = clamp(conf, 0.18, 0.92);
  var ev = mat.evidence && mat.evidence.nearest_formula ? " Nearest evidence: " + mat.evidence.nearest_formula + "." : "";
  var detail = "Stability " + stab.toFixed(2) + ", fade500 " + (fade500 * 100).toFixed(1) + "%, phase " + (mat.phaseState || "screened") + ", site error " + fmt((mat.siteError || 0) * 100, 1) + "%." + ev;
  return { confidence: conf, detail: detail };
}

function computeRecyclingConfidence(rec) {
  if (!rec || !Number.isFinite(rec.totalRecovered)) {
    return { confidence: 0.25, detail: "No recycling run is available yet." };
  }
  var total = Math.max(1e-9, Number(rec.totalRecovered));
  var lo = Number(rec.lo);
  var hi = Number(rec.hi);
  var widthRatio = (Number.isFinite(lo) && Number.isFinite(hi)) ? Math.max(0, hi - lo) / total : 0.30;
  var purity = clamp(Number(rec.purity) || 0, 0, 1);
  var margin = Number(rec.marginProxy) || 0;
  var mn = Array.isArray(rec.targets) && rec.targets.length ? clamp(Number(rec.targets[0]) || 0, 0, 1) : 0;
  var acid = num("acid-slider", 2.0);
  var temp = num("leach-slider", 80);
  var processPenalty = outsidePenalty(acid, 1.0, 3.2, 0.08) + outsidePenalty(temp, 55, 92, 0.08);
  var mcEnabled = !!(document.getElementById("sw-mc") && document.getElementById("sw-mc").checked);
  var conf = 0.60 - 0.42 * clamp(widthRatio, 0, 0.8) + 0.12 * purity + 0.08 * mn - processPenalty - (mcEnabled ? 0 : 0.06);
  if (margin < 0) conf -= 0.03;
  conf = clamp(conf, 0.20, 0.85);
  var detail = "Recovery interval width " + (widthRatio * 100).toFixed(1) + "% of mean, purity " + (purity * 100).toFixed(1) + "%, Mn " + (mn * 100).toFixed(1) + "%.";
  return { confidence: conf, detail: detail };
}

function computeUploadConfidence(byod) {
  if (!byod || !byod.predictions) return { confidence: 0.25, detail: "No upload prediction is available yet." };
  var pred = byod.predictions || {};
  var modelConf = clamp(Number(pred.confidence) || 0, 0, 1);
  var mask = Array.isArray(byod.feature_mask) ? byod.feature_mask : [];
  var names = Array.isArray(byod.feature_names) ? byod.feature_names : [];
  var availability = names.length ? clamp(mask.filter(Boolean).length / names.length, 0, 1) : 0;
  var schemaScore = clamp(Number((byod.schema || {}).score) || 0, 0, 1);
  var warnings = Array.isArray(byod.warnings) ? byod.warnings.length : 0;
  var warningPenalty = clamp(warnings * 0.06, 0, 0.30);
  var mode = String(pred.inference_mode || "rules_only");
  var modeBonus = mode === "checkpoint_plus_rules" ? 0.06 : 0;
  var conf = 0.48 * modelConf + 0.24 * availability + 0.18 * schemaScore + 0.10 * (1 - warningPenalty) + modeBonus;
  conf -= warningPenalty;
  conf = clamp(conf, 0.15, 0.90);
  var detail = "Model " + Math.round(modelConf * 100) + "%, feature coverage " + Math.round(availability * 100) + "%, schema " + Math.round(schemaScore * 100) + "%, warnings " + warnings + ".";
  return { confidence: conf, detail: detail };
}

// ── Ragone Plot (Energy vs Power density) ──────────────────────────────
function currentMaterialForAnalytics() {
  var selected = materialSelectionFromControls(true);
  var mat = window.__kfMaterials;
  if (mat && mat.comp && Math.abs(Number(mat.comp.Na) - selected.Na) < 1e-6 && Math.abs(Number(mat.comp.Mn) - selected.Mn) < 1e-6 && Math.abs(Number(mat.comp.Fe) - selected.Fe) < 1e-6 && !!mat.comp.al === selected.al && !!mat.comp.ti === selected.ti) {
    return mat;
  }
  var prop = scoreComposition(selected, 318.15, materialKnobs());
  prop.comp = selected;
  return prop;
}

function runRagone() {
  var mat = currentMaterialForAnalytics();
  var maxCr = num("ragone-max-cr", 5.0);
  var cellMassG = num("ragone-cell-mass", 120);
  var pts = [];
  var rateCap = clamp(Number(mat.rateCapability) || 0.55, 0.05, 1.0);
  var oxygenRisk = clamp(Number(mat.oxygenRisk) || 0, 0, 1);
  var step = Math.max(0.05, maxCr / 36);
  for (var cr = 0.1; cr <= maxCr + 1e-9; cr += step) {
    var ratePenalty = 1 / (1 + Math.pow(cr / Math.max(0.25, 2.2 * rateCap), 1.35));
    var thermalDerate = clamp(1 - 0.08 * oxygenRisk * (cr / Math.max(1, maxCr)), 0.72, 1);
    var qEff = Math.max(0, mat.Q0) * ratePenalty * thermalDerate;
    var vEff = Math.max(2.0, mat.avgVoltage - 0.035 * cr / Math.max(0.08, rateCap));
    var energy = qEff * vEff;
    var power = energy * cr;
    pts.push({ cRate: cr, energy: energy, power: power, rate_capability: rateCap, voltage: vEff });
  }
  window.__kfRagone = pts;
  var cv = makeCanvas("ragone-chart");
  if (cv) drawRagonePlot(cv, pts);
  var best = pts.reduce(function (a, b) { return a.energy > b.energy ? a : b; });
  var peakPower = pts.reduce(function (a, b) { return a.power > b.power ? a : b; });
  var cellEnergyWh = best.energy * cellMassG / 1000;
  setReadouts("ragone-readout", [
    { k: "Peak Energy", v: best.energy.toFixed(1) + " Wh/kg" },
    { k: "Cell Energy", v: cellEnergyWh.toFixed(2) + " Wh" },
    { k: "Peak Power", v: peakPower.power.toFixed(1) + " W/kg" },
    { k: "Rate limit", v: rateCap.toFixed(2) }
  ]);
  showToast("Ragone plot computed from current material physics up to " + maxCr.toFixed(1) + "C.", "ok");
}

function drawRagonePlot(cv, pts) {
  if (!cv || !pts.length) return;
  var ctx = cv.getContext("2d");
  var W = cv.width, H = cv.height;
  var p = { t: 28, r: 18, b: 34, l: 52 };
  var pw = W - p.l - p.r, ph = H - p.t - p.b;
  var xs = pts.map(function (pt) { return pt.power; });
  var ys = pts.map(function (pt) { return pt.energy; });
  var xMn = 0, xMx = Math.max.apply(null, xs) * 1.1;
  var yMn = Math.min.apply(null, ys) * 0.9, yMx = Math.max.apply(null, ys) * 1.05;
  ctx.clearRect(0, 0, W, H);
  drawGrid(ctx, p, pw, ph);
  // Fill area under curve
  ctx.beginPath();
  pts.forEach(function (pt, i) {
    var x = p.l + (pt.power - xMn) / (xMx - xMn) * pw;
    var y = p.t + ph - (pt.energy - yMn) / (yMx - yMn) * ph;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.lineTo(p.l + (pts[pts.length - 1].power - xMn) / (xMx - xMn) * pw, p.t + ph);
  ctx.lineTo(p.l, p.t + ph);
  ctx.closePath();
  ctx.fillStyle = "rgba(255,26,26,0.08)";
  ctx.fill();
  // Draw line
  ctx.beginPath();
  pts.forEach(function (pt, i) {
    var x = p.l + (pt.power - xMn) / (xMx - xMn) * pw;
    var y = p.t + ph - (pt.energy - yMn) / (yMx - yMn) * ph;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = "#ff1a1a";
  ctx.lineWidth = 2.5;
  ctx.shadowColor = "#ff1a1a";
  ctx.shadowBlur = 10;
  ctx.stroke();
  ctx.shadowBlur = 0;
  // Draw points at key C-rates
  var maxCr = pts[pts.length - 1].cRate;
  var sweepPoints = [0.2, 0.5, 1.0, 2.0, 3.0, 5.0].filter(function(v) { return v <= maxCr; });
  sweepPoints.forEach(function (target) {
    var closest = pts.reduce(function (a, b) { return Math.abs(a.cRate - target) < Math.abs(b.cRate - target) ? a : b; });
    var x = p.l + (closest.power - xMn) / (xMx - xMn) * pw;
    var y = p.t + ph - (closest.energy - yMn) / (yMx - yMn) * ph;
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#888";
    ctx.font = "9px JetBrains Mono, monospace";
    ctx.fillText(target + "C", x + 6, y - 6);
  });
  ctx.fillStyle = "#777";
  ctx.font = "10px JetBrains Mono, monospace";
  ctx.fillText("Ragone: Energy vs Power Density", p.l + 6, 15);
  ctx.fillText("Power (W/kg)", p.l + pw / 2 - 30, H - 6);
  ctx.fillText("E", 4, p.t + ph / 2);
}

function exportRagoneCSV() {
  var pts = window.__kfRagone;
  if (!pts || !pts.length) { showToast("Run Ragone first.", "warn"); return; }
  downloadCSV(pts.map(function (pt) {
    return { c_rate: pt.cRate.toFixed(2), energy_wh_kg: pt.energy.toFixed(1), power_w_kg: pt.power.toFixed(1) };
  }), "kineticsforge_ragone.csv");
}

// ── Vehicle Range Estimator ────────────────────────────────────────────
function runRangeEstimate() {
  var soh = clamp(num("range-soh", 1.0), 0.5, 1.05);
  var energyWhKg = Math.max(1, num("range-energy", 250));
  var packKg = Math.max(1, num("range-pack-kg", 495));
  var effKwhPer100km = Math.max(1, num("range-eff", 15));
  var targetRangeKm = num("range-target", 600);
  var motorKw = Math.max(1, num("range-motor-kw", 150));
  var mat = currentMaterialForAnalytics();
  var chemistryDerate = clamp(0.90 + 0.08 * (Number(mat.stability) || 0.5) - 0.06 * (Number(mat.oxygenRisk) || 0) - 0.05 * (Number(mat.chargeRisk) || 0), 0.65, 0.96);

  var pts = [];
  var vehicleMassKg = packKg + 1400; // pack + glider
  var cruisePowerKwReq = clamp(effKwhPer100km * (vehicleMassKg / 1850) * 1.05, 8, 45);
  var powerFactor = clamp(motorKw / Math.max(5, cruisePowerKwReq), 0.55, 1.0);
  for (var s = 1.0; s >= 0.5; s -= 0.02) {
    var usableKwh = s * energyWhKg * packKg / 1000 * chemistryDerate;
    var rangeEnergyKm = usableKwh / effKwhPer100km * 100;
    var rangePowerKm = rangeEnergyKm * powerFactor;
    pts.push({ soh: s, range_km: rangeEnergyKm, range_power_km: rangePowerKm, range_effective_km: Math.min(rangeEnergyKm, rangePowerKm), usable_kwh: usableKwh });
  }
  window.__kfRange = pts;
  var current = pts.find(function (p) { return Math.abs(p.soh - soh) < 0.015; }) || pts[0];
  var cv = makeCanvas("range-chart");
  if (cv) {
    drawMultiLine(cv, [
      { name: "energy-limited", values: pts.map(function (p) { return p.range_km; }), color: "#22c55e", glow: true },
      { name: "power-limited", values: pts.map(function (p) { return p.range_effective_km; }), color: "#38bdf8" }
    ], {
      yMin: 0, xMax: 100, title: "Range vs SOH (energy line + power-limited overlay)", yDigits: 0, legend: true,
      points: [{ x: (1 - soh) / 0.5 * 100, y: current.range_effective_km }], pointColor: "#ffffff"
    });
  }
  var rangeDiff = current.range_effective_km - targetRangeKm;
  var specificPower = motorKw * 1000 / packKg;
  // Power-limited acceleration estimate: F=ma, P=Fv => t(0-100) ~ 0.5*m*v^2/P
  var v100 = 100 / 3.6; // 100 km/h in m/s
  var t0to100 = 0.5 * vehicleMassKg * v100 * v100 / (motorKw * 1000 * 0.85);
  setReadouts("range-readout", [
    { k: "Current Range", v: current.range_effective_km.toFixed(1) + " km" },
    { k: "Energy-limited", v: current.range_km.toFixed(1) + " km" },
    { k: "Power-limited", v: current.range_power_km.toFixed(1) + " km" },
    { k: "Usable kWh", v: current.usable_kwh.toFixed(1) + " kWh" },
    { k: "vs Target", v: (rangeDiff >= 0 ? "+" : "") + rangeDiff.toFixed(0) + " km" },
    { k: "Specific Power", v: specificPower.toFixed(0) + " W/kg" },
    { k: "Cruise Power Req", v: cruisePowerKwReq.toFixed(1) + " kW" },
    { k: "0-100 km/h", v: t0to100.toFixed(1) + " s (est)" },
    { k: "Chemistry Derate", v: (chemistryDerate * 100).toFixed(0) + "%" }
  ]);
  showToast("Range: " + current.range_effective_km.toFixed(0) + " km vs target " + targetRangeKm + " km.", "ok");
}

// ── Electrolyte Recommendation ─────────────────────────────────────────
function runElectrolyteRecommend() {
  var mat = currentMaterialForAnalytics();
  var comp = mat.comp || {};
  var diagComp = diagnosticComposition();
  var dopantHint = (comp.ti || comp.ti_doped) ? "Ti" : (comp.al || comp.al_doped) ? "Al" : (diagComp.dopant_type || "generic");
  var el = document.getElementById("electrolyte-result");
  if (!el) return;
  var electrolyte, additives, rationale;
  if ((mat.oxygenRisk || 0) > 0.55 || (mat.p2Risk || 0) > 0.55) {
    electrolyte = "High-concentration NaFSI in DME/TMP blend";
    additives = ["1-2 wt% FEC", "dry handling below 20 ppm H2O"];
    rationale = "High phase or oxygen risk: reduce free-solvent activity and strengthen the oxidative CEI before cycling near the upper cutoff.";
  } else if ((mat.jtIndex || 0) > 0.35) {
    electrolyte = "Carbonate EC/EMC with 1.0 M NaPF6";
    var jt_additives = ["2 wt% FEC", "1 wt% VC"];
    if (dopantHint === "Ti") jt_additives.push("Reduced FEC (Ti lowers Na+ migration barrier)");
    if (dopantHint === "Al") jt_additives.push("Mn scavenger screen (Al pillar stabilizes but Mn3+ still dissolves)");
    else jt_additives.push("Mn scavenger screen");
    additives = jt_additives;
    rationale = "Mn3+-rich recipe: prioritize CEI quality and transition-metal dissolution control." + (dopantHint === "Ti" ? " Ti doping lowers Na+ desolvation energy; consider reduced FEC loading." : "");
  } else if (Number(comp.Fe) > Number(comp.Mn)) {
    electrolyte = "Glyme ether with 1.0 M NaTFSI";
    additives = (dopantHint === "Al") ? ["FEC confirmation screen", "No SN needed (Al pillar reduces lattice strain)"] : ["FEC confirmation screen"];
    rationale = "Fe-rich lower-voltage recipe: ether solvation can reduce Na+ desolvation penalty and improve rate response.";
  } else {
    electrolyte = "Carbonate EC/PC with 1.0 M NaClO4 or NaPF6";
    additives = ["3-5 wt% FEC", "1 wt% succinonitrile"];
    if (dopantHint === "Ti") additives.push("Ti-doped: consider lower FEC loading (3 wt%)");
    if (dopantHint === "Al") additives.push("Al-doped: add PS additive for cathode CEI reinforcement");
    rationale = "Balanced Mn/Fe layered oxide: carbonate baseline with FEC, then validate CEI stability at the selected upper cutoff.";
  }
  var compatibility = clamp(0.82 - 0.30 * (mat.oxygenRisk || 0) - 0.18 * (mat.chargeRisk || 0) + 0.10 * (mat.rateCapability || 0.5), 0, 1);
  var html = '<div style="padding:0.5rem">';
  html += '<div style="margin-bottom:0.5rem"><span style="color:#22c55e;font-weight:700">Recommended Electrolyte</span></div>';
  html += '<div style="font-size:0.82rem;color:#e8e8e8;font-weight:600;margin-bottom:0.4rem">' + escapeHtml(electrolyte) + '</div>';
  html += '<div style="font-size:0.72rem;color:#aaa;margin-bottom:0.5rem"><strong>Additives:</strong> ' + additives.map(escapeHtml).join(", ") + '</div>';
  html += '<div style="font-size:0.68rem;color:#777;border-left:2px solid #ff1a1a;padding-left:0.6rem">' + escapeHtml(rationale) + '</div>';
  html += '<div style="margin-top:0.6rem;font-size:0.64rem;color:#666">Compatibility ' + (compatibility * 100).toFixed(0) + '%, phase ' + escapeHtml(mat.phaseState || "screened") + ', dopant basis ' + escapeHtml(dopantHint) + '</div>';
  html += '</div>';
  el.innerHTML = html;
  showToast("Electrolyte: " + electrolyte.split(" with ")[0].trim(), "ok");
}

// ── Regional Climate Stress Profiles ───────────────────────────────────
var CLIMATE_SEEDS = {
  delhi_hot: { name: "Delhi", meanT: 29, seasonAmp: 12, diurnalAmp: 6.5, meanRH: 46, monsoonAmp: 28 },
  chennai_coastal: { name: "Chennai", meanT: 29, seasonAmp: 4, diurnalAmp: 3.2, meanRH: 72, monsoonAmp: 16 },
  mumbai_monsoon: { name: "Mumbai", meanT: 28, seasonAmp: 5, diurnalAmp: 3.5, meanRH: 76, monsoonAmp: 20 },
  jaipur_desert: { name: "Jaipur", meanT: 28, seasonAmp: 14, diurnalAmp: 8, meanRH: 38, monsoonAmp: 24 },
  leh_cold: { name: "Leh", meanT: 6, seasonAmp: 17, diurnalAmp: 8.5, meanRH: 35, monsoonAmp: 10 },
  guwahati_humid: { name: "Guwahati", meanT: 25, seasonAmp: 7, diurnalAmp: 3.5, meanRH: 78, monsoonAmp: 18 }
};

function generateClimateProfile(regionKey, days) {
  var seed = CLIMATE_SEEDS[regionKey] || CLIMATE_SEEDS.delhi_hot;
  var n = Math.max(24, days * 24);
  var temps = [];
  var rhs = [];
  var heatStress = [];
  var coldPlating = [];
  var moistureIngress = [];
  var naohRisk = [];
  var corrosionRisk = [];
  var offset = num("climate-temp-offset", 0.0);
  var chargeStress = num("climate-charge-stress", 1.0);
  var rhBias = num("climate-rh-bias", 0.0);
  for (var h = 0; h < n; h++) {
    var day = h / 24.0;
    var seasonal = seed.seasonAmp * Math.sin(2 * Math.PI * (day / 365 + 0.18));
    var diurnal = seed.diurnalAmp * Math.sin(2 * Math.PI * ((h % 24) - 14) / 24);
    var t = seed.meanT + seasonal + diurnal + offset;
    var monsoon = seed.monsoonAmp * Math.sin(2 * Math.PI * (day / 365 - 0.05));
    var rh = clamp(seed.meanRH + monsoon - 0.45 * diurnal + rhBias, 8, 98);
    // Humidity channels: moisture ingress, NaOH tendency, and collector corrosion proxy.
    var moisture = clamp((rh - 65) / 25, 0, 1) * (0.55 + 0.45 * clamp(chargeStress / 2.0, 0.4, 1.5));
    var naoh = clamp((rh - 72) / 18, 0, 1) * sigmoid((t - 30) / 5);
    var corrosion = clamp((rh - 75) / 16, 0, 1) * sigmoid((t - 34) / 4);
    var thermal = sigmoid((t - 42) / 3.5);
    var hs = thermal * (1 + 0.20 * moisture + 0.15 * naoh + 0.12 * corrosion) * chargeStress;
    var cp = 1 / (1 + Math.exp((t - 2) / 3));
    temps.push(t);
    rhs.push(rh);
    heatStress.push(clamp(hs, 0, 1.5));
    coldPlating.push(clamp(cp, 0, 1));
    moistureIngress.push(clamp(moisture, 0, 1.2));
    naohRisk.push(clamp(naoh, 0, 1));
    corrosionRisk.push(clamp(corrosion, 0, 1));
  }
  return {
    name: seed.name,
    temps: temps,
    rhs: rhs,
    heatStress: heatStress,
    coldPlating: coldPlating,
    moistureIngress: moistureIngress,
    naohRisk: naohRisk,
    corrosionRisk: corrosionRisk,
    hours: n
  };
}

function runClimateStress() {
  var region = document.getElementById("climate-region").value;
  var days = parseInt(document.getElementById("climate-days").value, 10) || 14;
  var profile = generateClimateProfile(region, days);
  // Subsample for plotting (max 400 points)
  var step = Math.max(1, Math.floor(profile.hours / 400));
  var tSub = [], hsSub = [];
  for (var i = 0; i < profile.hours; i += step) {
    tSub.push(profile.temps[i]);
    hsSub.push(profile.heatStress[i]);
  }
  var cv = makeCanvas("climate-chart");
  if (cv) {
    drawMultiLine(cv, [
      { name: "Temp °C", values: tSub, color: "#ff9f1a" },
      { name: "Heat stress", values: hsSub.map(function (v) { return v * 50; }), color: "#ff1a1a", glow: true }
    ], { yMin: -10, yMax: 60, title: profile.name + " — " + days + " day temperature + heat stress (×50)", legend: true, yDigits: 0 });
  }
  var meanT = mean(profile.temps);
  var maxT = Math.max.apply(null, profile.temps);
  var minT = Math.min.apply(null, profile.temps);
  var hsHours = profile.heatStress.filter(function (v) { return v > 0.1; }).length;
  var cpHours = profile.coldPlating.filter(function (v) { return v > 0.1; }).length;
  var humidHours = profile.rhs.filter(function (v) { return v > 75; }).length;
  var ingressHours = profile.moistureIngress.filter(function (v) { return v > 0.25; }).length;
  var naohHours = profile.naohRisk.filter(function (v) { return v > 0.25; }).length;
  var corrosionHours = profile.corrosionRisk.filter(function (v) { return v > 0.25; }).length;
  setReadouts("climate-readout", [
    { k: "Mean T", v: meanT.toFixed(1) + " °C" },
    { k: "Max T", v: maxT.toFixed(1) + " °C" },
    { k: "Heat Stress hrs", v: String(hsHours) },
    { k: "Cold Plating hrs", v: String(cpHours) },
    { k: "Mean RH", v: mean(profile.rhs).toFixed(0) + "%" },
    { k: "Humidity hrs", v: String(humidHours) },
    { k: "Moisture Ingress hrs", v: String(ingressHours) },
    { k: "NaOH Proxy hrs", v: String(naohHours) },
    { k: "Corrosion Proxy hrs", v: String(corrosionHours) }
  ]);
  showToast(profile.name + ": mean " + meanT.toFixed(1) + "°C, " + hsHours + " heat-stress hours.", "ok");
}

function runClimateCompare() {
  var days = parseInt(document.getElementById("climate-days").value, 10) || 14;
  var regions = Object.keys(CLIMATE_SEEDS);
  var barData = regions.map(function (key) {
    var p = generateClimateProfile(key, days);
    return {
      name: CLIMATE_SEEDS[key].name,
      meanT: mean(p.temps),
      maxT: Math.max.apply(null, p.temps),
      meanRH: mean(p.rhs),
      hsHours: p.heatStress.filter(function (v) { return v > 0.1; }).length,
      cpHours: p.coldPlating.filter(function (v) { return v > 0.1; }).length,
      humidHours: p.rhs.filter(function (v) { return v > 75; }).length,
      ingressHours: p.moistureIngress.filter(function (v) { return v > 0.25; }).length,
      naohHours: p.naohRisk.filter(function (v) { return v > 0.25; }).length,
      corrosionHours: p.corrosionRisk.filter(function (v) { return v > 0.25; }).length
    };
  });
  barData.sort(function (a, b) {
    var bScore = b.hsHours + 0.40 * b.humidHours + 0.55 * b.cpHours + 0.35 * b.ingressHours + 0.30 * b.naohHours + 0.30 * b.corrosionHours;
    var aScore = a.hsHours + 0.40 * a.humidHours + 0.55 * a.cpHours + 0.35 * a.ingressHours + 0.30 * a.naohHours + 0.30 * a.corrosionHours;
    return bScore - aScore;
  });
  var cv = makeCanvas("climate-compare-chart");
  if (!cv) return;
  var ctx = cv.getContext("2d");
  var W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  var pad = { t: 20, b: 40, l: 52, r: 10 };
  var pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
  var n = barData.length;
  var bw = Math.floor(pw / n * 0.7);
  var gap = Math.floor(pw / n * 0.3);
  var maxHs = Math.max.apply(null, barData.map(function (d) { return d.hsHours; })) || 1;
  barData.forEach(function (d, i) {
    var x = pad.l + i * (bw + gap) + gap / 2;
    var barH = (d.hsHours / maxHs) * ph;
    // Heat stress bar
    var grad = ctx.createLinearGradient(x, pad.t + ph - barH, x, pad.t + ph);
    grad.addColorStop(0, "#ff4444");
    grad.addColorStop(1, "#8b0000");
    ctx.fillStyle = grad;
    ctx.fillRect(x, pad.t + ph - barH, bw, barH);
    // Cold plating small bar
    var cpH = (d.cpHours / Math.max(maxHs, 1)) * ph;
    ctx.fillStyle = "rgba(56,189,248,0.5)";
    ctx.fillRect(x + bw - 6, pad.t + ph - cpH, 6, cpH);
    // Label
    ctx.fillStyle = "#aaa";
    ctx.font = "9px JetBrains Mono, monospace";
    ctx.textAlign = "center";
    ctx.fillText(d.name, x + bw / 2, H - 8);
    ctx.fillText(d.hsHours + "h", x + bw / 2, pad.t + ph - barH - 5);
    ctx.textAlign = "start";
  });
  ctx.fillStyle = "#777";
  ctx.font = "10px JetBrains Mono, monospace";
  ctx.fillText("Heat stress hours (red) + cold plating (blue) across India regions", pad.l, 14);
  var byHumidity = barData.slice().sort(function (a, b) { return b.humidHours - a.humidHours; })[0];
  var byCold = barData.slice().sort(function (a, b) { return b.cpHours - a.cpHours; })[0];
  var byIngress = barData.slice().sort(function (a, b) { return b.ingressHours - a.ingressHours; })[0];
  setReadouts("climate-readout", [
    { k: "Worst heat", v: barData[0].name + " (" + barData[0].hsHours + "h)" },
    { k: "Worst humidity", v: byHumidity.name + " (" + byHumidity.humidHours + "h)" },
    { k: "Worst cold", v: byCold.name + " (" + byCold.cpHours + "h)" },
    { k: "Worst moisture ingress", v: byIngress.name + " (" + byIngress.ingressHours + "h)" },
    { k: "Method", v: "T + RH + charge stress + moisture/NaOH/corrosion proxies" }
  ]);
}

window.addEventListener("DOMContentLoaded", function () {
  setTimeout(function () {
    animC(document.getElementById("counter-data"), 5462518);
    animC(document.getElementById("counter-models"), 14);
    animC(document.getElementById("counter-cells"), 555);
    animC(document.getElementById("counter-endpoints"), 14);
  }, 300);
  initArchitecture();
  fetchModelRegistry();
  initAPIEndpoints();
  initAssistant();
  updateNavHealth();
  setInterval(updateNavHealth, 60000);
  updateDiag();
  updateBMS();
  updateMat();
  updateRecycling();
  renderDecisionConsole();
});
