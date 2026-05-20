"""Small server-side assistant for the KineticsForge web UI.

The server does not persist private chat history. The browser may send recent
local turns with each request so follow-up questions can still make sense.
"""
import json
import math
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


PROJECT_KNOWLEDGE = [
    {
        "title": "What KineticsForge is",
        "keywords": "overview about what workbench kineticsforge purpose battery engineering",
        "text": (
            "KineticsForge is a battery physics workbench. It helps choose which "
            "degradation mechanism is dominating, which pack cell needs attention, "
            "which Na-ion cathode recipe is worth making, and which recycling recipe "
            "is worth running next. The production CPU server runs physics endpoints "
            "and can use trained checkpoints when the PyTorch runtime is available."
        ),
    },
    {
        "title": "Diagnostics",
        "keywords": "diagnostics degradation lifetime fade capacity soh ude p2 o2 jahnteller sei 4d mechanism surface state map",
        "text": (
            "The diagnostics panel simulates capacity fade for Na-ion cathodes. "
            "Important controls are temperature, C-rate, cycle count, P2-O2 phase "
            "transition, Jahn-Teller distortion, SEI growth, Butler-Volmer rate stress, and residual scale. "
            "Outputs include end-of-life capacity, total fade, knee point, RUL at 80% "
            "SOH, mechanism contributions, a mechanism state map, and a recommendation. "
            "The mechanism state map encodes cycle, SOC, mechanism intensity, and dominant "
            "loss channel on one 2D canvas; it is not a literal 4-spatial-dimensional object."
        ),
    },
    {
        "title": "BMS Pack Monitoring",
        "keywords": "bms battery management system pack thermal graph eis risk alert cell fault cell risk map highlighted c0 c1 c2 c3 c4 c5 c6 c7",
        "text": (
            "The BMS panel simulates a battery pack as a thermal graph. It tracks "
            "cell temperatures, EIS-like resistance drift, neighbor coupling, and risk "
            "scores. The cell risk map highlights cells with high risk, high temperature, "
            "or fault injection state. It answers which cell is becoming unsafe early "
            "enough to cool, isolate, or inspect."
        ),
    },
    {
        "title": "Materials Screening",
        "keywords": "materials cathode screening composition na mn fe al ti capacity voltage stability fade cost",
        "text": (
            "The materials panel scores Na-ion cathode compositions by capacity, "
            "voltage, stability, fade, cost, defect risk, and dopant effects. It is a "
            "prioritization tool for deciding which candidate is worth making next, not "
            "a substitute for lab validation."
        ),
    },
    {
        "title": "Recycling Optimizer",
        "keywords": "recycling optimizer leaching shrinking core bayesian recovery acid temperature particle mass cost",
        "text": (
            "The recycling panel estimates metal recovery from black mass using a "
            "shrinking-core leaching model, acid molarity, temperature, leach time, "
            "particle size, Bayesian recovery priors, Monte Carlo uncertainty, and an "
            "INR batch-cost proxy."
        ),
    },
    {
        "title": "API Surface",
        "keywords": "api endpoints health models predict simulate optimize screen chat upload byod fastapi",
        "text": (
            "The production FastAPI app exposes /health, /api/models, "
            "/api/predict/degradation, /api/simulate/bms, /api/optimize/recycling, "
            "/api/screen/cathode, /api/byod/analyze, /api/byod/analyze-full, and "
            "/api/byod/compare, /api/byod/batch, /api/byod/webhook/cycle, session exports, "
            "and /api/chat. Compatibility endpoints also exist for lifetime prediction, BMS "
            "alerts, recycling, and cathode screening."
        ),
    },
    {
        "title": "Upload Diagnostics",
        "keywords": "upload byod csv xlsx dataset schema features soh confidence cycle80 chemistry plating warnings",
        "text": (
            "The upload panel fingerprints cycler columns, extracts tier-1 features, "
            "computes dQ/dV peaks when voltage-capacity traces are available, reports "
            "SOH and cycle-to-80 estimates, and marks warnings when data is missing, "
            "specific-capacity based, truncated, or low confidence. If PyTorch is "
            "available, trained checkpoint outputs are attached next to the rule outputs."
        ),
    },
    {
        "title": "Decision Console",
        "keywords": "decision console action queue experiment tickets memo priority next owner gate critical watch ready",
        "text": (
            "The Decision Console is a browser-side operating layer. It does not run hidden models. "
            "It derives an action queue, owner, evidence, and next experiment from the current panel "
            "outputs and local run history. It can export experiment tickets as CSV and decision memos "
            "as Markdown or JSON."
        ),
    },
    {
        "title": "Data And Validation",
        "keywords": "data validation rows nasa pcoe isu ilcc battery life benchmark uncertainty provenance claim",
        "text": (
            "The UI lists about 5.09M indexed data rows across cycling, impedance, "
            "time-series, and cycle-summary sources. Claims should stay simulation-backed "
            "unless a report explicitly says it used real holdout data. Predictions should "
            "carry uncertainty and provenance."
        ),
    },
    {
        "title": "Deployment",
        "keywords": "deployment render free tier openrouter api key environment variable model server cloud",
        "text": (
            "The deploy path is a production CPU FastAPI app plus static HTML/CSS/JS and numpy "
            "physics endpoints. The chat assistant does not load an LLM locally. It calls "
            "OpenRouter only when OPENROUTER_API_KEY is configured. OPENROUTER_MODEL can "
            "override the default model."
        ),
    },
    {
        "title": "Limitations",
        "keywords": "limitations not medical safety production guarantee memory stateless context hallucination",
        "text": (
            "The assistant does not persist users server-side. It can use recent browser-supplied "
            "conversation turns for follow-up context. It should answer specific UI questions "
            "directly, avoid brochure language, and must not invent validation results or claim "
            "production battery safety certification."
        ),
    },
]


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "for", "from",
    "how", "i", "in", "is", "it", "like", "of", "on", "or", "should", "the", "to",
    "u", "use", "what", "when", "with", "you", "your",
}


def _tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1 and t not in STOPWORDS]


def retrieve_context(question: str, limit: int = 4) -> List[Dict[str, str]]:
    q_tokens = set(_tokens(question))
    scored = []
    for section in PROJECT_KNOWLEDGE:
        haystack = set(_tokens(section["title"] + " " + section["keywords"] + " " + section["text"]))
        score = len(q_tokens & haystack)
        if any(word in question.lower() for word in section["keywords"].split()):
            score += 1
        scored.append((score, section))
    scored.sort(key=lambda item: item[0], reverse=True)
    chosen = [section for score, section in scored if score > 0][:limit]
    if not chosen:
        chosen = [PROJECT_KNOWLEDGE[0], PROJECT_KNOWLEDGE[7], PROJECT_KNOWLEDGE[8]]
    return chosen


def _state_summary(state: Optional[Dict[str, Any]]) -> str:
    if not isinstance(state, dict):
        return ""
    parts = []
    for section in ("diagnostics", "bms", "materials", "recycling", "upload", "decisions"):
        value = state.get(section)
        if not isinstance(value, dict):
            continue
        fields = []
        for key, item in value.items():
            if item is None or item == "":
                continue
            if isinstance(item, float):
                if math.isfinite(item):
                    fields.append(f"{key}={item:.4g}")
            elif isinstance(item, (int, bool)):
                fields.append(f"{key}={item}")
            else:
                text = str(item).strip()
                if text:
                    fields.append(f"{key}={text[:220]}")
        if fields:
            parts.append(section + ": " + ", ".join(fields[:10]))
    conversation = state.get("conversation")
    if isinstance(conversation, list):
        turns = []
        for item in conversation[-8:]:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "user").strip()[:12]
            text = str(item.get("text") or "").strip()
            if text:
                turns.append(f"{role}: {text[:280]}")
        if turns:
            parts.append("recent conversation:\n" + "\n".join(turns))
    return "\n".join(parts)[:2400]


def _cell_details(state: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if not isinstance(state, dict):
        return {}
    bms = state.get("bms")
    if not isinstance(bms, dict):
        return {}
    details = bms.get("cell_details") or bms.get("cell_risk") or bms.get("cells_detail")
    if not isinstance(details, list):
        return {}
    out = {}
    for item in details:
        if not isinstance(item, dict):
            continue
        cell = str(item.get("cell", "")).upper()
        if re.fullmatch(r"C\d+", cell):
            if "temp_C" not in item and "temp" in item:
                item = {**item, "temp_C": item.get("temp")}
            out[cell] = item
    return out


def _as_float(value: Any, fallback: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return fallback
    return out if math.isfinite(out) else fallback


def _direct_ui_answer(question: str, state: Optional[Dict[str, Any]]) -> Optional[str]:
    q = question.lower().strip()
    cells = _cell_details(state)
    bms = state.get("bms", {}) if isinstance(state, dict) else {}
    diag = state.get("diagnostics", {}) if isinstance(state, dict) else {}
    mat = state.get("materials", {}) if isinstance(state, dict) else {}
    rec = state.get("recycling", {}) if isinstance(state, dict) else {}
    upload = state.get("upload", {}) if isinstance(state, dict) else {}
    decisions = state.get("decisions", {}) if isinstance(state, dict) else {}

    if upload and any(term in q for term in ["upload", "dataset", "file", "my cell", "cell healthy", "healthy", "soh", "schema", "features", "plating", "chemistry", "warnings"]):
        filename = upload.get("filename") or "the uploaded file"
        rows = upload.get("rows_read", "--")
        features = upload.get("features_present", 0)
        schema_score = _as_float(upload.get("schema_score"), float("nan"))
        soh = upload.get("soh")
        confidence = upload.get("confidence")
        cycle_80 = upload.get("cycle_80")
        warnings = upload.get("warnings") if isinstance(upload.get("warnings"), list) else []
        models = upload.get("model_outputs") if isinstance(upload.get("model_outputs"), dict) else {}
        m11 = models.get("M11_ElectrolyteHealth", {}) if isinstance(models.get("M11_ElectrolyteHealth"), dict) else {}
        m13 = models.get("M13_ChemIdentifier", {}) if isinstance(models.get("M13_ChemIdentifier"), dict) else {}
        plating = m11.get("checkpoint_sodium_plating_probability", m11.get("sodium_plating_probability"))
        chem = m13.get("checkpoint_class_id", m13.get("predicted_family", "unknown"))
        soh_text = _fmt_state_value(_as_float(soh, float("nan")) * 100, "%", 1)
        conf_text = _fmt_state_value(_as_float(confidence, float("nan")) * 100, "%", 0)
        verdict = "healthy enough for triage" if _as_float(soh, 0.0) >= 0.90 and _as_float(confidence, 0.0) >= 0.60 and not warnings else "needs cautious interpretation"
        warn_text = " No parser warnings." if not warnings else " Main warning: " + str(warnings[0])
        plating_text = "" if plating is None else f" Sodium plating risk is {_fmt_state_value(plating, digits=3)}."
        return (
            f"{filename} parsed {rows} rows with {features} features and schema score {_fmt_state_value(schema_score, digits=2)}. "
            f"SOH is {soh_text}, confidence is {conf_text}, cycle-to-80 is {cycle_80 if cycle_80 is not None else '--'}; this {verdict}. "
            f"Chemistry signal: {chem}.{plating_text}{warn_text}"
        )

    cell_match = re.search(r"\bc\s*([0-9]{1,2})\b", q)
    if cell_match and any(term in q for term in ["highlight", "red", "glow", "signify", "mean", "risk", "cell", "do", "should", "about", "with", "what", "why", "how", "action", "inspect", "cool", "isolate"]):
        cell_id = "C" + cell_match.group(1)
        detail = cells.get(cell_id)
        if detail:
            risk = _as_float(detail.get("risk"))
            temp = _as_float(detail.get("temp_C"), 25.0)
            threshold = bms.get("threshold")
            is_fault = bool(detail.get("fault"))
            hot = temp >= 50
            high = isinstance(threshold, (int, float)) and risk >= threshold
            reason = "risk crossed the action threshold" if high else "temperature is elevated" if hot else "it is currently the warmest/highest-risk visible cell"
            if is_fault:
                reason += " and it is the injected fault cell in this simulation"
            return (
                f"{cell_id} is highlighted because {reason}. "
                f"Current risk is {risk:.2f} and temperature is {temp:.1f} C. "
                "Treat it as an inspection/cooling candidate, not proof that the physical cell has failed."
            )
        decision = str(bms.get("decision", "")).strip()
        if decision and cell_id.lower() in decision.lower():
            return f"{cell_id} is highlighted because it is the current BMS action candidate. {decision}"
        return f"{cell_id} is the cell tile. Highlighting means the simulation sees higher risk or heat there than nearby cells."

    if "highlight" in q and "cell" in q:
        details = sorted(cells.values(), key=lambda d: max(_as_float(d.get("risk")), max(0.0, _as_float(d.get("temp_C"), 25.0) - 45) / 25), reverse=True)
        if details:
            top = details[0]
            return (
                f"The highlighted cell is the pack cell with the strongest current warning signal. "
                f"Here that is {top.get('cell')}: risk {_as_float(top.get('risk')):.2f}, temp {_as_float(top.get('temp_C'), 25.0):.1f} C. "
                "It means inspect/cool that cell first; it is not an automatic failure certificate."
            )
        return "A highlighted cell means the BMS simulation sees higher risk or heat there than in neighboring cells."

    if "risk map" in q or "riskmap" in q or ("cell" in q and "risk" in q):
        details = sorted(cells.values(), key=lambda d: _as_float(d.get("risk")), reverse=True)
        if details:
            top = details[0]
            answer = (
                "The cell risk map is the BMS triage view. Each tile is one pack cell; the number is risk from temperature, heat-rise slope, EIS resistance drift, and neighbor heating. "
                f"Right now {top.get('cell')} is highest at risk {_as_float(top.get('risk')):.2f}, temp {_as_float(top.get('temp_C'), 25.0):.1f} C."
            )
            threshold = bms.get("threshold")
            if isinstance(threshold, (int, float)):
                answer += f" The action threshold is {float(threshold):.2f}."
            return answer
        return "The cell risk map is the BMS triage view: one tile per cell, with color/glow increasing when thermal risk, EIS drift, or neighbor heating rises."

    # BMS non-determinism / different output each time
    if any(term in q for term in ["diff output", "different output", "different result", "every time", "not reproducible", "random", "inconsistent", "changes every", "non-deterministic"]):
        if any(term in q for term in ["bms", "cell", "risk", "pack", "map"]):
            seed = bms.get("seed", "not set")
            inject = bms.get("inject_fault", True)
            return (
                f"The BMS simulation uses a seeded pseudo-random number generator (seed={seed}) for initial cell temperatures, resistance offsets, and fault cell selection. "
                f"With the same seed and 'Inject Cell Failure' = {inject}, you will get identical results every run. "
                "To explore different scenarios, change the seed value in the advanced knobs or click '\U0001f3b2 New Seed'. "
                "The seed is shown in the simulation log output."
            )

    if decisions and any(term in q for term in ["decision console", "action queue", "priority", "prioritize", "what next", "do next", "next experiment", "experiment ticket", "pilot memo", "memo"]):
        items = decisions.get("items") if isinstance(decisions.get("items"), list) else []
        if items:
            top = items[0]
            severity = str(top.get("severity", "info")).upper()
            source = top.get("source", "Workbench")
            evidence = top.get("evidence", "--")
            action = top.get("action", "--")
            next_step = top.get("next", "--")
            open_actions = decisions.get("open_actions", 0)
            critical = decisions.get("critical_actions", 0)
            return (
                f"Top priority is **{source}** [{severity}]. Evidence: {evidence}. "
                f"Action: {action} Next experiment: {next_step} "
                f"The queue currently has {open_actions} open actions and {critical} critical gates."
            )
        return "The Decision Console is empty because no panel output has been run yet. Run a simulation or upload data, then refresh the console."

    if "4d" in q or "mechanism surface" in q or "mechanism state" in q or "mechanism map" in q or "state map" in q:
        decision = str(diag.get("decision", "")).strip()
        extra = f" In your current run: {decision}" if decision else ""
        return (
            "That plot is a mechanism state map, not a literal 4D object. It compresses four variables: x = cycle, y = SOC window, color = dominant loss mechanism, brightness/height = stress intensity. "
            "Use it to see when the model shifts from SEI/desolvation-dominated loss to P2-O2, JT, or residual-driven loss."
            + extra
        )

    # Voltage fade
    if any(term in q for term in ["voltage fade", "voltage drop", "average voltage", "discharge voltage"]):
        v_end = diag.get("voltage_end")
        if v_end and _as_float(v_end) > 0:
            return (
                f"Average discharge voltage starts at 3.34 V and decreases as cumulative losses grow. "
                f"The current end voltage is {_as_float(v_end):.3f} V. "
                "Each mechanism contributes: P2-O2 lowers voltage the most, followed by JT, SEI, and rate stress. "
                "It is not a separate output metric; it is derived from the cumulative loss terms."
            )
        return (
            "Average voltage fade is derived from cumulative mechanism losses. Run the diagnostics simulation first to see the voltage curve; "
            "it starts at 3.34 V and drops as P2-O2, JT, SEI, and rate losses accumulate."
        )

    # Knee point
    if "knee" in q and ("point" in q or "detect" in q or "what" in q):
        knee = diag.get("knee", diag.get("knee_point"))
        return (
            f"The knee point is where capacity fade accelerates non-linearly (second derivative < -1.6e-5). "
            f"Current value: {knee if knee else 'N/A (not detected yet, run simulation first)'}. "
            "If detected early, it means the cell is entering accelerated degradation and may not reach its target lifetime."
        )

    # RUL
    if "rul" in q or "remaining useful life" in q:
        rul = diag.get("rul80", diag.get("rul"))
        return (
            f"RUL at 80% SOH is the cycle number where capacity first drops below 0.80. "
            f"Current value: {rul if rul else 'not computed yet (run simulation)'}. "
            "If the value shows '>' followed by the cycle count, it means the cell has not reached 80% within the simulated range."
        )

    # EOL
    if "eol" in q or "end of life" in q:
        eol = diag.get("eol_capacity")
        fade = diag.get("fade")
        return (
            f"EOL (end-of-life) capacity is the simulated capacity after all cycles complete. "
            f"Current: {eol if eol else '--'}, total fade: {fade if fade else '--'}. "
            "This is a simulation result, not a measured value. Validate against real cycling data before trusting it."
        )

    if any(term in q for term in ["capacity fade", "degradation", "dominant mechanism", "mechanism contribution", "sei"]):
        decision = str(diag.get("decision", "")).strip()
        losses = {
            "SEI/desolvation": _as_float(diag.get("sei_loss")),
            "P2-O2": _as_float(diag.get("p2_loss")),
            "JT": _as_float(diag.get("jt_loss")),
            "rate": _as_float(diag.get("rate_loss")),
            "residual": _as_float(diag.get("residual_loss")),
        }
        top = max(losses.items(), key=lambda item: item[1])
        return (
            f"Capacity fade is computed by subtracting explicit loss terms each cycle. "
            f"In the current run the largest term is {top[0]} at {top[1]:.2f}% cumulative loss. "
            f"EOL capacity is {diag.get('eol_capacity', '--')} and fade is {diag.get('fade', '--')}. "
            + (decision if decision else "")
        )

    # Uncertainty / propagation
    if "uncertainty" in q or "propagation" in q or "confidence" in q or "error bar" in q:
        return (
            "Uncertainty propagation shows how input variations affect output results. The capacity chart has a shaded band from +/-15% coefficient perturbation. "
            "The recycling panel uses Monte Carlo sampling (200 draws) over feedstock and assay noise to produce a 90% recovery interval. "
            "These help prioritize which parameters need tighter lab measurement before trusting the simulation."
        )

    # Black mass
    if "black mass" in q:
        return (
            "Black mass is the powder produced after shredding and processing spent batteries. It contains mixed metal oxides (Mn, Fe, Na, Al, Cu). "
            "In the recycling panel it serves as the feedstock input. The leaching model estimates how much of each metal can be recovered from it."
        )

    # Bot identity
    if any(term in q for term in ["who are you", "what are you", "are you openrouter", "are you fallback", "are you a bot", "are you ai", "which model"]):
        return (
            "I'm KineticsForge Assist. I use deterministic project rules for exact UI questions and OpenRouter free-model routing for open-ended ones when configured. "
            "The server does not store chat history, but this browser can send recent local turns so follow-up questions work."
        )

    # Ideal / fake / too perfect data concerns
    if any(term in q for term in ["too ideal", "too perfect", "fake", "realistic", "real data", "not real", "made up", "fabricat"]):
        return (
            "The simulations use physics-informed models, not measured lab data. Recovery curves, fade rates, and risk scores come from calibrated equations, not from a specific battery. "
            "They will look smoother and more predictable than real experiments. Validate against your own cycling or leaching data before using these numbers for production decisions."
        )

    # EIS
    if "eis" in q or "impedance" in q or "nyquist" in q:
        return (
            "EIS (Electrochemical Impedance Spectroscopy) features feed the BMS risk model. The simulation tracks R_ct (charge transfer resistance) and R_SEI (SEI layer resistance). "
            "When their sum exceeds the R_ct gate threshold, the cell's risk score rises. Toggle 'EIS diagnostics' in the BMS panel to see the difference."
        )

    # Arrhenius
    if "arrhenius" in q:
        return (
            "Arrhenius scaling models temperature-dependent reaction rates as k = A * exp(-Ea / kT). "
            "In diagnostics, SEI growth uses Ea=0.56 eV by default. Higher temperature exponentially increases the rate. "
            "You can tune Ea in the advanced knobs panel."
        )

    # Butler-Volmer
    if "butler" in q or "volmer" in q or "overpotential" in q:
        return (
            "The Butler-Volmer term estimates rate stress from interfacial overpotential: eta = asinh(I / 2*i0). "
            "It mainly grows at high C-rate (>1.5C). At normal 1C cycling it should not be the dominant loss term. "
            "If it dominates, reduce C-rate or improve interfacial kinetics."
        )

    # P2-O2
    if "p2" in q and ("o2" in q or "phase" in q or "transition" in q):
        return (
            "P2-O2 is a structural phase transition in layered Na-ion cathodes that occurs at high SOC. "
            "It causes irreversible capacity loss and voltage hysteresis. The simulation gates it with a sigmoid around SOC_crit. "
            "To reduce it: lower upper cutoff voltage or add Al/Ti stabilization."
        )

    # Jahn-Teller
    if "jahn" in q or "teller" in q or "jt" in q:
        return (
            "Jahn-Teller distortion is a geometric distortion of Mn3+ octahedra that causes structural degradation. "
            "It increases with Mn content and temperature, and decreases with Fe substitution and dopants. "
            "The JT index in materials screening shows how exposed a composition is to this effect."
        )

    # Desolvation
    if "desolv" in q or "desolvation" in q or "solvation" in q:
        return (
            "Na+ desolvation is the energy barrier for sodium ions to shed their solvent shell and intercalate into the cathode. "
            "Higher barrier means slower kinetics and more loss. Fe substitution and dopants lower the barrier. "
            "It is coupled to SEI loss in the cumulative mechanism chart."
        )

    # Shrinking core
    if "shrinking" in q and "core" in q:
        return (
            "The shrinking-core model treats each black-mass particle as a reacting sphere with a growing product shell. "
            "Conversion follows 1 - (1-X)^(1/3) = k*C_acid*t, so smaller particles, higher acid concentration, longer time, and higher temperature all increase recovery."
        )

    # Thermal runaway
    if "thermal runaway" in q or "runaway" in q:
        return (
            "Thermal runaway is not explicitly modeled here. The BMS panel models thermal coupling between cells and flags risk before runaway. "
            "A cell crossing the risk threshold means it should be inspected or cooled, not that runaway has started."
        )

    # Score meaning
    if "score" in q and ("mean" in q or "what" in q or "how" in q):
        return (
            "The objective score is a weighted sum: capacity (32%), stability (32%), fade resistance (22%), and cost efficiency (14%), minus a charge-balance penalty. "
            "Higher is better. You can change the weights in the advanced knobs to match your priorities."
        )

    # Cost
    if "cost" in q and ("how" in q or "what" in q or "calculate" in q or "usd" in q or "inr" in q):
        return (
            "Cost is estimated from raw material prices (Na, Mn, Fe, dopant) per kg of cathode, divided by energy density to get $/kWh. "
            "For recycling, cost includes acid, heating energy, and processing per batch in INR. These are proxy estimates, not quotes."
        )

    if "pareto" in q:
        return (
            "The Pareto front compares cathode candidates where improving one objective usually hurts another. "
            "Here the main tradeoff is capacity versus stability/fade/cost. Points near the front are candidates worth deeper screening; the selected point shows your current Na/Mn/Fe recipe."
        )

    if "composition landscape" in q:
        return (
            "The composition landscape sweeps Na and Mn while Fe is adjusted by charge balance. Height and color are the objective score, so peaks are better synthesis candidates under the current weights."
        )

    if any(term in q for term in ["material", "composition", "cathode", "synthesis", "oxygen risk", "charge"]):
        if mat:
            fade500 = _as_float(mat.get("fade500_pct", mat.get("fade500")))
            if 0 < fade500 <= 1:
                fade500 *= 100
            return (
                f"The selected cathode is scored from capacity, stability, fade, cost, oxygen risk, and charge balance. "
                f"Current score is {_as_float(mat.get('score')):.3f}, stability {mat.get('stability', '--')}, "
                f"fade@500 {fade500:.1f}%, oxygen risk {_as_float(mat.get('oxygen_risk')):.2f}, "
                f"charge risk {_as_float(mat.get('charge_risk')):.2f}. "
                f"{mat.get('decision', '')}"
            )
        return "The materials screen ranks Na/Mn/Fe cathodes by capacity, stability, fade, oxygen risk, charge balance, and cost."

    if any(term in q for term in ["recycling", "recovery", "leach", "acid", "purity", "margin"]):
        if rec:
            purity = _as_float(rec.get("purity_proxy", rec.get("purity"))) * 100
            margin = _as_float(rec.get("margin_proxy_inr", rec.get("margin_proxy")))
            recovered = _as_float(rec.get("recovered_kg", rec.get("recovery_kg", rec.get("recovered_mass_kg"))))
            return (
                f"Recycling uses shrinking-core leaching: recovery rises with acid concentration, temperature, time, and smaller particles. "
                f"Current recovered mass is {recovered:.1f} kg, interval {rec.get('interval_kg') or '--'}, "
                f"purity proxy {purity:.1f}%, margin proxy INR {margin:.0f}. "
                f"{rec.get('decision', '')}"
            )
        return "The recycling screen estimates Mn/Fe/Na recovery from leach kinetics, then applies uncertainty, purity, and cost checks."

    if any(term in q for term in ["pending", "staged", "synthetic", "validation gate"]):
        return (
            "Those labels are readiness gates, not missing UI. Runnable panels work now in the app; simulation-only means the physics mirror works but real pack/lab validation is still required before claiming field performance."
        )

    # What is this
    if "what is this" in q or "what does this" in q or "what is it" in q or "what am i looking at" in q:
        panel = _active_panel(state)
        if panel == "bms":
            return "You are looking at the BMS pack monitoring panel. It simulates a battery pack as a thermal graph, tracking cell temperatures, EIS drift, and risk scores. Run the simulation to see which cell needs attention first."
        if panel == "materials":
            return "You are looking at the materials screening panel. It scores Na-ion cathode compositions by capacity, stability, fade, cost, and defect risk. Adjust Na/Mn/Fe sliders and run screening to compare candidates."
        if panel == "recycling":
            return "You are looking at the recycling optimizer. It estimates metal recovery from black mass using shrinking-core leaching kinetics, Bayesian priors, and Monte Carlo uncertainty."
        if panel == "upload":
            return "You are looking at the BYOD upload panel. It fingerprints cycler columns, extracts tier-1 battery features, plots dQ/dV when traces are available, and marks prediction confidence instead of hiding missing data."
        if panel == "decisions":
            return "You are looking at the Decision Console. It turns current panel outputs into an action queue with evidence, owner, and next experiment, then exports tickets or a memo without adding hidden validation claims."
        if panel == "diagnostics":
            return "You are looking at the diagnostics panel. It simulates capacity fade for Na-ion cathodes under specified temperature, C-rate, and cycle conditions, and identifies the dominant degradation mechanism."
        return "KineticsForge is a battery decision workbench. It helps choose: which degradation mechanism is dominating, which pack cell needs attention, which cathode is worth making, and which recycling recipe is worth running."

    # Chart / plot generic
    if any(term in q for term in ["chart", "plot", "graph", "curve", "line", "orange line", "red line"]):
        panel = _active_panel(state)
        if panel == "diagnostics":
            return "The diagnostics charts show: (1) capacity fade over cycles with an uncertainty band, (2) average discharge voltage, (3) cumulative loss by mechanism (SEI=orange, P2-O2=red, JT=purple, rate=green, residual=blue), and (4) the mechanism state map."
        if panel == "bms":
            return "The BMS charts show: (1) the 3D thermal coupling view with cell tiles, and (2) a trend line of max risk (red) and peak temperature (orange) over time."
        if panel == "materials":
            return "The materials charts show: (1) a Pareto scatter of capacity vs stability (red dots = front, your pick = large dot), and (2) a composition landscape where height/color = objective score."
        if panel == "recycling":
            return "The recycling chart shows shrinking-core conversion over leach time for Mn (red), Fe (orange), and Na (blue). Higher curves mean faster recovery."
        if panel == "upload":
            return "The upload panel chart is the dQ/dV fingerprint. Peaks are diagnostic voltage signatures; if the chart is empty, the upload did not include enough discharge voltage-capacity trace points."
        if panel == "decisions":
            return "The Decision Console tables are not model charts. They summarize current evidence, action gates, owners, and experiment tickets derived from the panel outputs."
        return "Each panel has specific charts. Navigate to a panel and ask again for details."

    # Calibration
    if "calibrat" in q:
        return (
            "Calibration fits the simulation coefficients (SEI scale, P2 rate, JT scale, stress exponent) to your experimental data. "
            "Paste cycle,capacity rows into the calibration box and click Calibrate. It runs a grid search and reports RMSE and R-squared."
        )

    # Download / export
    if "download" in q or "export" in q or "csv" in q:
        return (
            "Each panel has CSV/JSON export for its own result. Diagnostics exports cycle-by-cycle capacity and mechanism breakdown; Upload exports parsed summaries and features; BMS exports cell risk, temperature, and impedance per timestep; Materials exports selected composition properties; Recycling exports element-level recovery. "
            "The Decision Console can also export experiment tickets as CSV and decision memos as Markdown or JSON."
        )

    return None


UI_MODELS = [
    "liquid/lfm-2.5-1.2b-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "deepseek/deepseek-v4-flash:free",
]
PHYSICS_MODELS = [
    "deepseek/deepseek-v4-flash:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
]
GENERAL_MODELS = [
    "deepseek/deepseek-v4-flash:free",
    "google/gemma-4-26b-a4b-it:free",
    "meta-llama/llama-3.2-3b-instruct:free",
]


def _intent(question: str) -> str:
    q = question.lower().strip()
    if re.fullmatch(r"(hi|hello|hey|yo|sup|namaste|thanks|thank you)[!. ]*", q):
        return "greeting"
    if any(k in q for k in ["how do i", "how to", "use", "where", "navigate", "start", "button", "download"]):
        return "how_to"
    if any(k in q for k in ["should", "what next", "good", "bad", "worth", "synthesize", "run this", "do now"]):
        return "action"
    if any(k in q for k in ["compare", "versus", "vs", "better", "worse"]):
        return "compare"
    if any(k in q for k in ["why", "red", "glow", "highlight", "current", "result", "this", "chart", "plot"]):
        return "current"
    if any(k in q for k in ["explain", "what is", "formula", "equation", "model", "kinetics", "shrinking", "arrhenius", "butler", "sei", "p2", "jahn"]):
        return "concept"
    return "outside"


def _active_panel(state: Optional[Dict[str, Any]]) -> str:
    if isinstance(state, dict):
        section = str(state.get("section", "")).replace("sec-", "").strip().lower()
        if section:
            return section
    return "general"


def _fmt_state_value(value: Any, suffix: str = "", digits: int = 2) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    val = _as_float(value, float("nan"))
    if math.isfinite(val):
        return f"{val:.{digits}f}{suffix}"
    return "--"


def _local_answer(question: str, sections: List[Dict[str, str]], state: Optional[Dict[str, Any]] = None) -> str:
    intent = _intent(question)
    panel = _active_panel(state)
    direct = _direct_ui_answer(question, state)
    if direct:
        return direct
    if intent == "greeting":
        return "Hi. Ask me about the panel you are viewing, a highlighted cell, a formula, or what to change next."

    diag = state.get("diagnostics", {}) if isinstance(state, dict) else {}
    bms = state.get("bms", {}) if isinstance(state, dict) else {}
    mat = state.get("materials", {}) if isinstance(state, dict) else {}
    rec = state.get("recycling", {}) if isinstance(state, dict) else {}
    upload = state.get("upload", {}) if isinstance(state, dict) else {}

    if intent == "how_to":
        return "Use Upload for real cycler files, Diagnostics for editable physics simulations, BMS for cell risk, Materials for cathode screening, and Recycling for leach economics. Run the panel, then read the decision box and confidence/warning labels."
    if intent == "action":
        if panel == "bms" or (not mat and bms):
            return f"Use the highest-risk cell first. Current max risk is {_fmt_state_value(bms.get('max_risk'), digits=3)} against gate {_fmt_state_value(bms.get('threshold'), digits=2)}; cool or inspect if it crosses the gate."
        if panel == "materials" or (mat and panel not in ('bms', 'recycling', 'upload', 'diagnostics')):
            return f"Current composition is Na={_fmt_state_value(mat.get('na'))}, Mn={_fmt_state_value(mat.get('mn'))}, Fe={_fmt_state_value(mat.get('fe'))}. Score is {_fmt_state_value(mat.get('score'), digits=3)}; synthesize only if stability is high and fade/oxygen/charge risks are controlled."
        if panel == "recycling" or rec:
            return f"Current recovery is {_fmt_state_value(rec.get('recovered_kg'), ' kg', 1)}, purity proxy {_fmt_state_value(_as_float(rec.get('purity_proxy')) * 100, '%', 1)}, margin INR {_fmt_state_value(rec.get('margin_proxy_inr'), digits=0)}. Pilot only if recovery, purity, and margin are all positive."
        if panel == "upload" or upload:
            return f"The upload parser read {upload.get('rows_read', '--')} rows with schema score {_fmt_state_value(upload.get('schema_score'), digits=2)} and {upload.get('features_present', 0)} features present. Treat outputs with warnings as triage, then export the CSV for your lab notebook."
        return f"Current fade is {_fmt_state_value(diag.get('fade'))} at {_fmt_state_value(diag.get('temperature_C'), ' C', 0)} and {_fmt_state_value(diag.get('c_rate'), 'C', 1)}. Follow the dominant mechanism in the decision box before changing chemistry or cycling conditions."
    if intent == "current":
        if panel == "bms" and bms:
            return f"The BMS panel is showing {bms.get('cells', '--')} cells over {bms.get('duration_seconds', '--')} s. Max risk is {_fmt_state_value(bms.get('max_risk'), digits=3)}; the gate is {_fmt_state_value(bms.get('threshold'), digits=2)}."
        if panel == "materials" and mat:
            return f"The selected cathode is Na={_fmt_state_value(mat.get('na'))}, Mn={_fmt_state_value(mat.get('mn'))}, Fe={_fmt_state_value(mat.get('fe'))}. Capacity is {mat.get('capacity', '--')}, stability {mat.get('stability', '--')}, score {_fmt_state_value(mat.get('score'), digits=3)}."
        if panel == "recycling" and rec:
            return f"The recipe uses {rec.get('mass_kg', '--')} kg black mass, {rec.get('acid_molarity', '--')} M acid, and {rec.get('temperature_C', '--')} C. Recovery is {_fmt_state_value(rec.get('recovered_kg'), ' kg', 1)} with interval {rec.get('interval_kg') or '--'}."
        if panel == "upload" and upload:
            return f"The upload result is {upload.get('filename', 'the current file')}: {upload.get('rows_read', '--')} rows, {upload.get('features_present', 0)} features, SOH {_fmt_state_value(_as_float(upload.get('soh')) * 100, '%', 1)}, confidence {_fmt_state_value(_as_float(upload.get('confidence')) * 100, '%', 0)}."
        return f"Diagnostics is set to {_fmt_state_value(diag.get('temperature_C'), ' C', 0)}, {_fmt_state_value(diag.get('c_rate'), 'C', 1)}, {diag.get('cycles', '--')} cycles. Current EOL capacity is {diag.get('eol_capacity', '--')} and fade is {diag.get('fade', '--')}."
    if intent == "concept":
        q = question.lower()
        if "shrinking" in q or "leach" in q:
            return "The shrinking-core model treats each black-mass particle as a reacting core with a product layer around it. Conversion follows 1 - (1-X)^(1/3) = kCt, so smaller particles, higher acid, longer time, and higher temperature increase recovery."
        if "butler" in q or "bv" in q:
            return "The Butler-Volmer term estimates rate stress from interfacial overpotential. In this app it mainly grows at high C-rate, so it should not dominate a normal 1C run."
        if "sei" in q:
            return "SEI loss is irreversible sodium inventory loss from interphase growth. It follows Arrhenius temperature scaling plus square-root cycle growth, so hot long cycling raises it strongly."
        return "That plot compresses the panel physics into an engineering readout: inputs on the left, simulated response on the chart, and the decision box for what to change next."
    if intent == "compare":
        return "Compare by changing one control at a time and watching the decision box plus the main metric. For materials, compare score/fade/stability; for BMS, compare max risk and Tmax; for recycling, compare recovery, purity, and margin."
    return "I can help with the panels here: diagnostics, BMS, materials, or recycling. What are you looking at?"


def pick_model(question: str) -> List[str]:
    override = os.environ.get("OPENROUTER_MODEL", "").strip()
    intent = _intent(question)
    q = question.lower()
    if any(k in q for k in ["equation", "formula", "arrhenius", "butler", "volmer", "sei", "p2", "o2", "jahn", "shrinking", "kinetics", "thermal", "ode", "physics"]):
        models = PHYSICS_MODELS
    elif intent in {"how_to", "current", "greeting"} or any(k in q for k in ["button", "panel", "chart", "plot", "glow", "highlight", "download", "where"]):
        models = UI_MODELS
    else:
        models = GENERAL_MODELS
    ordered = [override] if override else []
    for model in models:
        if model and model not in ordered:
            ordered.append(model)
    return ordered or ["openrouter/free"]


def _guard_cloud_answer(content: str, deterministic: Optional[str]) -> str:
    if not deterministic:
        return content
    if deterministic.startswith("Hi."):
        return deterministic
    if content[:1] in {":", ";", ",", "-", "—"}:
        return deterministic
    nums = re.findall(r"\d+\.\d+|\d+%", deterministic)
    if nums and not any(n in content for n in nums[:4]):
        return deterministic
    if len(content.split()) > 95:
        return deterministic
    return content


def _messages(question: str, sections: List[Dict[str, str]], section: str = "general", state: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    project_context = "\n\n".join(f"{s['title']}: {s['text']}" for s in sections)
    state_context = _state_summary(state) or "No current UI state was supplied."
    deterministic = _direct_ui_answer(question, state)
    system = (
        "You are KineticsForge Assist — an expert electrochemical engineer and adaptive AI collaborator "
        "embedded in a battery physics workbench. You validate engineering concerns directly, call out "
        "flawed physical assumptions gently but firmly, and use structured formatting (bold for emphasis, "
        "bullet points for lists) to keep answers scannable. "
        "The server does not persist memory, but the prompt may include recent browser-supplied "
        "conversation turns; use them only for local follow-up context. Use the provided project context "
        "first for KineticsForge questions and the UI state second for current results. If the question "
        "is outside the project, answer normally but keep it concise. Do not invent validation, safety, "
        "funding, or production claims. When a claim is uncertain, say what would need validation. If a "
        "result is simulated, heuristic, or checkpoint-based, say so. Answer in 2-5 sentences; if the user "
        "wants more detail, they will ask. If a deterministic UI readout is supplied, preserve its numbers "
        "and action meaning. Never start with 'KineticsForge is' because the user already knows the app. "
        "If you don't know, say so in one sentence. Don't fabricate. Use plain, direct technical language, "
        "not brochure language. When showing numbers, reference the actual current simulation state."
    )
    if deterministic:
        state_context += "\n\nDeterministic UI readout to preserve: " + deterministic
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "Active section: " + section[:80] + "\n\nProject context:\n" + project_context + "\n\nCurrent UI state:\n" + state_context},
        {"role": "user", "content": question[:1200]},
    ]


def _memory_mode(state: Optional[Dict[str, Any]]) -> str:
    if isinstance(state, dict) and isinstance(state.get("conversation"), list) and state.get("conversation"):
        return "browser_recent_turns"
    return "off"


def answer_chat(question: str, section: str = "general", state: Optional[Dict[str, Any]] = None) -> Dict[str, object]:
    clean_question = (question or "").strip()
    sections = retrieve_context(clean_question)
    context_titles = [section["title"] for section in sections]
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    direct = _direct_ui_answer(clean_question, state)
    if not direct and _intent(clean_question) == "greeting":
        direct = _local_answer(clean_question, sections, state)

    if not api_key:
        return {
            "answer": direct or _local_answer(clean_question, sections, state),
            "source": "local_setup_fallback",
            "model": "none",
            "memory": _memory_mode(state),
            "context": context_titles,
            "setup_required": True,
        }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "KineticsForge Assistant",
        "X-OpenRouter-Title": "KineticsForge Assistant",
    }
    referer = os.environ.get("OPENROUTER_SITE_URL", "").strip()
    if referer:
        headers["HTTP-Referer"] = referer
    last_exc: Optional[Exception] = None
    for model in pick_model(clean_question):
        payload = {
            "model": model,
            "messages": _messages(clean_question, sections, section=section, state=state),
            "temperature": float(os.environ.get("OPENROUTER_TEMPERATURE", "0.20")),
            "max_tokens": min(360, int(os.environ.get("OPENROUTER_MAX_TOKENS", "260"))),
        }
        request = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=float(os.environ.get("OPENROUTER_TIMEOUT", "12"))) as response:
                data = json.loads(response.read().decode("utf-8"))
            content = (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
            if not content:
                raise ValueError("OpenRouter returned an empty response")
            content = _guard_cloud_answer(content, direct)
            return {
                "answer": content,
                "source": "openrouter",
                "model": data.get("model", model),
                "memory": _memory_mode(state),
                "context": context_titles,
                "setup_required": False,
            }
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
            last_exc = exc
            continue
    return {
        "answer": direct or _local_answer(clean_question, sections, state),
        "source": "local_setup_fallback",
        "model": "openrouter_retry_failed",
        "memory": _memory_mode(state),
        "context": context_titles,
        "setup_required": False,
        "warning": f"OpenRouter models failed; answered from compact fallback ({last_exc.__class__.__name__ if last_exc else 'unknown'}).",
    }
