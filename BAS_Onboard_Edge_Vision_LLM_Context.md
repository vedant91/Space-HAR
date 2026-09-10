# Bharatiya Antariksh Station (BAS) — Onboard Edge Vision & Procedure AI
## LLM Context Pack / Engineering Brief

**Document type:** Self-contained briefing for LLMs and engineers  
**Date context:** September 2026  
**Scope:** Local (onboard) camera AI for astronaut movement tracking, object tracking, procedure validation, and anomaly guidance — so closed-loop safety does not depend on Earth streaming  
**Usage:** Feed this entire file as context to an LLM when designing, coding, pitching, or reviewing the system. Do not assume access to any local code repository unless the user attaches it separately.

---

## 1. One-sentence mission

Build an **offline, onboard** vision system that watches a fixed (and later FPV) camera during a science/ops procedure, validates step sequence in real time, detects anomalies, and gives voice/UI guidance — because Earth-in-the-loop is too slow and too intermittent to prevent irreversible mistakes.

---

## 2. Problem framing (why local AI)

### 2.1 Latency reality (documented ranges)

| Regime | Typical delay | Implication |
|--------|---------------|-------------|
| LEO via TDRSS Ku-band | ~0.6–1 s on RF path; coverage not continuous | Too slow for tight “stop now” safety loops; outages break continuity |
| LEO practical end-to-end (video/ops/cloud AI) | Often **2–10+ s** depending on path, processing, and payload (HD ~1 s class; UHD events reported **10+ s**) | Supervisory coaching possible; closed-loop safety unreliable |
| Ground robotics teleop via ISS relays | Typically **2–10 s** RTT; operators use bump-and-wait | Proves delayed command loops are inefficient and unsafe for irreversible steps |
| Earth–Moon | ~**2.6 s** light RTT; ops studies often use 4–8 s | Predictive overlays help; true closed loop does not return |
| Mars | Minutes one-way | Local autonomy mandatory |

**Engineering thesis:** Perception + procedure state + anomaly + HMI must run **onboard**. Earth is an **asynchronous audit / model-update** channel, not the real-time brain.

### 2.2 Bandwidth ≠ control authority

High downlink rates still face: intermittent contact, priority contention, DTN store-and-forward, and human reaction after delay. Streaming raw experiment video to Earth for a ground model to shout “stop” is a **supervisory** loop, not a **safety** loop. Prefer shipping **structured logs + anomaly clips**, not continuous 4K.

### 2.3 Disaster modes latency cannot fix in time

Use as motivating cases (first-principles; not claimed historical BAS incidents):

- Irreversible experiment states (wrong valve/reagent sequence, contamination, sealed sample ruined)
- Micro-g object kinematics (floating tools entering intakes, racks, or crew space)
- Procedure skip / wrong step under fatigue when remote correction arrives after the bad state
- Any time-critical ECLSS / medical / suit urgency where seconds matter

### 2.4 BAS program context (documented as of 2025–2026 reporting)

- **Bharatiya Antariksh Station (BAS):** Indian modular LEO station
- **BAS-01** first module targeted ~**2028**; full ~**5-module** complex targeted ~**2035**
- Tied to Gaganyaan human spaceflight follow-on; racks planned for microgravity experiments (life sciences, pharma, materials, manufacturing, etc.)
- **IMEx-2026:** ISRO AO inviting Indian microgravity experiment proposals that may progress toward LEO / BAS opportunities
- Early modules will be **crew-time- and bandwidth-constrained** → onboard assistants have high leverage

---

## 3. Related systems (do not reinvent the wrong product)

| System | What it is | Local vs ground | Gap vs this project |
|--------|------------|-----------------|---------------------|
| **CIMON / CIMON-2** (DLR/Airbus/IBM) | Free-flyer voice assistant; shows procedures; mobile camera | Speech/AI historically ground cloud (~2 s after optimization); nav more onboard | Strong HMI precedent; weak as hard real-time **visual safety net** while cloud-dependent |
| **Astrobee** (NASA) | Free-flyer research platform; cameras; guest science | Autonomous or ground-commanded | Eyes/mobility ≠ procedure-state anomaly coach |
| **Sidekick** (NASA + Microsoft HoloLens) | Procedure Mode (local holographic steps) + Remote Expert (ground annotates FPV) | Procedure Mode can be standalone; Remote Expert needs link | Validates local procedure UX; Remote Expert is what latency breaks |
| **Int-Ball / Int-Ball2** (JAXA) | Cabin camera free-flyer | Imaging often ground-routed historically | Sensing, not full onboard procedure ML |
| **Xiaohang / Wukong AI** (China Tiangong programs) | Flying robot / onboard+ground LLM Q&A | Split orbit vs ground for urgent vs deep | Language/knowledge layer; not published as real-time camera anomaly gating |
| **Industrial edge inspection** (Jetson / MES / E-stop patterns) | Defect detect + reject in tens of ms | Fully local | Best **architectural** analogy for closed-loop visual safety |
| **HPE Spaceborne Computer-2** | Onboard HPC/AI on ISS; edge→cloud federated patterns | Onboard inference; Earth for heavy training/updates | Validates “process onboard, downlink insights” ops model |
| **MicroG-4M** (2025 benchmark) | Microgravity human action dataset/benchmark | N/A | Shows Earth-trained action models **degrade sharply** in micro-g semantics |

**Honest gap:** No widely published system is a mature **onboard, vision-first, procedure-state machine with hard anomaly gating and fail-safe “I’m unsure”** for crew science procedures. Pieces exist; the integrated safety product is the opportunity.

---

## 4. Product definition

### 4.1 Primary user

Astronaut / crew operator executing a **pre-defined multi-step experiment or maintenance protocol** at a rack/bench, with intermittent or delayed ground support.

### 4.2 Primary job-to-be-done

1. Track the current procedure step from camera evidence  
2. Suggest the next step  
3. Detect skip / wrong-order / missing-object / forbidden-zone / low-confidence situations  
4. Alert via voice (and UI) **immediately**  
5. Log structured outcomes + store video locally; optionally stream to a ground IP for monitoring only  
6. Never depend on Earth RTT for the closed loop  

### 4.3 Non-goals (MVP)

- Replacing flight computers or ECLSS control  
- Medical diagnosis from video  
- Fully autonomous robot manipulation  
- Cloud LLM as the real-time decision authority  

---

## 5. Capability map (what to build)

### P0 — must have for the thesis

- Procedure step recognition + wrong-step / skip / out-of-order detection  
- Hand / tool / sample ROI monitoring + forbidden-zone intrusion → HOLD  
- Confidence-gated alerts + “UNSURE → ask crew / hold” (never invent a step)  
- Timestamped structured log + synchronized anomaly video snippets  
- Offline standalone inference (edge)  

### P1 — strong differentiators

- Object identity + presence checklist (missing / wrong / left FOV)  
- Typed anomaly set: contamination proxy gestures, wrong connector mating, dwell-time exceeded, occlusion  
- Local voice coach + confirmations  
- Tablet/HUD Procedure Mode (Sidekick-lite)  
- Bandwidth-smart uplink policy (logs + clips, not continuous raw)  

### P2 — later

- Multi-person role awareness  
- Multi-camera / FPV fusion  
- Fatigue / tempo soft signals (advisory only; privacy-sensitive)  
- Free-flyer “camera agent” story  

### P3 — strategic

- Cross-experiment transferable backbone + per-experiment procedure packs  
- Predictive floating-object trajectories (needs micro-g analog demo)  
- Federated / delayed retrain with ground  

---

## 6. Recommended architecture (layers)

```
Camera / FPV frame
        │
        ├─► Classical CV (markers, HSV/color, ROIs, lid/open state, optical flow)
        │
        └─► Lightweight deep perception
                • pose / wrists-hands
                • object detect
                • short temporal step classifier (windowed)
                        │
                        ▼
              Evidence tokens (confidences, detections)
                        │
                        ▼
           Explicit Procedure State Machine (authority)
           hold-and-correct; never silent-skip
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   Voice alert     GUI / HUD      Structured log
   + confirm       live status    + anomaly clips
                                  (+ optional UDP/IP stream for MONITORING only)
```

### Design rules

1. **Models propose; FSM decides.**  
2. **Uncertainty is a first-class state.** Low confidence → HOLD / CONFIRM, not guess.  
3. **Earth is audit, not control.** Streaming is optional monitoring.  
4. **Hybrid perception.** Classical CV for critical, explainable object/lid/zone cues; deep models for pose/activity.  
5. **Shadow → advisory → gating.** Industrial inspection rollout pattern.  
6. **Watchdog.** If inference dies → degrade to checklist UI + raw recording; never fail open into false confidence.

---

## 7. Anomaly taxonomy (starter set)

| ID | Anomaly | Typical evidence | Suggested action |
|----|---------|------------------|------------------|
| A1 | Wrong step / out of order | Step classifier ≠ expected; object state contradicts | Voice warn + HOLD + show correct step |
| A2 | Skipped step | FSM advanced without precondition evidence | HOLD + require recovery path |
| A3 | Missing / wrong object | Checklist fail; wrong class in ROI | Warn; block advance |
| A4 | Forbidden-zone intrusion | Hand/tool enters sterile/hazard ROI | Immediate HOLD |
| A5 | Dwell / stuck | No progress beyond timeout | Soft prompt; escalate if persists |
| A6 | Object left FOV unexpectedly | Track loss during critical step | Warn; optional pause |
| A7 | Occlusion / bad view | Low detectability / camera blocked | Ask crew to clear view; abstain |
| A8 | Model disagreement / low confidence | Ensemble or score below threshold | CONFIRM with crew; do not auto-advance |
| A9 | Unexpected second actor | Extra person in frame (optional P2) | Advisory |

---

## 8. Procedure pack format (conceptual)

Each experiment should be a **data pack**, not a code fork:

```yaml
experiment_id: bas_box_sort_v1   # example id only
name: "Sample sealing protocol"
steps:
  - id: 1
    name: "Approach closed lid"
    required_objects: [main_box, lid_closed]
    forbidden_zones: []
    success_evidence: [hand_near_lid, lid_closed_visible]
    timeout_s: 30
  - id: 2
    name: "Open lid"
    preconditions: [step_1_complete]
    required_objects: [lid_open]
    ...
alerts:
  voice: true
  confirm_irreversible: true
uplink_policy:
  continuous_video: false
  anomaly_clips: true
  structured_log: true
```

**Goal:** Swap experiments without retraining the entire stack; retrain/adapt only perception heads or few-shot tool embeddings as needed.

---

## 9. Training & data strategy (without rare real micro-g video)

| Strategy | Role | Honesty note |
|----------|------|--------------|
| Synthetic renderer with exact joint/object GT | Pose / detector pretrain | Best free labels |
| Domain randomization (orientation, lighting, textures, camera extrinsics) | Robustness to micro-g orientations | Required; PAM/MicroG-4M show inversion/gravity priors fail |
| Gravity-mimic / harness / underwater / air-bearing analogs | Procedure semantics on real pixels | Analog ≠ true micro-g physics — say so |
| Public micro-g corpora (e.g. MicroG-4M) | Domain eval / fine-tune | Earth Kinetics/AVA transfer is fragile |
| Marker-augmented tools in demo | Bootstraps reliability for judges | Engineering pragmatism |
| Pseudo-labels from classical CV on real demo video | Bridge synthetic→real | Report quality metrics; masked loss for uncertain joints |
| Self-supervised “normal procedure” embeddings | Novel anomaly flags | Useful add-on, not sole safety authority |

**Do not claim** terrestrial HAR transfers unchanged to microgravity.

---

## 10. Compute, latency, and flight honesty

### Demo / SIH target

- Laptop CPU or **Jetson Orin Nano/NX-class**  
- ONNX / TensorRT  
- Aim **≤100 ms/frame** class for closed loop (industrial edge often targets tens of ms for reject loops; 10–15 FPS multi-stream is acceptable if alerts are timely)  
- Local ASR for voice if possible; cloud ASR is a demo convenience only  

### Flight-facing honesty

- Jetson-class is **COTS**, suitable for **payload AI / demo TRL narrative**, not as sole critical flight computer  
- Pair with rad-tolerant watchdog MCU/FPGA patterns, ECC, duty-cycle, graceful restart (Spaceborne lesson: software resilience + monitoring)  
- Critical actuations (if any ever) remain outside the GPU path  

---

## 11. HMI requirements

- **Voice:** next step; anomaly; severity (info / warn / hold)  
- **GUI:** live feed, step checklist, detections, latency/FPS, logs  
- **Confirmations** on irreversible steps  
- Language: assistive, interruptible — never nag into alert fatigue  
- Optional: tablet AR step highlight (Procedure Mode); Remote Expert only as secondary, link-dependent mode  

---

## 12. Logging & uplink policy

**Always local:**

- Structured event log (UTC timestamps; step id; status; confidence; anomaly codes)  
- Full video recording on local storage  
- Ring buffer for pre-roll around anomalies  

**To Earth (when link available):**

- Structured log  
- Anomaly clips only (default)  
- Optional low-rate monitoring stream to a specified IP (supervisory, not control)  

This makes the bandwidth thesis tangible in demos.

---

## 13. Evaluation metrics (judge / ops relevant)

Do **not** optimize only mAP.

| Metric | Why |
|--------|-----|
| Step sequence accuracy / FSM complete-correct rate | Core job |
| Time-to-detect anomaly (ms) | Latency thesis |
| Precision @ alert / false hold rate | Alert fatigue risk |
| Missed irreversible anomalies | Safety |
| Abstain quality (when uncertain, did it hold?) | Over-trust risk |
| Per-frame / end-to-end latency p50/p95 | Edge feasibility |
| Local vs delayed-coach race win rate | Narrative proof |

---

## 14. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| False alarms / alert fatigue | Shadow mode first; rate limits; only gate safety-critical classes |
| Over-trust | Confirm irreversible steps; UI says advisory; never auto-actuate hardware in MVP |
| Earth-gravity pose priors fail when inverted | Domain randomization; evaluate sideways/inverted subjects; MicroG-aware testing |
| Opaque end-to-end model as sole authority | Explicit procedure graph + evidence tokens |
| Privacy / crew acceptance | Purpose limitation; optional face blur in exported logs; crew-owned retention |
| Silent wrong outputs under compute faults | Watchdog; classical fallbacks; confidence thresholds |
| “Just stream 4K to ground” | Demo local vs +2–8 s delayed coach race |

---

## 15. Phased roadmap

### Phase 0 — Narrative + MVP closed loop

One protocol, one disaster story, one defended latency number. Single camera → perception → FSM → voice + log + GUI. Offline. Intentional mistake injection.

### Phase 1 — Anomaly depth + trust

Typed anomalies, confidence gating, anomaly clips, occlusion/orientation stress tests, shadow→advisory modes.

### Phase 2 — Experiment packs

Second protocol via pack format; prove transfer without full rebuild.

### Phase 3 — Dual view / FPV

Rack + wearable/FPV fusion; optional free-flyer camera agent story.

### Phase 4 — Ops realism

Uplink budget demo; model update story; failure-modes card; flight-honesty (COTS AI + watchdog) narrative.

---

## 16. Top build priorities (ranked)

1. **Local vs delayed coach race demo** (2–8 s injected delay) — makes the thesis undeniable  
2. **Procedure FSM + wrong-step/skip detector** with hold-and-correct  
3. **Forbidden-zone HOLD**  
4. **Confidence-gated UNSURE → CONFIRM policy**  
5. **Forensics ring-buffer + anomaly-clip uplink policy**  
6. **Synthetic + domain-randomized + marker training pipeline**  
7. **Edge export (ONNX/TensorRT) + watchdog stub**  
8. **Sidekick-lite tablet Procedure Mode**  

---

## 17. Suggested demo script (for humans or LLMs designing a pitch)

1. Show crew executing protocol with **onboard** coach — mistake at step N → alert in &lt;100–300 ms class, HOLD, recovery guidance.  
2. Replay same mistake with **Earth path simulated +3–8 s** — bad state already committed before “stop” arrives.  
3. Show structured log + 10–20 s anomaly clip as the only “downlink,” vs full raw video size.  
4. Show intentional low-confidence case → system asks for confirm instead of guessing.  
5. Flip lighting / body orientation → show robustness or honest failure + fallback.

---

## 18. Speculative vs documented (honesty box)

| Claim | Status |
|-------|--------|
| LEO RF path ~0.6–1 s; practical ops/video often multi-second | Documented (NASA networking / ISS video latency reports) |
| Ground robotics teleop delays ~2–10 s | Documented (ISECG / NASA robotics latency literature) |
| CIMON speech AI historically ground-cloud dependent | Documented |
| Sidekick Procedure Mode vs Remote Expert split | Documented (NASA) |
| BAS-01 ~2028, full station ~2035, microgravity racks | Documented (ISRO / government reporting) |
| MicroG-4M: terrestrial models degrade in micro-g | Documented (2025 paper/dataset) |
| No public full onboard vision procedure-anomaly safety product | Assessment (absence of evidence) |
| Specific BAS disaster scenarios | Motivating hypotheses — not historical claims |
| Jetson as crewed critical-path computer | Not established — frame as payload AI + supervisor |

---

## 19. Key public references (URLs)

- ISRO IMEx-2026: https://www.isro.gov.in/IndianMicrogravityExperiments_IMEx2026.html  
- DLR CIMON: https://www.dlr.de/en/research-and-transfer/projects-and-missions/horizons/cimon  
- NASA Astrobee: https://www.nasa.gov/astrobee/  
- NASA Sidekick: https://www.nasa.gov/news-release/nasa-microsoft-collaborate-to-bring-science-fiction-to-science-fact/  
- MicroG-4M: https://arxiv.org/html/2506.02845v4  
- HPE Spaceborne Computer-2 (edge AI / downlink reduction narrative): HPE newsroom releases on SBC-2  
- ISECG telerobotic time-delay assessment (ISS robotics 2–10 s class delays)  
- NASA NTRS materials on ISS video latency / DTN / HDTN networking  

---

## 20. Instructions for an LLM using this file

When helping the user:

1. Treat **onboard closed-loop procedure safety under latency** as the north star.  
2. Prefer **hybrid classical + deep** perception and an **explicit FSM**.  
3. Never recommend cloud/Earth RTT as the primary safety path.  
4. Prefer procedure packs, anomaly taxonomy, confidence gating, and measurable time-to-alert.  
5. Be honest about micro-g data limits and COTS compute vs flight certification.  
6. If generating code, default to offline inference, structured logging, voice/UI alerts, and a clear separation between perception evidence and FSM authority.  
7. If the user attaches an existing repo, map this brief onto it — but do not invent repo contents that were not provided.

---

**Bottom line:** This system is an **onboard industrial safety inspector + procedure state machine with a CIMON/Sidekick-class HMI**, not “stream video to Earth and run HAR.” Latency and link outages turn delayed correction into irreversible error; that is the design driver for every layer.
