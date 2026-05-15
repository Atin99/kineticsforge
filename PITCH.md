# KineticsForge - Pitch

## What this is

I'm building physics-informed software for industrial battery teams. Think of it as a decision layer that sits between your lab and your production line - it tells you which experiments are worth running, which cells need attention, and where your recycling margins are leaking.

## The problem

Battery manufacturers waste a lot of time and money running experiments blind. They test cathode compositions without knowing which degradation mechanism will dominate. Their BMS catches thermal events after they've started, not before. Their recycling plants run at 60-70% metal recovery because nobody models the feedstock uncertainty properly. And most ML tools in this space just memorize cycling curves without understanding the underlying physics, so they fall apart when conditions change.

## What I built

KineticsForge is a B2B platform with four core capabilities:

1. **Cathode lifetime prediction** - I use Universal Differential Equations that bake in real Na-ion physics (P2-O2 phase transitions, Jahn-Teller distortion, SEI growth kinetics) and then learn whatever the physics misses through a bounded neural residual. You get degradation curves with honest uncertainty bands, not just a single prediction.

2. **Pack-level BMS** - Instead of monitoring cells independently, I model the pack as a thermal graph. Cells are nodes, thermal connections are edges. Risk scores come from temperature, impedance drift, and neighbor effects. The goal is to flag the problem cell 30-60 minutes before it becomes a real problem.

3. **Cathode screening** - Scores Na/Mn/Fe compositions across capacity, stability, fade, cost, and defect risk simultaneously. Generates Pareto fronts so you can see the actual tradeoffs instead of optimizing one thing at a time.

4. **Recycling optimization** - Shrinking-core leaching kinetics with Bayesian priors on recovery rates. Monte Carlo propagation over feedstock uncertainty gives you a confidence interval on your recovery, not a number you cant trust. Includes cost/margin analysis so you know if a batch is worth running.

## Why UDE and not just ML

Traditional ML memorizes curves. When you change temperature or switch compositions, the predictions break. My approach encodes the physics I know (Arrhenius kinetics, phase transitions, diffusion equations) as the backbone, and uses a neural network to learn corrections for whatever the physics model gets wrong. This means I need way less training data, physically impossible predictions cant happen, and the model actually extrapolates to new conditions because the physics generalizes even when the data doesn't.

## Technical edge

| What | Why it matters |
|------|---------------|
| P2-O2 phase transition + JT distortion in the ODE | First-principles accuracy without DFT compute cost |
| Temporal graph network on pack topology | Catches failures that cell-level monitoring misses |
| End-to-end uncertainty propagation | Every prediction has bounds, not just a point estimate |
| 5M+ real cycling rows from 11 public datasets | Not training on synthetic data alone |
| India-first design (INR pricing, 45C ambient, monsoon humidity) | Built for the fastest growing battery market |

## Market size

Battery diagnostics is projected at $8.2B by 2028. India's battery market alone is $15.2B by 2030 per NITI Aayog. Battery recycling hits $23.2B by 2030. The slice I'm going after is industrial diagnostics + materials CRO + recycling analytics, roughly $3.4B addressable.

## How it makes money

- **API pricing** - charge per prediction. Diagnostics calls are cheap ($0.01), materials screening calls are higher ($0.10) because they run more compute.
- **Enterprise licenses** - annual platform access for OEMs, $50K-$200K/year depending on scope.
- **Lab partnerships** - I predict which compositions to synthesize, partner labs make them, and I learn from the results. This creates a data flywheel that's hard to replicate.

## Where things stand right now

Working webapp with all four panels running real physics simulations. 15+ API endpoints with authentication and rate limiting. 110 Python modules. Full real-data assembly pipeline with ~5M rows from 6 public research datasets (NASA, ISU, SNL, XJTU, Michigan, UL-PUR). Training infrastructure set up across multiple Kaggle accounts. The models are staged but still need full training runs on real data before I start making hard performance claims.

I'm being honest about this - simulation-backed is not the same as experimentally validated. My validation roadmap puts every candidate through XRD, ICP-OES, EIS, cycling, and pack fault injection before I'd claim production-ready numbers.

## What I need

Seed funding to:
1. Scale the real data corpus to 50M+ rows through OEM partnerships
2. Wet-lab validation of the top 10 predicted cathode compositions
3. Deploy the BMS module on actual battery packs for field testing
4. Production infrastructure for the API

## Contact

[To be filled]
