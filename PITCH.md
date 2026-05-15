# KineticsForge - Pitch

## What this is

I'm building physics-informed software for industrial battery teams. Think of it as a decision layer that sits between your lab and your production line - it tells you which experiments are worth running, which cells in your pack need attention, and where your recycling margins are leaking. The core idea is that nobody should be running blind experiments when physics can narrow things down first.

## The problem

Battery manufacturers waste a lot of time and money running experiments blind. They test cathode compositions without knowing which degradation mechanism will dominate at their operating conditions. Their BMS catches thermal events after they've started, not before when you can still do something about it. Their recycling plants run at 60-70% metal recovery because nobody bothers to model the feedstock uncertainty properly. And most ML tools in this space just memorize cycling curves without understanding the underlying physics, so they fall apart the moment conditions change from what they were trained on.

I've talked to enough people in this space to know that the gap isnt "more data" or "better models" - its that nobody has connected the physics to the decision making in a way thats actually usable by an engineering team that has other things to do.

## What I built

KineticsForge is a B2B platform with four core things it does:

1. **Cathode lifetime prediction** - I use Universal Differential Equations that bake in real Na-ion physics (P2-O2 phase transitions, Jahn-Teller distortion, SEI growth kinetics) and then learn whatever the physics misses through a bounded neural residual. You get degradation curves with honest uncertainty bands, not just a single prediction that pretends to be certain.

2. **Pack-level BMS** - Instead of monitoring cells independently like everyone else does, I model the pack as a thermal graph. Cells are nodes, thermal connections are edges. Risk scores come from temperature, impedance drift, and neighbor effects. The goal is to flag the problem cell 30-60 minutes before it becomes an actual problem.

3. **Cathode screening** - Scores Na/Mn/Fe compositions across capacity, stability, fade, cost, and defect risk all at the same time. Generates Pareto fronts so you can see the real tradeoffs instead of optimizing one metric and getting blindsided by another.

4. **Recycling optimization** - Shrinking-core leaching kinetics with Bayesian priors on recovery rates. Monte Carlo propagation over feedstock uncertainty gives you a confidence interval on your recovery, not a single number you cant trust. Includes cost/margin analysis so you know if a batch is worth running before you start it.

## Why UDE and not just ML

This is probably the most important technical decision I made. Traditional ML just memorizes curves. When you change temperature or switch to a different composition, the predictions break because it never saw that exact combination before. My approach encodes the physics I know (Arrhenius kinetics, phase transitions, diffusion equations) as the backbone, and uses a neural network to learn corrections for whatever the physics model gets wrong. The result is I need way less training data, physically impossible predictions literally cant happen because the physics constrains them, and the model actually extrapolates to new conditions because the physics part generalizes even when the training data doesnt cover everything.

## Technical edge

| What | Why it matters |
|------|---------------|
| P2-O2 phase transition + JT distortion in the ODE | First-principles accuracy without DFT compute cost |
| Temporal graph network on pack topology | Catches thermal failures that cell-level monitoring completely misses |
| End-to-end uncertainty propagation | Every prediction has bounds, not just a point estimate |
| 5M+ real cycling rows from 6 public research datasets | Not training on synthetic data alone |
| Calibration engine against published experimental data | Coefficients grounded in real measurements, not guesses |
| India-first design (INR pricing, 45C ambient, monsoon humidity) | Built for the fastest growing battery market, not just copy-pasting from US/EU assumptions |

## Data

I collected about 5 million real battery cycling rows from publicly available research datasets - NASA PCoE, ISU ILCC, Sandia National Labs, XJTU, University of Michigan, and UL/Purdue. Wrote custom normalization scripts for each one because they all use different formats, units, and column conventions. The assembled dataset covers 555 unique cells across 11 dataset keys. Also built a literature scraper and crystal structure harvester to pull published Na-ion cathode properties from papers and materials databases for the screening module.

## Market size

Battery diagnostics is projected at $8.2B by 2028. India's battery market alone is $15.2B by 2030 per NITI Aayog. Battery recycling hits $23.2B by 2030. The slice I'm going after is industrial diagnostics + materials CRO + recycling analytics, roughly $3.4B addressable.

## How it makes money

- **API pricing** - charge per prediction. Diagnostics calls are cheap ($0.01), materials screening calls are higher ($0.10) because they run more compute.
- **Enterprise licenses** - annual platform access for OEMs, $50K-$200K/year depending on scope.
- **Lab partnerships** - I predict which compositions to synthesize, partner labs make them, and I learn from the real results. This creates a data flywheel that gets harder to replicate the longer it runs.

## Where things stand right now

Working webapp with all four panels running real physics simulations. Full API with authentication and rate limiting. 110+ Python modules covering the physics engine, training infrastructure, data pipelines, and validation framework. Real-data assembly pipeline with ~5M rows from 6 public research datasets (NASA, ISU, SNL, XJTU, Michigan, UL-PUR). Training infrastructure across multiple Kaggle accounts with trained model checkpoints. Calibration tools for grounding physics coefficients against published experimental data.

I'm being upfront about this - the simulations give numbers in the right ballpark for Na-ion cathode behavior, and the physics structure is sound, but I'm still working on tightening coefficient calibration against experimental data. I dont claim hard performance numbers until proper holdout benchmarks pass my validation gates. Simulation-backed is not the same as experimentally validated and I wont pretend otherwise.

## What I need

Seed funding to:
1. Scale the real data corpus to 50M+ rows through OEM partnerships
2. Wet-lab validation of the top 10 predicted cathode compositions
3. Deploy the BMS module on actual battery packs for field testing
4. Production infrastructure for the API

## Contact

[To be filled]
