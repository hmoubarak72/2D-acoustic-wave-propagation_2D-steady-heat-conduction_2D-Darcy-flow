# 5-minute interview demo script

Goal: show technical depth (how surrogates fail and how to catch it), product judgment (who pays and why), and
go-to-market craft (assets a partner team can use), in that order. Open the live app before the call.

---

### 0:00-0:40 · Overview page: the thesis
> "AI surrogates can make simulation much faster. Adoption stalls on one question: can I trust this answer for
> my next case? PhysicsVeto is built around that question. It checks every prediction against the physics and
> sends the ones that fail back to the solver."

Point at the headline tiles. Read out the veto recall and the "bad answers that get through" tile.
Then say the honest part on purpose:
> "At this toy size, 64 by 64 on a CPU, the solver is actually faster than the network. I measured where that
> flips. The value is at production scale, and I didn't fake that number."

### 0:40-1:40 · Simulate & Compare
- Leave geology on *in-distribution*, press 🎲 once or twice.
- Show FD vs FNO side by side, the error map, and press ▶ Play on the animation.
- Show the error-over-time chart: "Error grows as the wavefield gets complicated. That's why an average accuracy
  number isn't enough."

### 1:40-3:20 · Physics Veto: the differentiator
1. Press **✅ Normal case**. Verdict ACCEPT, Trust Score high, all bars green.
   > "No answer key is used here. These are the wave equation, the boundary condition and energy balance."
2. Press **💥 Try to break it** (or choose *salt body*). The residual map lights up, bars go red, verdict VETO.
3. Read the ground-truth reveal: "The veto never saw the reference, and it was right."
4. Choose *amplitude drift*. Show that the PDE residual barely moves (about 2× normal, under its hard limit),
   because a scaled wavefield almost satisfies a linear wave equation. The model-disagreement bar is what fires.
   > "Physics checks have blind spots. A physically consistent but wrong answer can slip past the residual.
   > That's why there's a second, independently trained model and an OOD check. I report recall per failure
   > mode, so a customer sees exactly which check carries which failure."

### 3:20-4:20 · ROI & Business Case
- The veto rate defaults to a 10% planning assumption. Click the two measured options: 0% on clean in-distribution
  work (best case) and the stress-test mix (worst case). Real workloads sit in between.
- Drag the speedup slider from 100× to 1,000× and show that savings barely move, then raise the veto rate and
  show that they collapse.
  > "Effective speedup is one over veto rate plus one over speedup. With a 10% veto rate you never beat 10×.
  > So the product metric to drive is veto rate, not model speed. That changes the roadmap: active learning on
  > vetoed cases beats a faster network."
- Point at the tornado chart and the assumptions table: every input has a source.

### 4:20-5:00 · GTM Studio + close
- Switch vertical to *Automotive & aerospace* to show the assets adapt. Download the one-page brief.
- Close:
  > "The pilot motion is simple: a customer brings 1,000 historical runs of one workload, we train, calibrate
  > the veto and measure their veto rate in two weeks. That number is their business case. On the NVIDIA stack
  > the model and training move to PhysicsNeMo on GPUs. The veto layer stays the same."

---

**If asked "why not just use a PINN?"** Open the Benchmark Lab, Case B comparison. A classic PINN needs no
solver data, but it is trained again for every new layout, so each answer costs minutes of training instead of
one forward pass. Physics-informed training (the physics-loss FNO) lowers the residual on average but still
doesn't tell you when a specific prediction is wrong. The veto is complementary: it works with any surrogate.

**If asked "how are thresholds set?"** Each check is calibrated to the 95th percentile on held-out
in-distribution predictions. A customer deployment recalibrates on their own data.

**If asked about Cases B and C:** open "Thermal & Darcy cases". Same pipeline, same veto scoring, new physics:
steady heat conduction for electronics cooling and Darcy flow for reservoirs and CO2 storage. The energy check
becomes a conservation check (what leaves through the boundary must equal what the sources inject). Be upfront
that at 64×64 a sparse solve takes milliseconds, so these cases show that the trust layer transfers, not speed.
