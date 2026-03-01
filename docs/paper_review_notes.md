# Paper Review Pass — Issues Found

## Factual / Terminology Errors

1. **Line 523**: "Conditional per-grasp quality (throughput, alignment, stability)" — throughput is not a quality metric. Quality is defined on line 224 as q = α · ς. Should say something like "Conditional per-grasp metrics (throughput, alignment, stability)".

2. **Line 40 (abstract)**: "improving grasp quality by 16%" — need to verify: BC quality = 0.905 × 0.824 = 0.746, BCRL quality = 0.908 × 0.950 = 0.863. Improvement = (0.863-0.746)/0.746 = 15.7%. Close enough but could round to 16%.

3. **Line 40 (abstract)**: "grasp success rate by 5 percentage points" — BC = 82.5%, BCRL = 87.0%, diff = 4.5pp. Should say "by 4.5 percentage points" or round to "nearly 5 percentage points".

4. **Line 529**: "RL (Pose) converges to a mean episode reward of ~95–97" — this is training reward. Now that we have RL Raw PCD eval at reward 49.37, the text about "competitive per-grasp reward" needs context. Training reward ≠ eval reward.

5. **Line 556**: TODO still present — "Update with eval numbers for best RL variant". Now have Raw PCD numbers, can partially fill.

6. **Line 608**: "stalls at roughly 60% clearing" — eval shows 53.0% for Raw PCD. Update to "roughly 53%" or wait for all 3 RL variants. The abstract (line 37/61) also says "roughly 60%" — needs updating.

## Consistency Issues

7. **Line 499**: "From-scratch RL uses a symmetric actor-critic in which both actor and critic process point clouds through independent PointNet encoders." — This only applies to PCD variants. The Pose variant uses MLPs. Should clarify "For PCD variants, ..." or "both networks process observations through independent encoders."

8. **Line 379**: "The actor uses the same PointNet encoder and actor MLP as the BC policy; the critic uses an independent PointNet encoder with its own MLP head." — Same issue, only true for symmetric PCD RL.

9. **Line 533**: "All achieve roughly similar clearing (Table~\ref{tab:rl_ablation})" — only Raw PCD is filled in so far. Text should be updated once all 3 are available.

## Style Issues (per professor preferences)

10. **Line 35 (abstract)**: "contact-rich manipulation problem: the scene contains" — the colon after "problem" reads like an em dash substitute. Consider splitting into two sentences.

11. **Line 623**: "BC+RL exhibits a reliability--throughput trade-off" — uses en dash (--), fine per IEEE style, but check if this reads as intended.

## Missing/Stale Content

12. **Line 424**: TODO — BC→RL pipeline figure still needed.
13. **Line 429**: TODO — Parallel environments screenshot still needed.
14. **Line 492**: TODO — Clearing progression curves still needed.
15. **Line 545-546**: TODO — RL Pose and RL Seg PCD eval numbers still needed.
16. **Line 556**: TODO — RL discussion text needs numbers.
17. **Lines 568-580**: TODO — Robustness table needs all eval numbers.
18. **Line 596**: TODO — Per-remaining-logs figure still needed.

## Minor

19. **Line 362**: "The raw PCD model is selected by minimum validation loss (0.101 MSE, epoch 190)" — is this still the model used? Worth double-checking if it's bc_pointcloud_policy.pt.

20. **Line 637**: "improves grasp quality by 16%" — same as item 2, verify after all numbers are final.

21. **Line 635**: "stalling at roughly 60%" — same as item 6, update with final eval numbers.
