Run data of the report's section "Continuous forward points, better partners,
and a band closure" (graphite pincell, fine lifts 200x5, 4000, 800, 200).

- fp4.json, fp5.json, fp7*.json: partners of the multigroup flux (forward_points
  parts 0, 3, 4): ray-wise mismatch, and noise split into bin and within-bin parts.
- channels.json, pn.json: the first-collision emission by channel; the P1
  multigroup partner.
- res_*.json: RESOLVE partners (part 5): tuned (scan), generic, generic with the
  field's lifts (src), and bands (sub), against width x depth.
- fp8_*.json, fp9_*.json: noise of the RESOLVE and band partners (parts 7, 8);
  fp9_sl_W_L.json: band partners with log-spaced levels, W windows x L levels,
  with the band proposal.
- closure_16x8.json: the band sweep's closure against the forward points.

forward_partner_study.py reproduces each kind of run.
