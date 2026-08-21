# Inna's handwritten comments on Chapters 1-3 (scan, 2026-08-19)

Source: ~/Downloads/"George Siders Ch 1-3 Innas comments.pdf" (36 pages, rendered to
scratchpad/inna/p-NN.png). Transcribed from the handwriting; page numbers are PDF pages.

## Her general note (email)
1. Too concise: concision is good, but not at the cost of being unclear, ambiguous, or
   missing important information.
2. Underlined + "?" means she does not understand it or it makes no sense.
3. Ch3 especially uses terms Prof. Kovecses will not know, e.g. "endgame".
4. "Scripted policies" - is that standard ML/AI usage? If not, prefer terms familiar to a
   roboticist with a classical background.
5. Many places are too terse; other places explain in words what should be an equation,
   figure, flowchart, or block diagram, which classical roboticists read more easily.
6. The use of Isaac Lab must be motivated / explained.
7. Comments get denser as the chapters progress; apply the same lessons to later chapters.

## Page-by-page

### p2 (Ch1, Sec 1.1)
- Margin: "section 1.1 is a bit too brief for my taste but OK."
- Inline rewrite: strike "targeting" -> "for defining the target location for placing the
  grapple relative to the pile". (So: "a good strategy for defining the target location for
  placing the grapple relative to the pile must remain effective from the full, ordered
  stack down to the last scattered logs.")

### p3 (Ch1, Sec 1.2)
- "Three questions structure the work" -> strike "conducted", insert "reported in this thesis".
- Q3 "physical crane" -> "log-loading machine".
- "The experimental platform is" -> "The experimental platform employed to support this
  research is ..."; "operated with FPInnovations" -> "operated by FPInnovations, referred to
  as the crane".
- A "?" beside the paragraph and "1+1s" mark near "equipped with crane joint sensing" -
  wants this clarified.

### p4 (Ch1, Sec 1.3)
- Contribution 1: circles "bite" in "bite depth" with "?" - term unclear, define or replace.
- Contribution 2: "split this sentence" (the long "An Isaac Lab testbed ... and the systems
  work that carries a policy onto the machine" sentence).
- Underlines "the systems work that carries a policy onto the machine" + "don't understand
  this statement".

### p5 (Ch1, Sec 1.3-1.4)
- Circles "a floor-constrained refinement" + "not sure what that means".
- Underlines "a training environment aligned with the calibrated deployment configuration"
  + "back to training environment -> confusing".
- Contribution 3 bracketed whole + "this doesn't sound like a contribution but rather a
  finding or conclusion".
- Contribution 4: underlines "the learned policies finished every trial, while two of the
  three baseline trials ended in repeated grasps at rack structure with pile remaining" -
  same point: reads as a finding, not a contribution.

### p6 (Ch1, Sec 1.4)
- "the calibration campaign" -> "methodology".
- "Chapter 7 concludes and outlines future work" -> insert "with the main findings".

### p14 (Ch2)
- "log instance segmentation [28]" underlined + "any others?" - wants more references.
- Section 2.5 heading circled: "I feel like this section is better integrated into 2.1,
  either into 2.1.3 or somewhere around there."

### p18 (Ch3 opening) - the big structural note
- "the simulated testbed" -> "the simulated CRANE testbed".
- "The environment is implemented in Isaac Lab" -> "The SIMULATED environment ...".
- Circles "PhysX GPU dynamics": "don't understand the use of the word dynamics here."
- MAJOR: "I would suggest having a paragraph or 2, or a subsection, on Isaac Lab with some
  basic information about its physics modeling and ML features, and references to
  documentation."

### p19 (Sec 3.1 Crane Model)
- Fig 3.1: marks the Stick Camera label; "on the stick" / "on the mast" annotations - make
  the two camera mountings explicit in the figure.
- "a grapple suspended from a passive chain" -> "a 2 degree-of-freedom passive chain".
- "Link masses and joint limits are listed in Table 3.3" -> "should be Table 3.1 (first table
  in this chapter)" - TABLE ORDER IS WRONG.
- "of the grapple" insertion; "the passive chain shapes both control and evaluation" ->
  add "relative to the rest of the arm".

### p20 (Sec 3.2)
- Section title "Physics Modeling and Tuning" circled -> "would change to Environment Modeling".
- "several hundred rigid bodies" -> "or more"; "large-scale" suggested.
- "contact network" -> "contact system" (network struck out).

### p23 (Table 3.2)
- Caption -> "Per-episode variation in the pile AND LOG ASSETS".
- "what is a column?" (re per-column relief).
- "Jagged bumps" circled, "are these per episode?"
- "prims" circled + "?" - undefined jargon.
- "each run keeps one fixed mixture" - "run" circled: define run vs episode.
- Log size row: the "x{0.90,...}" notation circled + "?" - unclear notation.

### p26 (Sec 3.3 FSM / 3.4 Task Formulation) - densest page
- TOP: "illustrating these phases using graphs/plots, say for x, y, z, psi in task space,
  would be much more helpful."  -> SHE WANTS A TASK-SPACE TRAJECTORY FIGURE FOR THE FSM.
- "consequences of matching the machine" underlined: "what do you mean? matching how the
  real crane is currently functioning?"
- "simulation is grasp-and-remove throughout this thesis" -> "for all simulations carried
  out"; and "the logs are removed from the grapple at the end of the cycle".
- "On the machine" -> "on the real crane test-bed".
- "transport and deposition sequence" -> "where the grasped logs are moved to a location
  after the start ...".
- "the deposit-side phases present in the environment code are not exercised by any
  experiment reported here" circled: "need to be more clear on what that means + also
  explain why this is done + contrast to what happens on the real system."
- "Algorithm 3.1 states the simulated cycle" -> "of the FSM".
- RIGHT: "there are 2 cameras on the crane! => clarify which camera makes the observation in
  the GAZE phase, earlier as well."
- Sec 3.4: "don't quite understand what you are saying here."
- "the transition is whatever the physics and the FSM produce" -> "engine of Isaac Lab";
  and "how is this a consequence of how the machine operates?"

### p27 (Algorithm 3.1) - second big structural note
- "q is usually used for configuration space variables of a robotic arm (joint variables)".
- "H = 30 grasp attempts, above what the expert needs" -> insert "this value".
- MAJOR: "I don't quite see how this Section 3.4 fits in with the title of Chapter 3 and
  hence how it belongs in this chapter. The same is true for 3.5 and 3.6."

### p30 (Sec 3.4.3 Actions, Fig 3.4)
- "command the grapple into the bed" circled - unclear.
- "Yaw is represented by the doubled-angle pair" circled: "where / for what?"
- "the pi-periodic encoding maps" underlined: "are you referring to Eq. (3.1)?" -> insert
  "to that".

### p33 (Fig 3.5 outcomes)
- Inset legend symbols circled: "need to match the notation used in the body, or vice versa."
- "degradation sweeps of Chapter 6 test the policy against holes as well as displacement"
  circled: "which holes? you mean due to dropped pixels? which displacement?"
- Margin: "deployed policy" / "framing" note beside the sensing-geometry paragraph.

### p34 (Sec 3.6 Scripted Policies)
- "it cannot be deployed" -> "on the real crane testbed".
- The geometric-heuristic sentence underlined: "grammar, don't quite understand" +
  "improve for clarity".
- "places the grasp over it" -> "grapple".
- "digs d = 0.25 m below the local pile surface" -> "so below the highest z?"
- "the depth every row of the simulated comparison shares" circled: "which row are you
  talking about?"
- "empty cycles" and "the cap" circled - jargon, and "what is that?"
- MAJOR: "this is strange because you seem to be talking about results that will be
  presented in subsequent chapters. Why are you doing this?"  -> Ch3 forward-references
  Ch4/Ch6 results.

### p36 (Sec 3.6.2 geometric heuristic, end of Ch3)
- "the heuristic digs a field-tuned fixed depth" -> "for the geometric heuristic".
- "the simulated cloud of its deployment configuration" -> "the log rack?"
- Whole parameter-list paragraph bracketed: "This narrative is hard to follow: drawings,
  flowchart, equations might help."
- REPEATED MAJOR: "I really feel like Sections 3.4, 3.5 and 3.6 are better fit for
  integration in other chapters."

## Recurring themes to act on
1. STRUCTURE: Sections 3.4 (task formulation), 3.5 (randomization), 3.6 (scripted policies)
   do not fit a chapter titled "A Simulated Crane Testbed"; she says twice they belong
   elsewhere. Ch3 also forward-references results from Ch4 and Ch6.
2. ISAAC LAB: must be motivated, with a subsection on its physics/ML features + docs refs.
3. FIGURES OVER PROSE: task-space plots of the FSM phases; a flowchart/diagram for the
   heuristic; equations where currently there is narrative.
4. VOCABULARY: "endgame", "empty cycles", "the cap", "prims", "run" vs "episode", "bite
   depth", "scripted policies", "contact network", "row" (of a table) - define on first use
   or replace with classical-robotics terms.
5. THE TWO CAMERAS: say which camera produces the observation, in the figure and the text.
6. NOTATION: figure insets must match body notation; Table 3.3 referenced before Table 3.1
   (renumber); q reserved for joint variables.
7. REAL vs SIMULATED: say "real crane testbed" explicitly wherever the machine is meant;
   several places read ambiguously.
8. CONTRIBUTIONS: items 3 and 4 in Sec 1.3 read as findings, not contributions.

## Second pass: pages not covered in the first transcription

### p8 (Ch2, Sec 2.1.1-2.1.2)
- "one step is one complete grasp cycle" -> "one step OF THE POLICY is ...".
- "it is therefore bounded by the strategy it copies" -> "weaker than what?" and circles
  "a weaker bound" - the bound-vs-weaker-bound phrasing is unclear.

### p24 (Fig 3.3 pile shapes)
- No marks. (Figure accepted as is.)

### p25 (Sec 3.3 FSM) - TERMINOLOGY, the most serious page
- Section title: "is this the terminology Isaac Labs uses? It doesn't quite make sense."
- Circles "damped-least-squares differential IK controller in task space".
- "the IK target is the top of the passive chain" -> "you mean tip of telescope?"
- Circles "the IK link" + "?" ; "targets are specified for the grapple" circled:
  "what does this mean? the grapple has a finite dimension, it is not a point in space."
- Tolerances: "these are position tolerances, they only make sense for a point in space."
- "fixed joint-space viewing pose q_gaze": POSE vs CONFIGURATION - "the crane is a
  multi-body system and it can have a CONFIGURATION; POSE in robotics describes the
  position/orientation of a single rigid body."
- BOTTOM: "I am concerned Professor Kovecses will take objection to some of this
  terminology and I don't feel comfortable with it either. What exactly is IK link/target?"

### p28 (Sec 3.4.2 Observations)
- "the two settings" circled: "which two settings? simulated + real?"
- "and the simulator back-projects" -> "while".
- "Everything downstream is shared" -> "between the simulation and implementation on the
  real crane testbed?"
- "extrinsic T_base<-cam" -> "transform"; "cropped to the box [b_min, b_max]" circled:
  "how do 2 scalars define a box?"
- "Algorithm 3.2 ... Figure 3.4" -> "separate out into another sentence".
- "the cloud contains every surface" -> "points for every surface".
- "4K-dimensional vector" -> "say 4 x (x,y,z,psi)?"
- "which never runs on the machine" -> "on the real crane".
- FOOTER: "homogeneous transforms in robotics are usually indicated with bold symbols."

### p32 (Sec 3.4.4 outcomes / 3.5 randomization)
- "grasp success rate (fraction of cycles with n>0)" -> "defined as the"; "pile clearing
  rate" -> "computed as"; "the fraction of logs knocked out of bounds" -> "the NUMBER of
  logs knocked out of bounds AS A FRACTION OF TOTAL NUMBER OF LOGS".
- "Every policy trains in simulation and is judged on the crane" -> "is trained",
  "evaluated"; "the variation the simulator offers" -> "variability".
- "Between episodes the simulator redraws only the pile" -> "rebuilds ... as was described
  in Section 3.2.4".
- "the training pipeline adds its own augmentations" -> "of what?"
- "the axial noise model" -> "not sure what this means".
- MAJOR: "If you have to do so much cross-referencing into the future, maybe the
  organization is not the best."

### p35 (Alg 3.3 + Sec 3.6.2 geometric heuristic)
- "The heuristic re-implements the same strategy" -> "the same GRASP strategy AS THE
  PRIVILEGED EXPERT?"
- "it takes the highest supported candidate" -> "how can a log be unsupported?"
- "within a planar radius r_n" -> "circle in the ground plane of".
- "don't know if Prof. Kovecses will know the meaning of 'kludge'/'nudge'" (re wording on
  the next page).

## Additional recurring themes (second pass)
9. POSE vs CONFIGURATION: a multi-body system has a configuration; pose belongs to a single
   rigid body. Fix q_gaze wording and anything calling a joint vector a "pose".
10. IK LINK / IK TARGET is undefined and she explicitly asks what it is; the grapple is a
    body, not a point, so "position tolerance" needs restating (e.g. tolerance on a defined
    reference point of the grapple).
11. Bold symbols for homogeneous transforms, per robotics convention.
12. "cropped to the box [b_min, b_max]" - explain that these are opposite corners.
13. Cross-referencing forward is pervasive enough that she questions the organization
    (second independent flag, p32).

## Third pass: the last unread pages

### p7 (Ch2 opening)
- "receives a scalar reward" circled: "what is a scalar reward and why is it a scalar?"
  (asks for the reward to be explained, not just named).
- "The tasks in this thesis" -> insert "log-loading from large piles".
- "is useful" note beside the reward clause.

### p12 (Sec 2.2 forestry automation) - REFERENCES
- "forwarder" -> "machine".
- "leave open where to grasp" -> "leave open THE QUESTION OF where to grasp".
- MAJOR: "You only have 4 references and 2 of them are to Jebellat + me, which is not at
  all reasonable. => missing references => Elie's papers!"
  ACTION NEEDED FROM GEORGE: supply the additional Ayoub citations (only ayoub2023grasp is
  in thesis_references.bib today). A TODO marker sits at that spot in the chapter.

### p13 (Sec 2.3 grasping in clutter)
- "progressed from isolated objects to cluttered scenes" -> "applications involving those
  with isolated objects and those with cluttered scenes".
- "the mill yard" -> "the problem of grasping bunches of large logs from large log piles".
- "its clutter" -> "whose?" (ambiguous possessive).
- "the grapple captures bunches" -> "must pick up".
- "success is usually scored one grasp at a time" -> "in bin-picking types of problems".
- "from dense stack to empty rack" -> "from a dense LOG stack to an empty RACK".

### p11 (Fig 2.1 actor-critic)
- "spread off / need sentence here" - the figure sits immediately after a broken sentence;
  needs a lead-in sentence before the figure.

### p21 (Sec 3.2.1 log assets)
- Table 3.1 margin: "are you describing other environments as well or just this one?"
- "the deployment-era environment spawns" -> unclear, drop "deployment-era".
- "every log sharing one end-to-tip orientation" circled: "not sure what you mean".
- "impacts dissipate energy" -> "most of the PRE-IMPACT energy".

### p22 (Sec 3.2.2-3.2.4) - STRUCTURE, the one George flagged
- "convex decomposition for the rack" circled: "need a picture/figure of the rack, sample
  where" (wants the rack collision geometry shown).
- "for logs and grapple parts" -> "all?"
- "eliminating tunneling" -> "missed contact detection" (avoid the game-physics term).
- 3.2.3 heading, "Actuation" circled: "would put this into crane modeling".
- "implicit PhysX PD drives" margin: "any chance of writing down a control law?"
- "saturated PD velocity law with discrete stepping" circled: "not sure what you mean".
- "Piles are initialized by spawning logs ..." bracketed: "would put this into 3.2.4,
  Pile initialization + variation, or Pile Modeling".

### p29 (Alg 3.2 + Sec 3.4.3 actions)
- "the two are identical" -> "what are you referring to?"; "transfer gap" -> "sim-to-real";
  "Algorithm 3.2" -> "which summarizes the observation pipeline".
- "the grapple position" circled: "position of WHICH POINT?"
- "Admissible targets lie" -> "must lie".
- "the bed floor" -> "which bed?"
- "as already indicated in Section 3.3" (the clamp is stated twice).

### p31 (Sec 3.4.4 outcomes)
- "The scripted policies of Section 3.6 write their targets directly into this space"
  -> "don't quite understand the logic of this sentence".
- "After the lift" -> "at the end of the cycle"; "measured by" -> "evaluated by".
- "counts the logs held by the grapple (centers within a proximity radius)" -> "this is
  evaluated by ..." (say how).
- "the grapple jaw axis" circled: "has this axis been defined/drawn before? refer to
  Figure 3.5."
- "the alignment" -> "the alignment METRIC"; "scores the levelness of" -> "scores HOW
  LEVELLED the grapple IS".

### p1, p9, p10, p15, p16, p17
- No margin annotations (ink-density check: <= 0.3%, consistent with scanner speckle).

## Seed results that arrived during the review work (2026-08-20 night)

BC scoring head, deep protocol, three training seeds: 98.0 / 98.0 / 99.0 full clear
(mean 98.3, sd 0.6), cycles 18.93 +/- 0.02. The headline number is reproducible.

BC regression head, same treatment: 84.1 / 33.3 / 41.1 % of the pile cleared, full clear
33 / 0 / 0. Training converged normally in all three cases, so the head itself is
seed-fragile.

Both are relevant to how chapter 5 should be written: the architecture claim can be stated
as reproducibility rather than as a single-seed win, and the regression row should carry
its spread.
