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
