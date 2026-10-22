# Changelog

## [Unreleased] — 2026-10-21

### Added

**Local sensitivity analysis** (from `~/Downloads/sensitivityanalysis.ipynb`)
- `sensitivity.py`: `SensitivityAnalyzer` with `build_jacobian`, `jacobian`,
  `solve_base`, `solve_augmented`, `run` (parameter sweep), `elasticity`,
  `summary`, `finite_difference_check` and `case_inputs`. Also contains
  `SensitivityResult`, `activation` / `activation_derivatives` (sigmoid,
  tanh, hill), `sweep_values`, `MICROENVIRONMENT_CASES`, `CASE_BASE_INPUTS`
  and `notebook_initial_conditions()`.
- Extends the notebook's single-parameter logic to *k* parameters at once
  (S is n × k), arbitrary decay rates (−diag(δ) instead of −I), the UI
  thresholds and activation function, and the extra ∂φ/∂b term when the
  saturation rate is a parameter.
- With the notebook's initial conditions it reproduces the notebook's sweep
  to 1e-10 (q) and 1e-12 (S). It agrees with central finite differences to
  ~1e-6.
- New tab **Local sensitivity analysis** with its own *Run sensitivity
  analysis* button. It has multiselect boxes for parameters and output
  nodes, a swept parameter and range, the elasticity mode, the final time,
  and an optional finite-difference check. Outputs are a figure, a
  validation figure, a summary table, an info box and a JPG download.
- `sensitivity_analysis.ipynb`: the analysis with LaTeX cells for the
  mathematical framework. It can run the thesis microenvironment cases
  (Figures 5.15–5.20) and the alternative set from the original notebook's
  raw cell (Figures 8–13).
- `docs/sensitivity_analysis.md`: documentation of the core logic.
- `plotting.py`: `Plotter.plot_sensitivity_grid`,
  `Plotter.plot_sensitivity_fd_check`, `Plotter.node_color`, plus the
  module helpers `param_to_latex`, `elasticity_label` and `auto_palette`.
- `requirements.txt`: `sympy`.

**Bifurcation diagram (2-D)** (replaces the second tab that was also called
"Phase Space")
- Vary any input from `INPUT_GROUPS`, or several at the same time. The other
  inputs, decay rates, thresholds, initial conditions, activation and solver
  come from the UI (previously LPSE, SSRNA, IL4L, TGFBE and IL21E were fixed
  at 1 in code).
- Levels come from *From* / *To* / *Number of levels* or a custom list.
- The plot title and the `IFNγ=…` boxes inside the axes were removed.
- Colour bar: `<input> expression level` for one input,
  `Cytokines expression level` for several.
- Choosing other axes (master transcription factors) redraws the last
  diagram without solving again. The figure can be downloaded as JPG.
- `ThesisCode.py`: `Bifurcation_tab`, `run_bifurcation`,
  `replot_bifurcation`, `handle_bifurcation`. `data_structures.py`:
  `BifurcationTabComponents`.

### Changed
- **Phase Space (3-D)**: renamed, now an interactive Plotly figure.
  Trajectories are coloured by cytokine level (with a colour bar) and their
  shadows are drawn on the floor. Steady states are joined by a dashed line
  and the initial condition is marked. Axes are equal (cube) with a fixed
  camera. The axis selector now asks for 3 nodes (it said 2 and defaulted to
  2), and the *Plot Phase Space* button is wired.
- `Equations.py`: the 155 terms moved from `system_ode` into
  `NetworkEquations.interaction_terms(t, q, p, w, exp=np.exp)`, so sympy can
  reuse them. The numeric results are bit-identical.

## [Unreleased] — 2026-10-18

### Added

**Two new Gradio tabs ("scenario grids")**

Each tab runs four deterministic simulations and plots them as a 4 × 5 matplotlib
figure. Each row is one simulation and each column is one subnetwork from
`subnetworks.py`: `T Metabolism`, `Antigen presentation`,
`T Transcription factors`, `Th1` and `Treg`. Every row has a bold panel letter
outside the top-left corner of its first plot. The letters default to `A,B,C,D`
and can be changed from a textbox in the tab.

| Tab | Row 1 | Row 2 | Row 3 | Row 4 |
|---|---|---|---|---|
| **TCR/CD28 interaction times** | TCD8086 = 5, TMHC2 = 5 | TCD8086 = 15, TMHC2 = 15 | TCD8086 = 30, TMHC2 = 25 | TCD8086 = 45, TMHC2 = 35 |
| **Decay-rate scenarios** | all decay rates = 1 | TBET = 4 | TBET = 4, GATA3 = 4 | TBET = 4, GATA3 = 4, RORGT = 5 |

The settings that a row does not override come from the user's selections in
the rest of the interface: initial conditions, inputs, thresholds, activation
function and ODE solver. The interaction-time tab also uses the user's
decay-rate sliders. In the decay-rate tab, every node not listed in a row is
set to 1, as specified. Each tab also has a "Download figure as JPG" button.

**`data_structures.py`**
- `SimulationScenario`: a frozen dataclass that describes one row as overrides
  applied on top of the user's settings (`label`, `out_overrides`,
  `decay_overrides`, `reset_decay_rates`).
- `ScenarioGridTabComponents`: the Gradio components of a scenario-grid tab.
- `Tabs`: new fields `tcr` and `dec`.

**`plotting.py`**
- `Plotter.design_scenario_grid(rows, subnetworks, row_letters, with_bounds=False)`
  builds the grid. It reuses `_draw_branch_for_subnetwork` and
  `create_line_plot`, so every panel has exactly the same style as the
  Results-tab figures. Bar-chart subnetworks such as `Lymphocyte inputs` also
  work in a grid.
- `Plotter.add_panel_letter(axi, letter)` adds the bold panel letter.
- New style keys in `DEFAULT_STYLE`: `panel_letter_x`, `panel_letter_y`,
  `panel_letter_font_size`.

**`ThesisCode.py`**
- Class constants `SCENARIO_GRID_SUBNETWORKS`, `DEFAULT_ROW_LETTERS`,
  `INTERACTION_TIME_SCENARIOS` and `DECAY_RATE_SCENARIOS`.
- `Scenario_grid_tab(title, scenarios)`: one factory builds both tabs. The tab
  lists what each row overrides, generated from the scenario labels.
- `handle_scenario_grids(tabs, gd)`: wires the simulate and download buttons
  of both tabs.
- `_apply_scenario(sim, scenario)`: returns a copy of the user's settings with
  one scenario's overrides applied. The original settings are never changed.
- `_parse_row_letters(letters_str, n_rows)`: missing or empty letters fall back
  to A, B, C, …
- `_make_scenario_grid_callback(key, scenarios)`: builds the Gradio callback
  for a grid tab.
- `self.last_scenario_figs`: stores the last figure of each grid tab, used for
  downloads.

### Changed (refactors with no change in behaviour)

- `ThesisCode.py`
  - `_simulation_inputs(tabs, gd)` replaces the list of input components that
    was copied in `handle_simulation` and `handle_comparison_simulation`. The
    new tabs use it too.
  - `_unpack_simulation_values(values)` replaces the index slicing that was
    copied in `after_push_button` and `after_push_button_comparison`.
  - `_solve_simulation(sim)` replaces the copied `self.solve(...)` call in
    those two callbacks.
- `plotting.py`
  - `_create_figure_grid(n_rows, n_cols)` and `_legend_bbox()` are now shared
    by `_setup_subplots_grid` and `design_scenario_grid`. The Results and
    Comparison figures keep the same size and legends.

### Unchanged

`Equations.py`, `subnetworks.py`, `boolfuzzy_converter.py`, and every other tab
and callback.

### Verified

The test ran with the versions in `requirements.txt`, headless with the Agg
backend, building the whole UI with `launch()` stubbed out.
- Both grid callbacks render a 21 × 26 in figure in about 2 s, and the JPG
  download works.
- Overrides reach the solver. With MHC2E = CD8086E = 1, longer interaction
  windows lengthen TCR/CD28 signalling. Raising the decay rate of TBET, GATA3
  and RORGT lowers only those nodes.
- `after_push_button`, which was refactored, still returns all three figures.
