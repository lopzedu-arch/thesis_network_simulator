# Macrophage–T cell network simulator

A Gradio interface for simulating a fuzzy/probabilistic regulatory network of
macrophage and CD4⁺ T-cell signalling (155 nodes, 43 external inputs). The
network is solved as deterministic ODEs (`scipy.integrate.solve_ivp`) or as Itô
SDEs (`sdeint`). Results are plotted as publication-style matplotlib figures and
interactive Plotly charts.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python ThesisCode.py               # opens the interface at http://127.0.0.1:7860
```

## Project structure

| File | Role |
|---|---|
| `ThesisCode.py` | Entry point. `NetworkInterface` builds the Gradio tabs, wires their callbacks and runs the simulations. |
| `Equations.py` | `NetworkEquations`: node and input names, default values, the ODE system, `solve()` (deterministic) and `solve_ito()` (stochastic). |
| `plotting.py` | `Plotter` (all matplotlib/Plotly figures), `PlotConfig`, and the `DEFAULT_STYLE` dictionary of style values. |
| `subnetworks.py` | `SUBNETWORKS`: maps each subnetwork panel to the nodes, labels, colours and line styles it plots. |
| `data_structures.py` | Dataclasses that group the Gradio components of each tab (`*TabComponents`, `Tabs`) and `SimulationScenario`. |
| `boolfuzzy_converter.py` | Converts Boolean rules (`!`, `&`, `\|`) into probabilistic expressions. |
| `sensitivity.py` | `SensitivityAnalyzer`: forward (local) sensitivity analysis, i.e. symbolic Jacobian, augmented ODE, elasticities and finite-difference check. Also holds the thesis `MICROENVIRONMENT_CASES`. |
| `sensitivity_analysis.ipynb` | Notebook version of the analysis, with the LaTeX derivation. Runs the thesis microenvironment cases (Figures 5.15–5.20). |
| `docs/sensitivity_analysis.md` | Core logic of the sensitivity analysis: equations, implementation, validation, caveats. |

## Tabs

| Tab | What it does |
|---|---|
| Inputs | External inputs (cytokines, antigens, nutrients, drugs, interaction times) and per-node decay rates. |
| Initial conditions | Per-node initial values; random Gaussian presets. |
| Threshold values | Per-node activation threshold `u`. |
| Results | Deterministic simulation drawn as two subplot figures (subnetworks 0–10 and 10–20). |
| **TCR/CD28 interaction times** | 4 simulations varying `TCD8086` / `TMHC2`, one row each (see below). |
| **Decay-rate scenarios** | 4 simulations varying the TBET / GATA3 / RORGT decay rates, one row each (see below). |
| Deterministic vs Stochastic | ODE solution with a ±20 % band, overlaid with the mean of 100 Itô runs. |
| Stochastic Itô | Mean of 100 Itô runs with per-node noise σ. |
| Bandwidth analysis | Differentiation efficiency vs. noise level. |
| Convert and process your own boolean networks | Boolean → probabilistic rule converter. |
| Plot customization | Interactive Plotly chart of any nodes. |
| Phase Space (3-D) | Interactive 3-D trajectories of 3 master transcription factors as IFN-γ, IL-33, IL-4 and TGF-β rise together (drag to rotate). |
| **Bifurcation diagram (2-D)** | 2 master transcription factors, one trajectory per level of the selected input(s) (see below). |
| **Local sensitivity analysis** | Forward sensitivities and elasticities of chosen nodes with respect to one or more inputs (see below). |
| Additional settings | Activation function, ODE solver, initial-condition distribution. |

## Scenario grids

A scenario grid runs one deterministic simulation per row and draws the same
subnetworks in every row:

```
        T Metabolism  Antigen presentation  T Transcription factors  Th1  Treg
  A     [scenario 1]  ...
  B     [scenario 2]  ...
  C     [scenario 3]  ...
  D     [scenario 4]  ...
```

Every panel is drawn by the same code as the Results tab
(`Plotter.create_line_plot`), so colours, line styles, legends, axes and fonts
match the rest of the thesis figures. The bold letters are set in the
*Row letters* textbox (default `A,B,C,D`). The letters are separated by commas,
and any missing letter falls back to A, B, C, …

**Where the values come from.** When you press *Simulate scenarios*, the tab
reads the current initial conditions, inputs, decay rates, thresholds,
activation function and solver from the other tabs, just as the Results tab
does. Each row then applies its overrides on top of these settings:

| | TCR/CD28 interaction times | Decay-rate scenarios |
|---|---|---|
| Row 1 | TCD8086 = 5, TMHC2 = 5 | all decay rates = 1 |
| Row 2 | TCD8086 = 15, TMHC2 = 15 | TBET = 4, all others = 1 |
| Row 3 | TCD8086 = 30, TMHC2 = 25 | TBET = 4, GATA3 = 4, all others = 1 |
| Row 4 | TCD8086 = 45, TMHC2 = 35 | TBET = 4, GATA3 = 4, RORGT = 5, all others = 1 |

The decay-rate tab ignores the decay-rate sliders and starts every row from 1,
as the scenarios require. The interaction-time tab uses the decay-rate sliders
as set.

> In the default inputs `MHC2E` and `CD8086E` are 0 (no antigen), so the
> T-cell panels stay near zero. Set them to 1 in the *Inputs* tab to see
> TCR/CD28 activation.

### Changing or adding scenarios

The scenarios are plain data at the top of `NetworkInterface` in
`ThesisCode.py`:

```python
SCENARIO_GRID_SUBNETWORKS = ['T Metabolism', 'Antigen presentation',
                             'T Transcription factors', 'Th1', 'Treg']

INTERACTION_TIME_SCENARIOS = [
    SimulationScenario('TCD8086 = 5, TMHC2 = 5',
                       out_overrides={'TCD8086': 5, 'TMHC2': 5}),
    ...
]
```

`SimulationScenario` fields (`data_structures.py`):

| Field | Meaning |
|---|---|
| `label` | Text shown in the tab's description. |
| `out_overrides` | `{input name: value}`, using names from `Equations.networkparameters()` (e.g. `'TCD8086'`, `'IFNGE'`). |
| `decay_overrides` | `{node name: decay rate}`, using names from `Equations.Nodes()` (e.g. `'TBET'`). |
| `reset_decay_rates` | `True` starts every decay rate at 1.0 before the overrides are applied; `False` keeps the user's sliders. |

To add a whole new grid tab:

1. Define a new list of `SimulationScenario`s as a class constant.
2. Add a field to `Tabs` in `data_structures.py`, e.g.
   `ifn: ScenarioGridTabComponents`.
3. Create the tab in `gradio_interface()`:
   `ifn=self.Scenario_grid_tab("IFN-γ scenarios", self.IFN_SCENARIOS)`.
4. Add `('ifn', tabs.ifn, self.IFN_SCENARIOS)` to the list in
   `handle_scenario_grids()`.

The number of rows follows the length of the scenario list. The number of
columns follows `SCENARIO_GRID_SUBNETWORKS`, and any key of `SUBNETWORKS` can be
used there.

### Panel-letter style

The letter's position and size are set in `DEFAULT_STYLE` (`plotting.py`). The
coordinates are in axes units of the first panel of each row.

| Key | Default | Effect |
|---|---|---|
| `panel_letter_x` | `-0.22` | Horizontal position (negative = left of the axes). |
| `panel_letter_y` | `1.5` | Vertical position of the top of the letter (1.5 = top of the legend block). |
| `panel_letter_font_size` | `30` | Font size (always bold). |

## Bifurcation diagram (2-D)

Pick the 2 axes (TBET, GATA3, FOXP3, RORGT, BCL6) and one or more inputs to
vary. The *Input group* dropdown (`INPUT_GROUPS`, or *All groups*) filters
the input list. The selected inputs all take the same value at each level:
either *Number of levels* values evenly spaced between *From* and *To*, or
the comma-separated *Custom levels*. Everything else comes from the other
tabs: initial conditions, all other inputs (saturation rate, antigens,
nutrients, ...), decay rates, thresholds, activation function and solver.

The colour bar reads `<input> expression level` when one input is varied
and `Cytokines expression level` when several are. Changing the axes redraws
the last diagram without solving again.

## Local sensitivity analysis

For the selected parameters $p_j$ (any input; add more in the *Parameters*
box), the tab integrates $S_{ij}=\partial q_i/\partial p_j$ together with
the network, around the values set in the other tabs. *Swept parameter*
repeats the analysis for each value in *From*–*To* with step *Step*. Pick
*None* for a single run at the nominal inputs. The figure has one row per
output node: activity, then one elasticity panel per parameter. *Validate
with finite differences* adds a figure comparing the exact sensitivity with
central differences. The derivation is in `docs/sensitivity_analysis.md`.

The first run builds the symbolic Jacobian (≈ 1 s). Each run of the sweep
then takes about 0.7 s with one parameter.

## How a simulation flows through the code

```
Gradio button
  └─ inputs = _simulation_inputs(tabs, gd)        # sliders + dropdowns, fixed order
       └─ callback(*values)
            ├─ _unpack_simulation_values(values)  # → dict: nodes, out_params, decay_rates, ...
            ├─ _apply_scenario(sim, scenario)     # scenario grids only
            ├─ _solve_simulation(sim)             # → NetworkEquations.solve(...)
            └─ Plotter.design_*(...)              # → matplotlib Figure → gr.Plot
```

`_simulation_inputs` and `_unpack_simulation_values` must list the components
in the same order. Change both together if you add a new global input.
