from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import gradio as gr
@dataclass
class GeneralDropdowns:
    activation: gr.Dropdown
    method: gr.Dropdown
    distribution: gr.Dropdown


@dataclass
class InputsTabComponents:
    solve_btn: gr.Button
    reset_network: gr.Button
    reset_decay: gr.Button
    sliders_out: List[gr.Slider] = field(default_factory=list)
    sliders_input: List[gr.Slider] = field(default_factory=list)
    sliders_alpha: List[gr.Slider] = field(default_factory=list)


@dataclass
class InitialConditionsTabComponents:
    solve_btn: gr.Button
    restart_btn: gr.Button
    gaussian_dropdown: gr.Dropdown
    sliders_ic: List[gr.Slider] = field(default_factory=list)


@dataclass
class ThresholdsTabComponents:
    solve_btn: gr.Button
    reset_btn: gr.Button
    sliders: List[gr.Slider] = field(default_factory=list)


@dataclass
class ResultsTabComponents:
    solve_btn: gr.Button
    gplot: gr.Plot
    lympho_plot: gr.Plot
    download_btn_fig2: gr.Button
    download_fig2: gr.File
    download_btn_fig3: gr.Button
    download_fig3: gr.File


@dataclass
class ItoTabComponents:
    solve_btn: gr.Button
    seed: gr.Slider
    sigma_global: gr.Number
    apply_sigma_btn: gr.Button
    noise_sliders: List[gr.Slider] = field(default_factory=list)
    gplot: gr.Plot = None
    lympho_plot: gr.Plot = None
    download_btn_fig2: gr.Button = None
    download_fig2: gr.File = None
    download_btn_fig3: gr.Button = None
    download_fig3: gr.File = None


@dataclass
class BooleanToFuzzyTabComponents:
    procesar_btn: gr.Button
    limpiar_btn: gr.Button
    archivo_boolean: gr.File
    archivo_nodes: gr.File
    archivo_inputs: gr.File
    archivo_out_boolean: gr.File
    info_text: gr.Textbox


@dataclass
class CustomPlotTabComponents:
    solve_btn: gr.Button
    select_box: gr.CheckboxGroup
    figura: gr.Plot


@dataclass
class PhaseSpaceTabComponents:
    solve_btn: gr.Button
    select_box: gr.CheckboxGroup
    figura: gr.Plot

@dataclass
class BifurcationTabComponents:
    solve_btn: gr.Button
    select_box: gr.CheckboxGroup        # the 2 master transcription factors (axes)
    input_group: gr.Dropdown            # filters the choices of varied_inputs
    varied_inputs: gr.Dropdown          # inputs varied together (multiselect)
    sweep_start: gr.Number
    sweep_end: gr.Number
    sweep_points: gr.Number
    custom_values: gr.Textbox           # optional comma-separated levels
    figura: gr.Plot
    download_btn: gr.Button
    download_file: gr.File


@dataclass
class SensitivityTabComponents:
    solve_btn: gr.Button
    parameters: gr.Dropdown             # parameters p_j (multiselect)
    nodes: gr.Dropdown                  # output nodes, one figure row each
    sweep_param: gr.Dropdown
    sweep_start: gr.Number
    sweep_end: gr.Number
    sweep_step: gr.Number
    mode: gr.Dropdown                   # elasticity normalisation
    t_end: gr.Number
    n_show: gr.Slider
    fd_check: gr.Checkbox
    plot: gr.Plot
    fd_plot: gr.Plot
    table: gr.Dataframe
    info_text: gr.Textbox
    download_btn: gr.Button
    download_file: gr.File


@dataclass
class ComparisonTabComponents:
    solve_btn: gr.Button
    gplot: gr.Plot
    lympho_plot: gr.Plot
    download_btn_fig2: gr.Button
    download_fig2: gr.File
    download_btn_fig3: gr.Button
    download_fig3: gr.File
@dataclass
class BandwidthTabComponents:
    analyze_btn: gr.Button
    noise_input: gr.Textbox
    target_coverage: gr.Slider
    n_simulations: gr.Slider
    plot: gr.Plot
    results_table: gr.Dataframe
    info_text: gr.Textbox


@dataclass(frozen=True)
class SimulationScenario:
    """One row of a scenario grid: overrides applied on top of the user's settings.

    out_overrides      -> {out-of-network argument name: value}, e.g. {'TCD8086': 5}
    decay_overrides    -> {node name: decay rate}, e.g. {'TBET': 4}
    reset_decay_rates  -> if True, every decay rate starts at the default (1.0)
                          before decay_overrides are applied; otherwise the
                          decay-rate sliders chosen by the user are used.
    """
    label: str
    out_overrides: Dict[str, float] = field(default_factory=dict)
    decay_overrides: Dict[str, float] = field(default_factory=dict)
    reset_decay_rates: bool = False


@dataclass
class ScenarioGridTabComponents:
    solve_btn: gr.Button
    row_letters: gr.Textbox
    plot: gr.Plot
    download_btn: gr.Button
    download_file: gr.File


@dataclass
class Tabs:
    inp: InputsTabComponents
    ic:  InitialConditionsTabComponents
    thr: ThresholdsTabComponents
    res: ResultsTabComponents
    cmp: ComparisonTabComponents
    ito: ItoTabComponents
    bw:  BandwidthTabComponents
    bf:  BooleanToFuzzyTabComponents
    cp:  CustomPlotTabComponents
    ps:  PhaseSpaceTabComponents
    ps2: BifurcationTabComponents
    sens: SensitivityTabComponents
    tcr: ScenarioGridTabComponents
    dec: ScenarioGridTabComponents
