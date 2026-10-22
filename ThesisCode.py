from math import ceil
import string
import time
import numpy as np
import pandas as pd
import gradio as gr
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from subnetworks import SUBNETWORKS
from plotting import Plotter
from data_structures import (
    GeneralDropdowns,
    InputsTabComponents,
    InitialConditionsTabComponents,
    ThresholdsTabComponents,
    ResultsTabComponents,
    ItoTabComponents,
    BooleanToFuzzyTabComponents,
    CustomPlotTabComponents,
    PhaseSpaceTabComponents,
    BifurcationTabComponents,
    SensitivityTabComponents,
    ComparisonTabComponents,
    BandwidthTabComponents,
    ScenarioGridTabComponents,
    SimulationScenario,
    Tabs,
)
from Equations import NetworkEquations
from boolfuzzy_converter import BooleanToFuzzyProbabilistic
from sensitivity import ELASTICITY_MODES, SensitivityAnalyzer, sweep_values
class NetworkInterface(NetworkEquations):
    DEFAULT_MIN_SLIDER = 0.0
    DEFAULT_MAX_SLIDER = 1.0
    DEFAULT_SLIDER_STEP = 0.01
    DEFAULT_SLIDER_VALUE = 0.5

    DEFAULT_TIME_SPAN = (0, 55)
    DEFAULT_DECAY_RATE = 1.0
    DEFAULT_THRESHOLD_VALUE = 0.5

    GAUSSIAN_STD = 0.15
    GAUSSIAN_MEAN_0 = 0
    GAUSSIAN_MEAN_05 = 0.5
    GAUSSIAN_MEAN_1 = 1

    INPUT_GROUPS = {
        'Antigen microenviroment': sorted(['LPSE', 'LP', 'DSRNA', 'SSRNA', 'CPGDNA',
                                           'GMCSFE', 'IC3B', 'IGGC', 'IGAC', 'CITDSRNA', 'CITSSRNA']),
        'Cytokine microenviroment': ['IFNGE', 'IFNGL', 'IL1BE', 'IL4E', 'IL4L', 'IL6E', 'IL10E', 'IL10L', 'IL12E', 'IL18E', 'IL21E', 'IL33E', 'TGFBE','TNFA'],
        'Activation parameters': sorted(['b', 'TCD8086', 'TMHC2', 'MHC2E', 'CD8086E']),
        'Macrophage nutrients': sorted(['MO2', 'MGLC']),
        'Nutrients and molecular therapies': sorted(['GLC', 'O2', 'CIAP', 'GLN', 'MFA', 'FA',
                                                      'TRP', 'METF', 'RAPA', 'PRED', 'CS'])}

    HIDDEN_SLIDERS = ['IL10OUT', 'IL6OUT', 'IL33OUT', 'IL18OUT', 'IL12OUT']

    MASTER_TRANSCRIPTION_FACTORS = ['TBET', 'GATA3', 'FOXP3', 'RORGT', 'BCL6']
    ALL_INPUT_GROUPS = 'All groups'

    # --- Bifurcation diagram (2-D) ---
    DEFAULT_BIFURCATION_GROUP = 'Cytokine microenviroment'
    DEFAULT_BIFURCATION_INPUTS = ['IFNGE']
    DEFAULT_BIFURCATION_RANGE = (0.0, 1.0, 21)      # start, end, number of levels

    # --- Local sensitivity analysis ---
    NO_SWEEP = 'None (nominal inputs only)'
    DEFAULT_SENSITIVITY_PARAMETERS = ['IFNGE']
    DEFAULT_SENSITIVITY_NODES = ['TBET', 'IFNG']
    DEFAULT_SENSITIVITY_SWEEP = ('IFNGE', 0.9, 1.1, 0.1)   # param, start, end, step
    DEFAULT_SENSITIVITY_T_END = 50.0
    MAX_SENSITIVITY_NODES = 8
    MAX_SENSITIVITY_PARAMETERS = 6
    MAX_SWEEP_RUNS = 50

    MAX_OUT_SLIDERS_DEFAULT = [100]
    ACTIVATION_FUNCTIONS = ['sigmoid', 'tanh', 'hill']
    DEFAULT_ACTIVATION = 'sigmoid'
    ODE_SOLVERS = ['DOP853', 'LSODA']
    DEFAULT_SOLVER = 'LSODA'
    DISTRIBUTION_TYPES = ['normal', 'uniform', 'laplace']
    DEFAULT_DISTRIBUTION = 'uniform'

    # --- Scenario grids (one row per simulation, one column per subnetwork) ---
    #SCENARIO_GRID_SUBNETWORKS = ['T Metabolism', 'Antigen presentation','T Transcription factors', 'Th1', 'Treg']
    SCENARIO_GRID_SUBNETWORKS = ['Th1', 'Th2', 'Th17', 'Treg', 'Tfh']
    DEFAULT_ROW_LETTERS = 'A,B,C,D'
    INTERACTION_TIME_SCENARIOS = [
        SimulationScenario('TCD8086 = 5, TMHC2 = 5',
                           out_overrides={'TCD8086': 5, 'TMHC2': 5}),
        SimulationScenario('TCD8086 = 15, TMHC2 = 15',
                           out_overrides={'TCD8086': 15, 'TMHC2': 15}),
        SimulationScenario('TCD8086 = 30, TMHC2 = 25',
                           out_overrides={'TCD8086': 30, 'TMHC2': 25}),
        SimulationScenario('TCD8086 = 45, TMHC2 = 35',
                           out_overrides={'TCD8086': 45, 'TMHC2': 35}),
    ]
    DECAY_RATE_SCENARIOS = [
        SimulationScenario('All decay rates = 1',
                           reset_decay_rates=True),
        SimulationScenario('TBET = 4, others = 1',
                           decay_overrides={'TBET': 4}, reset_decay_rates=True),
        SimulationScenario('TBET = 4, GATA3 = 4, others = 1',
                           decay_overrides={'TBET': 4, 'GATA3': 4},
                           reset_decay_rates=True),
        SimulationScenario('TBET = 4, GATA3 = 4, RORγt = 5, others = 1',
                           decay_overrides={'TBET': 4, 'GATA3': 4, 'RORGT': 5},
                           reset_decay_rates=True),
    ]

    def __init__(self, features: dict):
        super().__init__()
        self.value_sliders = self.initial_conditions
        self.max_sliders_out = [100] + (len(self.out_of_network_arguments) - 3) * [1] + 2 * [100]
        self.value_sliders_out = self.initial_out
        self.distribution_type = None
        self.features = features
        self.subnetwork_options = list(features.keys())
        self.last_results = None
        self.last_stoch_results = None
        self.subnetwork_name = None
        self.last_fig2 = None
        self.last_fig3 = None
        self.last_scenario_figs = {}
        self.last_bifurcation = None
        self.last_bifurcation_fig = None
        self.last_sensitivity_fig = None
        self.sensitivity = SensitivityAnalyzer(self)

        self.plotter = Plotter(features, self._style_config())

        self.gradio_interface()

    def _style_config(self) -> dict:
        return {
            "out_args": self.out_of_network_arguments,
        }

    def _create_slider(self, label: str, minimum: float = DEFAULT_MIN_SLIDER,
                       maximum: float = DEFAULT_MAX_SLIDER,
                       value: float = DEFAULT_SLIDER_VALUE,
                       step: float = DEFAULT_SLIDER_STEP,
                       visible: bool = True) -> gr.Slider:
        return gr.Slider(
            minimum=minimum,
            maximum=maximum,
            step=step,
            label=label,
            value=value,
            interactive=True,
            visible=visible
        )

    def create_slider_group(self, labels, values, minimums, maximums,
                            exceptions=None, n_columns: int = 5):
        sliders = []
        n_rows = (len(labels) + n_columns - 1) // n_columns
        for row in range(n_rows):
            with gr.Row(variant="compact"):
                start_idx = row * n_columns
                end_idx = min((row + 1) * n_columns, len(labels))
                for idx in range(start_idx, end_idx):
                    visibility = exceptions is None or labels[idx] not in exceptions
                    slider = self._create_slider(
                        label=labels[idx],
                        minimum=minimums[idx] if isinstance(minimums, list) else minimums,
                        maximum=maximums[idx] if isinstance(maximums, list) else maximums,
                        value=values[idx],
                        visible=visibility
                    )
                    sliders.append(slider)
        return sliders

    # ============================================================
    # FACTORÍAS DE TABS
    # ============================================================

    def Inputs_tab(self, tab_title='Inputs',
                   n_columns_in_tab: int = 5) -> InputsTabComponents:
        with gr.Tab(tab_title):
            all_sliders_out = len(self.out_of_network_arguments) * [0]

            with gr.Row():
                with gr.Column():
                    solve_btn_in = gr.Button("Simulate network dynamics", variant="primary")
                with gr.Column():
                    reset_network_params = gr.Button('Reset input values', variant='secondary')

            with gr.Column():
                max_out_sliders_reference = dict(zip(self.out_of_network_arguments,
                                                     self.max_sliders_out))
                value_reference = dict(zip(self.out_of_network_arguments,
                                           self.value_sliders_out))

                all_sliders_input = []
                all_sliders_out = len(self.out_of_network_arguments) * [0]

                for group_name, group_labels in self.INPUT_GROUPS.items():
                    gr.Markdown(f'{group_name}')
                    group_values = [value_reference[e] for e in group_labels]
                    group_minimus = len(group_labels) * [0]
                    group_max = [max_out_sliders_reference[k] for k in group_labels]
                    all_sliders_input += self.create_slider_group(
                        group_labels, group_values, group_minimus, group_max,
                        exceptions=None, n_columns=5
                    )

                common_ref = dict(zip(self.out_of_network_arguments.tolist(),
                                      range(len(self.out_of_network_arguments))))
                for element in all_sliders_input:
                    old_idx = common_ref[element.label]
                    all_sliders_out[old_idx] = element

                for name, element in zip(self.out_of_network_names, all_sliders_out):
                    element.label = name

                with gr.Row():
                    reset_decay_rates_sliders = gr.Button('Reset decay rates',
                                                          variant='secondary')

                labels_dr = [f'Decay rate: ({par})' for par in self.array_of_elements]
                values_dr = [self.DEFAULT_DECAY_RATE for _ in range(self.n)]
                all_sliders_alpha = self.create_slider_group(
                    labels_dr, values_dr, minimums=0, maximums=10,
                    exceptions=None, n_columns=5
                )

            reset_network_params.click(
                fn=self.reset_network_parameters,
                outputs=all_sliders_out,
            )
            reset_decay_rates_sliders.click(
                fn=self.set_decay_rates,
                outputs=all_sliders_alpha,
            )

        return InputsTabComponents(
            solve_btn=solve_btn_in,
            reset_network=reset_network_params,
            reset_decay=reset_decay_rates_sliders,
            sliders_out=all_sliders_out,
            sliders_input=all_sliders_input,
            sliders_alpha=all_sliders_alpha,
        )

    def Initial_conditions_tab(self) -> InitialConditionsTabComponents:
        with gr.Tab("Initial conditions"):
            solve_btn_ic = gr.Button("Simulate network dynamics", variant="primary")

            with gr.Row():
                with gr.Column():
                    restart_initial_conditions = gr.Button(
                        'Reset initial conditions', variant='secondary'
                    )
                with gr.Column():
                    set_gaussian_ic = gr.Dropdown(
                        choices=['0', '0.5', '1'],
                        label='Gaussian distribution for initial conditions',
                        value='0',
                    )

            nodes_names = self.array_of_elements.tolist()
            nodes_values = self.value_sliders.tolist()
            all_sliders_ic = self.create_slider_group(
                labels=nodes_names,
                values=nodes_values,
                minimums=self.DEFAULT_MIN_SLIDER,
                maximums=self.DEFAULT_MAX_SLIDER,
                exceptions=self.HIDDEN_SLIDERS,
            )
            set_gaussian_ic.change(
                fn=self.set_gaussian_conditions,
                inputs=set_gaussian_ic,
                outputs=all_sliders_ic,
            )
            restart_initial_conditions.click(
                fn=self.restart_sliders_value,
                outputs=all_sliders_ic,
            )

        return InitialConditionsTabComponents(
            solve_btn=solve_btn_ic,
            restart_btn=restart_initial_conditions,
            gaussian_dropdown=set_gaussian_ic,
            sliders_ic=all_sliders_ic,
        )

    def Thresholds_tab(self) -> ThresholdsTabComponents:
        with gr.Tab("Threshold values"):
            with gr.Row():
                with gr.Column():
                    solve_btn_tv = gr.Button("Simulate network dynamics", variant="primary")
                with gr.Column():
                    reset_threshold_values = gr.Button('Reset threshold values',
                                                        variant='secondary')

            labels_um = [f'Threshold U ({par})' for par in self.array_of_elements]
            values_um = [self.DEFAULT_THRESHOLD_VALUE for _ in range(self.n)]
            all_sliders_threshold = self.create_slider_group(
                labels_um, values_um,
                minimums=self.DEFAULT_MIN_SLIDER,
                maximums=self.DEFAULT_MAX_SLIDER,
                exceptions=None, n_columns=5,
            )
            reset_threshold_values.click(
                fn=self.reset_threshold_values,
                outputs=all_sliders_threshold,
            )

        return ThresholdsTabComponents(
            solve_btn=solve_btn_tv,
            reset_btn=reset_threshold_values,
            sliders=all_sliders_threshold,
        )

    def Results_tab(self, title="Results") -> ResultsTabComponents:
        with gr.Tab(title):
            solve_btn_re = gr.Button("Simulate network dynamics", variant="primary")
            with gr.Row():
                GPlot = gr.Plot()
            with gr.Row():
                LymphoPlot = gr.Plot()
            with gr.Row():
                with gr.Column():
                    download_btn_fig2 = gr.Button(
                        "Download Figure 2 (Subplots 0-10) as JPG", variant="secondary"
                    )
                    download_fig2 = gr.File(label="Download Fig2", visible=True)
                with gr.Column():
                    download_btn_fig3 = gr.Button(
                        "Download Figure 3 (Subplots 10-20) as JPG", variant="secondary"
                    )
                    download_fig3 = gr.File(label="Download Fig3", visible=True)

        return ResultsTabComponents(
            solve_btn=solve_btn_re,
            gplot=GPlot,
            lympho_plot=LymphoPlot,
            download_btn_fig2=download_btn_fig2,
            download_fig2=download_fig2,
            download_btn_fig3=download_btn_fig3,
            download_fig3=download_fig3,
        )

    def Comparison_tab(self, title: str = "Deterministic vs Stochastic") -> ComparisonTabComponents:
        with gr.Tab(title):
            solve_btn = gr.Button("Simulate deterministic vs stochastic", variant="primary")
            with gr.Row():
                GPlot = gr.Plot()
            with gr.Row():
                LymphoPlot = gr.Plot()
            with gr.Row():
                with gr.Column():
                    download_btn_fig2 = gr.Button(
                        "Download Figure 2 (comparison 0-10) as JPG",
                        variant="secondary",
                    )
                    download_fig2 = gr.File(label="Download Fig2", visible=True)
                with gr.Column():
                    download_btn_fig3 = gr.Button(
                        "Download Figure 3 (comparison 10-20) as JPG",
                        variant="secondary",
                    )
                    download_fig3 = gr.File(label="Download Fig3", visible=True)

        return ComparisonTabComponents(
            solve_btn=solve_btn,
            gplot=GPlot,
            lympho_plot=LymphoPlot,
            download_btn_fig2=download_btn_fig2,
            download_fig2=download_fig2,
            download_btn_fig3=download_btn_fig3,
            download_fig3=download_fig3,
        )

    def set_all_sigma(self, value, *current_sigmas):
        if value is None:
            value = 0.0
        return [float(value)] * len(current_sigmas)

    def Stochastic_Ito_tab(self) -> ItoTabComponents:
        with gr.Tab("Stochastic Itô"):
            with gr.Row():
                solve_btn_ito = gr.Button("Simulate stochastic dynamics (Itô)",
                                          variant="primary")
            with gr.Row():
                seed_slider = gr.Slider(
                    minimum=0, maximum=9999, step=1, value=42,
                    label="Random seed",
                )

            gr.Markdown("### Noise intensity per node (diagonal of G)")

            noise_sliders = []
            n_cols = 5
            labels = list(self.array_of_elements)
            n_rows = (len(labels) + n_cols - 1) // n_cols

            with gr.Row():
                sigma_global = gr.Number(
                    value=0.0,
                    label="Set all σ sliders to this value",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    precision=2,
                )
                apply_sigma_global = gr.Button("Apply to all σ", variant="secondary")

            for row in range(n_rows):
                with gr.Row(variant="compact"):
                    start = row * n_cols
                    end = min((row + 1) * n_cols, len(labels))
                    for idx in range(start, end):
                        s = gr.Slider(
                            minimum=0.0, maximum=1.0, step=0.01, value=0.0,
                            label=f"σ ({labels[idx]})",
                            interactive=True,
                        )
                        noise_sliders.append(s)

            with gr.Row():
                GPlot_ito = gr.Plot()
            with gr.Row():
                LymphoPlot_ito = gr.Plot()
            with gr.Row():
                download_btn_fig2_ito = gr.Button("Download Figure 2 (Itô) as JPG",
                                                  variant="secondary")
                download_fig2_ito = gr.File(label="Download Fig2 Itô")
            with gr.Row():
                download_btn_fig3_ito = gr.Button("Download Figure 3 (Itô) as JPG",
                                                  variant="secondary")
                download_fig3_ito = gr.File(label="Download Fig3 Itô")

            apply_sigma_global.click(
                fn=self.set_all_sigma,
                inputs=[sigma_global] + noise_sliders,
                outputs=noise_sliders,
            )

        return ItoTabComponents(
            solve_btn=solve_btn_ito,
            seed=seed_slider,
            sigma_global=sigma_global,
            apply_sigma_btn=apply_sigma_global,
            noise_sliders=noise_sliders,
            gplot=GPlot_ito,
            lympho_plot=LymphoPlot_ito,
            download_btn_fig2=download_btn_fig2_ito,
            download_fig2=download_fig2_ito,
            download_btn_fig3=download_btn_fig3_ito,
            download_fig3=download_fig3_ito,
        )

    def Boolean_to_Fuzzy_tab(self) -> BooleanToFuzzyTabComponents:
        with gr.Tab("Convert and process your own boolean networks"):
            with gr.Row():
                procesar_btn = gr.Button("Process file", variant="primary", size="lg")
                limpiar_btn = gr.Button("Delete file", variant="secondary")
            with gr.Row():
                with gr.Column():
                    archivo_boolean = gr.File(
                        label="📄 Boolean rules file", file_types=[".txt"]
                    )
                with gr.Column():
                    archivo_nodes = gr.File(
                        label="📄 Nodes file", file_types=[".txt"]
                    )
                with gr.Column():
                    archivo_inputs = gr.File(
                        label="📄 Inputs file", file_types=[".txt"]
                    )
            with gr.Row():
                with gr.Column():
                    archivo_out_boolean = gr.File(
                        label="📥 Probablistic expressions file", interactive=False
                    )
            info_text = gr.Textbox(label="ℹ️ Information", lines=3)

        return BooleanToFuzzyTabComponents(
            procesar_btn=procesar_btn,
            limpiar_btn=limpiar_btn,
            archivo_boolean=archivo_boolean,
            archivo_nodes=archivo_nodes,
            archivo_inputs=archivo_inputs,
            archivo_out_boolean=archivo_out_boolean,
            info_text=info_text,
        )

    def Plot_custom_tab(self) -> CustomPlotTabComponents:
        with gr.Tab("Plot customization"):
            with gr.Row():
                solve_btn_pc = gr.Button("Simulate network dynamics", variant="primary")
            with gr.Row():
                select_box = gr.CheckboxGroup(
                    choices=list(self.array_of_elements),
                    value=['OXPHOS', 'AMPK', 'Glycolysis'],
                    label="Choose the nodes to plot (15 nodes max)",
                )
            with gr.Row():
                figura = gr.Plot()

        return CustomPlotTabComponents(
            solve_btn=solve_btn_pc,
            select_box=select_box,
            figura=figura,
        )

    def phase_space(self) -> PhaseSpaceTabComponents:
        with gr.Tab("Phase Space (3-D)"):
            with gr.Row():
                solve_btn_ps = gr.Button("Plot Phase Space", variant="primary")
            with gr.Row():
                select_box = gr.CheckboxGroup(
                    choices=self.MASTER_TRANSCRIPTION_FACTORS,
                    value=['TBET', 'GATA3', 'FOXP3'],
                    label="Choose the 3 axes for the Phase Space (3D representation)",
                )
            gr.Markdown("Drag to rotate, scroll to zoom. The dashed black line "
                        "joins the steady states reached at each cytokine level.")
            with gr.Row():
                with gr.Column():
                    figura_ps = gr.Plot()

        return PhaseSpaceTabComponents(
            solve_btn=solve_btn_ps,
            select_box=select_box,
            figura=figura_ps,
        )

    def _inputs_of_group(self, group: str) -> list:
        if group == self.ALL_INPUT_GROUPS:
            return [name for names in self.INPUT_GROUPS.values() for name in names]
        return list(self.INPUT_GROUPS[group])

    def Bifurcation_tab(self, title="Bifurcation diagram (2-D)") -> BifurcationTabComponents:
        start, end, points = self.DEFAULT_BIFURCATION_RANGE
        with gr.Tab(title):
            with gr.Row():
                solve_btn = gr.Button("Plot bifurcation diagram", variant="primary")
            gr.Markdown(
                "Initial conditions, inputs (saturation rate, nutrients, ...), "
                "decay rates, thresholds, activation function and solver are "
                "taken from the other tabs. Only the inputs selected below are "
                "changed: all of them take the same value at each level, and "
                "that level colours the trajectory.")
            with gr.Row():
                select_box = gr.CheckboxGroup(
                    choices=self.MASTER_TRANSCRIPTION_FACTORS,
                    value=['TBET', 'GATA3'],
                    label="Choose the 2 axes (master transcription factors)",
                )
            with gr.Row():
                input_group = gr.Dropdown(
                    choices=[self.ALL_INPUT_GROUPS] + list(self.INPUT_GROUPS),
                    value=self.DEFAULT_BIFURCATION_GROUP,
                    label="Input group",
                )
                varied_inputs = gr.Dropdown(
                    choices=self._inputs_of_group(self.DEFAULT_BIFURCATION_GROUP),
                    value=self.DEFAULT_BIFURCATION_INPUTS,
                    multiselect=True,
                    label="Inputs to vary (one or more, varied together)",
                )
            with gr.Row():
                sweep_start = gr.Number(value=start, label="From")
                sweep_end = gr.Number(value=end, label="To")
                sweep_points = gr.Number(value=points, precision=0, minimum=2,
                                         label="Number of levels")
                custom_values = gr.Textbox(
                    value="", label="Custom levels (comma-separated, overrides the range)")
            with gr.Row():
                figura = gr.Plot()
            with gr.Row():
                with gr.Column():
                    download_btn = gr.Button("Download figure as JPG", variant="secondary")
                    download_file = gr.File(label="Download figure", visible=True)

        return BifurcationTabComponents(
            solve_btn=solve_btn,
            select_box=select_box,
            input_group=input_group,
            varied_inputs=varied_inputs,
            sweep_start=sweep_start,
            sweep_end=sweep_end,
            sweep_points=sweep_points,
            custom_values=custom_values,
            figura=figura,
            download_btn=download_btn,
            download_file=download_file,
        )

    def Sensitivity_tab(self, title="Local sensitivity analysis") -> SensitivityTabComponents:
        sweep_param, start, end, step = self.DEFAULT_SENSITIVITY_SWEEP
        all_inputs = self._inputs_of_group(self.ALL_INPUT_GROUPS)
        with gr.Tab(title):
            with gr.Row():
                solve_btn = gr.Button("Run sensitivity analysis", variant="primary")
            gr.Markdown(
                "Forward (local) sensitivity analysis around the values set in the "
                "other tabs (initial conditions, inputs, decay rates, thresholds, "
                "activation function, solver). For every parameter $p_j$ selected "
                "below, $S_{ij}(t)=\\partial q_i/\\partial p_j$ is integrated "
                "together with the network. See `docs/sensitivity_analysis.md`.")
            with gr.Row():
                parameters = gr.Dropdown(
                    choices=all_inputs, value=self.DEFAULT_SENSITIVITY_PARAMETERS,
                    multiselect=True,
                    label="Parameters p_j (add more to extend the analysis)",
                )
                nodes = gr.Dropdown(
                    choices=self.array_of_elements.tolist(),
                    value=self.DEFAULT_SENSITIVITY_NODES, multiselect=True,
                    label="Output nodes (one figure row each)",
                )
            with gr.Row():
                sweep_dropdown = gr.Dropdown(
                    choices=[self.NO_SWEEP] + all_inputs, value=sweep_param,
                    label="Swept parameter (nominal value varied over the range)",
                )
                sweep_start = gr.Number(value=start, label="From")
                sweep_end = gr.Number(value=end, label="To")
                sweep_step = gr.Number(value=step, label="Step")
            with gr.Row():
                mode = gr.Dropdown(choices=ELASTICITY_MODES, value='relative_pointwise',
                                   label="Elasticity normalisation")
                t_end = gr.Number(value=self.DEFAULT_SENSITIVITY_T_END, minimum=1,
                                  label="Final time")
                n_show = gr.Slider(minimum=1, maximum=20, step=1, value=10,
                                   label="Max curves per panel")
                fd_check = gr.Checkbox(value=False,
                                       label="Validate with finite differences")
            with gr.Row():
                plot = gr.Plot(label="Activities and elasticity coefficients")
            with gr.Row():
                fd_plot = gr.Plot(label="Sensitivity ODE vs finite differences")
            with gr.Row():
                table = gr.Dataframe(label="Summary (min / max over time and sweep)")
            with gr.Row():
                info_text = gr.Textbox(label="ℹ️ Info", lines=3)
            with gr.Row():
                with gr.Column():
                    download_btn = gr.Button("Download figure as JPG", variant="secondary")
                    download_file = gr.File(label="Download figure", visible=True)

        return SensitivityTabComponents(
            solve_btn=solve_btn,
            parameters=parameters,
            nodes=nodes,
            sweep_param=sweep_dropdown,
            sweep_start=sweep_start,
            sweep_end=sweep_end,
            sweep_step=sweep_step,
            mode=mode,
            t_end=t_end,
            n_show=n_show,
            fd_check=fd_check,
            plot=plot,
            fd_plot=fd_plot,
            table=table,
            info_text=info_text,
            download_btn=download_btn,
            download_file=download_file,
        )

    def update_phase_space(self, selected_transcription_factors: list):
        if selected_transcription_factors is None or len(selected_transcription_factors) != 3:
            selected_transcription_factors = ['TBET', 'GATA3', 'FOXP3']

        out_params = dict(zip(self.out_of_network_arguments, self.initial_out))
        fixed_inputs = ['LPSE', 'SSRNA']
        for fix_inp in fixed_inputs:
            out_params[fix_inp] = 1

        variable_inputs = ['IFNGE', 'IL33E', 'IL4L', 'TGFBE']
        input_combinations = []
        levels = []
        for i in range(21):
            levels.append(0.05 * i)
            for variable in variable_inputs:
                out_params[variable] = 0.05 * i
            combination = list(out_params.values())
            input_combinations.append(combination)

        solutions_for_sp = []
        for l_inputs in input_combinations:
            solution_sp = self.solve(grout_of_the_network_arguments_eval=l_inputs)
            solution_sp = [solution_sp['nodes'][trans_f]['values']
                           for trans_f in selected_transcription_factors]
            solutions_for_sp.append(solution_sp)

        return self.plotter.plot_phase_space(selected_transcription_factors,
                                             solutions_for_sp, levels)

    def _bifurcation_levels(self, start, end, points, custom: str) -> list:
        if custom and custom.strip():
            try:
                levels = [float(x) for x in custom.split(',') if x.strip()]
            except ValueError:
                raise gr.Error("Custom levels must be numbers separated by commas.")
        else:
            if start is None or end is None or points is None or int(points) < 2:
                raise gr.Error("Set 'From', 'To' and at least 2 levels.")
            levels = np.linspace(float(start), float(end), int(points)).tolist()
        if len(levels) < 2:
            raise gr.Error("At least 2 levels are needed for a bifurcation diagram.")
        return levels

    @staticmethod
    def _bifurcation_colorbar_label(varied_inputs: list) -> str:
        if len(varied_inputs) == 1:
            return f'{varied_inputs[0]} expression level'
        return 'Cytokines expression level'

    def run_bifurcation(self, axes, varied_inputs, start, end, points, custom,
                        *values, progress=gr.Progress()):
        """Solve one simulation per level of the varied inputs; plot `axes`."""
        if not axes or len(axes) != 2:
            raise gr.Error("Select exactly 2 transcription factors as axes.")
        if not varied_inputs:
            raise gr.Error("Select at least one input to vary.")
        levels = self._bifurcation_levels(start, end, points, custom)

        sim = self._unpack_simulation_values(values)
        trajectories = {tf: [] for tf in self.MASTER_TRANSCRIPTION_FACTORS}
        for i, level in enumerate(levels):
            scenario = SimulationScenario(
                f'level {level}', out_overrides={name: level for name in varied_inputs})
            results = self._solve_simulation(self._apply_scenario(sim, scenario))
            for tf in trajectories:
                trajectories[tf].append(results['nodes'][tf]['values'])
            progress((i + 1) / len(levels), desc="Solving levels")

        self.last_bifurcation = {
            'levels': levels,
            'trajectories': trajectories,
            'label': self._bifurcation_colorbar_label(varied_inputs),
        }
        return self.replot_bifurcation(axes)

    def replot_bifurcation(self, axes):
        """Redraw the last bifurcation diagram with other axes (no new solves)."""
        if self.last_bifurcation is None or not axes or len(axes) != 2:
            return gr.skip()
        data = self.last_bifurcation
        solutions = [[data['trajectories'][tf][i] for tf in axes]
                     for i in range(len(data['levels']))]
        fig = self.plotter.plot_phase_space_2d(axes, solutions, data['levels'],
                                               colorbar_label=data['label'])
        self.last_bifurcation_fig = fig
        return fig

    def run_sensitivity(self, parameters, nodes, sweep_param, start, end, step,
                        mode, t_end, n_show, fd_check, *values,
                        progress=gr.Progress()):
        if not parameters:
            raise gr.Error("Select at least one parameter.")
        if not nodes:
            raise gr.Error("Select at least one output node.")
        if len(parameters) > self.MAX_SENSITIVITY_PARAMETERS:
            raise gr.Error(f"Select at most {self.MAX_SENSITIVITY_PARAMETERS} parameters.")
        if len(nodes) > self.MAX_SENSITIVITY_NODES:
            raise gr.Error(f"Select at most {self.MAX_SENSITIVITY_NODES} nodes.")
        if t_end is None or t_end <= 0:
            raise gr.Error("The final time must be positive.")

        if sweep_param == self.NO_SWEEP:
            sweep_param, levels = None, None
        else:
            try:
                levels = sweep_values(float(start), float(end), float(step))
            except (TypeError, ValueError) as e:
                raise gr.Error(f"Invalid sweep range: {e}")
            if len(levels) > self.MAX_SWEEP_RUNS:
                raise gr.Error(f"The sweep has {len(levels)} values; "
                               f"use at most {self.MAX_SWEEP_RUNS}.")

        sim = self._unpack_simulation_values(values)
        solver = dict(q0=sim['nodes'], decay=sim['decay_rates'],
                      thresholds=sim['threshold_values'], af=sim['activation'],
                      method=sim['method'], t_span=(0.0, float(t_end)))

        t0 = time.time()
        progress(0, desc="Building the symbolic Jacobian")
        self.sensitivity.build_jacobian()
        try:
            result = self.sensitivity.run(
                sim['out_params'], parameters, sweep_param, levels,
                progress=lambda f: progress(f, desc="Integrating sensitivities"),
                **solver)
        except RuntimeError as e:
            raise gr.Error(str(e))
        elapsed = time.time() - t0

        fig = self.plotter.plot_sensitivity_grid(result, nodes, mode, int(n_show))
        self.last_sensitivity_fig = fig
        table = pd.DataFrame(self.sensitivity.summary(result, nodes, mode)).round(6)

        n_runs = len(result.sweep_values)
        info = [f"{n_runs} augmented integration(s) of {self.n * (1 + len(parameters))} "
                f"states (155 nodes + 155 x {len(parameters)} sensitivities) "
                f"in {elapsed:.1f} s."]

        fd_fig = None
        if fd_check:
            p_check = result.p[n_runs // 2]
            checks = []
            for param in parameters:
                progress(0.5, desc=f"Finite differences for {param}")
                check = self.sensitivity.finite_difference_check(
                    p_check, param, nodes[0], **solver)
                checks.append(check)
                errors = ', '.join(f"Δ={d:.0e}: {np.max(np.abs(fd - check['exact'])):.1e}"
                                   for d, fd in check['fd'].items())
                info.append(f"d{nodes[0]}/d{param}: max |exact - FD| ({errors}); "
                            f"max |q_aug - q_base| = {check['augmented_vs_base']:.1e}.")
            fd_fig = self.plotter.plot_sensitivity_fd_check(checks)

        return fig, fd_fig, table, "\n".join(info)

    def general_dropdowns(self) -> GeneralDropdowns:
        activation = gr.Dropdown(
            choices=self.ACTIVATION_FUNCTIONS,
            label="Activation function",
            value=self.DEFAULT_ACTIVATION,
            render=False,
        )
        method = gr.Dropdown(
            choices=self.ODE_SOLVERS,
            label="ODE Solver",
            value=self.DEFAULT_SOLVER,
            render=False,
        )
        distribution = gr.Dropdown(
            choices=self.DISTRIBUTION_TYPES,
            value=self.DEFAULT_DISTRIBUTION,
            label='Distribution to simulate network dynamics',
            render=False,
        )
        return GeneralDropdowns(
            activation=activation,
            method=method,
            distribution=distribution,
        )

    def handle_ito(self, tabs: Tabs, gd: GeneralDropdowns) -> None:
        variables_to_control_ito = (
            tabs.ic.sliders_ic
            + tabs.inp.sliders_out
            + tabs.inp.sliders_alpha
            + tabs.thr.sliders
            + [gd.activation, tabs.cp.select_box]
            + [tabs.ito.seed]
            + tabs.ito.noise_sliders
        )
        tabs.ito.solve_btn.click(
            fn=self.after_push_ito,
            inputs=variables_to_control_ito,
            outputs=[tabs.cp.figura, tabs.ito.gplot, tabs.ito.lympho_plot],
        )

    def _simulation_inputs(self, tabs: Tabs, gd: GeneralDropdowns) -> list:
        """Components read by every deterministic simulation (order matters:
        it must match `_unpack_simulation_values`)."""
        return (
            tabs.ic.sliders_ic
            + tabs.inp.sliders_out
            + tabs.inp.sliders_alpha
            + tabs.thr.sliders
            + [gd.activation, gd.method, tabs.cp.select_box]
        )

    def handle_comparison_simulation(self, tabs: Tabs, gd: GeneralDropdowns) -> None:
        tabs.cmp.solve_btn.click(
            fn=self.after_push_button_comparison,
            inputs=self._simulation_inputs(tabs, gd),
            outputs=[tabs.cp.figura, tabs.cmp.gplot, tabs.cmp.lympho_plot],
        )

    def handle_global_dropdowns(self, gd: GeneralDropdowns, tabs: Tabs) -> None:
        gd.distribution.change(
            fn=self.set_distribution,
            inputs=gd.distribution,
            outputs=tabs.ic.sliders_ic,
        )

    def handle_plot_selectors(self, tabs: Tabs) -> None:
        tabs.cp.select_box.change(
            fn=self.update_general_plot,
            inputs=tabs.cp.select_box,
            outputs=tabs.cp.figura,
        )
        for trigger in [tabs.ps.select_box.change, tabs.ps.solve_btn.click]:
            trigger(
                fn=self.update_phase_space,
                inputs=tabs.ps.select_box,
                outputs=tabs.ps.figura,
            )

    def handle_bifurcation(self, tabs: Tabs, gd: GeneralDropdowns) -> None:
        tab = tabs.ps2
        tab.solve_btn.click(
            fn=self.run_bifurcation,
            inputs=[tab.select_box, tab.varied_inputs, tab.sweep_start,
                    tab.sweep_end, tab.sweep_points, tab.custom_values]
                   + self._simulation_inputs(tabs, gd),
            outputs=tab.figura,
        )
        tab.select_box.change(
            fn=self.replot_bifurcation,
            inputs=tab.select_box,
            outputs=tab.figura,
        )
        tab.input_group.change(
            fn=lambda group, selected: gr.update(
                choices=self._inputs_of_group(group),
                value=[x for x in (selected or []) if x in self._inputs_of_group(group)]),
            inputs=[tab.input_group, tab.varied_inputs],
            outputs=tab.varied_inputs,
        )
        tab.download_btn.click(
            fn=lambda: self.plotter.save_matplotlib_as_jpg(
                self.last_bifurcation_fig, "bifurcation_diagram.jpg")
            if self.last_bifurcation_fig else None,
            outputs=tab.download_file,
        )

    def handle_sensitivity(self, tabs: Tabs, gd: GeneralDropdowns) -> None:
        tab = tabs.sens
        tab.solve_btn.click(
            fn=self.run_sensitivity,
            inputs=[tab.parameters, tab.nodes, tab.sweep_param, tab.sweep_start,
                    tab.sweep_end, tab.sweep_step, tab.mode, tab.t_end,
                    tab.n_show, tab.fd_check]
                   + self._simulation_inputs(tabs, gd),
            outputs=[tab.plot, tab.fd_plot, tab.table, tab.info_text],
        )
        tab.download_btn.click(
            fn=lambda: self.plotter.save_matplotlib_as_jpg(
                self.last_sensitivity_fig, "sensitivity_analysis.jpg")
            if self.last_sensitivity_fig else None,
            outputs=tab.download_file,
        )

    def render_additional_settings(self, gd: GeneralDropdowns) -> None:
        with gr.Tab("Additional settings"):
            with gr.Row(variant="compact"):
                gd.activation.render()
                gd.method.render()
            with gr.Row(variant="compact"):
                gd.distribution.render()

    def handle_simulation(self, tabs: Tabs, gd: GeneralDropdowns) -> None:
        variables_to_control = self._simulation_inputs(tabs, gd)
        for button in [tabs.ic.solve_btn, tabs.inp.solve_btn, tabs.thr.solve_btn,
                       tabs.cp.solve_btn, tabs.res.solve_btn]:
            button.click(
                fn=self.after_push_button,
                inputs=variables_to_control,
                outputs=[tabs.cp.figura, tabs.res.gplot, tabs.res.lympho_plot],
            )

    def handle_downloads(self, tabs: Tabs) -> None:
        tabs.res.download_btn_fig2.click(
            fn=lambda: self.plotter.save_matplotlib_as_jpg(self.last_fig2, "subplots_0_10.jpg")
            if self.last_fig2 else None,
            outputs=tabs.res.download_fig2,
        )
        tabs.res.download_btn_fig3.click(
            fn=lambda: self.plotter.save_matplotlib_as_jpg(self.last_fig3, "subplots_10_20.jpg")
            if self.last_fig3 else None,
            outputs=tabs.res.download_fig3,
        )
        tabs.ito.download_btn_fig2.click(
            fn=lambda: self.plotter.save_matplotlib_as_jpg(self.last_fig2, "ito_subplots_0_10.jpg")
            if self.last_fig2 else None,
            outputs=tabs.ito.download_fig2,
        )
        tabs.ito.download_btn_fig3.click(
            fn=lambda: self.plotter.save_matplotlib_as_jpg(self.last_fig3, "ito_subplots_10_20.jpg")
            if self.last_fig3 else None,
            outputs=tabs.ito.download_fig3,
        )

    def handle_comparison_downloads(self, tabs: Tabs) -> None:
        tabs.cmp.download_btn_fig2.click(
            fn=lambda: self.plotter.save_matplotlib_as_jpg(
                self.last_fig2, "comparison_subplots_0_10.jpg")
            if self.last_fig2 else None,
            outputs=tabs.cmp.download_fig2,
        )
        tabs.cmp.download_btn_fig3.click(
            fn=lambda: self.plotter.save_matplotlib_as_jpg(
                self.last_fig3, "comparison_subplots_10_20.jpg")
            if self.last_fig3 else None,
            outputs=tabs.cmp.download_fig3,
        )

    def handle_scenario_grids(self, tabs: Tabs, gd: GeneralDropdowns) -> None:
        simulation_inputs = self._simulation_inputs(tabs, gd)
        for key, tab, scenarios in [
            ('interaction_times', tabs.tcr, self.INTERACTION_TIME_SCENARIOS),
            ('decay_rates', tabs.dec, self.DECAY_RATE_SCENARIOS),
        ]:
            tab.solve_btn.click(
                fn=self._make_scenario_grid_callback(key, scenarios),
                inputs=[tab.row_letters] + simulation_inputs,
                outputs=tab.plot,
            )
            tab.download_btn.click(
                fn=lambda key=key: self.plotter.save_matplotlib_as_jpg(
                    self.last_scenario_figs[key], f"{key}_grid.jpg")
                if key in self.last_scenario_figs else None,
                outputs=tab.download_file,
            )

    def handle_boolean_to_fuzzy(self, tabs: Tabs) -> None:
        convertidor = BooleanToFuzzyProbabilistic()
        tabs.bf.procesar_btn.click(
            fn=convertidor.process_file,
            inputs=[tabs.bf.archivo_boolean, tabs.bf.archivo_nodes,
                    tabs.bf.archivo_inputs],
            outputs=[tabs.bf.archivo_out_boolean, tabs.bf.info_text],
        )

    def handle_bandwidth_analysis(self, tabs: Tabs) -> None:
        tabs.bw.analyze_btn.click(
            fn=self.run_bandwidth_analysis,
            inputs=[tabs.bw.noise_input,
                    tabs.bw.target_coverage,
                    tabs.bw.n_simulations],
            outputs=[tabs.bw.plot,
                     tabs.bw.results_table,
                     tabs.bw.info_text],
        )

    def Bandwidth_analysis_tab(self,
                            title="Bandwidth analysis") -> BandwidthTabComponents:
        with gr.Tab(title):
            with gr.Row():
                analyze_btn = gr.Button("Run bandwidth analysis", variant="primary")

            with gr.Row():
                noise_input = gr.Textbox(
                    value="0.05,0.06,0.1,0.2,0.3,0.4",
                    label="Stochastic noise levels (comma-separated)",
                )
                target_coverage = gr.Slider(
                    minimum=0.5, maximum=1.0, step=0.01, value=0.95,
                    label="Target coverage",
                )
                n_simulations = gr.Slider(
                    minimum=10, maximum=500, step=10, value=100,
                    label="N simulations per level",
                )

            with gr.Row():
                plot = gr.Plot(label="Efficiency vs stochastic noise")

            with gr.Row():
                results_table = gr.Dataframe(
                    headers=["noise", "best_p", "efficiency", "coverage"],
                    datatype=["number", "number", "number", "number"],
                    label="Bandwidth analysis results",
                )

            with gr.Row():
                info_text = gr.Textbox(label="ℹ️ Info", lines=2)

        return BandwidthTabComponents(
            analyze_btn=analyze_btn,
            noise_input=noise_input,
            target_coverage=target_coverage,
            n_simulations=n_simulations,
            plot=plot,
            results_table=results_table,
            info_text=info_text,
        )

    def Scenario_grid_tab(self, title: str,
                          scenarios: list) -> ScenarioGridTabComponents:
        with gr.Tab(title):
            with gr.Row():
                with gr.Column():
                    solve_btn = gr.Button("Simulate scenarios", variant="primary")
                with gr.Column():
                    row_letters = gr.Textbox(
                        value=self.DEFAULT_ROW_LETTERS,
                        label="Row letters (comma-separated)",
                    )
            gr.Markdown(
                "Initial conditions, inputs, thresholds, decay rates, activation "
                "function and solver are taken from the other tabs; each row "
                "overrides them as follows:\n\n"
                + "\n".join(f"- **Row {i + 1}:** {sc.label}"
                            for i, sc in enumerate(scenarios))
            )
            with gr.Row():
                plot = gr.Plot()
            with gr.Row():
                with gr.Column():
                    download_btn = gr.Button("Download figure as JPG",
                                             variant="secondary")
                    download_file = gr.File(label="Download figure", visible=True)

        return ScenarioGridTabComponents(
            solve_btn=solve_btn,
            row_letters=row_letters,
            plot=plot,
            download_btn=download_btn,
            download_file=download_file,
        )

    def gradio_interface(self, n_columns: int = 5):
        with gr.Blocks(title="Network simulator") as demo:
            gd = self.general_dropdowns()
            with gr.Tabs():
                tabs = Tabs(
                    inp=self.Inputs_tab('Inputs', n_columns),
                    ic=self.Initial_conditions_tab(),
                    thr=self.Thresholds_tab(),
                    res=self.Results_tab("Results"),
                    tcr=self.Scenario_grid_tab("TCR/CD28 interaction times",
                                               self.INTERACTION_TIME_SCENARIOS),
                    dec=self.Scenario_grid_tab("Decay-rate scenarios",
                                               self.DECAY_RATE_SCENARIOS),
                    cmp=self.Comparison_tab("Deterministic vs Stochastic"),
                    ito=self.Stochastic_Ito_tab(),
                    bw=self.Bandwidth_analysis_tab("Bandwidth analysis"),
                    bf=self.Boolean_to_Fuzzy_tab(),
                    cp=self.Plot_custom_tab(),
                    ps=self.phase_space(),
                    ps2=self.Bifurcation_tab("Bifurcation diagram (2-D)"),
                    sens=self.Sensitivity_tab("Local sensitivity analysis"),
                )
                self.render_additional_settings(gd)
            self.handle_global_dropdowns(gd, tabs)
            self.handle_plot_selectors(tabs)
            self.handle_simulation(tabs, gd)
            self.handle_comparison_simulation(tabs, gd)
            self.handle_ito(tabs, gd)
            self.handle_scenario_grids(tabs, gd)
            self.handle_bifurcation(tabs, gd)
            self.handle_sensitivity(tabs, gd)
            self.handle_bandwidth_analysis(tabs)
            self.handle_downloads(tabs)
            self.handle_comparison_downloads(tabs)
            self.handle_boolean_to_fuzzy(tabs)
        return demo.launch(share=False)

    # ============================================================
    # CALLBACKS DE SIMULACIÓN
    # ============================================================

    def _unpack_simulation_values(self, values) -> dict:
        """Split the flat tuple produced by `_simulation_inputs` into named groups."""
        n, m = self.n, len(self.out_of_network_arguments)
        return {
            'nodes':            list(values[:n]),
            'out_params':       list(values[n:n + m]),
            'decay_rates':      list(values[n + m:n + m + n]),
            'threshold_values': list(values[n + m + n:n + m + 2 * n]),
            'activation':       values[n + m + 2 * n],
            'method':           values[n + m + 2 * n + 1],
            'selection':        values[n + m + 2 * n + 2],
        }

    def _solve_simulation(self, sim: dict) -> dict:
        return self.solve(
            t_span=self.DEFAULT_TIME_SPAN,
            t_eval=None,
            grinitial_conditions=sim['nodes'],
            grmethod=sim['method'],
            grout_of_the_network_arguments_eval=sim['out_params'],
            af=sim['activation'],
            decay_rates_list=sim['decay_rates'],
            grthreshold_values=sim['threshold_values'],
        )

    def _apply_scenario(self, sim: dict, scenario: SimulationScenario) -> dict:
        """Copy of `sim` with the scenario's input and decay-rate overrides."""
        out_index = {name: i for i, name in enumerate(self.out_of_network_arguments)}
        node_index = {name: i for i, name in enumerate(self.array_of_elements)}

        out_params = list(sim['out_params'])
        for name, value in scenario.out_overrides.items():
            out_params[out_index[name]] = value

        decay_rates = ([self.DEFAULT_DECAY_RATE] * self.n if scenario.reset_decay_rates
                       else list(sim['decay_rates']))
        for name, value in scenario.decay_overrides.items():
            decay_rates[node_index[name]] = value

        return {**sim, 'out_params': out_params,
                'decay_rates': np.asarray(decay_rates, dtype=float)}

    def _parse_row_letters(self, letters_str: str, n_rows: int) -> list:
        """'A,B,C,D' -> ['A', 'B', 'C', 'D']; missing letters fall back to A, B, C..."""
        letters = [s.strip() for s in (letters_str or '').split(',')]
        defaults = [s.strip() for s in self.DEFAULT_ROW_LETTERS.split(',')]
        defaults += list(string.ascii_uppercase[len(defaults):])
        return [letters[i] if i < len(letters) and letters[i] else defaults[i]
                for i in range(n_rows)]

    def _make_scenario_grid_callback(self, key: str, scenarios: list):
        def callback(letters_str, *values):
            sim = self._unpack_simulation_values(values)
            rows = []
            for scenario in scenarios:
                scenario_sim = self._apply_scenario(sim, scenario)
                rows.append((self._solve_simulation(scenario_sim),
                             scenario_sim['out_params']))
            fig = self.plotter.design_scenario_grid(
                rows, self.SCENARIO_GRID_SUBNETWORKS,
                self._parse_row_letters(letters_str, len(scenarios)))
            self.last_scenario_figs[key] = fig
            return fig
        return callback

    def after_push_button(self, *values):
        sim = self._unpack_simulation_values(values)
        out_params = sim['out_params']
        selection_box_plot = sim['selection']

        Resultados = self._solve_simulation(sim)
        self.last_results = Resultados
        Fig1 = self.plotter.plot_results(Resultados, selection_box_plot)
        Fig2 = self.plotter.design_and_personalize_subplots(
            Resultados, out_params, start_idx=0, end_idx=10, with_bounds=False)
        Fig3 = self.plotter.design_and_personalize_subplots(
            Resultados, out_params, start_idx=10, end_idx=20)
        self.last_fig2 = Fig2
        self.last_fig3 = Fig3
        return Fig1, Fig2, Fig3

    def after_push_button_comparison(self, *values):
        sim = self._unpack_simulation_values(values)
        out_params = sim['out_params']
        selection_box_plot = sim['selection']

        det_results = self._solve_simulation(sim)

        stoch_results = self._run_stochastic(
            sim['nodes'], out_params, sim['decay_rates'], sim['threshold_values'],
            sim['activation'], seed=42, noise_sigmas=[0.1] * self.n, N=100,
        )

        self.last_results = det_results
        self.last_stoch_results = stoch_results

        Fig1 = self.plotter.plot_results(det_results, selection_box_plot)
        Fig2 = self.plotter.design_and_personalize_comparison(
            det_results, stoch_results, out_params, 0, 10, with_bounds=True)
        Fig3 = self.plotter.design_and_personalize_comparison(
            det_results, stoch_results, out_params, 10, 20, with_bounds=True)

        self.last_fig2 = Fig2
        self.last_fig3 = Fig3
        return Fig1, Fig2, Fig3

    def _run_stochastic(self, nodes, out_params, decay_rates, thresholds,
                        activation, seed, noise_sigmas, N=100):
        t_eval = np.linspace(self.DEFAULT_TIME_SPAN[0],
                             self.DEFAULT_TIME_SPAN[1], 500)
        Resultados = {
            'time': t_eval,
            'nodes': {},
            'stochastic': True,
        }
        for node in self.array_of_elements:
            Resultados['nodes'][node] = {
                'values': np.zeros(len(t_eval)),
                'initial': 0.0, 'final': 0.0,
                'max': 0.0, 'min': 0.0, 'mean': 0.0,
            }
        for j in range(N):
            seed += 1
            simulation = self.solve_ito(
                t_span=self.DEFAULT_TIME_SPAN,
                grinitial_conditions=nodes,
                grout_of_the_network_arguments_eval=out_params,
                af=activation,
                decay_rates_list=decay_rates,
                grthreshold_values=thresholds,
                noise_sigmas=noise_sigmas,
                seed=seed,
            )
            for node in self.array_of_elements:
                for key in simulation['nodes'][node]:
                    Resultados['nodes'][node][key] += simulation['nodes'][node][key] / N
        return Resultados

    def after_push_ito(self, *values):
        n, m = self.n, len(self.out_of_network_arguments)
        nodes        = values[:n]
        out_params   = values[n:n + m]
        decay_rates  = values[n + m:n + m + n]
        thresholds   = values[n + m + n:n + m + 2 * n]
        activation   = values[n + m + 2 * n]
        selection    = values[n + m + 2 * n + 1]
        seed         = int(values[n + m + 2 * n + 2])
        noise_sigmas = list(values[n + m + 2 * n + 3:n + m + 2 * n + 3 + n])

        Resultados = self._run_stochastic(
            nodes, out_params, decay_rates, thresholds,
            activation, seed, noise_sigmas, N=100)

        self.last_results = Resultados
        Fig1 = self.plotter.plot_results(Resultados, selection)
        Fig2 = self.plotter.design_and_personalize_subplots(Resultados, out_params, 0, 10)
        Fig3 = self.plotter.design_and_personalize_subplots(Resultados, out_params, 10, 20)
        self.last_fig2 = Fig2
        self.last_fig3 = Fig3
        return Fig1, Fig2, Fig3

    def run_bandwidth_analysis(self, noise_str, target_cov, n_sim):
        try:
            noise_levels = [float(x.strip()) for x in noise_str.split(",") if x.strip()]
        except ValueError:
            return None, None, "Error: revisa el formato de los niveles de ruido."
        if not noise_levels:
            return None, None, "Error: no hay niveles de ruido."

        try:
            results = self.analyze_noise_bandwidth(
                noise_levels=noise_levels,
                target_coverage=float(target_cov),
                N=int(n_sim),
            )
        except Exception as e:
            return None, None, f" Error durante el análisis: {e}"

        rows = []
        for r in results:
            for node in r['efficiency']:
                rows.append({
                    "noise": r['noise'],
                    "node": node,
                    "p_min": r['p_min'].get(node),
                    "efficiency": r['efficiency'].get(node),
                    "coverage": r['coverage'].get(node),
                })
        import pandas as pd
        df = pd.DataFrame(rows)

        fig = self.plotter.plot_bandwidth_vs_noise(results)

        info = (f"Análisis completado para {len(noise_levels)} niveles "
                f"de ruido, N = {int(n_sim)}, target = {target_cov}.")
        return fig, df, info

    def set_gaussian_conditions(self, val: str):
        if val == '0':
            return np.absolute(np.random.normal(self.GAUSSIAN_MEAN_0, self.GAUSSIAN_STD, self.n)).tolist()
        elif val == '0.5':
            return np.absolute(np.random.normal(self.GAUSSIAN_MEAN_05, self.GAUSSIAN_STD, self.n)).tolist()
        elif val == '1':
            return np.absolute(np.random.normal(self.GAUSSIAN_MEAN_1, self.GAUSSIAN_STD, self.n)).tolist()

    def update_general_plot(self, selected_nodes: list) -> go.Figure:
        if not hasattr(self, 'last_results') or self.last_results is None:
            return go.Figure().update_layout(title="Please run a simulation first to plot selected nodes.")
        return self.plotter.plot_results(self.last_results, selected_nodes)

    def set_distribution(self, option):
        if option == 'normal':
            return np.absolute(np.random.normal(self.GAUSSIAN_MEAN_05, self.GAUSSIAN_STD, self.n)).tolist()
        elif option == 'uniform':
            return np.absolute(np.random.uniform(self.GAUSSIAN_MEAN_05, self.GAUSSIAN_STD, self.n)).tolist()
        elif option == 'laplace':
            return np.absolute(np.random.laplace(self.GAUSSIAN_MEAN_05, self.GAUSSIAN_STD, self.n)).tolist()

    def restart_sliders_value(self, *values):
        return self.value_sliders.tolist()

    def reset_network_parameters(self, *values):
        return self.value_sliders_out.tolist()

    def reset_threshold_values(self, *values):
        return np.full(self.n, self.DEFAULT_THRESHOLD_VALUE).tolist()

    def set_decay_rates(self, *values):
        return np.ones(self.n).tolist()

    def analyze_noise_bandwidth(self, noise_levels=None, nodes=None,
                                target_coverage=0.95, candidates=None,
                                N=100, base_seed=42):
        if noise_levels is None:
            noise_levels = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3]
        if nodes is None:
            nodes = ['TBET', 'GATA3', 'RORGT', 'FOXP3', 'BCL6']

        init_nodes = self.value_sliders.tolist()
        out_params = list(self.value_sliders_out)
        decay_rates = [self.DEFAULT_DECAY_RATE] * self.n
        thresholds = [self.DEFAULT_THRESHOLD_VALUE] * self.n
        activation = self.DEFAULT_ACTIVATION
        method = self.DEFAULT_SOLVER

        t_eval = np.linspace(self.DEFAULT_TIME_SPAN[0],
                             self.DEFAULT_TIME_SPAN[1], 500)

        det_results = self.solve(
            t_span=self.DEFAULT_TIME_SPAN,
            t_eval=t_eval,
            grinitial_conditions=init_nodes,
            grmethod=method,
            grout_of_the_network_arguments_eval=out_params,
            af=activation,
            decay_rates_list=decay_rates,
            grthreshold_values=thresholds,
        )

        results = []
        for i, noise in enumerate(noise_levels):
            stoch_results = self._run_stochastic(
                init_nodes, out_params, decay_rates, thresholds,
                activation,
                seed=base_seed + i,
                noise_sigmas=[noise] * self.n,
                N=N)

            p_min_per_node, p_ratio = self.find_band_percentage(
                det_results, stoch_results,
                nodes=nodes,
                target_coverage=target_coverage,
                candidates=candidates,
            )

            efficiencies = {node: float(1 - p) for node, p in p_min_per_node.items()}
            coverages = {node: p_ratio[node][p]
                         for node, p in p_min_per_node.items()}

            results.append({
                'noise': float(noise),
                'p_min': {n: float(p) for n, p in p_min_per_node.items()},
                'efficiency': efficiencies,
                'coverage': coverages,
            })

        return results

    def find_band_percentage(self, deterministic_results, stochastic_results,
                             nodes=None, target_coverage=0.95, candidates=None):
        if nodes is None:
            nodes = ['TBET', 'GATA3', 'RORGT', 'FOXP3', 'BCL6']
        if candidates is None:
            candidates = [i / 10 for i in range(1, 11)]

        p_min_per_node = {}
        p_ratio = {}

        for node in nodes:
            if node not in deterministic_results['nodes']:
                continue
            if node not in stochastic_results['nodes']:
                continue

            det_curve = np.asarray(deterministic_results['nodes'][node]['values'])
            stoch_curve = np.asarray(stochastic_results['nodes'][node]['values'])

            if det_curve.shape[0] != stoch_curve.shape[0]:
                raise ValueError(
                    f"Longitudes incompatibles en {node}: "
                    f"det={det_curve.shape[0]}, stoch={stoch_curve.shape[0]}"
                )

            p_ratio[node] = {}
            for p in candidates:
                lower = det_curve * (1 - p)
                upper = det_curve * (1 + p)
                inside = (stoch_curve >= lower) & (stoch_curve <= upper)
                p_ratio[node][p] = float(inside.mean())

            p_best = None
            for p in candidates:
                if p_ratio[node][p] >= target_coverage:
                    p_best = p
                    break
            if p_best is None:
                p_best = candidates[-1]

            p_min_per_node[node] = p_best

        return p_min_per_node, p_ratio
    def analyze_final_vs_max(self, noise_levels=None, nodes=None,
                         threshold=0.2, N=100, base_seed=42,
                         t_eval=None):
        if noise_levels is None:
            noise_levels = [0.05, 0.06, 0.1, 0.2, 0.3, 0.4]
        if nodes is None:
            nodes = ['TBET', 'GATA3', 'RORGT', 'FOXP3', 'BCL6']
        if t_eval is None:
            t_eval = np.linspace(self.DEFAULT_TIME_SPAN[0],
                                 self.DEFAULT_TIME_SPAN[1], 500)
    
        init_nodes = self.value_sliders.tolist()
        out_params = list(self.value_sliders_out)
        decay_rates = [self.DEFAULT_DECAY_RATE] * self.n
        thresholds = [self.DEFAULT_THRESHOLD_VALUE] * self.n
        activation = self.DEFAULT_ACTIVATION
    
        results = []
    
        for i, noise in enumerate(noise_levels):
            final_counts = {node: 0 for node in nodes}
            max_counts = {node: 0 for node in nodes}
    
            for j in range(N):
                simulation = self.solve_ito(
                    t_span=self.DEFAULT_TIME_SPAN,
                    t_eval=t_eval,
                    grinitial_conditions=init_nodes,
                    grout_of_the_network_arguments_eval=out_params,
                    af=activation,
                    decay_rates_list=decay_rates,
                    grthreshold_values=thresholds,
                    noise_sigmas=[noise] * self.n,
                    seed=base_seed + i * 1000 + j,
                )
    
                for node in nodes:
                    values = np.asarray(simulation['nodes'][node]['values'])
                    if values[-1] > threshold:
                        final_counts[node] += 1
                    if float(np.max(values)) > threshold:
                        max_counts[node] += 1
    
            final_fractions = {}
            max_fractions = {}
            final_stds = {}
            max_stds = {}
            differences = {}
    
            for node in nodes:
                k_f = final_counts[node]
                k_m = max_counts[node]
    
                f_f = k_f / N
                f_m = k_m / N
    
                final_fractions[node] = float(f_f)
                max_fractions[node] = float(f_m)
                final_stds[node] = float(np.sqrt(max(f_f * (1 - f_f), 0.0) / N))
                max_stds[node] = float(np.sqrt(max(f_m * (1 - f_m), 0.0) / N))
                differences[node] = float(f_m - f_f)
    
            results.append({
                'noise': float(noise),
                'final_fractions': final_fractions,
                'max_fractions': max_fractions,
                'final_stds': final_stds,
                'max_stds': max_stds,
                'final_counts': {node: int(c) for node, c in final_counts.items()},
                'max_counts': {node: int(c) for node, c in max_counts.items()},
                'differences': differences,
            })
    
        return results
import argparse

def main():
    parser = argparse.ArgumentParser(description="Macrophage–T cell network simulator")
    parser.add_argument("--share", action="store_true",
                        help="Crea un enlace público temporal (necesario en Colab)")
    args = parser.parse_args()

    app = NetworkInterface(SUBNETWORKS)
    demo = app.gradio_interface()      # el objeto gr.Blocks que hoy se lanza
    demo.launch(share=args.share)

if __name__ == "__main__":
    main()
