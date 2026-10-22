import colorsys
import string
import numpy as np
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from math import ceil
from dataclasses import dataclass
from typing import List, Sequence, Tuple
PHENOTYPE_LABELS = {
    'TBET':  'Th1',
    'GATA3': 'Th2',
    'RORGT': 'Th17',
    'FOXP3': 'Treg',
    'BCL6':  'Tfh',
}

PHENOTYPE_COLORS = {
    'TBET':  'green',
    'GATA3': 'purple',
    'RORGT': 'blue',
    'FOXP3': 'red',
    'BCL6':  'orange',
}

DEFAULT_STYLE = {
    # --- Plotly (plot_results) ---
    "plot_y_range": (-0.1, 1.1),
    "plot_height": 500,
    "title_x_position": 0.5,
    "title_y_position": 0.95,
    "max_nodes_to_plot": 16,
    "default_plot_options": ['OXPHOS', 'IL10', 'TGFB', 'RORGT', 'IL21', 'IL17', 'BCL6'],

    # --- Subplots matplotlib ---
    "subplot_columns": 5,
    "subplot_width": 21,
    "subplot_height_mult": 6.5,
    "subplot_tight_pad": 0.5,

    # --- Leyenda ---
    "legend_bbox_x": 0.0,
    "legend_bbox_y": 1.5,
    "legend_bbox_width": 1.0,
    "legend_bbox_height": 0.01,
    "legend_font_size": 16,
    "legend_title_font_size": 16,

    # --- Letras de panel (scenario grids) ---
    # Coordenadas en unidades del eje (transAxes) del primer subplot de cada fila.
    "panel_letter_x": -0.22,
    "panel_letter_y": 1.5,
    "panel_letter_font_size": 30,

    # --- Barras ---
    "bar_alpha": 0.85,
    "bar_color": "blue",
    "bar_y_limit": (-0.01, 1.1),

    # --- Cotas ---
    "bounding_percentage": 0.20,
    "bounding_width": 1.5,
    "bounding_alpha": 0.5,
    "bounding_fill_alpha": 0.20,

    # --- Comparación determinista/estocástica ---
    "stochastic_line_width": 1.5,
    "stochastic_line_alpha": 0.7,

    # --- Bandwidth plot ---
    "bandwidth_line_width": 2.5,
    "bandwidth_marker_size": 8,
    "bandwidth_line_color": "steelblue",

    # --- Phase space 3-D (Plotly) ---
    "phase_space_colorscale": "Viridis",
    "phase_space_3d_height": 750,

    # --- Sensitivity analysis ---
    "sensitivity_panel_size": (9.0, 7.0),
    "sensitivity_font_size": 16,
    "sensitivity_legend_font_size": 13,
}

# Node / input name -> LaTeX, used in the sensitivity figures.
LATEX_NAMES = {
    'IFNG': r'\mathrm{IFN}\gamma', 'IFNGE': r'\mathrm{IFN}\gamma',
    'IFNGL': r'\mathrm{IFN}\gamma_{L}', 'IFNA': r'\mathrm{IFN}\alpha',
    'TGFB': r'\mathrm{TGF}\beta', 'TGFBE': r'\mathrm{TGF}\beta_{e}',
    'IL4': r'\mathrm{IL4}', 'IL4E': r'\mathrm{IL4}_{e}', 'IL4L': r'\mathrm{IL4}_{L}',
    'IL6E': r'\mathrm{IL6}_{e}', 'IL10': r'\mathrm{IL10}',
    'IL10E': r'\mathrm{IL10}_{e}', 'IL10L': r'\mathrm{IL10}_{L}',
    'IL12': r'\mathrm{IL12}', 'IL12E': r'\mathrm{IL12}_{e}',
    'IL18E': r'\mathrm{IL18}_{e}', 'IL21': r'\mathrm{IL21}',
    'IL21E': r'\mathrm{IL21}_{e}', 'IL33': r'\mathrm{IL33}',
    'IL33E': r'\mathrm{IL33}_{e}', 'TNFA': r'\mathrm{TNF}\alpha',
    'TNFAOUT': r'\mathrm{TNF}\alpha', 'IL1B': r'\mathrm{IL1}\beta',
    'IL1BE': r'\mathrm{IL1}\beta_{e}', 'IL1BOUT': r'\mathrm{IL1}\beta',
    'O2': r'\mathrm{O_2}', 'MO2': r'\mathrm{O_2}^{M}', 'SSRNA': r'\mathrm{ssRNA}',
    'DSRNA': r'\mathrm{dsRNA}', 'LPSE': r'\mathrm{LPS}', 'b': r'\beta',
}


def param_to_latex(name: str) -> str:
    if name in LATEX_NAMES:
        return LATEX_NAMES[name]
    s = str(name).replace('γ', r'\gamma').replace('β', r'\beta').replace('α', r'\alpha')
    return r'\mathrm{' + s.replace('_', r'\_') + '}'


def elasticity_label(node_tex: str, param_tex: str, mode: str) -> str:
    derivative = f'(\\partial {node_tex} / \\partial {param_tex})'
    if mode == 'relative_pointwise':
        return f'${derivative} \\cdot {param_tex} / |{node_tex}(t)|$'
    if mode == 'relative_final':
        return f'${derivative} \\cdot {param_tex} / |{node_tex}_f|$'
    if mode == 'scaled':
        return f'${derivative} \\cdot {param_tex}$'
    return f'$\\partial {node_tex} / \\partial {param_tex}$'


def auto_palette(base_color, n, palette=None):
    """n distinct colours. The first one is `base_color`; the rest rotate its hue.

    palette: None -> anchored on base_color; str -> matplotlib colormap name;
             list -> explicit colours (cycled).
    """
    if n <= 0:
        return []
    if palette is not None:
        if isinstance(palette, (list, tuple)):
            cols = [mcolors.to_rgb(c) for c in palette]
            return [cols[k % len(cols)] for k in range(n)]
        cmap = mpl.colormaps[palette]
        if isinstance(cmap, mcolors.ListedColormap) and cmap.N <= 20:
            return [mcolors.to_rgb(cmap(k % cmap.N)) for k in range(n)]
        return [mcolors.to_rgb(c) for c in cmap(np.linspace(0.0, 0.9, n))]

    h, l, s = colorsys.rgb_to_hls(*mcolors.to_rgb(base_color))
    s_alt = s if s >= 0.25 else 0.65          # greys have no usable hue
    l_alt = float(np.clip(l, 0.35, 0.55))
    colors = [mcolors.to_rgb(base_color)]
    for k in range(1, n):
        colors.append(colorsys.hls_to_rgb((h + k / n) % 1.0, l_alt, s_alt))
    return colors

@dataclass
class PlotConfig:
    variables: List[str]
    labels: List[str]
    colors: List[str]
    styles: List[str]
    legfontsize: float
    legncol: int
    title: str
class Plotter:
    def __init__(self, features: dict, style: dict = None):
        self.features = features
        self.style = {**DEFAULT_STYLE, **(style or {})}

    # ------------------------------------------------------------
    # Barras
    # ------------------------------------------------------------
    def create_bar_chart(self, axi, labels, labels_mute, values, title,
                         bbox_position, ncol=1, y_limit=None):
        axi.set_xticks(range(len(labels)))
        axi.set_xticklabels(labels_mute if labels_mute else labels,
                            rotation=90, ha='center', fontsize=16)
        axi.tick_params(axis='both', labelsize=17)
        axi.bar(labels, values, color=self.style["bar_color"],
                alpha=self.style["bar_alpha"], label='_nolegend_')
        axi.legend(bbox_to_anchor=bbox_position, loc=1, ncol=ncol,
                   mode="expand", borderaxespad=0., title=title,
                   title_fontsize='xx-large')
        if y_limit:
            axi.set_ylim(*y_limit)
        return axi

    def create_macrophage_inputs_chart(self, axi, bars_reference, subr,
                                       bbox_position):
        labels = ['TGFBE', 'IFNGE', 'IL10E', 'IL4E', 'IL1BE', 'TNFA']
        labels_mute = ['TGFB', 'IFNG', 'IL10', 'IL4E', 'IL1B', 'TNFA']
        values = [bars_reference[label] for label in labels]
        return self.create_bar_chart(axi, labels, labels_mute, values, subr,
                                     bbox_position, ncol=1,
                                     y_limit=self.style["bar_y_limit"])

    def create_lymphocyte_inputs_chart(self, axi, bars_reference, subr,
                                       bbox_position):
        reference = ['IL12E', 'IL18E', 'IL33E', 'IL6E', 'IL10L', 'IL4L', 'IL21E']
        labels = ['TCR', 'IL12L', 'IL18L', 'IL33L', 'IL6L', 'IL10L', 'IL4L', 'IL21L']
        TCR_value = 1 if bars_reference.get('MHC2E') == 1 and bars_reference.get('CD8086E') else 0
        values = [bars_reference[ref] for ref in reference]
        values = [TCR_value] + values
        return self.create_bar_chart(axi, labels, None, values, subr,
                                     bbox_position, ncol=1,
                                     y_limit=self.style["bar_y_limit"])

    def create_lymphocyte_phenotypes_chart(self, axi, results, subr):
        config = self.features[subr]
        labels = config.labels
        axi.set_xticks(range(len(labels)))
        axi.set_xticklabels(labels, rotation=90, ha='right',
                            fontsize=self.style["legend_font_size"])
        phenotype_population = [results['nodes'][k]['max'] for k in config.variables
                                if k in results['nodes']]
        axi.bar(labels, phenotype_population, color=self.style["bar_color"])
        axi.legend(bbox_to_anchor=(self.style["legend_bbox_x"], 1.3,
                                   self.style["legend_bbox_width"], 0.09),
                   loc=1, ncol=1, mode="expand", borderaxespad=0.,
                   prop={'size': self.style["legend_font_size"]},
                   title=config.title, title_fontsize='xx-large')
        axi.tick_params(axis='both', labelsize=self.style["legend_font_size"])
        axi.set_ylim(*self.style["bar_y_limit"])
        return axi

    # ------------------------------------------------------------
    # Helpers de cotas
    # ------------------------------------------------------------
    def compute_bounds(self, values, percentage=None):
        if percentage is None:
            percentage = self.style["bounding_percentage"]
        values = np.asarray(values)
        upper = values * (1 + percentage)
        lower = values * (1 - percentage)
        return lower, upper

    def draw_bounds(self, axi, x, lower, upper, color):
        axi.plot(x, upper, color=color, linestyle=':',
                 linewidth=self.style["bounding_width"],
                 alpha=self.style["bounding_alpha"], label='_nolegend_')
        axi.plot(x, lower, color=color, linestyle=':',
                 linewidth=self.style["bounding_width"],
                 alpha=self.style["bounding_alpha"], label='_nolegend_')

    def _draw_bounds_block(self, axi, x, values, color, fill: bool):
        lower, upper = self.compute_bounds(values)
        if fill:
            axi.fill_between(x, lower, upper, color=color,
                             alpha=self.style["bounding_fill_alpha"],
                             label='_nolegend_')
        else:
            self.draw_bounds(axi, x, lower, upper, color=color)

    # ------------------------------------------------------------
    # Helpers de estilo de ejes
    # ------------------------------------------------------------
    def _style_axis_legend(self, axi, config, bbox_position, ncol):
        axi.legend(bbox_to_anchor=bbox_position, loc=1, ncol=ncol,
                   mode="expand", borderaxespad=0.,
                   prop={'size': self.style["legend_font_size"]},
                   title=config.title, title_fontsize='xx-large')
        axi.tick_params(axis='both', labelsize=17)
        axi.set_xlabel('Time', fontsize=17)
        axi.set_ylabel('Expression', fontsize=17)

    def add_panel_letter(self, axi, letter: str):
        """Bold panel letter (A, B, ...) outside the top-left corner of `axi`."""
        if not letter:
            return axi
        axi.text(self.style["panel_letter_x"], self.style["panel_letter_y"], letter,
                 transform=axi.transAxes,
                 fontsize=self.style["panel_letter_font_size"],
                 fontweight='bold', ha='right', va='top')
        return axi

    # ------------------------------------------------------------
    # Líneas determinista / comparación
    # ------------------------------------------------------------
    def create_line_plot(self, axi, results, subr, bbox_position, ncol,
                         with_bounds: bool = False):
        config = self.features[subr]
        for node, col, sty, label in zip(config.variables, config.colors,
                                         config.styles, config.labels):
            values = results['nodes'][node]['values']

            axi.plot(results['time'], values, label=label,
                     color=col, linestyle=sty, linewidth=3)

            if with_bounds:
                self._draw_bounds_block(axi, results['time'], values, col,
                                        fill=False)

            self._style_axis_legend(axi, config, bbox_position, ncol)

        axi.set_ylim(*self.style["plot_y_range"])
        axi.grid(False)
        return axi

    def create_comparison_line_plot(self, axi, results, subr, bbox_position, ncol,
                                    stochastic_results=None,
                                    with_bounds: bool = True):
        config = self.features[subr]
        for node, col, sty, label in zip(config.variables, config.colors,
                                         config.styles, config.labels):
            values = results['nodes'][node]['values']

            axi.plot(results['time'], values, label=label,
                     color=col, linestyle=sty, linewidth=3)

            if with_bounds:
                self._draw_bounds_block(axi, results['time'], values, col,
                                        fill=True)

            if stochastic_results is not None and node in stochastic_results['nodes']:
                axi.plot(stochastic_results['time'],
                         stochastic_results['nodes'][node]['values'],
                         color=col, linestyle='--',
                         linewidth=self.style["stochastic_line_width"],
                         alpha=self.style["stochastic_line_alpha"],
                         label='_nolegend_')

            self._style_axis_legend(axi, config, bbox_position, ncol)

        axi.set_ylim(*self.style["plot_y_range"])
        axi.grid(False)
        return axi

    # ------------------------------------------------------------
    # Composición de subplots
    # ------------------------------------------------------------
    def _legend_bbox(self) -> tuple:
        return (self.style["legend_bbox_x"], self.style["legend_bbox_y"],
                self.style["legend_bbox_width"], self.style["legend_bbox_height"])

    def _create_figure_grid(self, n_rows, n_cols):
        """2-D array of axes; width scales with n_cols, height with n_rows."""
        width = self.style["subplot_width"] * n_cols / self.style["subplot_columns"]
        return plt.subplots(
            n_rows, n_cols,
            figsize=(width, self.style["subplot_height_mult"] * n_rows),
            layout="constrained",
            squeeze=False,
        )

    def _setup_subplots_grid(self, inputs, start_idx, end_idx):
        sub_names = list(self.features.keys())[start_idx:end_idx]
        n_rows = ceil(len(sub_names) / self.style["subplot_columns"])
        fig2, axis = self._create_figure_grid(n_rows, self.style["subplot_columns"])
        axis = axis.flatten()
        positions_bbox_to_anchor = {subr: self._legend_bbox() for subr in self.features}
        bars_reference = dict(zip(self.style["out_args"], inputs))
        return fig2, axis, sub_names, positions_bbox_to_anchor, bars_reference

    def _hide_empty_axes(self, axis, n_drawn):
        for j in range(n_drawn, len(axis)):
            axis[j].set_visible(False)

    def _finalize_figure(self, fig2):
        plt.show()
        fig2.get_layout_engine().set(h_pad=self.style["subplot_tight_pad"])
        return fig2

    def _draw_branch_for_subnetwork(self, axi, subr, config,
                                    results, bars_reference, bbox_position,
                                    line_fn, **line_fn_kwargs):
        if subr == 'Macrophage inputs':
            self.create_macrophage_inputs_chart(
                axi, bars_reference, config.title, bbox_position)
        elif subr == 'Lymphocyte inputs':
            self.create_lymphocyte_inputs_chart(
                axi, bars_reference, config.title, bbox_position)
        elif subr == 'Lymphocyte phenotype':
            self.create_lymphocyte_phenotypes_chart(axi, results, subr)
        else:
            line_fn(axi, results, subr, bbox_position, config.legncol,
                    **line_fn_kwargs)

    def _design_subplots(self, results, inputs, start_idx, end_idx,
                         line_fn, with_bounds: bool = False,
                         stochastic_results=None):
        fig2, axis, sub_names, positions_bbox_to_anchor, bars_reference = \
            self._setup_subplots_grid(inputs, start_idx, end_idx)

        for axi, subr in zip(axis, sub_names):
            config = self.features[subr]
            kwargs = {"with_bounds": with_bounds}
            if stochastic_results is not None:
                kwargs["stochastic_results"] = stochastic_results
            self._draw_branch_for_subnetwork(
                axi, subr, config, results, bars_reference,
                positions_bbox_to_anchor[subr],
                line_fn=line_fn,
                **kwargs,
            )

        self._hide_empty_axes(axis, len(sub_names))
        return self._finalize_figure(fig2)

    def design_and_personalize_subplots(self, results, inputs, start_idx, end_idx,
                                        with_bounds: bool = False):
        return self._design_subplots(
            results, inputs, start_idx, end_idx,
            line_fn=self.create_line_plot,
            with_bounds=with_bounds,
        )

    def design_and_personalize_comparison(self, deterministic_results,
                                          stochastic_results, inputs,
                                          start_idx, end_idx,
                                          with_bounds: bool = True):
        return self._design_subplots(
            deterministic_results, inputs, start_idx, end_idx,
            line_fn=self.create_comparison_line_plot,
            with_bounds=with_bounds,
            stochastic_results=stochastic_results,
        )

    def design_scenario_grid(self, rows: Sequence[Tuple[dict, Sequence[float]]],
                             subnetworks: Sequence[str],
                             row_letters: Sequence[str],
                             with_bounds: bool = False):
        """One row per simulation, one column per subnetwork.

        rows        -> [(results, inputs), ...]; `inputs` are the out-of-network
                       arguments used in that simulation (needed by bar charts).
        subnetworks -> keys of `self.features` drawn in every row.
        row_letters -> bold letter placed at the top-left of each row.
        """
        fig, axis = self._create_figure_grid(len(rows), len(subnetworks))
        bbox_position = self._legend_bbox()

        for row_axes, (results, inputs), letter in zip(axis, rows, row_letters):
            bars_reference = dict(zip(self.style["out_args"], inputs))
            for axi, subr in zip(row_axes, subnetworks):
                self._draw_branch_for_subnetwork(
                    axi, subr, self.features[subr], results, bars_reference,
                    bbox_position,
                    line_fn=self.create_line_plot,
                    with_bounds=with_bounds,
                )
            self.add_panel_letter(row_axes[0], letter)

        return self._finalize_figure(fig)

    # ------------------------------------------------------------
    # Plotly
    # ------------------------------------------------------------
    def plot_results(self, results: dict = None, options: list = None) -> go.Figure:
        if options is None:
            options = self.style["default_plot_options"]

        if len(options) > self.style["max_nodes_to_plot"]:
            return go.Figure().update_layout(
                title="Error: Change one of the nodes to add it to your graphic")

        fig = go.Figure()
        t = results['time']

        for option in options:
            fig.add_trace(go.Scatter(
                x=t, y=results['nodes'][option]['values'],
                mode='lines', name=option, line=dict(width=2),
                hovertemplate=f'{option}<br>Time: %{{x:.2f}}<br>Value: %{{y:.4f}}'
            ))

        title_config = {
            'text': "Nodes expression",
            'x': self.style["title_x_position"],
            'y': self.style["title_y_position"],
            'xanchor': 'center',
            'yanchor': 'top'
        }

        fig.update_layout(
            title=title_config,
            yaxis=dict(range=self.style["plot_y_range"], autorange=False),
            xaxis_title='Time',
            yaxis_title='Expression',
            hovermode='x unified',
            template='plotly_white',
            height=self.style["plot_height"]
        )

        return fig

    # ------------------------------------------------------------
    # Phase space
    # ------------------------------------------------------------
    def plot_phase_space(self, selected_transcription_factors,
                         solutions_for_sp, param_values=None,
                         colorbar_label='Cytokines expression level') -> go.Figure:
        """Interactive 3-D phase space (drag to rotate, scroll to zoom).

        Each trajectory is coloured by its input level; its shadow is drawn
        on the floor of the box for depth, and the end points (steady states)
        are joined by a dashed line that traces the branch of attractors.
        """
        if param_values is None:
            param_values = list(range(len(solutions_for_sp)))
        param_values = np.asarray(param_values, dtype=float)
        vmin, vmax = float(param_values.min()), float(param_values.max())
        span = (vmax - vmin) or 1.0
        colorscale = self.style["phase_space_colorscale"]
        x_name, y_name, z_name = selected_transcription_factors

        all_xyz = np.concatenate([np.asarray(sol, dtype=float)
                                  for sol in solutions_for_sp], axis=1)
        lo, hi = all_xyz.min(axis=1), all_xyz.max(axis=1)
        pad = np.maximum((hi - lo) * 0.05, 0.02)
        lo, hi = lo - pad, hi + pad
        z_floor = lo[2]

        fig = go.Figure()
        for sol, value in zip(solutions_for_sp, param_values):
            x, y, z = (np.asarray(v, dtype=float) for v in sol)
            color = sample_colorscale(colorscale, (value - vmin) / span)[0]
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=np.full_like(z, z_floor), mode='lines',
                line=dict(color='rgba(110,110,110,0.30)', width=2),
                hoverinfo='skip', showlegend=False))
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z, mode='lines',
                line=dict(color=color, width=6), showlegend=False,
                hovertemplate=(f'{colorbar_label}: {value:.3f}<br>'
                               f'{x_name}: %{{x:.3f}}<br>{y_name}: %{{y:.3f}}<br>'
                               f'{z_name}: %{{z:.3f}}<extra></extra>')))

        order = np.argsort(param_values)
        ends = np.array([[np.asarray(s[k])[-1] for k in range(3)]
                         for s in solutions_for_sp])[order]
        starts = np.array([[np.asarray(s[k])[0] for k in range(3)]
                           for s in solutions_for_sp])
        fig.add_trace(go.Scatter3d(
            x=ends[:, 0], y=ends[:, 1], z=ends[:, 2], mode='lines+markers',
            name='Steady states',
            line=dict(color='black', width=3, dash='dash'),
            marker=dict(size=5, color=param_values[order], colorscale=colorscale,
                        cmin=vmin, cmax=vmax,
                        line=dict(color='black', width=1),
                        colorbar=dict(title=dict(text=colorbar_label, side='right'),
                                      thickness=18, len=0.75)),
            customdata=param_values[order],
            hovertemplate=(f'Steady state<br>{colorbar_label}: %{{customdata:.3f}}<br>'
                           f'{x_name}: %{{x:.3f}}<br>{y_name}: %{{y:.3f}}<br>'
                           f'{z_name}: %{{z:.3f}}<extra></extra>')))
        fig.add_trace(go.Scatter3d(
            x=starts[:, 0], y=starts[:, 1], z=starts[:, 2], mode='markers',
            name='Initial condition',
            marker=dict(size=6, color='white', symbol='diamond',
                        line=dict(color='black', width=2)),
            hovertemplate='Initial condition<extra></extra>'))

        def axis(title, rng):
            return dict(title=dict(text=title, font=dict(size=15)),
                        range=list(rng), backgroundcolor='#f3f3f3',
                        gridcolor='white', zerolinecolor='#bbbbbb',
                        showbackground=True, tickfont=dict(size=11))

        fig.update_layout(
            template='plotly_white',
            height=self.style["phase_space_3d_height"],
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
            scene=dict(
                xaxis=axis(x_name, (lo[0], hi[0])),
                yaxis=axis(y_name, (lo[1], hi[1])),
                zaxis=axis(z_name, (lo[2], hi[2])),
                aspectmode='cube',
                camera=dict(eye=dict(x=1.55, y=-1.55, z=0.95)),
            ),
        )
        return fig

    def plot_phase_space_2d(self, selected_transcription_factors,
                            solutions_for_sp, param_values,
                            colorbar_label='Cytokines expression level'):
        """Bifurcation diagram: one trajectory per input level, coloured by it."""
        fig_2d, ax = plt.subplots(figsize=(12, 9))

        norm = plt.Normalize(vmin=min(param_values), vmax=max(param_values))
        cmap = plt.cm.viridis

        for i, sol in enumerate(solutions_for_sp):
            x_values = np.asarray(sol[0])
            y_values = np.asarray(sol[1])
            color = cmap(norm(param_values[i]))

            ax.plot(x_values, y_values,
                    color=color, alpha=0.85, linewidth=2.5, linestyle='-')

            dx = np.diff(x_values)
            dy = np.diff(y_values)
            magnitudes = np.sqrt(dx**2 + dy**2)
            magnitudes[magnitudes == 0] = 1e-10
            dx_norm = dx / magnitudes
            dy_norm = dy / magnitudes

            step = 4
            x_quiv = x_values[:-1:step]
            y_quiv = y_values[:-1:step]
            u_quiv = dx_norm[::step]
            v_quiv = dy_norm[::step]

            scale = 0.008
            ax.quiver(x_quiv, y_quiv, u_quiv * scale, v_quiv * scale,
                      color=color, alpha=0.8, width=0.004,
                      headwidth=4, headlength=4, headaxislength=3.5,
                      scale=1, zorder=10)
            ax.scatter(x_values[0], y_values[0],
                       color=color, s=80, edgecolor='black',
                       linewidth=1.0, zorder=5)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(colorbar_label, fontsize=14)
        cbar.ax.tick_params(labelsize=12)

        ax.set_xlabel(f'{selected_transcription_factors[0]}', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'{selected_transcription_factors[1]}', fontsize=14, fontweight='bold')
        ax.tick_params(axis='both', labelsize=12)

        ax.set_facecolor('#f5f5f5')
        ax.grid(True, alpha=0.3, linestyle='--', color='#888888')

        plt.tight_layout()
        return fig_2d

    # ------------------------------------------------------------
    # Local sensitivity analysis
    # ------------------------------------------------------------
    def node_color(self, node: str, fallback: str = 'steelblue') -> str:
        """First colour assigned to `node` in SUBNETWORKS."""
        for config in self.features.values():
            if node in config.variables:
                return config.colors[config.variables.index(node)]
        return fallback

    def _sensitivity_axes(self, n_rows, n_cols):
        w, h = self.style["sensitivity_panel_size"]
        return plt.subplots(n_rows, n_cols, figsize=(w * n_cols, h * n_rows),
                            squeeze=False, layout='constrained')

    def _sensitivity_legend(self, ax, ncol=2):
        ax.legend(loc='best', ncol=ncol, frameon=True, framealpha=0.9,
                  edgecolor='0.55', facecolor='white',
                  prop={'size': self.style["sensitivity_legend_font_size"]},
                  handlelength=2.0, columnspacing=1.4, labelspacing=0.35)

    def _sensitivity_style_axis(self, ax, letter, title, ylabel):
        fs = self.style["sensitivity_font_size"]
        ax.set_title(title, fontsize=fs + 2, pad=12)
        ax.set_title(f'{letter})', loc='left', fontsize=fs + 1,
                     fontweight='bold', pad=12)
        ax.set_xlabel('$t$', fontsize=fs)
        ax.set_ylabel(ylabel, fontsize=fs)
        ax.tick_params(axis='both', labelsize=fs - 2)
        ax.grid(False)

    def plot_sensitivity_grid(self, result, nodes, mode='relative_pointwise',
                              n_show=10, palette=None):
        """One row per node: activity, then one elasticity panel per parameter.

        With one parameter and two nodes this is the notebook's
        "activities + elasticity" figure (2 x 2).
        """
        from sensitivity import SensitivityAnalyzer

        params = result.param_names
        n_sweep = len(result.sweep_values)
        idx = np.linspace(0, n_sweep - 1, min(n_show, n_sweep)).astype(int)
        sweep_tex = (param_to_latex(result.sweep_param)
                     if result.sweep_param else None)

        def label(i):
            if sweep_tex is None:
                return 'Nominal inputs'
            return f'${sweep_tex} = {result.sweep_values[i]:.3f}$'

        fig, axes = self._sensitivity_axes(len(nodes), 1 + len(params))
        letters = iter(string.ascii_uppercase * 3)

        for r, node in enumerate(nodes):
            node_tex = param_to_latex(node)
            colors = auto_palette(self.node_color(node), len(idx), palette)
            q = result.q[:, result.node_index(node), :]

            ax = axes[r, 0]
            for k, i in enumerate(idx):
                ax.plot(result.t, q[i], color=colors[k], lw=1.8, label=label(i))
            ax.set_ylim(-0.05, 1.05)
            self._sensitivity_legend(ax)
            self._sensitivity_style_axis(ax, next(letters), f'${node_tex}$',
                                         f'${node_tex}$')

            for c, param in enumerate(params, start=1):
                p_tex = param_to_latex(param)
                e = SensitivityAnalyzer.elasticity(result, node, param, mode)
                ax = axes[r, c]
                for k, i in enumerate(idx):
                    ax.plot(result.t, e[i], color=colors[k], lw=1.8, label=label(i))
                ax.axhline(0, color='gray', ls=':', lw=0.8)
                self._sensitivity_legend(ax)
                title = 'Sensitivity' if mode == 'raw' else 'Elasticity coefficient'
                if len(params) > 1:
                    title += f' (${p_tex}$)'
                self._sensitivity_style_axis(ax, next(letters), title,
                                             elasticity_label(node_tex, p_tex, mode))
        return fig

    def plot_sensitivity_fd_check(self, checks):
        """Exact sensitivity vs central finite differences, one panel per check."""
        fig, axes = self._sensitivity_axes(1, len(checks))
        styles = ['--', '-.', ':', (0, (3, 1, 1, 1))]
        for c, (ax, check) in enumerate(zip(axes[0], checks)):
            node_tex = param_to_latex(check['node'])
            p_tex = param_to_latex(check['param'])
            ax.plot(check['t'], check['exact'], color='black', lw=2.0,
                    label='Exact (sensitivity ODE)')
            for m, (d, fd) in enumerate(check['fd'].items()):
                err = np.max(np.abs(fd - check['exact']))
                ax.plot(check['t'], fd, lw=1.5, ls=styles[m % len(styles)],
                        label=f'FD $\\Delta {p_tex} = {d:.0e}$  ({err:.1e})')
            ax.axhline(0, color='gray', ls=':', lw=0.8)
            self._sensitivity_legend(ax, ncol=1)
            self._sensitivity_style_axis(
                ax, string.ascii_uppercase[c], 'Comparison of sensitivity techniques',
                f'$\\partial {node_tex} / \\partial {p_tex}$')
        return fig

    # ------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------
    def save_matplotlib_as_jpg(self, fig: plt.Figure,
                               filename: str = "plot.jpg") -> str:
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        fig.savefig(temp_file.name, format='jpg', dpi=300, bbox_inches='tight')
        return temp_file.name

    # ------------------------------------------------------------
    # Bandwidth
    # ------------------------------------------------------------
    def plot_bandwidth_vs_noise(self, analysis_results, nodes=None,
                            node_label_map=None,
                            node_color_map=None,
                            shaded_range=(20, 30),
                            shaded_alpha=0.08,
                            shaded_color='gray',
                            figsize=(10, 6)):
        if node_label_map is None:
            node_label_map = PHENOTYPE_LABELS
        if node_color_map is None:
            node_color_map = PHENOTYPE_COLORS

        if not analysis_results:
            return plt.figure()

        if nodes is None:
            nodes = list(analysis_results[0]['efficiency'].keys())

        noises_pct = np.array([r['noise'] * 100 for r in analysis_results])

        fig, ax = plt.subplots(figsize=figsize)
        ax.axvspan(shaded_range[0], shaded_range[1],
                   color=shaded_color, alpha=shaded_alpha,
                   edgecolor='none', zorder=0)

        markers = ['o', 's', '^', 'D', 'v']

        for i, node in enumerate(nodes):
            label = node_label_map.get(node, node)
            color = node_color_map.get(node, 'black')
            marker = markers[i % len(markers)]
            effs = np.array([r['efficiency'].get(node, np.nan) * 100
                             for r in analysis_results])
            valid = ~np.isnan(effs)
            if not valid.any():
                continue
            ns = noises_pct[valid]
            es = effs[valid]

            ax.plot(ns, es,
                    marker=marker, markersize=8, linewidth=2.5,
                    color=color,
                    markeredgecolor='white', markeredgewidth=1.0,
                    label=label, zorder=3)

        ax.set_xlim(0, 30)
        ax.set_xticks(np.arange(0, 31, 5))
        ax.set_xlabel('Noise level (%)', fontsize=14, fontweight='bold')

        ax.set_ylim(0, 105)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_ylabel('Differentiation efficiency', fontsize=14, fontweight='bold')

        ax.grid(True, alpha=0.3, linestyle='--', color='#888888')
        ax.set_facecolor('#f9f9f9')
        ax.legend(loc='best', fontsize=12, title='Subset')

        plt.tight_layout()
        return fig

    def plot_fraction_vs_noise(self, analysis_results,
                               node_label_map=None,
                               node_color_map=None,
                               shaded_range=(20, 30),
                               figsize=(10, 6)):
        if node_label_map is None:
            node_label_map = PHENOTYPE_LABELS
        if node_color_map is None:
            node_color_map = PHENOTYPE_COLORS

        if not analysis_results:
            return plt.figure()

        noises_pct = np.array([r['noise'] * 100 for r in analysis_results])

        fig, ax = plt.subplots(figsize=figsize)

        ax.axvspan(shaded_range[0], shaded_range[1],
                   color='gray', alpha=0.20, zorder=0)

        for node, label in node_label_map.items():
            color = node_color_map.get(node, 'black')
            means = np.array([
                r['fractions'].get(node, np.nan) * 100
                for r in analysis_results
            ])
            stds = np.array([
                r['stds'].get(node, np.nan) * 100
                for r in analysis_results
            ])

            ax.errorbar(
                noises_pct, means, yerr=stds,
                marker='o', markersize=8, linewidth=2.5,
                color=color, ecolor=color,
                capsize=4, capthick=1.5, elinewidth=1.5,
                label=label, zorder=3,
            )

        ax.set_xlim(0, 30)
        ax.set_xticks(np.arange(0, 31, 5))
        ax.set_xlabel('Noise level (%)', fontsize=14, fontweight='bold')

        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_ylabel('Differentiation efficiency', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', color='#888888')
        ax.set_facecolor('#f9f9f9')
        ax.legend(loc='best', fontsize=12, title='Subset')

        plt.tight_layout()
        return fig
