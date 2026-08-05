import tempfile
from datetime import datetime

import faicons
from matplotlib import colormaps
from shiny import Inputs, Outputs, Session, module, reactive, render, ui

from shared import comps, styles, utils, views
from shared.consts import PLOT_DEFAULTS, Colorby, DiagramView


@module.ui
def ui_() -> ui.Tag:
    return ui.div(
        ui.row(
            ui.column(
                3,
                ui.card(
                    ui.card_header("Common plot options"),
                    ui.div(
                        comps.selection(
                            id="view",
                            label="Choose a diagram to view",
                            choices=list(DiagramView),
                            selected=DiagramView.FULL,
                            width="100%",
                        ),
                        comps.selection(
                            id="colorby",
                            label="Color by",
                            choices=[c for c in Colorby],
                            selected=PLOT_DEFAULTS["colorby"],
                            width="100%",
                        ),
                        comps.selection(
                            id="colormap",
                            label="Choose a colormap",
                            choices=sorted(list(colormaps)),
                            selected=PLOT_DEFAULTS["colormap"],
                            width="100%",
                        ),
                        ui.input_slider(
                            id="opacity",
                            label="Diagram opacity",
                            min=0.0,
                            max=1.0,
                            value=PLOT_DEFAULTS["opacity"],
                            ticks=True,
                            width="100%",
                        ),
                        ui.input_switch(
                            id="add_final_seed_positions",
                            label="Add final seed positions",
                            width="100%",
                        ),
                        comps.input_action_button(
                            id="reset_plot_options",
                            label="Reset common plot options to defaults",
                            icon="gear",
                        ),
                    ),
                ),
                ui.output_ui("extra_opts"),
            ),
            ui.column(
                9,
                ui.navset_card_underline(
                    ui.nav_spacer(),
                    ui.nav_panel("Microstructure", ui.output_ui("diagram_tab_card")),
                    ui.nav_panel("Metrics", ui.output_ui("metrics_tab_card")),
                ),
            ),
        ),
    )


@module.server
def server(
    input: Inputs,
    output: Outputs,
    session: Session,
    processed_data: reactive.Value,
) -> None:

    @reactive.calc
    def _diagram() -> utils.Diagram | Exception | None:
        d = processed_data()
        if not isinstance(d, utils.ProcessedData):
            return d

        viewer = utils.DiagramViewer(
            processed_data=d,
            colorby=input.colorby(),
            colormap=input.colormap(),
            opacity=input.opacity(),
            add_final_seed_positions=input.add_final_seed_positions(),
        )

        domain = d.config.domain

        match input.view():
            case DiagramView.FULL:
                return viewer.full()

            case DiagramView.THRESHED:
                min_, max_ = viewer._clim

                threshed_value = input.threshed_value()
                vmin = utils.unscale_from_minmax(
                    scaled=threshed_value[0], min_=min_, max_=max_
                )
                vmax = utils.unscale_from_minmax(
                    scaled=threshed_value[1], min_=min_, max_=max_
                )

                try:
                    return viewer.threshed(value=(vmin, vmax), invert=input.invert())
                except Exception as e:
                    return Exception(
                        "Error in parsing plot options, possibly no grain size matches the specified thresholds."
                        "More information: " + str(e),
                    )

            case DiagramView.SLICE:
                # when slice view is selected (in 3D mode) and user
                # changes mode to 2D (and then press generate); default to
                # full view and quit.
                if len(domain) == 2:
                    return Exception(
                        "Slicing is only available for 3D case. Please choose a different view."
                    )

                return viewer.slice_(
                    slice_normal=input.slice_normal(), slice_value=input.slice_value()
                )

            case DiagramView.CLIP:
                if len(domain) == 2 and input.clip_normal() == utils.COORDINATES[-1]:
                    return Exception(
                        f"2D mode is activated whereas clip normal is set to {utils.COORDINATES[-1]}. "
                        "Defaulting back to full diagram view."
                    )

                a, b = utils.compute_cut_interval(
                    input.clip_normal(), utils.COORDINATES[: len(domain)], domain
                )
                if not (a < input.clip_value() < b):
                    return Exception(
                        "Error in parsing plot options. Clips cannot be created on the selected coordinate boundaries."
                    )

                try:
                    return viewer.clip(
                        clip_normal=input.clip_normal(),
                        clip_value=input.clip_value(),
                        invert=input.invert(),
                        add_remains_as_wireframe=input.add_remains_as_wireframe(),
                    )
                except Exception as e:
                    return Exception(
                        "Error in parsing plot options, possibly clip value is too small for the given domain."
                        "More information: " + str(e),
                    )

            case _:
                return Exception(
                    f"Invalid view '{input.view()}' provided. Value must be one of {', '.join(DiagramView)}."
                )

    @reactive.calc
    def _metrics() -> utils.Metrics | Exception | None:
        diagram = _diagram()
        if not isinstance(diagram, utils.Diagram):
            return diagram

        return utils.calculate_metrics(diagram)

    @reactive.effect
    @reactive.event(input.reset_plot_options)
    def _():
        ui.update_select(id="view", selected=PLOT_DEFAULTS["view"])
        ui.update_select(id="colorby", selected=PLOT_DEFAULTS["colorby"])
        ui.update_select(id="colormap", selected=PLOT_DEFAULTS["colormap"])
        ui.update_slider(id="opacity", value=PLOT_DEFAULTS["opacity"])
        ui.update_switch(
            id="add_final_seed_positions",
            value=PLOT_DEFAULTS["add_final_seed_positions"],
        )

    @reactive.effect
    def _():
        choices = list(Colorby)
        selected = input.colorby()

        if input.view() == DiagramView.THRESHED:
            choices.remove(Colorby.RANDOM)
            choices.remove(Colorby.PHASES)

            if input.colorby() in (Colorby.RANDOM, Colorby.PHASES):
                selected = None

        ui.update_select(id="colorby", choices=choices, selected=selected)

    @render.ui
    def slice_value_slider() -> ui.Tag | None:
        d = processed_data()
        if not isinstance(d, utils.ProcessedData):
            return

        domain = d.config.domain
        coordinates = utils.COORDINATES[: len(domain)]

        if input.slice_normal() not in coordinates:
            return

        a, b = utils.compute_cut_interval(input.slice_normal(), coordinates, domain)

        return ui.input_slider(
            id="slice_value",
            label="Value along the selected normal",
            value=PLOT_DEFAULTS["slice_value"],
            min=a,
            max=b,
            ticks=True,
            width="100%",
        )

    @render.ui
    def clip_value_slider() -> ui.Tag | None:
        d = processed_data()
        if not isinstance(d, utils.ProcessedData):
            return

        domain = d.config.domain
        coordinates = utils.COORDINATES[: len(domain)]

        if input.clip_normal() not in coordinates:
            return

        a, b = utils.compute_cut_interval(
            input.clip_normal(),
            coordinates,
            domain,
        )
        return ui.input_slider(
            id="clip_value",
            label="Value along the selected normal",
            value=(a + b) / 2,
            min=a,
            max=b,
            ticks=True,
            width="100%",
        )

    @render.ui
    def extra_opts() -> ui.Tag | None:
        d = processed_data()
        if not isinstance(d, utils.ProcessedData):
            return

        domain = d.config.domain

        opts = ()
        match input.view():
            case DiagramView.FULL:
                return

            case DiagramView.THRESHED:
                opts += (
                    ui.input_slider(
                        id="threshed_value",
                        label=ui.div(
                            f"Slide to select grains whose {input.colorby()} fall in the selected range.",
                            ui.br(),
                            ui.help_text(
                                "Note that the values are rescaled to be in [0, 1] on the slider."
                            ),
                        ),
                        min=0.0,
                        max=1.0,
                        value=(0.0, 1.0),
                        ticks=True,
                        drag_range=False,
                        width="100%",
                    ),
                    views.invert_switch("invert"),
                )

            case DiagramView.SLICE:
                opts += (
                    comps.selection(
                        id="slice_normal",
                        label="Choose a normal for slicing",
                        choices=list(utils.COORDINATES[: len(domain)]),
                        selected=utils.COORDINATES[0],
                        width="100%",
                    ),
                    ui.output_ui("slice_value_slider"),
                )
            case DiagramView.CLIP:
                opts += (
                    comps.selection(
                        id="clip_normal",
                        label="Choose a normal for clipping",
                        choices=list(utils.COORDINATES[: len(domain)]),
                        selected=utils.COORDINATES[0],
                        width="100%",
                    ),
                    ui.output_ui("clip_value_slider"),
                    ui.row(
                        ui.column(5, views.invert_switch("invert")),
                        ui.column(
                            7,
                            ui.input_switch(
                                id="add_remains_as_wireframe",
                                label="Add remains",
                                width="100%",
                            ),
                        ),
                    ),
                )

        return ui.card(ui.card_header("Selected view plot options"), ui.div(*opts))

    @render.ui
    def metrics_tab_card() -> ui.Tag | None:
        if input.view() in [DiagramView.CLIP, DiagramView.THRESHED]:
            return ui.help_text(
                ui.markdown(
                    f"""
                    Metrics are not available for {input.view()} view. You can
                    only view the {input.view()} microstructure
                    by selecting the **Microstructure** button.
                    """
                )
            )

        metrics = _metrics()
        if metrics is None:
            return views.no_processed_data_text()

        if isinstance(metrics, Exception):
            comps.show_notification(str(metrics), type_="error")
            return

        @render.plot
        def plot_metrics():
            return metrics.fig

        @render.download(
            filename=lambda: (
                f"synthetmic-gui-metrics-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.zip"
            ),
            media_type="application/zip",
        )
        def download_metrics():
            yield utils.create_metrics_bytes(
                fig=metrics.fig, plot_data=metrics.plot_data
            )

        download_popover = ui.popover(
            comps.input_action_button(
                id="metrics_download_trigger",
                label="Download",
                icon="download",
                width=None,
            ),
            ui.markdown("Are you sure you want to download the current metrics?"),
            ui.help_text(
                ui.markdown(
                    """
                    This will download the current metrics. This involves downloading the plots in various 
                    file formats as well as the underlying data as json.
                    All outputs will be downloaded in .zip format which can be easily unzipped for futher processing.
                    """
                )
            ),
            comps.download_button(
                id="download_metrics",
                label="Yes, download the current metrics",
            ),
        )

        stats = []

        measure = utils.get_domain_measure(
            space_dim=metrics.space_dim, title=False, plural=False
        )
        for t, v in zip(
            [
                f"Max {measure} % error",
                f"Mean {measure} % error",
                f"Mean of grain {measure}s",
                f"Standard deviation of grain {measure}s",
                f"90th percentile of grain {measure}s",
                "Mean of ECDs",
                "Standard deviation of ECDs",
                "90th percentile of ECDs",
            ],
            [
                metrics.max_percentage_error,
                metrics.mean_percentage_error,
                metrics.fitted_volumes_mean,
                metrics.fitted_volumes_std,
                metrics.fitted_volumes_90p,
                metrics.ecds_mean,
                metrics.ecds_std,
                metrics.ecds_d90,
            ],
        ):
            if isinstance(v, float):
                v_formated = utils.format_to_standard_form(v, 2)
            elif v is None:
                v_formated = "NA"

            else:
                v_formated = v

            stats.append(
                ui.value_box(
                    title=t,
                    value=v_formated,
                    full_screen=False,
                    showcase=faicons.icon_svg("chart-line"),
                    theme="blue",
                )
            )

        return ui.div(
            ui.row(ui.layout_column_wrap(*stats[:4])),
            ui.row(ui.layout_column_wrap(*stats[4:])),
            ui.card(
                ui.card_body(ui.output_plot("plot_metrics")),
                ui.card_footer(download_popover, style="border: none"),
                full_screen=True,
                style=styles.metrics_card,
            ),
        )

    @render.ui
    def diagram_tab_card() -> ui.Tag | None:
        diagram = _diagram()
        if diagram is None:
            return views.no_processed_data_text()

        if isinstance(diagram, Exception):
            comps.show_notification(str(diagram), type_="error")
            return

        @render.ui
        def display_diagram():
            with tempfile.NamedTemporaryFile(suffix=".html", delete=True) as f:
                diagram.plotter.export_html(f.name)
                f.seek(0)
                html = f.read().decode("utf-8")

            return ui.tags.iframe(
                srcdoc=html, style="width: 100%; height: 100%; border: none;"
            )

        @render.download(
            filename=lambda: (
                f"synthetmic-gui-output-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.zip"
            ),
            media_type="application/zip",
        )
        def download_diagram():
            yield utils.create_diagram_bytes(diagram)

        download_popover = ui.popover(
            comps.input_action_button(
                id="diagram_download_trigger",
                label="Download",
                icon="download",
                width=None,
            ),
            ui.markdown("Are you sure you want to download the current diagram view?"),
            ui.help_text(
                ui.markdown(
                    "This will download the current diagram view and its properties "
                    "(such as seeds, volumes, vertices, etc) in various formats. "
                    "All outputs will be downloaded in .zip format which can be easily unzipped for futher processing."
                )
            ),
            comps.download_button(
                id="download_diagram",
                label="Yes, download the current diagram and properties",
            ),
        )

        return ui.card(
            ui.card_body(ui.output_ui("display_diagram")),
            ui.card_footer(download_popover, style="border: none"),
            full_screen=True,
            style=styles.diagram_card,
        )
