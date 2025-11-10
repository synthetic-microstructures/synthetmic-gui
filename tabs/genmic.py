from datetime import datetime

import faicons
from matplotlib import colormaps
from shiny import Inputs, Outputs, Session, module, reactive, render, ui

import shared.controls as ct
from shared import styles, utils, views


@module.ui
def page_ui() -> ui.Tag:
    return ui.tags.div(
        ui.output_ui("tab_radio_buttons"),
        ui.row(
            ui.column(
                3,
                ui.output_ui("common_opts"),
                ui.output_ui("extra_opts"),
            ),
            ui.column(
                9,
                ui.output_ui("selected_tab"),
            ),
        ),
    )


@module.server
def server(
    input: Inputs,
    output: Outputs,
    session: Session,
    fitted_data: tuple[utils.SynthetMicData, utils.LaguerreDiagramGenerator],
    global_generate: reactive.Value,
) -> None:
    @reactive.calc
    def _calculate_diagram() -> utils.Diagram | Exception:
        data, generator = fitted_data
        match input.view():
            case ct.DiagramView.FULL:
                return utils.generate_full_diagram(
                    data=data,
                    generator=generator,
                    colorby=input.colorby(),
                    colormap=input.colormap(),
                    opacity=input.opacity(),
                    add_final_seed_positions=input.add_final_seed_positions(),
                )

            case ct.DiagramView.SLICE:
                # when slice view is selected (in 3D mode) and user
                # changes mode to 2D (and then press generate); default to
                # full view and quit.
                if len(data.domain) == 2:
                    return Exception(
                        "Slicing is only available for 3D case. Defaulting back to full diagram."
                    )

                return utils.generate_slice_diagram(
                    data=data,
                    generator=generator,
                    colorby=input.colorby(),
                    slice_normal=input.slice_normal(),
                    slice_value=input.slice_value(),
                    colormap=input.colormap(),
                    opacity=input.opacity(),
                )

            case ct.DiagramView.CLIP:
                if (
                    len(data.domain) == 2
                    and input.clip_normal() == utils.COORDINATES[-1]
                ):
                    return Exception(
                        f"2D mode is activated whereas clip normal is set to {utils.COORDINATES[-1]}. "
                        "Defaulting back to full diagram view."
                    )

                a, b = utils.compute_cut_interval(
                    input.clip_normal(),
                    utils.COORDINATES[: len(data.domain)],
                    data.domain,
                )
                if not (a < input.clip_value() < b):
                    return Exception(
                        "Error in parsing plot options. Clips cannot be created on the selected coordinate boundaries."
                    )

                try:
                    return utils.generate_clip_diagram(
                        data=data,
                        generator=generator,
                        colorby=input.colorby(),
                        clip_normal=input.clip_normal(),
                        clip_value=input.clip_value(),
                        invert=input.invert(),
                        add_remains_as_wireframe=input.add_remains_as_wireframe(),
                        colormap=input.colormap(),
                        opacity=input.opacity(),
                    )
                except Exception as e:
                    return Exception(
                        "Error in parsing plot options, possibly clip value is too small for the given domain."
                        "More information: " + str(e),
                    )

            case _:
                return Exception(
                    f"Invalid view '{input.view()}' provided. Value must be one of {', '.join(ct.DiagramView)}."
                )

    @reactive.calc
    def _calculate_metrics() -> utils.Metrics | Exception:
        diagram = _calculate_diagram()
        if isinstance(diagram, Exception):
            return diagram

        return utils.calculate_metrics(diagram)

    @reactive.effect
    @reactive.event(global_generate)
    def _():
        data, _ = fitted_data
        # to ensure the current plot settings are remembered after
        # reclicking generate button
        if input.view() == ct.DiagramView.SLICE and len(data.domain) == 2:
            ui.update_select(
                id="view",
                selected=ct.DiagramView.FULL,
            )
        else:
            ui.update_select(id="view", selected=input.view())

        ui.update_radio_buttons(id="tab", selected=input.tab())
        ui.update_select(id="colorby", selected=input.colorby())
        ui.update_select(id="colormap", selected=input.colormap())
        ui.update_slider(id="opacity", value=input.opacity())

    @reactive.effect
    @reactive.event(input.reset_plot_options)
    def _():
        ui.update_select(id="view", selected=ct.PLOT_DEFAULTS["view"])
        ui.update_select(id="colorby", selected=ct.PLOT_DEFAULTS["colorby"])
        ui.update_select(id="colormap", selected=ct.PLOT_DEFAULTS["colormap"])
        ui.update_slider(id="opacity", value=ct.PLOT_DEFAULTS["opacity"])

    @render.ui
    def common_opts() -> ui.Tag:
        data, _ = fitted_data
        view_choices = [v for v in ct.DiagramView]
        if len(data.domain) == 2:
            view_choices.remove(ct.DiagramView.SLICE)

        return ui.card(
            ui.card_header("Common plot options"),
            ui.tags.div(
                views.create_selection(
                    id="view",
                    label="Choose a diagram to view",
                    choices=view_choices,
                    selected=ct.DiagramView.FULL,
                    width="100%",
                ),
                views.create_selection(
                    id="colorby",
                    label="Color by",
                    choices=[c for c in ct.Colorby],
                    selected=ct.PLOT_DEFAULTS["colorby"],
                    width="100%",
                ),
                views.create_selection(
                    id="colormap",
                    label="Choose a colormap",
                    choices=sorted(list(colormaps)),
                    selected=ct.PLOT_DEFAULTS["colormap"],
                    width="100%",
                ),
                ui.input_slider(
                    id="opacity",
                    label="Diagram opacity",
                    min=0.0,
                    max=1.0,
                    value=ct.PLOT_DEFAULTS["opacity"],
                    ticks=True,
                    width="100%",
                ),
                views.create_input_action_button(
                    id="reset_plot_options",
                    label="Reset common plot options to defaults",
                    icon="gear",
                ),
            ),
        )

    @render.ui
    def slice_value_slider() -> ui.Tag | None:
        data, _ = fitted_data
        coordinates = utils.COORDINATES[: len(data.domain)]

        if input.slice_normal() not in coordinates:
            return

        a, b = utils.compute_cut_interval(
            input.slice_normal(),
            coordinates,
            data.domain,
        )

        return ui.input_slider(
            id="slice_value",
            label="Value along the selected normal",
            value=ct.PLOT_DEFAULTS["slice_value"],
            min=a,
            max=b,
            ticks=True,
            width="100%",
        )

    @render.ui
    def clip_value_slider() -> ui.Tag | None:
        data, _ = fitted_data
        coordinates = utils.COORDINATES[: len(data.domain)]

        if input.clip_normal() not in coordinates:
            return

        a, b = utils.compute_cut_interval(
            input.clip_normal(),
            coordinates,
            data.domain,
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
    def extra_opts() -> ui.Tag:
        data, _ = fitted_data

        opts = ()
        match input.view():
            case ct.DiagramView.FULL:
                opts += (
                    ui.input_switch(
                        id="add_final_seed_positions",
                        label="Add final seed positions",
                        width="100%",
                    ),
                )

            case ct.DiagramView.SLICE:
                opts += (
                    views.create_selection(
                        id="slice_normal",
                        label="Choose a normal for slicing",
                        choices=list(utils.COORDINATES[: len(data.domain)]),
                        selected=utils.COORDINATES[0],
                        width="100%",
                    ),
                    ui.output_ui("slice_value_slider"),
                )
            case ct.DiagramView.CLIP:
                opts += (
                    views.create_selection(
                        id="clip_normal",
                        label="Choose a normal for clipping",
                        choices=list(utils.COORDINATES[: len(data.domain)]),
                        selected=utils.COORDINATES[0],
                        width="100%",
                    ),
                    ui.output_ui("clip_value_slider"),
                    ui.row(
                        ui.column(
                            5,
                            ui.input_switch(
                                id="invert",
                                label="Invert",
                                width="100%",
                            ),
                        ),
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

        return ui.card(ui.card_header("Selected view plot options"), ui.tags.div(*opts))

    @render.ui
    def metrics_tab_card() -> ui.Tag | None:
        metrics = _calculate_metrics()
        if isinstance(metrics, Exception):
            views.create_error_notification(str(metrics))
            return

        @render.plot
        def plot_metrics():
            return metrics.fig

        @render.download(
            filename=lambda: f"synthetmic-gui-metrics-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.zip",
            media_type="application/zip",
        )
        def download_metrics():
            yield utils.create_metrics_bytes(
                fig=metrics.fig, plot_data=metrics.plot_data
            )

        download_popover = ui.popover(
            ui.span(
                faicons.icon_svg(
                    "download",
                    fill=ct.FILL_COLOUR,
                ),
                style=styles.diagram_download_trigger,
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
            views.create_download_button(
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
                metrics.fitted_volumes_90_percentile,
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
                    showcase=faicons.icon_svg("square-poll-horizontal"),
                )
            )

        if input.view() == ct.DiagramView.CLIP:
            return ui.card(
                ui.card_header(
                    faicons.icon_svg("circle-exclamation", fill=ct.FILL_COLOUR),
                    "A note on metrics",
                ),
                ui.markdown(
                    """
                    Metrics are not available for clip view. You can only view the clipped microstructure
                    by selecting the **Microstructure** tab.
                    """
                ),
            )

        return ui.tags.div(
            ui.row(ui.layout_column_wrap(*stats[:4])),
            ui.row(ui.layout_column_wrap(*stats[4:])),
            ui.card(
                ui.output_plot("plot_metrics"),
                download_popover,
                full_screen=True,
                style=styles.metrics_card,
            ),
        )

    @render.ui
    def diagram_tab_card() -> ui.Tag | None:
        diagram = _calculate_diagram()
        if isinstance(diagram, Exception):
            views.create_error_notification(str(diagram))
            return

        @render.ui
        def display_diagram():
            return ui.HTML(diagram.plotter.export_html(filename=None).read())

        @render.download(
            filename=lambda: f"synthetmic-gui-output-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.zip",
            media_type="application/zip",
        )
        def download_diagram():
            yield utils.create_diagram_bytes(diagram)

        download_popover = ui.popover(
            ui.span(
                faicons.icon_svg(
                    "download",
                    fill=ct.FILL_COLOUR,
                ),
                style=styles.diagram_download_trigger,
            ),
            ui.markdown("Are you sure you want to download the current diagram view?"),
            ui.help_text(
                ui.markdown(
                    "This will download the current diagram view and its properties "
                    "(such as seeds, volumes, vertices, etc) in various formats. "
                    "All outputs will be downloaded in .zip format which can be easily unzipped for futher processing."
                )
            ),
            views.create_download_button(
                id="download_diagram",
                label="Yes, download the current diagram and properties",
            ),
        )

        return ui.card(
            ui.output_ui("display_diagram"),
            download_popover,
            full_screen=True,
            style=styles.diagram_card,
        )

    @render.ui
    def tab_radio_buttons() -> ui.Tag:
        return ui.div(
            ui.input_radio_buttons(
                id="tab",
                label=None,
                choices=[t for t in ct.Tab],
                inline=True,
            ),
            style=styles.tab_radio_button,
        )

    @render.ui
    def selected_tab() -> ui.Tag:
        tabs = dict(
            zip(
                ct.Tab,
                (ui.output_ui("diagram_tab_card"), ui.output_ui("metrics_tab_card")),
            )
        )

        return tabs[input.tab()]
