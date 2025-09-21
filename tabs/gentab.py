from datetime import datetime

import faicons as fa
import numpy as np
from matplotlib import colormaps
from shiny import Inputs, Outputs, Session, module, reactive, render, ui

import shared.controls as ct
from shared import styles, views
from shared.utils import (
    COORDINATES,
    Diagram,
    LaguerreDiagramGenerator,
    SynthetMicData,
    create_diagram_download_bytes,
    create_prop_download_bytes,
    generate_full_diagram,
    generate_slice_diagram,
)


@module.ui
def tab() -> ui.Tag:
    return ui.output_ui("main")


@module.server
def server(
    input: Inputs,
    output: Outputs,
    session: Session,
    _fitted_data: tuple[SynthetMicData, LaguerreDiagramGenerator],
):
    def is_slice_view(view: str, data: np.ndarray) -> bool:
        return (view == ct.DiagramView.SLICE) and (len(data) == 3)

    @reactive.calc
    def _diagram() -> Diagram:
        data, generator = _fitted_data
        if is_slice_view(input.view(), data.domain):
            # ensure that the current slide value is always in the correct range
            a, b = _interval()
            slice_center = input.slice_center()
            if not (a <= slice_center <= b):
                slice_center = ct.PLOT_DEFAULTS["slice_center"]

            return generate_slice_diagram(
                data=data,
                generator=generator,
                slice_value=slice_center,
                slice_normal=input.slice_normal(),
                colorby=input.colorby(),
                colormap=input.colormap(),
                add_final_seed_positions=input.addpositions(),
                opacity=input.opacity(),
            )

        return generate_full_diagram(
            data=data,
            generator=generator,
            colorby=input.colorby(),
            colormap=input.colormap(),
            add_final_seed_positions=input.addpositions(),
            opacity=input.opacity(),
        )

    @reactive.calc
    @reactive.event(input.slice_normal)
    def _interval() -> tuple[float, float]:
        data, _ = _fitted_data

        domain_map = dict(zip(COORDINATES, data.domain))
        a, b = domain_map[input.slice_normal()]

        return float(a), float(b)

    @reactive.effect
    def _():
        data, _ = _fitted_data
        if is_slice_view(input.view(), data.domain):
            ui.update_select(id="slice_normal", selected=input.slice_normal())
            # since the slide center will be adjusted based on the inputs both from the main
            # slider and pot option slider; don't remember the previous value.

        ui.update_select(id="view", selected=input.view())
        ui.update_select(id="colorby", selected=input.colorby())
        ui.update_select(id="colormap", selected=input.colormap())
        ui.update_switch(id="addpositions", value=input.addpositions())
        ui.update_slider(id="opacity", value=input.opacity())

    @reactive.effect
    @reactive.event(input.reset_plot_options)
    def _():
        data, _ = _fitted_data

        if is_slice_view(input.view(), data.domain):
            ui.update_select(
                id="slice_normal", selected=ct.PLOT_DEFAULTS["slice_normal"]
            )
            ui.update_slider(id="slice_center", value=ct.PLOT_DEFAULTS["slice_center"])

        ui.update_select(id="colorby", selected=ct.PLOT_DEFAULTS["colorby"])
        ui.update_select(id="colormap", selected=ct.PLOT_DEFAULTS["colormap"])
        ui.update_switch(id="addpositions", value=ct.PLOT_DEFAULTS["addpositions"])
        ui.update_slider(id="opacity", value=ct.PLOT_DEFAULTS["opacity"])

    @reactive.effect
    @reactive.event(input.goto_download_diagram)
    def _():
        diagram_download_modal = ui.modal(
            views.create_selection(
                id="fig_extension",
                label="Download full diagram as",
                choices=[e for e in ct.FigureExtension],
                selected=ct.PLOT_DEFAULTS["fig_extension"],
            ),
            title="Download diagram",
            easy_close=True,
            footer=ui.download_button(
                id="download_diagram",
                label="Download",
                icon=fa.icon_svg("download"),
                class_="btn btn-primary",
            ),
            size="s",
            fade=True,
        )

        ui.modal_show(diagram_download_modal)

    @reactive.effect
    @reactive.event(input.goto_download_diagram_prop)
    def _():
        ui.modal_show(
            ui.modal(
                views.create_selection(
                    id="prop_extension",
                    label=(
                        "Download properties as",
                        ui.help_text(
                            ui.markdown(
                                "Note that the vertices will always be saved as json, with key as cell IDs and values as vertices. "
                                "In the case of 3D, vertices are saved for each faces in a grain."
                            ),
                        ),
                    ),
                    choices=[e for e in ct.PropertyExtension],
                    selected=ct.PLOT_DEFAULTS["prop_extension"],
                ),
                footer=ui.download_button(
                    id="download_diagram_property",
                    label="Download",
                    icon=fa.icon_svg("download"),
                    class_="btn btn-primary",
                ),
                title="Download diagram properties",
                easy_close=True,
                size="s",
                fade=True,
            )
        )

    @render.ui
    def display_diagram():
        return ui.HTML(_diagram().plotter.export_html(filename=None).read())  # type: ignore

    @render.download(
        filename=lambda: f"main-diagram-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.{input.fig_extension()}"
    )
    def download_diagram():
        yield create_diagram_download_bytes(
            _diagram().mesh, _diagram().plotter, input.fig_extension()
        )

    @render.download(
        filename=lambda: f"diagram-property-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.zip",
        media_type="application/zip",
    )
    def download_diagram_property():
        yield create_prop_download_bytes(_diagram(), input.prop_extension())

    @render.ui
    def view_selection():
        data, _ = _fitted_data
        return views.create_selection(
            id="view",
            label="Choose a diagram to view",
            choices=[s for s in ct.DiagramView]
            if len(data.domain) == 3  # ensure that slice option only show for 3d case
            else [
                ct.DiagramView.FULL,
            ],
            selected=ct.DiagramView.FULL,
        )

    @render.ui
    def slice_normal_selection():
        return views.create_selection(
            id="slice_normal",
            label="Choose a coordinate along which the slice will be created",
            choices=list(COORDINATES),
            selected=COORDINATES[0],
        )

    @render.ui
    def slice_center_slider():
        a, b = _interval()
        return ui.input_slider(
            id="slice_center",
            label="Slide to select value along slice coordinate",
            value=ct.PLOT_DEFAULTS["slice_center"],
            min=a,
            max=b,
            ticks=True,
        )

    @render.ui
    def opts():
        opts = []

        data, _ = _fitted_data
        if is_slice_view(input.view(), data.domain):
            opts.extend(
                [
                    ui.output_ui("slice_normal_selection"),
                    ui.output_ui("slice_center_slider"),
                ]
            )

        opts.extend(
            [
                views.create_selection(
                    id="colorby",
                    label="Color by",
                    choices=[c for c in ct.Colorby],
                    selected=ct.PLOT_DEFAULTS["colorby"],
                ),
                views.create_selection(
                    id="colormap",
                    label="Choose a colormap",
                    choices=sorted(list(colormaps)),
                    selected=ct.PLOT_DEFAULTS["colormap"],
                ),
                ui.input_switch(
                    "addpositions",
                    "Add final seed positions",
                    ct.PLOT_DEFAULTS["addpositions"],
                ),
                ui.input_slider(
                    id="opacity",
                    label="Diagram opacity",
                    min=0.0,
                    max=1.0,
                    value=ct.PLOT_DEFAULTS["opacity"],
                    ticks=True,
                ),
                ui.input_action_button(
                    id="reset_plot_options",
                    label="Reset plot options to defaults",
                    icon=fa.icon_svg("gear"),
                    class_="btn btn-primary",
                ),
            ]
        )

        return opts

    @render.ui
    def main():
        return ui.tags.div(
            ui.row(
                ui.column(
                    3,
                    ui.card(
                        ui.output_ui("view_selection"),
                        ui.output_ui("opts"),
                        style=styles.tab_card,
                    ),
                ),
                ui.column(
                    9,
                    ui.card(
                        ui.output_ui("display_diagram"),
                        ui.card_header(
                            ui.popover(
                                ui.span(
                                    fa.icon_svg(
                                        "download",
                                        fill=ct.FILL_COLOUR,
                                        height="20px",
                                        width="20px",
                                    ),
                                    style="position:absolute; top: 5px; right: 7px;",
                                ),
                                ui.row(
                                    ui.column(
                                        6,
                                        ui.input_action_button(
                                            id="goto_download_diagram",
                                            label="Download diagram",
                                            icon=fa.icon_svg("download"),
                                            class_="btn btn-primary",
                                        ),
                                    ),
                                    ui.column(
                                        6,
                                        ui.input_action_button(
                                            id="goto_download_diagram_prop",
                                            label="Download properties",
                                            icon=fa.icon_svg("download"),
                                            class_="btn btn-primary",
                                        ),
                                    ),
                                ),
                            ),
                        ),
                        style=styles.tab_card,
                        full_screen=True,
                    ),
                ),
            )
        )
