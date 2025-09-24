from datetime import datetime

import faicons as fa
from matplotlib import colormaps
from shiny import Inputs, Outputs, Session, module, reactive, render, ui
from shiny_validate import InputValidator, check

import shared.controls as ct
from shared import styles, utils, views


@module.ui
def page_ui() -> ui.Tag:
    return ui.row(
        ui.column(
            3,
            ui.output_ui("common_opts"),
            ui.output_ui("extra_opts"),
            ui.output_ui("update_btns"),
        ),
        ui.column(
            9,
            ui.output_ui("diagram_view_card"),
        ),
    )


@module.server
def server(
    input: Inputs,
    output: Outputs,
    session: Session,
    fitted_data: tuple[utils.SynthetMicData, utils.LaguerreDiagramGenerator],
    global_generate: reactive.Value,
):
    iv = InputValidator()
    _diagram = reactive.Value(
        utils.generate_full_diagram(
            data=fitted_data[0],
            generator=fitted_data[1],
            colormap=ct.PLOT_DEFAULTS["colormap"],
            opacity=ct.PLOT_DEFAULTS["opacity"],
            add_final_seed_positions=False,
        )
    )

    @reactive.effect
    @reactive.event(input.update_plot_options)
    def _():
        data, generator = fitted_data
        match input.view():
            case ct.DiagramView.FULL:
                _diagram.set(
                    utils.generate_full_diagram(
                        data=data,
                        generator=generator,
                        colormap=input.colormap(),
                        opacity=input.opacity(),
                        add_final_seed_positions=input.add_final_seed_positions(),
                    )
                )

            case ct.DiagramView.SLICE:
                # when slice view is selected (in 3D mode) and user
                # changes mode to 2D (and then press generate); default to
                # full view and quit.
                if len(data.domain) == 2:
                    views.create_error_notification(
                        "Slicing is only available for 3D case. Defaulting back to full diagram."
                    )

                else:
                    a, b = utils.compute_cut_interval(
                        input.slice_normal(), utils.COORDINATES, data.domain
                    )
                    iv.add_rule(
                        "slice_value",
                        check.compose_rules(
                            utils.required(), utils.between(left=a, right=b)
                        ),
                    )
                    iv.enable()
                    if iv.is_valid():
                        _diagram.set(
                            utils.generate_slice_diagram(
                                data=data,
                                generator=generator,
                                slice_normal=input.slice_normal(),
                                slice_value=input.slice_value(),
                                colormap=input.colormap(),
                                opacity=input.opacity(),
                            )
                        )

            case ct.DiagramView.CLIP:
                if (
                    len(data.domain) == 2
                    and input.clip_normal() == utils.COORDINATES[-1]
                ):
                    views.create_error_notification(
                        f"2D mode is activated whereas clip normal is set to {utils.COORDINATES[-1]}. "
                        "Defaulting back to full diagram view."
                    )
                else:
                    a, b = utils.compute_cut_interval(
                        input.clip_normal(),
                        utils.COORDINATES[: len(data.domain)],
                        data.domain,
                    )
                    iv.add_rule(
                        "clip_value",
                        check.compose_rules(
                            utils.required(),
                            utils.between(left=a, right=b, both_open=True),
                        ),
                    )
                    iv.enable()
                    if iv.is_valid():
                        try:
                            _diagram.set(
                                utils.generate_clip_diagram(
                                    data=data,
                                    generator=generator,
                                    clip_normal=input.clip_normal(),
                                    clip_value=input.clip_value(),
                                    invert=input.invert(),
                                    add_remains_as_wireframe=input.add_remains_as_wireframe(),
                                    colormap=input.colormap(),
                                    opacity=input.opacity(),
                                )
                            )
                        except Exception as e:
                            views.create_error_notification(
                                "Error in parsing plot options, possibly clip value is too small for the given domain."
                                "More information: " + str(e),
                            )

    @render.ui
    def display_diagram():
        return ui.HTML(_diagram().plotter.export_html(filename=None).read())  # type: ignore

    @render.download(
        filename=lambda: f"synthetmic-gui-output-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.zip",
        media_type="application/zip",
    )
    def download_diagram():
        yield utils.create_full_download_bytes(_diagram())

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

        ui.update_select(id="colormap", selected=input.colormap())
        ui.update_slider(id="opacity", value=input.opacity())

    @reactive.effect
    @reactive.event(input.reset_plot_options)
    def _():
        ui.update_select(id="view", selected=ct.PLOT_DEFAULTS["view"])
        ui.update_select(id="colormap", selected=ct.PLOT_DEFAULTS["colormap"])
        ui.update_slider(id="opacity", value=ct.PLOT_DEFAULTS["opacity"])

    @render.ui
    def common_opts():
        data, _ = fitted_data
        view_choices = [v for v in ct.DiagramView]
        if len(data.domain) == 2:
            view_choices.remove(ct.DiagramView.SLICE)

        return ui.card(
            ui.card_header("Common plot options"),
            views.create_selection(
                id="view",
                label="Choose a diagram to view",
                choices=view_choices,
                selected=ct.DiagramView.FULL,
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
        )

    @render.ui
    def extra_opts():
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
                        choices=list(utils.COORDINATES),
                        selected=utils.COORDINATES[0],
                        width="100%",
                    ),
                    ui.input_numeric(
                        id="slice_value",
                        label="Value along the selected normal",
                        value=ct.PLOT_DEFAULTS["slice_value"],
                        width="100%",
                    ),
                )
            case ct.DiagramView.CLIP:
                data, _ = fitted_data
                min_lb = min(min(arr) for arr in data.domain)
                min_ub = min(max(arr) for arr in data.domain)
                clip_default_value = min_lb + 0.5 * min_ub
                opts += (
                    views.create_selection(
                        id="clip_normal",
                        label="Choose a normal for clipping",
                        choices=list(utils.COORDINATES[: len(data.domain)]),
                        selected=utils.COORDINATES[0],
                        width="100%",
                    ),
                    ui.input_numeric(
                        id="clip_value",
                        label="Value along the selected normal",
                        value=clip_default_value,
                        width="100%",
                    ),
                    ui.input_switch(
                        id="invert",
                        label="Invert clip",
                        width="100%",
                    ),
                    ui.input_switch(
                        id="add_remains_as_wireframe",
                        label="Add remains as wireframe",
                        width="100%",
                    ),
                )

        return ui.card(ui.card_header("Slice and clip options"), *opts)

    @render.ui
    def update_btns():
        return ui.card(
            ui.card_header("Update center"),
            ui.input_task_button(
                id="update_plot_options",
                label="Update plot",
                icon=fa.icon_svg("arrows-spin"),
                width="100%",
                class_="btn btn-primary",
            ),
            ui.input_action_button(
                id="reset_plot_options",
                label="Reset common plot options to defaults",
                icon=fa.icon_svg("gear"),
                width="100%",
                class_="btn btn-primary",
            ),
        )

    @render.ui
    def diagram_view_card():
        download_popover = ui.popover(
            ui.span(
                fa.icon_svg(
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
            ui.download_button(
                id="download_diagram",
                label="Yes, download the current diagram and properties",
                icon=fa.icon_svg("download"),
                class_="btn btn-primary",
            ),
        )

        return ui.card(
            ui.output_ui("display_diagram"),
            ui.card_header(download_popover),
            full_screen=True,
            style=styles.diagram_card,
        )
