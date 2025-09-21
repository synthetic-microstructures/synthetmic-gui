from datetime import datetime

import faicons as fa
from shiny import Inputs, Outputs, Session, module, reactive, render, ui
from shiny_validate import InputValidator

from shared import controls as ct
from shared import utils, views


@module.ui
def tab_ui() -> ui.Tag:
    return views.create_diagram_view_card(
        views.create_selection(
            id="slice_normal",
            label=(
                "Choose a coordinate or normal",
                ui.p(),
                ui.help_text("This the normal along which the slice will be created."),
            ),
            choices=list(utils.COORDINATES),
            selected=utils.COORDINATES[0],
            width="100%",
        ),
        ui.input_numeric(
            id="slice_value",
            label=(
                "Value along the selected normal or coordinate",
                ui.p(),
                ui.help_text(
                    "This is the value along the selected normal where the slice will be created.",
                    "Value must be in the range of the selected coordinate or normal.",
                ),
            ),
            value=ct.PLOT_DEFAULTS["slice_value"],
            width="100%",
        ),
        ui.input_action_button(
            id="apply_slice_opts",
            label="Update plot",
            icon=fa.icon_svg("arrows-spin"),
            width="100%",
            class_="btn btn-primary",
        ),
        diagram_display_id="display_slice",
        download_btn_id="yield_slice",
    )


@module.server
def server(
    input: Inputs,
    output: Outputs,
    session: Session,
    fitted_data: tuple[utils.SynthetMicData, utils.LaguerreDiagramGenerator],
    colorby: str,
    colormap: str,
    opacity: float,
):
    _diagram = reactive.Value(
        utils.generate_slice_diagram(
            data=fitted_data[0],
            generator=fitted_data[1],
            slice_normal=ct.PLOT_DEFAULTS["slice_normal"],
            slice_value=ct.PLOT_DEFAULTS["slice_value"],
            colorby=colorby
            if colorby in (ct.Colorby.RANDOM, ct.Colorby.FITTED_VOLUMES)
            else ct.Colorby.FITTED_VOLUMES,
            colormap=colormap,
            opacity=opacity,
        )
    )

    iv = InputValidator()
    iv.add_rule("slice_value", utils.gte(rhs=0.0))

    @reactive.effect
    @reactive.event(input.apply_slice_opts)
    def _():
        data, generator = fitted_data

        a, b = utils.compute_cut_interval(
            input.slice_normal(), utils.COORDINATES, data.domain
        )
        iv.add_rule("slice_value", utils.between(left=a, right=b))
        iv.enable()
        if not iv.is_valid():
            ui.notification_show(
                "Error in parsing plot options. Please check your inputs and try again."
            )
            return

        _diagram.set(
            utils.generate_slice_diagram(
                data=data,
                generator=generator,
                slice_normal=input.slice_normal(),
                slice_value=input.slice_value(),
                colorby=colorby
                if colorby in (ct.Colorby.RANDOM, ct.Colorby.FITTED_VOLUMES)
                else ct.Colorby.FITTED_VOLUMES,
                colormap=colormap,
                opacity=opacity,
            )
        )

    @render.ui
    def display_slice():
        return ui.HTML(utils.plotter_to_html(_diagram().plotter, "slice"))

    @render.download(
        filename=lambda: f"slice-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.zip",
        media_type="application/zip",
    )
    def yield_slice():
        yield utils.create_full_download_bytes(_diagram())
