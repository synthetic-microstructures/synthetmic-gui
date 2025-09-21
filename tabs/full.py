from datetime import datetime

from shiny import Inputs, Outputs, Session, module, reactive, render, ui

from shared import utils, views


@module.ui
def tab_ui() -> ui.Tag:
    return views.create_diagram_view_card(
        views.create_add_seed_positions_switch("add_final_seed_positions"),
        diagram_display_id="display_diagram",
        download_btn_id="yield_diagram",
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
    @reactive.calc
    def _diagram() -> utils.Diagram:
        data, generator = fitted_data

        return utils.generate_full_diagram(
            data=data,
            generator=generator,
            colorby=colorby,
            colormap=colormap,
            add_final_seed_positions=input.add_final_seed_positions(),
            opacity=opacity,
        )

    @render.ui
    def display_diagram():
        return ui.HTML(utils.plotter_to_html(_diagram().plotter, "full"))

    @render.download(
        filename=lambda: f"full-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.zip",
        media_type="application/zip",
    )
    def yield_diagram():
        yield utils.create_full_download_bytes(_diagram())
