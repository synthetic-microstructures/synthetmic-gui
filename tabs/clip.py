from shiny import Inputs, Outputs, Session, module, ui

from shared import utils


@module.ui
def tab_ui() -> ui.Tag:
    return ui.output_ui("main_ui")


@module.server
def server(
    input: Inputs,
    output: Outputs,
    session: Session,
    fitted_data: tuple[utils.SynthetMicData, utils.LaguerreDiagramGenerator],
    clip_normal: str,
    clip_value: float,
    colorby: str,
    colormap: str,
    opacity: float,
    add_final_seed_positions: bool,
):
    pass
