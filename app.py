from pathlib import Path

from shiny import App, Inputs, Outputs, Session, reactive, render, ui

from modules import laguerre, voronoi
from shared import comps, styles, views
from shared.consts import APP_NAME, DiagramType

page_dependencies = ui.head_content(
    ui.tags.link(rel="stylesheet", type="text/css", href="layout.css"),
    ui.tags.link(rel="icon", type="image/png", href="favicon.ico"),
    ui.tags.style(styles.popover),
    ui.HTML(Path(Path(__file__).parent, "www", "ga", "analytics.html").read_text()),
)

app_ui = ui.page_fillable(
    ui.output_ui("page"),
    comps.create_input_action_button(
        id="change_diagram", label="Change diagram type", icon="arrow-right-arrow-left"
    ),
    page_dependencies,
    title=APP_NAME,
)


def server(input: Inputs, output: Outputs, session: Session):
    views.info_modal(diagram_id="diagram")

    @reactive.effect
    @reactive.event(input.change_diagram)
    def _():
        views.info_modal(diagram_id="diagram")

    laguerre.server("laguerre")
    voronoi.server("voronoi")

    @render.ui
    def page() -> ui.Tag:
        register = {
            DiagramType.LAGUERRE: laguerre.ui_("laguerre"),
            DiagramType.VORONOI: voronoi.ui_("voronoi"),
        }

        return register[input.diagram()]


app = App(app_ui, server, static_assets=Path(__file__).parent / "www")
