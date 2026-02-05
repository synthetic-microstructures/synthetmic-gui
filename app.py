from pathlib import Path

from shiny import App, Inputs, Outputs, Session, ui

from modules import laguerre, voronoi
from shared import styles, views
from shared.consts import APP_NAME

page_dependencies = ui.head_content(
    ui.tags.link(rel="stylesheet", type="text/css", href="layout.css"),
    ui.tags.link(rel="icon", type="image/png", href="favicon.ico"),
    ui.tags.style(styles.popover),
    ui.HTML(Path(Path(__file__).parent, "www", "ga", "analytics.html").read_text()),
)

app_ui = ui.page_fillable(
    ui.navset_bar(
        ui.nav_panel(
            "Laguerre Diagram",
            laguerre.ui_("laguerre"),
        ),
        ui.nav_panel("Voronoi Diagram", voronoi.ui_("voronoi")),
        navbar_options=ui.navbar_options(
            underline=True,
        ),
        title=APP_NAME,
    ),
    page_dependencies,
    title=APP_NAME,
)


def server(input: Inputs, output: Outputs, session: Session):
    views.info_modal()

    laguerre.server("laguerre")
    voronoi.server("voronoi")


app = App(app_ui, server, static_assets=Path(__file__).parent / "www")
