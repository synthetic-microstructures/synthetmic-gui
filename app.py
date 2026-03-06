from pathlib import Path

from shiny import App, Inputs, Outputs, Session, ui

from modules import home, main
from shared.consts import APP_NAME

page_dependencies = ui.head_content(
    ui.tags.link(rel="stylesheet", type="text/css", href="layout.css"),
    ui.tags.link(rel="stylesheet", type="text/css", href="style.css"),
    ui.HTML(Path(Path(__file__).parent, "www", "ga", "analytics.html").read_text()),
)


app_ui = ui.page_fillable(
    page_dependencies,
    ui.navset_bar(
        ui.nav_panel(
            "Home",
            home.ui_("home"),
        ),
        ui.nav_panel(
            "Microstructure generation",
            main.ui_("main"),
        ),
        title=APP_NAME,
        navbar_options=ui.navbar_options(underline=True),
    ),
    ui.head_content(
        ui.tags.link(rel="icon", type="image/png", href="favicon.ico"),
    ),
    title=APP_NAME,
)


def server(input: Inputs, output: Outputs, session: Session):
    home.server("home")
    main.server("main")


app = App(
    app_ui,
    server,
    static_assets={"/": Path(__file__).parent / "www"},
)
