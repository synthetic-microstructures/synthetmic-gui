from shiny import ui

from shared import comps
from shared.consts import APP_NAME


def help() -> None:
    ui.modal_show(
        ui.modal(
            ui.markdown(f"### {APP_NAME}"),
            ui.hr(),
            ui.markdown(
                """
                #### Usage

                Using this app's tab is extremely easy. It can be done in **4 steps**:
                1. Specify both the space and box dimension. You can optionally
                choose whether the underlying domain should be periodic in all directions.
                1. Specify the number of grains or cells.
                1. Click on **Generate miscrostructure** button to generate synthetic miscrostructure.
                1. Click on any of the **download buttons** to either download
                the generated diagram (in different formats!) or the diagram properties
                (like centroids, vertices, etc; also in differnt formats!).

                That is it!

                Enjoy generating microstructures!
                """
            ),
            size="m",
            easy_close=True,
            footer=ui.modal_button(
                "Close",
                class_="btn btn-primary",
            ),
        )
    )


def sidebar() -> ui.Tag:
    return comps.group_ui_elements(
        ui.input_numeric(
            id="n_grains",
            label="Number of grains",
            value=1000,
        ),
        title="Grains",
        help_text="Number of grains or cells in the Voronoi diagram.",
    )
