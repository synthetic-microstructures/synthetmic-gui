from shiny import ui

from shared import comps


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
