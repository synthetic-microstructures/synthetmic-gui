from typing import Any

import pandas as pd
from shiny import ui

from shared import comps
from shared.consts import (
    APP_NAME,
    APP_VERSION,
    DiagramType,
    Distribution,
)
from shared.utils import COORDINATES, VOLUMES


def seeds_summary(seeds: pd.DataFrame | None, table_id: str) -> ui.Tag:
    if seeds is None:
        return ui.help_text(
            "No seeds uploaded yet. Information about seeds will be displayed here after upload."
        )

    return ui.output_table(table_id)


def app_version_text() -> ui.Tag:
    return (ui.help_text(f"{APP_NAME} {APP_VERSION}"),)


def create_help(help_id: str) -> ui.Tag:
    return ui.card(
        ui.card_header("Help center"),
        ui.markdown(
            f"Need help? Read how to use {APP_NAME} by clicking the button below."
        ),
        comps.create_input_action_button(
            id=help_id,
            label="Click here for help and to see some examples",
            icon="lightbulb",
        ),
    )


def create_dist_selection(
    id: str, label: str = "Choose a volume distribution"
) -> ui.Tag:
    return comps.create_selection(
        id=id,
        label=label,
        choices=[d for d in Distribution],
        selected=Distribution.CONSTANT,
    )


def create_numeric_input(
    ids: list[str], labels: list[str], defaults: list[Any]
) -> ui.Tag:
    width = 12 // len(ids)
    cols = [
        ui.column(
            width,
            ui.input_numeric(
                id=id,
                label=label,
                value=value,
            ),
        )
        for id, label, value in zip(ids, labels, defaults)
    ]

    return ui.row(*cols)


def box_help_text() -> ui.HTML:
    return ui.markdown(
        """
        **Length**, **Breadth**, and **Height** are, respectively, the length, breadth, and height of the box. 

        **Periodicity** helps turn on periodicity of the domain in any of the given coordinates.
       """
    )


def grains_help_text() -> ui.HTML:
    return ui.markdown(
        f"""
        Here, you control the number and distribution of grains, as well as the
        distribution of volumes.

        There are three options in achieving these controls: **single**, **dual**, and
        **upload**.

        If single is selected, you will be prompted to enter the number of grains and select
        from the available volume distributions.

        In case of dual, you will have the flexibility of entering both the number of grains
        and volume distribution for each phase. Note that each phase can have a different
        distribution.

        If you choose to upload volumes instead, then the uploaded csv or txt file must have **only
        one column** named '{VOLUMES}', and all its values must be float.

        We support three distributions: **constant**, **uniform**, and **lognormal**.

        For constant, all volumes will be the same.

        In case of uniform distribution, **Low** and **High** are the
        boundaries of the distribution such that the generated values
        will always fall between them.

        In case of lognormal distributions, the **ECD Mean** and
        **ECD Std** are, respectively, the mean and standard deviation
        of the corresponding distributions.

        In all distribution cases, sampled values are scaled such that their sum equals the volume
        of the domain or box provided.
        """
    )


def algo_help_text() -> ui.HTML:
    return ui.markdown(
        f"""
        Seeds are the initial locations or positions of the grains or cells.
        These are needed for the generator to run.

        There are two ways to initialize seeds: **random** and **upload**.

        The random initialization will generate seeds (uniformly) randomly in the
        domain ([0, Length), [0, Breadth), [0, Height)). If you want the seeds to be reproducible,
        give a positive integer in the **Seeds random state** box.

        If seeds are uploaded, then these will be used in generating the miscrostructure instead.
        The csv or txt file **must only contain** column names as {list(COORDINATES)[:2]} for 2D case and
        {list(COORDINATES)} for 3D case. The columns are the coordinates of the seeds. All
        values must be float.

        **Volume tolerance**: relative percentage error for volumes (for Laguerre diagram).

        **Lloyd iterations**: number of iterations of Lloyd's algorithm (move each seed to the
        centroid of its cell).

        **Damp parameter**: the damping parametr of the damped Lloyd step; value must be between
        0 and 1 (inclusive).
        """
    )


def feedback_text() -> ui.Tag:
    return ui.card(
        ui.card_header("We'd love your feedback!"),
        ui.markdown(
            """
            If you encounter any bugs or have suggestions, please
            click [here](https://github.com/synthetic-microstructures/synthetmic-gui/issues)
            to report them on app's GitHub. Thank you for helping us improve the app!

            Check out the source code [here](https://github.com/synthetic-microstructures/synthetmic-gui)
            on GitHub and see what's under the hood!
            """
        ),
    )


def about_text(app_name: str) -> ui.HTML:
    return ui.markdown(
        f"""
        #### About

        **{app_name}** is a web app for generating 2D and 3D synthetic
        polycrystalline microstructures using Laguerre tessellations.
        It uses the fast algorithms (developed in this
        [paper](https://www.tandfonline.com/doi/full/10.1080/14786435.2020.1790053))
        for generating grains of prescribed volumes using optimal transport theory.
        It is built on top of [SynthetMic](https://github.com/synthetic-microstructures/synthetmic)
        and [pysdot](https://github.com/sd-ot/pysdot) packages.
        """
    )


def info_modal(diagram_id: str) -> None:
    ui.modal_show(
        ui.modal(
            ui.markdown(f"### {APP_NAME}"),
            ui.hr(),
            about_text(APP_NAME),
            comps.create_selection(
                id=diagram_id,
                label="Choose a type of microstructure",
                choices=[d for d in DiagramType],
                selected=DiagramType.LAGUERRE,
            ),
            size="m",
            easy_close=False,
            footer=ui.modal_button(
                "Close",
                class_="btn btn-primary",
            ),
        )
    )


def create_periodic_input(ids: list[str], labels: list[str]) -> ui.Tag:
    width = 12 // len(ids)
    cols = [
        ui.column(
            width,
            ui.input_switch(
                id=id,
                label=label,
            ),
        )
        for id, label in zip(ids, labels)
    ]

    return ui.row(*cols)


def invalid_input_text() -> str:
    return "Invalid inputs. Please check your inputs and try again."
