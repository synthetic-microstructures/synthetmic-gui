from typing import Any

import faicons as fa
from shiny import ui

import shared.controls as ct
from shared.utils import COORDINATES, VOLUMES


def create_selection(
    id: str,
    label: ui.TagChild,
    choices: list[Any],
    selected: Any,
) -> ui.Tag:
    return ui.input_select(
        id=id,
        label=label,
        choices=choices,
        selected=selected,
        selectize=False,
        remove_button=False,
        multiple=False,
    )


def create_dist_selection(
    id: str, label: str = "Choose a volume distribution"
) -> ui.Tag:
    return create_selection(
        id=id,
        label=label,
        choices=[d for d in ct.Distribution],
        selected=ct.Distribution.CONSTANT,
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

        In case of normal and lognormal distributions, the **Mean** and
        **Std** are, respectively, the mean and standard deviation
        of the corresponding distributions.

        In all distribution cases, sampled values are scaled such that their sum equals the volume
        of the domain or box provided.
        """
    )


def algo_help_text() -> ui.HTML:
    return ui.markdown(
        f"""
        Seeds are the initial locations or positions of the Laguerre cells.
        These are needed for the generator to run.

        There are two ways to initialize seeds: **random** and **upload**.

        The random initialization will generate seeds (uniformly) randomly in the
        domain ([0, Length), [0, Breadth), [0, Height)). If you want the seeds to be reproducible,
        give a positive integer in the **Seeds random state** box.

        If seeds are uploaded, then these will be used in generating the miscrostructure instead.
        The csv or txt file **must only contain** column names as {list(COORDINATES)[:2]} for 2D case and
        {list(COORDINATES)} for 3D case. The columns are the coordinates of the seeds. All
        values must be float.
        
        **Volume tolerance**: relative percentage error for volumes.

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
            If you encounter any bugs or have suggestions, please click [here](https://github.com/synthetic-microstructures/synthetmic-gui/issues)
            to report them on app's GitHub. Thank you for helping us improve the app!

            Check out the source code [here](https://github.com/synthetic-microstructures/synthetmic-gui) on GitHub and see what's under the hood!
            """
        ),
    )


def about_text() -> ui.HTML:
    return ui.markdown(
        """
        #### About

        **SynthetMic-GUI** is a web app for generating 2D and 3D synthetic polycrystalline microstructures using Laguerre tessellations.
        It uses the fast algorithms (developed in this [paper](https://www.tandfonline.com/doi/full/10.1080/14786435.2020.1790053))
        for generating grains of prescribed volumes using optimal transport theory. It is built on
        top of [SynthetMic](https://github.com/synthetic-microstructures/synthetmic) and [pysdot](https://github.com/sd-ot/pysdot) packages.
        """
    )


def usage_text() -> ui.HTML:
    return ui.markdown(
        """
        #### Usage

        Using the app is extremely easy. It can be done in **4 steps**:
        1. Specify both the space and box dimension. You can optionally choose whether the underlying domain should be periodic in all directions.
        1. Specify the number of grains and how target volumes will be generated. We support single and dual phase volumes specification. You
        can also upload your custom target volumes instead.
        1. Click on **Generate miscrostructure** button to generate synthetic miscrostructure.
        1. Click on any of the **download buttons** to either download the generated diagram (in different formats!) or the diagram properties
        (like centroids, vertices, etc; also in differnt formats!).

        That is it!

        Enjoy generating microstructures!
        """
    )


def how_text() -> ui.Tag:
    return ui.accordion(
        ui.accordion_panel(
            "Need help? Read how to use SynthetMic-GUI",
            about_text(),
            ui.hr(),
            usage_text(),
            icon=fa.icon_svg("lightbulb", fill=ct.FILL_COLOUR),
        ),
        open=False,
    )


def group_ui_elements(
    *args, title: ui.Tag | str, help_text: ui.Tag | str | ui.HTML
) -> ui.Tag:
    return ui.card(
        ui.card_header(
            title,
            ui.popover(
                ui.span(
                    fa.icon_svg("circle-info", fill=ct.FILL_COLOUR),
                    style="position:absolute; top: 5px; right: 7px;",
                ),
                help_text,
            ),
        ),
        *args,
    )


def create_dist_param(dist: str, id_prefix: str) -> ui.Tag:
    text = f"{dist} distribution selected;"
    match dist:
        case ct.Distribution.CONSTANT:
            return ui.help_text(f"{text} all volumes will be equal for this phase.")

        case ct.Distribution.UNIFORM:
            return ui.tags.div(
                create_numeric_input(
                    ids=[f"{id_prefix}_{p}" for p in ("low", "high")],
                    labels=["Low", "High"],
                    defaults=[1, 2],
                ),
                ui.help_text(
                    f"{text} volumes will be distibuted uniformly in [Low, High)."
                ),
            )

        case ct.Distribution.LOGNORMAL:
            return ui.tags.div(
                create_numeric_input(
                    ids=[f"{id_prefix}_{p}" for p in ("mean", "std")],
                    labels=["Mean", "Std"],
                    defaults=[1, 0.35],
                ),
                ui.help_text(
                    f"{text} volumes will be distibuted lorgnormally in with  mean 'Mean' standard deviation 'Std'."
                ),
            )

        case _:
            raise ValueError(
                f"Mismatch dist: {dist}. Input must be one of {', '.join(ct.Distribution)}."
            )


def create_upload_handler(id: str, label: str) -> ui.Tag:
    return ui.input_file(
        id,
        label,
        accept=[".csv", ".txt"],
        multiple=False,
    )


def info_modal() -> None:
    ui.modal_show(
        ui.modal(
            ui.markdown("### SynthetMic-GUI"),
            ui.hr(),
            about_text(),
            ui.hr(),
            usage_text(),
            size="l",
            easy_close=True,
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
