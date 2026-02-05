from typing import Any

import pandas as pd
from shiny import ui

from shared import comps
from shared.consts import (
    APP_NAME,
    APP_VERSION,
    Dimension,
    Distribution,
    ExampleDataName,
    PropertyExtension,
)
from shared.utils import COORDINATES, VOLUMES, format_to_standard_form, qp


def box_dim(dim: str) -> ui.Tag:
    ids = ("length", "breadth")
    labels = tuple([id.title() for id in ids])
    defaults = (1.0, 1.0)
    if dim == Dimension.THREE_D:
        ids += ("height",)
        labels += ("Height",)
        defaults += (1.0,)

    return create_numeric_input(ids, labels, defaults)


def periodicity(dim: str) -> ui.Tag:
    space_dim = 2 if dim == Dimension.TWO_D else 3
    ids = [f"is_{c}_periodic" for c in COORDINATES[:space_dim]]
    labels = [f"{c}-coordinate" for c in COORDINATES[:space_dim]]

    return ui.tags.div(
        ui.markdown("Periodicity"),
        create_periodic_input(ids, labels),
    )


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


def voronoi_help() -> None:
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


def laguerre_help(
    data_selection_id: str, data_extension_id: str, data_card_id: str
) -> None:
    ui.modal_show(
        ui.modal(
            ui.markdown(f"### {APP_NAME}"),
            ui.hr(),
            ui.markdown(
                """
                #### Usage

                Using the app is extremely easy. It can be done in **4 steps**:
                1. Specify both the space and box dimension. You can optionally
                choose whether the underlying domain should be periodic in all directions.
                1. Specify the number of grains and how target volumes will be generated.
                We support single and dual phase volumes specification. You
                can also upload your custom target volumes instead.
                1. Click on **Generate miscrostructure** button to generate synthetic miscrostructure.
                1. Click on any of the **download buttons** to either download
                the generated diagram (in different formats!) or the diagram properties
                (like centroids, vertices, etc; also in differnt formats!).

                That is it!

                Enjoy generating microstructures!
                """
            ),
            ui.hr(),
            ui.markdown("#### Starting point"),
            ui.markdown(
                """
                Don't know where to start yet?

                Don't worry, we've got you covered!

                Select and download one of our example data below. You can then upload them
                in the main app to generate the corresponding microstructure. When you click the 
                download button, the seeds, volumes and domain of the underlying microstructure will be
                downloaded in the selected format, which are zipped. Unzip to access the files.

                Ensure you enter the correct dimension in the 'Box dimension' input. You can read this from 
                the dimension.txt or dimension.csv file.
                """
            ),
            ui.row(
                ui.column(
                    4,
                    ui.card(
                        ui.card_header("Options center"),
                        comps.create_selection(
                            id=data_selection_id,
                            label="Choose an example data to download",
                            choices=[e for e in ExampleDataName],
                            selected=ExampleDataName.BASIC,
                            width="100%",
                        ),
                        comps.create_selection(
                            id=data_extension_id,
                            label="Choose a file extension for the example data",
                            choices=[e for e in PropertyExtension],
                            selected=PropertyExtension.CSV,
                            width="100%",
                        ),
                    ),
                ),
                ui.column(
                    8,
                    ui.output_ui(data_card_id),
                ),
            ),
            size="l",
            easy_close=True,
            footer=ui.modal_button(
                "Close",
                class_="btn btn-primary",
            ),
        )
    )


def create_dist_param(
    dist: str,
    id_prefix: str,
) -> ui.Tag:
    text = f"{dist} distribution selected;"
    match dist:
        case Distribution.CONSTANT:
            return ui.help_text(f"{text} all volumes will be equal for this phase.")

        case Distribution.UNIFORM:
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

        case Distribution.LOGNORMAL:
            return ui.tags.div(
                create_numeric_input(
                    ids=[f"{id_prefix}_{p}" for p in ("mean", "std")],
                    labels=["ECD Mean", "ECD Std"],
                    defaults=[1, 0.35],
                ),
                ui.help_text(
                    f"""{text} Equivalent Circle Diameters (ECDs) will be sampled from
                    a lorgnormal distribution with mean 'ECD mean' and standard deviation 'ECD std'.
                    """
                ),
            )

        case _:
            raise ValueError(
                f"Mismatch dist: {dist}. Input must be one of {', '.join(Distribution)}."
            )


def info_modal() -> None:
    ui.modal_show(
        ui.modal(
            ui.markdown(f"### {APP_NAME}"),
            ui.hr(),
            about_text(APP_NAME),
            size="m",
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


def create_example_data_card(
    name: str,
    image_id: str,
    download_id: str,
) -> ui.Tag:
    common_tags = """
        volume tolerance=1.0%, Lloyd iterations=20, damp param=1.0, 
        colorby=fitted volumes, colormap=plasma.
        """

    match name:
        case ExampleDataName.BASIC:
            info = """
                This is an example of a basic synthetic microstructure with a 
                random seed initialization and constant volume distribution.
                """
            tags = "2D, random seed, constant volumes, "

        case ExampleDataName.RANDOM:
            info = """
                Random distribution of initial seed locations. Here the initial
                generator locations of the large and small grains are uniformly
                distributed over the corresponding domain.
                """
            tags = "2D, dual-phase volumes, random seeds, "

        case ExampleDataName.BANDED:
            info = """
                Banded distribution of initial seed locations.
                Here, the different sized grains have initial generator
                locations that lie inside bands within the domain.
                The sizes of the bands have been chosen so that there are
                approximately equal numbers of small grains within each
                small-grain band and approximately equal numbers of large grains
                within each large-grain band.
                """
            tags = "2D, dual-phase volumes, banded seeds, "

        case ExampleDataName.CLUSTERED:
            info = """
                Clustered distribution of initial seed locations. 
                Here, the smaller grains have initial generator locations
                that lie inside non-overlapping discs.
                """
            tags = "2D, dual-phase volumes, clustered seeds, "

        case ExampleDataName.MIXED:
            info = """
                A mixed distribution: the initial generators are
                such that the larger grains are arranged in bands and
                the smaller grains are a combination of the banded and random distributions.
                """
            tags = "2D, dual-phase volumes, mixed seeds, "

        case ExampleDataName.INCREASING:
            info = """
                The initial seed locations are distributed such that the 𝑥-coordinate
                increases with grain size.
                """
            tags = "2D, multi-phase volumes, increasing grain size, "

        case ExampleDataName.MIDDLE:
            info = """
                The initial seed locations are distributed such that
                the larger grains are found in the middle of the domain.
                """
            tags = "2D, multi-phase volumes, divergent grain size, "

        case ExampleDataName.DP:
            info = """
                An RVE of a dual-phase material with a banded microstructure.
                """
            tags = "3D, RVE, dual-phase, banded structure, "

        case ExampleDataName.LOGNORMAL:
            info = """
                An RVE in which the grain volumes have approximately lognormal distribution
                The coefficient of variation of the volumes (the ratio of the standard
                deviation to the mean) is 1.4.
                """
            tags = "3D, RVE, lognormal volumes, banded structure, "

        case ExampleDataName.EBSD:
            info = """
            In this example we fit a Laguerre diagram to an EBSD image of a
            single-phase steel. The 'target volumes'
            are the areas of the grains in the EBSD image. The 'seeds' are the centroids of the grains
            in the EBSD image. The EBSD data is taken from this [paper](https://doi.org/10.1051/m2an/2025004).
            """
            tags = "2D, non-periodic, volume upload, seed upload, volume tolerance = 1, Lloyd iterations = 0, damp param = 1"

        case _:
            raise ValueError(
                f"Invalid data name '{name}'; name must be one of [{', '.join(ExampleDataName)}]."
            )

    return ui.card(
        ui.card_header(name),
        ui.output_image(image_id, fill=True, height="100%", width="100%"),
        ui.markdown(info),
        ui.help_text(
            f"Tags: {tags}{'' if name == ExampleDataName.EBSD else common_tags}"
        ),
        comps.create_download_button(
            id=download_id,
            label="Download data",
        ),
    )


def compute_d90_text(mean: float, std: float) -> str | None:
    text = """
            D90 for the lognormal distribution of ECDs with mean {}
            and std {} is {}. 
            """
    if any([i is None for i in (mean, std)]) or mean == 0:
        return

    d90 = qp(mean=mean, std=std, p=0.9)
    d90 = format_to_standard_form(d90, precision=2)

    return text.format(mean, std, d90)
