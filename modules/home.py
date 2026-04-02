from pathlib import Path
from typing import Generator

from shiny import Inputs, Outputs, Session, module, render, ui
from shiny.types import ImgData

from shared import comps, utils, views
from shared.consts import (
    ALGO_PAPER_LINK,
    APP_AUTHORS,
    APP_LINK,
    APP_NAME,
    APP_VERSION,
    HW_LINK,
    MACS_LINK,
    MDICE_LINK,
    MS_LINK,
    UOG_LINK,
    ExampleDataName,
    PropertyExtension,
)


def citation_text() -> ui.TagList:
    return ui.TagList(
        ui.tags.h4("Citation"),
        ui.markdown(
            f"""
            If you use this app in any publication or report, please cite it as follows:

            > {APP_AUTHORS} (2025). *{APP_NAME} ({APP_VERSION})* [Software]. Avaialable from {APP_LINK}.
            """
        ),
    )


def affliation_text() -> ui.TagList:
    return ui.TagList(
        ui.tags.h4("About the research team"),
        ui.row(
            *[
                ui.column(
                    width,
                    comps.anchor_tag(
                        ui.tags.img(
                            src=str(Path("static", "img", img)),
                            height="auto" if link == UOG_LINK else "130px",
                            width="auto",
                        ),
                        href=link,
                    ),
                    offset=0,
                )
                for width, img, link in zip(
                    (4, 3, 5),
                    (
                        "MDICE_logo.png",
                        "HWU_logo.jpg",
                        "UOG_logo.png",
                    ),
                    (MDICE_LINK, HW_LINK, UOG_LINK),
                )
            ],
            style="align-items: flex-end;",
        ),
        ui.tags.br(),
        ui.markdown(
            f"""
            This app has been developed by researchers in the
            {comps.anchor_html("School of Mathematical and Computer Sciences", MACS_LINK)}
            at {comps.anchor_html("Heriot-Watt University", HW_LINK)}, and the
            {comps.anchor_html("School of Mathematics and Statistics", MS_LINK)}
            at the {comps.anchor_html("University of Glasgow", UOG_LINK)}, with technical support from the
            {comps.anchor_html("Mathematical-Driven Innovation Center (M-DICE)", MDICE_LINK)}.

            M-DICE is where cutting-edge Mathematics and Statistics expertise drives industrial innovation,
            addresses real-world challenges, and fosters interdisciplinary research.
            To build a powerful tool like this app, please get in touch using this email: m-dice@hw.ac.uk.
            """
        ),
    )


def feedback_text() -> ui.TagList:
    return ui.TagList(
        ui.tags.h4("We'd love to hear from you!"),
        ui.tags.div(
            ui.markdown(
                f"""
            If you encounter any bug or have suggestions, please
            click {comps.anchor_html("here", "https://github.com/synthetic-microstructures/synthetmic-gui/issues")}
            to report them on our hub (if you have a GitHub account).

            Alternatively, you can contact any of the app maintainers below:

            - Rasheed Ibraheem (R.Ibraheem@hw.ac.uk)
            - David Bourne (D.Bourne@hw.ac.uk)
            - Steven Roper (Steven.Roper@glasgow.ac.uk)

            Check out the source code {comps.anchor_html("here", "https://github.com/synthetic-microstructures/synthetmic-gui")}
            on GitHub and see what's under the hood. If you like what we do, please leave us a star!

            Thank you for helping us improve the app!
            """
            ),
        ),
    )


def about_text() -> ui.TagList:
    return ui.TagList(
        ui.tags.h4("About"),
        ui.markdown(
            f"""
        **{APP_NAME}** is a web app for generating 2D and 3D synthetic
        polycrystalline microstructures using Voronoi and Laguerre tessellations.
        It uses the fast algorithms (developed in this
        {comps.anchor_html("paper", ALGO_PAPER_LINK)})
        for generating grains of prescribed volumes using optimal transport theory.
        It is built on top of the {comps.anchor_html("SynthetMic", "https://github.com/synthetic-microstructures/synthetmic")}
        and {comps.anchor_html("pysdot", "https://github.com/sd-ot/pysdot")} packages.
        """
        ),
    )


def usage_text() -> ui.TagList:
    return ui.TagList(
        ui.tags.h4("Usage"),
        ui.markdown(
            """
        Using the app is extremely easy. It can be done in **5 steps**:
        1. Click on the **Microstructure generation** tab above to begin.
        1. Specify the space dimension and box size. You can optionally
        choose whether the underlying domain is periodic in a given coordinate.
        1. Specify the number of grains and how the target volumes will be generated.
        We support single- and dual-phase microstructures. You
        can also upload your custom target volumes instead.
        1. Click on the **Generate microstructure** button.
        1. Click on the **download buttons** to download
        the generated diagram and the diagram properties
        (like centroids, vertices, etc.).

        That is it! Enjoy generating microstructures!"""
        ),
    )


def starting_point_text() -> ui.TagList:
    return ui.TagList(
        ui.h4("Upload option: Examples"),
        ui.markdown(
            """
        Here we provide some examples to showcase how to use the **upload** options of the app.
        This is useful when you want to upload your own target volumes and/or seeds to have more
        control on the generated microstructures.

        Select and download one of our example data sets below. You can then upload the data
        in the main app to generate the corresponding microstructure. When you click the 
        download button, the seeds, volumes and domain of the underlying microstructure will be
        downloaded in the selected format, which are zipped. Unzip to access the files.

        Ensure you enter the correct dimension in the **Box dimension** input.
        You can read this from the *dimension.txt* or *dimension.csv* file."""
        ),
    )


def create_example_data_card(
    name: str,
    image_id: str,
) -> ui.Tag:
    common_tags = """
        volume tolerance=1.0%, Lloyd iterations=20, damp param=1.0, 
        colorby=fitted volumes, colormap=plasma.
        """

    register = {
        ExampleDataName.BASIC: dict(
            info=f"""
                This is an example of a basic synthetic microstructure with a 
                random seed initialization and constant volume distribution.
                For more information, see Figure 2 of this
                {comps.anchor_html("paper", ALGO_PAPER_LINK)}.
                """,
            tags="Laguerre, 2D, random seed, constant volumes, ",
        ),
        ExampleDataName.RANDOM: dict(
            info=f"""
                Random distribution of initial seed locations. Here the initial
                generator locations of the large and small grains are uniformly
                distributed over the corresponding domain. For more information, see
                Figure 4 of this {comps.anchor_html("paper", ALGO_PAPER_LINK)}.
                """,
            tags="Laguerre, 2D, dual-phase volumes, random seeds, ",
        ),
        ExampleDataName.BANDED: dict(
            info=f"""
                Banded distribution of initial seed locations.
                Here, the different sized grains have initial generator
                locations that lie inside bands within the domain.
                The sizes of the bands have been chosen so that there are
                approximately equal numbers of small grains within each
                small-grain band and approximately equal numbers of large grains
                within each large-grain band. For more information, see
                Figure 4 of this {comps.anchor_html("paper", ALGO_PAPER_LINK)}.
                """,
            tags="Laguerre, 2D, dual-phase volumes, banded seeds, ",
        ),
        ExampleDataName.CLUSTERED: dict(
            info=f"""
                Clustered distribution of initial seed locations. 
                Here, the smaller grains have initial generator locations
                that lie inside non-overlapping discs. For more information, see
                Figure 4 of this {comps.anchor_html("paper", ALGO_PAPER_LINK)}.
                """,
            tags="Laguerre, 2D, dual-phase volumes, clustered seeds, ",
        ),
        ExampleDataName.MIXED: dict(
            info=f"""
                A mixed distribution: the initial generators are
                such that the larger grains are arranged in bands and
                the smaller grains are a combination of the banded and random distributions.
                For more information, see Figure 4 of this
                {comps.anchor_html("paper", ALGO_PAPER_LINK)}.
                """,
            tags="Laguerre, 2D, dual-phase volumes, mixed seeds, ",
        ),
        ExampleDataName.INCREASING: dict(
            info=f"""
                The initial seed locations are distributed such that the 𝑥-coordinate
                increases with grain size. For more information, see Figure 5 of this
                {comps.anchor_html("paper", ALGO_PAPER_LINK)}.
                """,
            tags="Laguerre, 2D, multi-phase volumes, increasing grain size, ",
        ),
        ExampleDataName.MIDDLE: dict(
            info=f"""
                The initial seed locations are distributed such that
                the larger grains are found in the middle of the domain.
                For more information, see Figure 5 of this
                {comps.anchor_html("paper", ALGO_PAPER_LINK)}.
                """,
            tags="Laguerre, 2D, multi-phase volumes, divergent grain size, ",
        ),
        ExampleDataName.DUAL_PHASE: dict(
            info=f"""
                An RVE of a dual-phase material with a banded microstructure.
                For more information, see Figure 5.4 of this
                {comps.anchor_html("paper", ALGO_PAPER_LINK)}.
                """,
            tags="Laguerre, 3D, RVE, dual-phase, banded structure, ",
        ),
        ExampleDataName.LOGNORMAL: dict(
            info=f"""
                An RVE in which the grain volumes have an approximately lognormal distribution.
                The coefficient of variation of the volumes (the ratio of the standard
                deviation to the mean) is 1.4.
                For more information, see Example 5.5 of this
                {comps.anchor_html("paper", ALGO_PAPER_LINK)}
                and this {
                comps.anchor_html(
                    "notebook",
                    "https://github.com/synthetic-microstructures/synthetmic/blob/main/examples/notebooks/laguerre/Lognormal_microstructure.ipynb",
                )
            }.
                """,
            tags="Laguerre, 3D, RVE, lognormal volumes, ",
        ),
        ExampleDataName.EBSD: dict(
            info=f"""
            In this example we fit a Laguerre diagram to an EBSD image of a
            single-phase steel. The 'target volumes'
            are the areas of the grains in the EBSD image. The 'seeds' are the
            centroids of the grains in the EBSD image. The EBSD data is taken
            from this {comps.anchor_html("paper", "https://doi.org/10.1051/m2an/2025004")}.
            """,
            tags="Laguerre, 2D, non-periodic, volume upload, seed upload, volume tolerance = 1, Lloyd iterations = 0, damp param = 1",
        ),
        ExampleDataName.BANDED_PERIODIC: dict(
            info=f"""
            This is an example of a banded periodic microstructure. The diagram is
            periodic in one direction, the volumes within each band are drawn from a
            log-normal distribution. There are 8000 grains in the central band and
            1000 grains in each of the other bands. For more information,
            see this {
                comps.anchor_html(
                    "notebook",
                    "https://github.com/synthetic-microstructures/synthetmic/blob/main/examples/notebooks/laguerre/Banded_periodic.ipynb",
                )
            }.
            """,
            tags="""3D, Laguerre, x-periodic, seed upload, volume upload, banded
            structure, lognormal, """,
        ),
    }

    return ui.card(
        ui.card_header(f"Selected example: {name}"),
        ui.output_image(image_id, fill=True, height="100%", width="100%"),
        ui.markdown(register[name]["info"]),
        ui.help_text(
            f"Tags: {register[name]['tags']}{'' if name == ExampleDataName.EBSD else common_tags}"
        ),
    )


@module.ui
def ui_() -> ui.Tag:
    return ui.tags.div(
        ui.tags.h3(APP_NAME),
        ui.markdown(
            "An interactive GUI for generating synthetic polycrystalline microstructures."
        ),
        ui.tags.hr(),
        about_text(),
        ui.tags.hr(),
        usage_text(),
        ui.tags.hr(),
        starting_point_text(),
        ui.row(
            ui.column(
                4,
                ui.card(
                    ui.card_header("Options"),
                    comps.selection(
                        id="example_data",
                        label="Choose an example data to download",
                        choices=[e for e in ExampleDataName],
                        selected=ExampleDataName.EBSD,
                        width="100%",
                    ),
                    comps.selection(
                        id="example_data_extension",
                        label="Choose a file extension for the example data",
                        choices=[e for e in PropertyExtension],
                        selected=PropertyExtension.CSV,
                        width="100%",
                    ),
                    comps.download_button(
                        id="download_example_data",
                        label="Download data",
                    ),
                ),
            ),
            ui.column(
                8,
                ui.output_ui("example_data_card"),
            ),
        ),
        ui.tags.hr(),
        affliation_text(),
        ui.tags.hr(),
        citation_text(),
        ui.tags.hr(),
        feedback_text(),
        ui.tags.hr(),
        views.app_version_text(),
        class_="container-sm",
        style="max-width: 1000px; overflow: auto;",
    )


@module.server
def server(
    input: Inputs,
    output: Outputs,
    session: Session,
) -> None:
    @render.download(
        filename=lambda: f"synthetmic-gui-example-{input.example_data()}.zip",
        media_type="application/zip",
    )
    def download_example_data() -> Generator[bytes, None, None]:
        yield utils.create_example_data_bytes(
            name=input.example_data(), file_extension=input.example_data_extension()
        )

    @render.image
    def example_data_image() -> ImgData:
        return utils.load_image(
            Path().resolve() / "assets" / "imgs" / f"{input.example_data()}.png",
        )

    @render.ui
    def example_data_card() -> ui.Tag:
        return create_example_data_card(
            name=input.example_data(),
            image_id="example_data_image",
        )
