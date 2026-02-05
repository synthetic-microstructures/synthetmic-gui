from dataclasses import asdict

import numpy as np
import pandas as pd
from shiny import Inputs, Outputs, Session, module, reactive, render, ui
from shiny.types import FileInfo
from shiny_validate import InputValidator
from synthetmic import VoronoiDiagramGenerator
from synthetmic.data.utils import SynthetMicData, sample_random_seeds

from modules import common
from shared import comps, utils, views
from shared.consts import Dimension, SeedInitializer


@module.ui
def ui_() -> ui.Tag:
    sidebar = comps.create_sidebar(
        ui.tags.div(
            views.create_help("help"),
            comps.group_ui_elements(
                comps.create_selection(
                    id="dim",
                    label="Choose a dimension",
                    choices=[d for d in Dimension],
                    selected=Dimension.THREE_D,
                ),
                ui.output_ui("box_dim"),
                ui.output_ui("periodicity"),
                title="Box dimension",
                help_text=views.box_help_text(),
            ),
            comps.group_ui_elements(
                ui.input_numeric(
                    id="n_grains",
                    label="Number of grains",
                    value=1000,
                ),
                title="Grains",
                help_text="Number of grains or cells in the Voronoi diagram.",
            ),
            comps.group_ui_elements(
                comps.create_selection(
                    id="seeds_init",
                    label="Choose how seeds are initialized",
                    choices=[i for i in SeedInitializer],
                    selected=SeedInitializer.RANDOM,
                ),
                ui.panel_conditional(
                    f"input.seeds_init === '{SeedInitializer.RANDOM}'",
                    ui.tags.div(
                        ui.input_numeric(
                            id="random_state",
                            label="Seeds random state",
                            value=None,
                        ),
                        ui.help_text(
                            "Seeds will be randomly generated in the specified box above."
                        ),
                    ),
                ),
                ui.panel_conditional(
                    f"input.seeds_init === '{SeedInitializer.UPLOAD}'",
                    ui.tags.div(
                        comps.create_upload_handler(
                            "uploaded_seeds",
                            "Uplaod seeds as a csv or txt file",
                        ),
                        ui.output_ui("uploaded_seeds_summary"),
                    ),
                ),
                views.create_numeric_input(
                    ["n_iter", "damp_param"],
                    ["Lloyd iterations", "Damp param"],
                    [5, 1.0],
                ),
                title="Algorithm",
                help_text=views.algo_help_text(),
            ),
            comps.create_input_task_button(
                id="generate",
                label="Generate microstructure",
                icon="person-running",
            ),
            ui.input_dark_mode(mode="light"),
            views.feedback_text(),
            views.app_version_text(),
        ),
        id="sidebar",
    )

    return comps.create_page_sidebar(common.page_ui("common"), sidebar=sidebar)


@module.server
def server(input: Inputs, output: Outputs, session: Session) -> None:
    _uploaded_seeds = reactive.Value(value=None)
    iv = InputValidator()
    iv.add_rule("length", utils.req_gt(rhs=0))
    iv.add_rule("breadth", utils.req_gt(rhs=0))
    iv.add_rule("damp_param", utils.req_between(left=0.0, right=1.0))
    iv.add_rule("n_iter", utils.req_int_gte(rhs=0))
    iv.add_rule("n_grains", utils.req_int_gt(rhs=0))

    @reactive.effect
    def _() -> None:
        file: list[FileInfo] | None = input.uploaded_seeds()
        if file is not None:
            _uploaded_seeds.set(pd.read_csv(file[0]["datapath"]))

    @reactive.effect
    @reactive.event(input.help)
    def _() -> None:
        views.voronoi_help()

    @render.ui
    def box_dim() -> ui.Tag:
        return views.box_dim(input.dim())

    @render.ui
    def periodicity() -> ui.Tag:
        return views.periodicity(input.dim())

    @render.table
    def seeds_summary_table() -> pd.DataFrame:
        return utils.summarize_df(_uploaded_seeds())

    @render.ui
    def uploaded_seeds_summary() -> ui.Tag:
        return views.seeds_summary(_uploaded_seeds(), "seeds_summary_table")

    @reactive.calc
    @reactive.event(input.generate)
    def _fit() -> tuple[SynthetMicData, VoronoiDiagramGenerator] | Exception:
        if input.dim() == Dimension.THREE_D:
            iv.add_rule("height", utils.req_gt(rhs=0))

        iv.enable()
        if not iv.is_valid():
            return Exception(
                "Invalid inputs. Please check all fields for the required values."
            )

        box_dim = utils.parse_box_dim(
            utils.Box(
                length=input.length(), breadth=input.breadth(), height=input.height()
            ),
            dim=input.dim(),
        )
        space_dim = len(box_dim)
        domain = np.array([[0, d] for d in box_dim])
        periodic = utils.parse_periodicity(
            utils.Periodic(
                is_x_periodic=input.is_x_periodic(),
                is_y_periodic=input.is_y_periodic(),
                is_z_periodic=input.is_z_periodic(),
            ),
            dim=input.dim(),
        )

        match input.seeds_init():
            case SeedInitializer.RANDOM:
                iv.add_rule("random_state", utils.int_gte(rhs=0, allow_none=True))
                iv.enable()

                if not iv.is_valid():
                    return Exception(
                        "Invalid random seed. Please check your input and try again."
                    )

                seeds = sample_random_seeds(
                    domain=domain,
                    n_grains=input.n_grains(),
                    random_state=input.random_state(),
                )

            case SeedInitializer.UPLOAD:
                if _uploaded_seeds() is None:
                    return Exception("Seeds not uploaded. Upload seeds and try again.")

                val_out = utils.validate_df(
                    _uploaded_seeds(),
                    expected_colnames=list(utils.COORDINATES)[:space_dim],
                    expected_dim=(input.n_grains(), space_dim),
                    expected_type="float",
                    file="seeds",
                    bounds=dict(zip(utils.COORDINATES[:space_dim], domain)),
                )
                if isinstance(val_out, str):
                    return Exception(val_out)

                seeds = _uploaded_seeds()

            case _:
                return Exception(
                    f"Mismatch seed initializer: {input.seeds_init()}. Input must be one of {', '.join(SeedInitializer)}."
                )

        return utils.fit(
            domain=domain,
            seeds=seeds,
            volumes=None,
            periodic=list(periodic),
            tol=None,
            n_iter=input.n_iter(),
            damp_param=float(input.damp_param()),
        )

    @reactive.effect
    @reactive.event(input.generate)
    async def _():
        await session.send_custom_message(
            "ga_event",
            asdict(
                utils.Event(
                    name="voronoi_generate",
                    category="button",
                    label="generate voronoi diagram",
                )
            ),
        )

    @reactive.effect
    @reactive.event(input.generate)
    def _():
        f = _fit()
        if isinstance(f, Exception):
            comps.create_error_notification(str(f))

        else:
            ui.update_sidebar(id="sidebar", show=False)
            common.server("common", f, input.generate)
