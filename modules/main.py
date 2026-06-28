from dataclasses import asdict

import numpy as np
import pandas as pd
from shiny import Inputs, Outputs, Session, module, reactive, render, ui
from shiny.types import FileInfo
from shiny_validate import InputValidator
from synthetmic.data.utils import SynthetMicData, sample_random_seeds

from modules import common, laguerre, voronoi
from shared import comps, utils, views
from shared.consts import (
    DiagramType,
    Dimension,
    Distribution,
    Phase,
    SeedInitializer,
)


@module.ui
def ui_():
    sidebar = comps.sidebar(
        ui.div(
            comps.group_ui_elements(
                comps.selection(
                    id="diagram",
                    label="Choose a diagram type",
                    choices=list(DiagramType),
                    selected=DiagramType.LAGUERRE,
                ),
                title="Diagram",
                help_text="Your choice here will determine available options and inputs.",
            ),
            comps.group_ui_elements(
                comps.selection(
                    id="dim",
                    label="Choose a dimension",
                    choices=list(Dimension),
                    selected=Dimension.THREE_D,
                ),
                ui.output_ui("box_dim"),
                ui.output_ui("periodicity"),
                title="Box dimension",
                help_text=views.box_help_text(),
            ),
            ui.panel_conditional(
                f"input.diagram === '{DiagramType.LAGUERRE}'", laguerre.sidebar()
            ),
            ui.panel_conditional(
                f"input.diagram === '{DiagramType.VORONOI}'", voronoi.sidebar()
            ),
            comps.group_ui_elements(
                comps.selection(
                    id="seeds_init",
                    label="Choose how seeds are initialized",
                    choices=list(SeedInitializer),
                    selected=SeedInitializer.RANDOM,
                ),
                ui.panel_conditional(
                    f"input.seeds_init === '{SeedInitializer.RANDOM}'",
                    ui.div(
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
                    ui.div(
                        comps.upload_handler(
                            "uploaded_seeds",
                            "Uplaod seeds as a csv or txt file",
                        ),
                        ui.output_ui("uploaded_seeds_summary"),
                    ),
                ),
                ui.output_ui("algorithm"),
                title="Algorithm",
                help_text=views.algo_help_text(),
            ),
            comps.input_task_button(
                id="generate",
                label="Generate microstructure",
                icon="arrow-right",
            ),
            ui.p(),
            views.app_version_text(),
        ),
        id="sidebar",
    )
    return comps.page_sidebar(common.ui_("common"), sidebar=sidebar)


@module.server
def server(input: Inputs, output: Outputs, session: Session) -> None:
    iv = InputValidator()
    iv.add_rule("length", utils.req_gt(rhs=0))
    iv.add_rule("breadth", utils.req_gt(rhs=0))
    iv.add_rule("damp_param", utils.req_between(left=0.0, right=1.0))
    iv.add_rule("n_iter", utils.req_int_gte(rhs=0))

    _uploaded_seeds = reactive.Value(value=None)
    _uploaded_volumes = reactive.Value(value=None)
    _processed_data = reactive.Value(value=None)

    @reactive.effect
    def _() -> None:
        file: list[FileInfo] | None = input.uploaded_seeds()
        if file is not None:
            _uploaded_seeds.set(pd.read_csv(file[0]["datapath"]))

    @reactive.effect
    def _() -> None:
        file: list[FileInfo] | None = input.uploaded_volumes()

        if file is not None:
            _uploaded_volumes.set(pd.read_csv(file[0]["datapath"]))

    @reactive.effect
    @reactive.event(input.generate)
    def _() -> None:
        if input.dim() == Dimension.THREE_D:
            iv.add_rule("height", utils.req_gt(rhs=0))

        iv.enable()
        if not iv.is_valid():
            _processed_data.set(Exception(views.invalid_input_text()))
            return

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

        if input.diagram() == DiagramType.LAGUERRE:
            out = laguerre.volumes(
                input=input,
                iv=iv,
                domain_vol=np.prod(box_dim),
                space_dim=space_dim,
                uploaded_volumes=_uploaded_volumes(),
            )
            if isinstance(out, Exception):
                _processed_data.set(out)
                return

            volumes, phases = out
            n_grains = len(volumes)
            tol = input.tol()

        else:
            iv.add_rule("n_grains", utils.req_int_gt(rhs=0))
            iv.enable()
            if not iv.is_valid():
                _processed_data.set(Exception(views.invalid_input_text()))
                return

            n_grains = input.n_grains()
            volumes = None
            phases = np.zeros(n_grains)
            tol = None

        if input.seeds_init() == SeedInitializer.RANDOM:
            iv.add_rule("random_state", utils.int_gte(rhs=0, allow_none=True))
            iv.enable()

            if not iv.is_valid():
                _processed_data.set(
                    Exception(
                        "Invalid random seed. Please check your input and try again."
                    )
                )
                return

            seeds = sample_random_seeds(
                domain=domain,
                n_grains=n_grains,
                random_state=input.random_state(),
            )

        else:
            seeds = _uploaded_seeds()
            if seeds is None:
                _processed_data.set(
                    Exception("Seeds not uploaded. Upload seeds and try again.")
                )
                return

            coords = list(utils.COORDINATES)[:space_dim]
            res = utils.validate_upload(
                seeds,
                expected_colnames=coords,
                expected_dim=(n_grains, space_dim),
                file="seeds",
                bounds=dict(zip(coords, domain)),
            )
            if isinstance(res, Exception):
                _processed_data.set(res)
                return

            seeds = seeds[coords].values

        # In case of Laguerre, check if len of volumes and seeds match
        if input.diagram() == DiagramType.LAGUERRE:
            if len(volumes) != len(seeds):
                _processed_data.set(
                    Exception(
                        f"""The number of samples in seeds and grain volumes do not match:
                  len(seeds)={len(seeds)}, len(volumes)={len(volumes)}"""
                    )
                )
                return

        config = utils.AlgorithmConfig(
            tol=tol,
            n_iter=input.n_iter(),
            damp_param=input.damp_param(),
            verbose=False,
        )
        sm_data = SynthetMicData(
            domain=domain,
            seeds=seeds,
            volumes=volumes,
            periodic=list(periodic),
            init_weights=None,
        )
        sdg = utils.ShinyDiagramGenerator(config)
        sdg.init_pbar()
        generator = sdg.fit(sm_data)
        sdg.close_pbar()

        _processed_data.set(
            utils.ProcessedData(sm_data=sm_data, generator=generator, phases=phases)
        )

    @reactive.effect
    @reactive.event(input.generate)
    async def _():
        await session.send_custom_message(
            "ga_event",
            asdict(
                utils.Event(
                    name="generate_button",
                    category="button",
                    label="generate diagram",
                )
            ),
        )

    @render.ui
    def box_dim() -> ui.Tag:
        ids = ("length", "breadth")
        labels = tuple([id.title() for id in ids])
        defaults = (1.0, 1.0)
        if input.dim() == Dimension.THREE_D:
            ids += ("height",)
            labels += ("Height",)
            defaults += (1.0,)

        return views.numeric_input(ids, labels, defaults)

    @render.ui
    def periodicity() -> ui.Tag:
        space_dim = 2 if input.dim() == Dimension.TWO_D else 3
        ids = [f"is_{c}_periodic" for c in utils.COORDINATES[:space_dim]]

        return ui.div(
            ui.markdown("Periodicity"),
            views.periodic_input(ids, utils.COORDINATES[:space_dim]),
        )

    @render.ui
    def algorithm() -> ui.Tag:
        ids = ("n_iter", "damp_param")
        labels = ("Lloyd iterations", "Damp parameter")
        defaults = (5, 1.0)

        if input.diagram() == DiagramType.LAGUERRE:
            ids += ("tol",)
            labels += ("Tolerance",)
            defaults += (1.0,)

        return views.numeric_input(ids=ids, labels=labels, defaults=defaults)

    @render.table
    def seeds_summary_table() -> pd.DataFrame:
        return utils.summarize_df(_uploaded_seeds())

    @render.ui
    def uploaded_seeds_summary() -> ui.Tag:
        return views.seeds_summary(_uploaded_seeds(), "seeds_summary_table")

    @render.ui
    def single_phase_dist_param() -> ui.Tag:
        return laguerre.create_dist_params(input.single_phase_dist(), "single_phase")

    @render.ui
    def phase1_dist_param() -> ui.Tag:
        return laguerre.create_dist_params(input.phase1_dist(), "phase1")

    @render.ui
    def phase2_dist_param() -> ui.Tag:
        return laguerre.create_dist_params(input.phase2_dist(), "phase2")

    @render.table
    def volumes_summary_table() -> pd.DataFrame:
        return utils.summarize_df(_uploaded_volumes())

    @render.ui
    def uploaded_volumes_summary() -> ui.Tag:
        if _uploaded_volumes() is None:
            return ui.help_text(
                "No volumes uploaded yet. Information about volumes will be displayed here after upload."
            )

        return ui.output_table("volumes_summary_table")

    @render.text
    def volume_percentage_text() -> str | None:
        if any(
            [
                input.phase1_vol_ratio() is None,
                input.phase2_vol_ratio() is None,
                input.phase() == Phase.UPLOAD,
            ]
        ):
            return

        if input.phase() == Phase.SINGLE:
            return "Single phase volume percentage is 100% of the domain volume."

        phase1_vol_percent = (
            input.phase1_vol_ratio()
            * 100
            / (input.phase1_vol_ratio() + input.phase2_vol_ratio())
        )

        return (
            f"Phase 1 volume percentage is {phase1_vol_percent:.2f}%; Phase 2 volume percentage is {100.0 - phase1_vol_percent:.2f}% "
            "of the domain volume."
        )

    @render.text
    def d90_text() -> str | None:
        if (
            input.phase() == Phase.SINGLE
            and input.single_phase_dist() == Distribution.LOGNORMAL
        ):
            return laguerre.compute_d90_text(
                mean=input.single_phase_mean(), std=input.single_phase_std()
            )

        if input.phase() == Phase.DUAL:
            msg = []
            for i, p in enumerate((input.phase1_dist(), input.phase2_dist()), start=1):
                if p == Distribution.LOGNORMAL:
                    text = laguerre.compute_d90_text(
                        mean=getattr(input, f"phase{i}_mean")(),
                        std=getattr(input, f"phase{i}_std")(),
                    )
                    if text is not None:
                        msg.append(f"Phase {i} {text}")

            if msg:
                return " ".join(msg)

    common.server("common", _processed_data)
