import pathlib

import faicons as fa
import numpy as np
import pandas as pd
from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from shiny.types import FileInfo
from shiny_validate import InputValidator, check

import shared.controls as ct
from shared import styles, utils, views
from tabs import gentab, metrictab

sidebar = ui.sidebar(
    views.how_text(),
    views.group_ui_elements(
        ui.output_ui("space_dim"),
        ui.output_ui("box_dim"),
        ui.output_ui("periodicity"),
        title="Box dimension",
        help_text=views.box_help_text(),
    ),
    views.group_ui_elements(
        views.create_selection(
            id="phase",
            label="Choose a phase",
            choices=[p for p in ct.Phase],
            selected=ct.Phase.SINGLE,
        ),
        ui.output_ui("grain_vol_input"),
        ui.output_text("volume_percentage_text"),
        title="Grains",
        help_text=views.grains_help_text(),
    ),
    views.group_ui_elements(
        views.create_selection(
            id="seeds_init",
            label="Choose how seeds are initialized",
            choices=[i for i in ct.SeedInitializer],
            selected=ct.SeedInitializer.RANDOM,
        ),
        ui.output_ui("seeds_input"),
        views.create_numeric_input(
            ["tol", "n_iter", "damp_param"],
            ["Volume tolerance", "Lloyd iterations", "Damp param"],
            [1.0, 5, 1.0],
        ),
        title="Algorithm",
        help_text=views.algo_help_text(),
    ),
    ui.input_task_button(
        id="generate",
        label="Generate microstructure",
        icon=fa.icon_svg("person-running"),
        class_="btn btn-primary",
    ),
    ui.input_dark_mode(mode="light"),
    views.feedback_text(),
    width=560,
    id="sidebar",
)

app_ui = ui.page_navbar(
    ui.nav_panel(ct.Tab.GEN_MIC, gentab.tab("gentab")),
    ui.nav_panel(ct.Tab.METRICS_AND_PLOTS, metrictab.tab("metrictab")),
    ui.head_content(ui.tags.link(rel="icon", type="image/png", href="favicon.ico")),
    ui.head_content(ui.tags.style(styles.popover_modal_navbar)),
    id="tab",
    sidebar=sidebar,
    title=ui.tags.div(
        *("SynthetMic-GUI", ui.br(), ui.help_text(utils.get_app_version()))
    ),
    fillable=True,
    fillable_mobile=True,
)


def server(input: Inputs, output: Outputs, session: Session):
    # ....................................................................
    #  some validation rules to be reused
    # ...................................................................
    def req_gt(rhs: float):
        return check.compose_rules(utils.required(), utils.gt(rhs=rhs))

    def req_int_gt(rhs: float):
        return check.compose_rules(
            utils.required(),
            utils.integer(),
            utils.gt(rhs=rhs),
        )

    def req_int_gte(rhs: float):
        return check.compose_rules(
            utils.required(),
            utils.integer(),
            utils.gte(rhs=rhs),
        )

    def req_between(left: float, right: float):
        return check.compose_rules(
            utils.required(),
            utils.between(left=left, right=right),
        )

    iv = InputValidator()
    iv.add_rule("length", req_gt(rhs=0))
    iv.add_rule("breadth", req_gt(rhs=0))
    iv.add_rule("tol", req_gt(rhs=0))
    iv.add_rule("damp_param", req_between(left=0.0, right=1.0))
    iv.add_rule("n_iter", req_int_gte(rhs=0))

    # ------------------------------------------------------------------------------
    # some helper functions for parsing inputs; will be reused
    # ------------------------------------------------------------------------------
    def parse_box_dim() -> tuple[float, ...]:
        box_dim = (input.length(), input.breadth())
        if input.dim() == ct.Dimension.THREE_D:
            box_dim += (input.height(),)

        return box_dim

    def parse_periodicity() -> tuple[bool, ...]:
        periodic = (input.is_x_periodic(), input.is_y_periodic())
        if input.dim() == ct.Dimension.THREE_D:
            periodic += (input.is_z_periodic(),)

        return periodic

    # ....................................................................
    # reactive and and side effect calculations
    # ...................................................................

    # define reactive variables for holding uploaded seeds and volumes
    _uploaded_seeds = reactive.Value(value=None)
    _uploaded_volumes = reactive.Value(value=None)

    @reactive.effect
    def _() -> None:
        file: list[FileInfo] | None = input.uploaded_seeds()
        if file is not None:
            _uploaded_seeds.set(pd.read_csv(file[0]["datapath"]))  # type: ignore

    @reactive.effect
    def _() -> None:
        file: list[FileInfo] | None = input.uploaded_volumes()

        if file is not None:
            _uploaded_volumes.set(pd.read_csv(file[0]["datapath"]))  # type: ignore

    views.info_modal()

    @reactive.calc
    @reactive.event(input.generate)
    def _fitted_data() -> (
        tuple[utils.SynthetMicData, utils.LaguerreDiagramGenerator] | str
    ):
        def add_dist_param_to_iv(dist: str, id_prefix: str, **kwargs) -> dict:
            match dist:
                case ct.Distribution.UNIFORM:
                    iv.add_rule(f"{id_prefix}_low", req_gt(rhs=0))
                    iv.add_rule(f"{id_prefix}_high", req_gt(rhs=0))

                    return {
                        k: v()
                        for k, v in zip(
                            ("low", "high"), kwargs[ct.Distribution.UNIFORM]
                        )
                    }

                case ct.Distribution.LOGNORMAL:
                    iv.add_rule(f"{id_prefix}_mean", req_gt(rhs=0))
                    iv.add_rule(f"{id_prefix}_std", req_gt(rhs=0))

                    return {
                        k: v()
                        for k, v in zip(
                            ("mean", "std"), kwargs[ct.Distribution.LOGNORMAL]
                        )
                    }

                case _:
                    return {}

        state_vars = {}

        box_dim = parse_box_dim()
        periodic = parse_periodicity()
        if input.dim() == ct.Dimension.THREE_D:
            iv.add_rule("height", req_gt(rhs=0))

        state_vars["periodic"] = periodic
        state_vars["space_dim"] = len(box_dim)
        state_vars["box_dim"] = box_dim
        state_vars["domain_vol"] = np.prod(box_dim)
        state_vars["domain"] = np.array([[0, d] for d in box_dim])

        match input.phase():
            case ct.Phase.SINGLE:
                iv.add_rule("single_phase_n_grains", req_int_gt(rhs=0))

                kwargs = add_dist_param_to_iv(
                    input.single_phase_dist(),
                    "single_phase",
                    **dict(
                        zip(
                            [ct.Distribution.UNIFORM, ct.Distribution.LOGNORMAL],
                            [
                                (input.single_phase_low, input.single_phase_high),
                                (input.single_phase_std, input.single_phase_std),
                            ],
                        )
                    ),
                )

                iv.enable()
                if iv.is_valid():
                    volumes = utils.sample_single_phase_vols(
                        input.single_phase_dist(),
                        input.single_phase_n_grains(),
                        state_vars["domain_vol"],  # type: ignore
                        **kwargs,
                    )

                    state_vars["n_grains"] = input.single_phase_n_grains()
                    state_vars["volumes"] = volumes

            case ct.Phase.DUAL:
                for n in (1, 2):
                    iv.add_rule(f"phase{n}_n_grains", req_int_gt(rhs=0))
                    iv.add_rule(f"phase{n}_vol_ratio", req_gt(rhs=0))

                phase1_kwargs = add_dist_param_to_iv(
                    input.phase1_dist(),
                    "phase1",
                    **dict(
                        zip(
                            [ct.Distribution.UNIFORM, ct.Distribution.LOGNORMAL],
                            [
                                (input.phase1_low, input.phase1_high),
                                (input.phase1_std, input.phase1_std),
                            ],
                        )
                    ),
                )
                phase2_kwargs = add_dist_param_to_iv(
                    input.phase2_dist(),
                    "phase2",
                    **dict(
                        zip(
                            [ct.Distribution.UNIFORM, ct.Distribution.LOGNORMAL],
                            [
                                (input.phase2_low, input.phase2_high),
                                (input.phase2_std, input.phase2_std),
                            ],
                        )
                    ),
                )

                iv.enable()
                if iv.is_valid():
                    volumes = utils.sample_dual_phase_vols(
                        (input.phase1_dist(), input.phase2_dist()),
                        (input.phase1_n_grains(), input.phase2_n_grains()),
                        (input.phase1_vol_ratio(), input.phase2_vol_ratio()),
                        state_vars["domain_vol"],  # type: ignore
                        (phase1_kwargs, phase2_kwargs),
                    )

                    state_vars["n_grains"] = sum(
                        [input.phase1_n_grains(), input.phase2_n_grains()]
                    )
                    state_vars["volumes"] = volumes

            case ct.Phase.UPLOAD:
                iv.enable()
                if iv.is_valid():
                    if _uploaded_volumes() is None:
                        return "Grain volumes not uploaded. Upload grain volumes and try again."

                    # validate the volumes
                    val_out = utils.validate_df(
                        _uploaded_volumes(),  # type: ignore
                        expected_colnames=[utils.VOLUMES],
                        expected_dim=None,
                        expected_type="float",
                        file="volumes",
                        bounds=None,
                    )
                    if isinstance(val_out, str):
                        return val_out

                    # check if domain volume is close to the sum of the uploaded volumes.
                    VOL_DIFF_TOL = 1e-6
                    volumes = _uploaded_volumes()[  # type: ignore
                        utils.VOLUMES
                    ].values  # get the underlying numpy array
                    diff = abs(volumes.sum() - state_vars["domain_vol"])
                    if diff > VOL_DIFF_TOL:
                        return f"""Mismatch total volume: domain volume is {state_vars["domain_vol"]} whereas total uploaded volume is {volumes.sum()};
                        a difference of {diff}. Volume difference must be at most {VOL_DIFF_TOL:.2e}."""

                    state_vars["n_grains"] = len(volumes)
                    state_vars["volumes"] = volumes

            case _:
                return f"Mismatch phase: {input.phase()}. Input must be one of {', '.join(ct.Phase)}."

        # check the validity of user inputs
        if not {"n_grains", "volumes"}.issubset(state_vars):
            return "Invalid inputs. Please check all fields for the required values."

        # deal with the seeds
        match input.seeds_init():
            case ct.SeedInitializer.RANDOM:
                iv.add_rule("random_state", utils.integer(allow_none=True))

                _, seeds = utils.sample_seeds(
                    state_vars["n_grains"],
                    input.random_state(),
                    *state_vars["box_dim"],  # type: ignore
                )
                state_vars["seeds"] = seeds

            case ct.SeedInitializer.UPLOAD:
                if _uploaded_seeds() is None:
                    return "Seeds not uploaded. Upload seeds and try again."

                # validate the seeds
                val_out = utils.validate_df(
                    _uploaded_seeds(),  # type: ignore
                    expected_colnames=list(utils.COORDINATES)[  # type: ignore
                        : state_vars["space_dim"]
                    ],
                    expected_dim=(
                        state_vars["n_grains"],
                        state_vars["space_dim"],
                    ),
                    expected_type="float",
                    file="seeds",
                    bounds=dict(
                        zip(
                            utils.COORDINATES[: state_vars["space_dim"]],
                            state_vars["domain"],
                        )
                    ),
                )
                if isinstance(val_out, str):
                    return val_out

                # add domain and seeds to state_vars since they have been uploaded
                seeds = _uploaded_seeds()
                state_vars["seeds"] = seeds.values  # type: ignore  # get the underlying numpy array

            case _:
                return f"Mismatch seed initializer: {input.seeds_init()}. Input must be one of {', '.join(ct.SeedInitializer)}."

        # finally check if volumes and seeds dim match; very important for the uploads
        if len(state_vars["volumes"]) != len(state_vars["seeds"]):
            return f"""The number of samples in seeds and grain volumes do not match:
              len(seeds)={len(state_vars["seeds"])}, len(volumes)={len(state_vars["volumes"])}"""

        return utils.fit_data(
            domain=state_vars["domain"],
            seeds=state_vars["seeds"],
            volumes=state_vars["volumes"],
            periodic=list(state_vars["periodic"]),
            tol=float(input.tol()),
            n_iter=input.n_iter(),
            damp_param=float(input.damp_param()),
        )

    # ....................................................................
    # reactive and non-reactive uis
    # ...................................................................

    @render.ui
    def box_dim() -> ui.Tag:
        ids = ["length", "breadth"]
        labels = [id.title() for id in ids]
        defaults = [1.0, 1.0]
        if input.dim() == ct.Dimension.THREE_D:
            ids.append("height")
            labels.append("Height")
            defaults.append(1.0)

        return views.create_numeric_input(ids, labels, defaults)

    @render.ui
    def periodicity() -> ui.Tag:
        dim_value = 2 if input.dim() == ct.Dimension.TWO_D else 3
        ids = [f"is_{c}_periodic" for c in utils.COORDINATES[:dim_value]]
        labels = [f"{c}-coordinate" for c in utils.COORDINATES[:dim_value]]
        return ui.tags.div(
            ui.markdown("Periodicity"),
            views.create_periodic_input(ids, labels),
        )

    @render.ui
    def space_dim() -> ui.Tag:
        return views.create_selection(
            id="dim",
            label="Choose a dimension",
            choices=[d for d in ct.Dimension],
            selected=ct.Dimension.THREE_D,
        )

    @render.ui
    def uploaded_seeds_summary():
        @render.table
        def seeds_summary_table():
            return utils.summarize_df(_uploaded_seeds())  # type: ignore

        if _uploaded_seeds() is None:
            return ui.help_text(
                "No seeds uploaded yet. Information about seeds will be displayed here after upload."
            )

        return ui.output_table("seeds_summary_table")

    @render.ui
    def seeds_input() -> ui.Tag:
        if input.seeds_init() == ct.SeedInitializer.RANDOM:
            return ui.tags.div(
                ui.input_numeric(
                    id="random_state",
                    label="Seeds random state",
                    value=None,  # type: ignore
                ),
                ui.help_text(
                    "Seeds will be randomly generated in the specified box above."
                ),
            )

        return ui.tags.div(
            views.create_upload_handler(
                "uploaded_seeds",
                "Uplaod seeds as a csv or txt file",
            ),
            ui.output_ui("uploaded_seeds_summary"),
        )

    @render.ui
    def single_phase_dist_param() -> ui.Tag:
        return views.create_dist_param(input.single_phase_dist(), "single_phase")

    @render.ui
    def phase1_dist_param() -> ui.Tag:
        return views.create_dist_param(input.phase1_dist(), "phase1")

    @render.ui
    def phase2_dist_param() -> ui.Tag:
        return views.create_dist_param(input.phase2_dist(), "phase2")

    @render.ui
    def uploaded_volumes_summary():
        @render.table
        def volumes_summary_table():
            return utils.summarize_df(_uploaded_volumes())  # type: ignore

        if _uploaded_volumes() is None:
            return ui.help_text(
                "No volumes uploaded yet. Information about volumes will be displayed here after upload."
            )

        return ui.output_table("volumes_summary_table")

    @render.ui
    def grain_vol_input() -> ui.Tag:
        def phase_input(n: int) -> ui.Tag:
            return ui.row(
                ui.column(
                    4,
                    views.create_dist_selection(
                        id=f"phase{n}_dist", label=f"Phase {n} volume distribution"
                    ),
                ),
                ui.column(
                    4,
                    ui.input_numeric(
                        id=f"phase{n}_n_grains",
                        label=f"Phase {n} number of grains",
                        value=500,
                    ),
                ),
                ui.column(
                    4,
                    ui.input_numeric(
                        id=f"phase{n}_vol_ratio",
                        label=f"Phase {n} volume ratio",
                        value=1,
                    ),
                ),
            )

        if input.phase() == ct.Phase.SINGLE:
            ins = [
                ui.row(
                    ui.column(
                        6,
                        ui.input_numeric(
                            id="single_phase_n_grains",
                            label="Number of grains",
                            value=1000,
                        ),
                    ),
                    ui.column(6, views.create_dist_selection(id="single_phase_dist")),
                ),
                ui.output_ui("single_phase_dist_param"),
            ]

            return ui.tags.div(*ins)

        elif input.phase() == ct.Phase.UPLOAD:
            return ui.tags.div(
                views.create_upload_handler(
                    "uploaded_volumes",
                    "Uplaod volumes as a csv or txt file",
                ),
                ui.output_ui("uploaded_volumes_summary"),
            )

        ins = []
        for n in (1, 2):
            ins.extend((phase_input(n), ui.output_ui(f"phase{n}_dist_param")))
            if n == 1:
                ins.append(ui.hr())

        return ui.tags.div(*ins)

    @render.text
    def volume_percentage_text() -> str | None:
        if any(
            [
                input.phase1_vol_ratio() is None,
                input.phase2_vol_ratio() is None,
                input.phase() == ct.Phase.UPLOAD,
            ]
        ):
            return

        if input.phase() == ct.Phase.SINGLE:
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

    @reactive.effect
    @reactive.event(input.generate)
    def _():
        fitted_data = _fitted_data()
        if isinstance(fitted_data, str):
            ui.notification_show(fitted_data, type="error", duration=None)
            return

        gentab.server("gentab", fitted_data)
        metrictab.server("metrictab", fitted_data)


app = App(app_ui, server, static_assets={"/": f"{pathlib.Path(__file__).parent}/www"})
