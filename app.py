import io
import os
import pathlib
import tempfile
import zipfile
from datetime import datetime

import faicons as fa
import numpy as np
import pandas as pd
from matplotlib import colormaps
from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from shiny.types import FileInfo
from shiny_validate import InputValidator, check

import shared.controls as ct
from shared import utils, views

sidebar = ui.sidebar(
    views.how_text(),
    views.group_ui_elements(
        ui.output_ui("space_dim"),
        ui.output_ui("box_dim"),
        ui.input_switch("periodic", "Periodic", False),
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
            [0.1, 5, 1.0],
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
    ui.nav_panel(ct.Tab.GEN_MIC, ui.output_ui(id="gen_mic_tab")),
    ui.nav_panel(ct.Tab.METRICS_AND_PLOTS, ui.output_ui(id="metrics_and_plots_tab")),
    ui.head_content(ui.tags.link(rel="icon", type="image/png", href="favicon.ico")),
    ui.head_content(
        ui.tags.style(
            """
        .popover {
            max-width: 400px !important;
            width: 400px !important;
        }
        .modal-dialog {
                margin-top: 20px !important;
                overflow-y: hidden !important;
        } 
        .navbar-nav {
            justify-content: center !important;
        }
        """
        )
    ),
    sidebar=sidebar,
    title="SynthetMic-GUI",
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

    # ....................................................................
    # reactive and and side effect calculations
    # ...................................................................

    # define reactive variables for holding uploaded seeds and volumes
    _uploaded_seeds = reactive.Value(value=None)
    _uploaded_volumes = reactive.Value(value=None)
    _generated_plotters = reactive.Value(value=None)

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

    views.info_modal()

    @reactive.calc
    @reactive.event(input.generate)
    def _generated_diagram() -> utils.Diagram | str:
        def add_dist_param_to_iv(dist: str, id_prefix: str, **kwargs) -> dict:
            match dist:
                case ct.Distribution.UNIFORM:
                    iv.add_rule(f"{id_prefix}_low", req_gt(rhs=0))
                    iv.add_rule(f"{id_prefix}_high", req_gt(rhs=0))

                    return {
                        k: v()
                        for k, v in zip(
                            ("low", "high"), kwargs.get(ct.Distribution.UNIFORM)
                        )
                    }

                case ct.Distribution.LOGNORMAL:
                    iv.add_rule(f"{id_prefix}_mean", req_gt(rhs=0))
                    iv.add_rule(f"{id_prefix}_std", req_gt(rhs=0))

                    return {
                        k: v()
                        for k, v in zip(
                            ("mean", "std"), kwargs.get(ct.Distribution.LOGNORMAL)
                        )
                    }

                case _:
                    return {}

        state_vars = {}

        box_dim = [input.length(), input.breadth()]
        if input.dim() == ct.Dimension.THREE_D:
            iv.add_rule("height", req_gt(rhs=0))
            box_dim.append(input.height())

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
                        state_vars.get("domain_vol"),
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
                        state_vars.get("domain_vol"),
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
                        _uploaded_volumes(),
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
                    volumes = _uploaded_volumes()[
                        utils.VOLUMES
                    ].values  # get the underlying numpy array
                    diff = abs(volumes.sum() - state_vars.get("domain_vol"))
                    if diff > VOL_DIFF_TOL:
                        return f"""Mismatch total volume: domain volume is {state_vars.get("domain_vol")} whereas total uploaded volume is {volumes.sum()};
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
                    state_vars.get("n_grains"),
                    input.random_state(),
                    *state_vars.get("box_dim"),
                )
                state_vars["seeds"] = seeds

            case ct.SeedInitializer.UPLOAD:
                if _uploaded_seeds() is None:
                    return "Seeds not uploaded. Upload seeds and try again."

                # validate the seeds
                val_out = utils.validate_df(
                    _uploaded_seeds(),
                    expected_colnames=list(utils.COORDINATES)[
                        : state_vars.get("space_dim")
                    ],
                    expected_dim=(
                        state_vars.get("n_grains"),
                        state_vars.get("space_dim"),
                    ),
                    expected_type="float",
                    file="seeds",
                    bounds=dict(
                        zip(
                            utils.COORDINATES[: state_vars.get("space_dim")],
                            state_vars.get("domain"),
                        )
                    ),
                )
                if isinstance(val_out, str):
                    return val_out

                # add domain and seeds to state_vars since they have been uploaded
                seeds = _uploaded_seeds()
                state_vars["seeds"] = seeds.values  # get the underlying numpy array

            case _:
                return f"Mismatch seed initializer: {input.seeds_init()}. Input must be one of {', '.join(ct.SeedInitializer)}."

        # finally check if volumes and seeds dim match; very important for the uploads
        if len(state_vars.get("volumes")) != len(state_vars.get("seeds")):
            return f"""The number of samples in seeds and grain volumes do not match:
              len(seeds)={len(state_vars.get("seeds"))}, len(volumes)={len(state_vars.get("volumes"))}"""

        return utils.generate_diagram(
            domain=state_vars.get("domain"),
            seeds=state_vars.get("seeds"),
            volumes=state_vars.get("volumes"),
            periodic=input.periodic(),
            tol=input.tol(),
            n_iter=input.n_iter(),
            damp_param=input.damp_param(),
        )

    @reactive.effect
    def _():
        diagram = _generated_diagram()
        mesh, plotters = utils.plot_diagram(
            generator=diagram.generator,
            target_volumes=diagram.target_volumes,
            colorby=input.colorby(),
            colormap=input.colormap(),
            add_final_seed_positions=input.addpositions(),
            opacity=input.opacity(),
        )

        _generated_plotters.set((mesh, plotters))

        # ensure most recent plot settings are remembered
        ui.update_select(id="slice", selected=input.slice())
        ui.update_select(id="colorby", selected=input.colorby())
        ui.update_select(id="colormap", selected=input.colormap())
        ui.update_switch(id="addpositions", value=input.addpositions())
        ui.update_slider(id="opacity", value=input.opacity())
        ui.update_select(id="fig_extension", selected=input.fig_extension())
        ui.update_select(id="prop_extension", selected=input.prop_extension())

    @reactive.effect
    @reactive.event(input.reset_plot_options)
    def _():
        ui.update_select(id="slice", selected=ct.PLOT_DEFAULTS.get("slice"))
        ui.update_select(id="colorby", selected=ct.PLOT_DEFAULTS.get("colorby"))
        ui.update_select(id="colormap", selected=ct.PLOT_DEFAULTS.get("colormap"))
        ui.update_switch(id="addpositions", value=ct.PLOT_DEFAULTS.get("addpositions"))
        ui.update_slider(id="opacity", value=ct.PLOT_DEFAULTS.get("opacity"))
        ui.update_select(
            id="fig_extension", selected=ct.PLOT_DEFAULTS.get("fig_extension")
        )
        ui.update_select(
            id="prop_extension", selected=ct.PLOT_DEFAULTS.get("prop_extension")
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
            return utils.summarize_df(_uploaded_seeds())

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
                    id="random_state", label="Seeds random state", value=None
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
            return utils.summarize_df(_uploaded_volumes())

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

    @render.ui
    def display_diagram():
        _, plotters = _generated_plotters()
        return ui.HTML(plotters.get(input.slice()).export_html(filename=None).read())

    @render.plot
    def vol_dist_plot():
        diagram = _generated_diagram()

        return utils.plot_volume_dist(diagram)

    @render.download(
        filename=lambda: f"full-diagram-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.{input.fig_extension()}",
    )
    def download_full_diagram():
        mesh, plotters = _generated_plotters()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=f".{input.fig_extension()}", delete=True
        ) as tmp_file:
            filename = tmp_file.name

            match input.fig_extension():
                case (
                    ct.FigureExtension.PDF
                    | ct.FigureExtension.EPS
                    | ct.FigureExtension.SVG
                ):
                    plotters.get(utils.Slice.FULL).save_graphic(filename)

                case ct.FigureExtension.HTML:
                    plotters.get(utils.Slice.FULL).export_html(filename)

                case ct.FigureExtension.VTK:
                    mesh.save(filename, binary=False)

                case _:
                    os.unlink(
                        filename
                    )  # ensure the file is deleted incase of wrong input
                    raise ValueError(
                        f"Mismatch extension: {input.fig_extension()}. Input must be one of {', '.join(ct.FigureExtension)}."
                    )

            with open(filename, "rb") as f:
                content = f.read()

        yield content

    @render.download(
        filename=lambda: f"diagram-property-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.zip",
        media_type="application/zip",
    )
    def download_diagram_property():
        diagram = _generated_diagram()
        diagram_prop = utils.extract_property_as_df(diagram)

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            for fname, df in diagram_prop.items():
                buffer = io.BytesIO()
                df.to_csv(buffer, index=False)
                buffer.seek(0)
                zipf.writestr(f"{fname}.{input.prop_extension()}", buffer.getvalue())
                buffer.close()

        zip_buffer.seek(0)

        yield zip_buffer.getvalue()

    def get_tab(tab: str) -> ui.Tag:
        diagram = _generated_diagram()

        if isinstance(diagram, str):
            ui.notification_show(diagram, type="error", duration=None)
            return

        diagram_ctrl = views.create_selection(
            id="slice",
            label="Choose a diagram to view",
            choices=[s for s in ct.Slice],
            selected=ct.Slice.FULL,
        )

        metrics = ui.layout_column_wrap(
            *[
                ui.value_box(
                    title=t,
                    value=utils.format_to_standard_form(v, 2)
                    if t
                    in (
                        "Max percentage error",
                        "Mean percentage error",
                    )
                    else f"{v:.2f}",
                    full_screen=False,
                    showcase=fa.icon_svg("magnifying-glass"),
                    height="160px",
                )
                for t, v in zip(
                    [
                        "Max percentage error",
                        "Mean percentage error",
                        "Sum of target volumes",
                        "Sum of fitted volumes",
                    ],
                    [
                        diagram.max_percentage_error,
                        diagram.mean_percentage_error,
                        diagram.target_volumes.sum(),
                        diagram.fitted_volumes.sum(),
                    ],
                )
            ]
        )

        match tab:
            case ct.Tab.GEN_MIC:
                return ui.tags.div(
                    ui.row(
                        ui.column(
                            3,
                            ui.card(
                                diagram_ctrl,
                                views.create_selection(
                                    id="colorby",
                                    label="Color by",
                                    choices=[c for c in ct.Colorby],
                                    selected=ct.PLOT_DEFAULTS.get("colorby"),
                                ),
                                views.create_selection(
                                    id="colormap",
                                    label="Choose a colormap",
                                    choices=sorted(list(colormaps)),
                                    selected=ct.PLOT_DEFAULTS.get("colormap"),
                                ),
                                ui.input_switch(
                                    "addpositions",
                                    "Add final seed positions",
                                    ct.PLOT_DEFAULTS.get("addpositions"),
                                ),
                                ui.input_slider(
                                    id="opacity",
                                    label="Diagram opacity",
                                    min=0.0,
                                    max=1.0,
                                    value=ct.PLOT_DEFAULTS.get("opacity"),
                                    ticks=True,
                                ),
                                views.create_selection(
                                    id="fig_extension",
                                    label="Download full diagram as",
                                    choices=[e for e in ct.FigureExtension],
                                    selected=ct.PLOT_DEFAULTS.get("fig_extension"),
                                ),
                                ui.download_button(
                                    id="download_full_diagram",
                                    label="Download full diagram",
                                    icon=fa.icon_svg("download"),
                                    class_="btn btn-primary",
                                ),
                                views.create_selection(
                                    id="prop_extension",
                                    label="Download properties as",
                                    choices=[e for e in ct.PropertyExtension],
                                    selected=ct.PLOT_DEFAULTS.get("prop_extension"),
                                ),
                                ui.download_button(
                                    id="download_diagram_property",
                                    label="Download properties",
                                    icon=fa.icon_svg("download"),
                                    class_="btn btn-primary",
                                ),
                                ui.input_action_button(
                                    id="reset_plot_options",
                                    label="Reset plot options to defaults",
                                    icon=fa.icon_svg("gear"),
                                    class_="btn btn-primary",
                                ),
                                style="height: 900px; overflow: hidden;",
                            ),
                        ),
                        ui.column(
                            9,
                            ui.card(
                                ui.output_ui(
                                    "display_diagram",
                                ),
                                style="height: 900px; overflow: hidden;",
                                full_screen=True,
                            ),
                        ),
                    ),
                )
            case ct.Tab.METRICS_AND_PLOTS:
                return ui.tags.div(
                    metrics,
                    ui.card(
                        ui.output_plot("vol_dist_plot"),
                        style="height: 600px; overflow: hidden;",
                    ),
                )

            case _:
                raise ValueError(
                    f"Invalid tab: {tab}; tab must be one of [{', '.join(ct.Tab)}]"
                )

    @render.ui
    def gen_mic_tab() -> ui.Tag:
        return get_tab(tab=ct.Tab.GEN_MIC)

    @render.ui
    def metrics_and_plots_tab() -> ui.Tag:
        return get_tab(tab=ct.Tab.METRICS_AND_PLOTS)


app = App(app_ui, server, static_assets={"/": f"{pathlib.Path(__file__).parent}/www"})
