import numpy as np
import pandas as pd
from shiny import Inputs, ui
from shiny_validate import InputValidator

from shared import comps, utils, views
from shared.consts import (
    Distribution,
    Phase,
)


def phase_input(n: int) -> ui.Tag:
    return ui.row(
        ui.column(
            4,
            views.dist_selection(
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


def sidebar() -> ui.Tag:
    return comps.group_ui_elements(
        comps.selection(
            id="phase",
            label="Choose a phase",
            choices=[p for p in Phase],
            selected=Phase.SINGLE,
        ),
        ui.panel_conditional(
            f"input.phase === '{Phase.SINGLE}'",
            ui.tags.div(
                ui.row(
                    ui.column(
                        6,
                        ui.input_numeric(
                            id="single_phase_n_grains",
                            label="Number of grains",
                            value=1000,
                        ),
                    ),
                    ui.column(6, views.dist_selection(id="single_phase_dist")),
                ),
                ui.output_ui("single_phase_dist_param"),
            ),
        ),
        ui.panel_conditional(
            f"input.phase === '{Phase.UPLOAD}'",
            ui.tags.div(
                comps.upload_handler(
                    "uploaded_volumes",
                    "Upload volumes as a csv or txt file",
                ),
                ui.output_ui("uploaded_volumes_summary"),
            ),
        ),
        ui.panel_conditional(
            f"input.phase === '{Phase.DUAL}'",
            ui.tags.div(
                phase_input(n=1),
                ui.output_ui("phase1_dist_param"),
                ui.hr(),
                phase_input(n=2),
                ui.output_ui("phase2_dist_param"),
            ),
        ),
        ui.output_text("d90_text"),
        ui.output_text("volume_percentage_text"),
        title="Grains",
        help_text=views.grains_help_text(),
    )


def add_dist_params(iv: InputValidator, dist: str, id_prefix: str, **kwargs) -> dict:
    match dist:
        case Distribution.UNIFORM:
            iv.add_rule(f"{id_prefix}_low", utils.req_gt(rhs=0))
            iv.add_rule(f"{id_prefix}_high", utils.req_gt(rhs=0))

            return {
                k: v() for k, v in zip(("low", "high"), kwargs[Distribution.UNIFORM])
            }

        case Distribution.LOGNORMAL:
            iv.add_rule(f"{id_prefix}_mean", utils.req_gt(rhs=0))
            iv.add_rule(f"{id_prefix}_std", utils.req_gt(rhs=0))

            return {
                k: v() for k, v in zip(("mean", "std"), kwargs[Distribution.LOGNORMAL])
            }

        case _:
            return {}


def create_dist_params(
    dist: str,
    id_prefix: str,
) -> ui.Tag:
    text = f"{dist} distribution selected;"
    match dist:
        case Distribution.CONSTANT:
            return ui.help_text(f"{text} all volumes will be equal for this phase.")

        case Distribution.UNIFORM:
            return ui.tags.div(
                views.numeric_input(
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
                views.numeric_input(
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


def volumes(
    input: Inputs,
    iv: InputValidator,
    domain_vol: float,
    space_dim: int,
    uploaded_volumes: pd.DataFrame | None,
) -> np.ndarray | Exception:
    iv.add_rule("tol", utils.req_gt(rhs=0))

    match input.phase():
        case Phase.SINGLE:
            iv.add_rule("single_phase_n_grains", utils.req_int_gt(rhs=0))

            kwargs = add_dist_params(
                iv=iv,
                dist=input.single_phase_dist(),
                id_prefix="single_phase",
                **dict(
                    zip(
                        [Distribution.UNIFORM, Distribution.LOGNORMAL],
                        [
                            (input.single_phase_low, input.single_phase_high),
                            (input.single_phase_mean, input.single_phase_std),
                        ],
                    )
                ),
            )

            iv.enable()
            if not iv.is_valid():
                return Exception(views.invalid_input_text())

            volumes = utils.sample_single_phase_vols(
                dist=input.single_phase_dist(),
                n_grains=input.single_phase_n_grains(),
                domain_vol=domain_vol,
                space_dim=space_dim,
                **kwargs,
            )

            return volumes

        case Phase.DUAL:
            for n in (1, 2):
                iv.add_rule(f"phase{n}_n_grains", utils.req_int_gt(rhs=0))
                iv.add_rule(f"phase{n}_vol_ratio", utils.req_gt(rhs=0))

            phase1_kwargs = add_dist_params(
                iv=iv,
                dist=input.phase1_dist(),
                id_prefix="phase1",
                **dict(
                    zip(
                        [Distribution.UNIFORM, Distribution.LOGNORMAL],
                        [
                            (input.phase1_low, input.phase1_high),
                            (input.phase1_mean, input.phase1_std),
                        ],
                    )
                ),
            )
            phase2_kwargs = add_dist_params(
                iv=iv,
                dist=input.phase2_dist(),
                id_prefix="phase2",
                **dict(
                    zip(
                        [Distribution.UNIFORM, Distribution.LOGNORMAL],
                        [
                            (input.phase2_low, input.phase2_high),
                            (input.phase2_mean, input.phase2_std),
                        ],
                    )
                ),
            )

            iv.enable()
            if not iv.is_valid():
                return Exception(views.invalid_input_text())

            volumes = utils.sample_dual_phase_vols(
                dist=(input.phase1_dist(), input.phase2_dist()),
                n_grains=(input.phase1_n_grains(), input.phase2_n_grains()),
                vol_ratio=(input.phase1_vol_ratio(), input.phase2_vol_ratio()),
                domain_vol=domain_vol,
                space_dim=space_dim,
                dist_kwargs=(phase1_kwargs, phase2_kwargs),
            )

            return volumes

        case Phase.UPLOAD:
            iv.enable()
            if not iv.is_valid():
                return Exception(views.invalid_input_text())

            if uploaded_volumes is None:
                return Exception(
                    "Grain volumes not uploaded. Upload grain volumes and try again."
                )

            val_out = utils.validate_df(
                uploaded_volumes,
                expected_colnames=[utils.VOLUMES],
                expected_dim=None,
                expected_type="float",
                file="volumes",
                bounds=None,
            )
            if isinstance(val_out, str):
                return Exception(val_out)

            VOL_DIFF_TOL = 1e-6
            volumes = uploaded_volumes[utils.VOLUMES].values
            diff = abs(volumes.sum() - domain_vol)
            if diff > VOL_DIFF_TOL:
                return Exception(f"""Mismatch total volume: domain volume is
                    {domain_vol} whereas total uploaded volume is {volumes.sum()};
                    a difference of {diff}. Volume difference must be at most {VOL_DIFF_TOL:.2e}.
                    """)

            return volumes

        case _:
            return Exception(
                f"Mismatch phase: {input.phase()}. Input must be one of {', '.join(Phase)}."
            )


def compute_d90_text(mean: float, std: float) -> str | None:
    text = """
            D90 for the lognormal distribution of ECDs with mean {}
            and std {} is {}. 
            """
    if any([i is None for i in (mean, std)]) or mean == 0:
        return

    d90 = utils.qp(mean=mean, std=std, p=0.9)
    d90 = utils.format_to_standard_form(d90, precision=2)

    return text.format(mean, std, d90)
