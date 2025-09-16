import faicons as fa
from shiny import Inputs, Outputs, Session, module, render, ui

from shared import styles
from shared.utils import (
    LaguerreDiagramGenerator,
    SynthetMicData,
    format_to_standard_form,
    plot_volume_dist,
)


@module.ui
def tab() -> ui.Tag:
    return ui.output_ui("main")


@module.server
def server(
    input: Inputs,
    output: Outputs,
    session: Session,
    _fitted_data: tuple[SynthetMicData, LaguerreDiagramGenerator],
):
    @render.plot
    def vol_dist_plot():
        data, generator = _fitted_data
        return plot_volume_dist(
            seeds=data.seeds,
            target_volumes=data.volumes,
            fitted_volumes=generator.get_fitted_volumes(),
            vertices=generator.get_vertices(),
        )

    @render.ui
    def main():
        data, generator = _fitted_data
        metrics = ui.layout_column_wrap(
            *[
                ui.value_box(
                    title=t,
                    value=format_to_standard_form(v, 2)  # type: ignore
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
                        generator.max_percentage_error_,
                        generator.mean_percentage_error_,
                        data.volumes.sum(),
                        generator.get_fitted_volumes().sum(),
                    ],
                )
            ]
        )

        return ui.tags.div(
            metrics,
            ui.card(ui.output_plot("vol_dist_plot"), style=styles.tab_card),
        )
