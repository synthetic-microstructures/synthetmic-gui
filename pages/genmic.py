import faicons as fa
import numpy as np
from matplotlib import colormaps
from shiny import Inputs, Outputs, Session, module, reactive, render, ui

import shared.controls as ct
from shared import utils, views
from tabs import clip, full, slice_


@module.ui
def page_ui() -> ui.Tag:
    return ui.row(
        ui.column(
            3,
            ui.output_ui("common_opts"),
        ),
        ui.column(
            9,
            ui.output_ui("diagram_view"),
        ),
    )


@module.server
def server(
    input: Inputs,
    output: Outputs,
    session: Session,
    fitted_data: tuple[utils.SynthetMicData, utils.LaguerreDiagramGenerator],
    global_generate: reactive.Value,
):
    def is_view(lhs: str, rhs: str) -> bool:
        return lhs == rhs

    def is_3d(domain: np.ndarray):
        return len(domain) == 3

    # @reactive.calc
    # @reactive.event(input.clip_normal)
    # def _clip_interval() -> tuple[float, float]:
    #     data, _ = fitted_data
    #
    #     return utils.compute_cut_interval(
    #         input.clip_normal(), utils.COORDINATES, data.domain
    #     )
    #
    # @render.ui
    # def slice_normal_selection():
    #     return views.create_selection(
    #         id="slice_normal",
    #         label="Choose a coordinate along which the slice will be created",
    #         choices=list(utils.COORDINATES),
    #         selected=utils.COORDINATES[0],
    #     )
    #
    # @render.ui
    # def clip_normal_selection():
    #     return views.create_selection(
    #         id="clip_normal",
    #         label="Choose a coordinate along which the clip will be created",
    #         choices=list(utils.COORDINATES),
    #         selected=utils.COORDINATES[0],
    #     )
    #
    # @render.ui
    # def slice_value_slider():
    #     a, b = _slice_interval()
    #     return ui.input_slider(
    #         id="slice_value",
    #         label="Slide to select value along slice coordinate",
    #         value=ct.PLOT_DEFAULTS["slice_value"],
    #         min=a,
    #         max=b,
    #         ticks=True,
    #     )

    # @render.ui
    # def clip_value_slider():
    #     a, b = _clip_interval()
    #     TOL = 0.001  # tolerance set to avoid creating clip on the boundary
    #     return ui.input_slider(
    #         id="clip_value",
    #         label="Slide to select value along clip coordinate",
    #         value=ct.PLOT_DEFAULTS["clip_value"],
    #         min=a + TOL,
    #         max=b - TOL,
    #         ticks=True,
    #     )

    @reactive.effect
    @reactive.event(input.colorby, input.opacity, input.colormap)
    def _():
        match input.view():
            case ct.DiagramView.FULL:
                full.server(
                    "full",
                    fitted_data=fitted_data,
                    colorby=input.colorby(),
                    colormap=input.colormap(),
                    opacity=input.opacity(),
                )

            case ct.DiagramView.SLICE:
                slice_.server(
                    "slice",
                    fitted_data=fitted_data,
                    colorby=input.colorby(),
                    colormap=input.colormap(),
                    opacity=input.opacity(),
                )

            case ct.DiagramView.CLIP:
                clip.server(
                    "clip",
                    fitted_data=fitted_data,
                    clip_normal=input.clip_normal(),
                    clip_value=input.clip_value(),
                    colorby=input.colorby(),
                    colormap=input.colormap(),
                    add_final_seed_positions=input.addpositions(),
                    opacity=input.opacity(),
                )

    @reactive.effect
    def _():
        # update the choices for colorby to exclude color by
        # target volumes and errors when view is set to slice
        colorby_choices = (
            [ct.Colorby.FITTED_VOLUMES, ct.Colorby.RANDOM]
            if input.view() == ct.DiagramView.SLICE
            else [c for c in ct.Colorby]
        )
        ui.update_select(
            id="colorby",
            choices=colorby_choices,  # type: ignore
            selected=input.colorby(),
        )

    @reactive.effect
    @reactive.event(global_generate)
    def _():
        # to ensure the current plot settings are remembered after
        # reclicking generate button
        ui.update_navs(id="view", selected=input.view())
        ui.update_select(
            id="colorby",
            selected=input.colorby(),
        )

        ui.update_select(id="colormap", selected=input.colormap())
        ui.update_slider(id="opacity", value=input.opacity())

    @reactive.effect
    @reactive.event(input.reset_plot_options)
    def _():
        data, _ = fitted_data

        # if input.view() == ct.DiagramView.SLICE and is_3d(data.domain):
        #     ui.update_select(
        #         id="slice_normal", selected=ct.PLOT_DEFAULTS["slice_normal"]
        #     )
        #     ui.update_slider(id="slice_value", value=ct.PLOT_DEFAULTS["slice_value"])
        #
        # if input.view() == ct.DiagramView.CLIP:
        #     ui.update_select(id="clip_normal", selected=ct.PLOT_DEFAULTS["clip_normal"])
        #     ui.update_slider(id="clip_value", value=ct.PLOT_DEFAULTS["clip_value"])
        #
        ui.update_select(id="colorby", selected=ct.PLOT_DEFAULTS["colorby"])
        ui.update_select(id="colormap", selected=ct.PLOT_DEFAULTS["colormap"])
        ui.update_slider(id="opacity", value=ct.PLOT_DEFAULTS["opacity"])

    @render.ui  # FIX: why cant I move this to the module.ui?
    def common_opts():
        return (
            views.create_selection(
                id="colorby",
                label="Color by",
                choices=[c for c in ct.Colorby],
                selected=ct.PLOT_DEFAULTS["colorby"],
            ),
            views.create_selection(
                id="colormap",
                label="Choose a colormap",
                choices=sorted(list(colormaps)),
                selected=ct.PLOT_DEFAULTS["colormap"],
            ),
            ui.input_slider(
                id="opacity",
                label="Diagram opacity",
                min=0.0,
                max=1.0,
                value=ct.PLOT_DEFAULTS["opacity"],
                ticks=True,
            ),
            ui.p(),
            ui.input_action_button(
                id="reset_plot_options",
                label="Reset plot options to defaults",
                icon=fa.icon_svg("gear"),
                class_="btn btn-primary",
            ),
        )

    # @render.ui
    # def extra_opts():
    #     opts = []
    #
    #     if input.view() == ct.DiagramView.SLICE:
    #         opts.extend(
    #             [
    #                 ui.output_ui("slice_normal_selection"),
    #                 ui.output_ui("slice_value_slider"),
    #             ]
    #         )
    #
    #     elif input.view() == ct.DiagramView.CLIP:
    #         opts.extend(
    #             [
    #                 ui.output_ui("clip_normal_selection"),
    #                 ui.output_ui("clip_value_slider"),
    #             ]
    #         )

    # if input.view() != ct.DiagramView.SLICE:
    #     opts.append(
    #         ui.input_switch(
    #             "addpositions",
    #             "Add final seed positions",
    #             ct.PLOT_DEFAULTS["addpositions"],
    #         ),
    #     )
    #
    #    return opts

    @render.ui
    def diagram_view():
        data, _ = fitted_data
        nav_controls = [
            ui.nav_spacer(),
            ui.nav_panel(
                ct.DiagramView.FULL,
                full.tab_ui("full"),
            ),
            ui.nav_panel(
                ct.DiagramView.CLIP,
                clip.tab_ui("clip"),
            ),
        ]

        if is_3d(data.domain):
            nav_controls.append(
                ui.nav_panel(
                    ct.DiagramView.SLICE,
                    slice_.tab_ui("slice"),
                ),
            )

        return ui.navset_card_pill(*nav_controls, id="view")
