from datetime import datetime

import faicons as fa
import numpy as np
import pandas as pd
from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from shiny_validate import InputValidator, check

import shared.controls as ct
from shared import utils, views

side_bar = ui.sidebar(
    views.group_ui_elements(
        views.create_dim_selection(id="dim"),
        ui.output_ui(id="domain_input"),
        ui.input_switch("periodic", "Periodic", False),
        title="Domain",
        help_text=views.domain_help_text(),
    ),
    views.group_ui_elements(
        views.create_numeric_input(
            ["n_grains", "grain_ratio", "volume_ratio"],
            ["Total grains", "Grain ratio", "Volume ratio"],
            [1000, 1, 1],
        ),
        title="Grains",
        help_text=views.grains_help_text(),
    ),
    views.group_ui_elements(
        views.create_numeric_input(
            ["tol", "n_iter", "damp_param"],
            ["Tolerance", "Iterations", "Damp param"],
            [0.1, 5, 1.0],
        ),
        title="Algorithm",
        help_text=views.algo_help_text(),
    ),
    ui.input_task_button(
        id="generate",
        label="Generate microstructure",
        # auto_reset=False,
        class_="btn btn-primary",
        icon=fa.icon_svg("person-running"),
    ),
    ui.input_dark_mode(mode="light"),
    width=450,
    id="sidebar",
)
app_ui = ui.page_sidebar(
    side_bar,
    ui.output_ui(id="main_ui"),
    title="Synthetic microstructure generator",
    fillable=True,
    fillable_mobile=True,
)


def server(input: Inputs, output: Outputs, session: Session):
    iv = InputValidator()  # FIXME: check why check.gt does not work
    iv.add_rule("length", check.compose_rules(utils.required(), utils.gt(rhs=0)))
    iv.add_rule("breadth", check.compose_rules(utils.required(), utils.gt(rhs=0)))
    iv.add_rule(
        "n_grains",
        check.compose_rules(
            utils.required(),
            utils.integer(),
            utils.gt(rhs=0),
        ),
    )
    iv.add_rule("grain_ratio", check.compose_rules(utils.required(), utils.gt(rhs=0)))
    iv.add_rule("volume_ratio", check.compose_rules(utils.required(), utils.gt(rhs=0)))
    iv.add_rule("tol", check.compose_rules(utils.required(), utils.gt(rhs=0)))
    iv.add_rule(
        "damp_param",
        check.compose_rules(
            utils.required(),
            utils.between(left=0.0, right=1.0),
        ),
    )
    iv.add_rule(
        "n_iter",
        check.compose_rules(
            utils.required(),
            utils.integer(),
            utils.gt(rhs=0),
        ),
    )

    @render.ui
    def domain_input():
        ids = ["length", "breadth"]
        labels = [id.title() for id in ids]
        defaults = [3.0, 3.0]
        if input.dim() == ct.Dimension.THREE_D:
            ids.append("height")
            labels.append("Height")
            defaults.append(3.0)

        return views.create_numeric_input(ids, labels, defaults)

    # @render.ui
    # def distparam_input():
    #     if input.dist().lower() == ct.Distribution.UNIFORM:
    #         return views.create_distparam_input("low", "high")

    #     return views.create_distparam_input("mean", "std")

    # @render.ui
    # def ratio_input():
    #     if input.phase().lower() == ct.Phase.SINGLE:
    #         return ui.help_text(
    #             "Single phase selected. All grains will have the same volume."
    #         )

    #     return views.create_ratio_input("volume_ratio", "grain_ratio")

    @reactive.calc
    @reactive.event(input.generate)
    def _generate_diagram() -> tuple[utils.OutputData | None, str | None]:
        if input.dim() == ct.Dimension.THREE_D:
            iv.add_rule(
                "height", check.compose_rules(utils.required(), utils.gt(rhs=0))
            )

        iv.enable()
        if iv.is_valid():
            args = [input.length(), input.breadth()]
            if input.dim() == ct.Dimension.THREE_D:
                args.append(input.height())

            return utils.generate_diagram(
                input.n_grains(),
                input.grain_ratio(),
                input.volume_ratio(),
                input.tol(),
                input.n_iter(),
                input.damp_param(),
                input.periodic(),
                *args,
            ), None

        return None, "Invalid inputs. Please all fields for the required values."

    @render.ui
    def display_diagram():
        diagram, _ = _generate_diagram()
        return ui.HTML(diagram.diagram_htmls.get(input.slice().lower()))

    # @render.plot
    # def vol_dist_plot():
    #     fig, ax = plt.subplots()
    #     diagram, _ = _generate_diagram()

    #     for vol, label in zip(
    #         (diagram.actual_volumes, diagram.fitted_volumes),
    #         ("Actual", "Fitted"),
    #     ):
    #         plt.hist(vol, label=label)
    #         # sns.kdeplot(data=vol, fill=True, alpha=0.5, ax=ax, label=label)

    #     ax.set_xlabel("Volumes")
    #     ax.legend(frameon=False)

    #     return fig
    # @render_widget
    # def vol_dist_plot():
    #     diagram, _ = _generate_diagram()
    #     df = pd.DataFrame()
    #     df["Volumes"] = (
    #         diagram.actual_volumes.tolist() + diagram.fitted_volumes.tolist()
    #     )
    #     df["Type"] = ["Actual"] * len(diagram.actual_volumes) + ["Fitted"] * len(
    #         diagram.fitted_volumes
    #     )

    #     fig = px.histogram(
    #         df,
    #         x="Volumes",
    #         color="Type",
    #         # histnorm="probability density",  # Normalize for better comparison
    #         opacity=0.7,
    #         barmode="overlay",
    #     )
    #     return fig

    @render.download(
        filename=lambda: f"centroid-volume-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.csv",
        media_type="text/csv",
    )
    def download_centroid_vol():
        diagram, _ = _generate_diagram()

        columns = ["x-coordinate", "y-coordinate"]
        if diagram.centroids.shape[1] == 3:
            columns.append("z-coordinate")
        columns.extend(["actual volumes", "fitted volumes"])

        df = pd.DataFrame(
            data=np.column_stack(
                (diagram.centroids, diagram.actual_volumes, diagram.fitted_volumes)
            ),
            columns=columns,
        )

        yield df.to_csv().encode("utf-8")

    @render.ui
    def main_ui():
        diagram, error = _generate_diagram()

        if error is not None:
            ui.notification_show(error, type="error")
            return

        diagram_ctrl = ui.input_radio_buttons(
            id="slice",
            label="Choose a diagram to view",
            choices=[s.capitalize() for s in utils.Slice],
            inline=False,
        )

        metrics = ui.layout_column_wrap(
            *[
                ui.value_box(
                    title=t,
                    value=f"{v:.4f}",
                    full_screen=False,
                    showcase=fa.icon_svg("magnifying-glass"),
                )
                for t, v in zip(
                    [
                        "Max percentage error",
                        "Mean percentage error",
                        "Total actual volume",
                        "Total fitted volume",
                    ],
                    [
                        diagram.max_percentage_error,
                        diagram.mean_percentage_error,
                        diagram.actual_volumes.sum(),
                        diagram.fitted_volumes.sum(),
                    ],
                )
            ]
        )

        return ui.tags.div(
            metrics,
            ui.row(
                ui.column(
                    3,
                    diagram_ctrl,
                    ui.download_button(
                        id="download_centroid_vol",
                        label="Down centroids and volumes",
                        icon=fa.icon_svg("download"),
                        class_="btn btn-primary",
                    ),
                ),
                ui.column(
                    9,
                    ui.card(
                        ui.output_ui(
                            "display_diagram",
                        ),
                        # height="600px",
                        style="height: 500px; overflow: hidden;",
                        full_screen=True,
                    ),
                ),
            ),
            # ui.card(output_widget("vol_dist_plot")),
        )


app = App(app_ui, server)
