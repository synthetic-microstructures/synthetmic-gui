from typing import Any

from shiny import ui

import shared.controls as ct

# def create_selection(id: str, label: str, choices: list[Any], selected: Any) -> ui.Tag:
#     return ui.input_select(
#         id=id,
#         label=label,
#         choices=choices,
#         selected=selected,
#         multiple=False,
#         selectize=True,
#     )


# def create_input_numeric(id: str, label: str, value: float | int | None = None):
#     return ui.input_numeric(
#         id=id,
#         label=label,
#         value=value,
#     )


def create_dim_selection(id: str) -> ui.Tag:
    return ui.input_select(
        id=id,
        label="Choose a dimension",
        choices=[d for d in ct.Dimension],
        selected=ct.Dimension.TWO_D,
        multiple=False,
        selectize=True,
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


# def create_dist_selection(id: str) -> ui.Tag:
#     return create_selection(
#         id=id,
#         label="Choose a distribution",
#         choices=[d.title() for d in ct.Distribution],
#         selected=ct.Distribution.UNIFORM,
#     )


# def create_distparam_input(*args) -> ui.Tag:
#     cols = [
#         ui.column(
#             6,
#             create_input_numeric(
#                 id=id,
#                 label=id.title(),
#             ),
#         )
#         for id in args
#     ]

#     return ui.row(*cols)


# def create_phase_selection(*args) -> ui.Tag:
#     cols = [
#         ui.column(
#             6,
#             create_selection(
#                 id=args[0],
#                 label="Choose a phase",
#                 choices=[p.title() for p in ct.Phase],
#                 selected=ct.Phase.SINGLE,
#             ),
#         ),
#         ui.column(
#             6,
#             create_input_numeric(
#                 id=args[1],
#                 label="Number of grains",
#             ),
#         ),
#     ]
#     return ui.row(*cols)


# def create_ratio_input(*args) -> ui.Tag:
#     cols = [
#         ui.column(
#             6,
#             create_input_numeric(
#                 id=id,
#                 label=id.replace("_", " ").capitalize(),
#             ),
#         )
#         for id in args
#     ]
#     return ui.row(*cols)


# def create_algo_input(**kwargs) -> ui.Tag:
#     cols = [
#         ui.column(
#             4,
#             create_input_numeric(
#                 id=id,
#                 label=label,
#             ),
#         )
#         for id, label in kwargs.items()
#     ]

#     return ui.row(*cols)


def domain_help_text() -> ui.Tag:
    return ui.markdown(
        """
        This is the dimension of the synthetic microstructure.

        **Length**, **Breadth**, and **Height** are, respectively, the length,
        breadth, and height of the box that defines the domain.

        Turn on **Periodic** to ensure periodicity of the domain in all directions.
        """
    )


def dist_help_text() -> ui.Tag:
    return ui.markdown(
        """
        This is the distribution of the position of seeds.

        In case of **Uniform distribution**, **Low** and **High** are the
        boundaries of the distribution such that the generated values
        will always fall between them.

        In case of **Normal** and **Lognormal** distributions, the **Mean** and
        **Standard deviation** are, respectively, the mean and standard deviation
        of the corresponding distributions.
        """
    )


def grains_help_text() -> ui.Tag:
    return ui.markdown(
        """
        **Total grains**: this is the total number of grains in the synthetic microstructure.

        **Grain ratio**: this is grain ratio for the idealised microstructure. If the number
        of smaller grains is n, then the number of bigger grains will be grain ratio * n (approx.).

        **Volume ratio**: this is the volume ratio for the idealised microstructure. If the volume of each smaller
        grain is v then that of the bigger grain will be volume ratio * v (approx.). A volume ratio of 1 indicates a single phase
        microstructure while a volume ratio > 1 indicates a dual-phase microstructure.
        """
    )


def algo_help_text() -> ui.Tag:
    return ui.markdown(
        """
        **Tolerance**: relative percentage error for volumes.

        *Iterations*: number of iterations of Lloyd's algorithm (move each seed to the
        centroid of its cell).

        *Damping parameter*: the damping parametr of the damped Lloyd step; value must be between
        0 and 1 (inclusive at both ends)
        )
        """
    )


def group_ui_elements(*args, title: ui.Tag, help_text: ui.Tag) -> ui.Tag:
    qn_circle_fill = ui.HTML(
        '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-question-circle-fill mb-1" viewBox="0 0 16 16"><path d="M16 8A8 8 0 1 1 0 8a8 8 0 0 1 16 0zM5.496 6.033h.825c.138 0 .248-.113.266-.25.09-.656.54-1.134 1.342-1.134.686 0 1.314.343 1.314 1.168 0 .635-.374.927-.965 1.371-.673.489-1.206 1.06-1.168 1.987l.003.217a.25.25 0 0 0 .25.246h.811a.25.25 0 0 0 .25-.25v-.105c0-.718.273-.927 1.01-1.486.609-.463 1.244-.977 1.244-2.056 0-1.511-1.276-2.241-2.673-2.241-1.267 0-2.655.59-2.75 2.286a.237.237 0 0 0 .241.247zm2.325 6.443c.61 0 1.029-.394 1.029-.927 0-.552-.42-.94-1.029-.94-.584 0-1.009.388-1.009.94 0 .533.425.927 1.01.927z"/></svg>'
    )

    return ui.card(
        ui.card_header(
            title,
            ui.popover(
                ui.span(
                    qn_circle_fill,
                    style="position:absolute; top: 5px; right: 7px;",
                ),
                help_text,
            ),
        ),
        *args,
    )
