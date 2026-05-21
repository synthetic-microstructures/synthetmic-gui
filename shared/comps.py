from typing import Any

import faicons as fa
from shiny import ui


def sidebar(*args, **kwargs) -> ui.Sidebar:
    return ui.sidebar(
        *args,
        position="left",
        width=400,
        title=None,
        fillable=True,
        **kwargs,
    )


def download_button(id: str, label: str, icon: str = "download") -> ui.Tag:
    return ui.download_button(
        id=id,
        label=label,
        icon=fa.icon_svg(icon),
        width="100%",
        class_="btn btn-primary",
    )


def input_action_button(
    id: str, label: str, icon: str, width: str | None = "100%"
) -> ui.Tag:
    return ui.input_action_button(
        id=id,
        label=label,
        icon=fa.icon_svg(icon),
        width=width,
        class_="btn btn-primary",
    )


def input_task_button(id: str, label: str, icon: str) -> ui.Tag:
    return ui.input_task_button(
        id=id,
        label=label,
        icon=fa.icon_svg(icon),
        width="100%",
        class_="btn btn-primary",
    )


def error_notification(msg: str) -> None:
    ui.notification_show(
        msg,
        type="error",
        duration=5,
    )

    return None


def selection(
    id: str, label: ui.TagChild, choices: list[Any], selected: Any, **props
) -> ui.Tag:
    return ui.input_select(
        id=id,
        label=label,
        choices=choices,
        selected=selected,
        multiple=False,
        **props,
    )


def selectize(
    id: str, label: ui.TagChild, choices: list[Any], selected: Any = None, **props
) -> ui.Tag:
    return ui.input_selectize(
        id=id,
        label=label,
        choices=choices,
        selected=choices[0] if selected is None else selected,
        multiple=False,
        remove_button=False,
        options={
            "maxOptions": 200,
            "placeholder": "Enter text to search",
            "create": False,
        },
        **props,
    )


def page_sidebar(*args, sidebar: ui.Sidebar, **props) -> ui.Tag:
    return ui.page_sidebar(
        sidebar,
        *args,
        title=None,
        fillable=True,
        fillable_mobile=True,
        **props,
    )


def upload_handler(id: str, label: str) -> ui.Tag:
    return ui.input_file(
        id,
        label,
        accept=[".csv", ".txt"],
        multiple=False,
    )


def group_ui_elements(
    *args, title: ui.Tag | str, help_text: ui.Tag | str | ui.HTML
) -> ui.Tag:
    return ui.tags.div(
        ui.tags.div(
            ui.tags.h4(title, class_="m-0"),
            ui.popover(
                ui.span(
                    fa.icon_svg("circle-info"),
                    class_="text-muted",
                    style="cursor: pointer;",
                ),
                help_text,
                title=title,
            ),
            style="""
                display: flex;
                align-items: center;
                gap: 0.5rem;
                margin-bottom: 0.5rem;
            """,
        ),
        *args,
        ui.hr(),
    )


def anchor_tag(*args, href: str, **props) -> ui.Tag:
    return ui.tags.a(*args, href=href, target="_blank", **props)


def anchor_html(text: str, link: str) -> str:
    return f'<a href="{link}" target="_blank" rel="noopener noreferrer">{text}</a>'
