from typing import Any

import faicons as fa
from shiny import ui

from shared.consts import FILL_COLOUR


def sidebar(*args, **kwargs) -> ui.Sidebar:
    return ui.sidebar(
        *args,
        position="left",
        width=560,
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


def input_action_button(id: str, label: str, icon: str) -> ui.Tag:
    return ui.input_action_button(
        id=id,
        label=label,
        icon=fa.icon_svg(icon),
        width="100%",
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
    return ui.card(
        ui.card_header(
            title,
            ui.popover(
                ui.span(
                    fa.icon_svg("circle-info", fill=FILL_COLOUR),
                    style="position:absolute; top: 5px; right: 7px;",
                ),
                help_text,
                title=title,
            ),
        ),
        *args,
    )


def anchor_tag(*args, href: str, **props) -> ui.Tag:
    return ui.tags.a(*args, href=href, target="_blank", **props)


def anchor_html(text: str, link: str) -> str:
    return f'<a href="{link}" target="_blank" rel="noopener noreferrer">{text}</a>'
