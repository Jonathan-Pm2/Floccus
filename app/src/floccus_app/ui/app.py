"""Floccus Streamlit interface. Launch with `floccus ui`."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st
from biofloc_review.protocol import CHECKS, CRITERION_TYPES, LANGUAGES, ProtocolError

from floccus_app import ui_logic
from floccus_app.i18n import t
from floccus_app.projects import ProjectError, create_project, list_projects, projects_home


def _home() -> Path:
    argv = sys.argv
    if "--home" in argv:
        return Path(argv[argv.index("--home") + 1])
    return projects_home()


st.set_page_config(page_title="Floccus", layout="wide")
home = _home()

with st.sidebar:
    lang = st.radio(t("language", "es") + " / Language", LANGUAGES, horizontal=True, key="lang",
                    format_func=lambda x: {"es": "Español", "en": "English"}[x])
    st.title(t("app_title", lang))
    projects = list_projects(home)
    names = [folder for folder, _ in projects]
    # A project created in the form below is selected on the next run, before
    # the selector exists (Streamlit forbids changing a drawn widget's key).
    if st.session_state.get("pending_project") in names:
        st.session_state["project"] = st.session_state.pop("pending_project")
    current = st.selectbox(t("project", lang), names, key="project") if names else None
    if not names:
        st.info(t("no_projects", lang))
    with st.expander(t("new_project", lang), expanded=not names):
        with st.form("new_project"):
            new_name = st.text_input(t("project_name", lang))
            start = st.radio(t("start_from", lang), ["blank", "biofloc"],
                             format_func=lambda x: t("biofloc_example" if x == "biofloc" else "blank", lang))
            if st.form_submit_button(t("create", lang)):
                try:
                    create_project(home, new_name.strip(), "biofloc" if start == "biofloc" else None)
                    st.session_state["pending_project"] = new_name.strip()
                    st.rerun()
                except (ProjectError, ProtocolError) as exc:
                    st.error(str(exc))

tabs = st.tabs([t(k, lang) for k in
                ("tab_project", "tab_protocol", "tab_run", "tab_results", "tab_review", "tab_llm")])

project = home / current if current else None
protocol = ui_logic.load_project_protocol(project) if project else None

with tabs[0]:
    if protocol:
        st.subheader(protocol.name)
        st.write(f"**{t('topic', lang)}:** {protocol.topic}")
        st.write(f"**{t('prompt_language', lang)}:** {protocol.language}")
        counts = Counter(c.type for c in protocol.criteria)
        cols = st.columns(3)
        for col, ctype in zip(cols, CRITERION_TYPES):
            col.metric(t(ctype, lang), counts.get(ctype, 0))
        st.caption(f"{t('folder', lang)}: {project.resolve()}")
    else:
        st.info(t("no_projects", lang))

with tabs[1]:
    if protocol:
        name = st.text_input(t("protocol_name", lang), protocol.name, key=f"pname-{current}")
        topic = st.text_input(t("topic", lang), protocol.topic, key=f"ptopic-{current}")
        plang = st.selectbox(t("prompt_language", lang), LANGUAGES, index=LANGUAGES.index(protocol.language),
                             key=f"plang-{current}")
        st.caption(t("criteria_help", lang))
        table = pd.DataFrame(ui_logic.protocol_to_rows(protocol), columns=ui_logic.COLUMNS)
        edited = st.data_editor(
            table, num_rows="dynamic", use_container_width=True, key=f"criteria-{current}",
            column_config={
                "type": st.column_config.SelectboxColumn(options=list(CRITERION_TYPES), required=True),
                "check": st.column_config.SelectboxColumn(options=["", *CHECKS]),
                "question": st.column_config.TextColumn(width="large"),
                "justification": st.column_config.TextColumn(width="medium"),
            },
        )
        rows = edited.to_dict("records")
        c1, c2 = st.columns(2)
        if c1.button(t("validate", lang)):
            try:
                ui_logic.rows_to_protocol(name, plang, topic, rows)
                st.success(t("valid", lang))
            except ProtocolError as exc:
                st.error(f"{t('invalid', lang)}: {exc}")
        if c2.button(t("save", lang), type="primary"):
            error = ui_logic.save_rows(project, name, plang, topic, rows)
            if error:
                st.error(f"{t('invalid', lang)}: {error}")
            else:
                st.success(t("saved", lang))

for tab, key in ((tabs[2], "run_soon"), (tabs[3], "results_soon"), (tabs[4], "review_soon")):
    with tab:
        st.info(f"🚧 {t('under_construction', lang)}")
        st.write(t(key, lang))

with tabs[5]:
    s = ui_logic.llm_settings()
    st.write(f"**{t('provider', lang)}:** {s['provider']}")
    st.write(f"**{t('model', lang)}:** {s['model']}")
    st.write(f"**{t('url', lang)}:** {s['url']}")
    st.write(f"**{t('api_key', lang)}:** {s['api_key']}")
    st.caption(t("llm_help", lang))
