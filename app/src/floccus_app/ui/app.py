"""Floccus Streamlit interface. Launch with `floccus ui`."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st
from biofloc_review.protocol import CHECKS, CRITERION_TYPES, LANGUAGES, ProtocolError

from floccus_app import branding
from floccus_app import results as res
from floccus_app import ui_logic
from floccus_app.i18n import t
from floccus_app.projects import ProjectError, create_project, list_projects, projects_home


def _home() -> Path:
    argv = sys.argv
    if "--home" in argv:
        return Path(argv[argv.index("--home") + 1])
    return projects_home()


st.set_page_config(page_title="Floccus", page_icon="🔬", layout="wide")
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
    st.divider()
    st.caption(t("theme_hint", lang))
    st.caption(t("footer", lang).format(version=branding.version(), license=branding.LICENSE,
                                        doi=branding.DOI, doi_url=branding.DOI_URL, repo=branding.REPO_URL))

st.markdown(f"## 🔬 {branding.APP_NAME} <span style='font-size:0.55em;opacity:0.6'>v{branding.version()}</span>",
            unsafe_allow_html=True)
st.caption(t("tagline", lang))

all_tabs = st.tabs([t(k, lang) for k in
                ("tab_start", "tab_project", "tab_protocol", "tab_run", "tab_results", "tab_review", "tab_llm")])
start_tab, tabs = all_tabs[0], all_tabs[1:]

with start_tab:
    st.write(t("start_what", lang))
    st.subheader(t("start_steps_title", lang))
    st.markdown(t("start_steps", lang).format(gentle=branding.GENTLE_SHELL_URL, repo=branding.REPO_URL))
    st.subheader(t("feedback_title", lang))
    st.write(t("feedback_text", lang))
    b1, b2 = st.columns(2)
    b1.link_button(t("feedback_bug", lang), branding.feedback_url("bug", t("tab_start", lang), lang))
    b2.link_button(t("feedback_idea", lang), branding.feedback_url("idea", t("tab_start", lang), lang))
    st.subheader(t("cite_title", lang))
    st.markdown(t("cite_text", lang).format(doi=branding.DOI, doi_url=branding.DOI_URL)
                + f" · {branding.LICENSE}")

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
            table, num_rows="dynamic", width="stretch", key=f"criteria-{current}",
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

run = res.load_results(project) if project else None


def _fmt(n: int) -> str:
    return f"{n:,}"


def _run_controls() -> None:
    from floccus_app import cost, pipeline

    counts = pipeline.read_counts(project)
    active = pipeline.active_run(project)

    st.subheader(t("run_exports", lang))
    st.caption(t("run_exports_help", lang))
    with st.form(f"upload-{current}", clear_on_submit=True):
        source = st.selectbox(t("source", lang), list(pipeline.SOURCES),
                              format_func=lambda s: res.SOURCE_LABELS.get(s, s.capitalize()))
        files = st.file_uploader(t("files", lang), accept_multiple_files=True,
                                 type=["csv", "xls", "xlsx"])
        if st.form_submit_button(t("save_files", lang)) and files:
            try:
                for f in files:
                    pipeline.add_export(project, source, f.name, f.getvalue())
                st.success(t("saved_files", lang))
            except pipeline.PipelineError as exc:
                st.error(str(exc))
    exports = pipeline.list_exports(project)
    if exports:
        for src, names in exports.items():
            st.write(f"**{res.SOURCE_LABELS.get(src, src)}:** {', '.join(names)}")
    else:
        st.caption(t("no_exports", lang))

    st.subheader(t("run_steps", lang))
    if active:
        st.warning(f"{t('running', lang)}: {active.get('step')}")
    limit = st.number_input(t("limit", lang), min_value=0, value=0, step=10, key=f"limit-{current}")
    cols = st.columns(4)
    steps = (("btn_import", "import"), ("btn_dedup", "dedup"), ("btn_l1", "screen-l1"), ("btn_l2", "screen-l2"))
    for col, (label, step) in zip(cols, steps):
        if col.button(t(label, lang), disabled=bool(active), width="stretch", key=f"{step}-{current}"):
            ui_logic.launch_step(home, current, step, int(limit) if step.startswith("screen") else None)
            st.toast(t("launched", lang))
            st.rerun()
    ident = counts.get("identification", {})
    dedup = counts.get("deduplication", {})
    final = counts.get("final_specificity_mode", {})
    cols = st.columns(4)
    cols[0].metric(t("m_identified", lang), _fmt(ident.get("total", 0)))
    cols[1].metric(t("m_unique", lang), _fmt(dedup.get("after_round2_ieee_merges", 0)))
    cols[2].metric(t("m_include", lang), _fmt(final.get("include", 0)))
    cols[3].metric(t("m_review", lang), _fmt(final.get("human_review", 0)))

    st.subheader(t("cost_title", lang))
    settings = ui_logic.llm_settings()
    n_unique = dedup.get("after_round2_ieee_merges")
    if settings["provider"] != "anthropic":
        st.caption(t("cost_local", lang))
    elif not n_unique:
        st.caption(t("cost_need_dedup", lang))
    else:
        c1, c2 = st.columns(2)
        price_in = c1.number_input(t("price_in", lang), min_value=0.0, value=0.0, step=0.5, key="price_in")
        price_out = c2.number_input(t("price_out", lang), min_value=0.0, value=0.0, step=0.5, key="price_out")
        st.caption(t("price_help", lang))
        n = min(n_unique, int(limit)) if limit else n_unique
        e = cost.estimate(protocol, n, cost.DEFAULT_LAYER2_SHARE, cost.DEFAULT_USER_CHARS, price_in, price_out)
        st.markdown(t("cost_result", lang).format(calls=_fmt(e.calls), tin=_fmt(e.input_tokens),
                                                  tout=_fmt(e.output_tokens), cost=f"{e.cost:,.2f}"))

    st.subheader(t("progress_title", lang))

    @st.fragment(run_every=2 if active else None)
    def _progress() -> None:
        prog = pipeline.read_progress(project)
        if not prog:
            return
        total = max(int(prog.get("total", 0)), 1)
        done = int(prog.get("done", 0))
        st.progress(min(done / total, 1.0), text=f"{prog.get('step')} · {done:,}/{total:,} · "
                                                    f"{t('status', lang)}: {prog.get('status')}")
        if prog.get("errors"):
            st.warning(f"{t('errors', lang)}: {prog['errors']} — {t('errors_warn', lang)}")
        tail = ui_logic.log_tail(home, current, str(prog.get("step", "")))
        if tail:
            with st.expander(t("log", lang)):
                st.code(tail)

    _progress()


with tabs[2]:
    if run and run.kind == "published":
        st.success(t("run_done_example", lang))
        steps = pd.DataFrame(res.pipeline_steps(run.counts, lang)).rename(columns={
            "step": t("step", lang), "what": t("what", lang), "in": t("input", lang), "out": t("output", lang)})
        st.dataframe(steps, hide_index=True, width="stretch")
        if run.info:
            st.write(f"**{t('run_model_used', lang)}:** {run.info.get('provider', '?').capitalize()} · "
                     f"`{run.info.get('generation_model', '?')}` · `{run.info.get('embedding_model', '?')}` · "
                     f"temperature {run.info.get('temperature', '?')}")
    elif project:
        _run_controls()
    else:
        st.info(t("no_projects", lang))

with tabs[3]:
    if run:
        if run.kind == "latest":
            st.caption(t("own_results_note", lang))
        c, f = run.counts, run.counts["final_specificity_mode"]
        cols = st.columns(6)
        for col, (key, value) in zip(cols, (
            ("m_identified", c["identification"]["total"]), ("m_unique", res._unique(c)),
            ("m_include", f["include"]), ("m_review", f["human_review"]),
            ("m_exclude", f["exclude"]), ("m_eligible", f["to_eligibility"]),
        )):
            col.metric(t(key, lang), _fmt(value))
        st.plotly_chart(res.fig_flow(c, lang), width="stretch")
        g1, g2 = st.columns(2)
        g1.plotly_chart(res.fig_sources(c, lang), width="stretch")
        g2.plotly_chart(res.fig_final_decisions(c, lang), width="stretch")
        g3, g4 = st.columns(2)
        g3.plotly_chart(res.fig_criteria(run.decisions, lang, [c.id for c in protocol.criteria]),
                        width="stretch")
        if run.eligible is not None:
            g4.plotly_chart(res.fig_clusters(run.eligible, lang), width="stretch")
        st.subheader(t("decisions_table", lang))
        table = res.active_decisions(run.decisions)
        options = sorted(table["final_decision"].dropna().unique())
        chosen = st.multiselect(t("filter_decision", lang), options, default=options, key="filter_decision")
        shown = table[table["final_decision"].isin(chosen)]
        st.caption(f"{_fmt(len(shown))} {t('records', lang)}")
        st.dataframe(shown, hide_index=True, width="stretch", height=360)
        st.download_button(t("download_csv", lang), shown.to_csv(index=False).encode("utf-8"),
                           file_name="screening_decisions.csv", mime="text/csv")
    else:
        st.info(f"🚧 {t('under_construction', lang)}")
        st.write(t("results_soon", lang))

with tabs[4]:
    if run and run.validation is not None:
        st.write(t("validation_intro", lang) + f" ({_fmt(len(run.validation))} {t('records', lang)})")
        v1, v2 = st.columns(2)
        v1.write(f"**{t('reviewer_1', lang)}**")
        v1.dataframe(res.validation_crosstab(run.validation, "reviewer_1"), width="stretch")
        v2.write(f"**{t('reviewer_2', lang)}**")
        v2.dataframe(res.validation_crosstab(run.validation, "reviewer_2"), width="stretch")
    else:
        st.info(f"🚧 {t('under_construction', lang)}")
        st.write(t("review_soon", lang))

with tabs[5]:
    st.markdown(t("model_used", lang))
    st.markdown(t("model_alternatives", lang))
    s = ui_logic.llm_settings()
    st.write(f"**{t('model_current', lang)}:** {s['provider'].capitalize()} · `{s['model']}`")
