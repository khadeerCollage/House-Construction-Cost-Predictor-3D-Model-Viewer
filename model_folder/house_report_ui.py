"""
The four result tabs under the blueprint — Room Schedule, Vastu Engineering, Cost & BOQ,
Best House Recommendation — drawn from floorplan_generator.house_report (measured from the
drawing), CostEngine.boq_for_plan (take-off priced and split into stages) and the engine's
own alternative layouts (the same re-plan machinery as Smart Re-Plan), so a "better design"
is a real design the owner can switch to with one tap.
"""

import csv
import io
from html import escape
from typing import Any, Dict, List, Optional

import streamlit as st

from floorplan_generator import house_report as hr
from model_folder.cost_engine import CostEngine

ALTS = "report_alternatives"          # {"plan_id", "options"} — designs our engine made for this plot
FLASH = "report_flash"

_CSS = """
<style>
.hr-chips{display:flex;flex-wrap:wrap;gap:8px;margin:4px 0 12px}
.hr-chip{background:#111827;border:1px solid #1F2937;border-radius:10px;padding:8px 12px;min-width:118px}
.hr-chip b{display:block;color:#F8FAFC;font-size:17px;line-height:1.2}
.hr-chip span{color:#94A3B8;font-size:11.5px}
.hr-gtitle{color:#94A3B8;font-size:12px;letter-spacing:.06em;text-transform:uppercase;margin:12px 0 6px}
.hr-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(178px,1fr));gap:8px}
.hr-card{background:#111827;border:1px solid #1F2937;border-radius:10px;padding:9px 10px;position:relative}
.hr-card .t{color:#F1F5F9;font-size:13px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;padding-right:44px}
.hr-card .shape{position:absolute;top:8px;right:9px}
.hr-card.np .t{padding-right:0}
.hr-card .a{color:#F8FAFC;font-size:16px;font-weight:700;margin-top:4px}
.hr-card .a small{color:#94A3B8;font-size:11.5px;font-weight:400}
.hr-card .d{color:#94A3B8;font-size:11.5px;margin-top:1px}
.hr-badges{display:flex;gap:5px;margin-top:6px;flex-wrap:wrap}
.hr-b{font-size:10.5px;border-radius:6px;padding:1px 6px;border:1px solid #334155;color:#CBD5E1}
.hr-b.ok{border-color:#166534;color:#86EFAC}.hr-b.low{border-color:#92400E;color:#FCD34D}
.hr-row{display:flex;align-items:center;gap:10px;padding:6px 0;border-bottom:1px solid #1F2937}
.hr-row .n{flex:1;color:#E2E8F0;font-size:13px}.hr-row .n small{color:#94A3B8;display:block;font-size:11.5px}
.hr-bar{height:6px;border-radius:3px;background:#1F2937;flex:0 0 90px;overflow:hidden}
.hr-bar i{display:block;height:100%;border-radius:3px}
.hr-pts{flex:0 0 64px;text-align:right;color:#FCA5A5;font-size:12.5px;font-variant-numeric:tabular-nums}
.hr-check{background:#111827;border:1px solid #1F2937;border-left-width:4px;border-radius:8px;padding:8px 12px;margin-bottom:6px}
.hr-check .r{color:#F1F5F9;font-size:13.5px;font-weight:600}.hr-check .e{color:#CBD5E1;font-size:12.5px}
.hr-check .w{color:#94A3B8;font-size:12px}.hr-check .f{color:#7DD3FC;font-size:12px}
.hr-split{display:flex;height:16px;border-radius:8px;overflow:hidden;margin:6px 0 4px}
.hr-split i{display:block;height:100%;border-right:2px solid #0B1220}
.hr-legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12px;color:#CBD5E1}
.hr-legend em{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
.hr-dim{display:grid;grid-template-columns:150px 1fr 44px;gap:10px;align-items:center;padding:7px 0;border-bottom:1px solid #1F2937}
.hr-dim .l{color:#E2E8F0;font-size:13px}.hr-dim .v{color:#F8FAFC;font-weight:700;text-align:right;font-size:13px}
.hr-dim .ev{grid-column:2 / 4;color:#94A3B8;font-size:11.5px;margin-top:-4px}
.hr-thumb{background:#0A2342;border-radius:8px;padding:4px;border:1px solid #334155}
.hr-thumb svg{width:100%;height:auto}
</style>
"""


def _money(v: float, country: str, compact: bool = True) -> str:
    return CostEngine.format_money(v, country or "India", compact=compact)


def _shape_svg(w: float, h: float, box: int = 34) -> str:
    """Tiny to-scale outline of a room (shape at a glance)."""
    k = box / max(w, h, 0.1)
    rw, rh = max(4, w * k), max(4, h * k)
    return (f'<svg class="shape" width="{box}" height="{box}" viewBox="0 0 {box} {box}">'
            f'<rect x="{(box - rw) / 2:.1f}" y="{(box - rh) / 2:.1f}" width="{rw:.1f}" height="{rh:.1f}" rx="2" '
            f'fill="#38BDF8" fill-opacity="0.18" stroke="#38BDF8" stroke-width="1.2"/></svg>')


def _bar(pct: float, colour: str) -> str:
    return f'<div class="hr-bar"><i style="width:{max(2, min(100, pct)):.0f}%;background:{colour}"></i></div>'


def _csv(rows: List[Dict]) -> bytes:
    buf = io.StringIO()
    if rows:
        w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return buf.getvalue().encode("utf-8-sig")


# =============================================================================
# Shared: alternative designs our engine makes for this plot
# =============================================================================

def _alternatives(plan, cost_params: Dict[str, Any]) -> Optional[List[Dict]]:
    cache = st.session_state.get(ALTS)
    if cache and cache.get("plan_id") == id(plan):
        return cache["options"]
    return None


def _compute_alternatives(plan, cost_params: Dict[str, Any]) -> None:
    from model_folder.intent_studio import build_options
    with st.spinner("Making other layouts for your plot and checking each one…"):
        opts = build_options(plan, [], cost_params, k=4)
    out = []
    for o in opts:
        if not o["report"].get("changes"):          # identical to the current design
            continue
        o["health"] = hr.health_card(o["plan"], o.get("cost"))
        out.append(o)
    st.session_state[ALTS] = {"plan_id": id(plan), "options": out}


def _use(option: Dict, cost_params: Dict[str, Any], why: str) -> None:
    from model_folder.intent_studio import commit
    commit(option, cost_params)
    st.session_state.pop(ALTS, None)
    st.session_state[FLASH] = f"✅ Switched to the {why}. The blueprint, downloads and every tab now show it — " \
                              "Undo is in the Smart Re-Plan section."


def _changes_text(option: Dict, limit: int = 4) -> str:
    ch = option["report"].get("changes") or []
    z = hr.ZONE_WORDS
    parts = [f"{c['room']}: {z.get(c['from'], c['from'])} → {z.get(c['to'], c['to'])}" for c in ch[:limit]]
    return "; ".join(parts) + (f" (+{len(ch) - limit} more)" if len(ch) > limit else "")


# =============================================================================
# 1. Room schedule
# =============================================================================

def render_room_schedule(plan) -> None:
    rs = hr.room_schedule(plan)
    t = rs["totals"]
    st.markdown(_CSS, unsafe_allow_html=True)
    chips = [(f"{t['carpet']:.1f} m²", f"carpet · {t['carpet'] * hr.SQFT:,.0f} sq ft"),
             (f"{t['built_up']:.1f} m²", f"built-up · {t['built_up'] * hr.SQFT:,.0f} sq ft"),
             (f"{t['efficiency']:.0f}%", "of built-up is usable"),
             (f"{t['circulation_pct']:.0f}%", "passages & foyer"),
             (f"{t['rooms']}", "spaces inside"),
             (f"{t['outdoor_m2']:.0f} m²", "outdoor features")]
    st.markdown('<div class="hr-chips">' + "".join(f'<div class="hr-chip"><b>{a}</b><span>{b}</span></div>'
                                                   for a, b in chips) + "</div>", unsafe_allow_html=True)
    low = [r for r in rs["rows"] if r["nbc"]["status"] == "low"]
    if low:
        st.warning("⚠️ **A little under NBC 2016 sizes (inside, wall to wall):** " +
                   "; ".join(f"{r['name']} — {r['nbc']['text'].split(': ', 1)[-1]}" for r in low) +
                   ".  \nFix it in **🧠 Smart Re-Plan**: tap the room → **➕ Bigger**.")
    else:
        st.success("✅ Every room meets the NBC 2016 Part 3 minimum sizes.")

    for _, title, rows in rs["groups"]:
        cards = []
        for r in rows:
            nb = r["nbc"]
            badge = {"ok": '<span class="hr-b ok">✓ NBC</span>', "low": '<span class="hr-b low">⚠ NBC</span>'}.get(
                nb["status"], "")
            cards.append(
                f'<div class="hr-card" title="{escape(nb["text"])}">{_shape_svg(r["w"], r["h"])}'
                f'<div class="t">{r["icon"]} {escape(r["name"])}</div>'
                f'<div class="a">{r["area"]:.1f} m² <small>· {r["sqft"]} sq ft</small></div>'
                f'<div class="d">{r["w"]:.2f} × {r["h"]:.2f} m · {r["w_ft"]} × {r["h_ft"]}</div>'
                f'<div class="hr-badges"><span class="hr-b">{hr.ZONE_WORDS.get(r["zone"], r["zone"])}</span>{badge}</div>'
                f'</div>')
        st.markdown(f'<div class="hr-gtitle">{title}</div><div class="hr-grid">{"".join(cards)}</div>',
                    unsafe_allow_html=True)
    st.caption("Sizes are inside, wall to wall (what you can furnish); the drawing labels show wall centre-lines. "
               "Hover a card for its NBC rule.")
    table = [{"Room": r["name"], "Inside size (m)": f"{r['w']:.2f} × {r['h']:.2f}",
              "Inside size (ft)": f"{r['w_ft']} × {r['h_ft']}", "Area (m²)": r["area"], "Area (sq ft)": r["sqft"],
              "Zone": r["zone"], "NBC 2016": r["nbc"]["text"]} for r in rs["rows"]]
    with st.expander("📋 Full room table & download"):
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download room schedule (.csv)", _csv(table),
                           file_name=f"{plan.bhk_config}_room_schedule.csv", mime="text/csv")


# =============================================================================
# 2. Vastu engineering
# =============================================================================

def _score_ring(score: float, label: str) -> str:
    c = "#22C55E" if score >= 85 else "#84CC16" if score >= 70 else "#F59E0B" if score >= 50 else "#EF4444"
    r, circ = 44, 2 * 3.14159 * 44
    return (f'<svg width="120" height="120" viewBox="0 0 120 120"><circle cx="60" cy="60" r="{r}" fill="none" '
            f'stroke="#1F2937" stroke-width="10"/><circle cx="60" cy="60" r="{r}" fill="none" stroke="{c}" '
            f'stroke-width="10" stroke-linecap="round" stroke-dasharray="{circ * score / 100:.1f} {circ:.1f}" '
            f'transform="rotate(-90 60 60)"/><text x="60" y="60" fill="#F8FAFC" font-size="26" font-weight="700" '
            f'text-anchor="middle">{score:.0f}</text><text x="60" y="80" fill="#94A3B8" font-size="11" '
            f'text-anchor="middle">{label}</text></svg>')


def render_vastu(plan, cost_params: Dict[str, Any]) -> None:
    from floorplan_generator.vastu_analyzer import VastuAnalyzer
    st.markdown(_CSS, unsafe_allow_html=True)
    if st.session_state.get(FLASH):
        st.success(st.session_state.pop(FLASH))       # shown once
    analyzer = VastuAnalyzer(plan)
    rec = analyzer.get_house_recommendation()
    led = hr.vastu_ledger(plan)
    vc = hr.vastu_checks(plan)
    cnt = vc["counts"]
    score = float(plan.vastu_score or 0)

    head1, head2 = st.columns([1, 3])
    with head1:
        st.markdown(_score_ring(score, "of 100"), unsafe_allow_html=True)
    with head2:
        st.markdown(f"#### {rec['vastu_rating_grade']}")
        st.markdown(f"**{cnt['pass']} rules met** · {cnt['warn']} to watch · {cnt['fail']} to fix — every rule "
                    "below is measured on your own drawing, and the score is the same formula the designer used.")
        lost_total = 100 - score
        if lost_total < 0.5:
            st.caption("A perfect score: every room sits in its ideal zone.")
        else:
            biggest = led["items"][0] if led["items"] else None
            if biggest:
                st.caption(f"The {lost_total:.1f} missing points come mostly from **{biggest['item']}** in the "
                           f"{hr.ZONE_WORDS.get(biggest['zone'], biggest['zone'])} ({biggest['lost']:.1f} pts).")

    c1, c2 = st.columns([1.15, 1], gap="medium")
    with c1:
        st.markdown("###### 🧭 Your house on its Vastu mandala")
        st.markdown(hr.vastu_plan_svg(plan), unsafe_allow_html=True)
        st.caption("The 3 × 3 grid is laid on the house itself (not the plot). Colours show how well each room "
                   "sits in its zone; the dashed circle is the Brahmasthan, the yellow mark the main door.")
    with c2:
        st.markdown("###### 📉 Where the points go")
        rows = []
        for it in led["items"]:
            if it["lost"] < 0.05:
                continue
            best = ", ".join(it["best"][:3]) if it["best"] else "—"
            colour = "#EF4444" if it["taboo"] or it["score"] < 0 else "#F59E0B" if it["score"] < 5 else "#84CC16"
            where = hr.ZONE_WORDS.get(it["zone"], it["zone"]) if it["key"] != "ENTRANCE" else f"score {it['score']:+.0f}/10"
            rows.append(f'<div class="hr-row"><div class="n">{escape(it["item"])}<small>in {where}'
                        f'{"" if it["key"] == "ENTRANCE" else " · best: " + best}</small></div>'
                        f'{_bar(it["lost"] * 10, colour)}<div class="hr-pts">−{it["lost"]:.1f}</div></div>')
        for p in led["penalties"]:
            rows.append(f'<div class="hr-row"><div class="n">{escape(p["item"])}</div>{_bar(p["lost"] * 10, "#EF4444")}'
                        f'<div class="hr-pts">−{p["lost"]:.1f}</div></div>')
        if rows:
            st.markdown("".join(rows[:9]), unsafe_allow_html=True)
            if len(rows) > 9:
                with st.expander(f"{len(rows) - 9} smaller items"):
                    st.markdown("".join(rows[9:]), unsafe_allow_html=True)
        st.caption(f"100 − {100 - led['score']:.1f} = **{led['score']:.1f}**. Heavier rooms count more "
                   "(master bedroom, kitchen, main door ×3; pooja ×2.5; living ×2).")

    # ── a better-Vastu version, made by the engine ──────────────────────
    st.markdown("###### ✨ Can my house get a better Vastu?")
    from model_folder.intent_studio import plan_supports_intents
    if not plan_supports_intents(plan):
        st.info("This plot used the compact fallback layout, so other arrangements can't be generated for it.")
    else:
        opts = _alternatives(plan, cost_params)
        if opts is None:
            st.button("🔍 Find a better-Vastu version of my house", key="vastu_find",
                      on_click=_compute_alternatives, args=(plan, cost_params))
            st.caption("Our designer re-arranges the same rooms on the same plot and keeps the best result.")
        else:
            best = max(opts, key=lambda o: o["plan"].vastu_score, default=None)
            if best is None or best["plan"].vastu_score < score + 0.5:
                st.success(f"✅ Your layout already has the best Vastu ({score:.0f}/100) our designer can reach for "
                           f"this plot and these rooms — {len(opts)} other arrangements scored lower.")
            else:
                _option_card(best, plan, cost_params, "better-Vastu design", key="vastu_use",
                             headline=f"Vastu {score:.0f} → **{best['plan'].vastu_score:.0f}** "
                                      f"(+{best['plan'].vastu_score - score:.1f})")

    st.markdown("###### 📏 Vastu rules checked on your drawing")
    border = {"pass": "#22C55E", "warn": "#F59E0B", "fail": "#EF4444"}
    icon = {"pass": "✅", "warn": "⚠️", "fail": "❌"}
    html = []
    for c in sorted(vc["checks"], key=lambda c: {"fail": 0, "warn": 1, "pass": 2}[c["status"]]):
        html.append(f'<div class="hr-check" style="border-left-color:{border[c["status"]]}">'
                    f'<div class="r">{icon[c["status"]]} {escape(c["rule"])}</div><div class="e">{escape(c["evidence"])}</div>'
                    f'<div class="w">{escape(c["why"])}</div>' + (f'<div class="f">👉 {escape(c["fix"])}</div>' if c["fix"] else "")
                    + "</div>")
    st.markdown("".join(html), unsafe_allow_html=True)
    st.markdown("**Check these on site** (a drawing cannot show them):\n" +
                "\n".join(f"- {s}" for s in vc["site_only"]))

    with st.expander("📜 Classical audit — 32-pada door, five elements, remedies & certificate"):
        entrance = analyzer.audit_entrance()
        st.markdown(f"**Main door:** pada {entrance['pada_id']} · {entrance['deity']} ({entrance['direction']} side) · "
                    f"{entrance['rating'].replace('_', ' ').title()} ({entrance['score']:+d}/10) — {entrance['effect']}")
        bal = analyzer.get_elemental_balance()
        cols = st.columns(len(bal))
        for col, (name, b) in zip(cols, bal.items()):
            col.metric(name.split()[0], f"{b['percentage']:.0f}%", b["health"])
        defects = analyzer.get_defects_and_remedies()
        if defects:
            st.markdown("**Remedies without demolition (Parihara):**")
            for d in defects:
                st.markdown(f"- **{d['defect']}** [{d['severity']}] — {d['impact']}  \n  *Remedy:* {d['remedy']}")
        else:
            st.success("No classical defects (doshas) found.")
        st.markdown(analyzer.render_mandala_svg(), unsafe_allow_html=True)
        st.download_button("📜 Download Vastu audit certificate (.txt)", analyzer.generate_certificate_text(),
                           file_name=f"{plan.bhk_config}_Vastu_Audit_Certificate.txt", mime="text/plain")


def _option_card(option: Dict, plan, cost_params: Dict[str, Any], why: str, key: str, headline: str) -> None:
    country = cost_params.get("country") or "India"
    a, b = st.columns([1, 1.3], gap="medium")
    with a:
        st.markdown(f'<div class="hr-thumb">{option["svg"]}</div>', unsafe_allow_html=True)
    with b:
        st.markdown(headline)
        moved = _changes_text(option)
        if moved:
            st.markdown(f"**What moves:** {moved}")
        d = option.get("d_cost", 0)
        st.markdown(f"**Cost:** {'same as now' if abs(d) < 1000 else ('+' if d > 0 else '−') + _money(abs(d), country)}"
                    + (f" · carpet {option['d_carpet']:+.1f} m²" if abs(option.get("d_carpet", 0)) >= 0.1 else ""))
        h_now = hr.health_card(plan)
        h_new = option.get("health") or hr.health_card(option["plan"])
        diffs = [(n["label"], o["score"] - n["score"]) for n, o in zip(h_now["dims"], h_new["dims"])
                 if abs(o["score"] - n["score"]) >= 3]
        if diffs:
            st.markdown("**Also changes:** " + ", ".join(f"{l} {d:+.0f}" for l, d in diffs))
        for w in (option["report"].get("warnings") or [])[:2]:
            st.caption(f"⚠️ {w}")
        st.button(f"✅ Use this design", key=key, type="primary", on_click=_use, args=(option, cost_params, why))


# =============================================================================
# 3. Cost & material BOQ
# =============================================================================

def render_cost(plan, cost_data: Dict[str, Any], cost_params: Dict[str, Any], quality: str, floors: int) -> None:
    import altair as alt
    st.markdown(_CSS, unsafe_allow_html=True)
    country = cost_data["country"]
    disp = cost_data["display"]
    md = lambda s: s.replace("$", r"\$")   # noqa: E731
    boq = CostEngine.boq_for_plan(plan, cost_data, quality)

    chips = [(disp["total_compact"], f"total project cost · {disp['rate']}"),
             (_money(cost_data["construction_cost"], country), "construction (hard) cost")]
    if boq.get("available"):
        chips += [(_money(boq["materials_total"], country),
                   f"materials · {100 * boq['materials_total'] / max(1, boq['building_cost']):.0f}% of building"),
                  (f"≈ {boq['weeks'] / 4.33:.0f} months", f"build time · {boq['weeks']} weeks")]
    st.markdown('<div class="hr-chips">' + "".join(f'<div class="hr-chip"><b>{md(a)}</b><span>{md(b)}</span></div>'
                                                   for a, b in chips) + "</div>", unsafe_allow_html=True)
    st.caption(md(f"{cost_data['region']}, {country} · {cost_data['quality_level']} · excludes land. "
                  f"{cost_data['description']}"))

    if boq.get("available"):
        st.markdown("###### 🧱 Main materials for your house — measured from the drawing")
        site_parts = {k: 0.0 for k in ("Cement", "Bricks")}
        for l in boq["lines"]:
            if l["group"] == "Site works":
                site_parts["Cement"] += l["cement_bags"]
                site_parts["Bricks"] += l["bricks"]
        cards = []
        for mt in boq["materials"]:
            q = mt["qty"]
            qty = f"{q / 1000:.2f} t" if mt["unit"] == "kg" and q >= 1000 else f"{q:,.0f} {mt['unit']}"
            note = f"{mt['per_sqft']:g} per sq ft"
            if mt.get("thumb_qty"):
                note += f" · thumb rule {mt['thumb_qty']:,.0f}"
            if site_parts.get(mt["key"]):
                note += f" · incl. {site_parts[mt['key']]:,.0f} for the compound wall"
            cards.append(f'<div class="hr-card np"><div class="t">{mt["label"]}</div>'
                         f'<div class="a">{qty}</div><div class="d" style="color:#E2E8F0">'
                         f'{_money(mt["amount"], country)} at ₹{mt["rate"]:,.0f}'
                         f'{" each" if mt["unit"] == "nos" else "/" + mt["unit"].rstrip("s")}</div>'
                         f'<div class="d">{note}</div></div>')
        st.markdown(f'<div class="hr-grid">{"".join(cards)}</div>', unsafe_allow_html=True)
        f = boq["finishes"]
        st.caption(f"Also: concrete {f['concrete_m3']:.1f} m³ · brickwork {f['brickwork_m3']:.1f} m³ · tiles "
                   f"{f['floor_tiles_m2']:.0f} m² · paint ≈ {f['paint_litres']:.0f} L · wall putty ≈ {f['putty_kg']:.0f} kg. "
                   "Rates at today's local prices; quantities include normal wastage.")

        st.markdown("###### 📅 Stage-by-stage cost & cash plan")
        stages = boq["stages"]
        chart = alt.Chart(alt.Data(values=[
            {"Stage": f"{i + 1}. {s['label']}", "Cost": s["amount"], "Share": f"{s['pct']:.0f}%",
             "Weeks": f"wk {s['start_week'] + 1}–{s['start_week'] + s['weeks']}",
             "Amount": _money(s["amount"], country), "order": i} for i, s in enumerate(stages)])).mark_bar(
            color="#38BDF8", cornerRadiusEnd=4, height=14).encode(
            y=alt.Y("Stage:N", sort=alt.SortField("order"), title=None, axis=alt.Axis(labelLimit=260)),
            x=alt.X("Cost:Q", title=None, axis=alt.Axis(labels=False, ticks=False, grid=False),
                    scale=alt.Scale(domain=[0, 1.28 * max(s["amount"] for s in stages)])),   # room for labels
            tooltip=["Stage:N", "Amount:N", "Share:N", "Weeks:N"])
        labels = chart.mark_text(align="left", dx=4, color="#CBD5E1", fontSize=11).encode(text="Amount:N")
        st.altair_chart((chart + labels).properties(height=30 * len(stages)).configure_view(stroke=None),
                        use_container_width=True)
        stage_rows = [{"#": i + 1, "Stage": s["label"], "What happens": s["what"],
                       "Weeks": f"{s['start_week'] + 1}–{s['start_week'] + s['weeks']}",
                       "Cement (bags)": s["cement_bags"] or "", "Steel (kg)": s["steel_kg"] or "",
                       "Bricks": s["bricks"] or "", "Cost": _money(s["amount"], country),
                       "Paid so far": _money(s["cumulative"], country)} for i, s in enumerate(stages)]
        st.dataframe(stage_rows, use_container_width=True, hide_index=True)

        st.markdown("###### 💰 Where the money goes")
        soft = cost_data.get("soft_costs", {})
        soft_total = sum(soft.get(k, 0) for k in ("fees", "approvals", "contingency", "tax"))
        parts = [("Materials (measured)", boq["materials_total"], "#38BDF8"),
                 ("Labour, contractor & tools", boq["labour_total"], "#A78BFA"),
                 ("Site works & amenities", boq["site_cost"], "#34D399"),
                 ("Fees, approvals & contingency", soft_total, "#FBBF24")]
        tot = max(1.0, sum(v for _, v, _ in parts))
        st.markdown('<div class="hr-split">' + "".join(
            f'<i style="width:{100 * v / tot:.1f}%;background:{c}" title="{n}"></i>' for n, v, c in parts if v > 0)
            + '</div><div class="hr-legend">' + "".join(
            f'<span><em style="background:{c}"></em>{n}: <b>{_money(v, country)}</b> ({100 * v / tot:.0f}%)</span>'
            for n, v, c in parts if v > 0) + "</div>", unsafe_allow_html=True)

        with st.expander("📋 Detailed bill of quantities — every line and how it was measured"):
            rows = [{"Work": l["group"], "Item": l["item"], "Quantity": l["qty"], "Unit": l["unit"],
                     "How it was measured": l["how"], "Cement (bags)": round(l["cement_bags"], 1),
                     "Steel (kg)": round(l["steel_kg"]), "Bricks": round(l["bricks"]),
                     "Material cost": round(l["amount"])} for l in boq["lines"]]
            st.dataframe(rows, use_container_width=True, hide_index=True, height=420)
            st.download_button("⬇️ Download BOQ (.csv)", _csv(rows), file_name=f"{plan.bhk_config}_BOQ.csv",
                               mime="text/csv")
            st.caption("Mixes: RCC M20 1:1.5:3, PCC 1:4:8, brickwork CM 1:6, plaster 1:6 inside / 1:4 outside. "
                       "Steel by member (footing 70, column 170, beam 125, slab 85 kg/m³). "
                       "Material costs exclude labour; the stage costs above include it.")
    else:
        st.info(boq.get("reason", ""))
        materials = cost_data["materials"]
        cols = st.columns(len(materials) or 1)
        for col, (k, info) in zip(cols, materials.items()):
            col.metric(info.get("label", k), f"{info.get('quantity', 0):,} {info.get('unit', '')}",
                       _money(info.get("cost", 0), country))

    with st.expander("🧮 How the total is calculated · cost in other cities"):
        st.table([{"Step": s["step"], "Basis": md(s["basis"]), "Value": md(s["value"])} for s in cost_data["calculation"]])
        tot = max(1.0, cost_data["total_cost"])
        st.table([{"Item": k, "Amount": md(_money(v, country, False)), "Share": f"{100 * v / tot:.1f}%"}
                  for k, v in cost_data["cost_breakdown"].items()])
        matrix = CostEngine.region_matrix_for_plan(plan, country, floors)
        st.dataframe([{"City / Region": ("▶ " if r["region"] == cost_data["region"] else "") + r["region"],
                       "Index": f"×{r['index']:.2f}",
                       **{q: _money(r[q], country) for q in ("Economy", "Standard", "Premium", "Luxury")}}
                      for r in matrix], use_container_width=True, hide_index=True,
                     height=min(420, 38 + 35 * len(matrix)))
    for note in cost_data.get("notes", []):
        st.caption(md(note))


# =============================================================================
# 4. Best house recommendation
# =============================================================================

def render_recommendations(plan, cost_data: Dict[str, Any], cost_params: Dict[str, Any], quality: str) -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
    if st.session_state.get(FLASH):
        st.success(st.session_state.pop(FLASH))       # shown once
    country = cost_data["country"]
    hc = hr.health_card(plan, cost_data)
    a, b = st.columns([1, 3])
    with a:
        st.markdown(_score_ring(hc["overall"], "house health"), unsafe_allow_html=True)
    with b:
        st.markdown(f"#### {hc['grade']} house for your plot")
        st.markdown("Six things that make a home good to live in, each **measured on your drawing**. "
                    "The same score is used to compare every design our engine can make for your plot.")
    colours = lambda s: "#22C55E" if s >= 85 else "#84CC16" if s >= 70 else "#F59E0B" if s >= 50 else "#EF4444"  # noqa: E731
    st.markdown("".join(
        f'<div class="hr-dim"><div class="l">{d["icon"]} {escape(d["label"])}</div>{_bar(d["score"], colours(d["score"]))}'
        f'<div class="v">{d["score"]}</div><div class="ev">{escape(d["evidence"])}</div></div>' for d in hc["dims"]),
        unsafe_allow_html=True)

    st.markdown("###### 🏆 The best design for your plot")
    from model_folder.intent_studio import plan_supports_intents
    if not plan_supports_intents(plan):
        st.info("This plot used the compact fallback layout, so other arrangements can't be compared.")
    else:
        opts = _alternatives(plan, cost_params)
        if opts is None:
            st.button("🔍 Compare my design with other layouts for this plot", key="rec_find", type="primary",
                      on_click=_compute_alternatives, args=(plan, cost_params))
            st.caption("Our designer re-arranges the same rooms on the same plot in different ways; "
                       "each result gets the same health check and cost estimate.")
        else:
            rows = [{"Design": "▶ Your current design", "Health": hc["overall"], "Vastu": round(plan.vastu_score),
                     **{d["label"]: d["score"] for d in hc["dims"] if d["key"] in ("light", "efficiency")},
                     "Cost": cost_data["display"]["total_compact"], "Rooms moved": 0}]
            for o in opts:
                h = o["health"]
                rows.append({"Design": o["report"].get("label", "Alternative").split(" —")[0],
                             "Health": h["overall"], "Vastu": round(o["plan"].vastu_score),
                             **{d["label"]: d["score"] for d in h["dims"] if d["key"] in ("light", "efficiency")},
                             "Cost": (o.get("cost") or {}).get("display", {}).get("total_compact", "—"),
                             "Rooms moved": len(o["report"].get("changes") or [])})
            st.dataframe(rows, use_container_width=True, hide_index=True)
            best = max(opts, key=lambda o: (o["health"]["overall"], o["plan"].vastu_score), default=None)
            if best is None or best["health"]["overall"] <= hc["overall"]:
                st.success(f"✅ **Your current design is the best** of {len(opts) + 1} layouts our engine made "
                           f"for this plot (health {hc['overall']}).")
            else:
                _option_card(best, plan, cost_params, "recommended design", key="rec_use",
                             headline=f"**Recommended:** house health {hc['overall']} → "
                                      f"**{best['health']['overall']}**, Vastu {plan.vastu_score:.0f} → "
                                      f"{best['plan'].vastu_score:.0f}")

    st.markdown("###### 💡 Advice for your house")
    for icon, text in _advice(plan, cost_data, cost_params, quality):
        st.markdown(f"{icon} {text}")


def _advice(plan, cost_data: Dict[str, Any], cost_params: Dict[str, Any], quality: str) -> List[tuple]:
    """Measured, plan-specific advice: structure, future floor, fixes, light, amenities, budget."""
    out = []
    country = cost_data["country"]
    eng = getattr(plan, "engineering", None) or {}
    if eng.get("columns"):
        out.append(("🏗️", f"**Structure:** {eng['columns']} RCC columns (230 × 300 mm), {eng.get('slab_mm')} mm roof "
                          f"slab, beams 230 × {eng.get('beam_mm', [0, 0])[1]} mm; largest room span "
                          f"{eng.get('max_span_m', 0):.1f} m. Get the bar bending schedule signed by a structural engineer."))
    floors = int(getattr(plan, "num_floors", 1) or 1)
    if floors == 1 and country == "India":
        try:
            now = CostEngine.boq_for_plan(plan, cost_data, quality)
            fut = CostEngine.boq_for_plan(plan, cost_data, quality, design_floors=2)
            f_now = sum(l["amount"] for l in now["lines"] if l["stage"] == "foundation")
            f_fut = sum(l["amount"] for l in fut["lines"] if l["stage"] == "foundation")
            extra = (f_fut - f_now) * (1 + CostEngine.STAGE_LABOUR["foundation"])
            if extra > 0:
                out.append(("⬆️", f"**Planning a first floor later?** Size the footings and columns for G+1 now — "
                                  f"about **{_money(extra, country)}** more today, instead of strengthening the "
                                  "foundation later (which is rarely possible)."))
        except Exception:
            pass
    rs = hr.room_schedule(plan)
    if rs["totals"]["nbc_low"]:
        out.append(("📏", "**Bring these up to NBC size:** " + ", ".join(rs["totals"]["nbc_low"]) +
                          " — Smart Re-Plan → tap the room → ➕ Bigger."))
    for n in getattr(plan, "design_notes", []) or []:
        if n.startswith("Mechanical exhaust") or n.startswith("Daylight warning"):
            out.append(("🌬️", n))
    if rs["totals"]["circulation_pct"] > 14:
        out.append(("🚶", f"Passages take {rs['totals']['circulation_pct']:.0f}% of the floor — in Smart Re-Plan, "
                          "making the living room bigger lets the designer absorb some of it."))
    try:
        from floorplan_generator.site_feasibility import check_site
        gp = dict(getattr(plan, "generation_params", {}) or {})
        comps = gp.get("additional_components") or {}
        fit = check_site(plan.plot_width, plan.plot_height, plan.orientation, plan.bhk_config,
                         gp.get("family_profile"), gp.get("building_type", "Independent House"),
                         floors, gp.get("quality_level", "Standard"), plan.vastu_enabled,
                         gp.get("has_lift", False), gp.get("lift_capacity", "4 Person"),
                         gp.get("stair_type", "dog_leg"), components=comps, unlock_hints=False)
        from floorplan_generator.site_feasibility import LABEL
        can = [LABEL[k] for k, f in fit.amenities.items()
               if f.status == "ok" and not (comps.get(k) or {}).get("enabled") and k != "generator_room"]
        if can:
            out.append(("🌿", "**Also fits on your plot:** " + ", ".join(can) +
                              " — switch them on in the amenities section and generate again."))
    except Exception:
        pass
    comp = cost_data.get("comparison") or {}
    q = cost_data.get("quality_level")
    if comp and q in comp:
        now = comp[q]
        from model_folder.cost_engine import QUALITY_SHORT
        alts = [(QUALITY_SHORT.get(k, k), v - now) for k, v in comp.items() if k != q]
        txt = " · ".join(f"{name}: {'+' if d > 0 else '−'}{_money(abs(d), country)}" for name, d in alts)
        out.append(("💰", f"**Finish level changes the price most:** {txt} (same house, same city)."))
    return out
