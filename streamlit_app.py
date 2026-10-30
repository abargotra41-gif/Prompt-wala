"""
app.py
AI-Powered Kirana Ordering Assistant - Streamlit Educational Prototype.
Unified End-to-End Workflow:
1. Inbound WhatsApp Message Entry (with sample picker, safety filters & blank reset)
2. Auto-Selected Grounded Order Review (with editable quantities, spelling mistake 'Do you mean this?', candidate clarification, & drop item option)
3. Store-History Reorder Recommendations (added seamlessly to the final order)
4. Payment Method Selection (Pay Now vs Pay Later):
   - Pay Now: Immediate settlement without credit check.
   - Pay Later: ML Credit Period Assessment (Logistic Regression vs Decision Tree),
     ₹10,000 Overdue Balance Guardrail & ₹30,000 Transaction Ceiling Guardrail.
   - Exact Grand Total, Affirmative Human Confirmation & 1-Second Money/Coin Animation on Dispatch.
"""

from __future__ import annotations
import csv
import os
import uuid
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Set

import pandas as pd
import streamlit as st

from catalogue import Catalogue, Product
from conversion import UnitConverter
from calculations import calculate_line_item, calculate_grand_total, LineCalculation
from validation import OrderLine, AuditLogger, validate_and_sanitize_input
from order_parser import OrderParser
from predictor import ReorderPredictor, ReorderNudge
from credit_predictor import CreditPeriodPredictor, CreditAssessment


# Page configuration
st.set_page_config(
    page_title="Kirana AI Ordering Assistant",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Singletons
@st.cache_resource
def get_catalogue() -> Catalogue:
    return Catalogue()


@st.cache_resource
def get_converter() -> UnitConverter:
    return UnitConverter()


@st.cache_resource
def get_predictor() -> ReorderPredictor:
    cat = get_catalogue()
    return ReorderPredictor(cat)


@st.cache_resource
def get_credit_predictor() -> CreditPeriodPredictor:
    return CreditPeriodPredictor()


@st.cache_data
def load_stores_metadata() -> List[dict]:
    stores = []
    csv_path = os.path.join(os.path.dirname(__file__), "data", "Stores.csv")
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                stores.append(r)
    return stores


@st.cache_data
def load_sample_messages() -> List[dict]:
    msgs = []
    path = os.path.join(os.path.dirname(__file__), "data", "WhatsApp_Messages.csv")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                msgs.append(r)
    return msgs


catalogue = get_catalogue()
converter = get_converter()
predictor = get_predictor()
credit_predictor = get_credit_predictor()
stores_data = load_stores_metadata()
backend_messages = load_sample_messages()

order_parser = OrderParser(
    catalogue=catalogue,
    converter=converter,
    store_history=predictor.store_cycle_history,
    store_sku_history=predictor.store_sku_history,
)

stores_dict = {str(s["Store_ID"]): s for s in stores_data}


# -------------------------------------------------------------
# Session State Initialization
# -------------------------------------------------------------
if "audit_logger" not in st.session_state:
    st.session_state.audit_logger = AuditLogger()

if "order_lines" not in st.session_state:
    st.session_state.order_lines = []

if "order_meta" not in st.session_state:
    st.session_state.order_meta = {}

if "raw_message_input" not in st.session_state:
    st.session_state.raw_message_input = "Bhaiya 2 dhaba Fortune sunflower oil 1L aur 5 Maggi bhej dena"

if "selected_store_id" not in st.session_state:
    st.session_state.selected_store_id = "STORE001"

if "submission_completed" not in st.session_state:
    st.session_state.submission_completed = False

if "last_submission_details" not in st.session_state:
    st.session_state.last_submission_details = None

if "dismissed_nudge_ids" not in st.session_state:
    st.session_state.dismissed_nudge_ids = set()

if "order_version" not in st.session_state:
    st.session_state.order_version = 1

if "trigger_coin_animation" not in st.session_state:
    st.session_state.trigger_coin_animation = False


def recompute_current_order() -> None:
    """Recomputes deterministic conversion and calculations for all lines."""
    for line in st.session_state.order_lines:
        order_parser._finalize_line_calculations(line, st.session_state.selected_store_id)


def execute_order_parse(text: str, store_id: str, app_mode: str, api_key: Optional[str], model: str) -> None:
    """Triggers order parsing and initializes a clean, unconfirmed order draft."""
    if not text or not text.strip():
        st.session_state.order_lines = []
        st.session_state.order_meta = {}
        return

    res = order_parser.parse_order(
        raw_message=text,
        store_id=store_id,
        use_llm=(app_mode == "Configured LLM API"),
        api_key=api_key,
        model_name=model,
    )
    if res.get("error"):
        st.error(f"Input validation error: {res['error']}")
        return

    lines = []
    for itm in res["items"]:
        l = OrderLine(
            line_id=itm["line_id"],
            message_id=itm["message_id"],
            store_id=itm["store_id"],
            original_item_text=itm["original_item_text"],
            product_mention=itm["product_mention"],
            requested_quantity_value=Decimal(itm["requested_quantity_value"]) if itm.get("requested_quantity_value") else None,
            requested_quantity_text=itm.get("requested_quantity_text", ""),
            requested_unit_text=itm.get("requested_unit_text"),
            canonical_product_name=itm.get("canonical_product_name"),
            sku=itm.get("sku"),
            brand=itm.get("brand"),
            pack_size=itm.get("pack_size"),
            match_status=itm.get("match_status", "NEEDS_CLARIFICATION"),
            match_evidence=itm.get("match_evidence"),
            confidence=float(itm.get("confidence", 0.0)),
            candidate_skus=itm.get("candidate_skus", []),
            candidate_names=itm.get("candidate_names", []),
            conversion_rule_id=itm.get("conversion_rule_id"),
            converted_quantity=Decimal(itm["converted_quantity"]) if itm.get("converted_quantity") else None,
            ordering_unit=itm.get("ordering_unit"),
            unit_price=Decimal(itm["unit_price"]) if itm.get("unit_price") else None,
            price_basis=itm.get("price_basis"),
            line_total=Decimal(itm["line_total"]) if itm.get("line_total") else None,
            clarification_required=bool(itm.get("clarification_required", False)),
            clarification_question=itm.get("clarification_question"),
            is_duplicate=bool(itm.get("is_duplicate", False)),
            line_status=itm.get("line_status", "Ready for review"),
            validation_messages=itm.get("validation_messages", []),
            spelling_suggestion=itm.get("spelling_suggestion"),
        )
        lines.append(l)

    st.session_state.order_lines = lines
    st.session_state.order_meta = {
        "message_id": res["message_id"],
        "store_id": res["store_id"],
        "language_style": res["language_style"],
        "has_injection_risk": res["has_injection_risk"],
        "injection_details": res["injection_details"],
        "raw_text": res["raw_text"],
    }
    st.session_state.submission_completed = False
    st.session_state.last_submission_details = None
    # Reset confirmation checkbox for new orders by bumping version
    st.session_state.order_version += 1

    st.session_state.audit_logger.log(
        event_type="ORDER_PARSED",
        status="Success" if not res["has_injection_risk"] else "Flagged - Injection Risk",
        details=f"Auto-parsed {len(lines)} line items. Language: {res['language_style']}.",
        message_id=res["message_id"],
        store_id=res["store_id"],
    )


# Initial parse only once on first boot
if "first_boot_done" not in st.session_state:
    st.session_state.first_boot_done = True
    if st.session_state.raw_message_input:
        execute_order_parse(st.session_state.raw_message_input, st.session_state.selected_store_id, "Demo / Rule-Grounded Mode", None, "default")


# -------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------
with st.sidebar:
    st.title("🛒 Kirana Ordering Assistant")
    st.caption("FAIB D3 Midterm Project - 300 Stores Credit Guardrails")
    st.divider()

    # Store Selector (supports all 300 stores)
    store_options = [str(s["Store_ID"]) for s in stores_data] if stores_data else ["STORE001"]
    current_store_idx = store_options.index(st.session_state.selected_store_id) if st.session_state.selected_store_id in store_options else 0

    def format_store_label(sid: str) -> str:
        s = stores_dict.get(sid, {})
        name = s.get("Store_Name", sid)
        city = s.get("City") or s.get("Location_City", "")
        overdue = float(s.get("Overdue_Balance_INR", 0.0) or 0.0)
        on_time = float(s.get("On_Time_Payment_Rate", 0.0) or 0.0) * 100
        if sid == "STORE002":
            status_tag = "🔴 Poor Credit History"
        elif overdue > 10000:
            status_tag = f"⚠️ Overdue > ₹10k (₹{overdue:,.0f})"
        else:
            status_tag = f"On-Time {on_time:.0f}%"
        return f"{sid} - {name} ({city}) [{status_tag}]"

    selected_store = st.selectbox(
        "🏪 Select Kirana Outlet (300 Stores)",
        options=store_options,
        index=current_store_idx,
        format_func=format_store_label,
        help="Select retail store profile for historical reordering context & custom packing/credit rules",
    )
    if selected_store != st.session_state.selected_store_id:
        st.session_state.selected_store_id = selected_store
        st.session_state.dismissed_nudge_ids = set()
        st.session_state.order_version += 1
        recompute_current_order()
        st.rerun()

    store_info = stores_dict.get(selected_store, {})
    if store_info:
        overdue_val = float(store_info.get("Overdue_Balance_INR", 0.0) or 0.0)
        limit_val = float(store_info.get("Credit_Limit_INR", 0.0) or 0.0)
        on_time_pct = float(store_info.get("On_Time_Payment_Rate", 0.0) or 0.0) * 100
        behavior = store_info.get("Payment_Behaviour_History", "Standard")

        if selected_store == "STORE002":
            credit_health_badge = "🔴 High Risk / Poor Payment History"
        elif overdue_val > 10000:
            credit_health_badge = "🔴 Blocked (> ₹10k Overdue Balance)"
        elif on_time_pct >= 90:
            credit_health_badge = "🟢 Excellent (Low Risk)"
        else:
            credit_health_badge = "🟡 Moderate / Under Review"

        st.info(
            f"**Store:** {store_info.get('Store_Name', selected_store)}\n\n"
            f"**City / Type:** {store_info.get('City', '')} | {store_info.get('Store_Type', 'Kirana')}\n\n"
            f"**Credit Health:** {credit_health_badge}\n\n"
            f"**Past Payment Behaviour:** *{behavior}*\n\n"
            f"**On-Time Rate:** {on_time_pct:.0f}% | **Avg Delay:** {float(store_info.get('Average_Days_Late', 0.0) or 0.0):.1f} days\n\n"
            f"**Credit Limit:** ₹ {limit_val:,.2f} | **Overdue:** ₹ {overdue_val:,.2f}"
        )

    st.divider()

    # Operating Mode
    app_mode = st.radio(
        "⚙️ Assistant Mode",
        options=["Demo / Rule-Grounded Mode", "Configured LLM API"],
        index=0,
        help="Demo mode runs 100% locally with verified deterministic rules. LLM mode requires an API key.",
    )

    api_key = None
    model_name = "gemini-1.5-flash"
    if app_mode == "Configured LLM API":
        api_key = st.text_input("API Key", type="password", help="Enter Gemini / OpenAI API key. Stored only in session.")
        model_name = st.selectbox("Model", ["gemini-1.5-flash", "gemini-1.5-pro", "gpt-4o-mini", "gpt-4o"])

    st.divider()

    # Quick Navigation
    view_selection = st.radio(
        "📑 View / Tab",
        ["📦 Order Processing", "💳 Credit ML Model & Guardrails", "📊 Reorder Model Benchmark", "📜 Session Audit Log", "📚 Product Catalogue & Rules"],
        index=0,
    )


# -------------------------------------------------------------
# TAB 1: ORDER PROCESSING & WORKFLOW (STEPS 1 TO 4)
# -------------------------------------------------------------
if view_selection == "📦 Order Processing":
    # 1-Second Money / Coin Celebration Animation when Order Placed
    if st.session_state.trigger_coin_animation:
        st.session_state.trigger_coin_animation = False
        st.balloons()
        coin_animation_html = """
        <div id="coin-celebration-overlay" style="
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            pointer-events: none;
            z-index: 9999999;
            overflow: hidden;
            background: rgba(0, 0, 0, 0.10);
            animation: fadeOverlay 1.3s ease-out forwards;
        ">
            <style>
                @keyframes coinRain {
                    0% { transform: translateY(-120px) rotate(0deg) scale(0.6); opacity: 1; }
                    80% { opacity: 1; }
                    100% { transform: translateY(115vh) rotate(1080deg) scale(1.3); opacity: 0; }
                }
                @keyframes fadeOverlay {
                    0% { opacity: 1; }
                    75% { opacity: 1; }
                    100% { opacity: 0; display: none; }
                }
                .falling-item {
                    position: absolute;
                    animation: coinRain 1.1s cubic-bezier(0.25, 0.46, 0.45, 0.94) forwards;
                    font-size: 46px;
                    filter: drop-shadow(0 6px 12px rgba(255, 215, 0, 0.9));
                    user-select: none;
                }
            </style>
            <div class="falling-item" style="left: 4%; animation-delay: 0.0s;">🪙</div>
            <div class="falling-item" style="left: 12%; animation-delay: 0.12s;">💰</div>
            <div class="falling-item" style="left: 19%; animation-delay: 0.04s;">₹</div>
            <div class="falling-item" style="left: 27%; animation-delay: 0.18s;">🪙</div>
            <div class="falling-item" style="left: 35%; animation-delay: 0.08s;">💵</div>
            <div class="falling-item" style="left: 43%; animation-delay: 0.22s;">🪙</div>
            <div class="falling-item" style="left: 50%; animation-delay: 0.02s;">₹</div>
            <div class="falling-item" style="left: 58%; animation-delay: 0.16s;">💰</div>
            <div class="falling-item" style="left: 66%; animation-delay: 0.06s;">🪙</div>
            <div class="falling-item" style="left: 74%; animation-delay: 0.20s;">💵</div>
            <div class="falling-item" style="left: 82%; animation-delay: 0.10s;">🪙</div>
            <div class="falling-item" style="left: 90%; animation-delay: 0.03s;">₹</div>
            <div class="falling-item" style="left: 96%; animation-delay: 0.14s;">💰</div>
        </div>
        """
        st.markdown(coin_animation_html, unsafe_allow_html=True)

    st.header("Kirana Replenishment Order Assistant")
    st.markdown(
        "> **Educational Prototype Notice:** Converts informal WhatsApp replenishment messages into auditable, "
        "verified orders with ML-predicted credit terms and deterministic policy guardrails."
    )

    # ---------------------------------------------------------
    # STEP 1: INBOUND WHATSAPP ORDER MESSAGE ENTRY
    # ---------------------------------------------------------
    with st.expander("📥 Step 1: Inbound WhatsApp Order Message", expanded=True):
        col_m1, col_m2 = st.columns([3, 1])

        with col_m2:
            st.markdown("##### 💬 Sample Messages")
            sample_labels = [f"#{idx+1}: {m.get('Category_Note', 'Order')} ({m.get('Store_ID', 'Any')})" for idx, m in enumerate(backend_messages)]
            selected_sample_idx = st.selectbox(
                "Choose authentic sample:",
                options=range(len(sample_labels)),
                format_func=lambda i: sample_labels[i],
                index=0,
            )
            if st.button("📋 Load Sample Message", use_container_width=True):
                chosen = backend_messages[selected_sample_idx]
                st.session_state.raw_message_input = chosen["WhatsApp_Message"]
                if chosen.get("Store_ID") in store_options:
                    st.session_state.selected_store_id = chosen["Store_ID"]
                execute_order_parse(chosen["WhatsApp_Message"], st.session_state.selected_store_id, app_mode, api_key, model_name)
                st.rerun()

        with col_m1:
            raw_text = st.text_area(
                "Inbound Replenishment Message (English, Hindi, Hinglish, vernacular units like 'peti', 'dhaba/dhabba'):",
                value=st.session_state.raw_message_input,
                height=110,
                key="msg_text_input_area",
            )
            btn_c1, btn_c2, btn_c3 = st.columns([1, 1, 2])
            with btn_c1:
                if st.button("⚡ Parse Order", type="primary", use_container_width=True):
                    st.session_state.raw_message_input = raw_text
                    execute_order_parse(raw_text, st.session_state.selected_store_id, app_mode, api_key, model_name)
                    st.rerun()
            with btn_c2:
                # Reset button comes as completely blank for a fresh order
                if st.button("🔄 Reset Order (Blank)", use_container_width=True):
                    st.session_state.raw_message_input = ""
                    st.session_state.order_lines = []
                    st.session_state.order_meta = {}
                    st.session_state.submission_completed = False
                    st.session_state.last_submission_details = None
                    st.session_state.order_version += 1
                    st.rerun()
            with btn_c3:
                if st.session_state.order_meta:
                    lang = st.session_state.order_meta.get("language_style", "English")
                    st.caption(f"Detected Dialect: **{lang}** | Message ID: **{st.session_state.order_meta.get('message_id')}**")
                    if st.session_state.order_meta.get("has_injection_risk"):
                        st.error(f"🛡️ Guardrail alert: {st.session_state.order_meta.get('injection_details')}")

    st.divider()

    # ---------------------------------------------------------
    # STEP 2: AUTO-SELECTED GROUNDED ORDER REVIEW
    # ---------------------------------------------------------
    st.subheader("📋 Step 2: Auto-Selected Grounded Order Review")

    if not st.session_state.order_lines:
        st.info("No order items in draft. Enter or select a message above and click **Parse Order**.")
    else:
        st.caption("Review extracted products, adjust quantities (auto-recalculates price), clarify spelling mistakes or ambiguous candidates, or drop unresolved items.")

        for idx, line in enumerate(st.session_state.order_lines):
            status_color = "🟢" if line.line_status == "Ready for review" else ("🟡" if line.line_status == "Needs clarification" else "🔴")

            with st.container(border=True):
                top_c1, top_c2, top_c3 = st.columns([3, 2, 1])
                with top_c1:
                    st.markdown(f"**Item {idx+1}: {line.product_mention}** &nbsp; `{line.line_id}`")
                    st.caption(f"Original text: *\"{line.original_item_text}\"*")
                with top_c2:
                    st.markdown(f"Status: **{status_color} {line.line_status}**")
                    if line.canonical_product_name:
                        st.caption(f"Matched: **{line.canonical_product_name}** (`{line.sku}`)")
                    elif line.match_status == "UNKNOWN_PRODUCT":
                        st.caption("⚠️ Product not found in distributor catalogue")
                with top_c3:
                    # Drop item button for every item (especially unknown / ambiguous)
                    if st.button("🗑️ Drop Item", key=f"btn_drop_{line.line_id}", help="Drop this item from order draft", type="secondary", use_container_width=True):
                        st.session_state.order_lines = [l for l in st.session_state.order_lines if l.line_id != line.line_id]
                        st.session_state.audit_logger.log(
                            event_type="ITEM_DROPPED",
                            status="Removed by shopkeeper",
                            details=f"Dropped line item '{line.product_mention}' ({line.line_id}) from order draft.",
                            message_id=st.session_state.order_meta.get("message_id"),
                            store_id=st.session_state.selected_store_id,
                        )
                        st.toast(f"Dropped '{line.product_mention}' from order.")
                        recompute_current_order()
                        st.rerun()

                # Middle row: Quantity editing & price breakdown
                q_col1, q_col2, q_col3, q_col4 = st.columns([2, 2, 2, 2])
                with q_col1:
                    # Editable quantity input - recalculates line total and grand total immediately
                    current_qty_val = float(line.requested_quantity_value) if line.requested_quantity_value is not None else (float(line.converted_quantity) if line.converted_quantity is not None else 1.0)
                    unit_label = line.requested_unit_text or line.ordering_unit or "units"
                    new_qty = st.number_input(
                        f"Quantity ({unit_label})",
                        min_value=1.0,
                        step=1.0,
                        value=max(1.0, current_qty_val),
                        key=f"input_qty_{line.line_id}_{st.session_state.order_version}",
                    )
                    if Decimal(str(new_qty)) != line.requested_quantity_value:
                        line.requested_quantity_value = Decimal(str(new_qty))
                        line.user_edited = True
                        order_parser._finalize_line_calculations(line, st.session_state.selected_store_id)
                        st.session_state.audit_logger.log(
                            event_type="QUANTITY_EDITED",
                            status="Success",
                            details=f"Line {line.line_id}: Quantity updated to {new_qty} {unit_label}.",
                            message_id=st.session_state.order_meta.get("message_id"),
                            store_id=st.session_state.selected_store_id,
                        )
                        st.rerun()

                with q_col2:
                    if line.converted_quantity is not None and line.ordering_unit:
                        conv_text = f"**{line.converted_quantity} {line.ordering_unit}**"
                        if line.conversion_rule_id:
                            conv_text += f"\n\n*(Rule: {line.conversion_rule_id})*"
                        st.markdown(f"**Ordering Units:**\n{conv_text}")
                    else:
                        st.markdown("**Ordering Units:**\n*Pending resolution*")

                with q_col3:
                    if line.unit_price is not None:
                        st.markdown(f"**Unit Price:**\n₹ {line.unit_price:,.2f} / {line.ordering_unit or 'unit'}")
                    else:
                        st.markdown("**Unit Price:**\n*Unavailable*")

                with q_col4:
                    if line.line_total is not None:
                        st.markdown(f"**Line Total:**\n### ₹ {line.line_total:,.2f}")
                    else:
                        st.markdown("**Line Total:**\n### —")

                # Validation messages and clarification questions
                if line.validation_messages and line.clarification_required:
                    for msg in line.validation_messages:
                        st.warning(f"ℹ️ {msg}")

                # Closest Spelling Suggestion: "Do you mean this which is most close to that word"
                closest_suggs = catalogue.get_spelling_suggestions(line.product_mention, top_n=2)
                if closest_suggs and (line.match_status != "MATCHED" or line.spelling_suggestion):
                    closest_name = closest_suggs[0]
                    closest_prod = None
                    for p in catalogue.products.values():
                        if p.product_name == closest_name:
                            closest_prod = p
                            break

                    st.info(f"💡 **Spelling Assistant:** Did you mean this: **'{closest_name}'** (closest match to *'{line.product_mention}'*)?")
                    if closest_prod:
                        if st.button(f"👉 Yes, change to '{closest_name}'", key=f"btn_spell_{line.line_id}_{closest_prod.sku}"):
                            line.sku = closest_prod.sku
                            line.canonical_product_name = closest_prod.product_name
                            line.brand = closest_prod.brand
                            line.pack_size = closest_prod.display_pack
                            line.match_status = "MATCHED"
                            line.clarification_required = False
                            line.line_status = "Ready for review"
                            line.validation_messages = []
                            line.clarification_question = None
                            line.spelling_suggestion = None
                            order_parser._finalize_line_calculations(line, st.session_state.selected_store_id)
                            st.session_state.audit_logger.log(
                                event_type="SPELLING_CORRECTION_APPLIED",
                                status="Success",
                                details=f"Line {line.line_id}: Shopkeeper accepted spelling correction '{closest_name}' for '{line.product_mention}'.",
                                message_id=st.session_state.order_meta.get("message_id"),
                                store_id=st.session_state.selected_store_id,
                            )
                            st.toast(f"Updated to {closest_name}!")
                            st.rerun()

                # Candidate Clarification Selection (Ambiguous Items)
                if line.clarification_required and line.candidate_skus:
                    st.markdown(f"❓ **Clarification needed:** {line.clarification_question or 'Please select the intended product:'}")
                    cand_cols = st.columns(len(line.candidate_skus) + 1)
                    for c_idx, cand_sku in enumerate(line.candidate_skus):
                        cand_prod = catalogue.get_by_sku(cand_sku)
                        cand_name = cand_prod.product_name if cand_prod else cand_sku
                        with cand_cols[c_idx]:
                            if st.button(f"✅ Select: {cand_name}", key=f"cand_{line.line_id}_{cand_sku}", use_container_width=True):
                                line.sku = cand_sku
                                line.canonical_product_name = cand_prod.product_name if cand_prod else cand_sku
                                line.brand = cand_prod.brand if cand_prod else None
                                line.pack_size = cand_prod.display_pack if cand_prod else None
                                line.match_status = "MATCHED"
                                line.clarification_required = False
                                line.line_status = "Ready for review"
                                line.validation_messages = []
                                line.clarification_question = None
                                order_parser._finalize_line_calculations(line, st.session_state.selected_store_id)
                                st.session_state.audit_logger.log(
                                    event_type="CLARIFICATION_RESOLVED",
                                    status="Success",
                                    details=f"Line {line.line_id}: Shopkeeper confirmed SKU {cand_sku} ({cand_name}).",
                                    message_id=st.session_state.order_meta.get("message_id"),
                                    store_id=st.session_state.selected_store_id,
                                )
                                st.toast(f"Resolved to {cand_name}")
                                st.rerun()

                elif line.match_status == "UNKNOWN_PRODUCT":
                    st.error("❓ Unknown product: This item does not match any SKU in the catalogue.")
                    st.markdown("Use the **🗑️ Drop Item** button above to remove this item so the order can proceed.")

        st.divider()

        # ---------------------------------------------------------
        # STEP 3: STORE-HISTORY REORDER RECOMMENDATIONS
        # ---------------------------------------------------------
        st.subheader("💡 Step 3: Store-History Reorder Recommendations")
        current_skus = {l.sku for l in st.session_state.order_lines if l.sku}
        nudges = predictor.get_reorder_nudges(st.session_state.selected_store_id, current_basket_skus=current_skus)
        active_nudges = [n for n in nudges if n.nudge_id not in st.session_state.dismissed_nudge_ids]

        if not active_nudges:
            st.caption("No additional reorder reminders for this store's ordering cycle.")
        else:
            st.caption("Suggested replenishment items based on verified historical purchase cycles (added directly to final order):")
            nudge_cols = st.columns(min(3, len(active_nudges)))
            for n_idx, nudge in enumerate(active_nudges[:3]):
                with nudge_cols[n_idx % 3]:
                    with st.container(border=True):
                        st.markdown(f"**{nudge.product_name}**")
                        st.caption(f"Category: {nudge.category} | Confidence: **{nudge.confidence_score * 100:.0f}%**")
                        st.markdown(f"Suggested: **{nudge.suggested_quantity} {nudge.ordering_unit}** @ ₹ {nudge.unit_price_inr}")
                        st.caption(f"*{nudge.reason}*")
                        btn_a, btn_d = st.columns(2)
                        with btn_a:
                            if st.button("➕ Add to Order", key=f"btn_add_nudge_{nudge.nudge_id}", use_container_width=True):
                                cat_p = catalogue.get_by_sku(nudge.sku)
                                new_line = OrderLine(
                                    line_id=f"LINE-REC-{len(st.session_state.order_lines)+1:03d}",
                                    message_id=st.session_state.order_meta.get("message_id", "MSG-NUDGE"),
                                    store_id=st.session_state.selected_store_id,
                                    original_item_text=f"[Reorder Suggestion] {nudge.suggested_quantity} {nudge.ordering_unit} {nudge.product_name}",
                                    product_mention=nudge.product_name,
                                    requested_quantity_value=nudge.suggested_quantity,
                                    requested_quantity_text=str(nudge.suggested_quantity),
                                    requested_unit_text=nudge.ordering_unit,
                                    canonical_product_name=nudge.product_name,
                                    sku=nudge.sku,
                                    brand=cat_p.brand if cat_p else "",
                                    pack_size=cat_p.display_pack if cat_p else "",
                                    match_status="MATCHED",
                                    clarification_required=False,
                                    line_status="Ready for review",
                                )
                                order_parser._finalize_line_calculations(new_line, st.session_state.selected_store_id)
                                st.session_state.order_lines.append(new_line)
                                st.session_state.dismissed_nudge_ids.add(nudge.nudge_id)
                                st.session_state.audit_logger.log(
                                    event_type="REORDER_NUDGE_ADDED",
                                    status="Success",
                                    details=f"Added suggested product {nudge.product_name} ({nudge.sku}) qty {nudge.suggested_quantity}.",
                                    message_id=st.session_state.order_meta.get("message_id"),
                                    store_id=st.session_state.selected_store_id,
                                )
                                st.toast(f"Added {nudge.product_name} to final order!")
                                st.rerun()
                        with btn_d:
                            if st.button("✕ Dismiss", key=f"btn_dis_nudge_{nudge.nudge_id}", use_container_width=True):
                                st.session_state.dismissed_nudge_ids.add(nudge.nudge_id)
                                st.session_state.audit_logger.log(
                                    event_type="REORDER_NUDGE_DISMISSED",
                                    status="Dismissed",
                                    details=f"Dismissed nudge for {nudge.product_name} ({nudge.sku}).",
                                    message_id=st.session_state.order_meta.get("message_id"),
                                    store_id=st.session_state.selected_store_id,
                                )
                                st.rerun()

        st.divider()

        # ---------------------------------------------------------
        # STEP 4: PAYMENT METHOD SELECTION & CONFIRMATION
        # ---------------------------------------------------------
        st.subheader("💳 Step 4: Payment Method & Final Confirmation")

        # Convert lines to LineCalculation for exact deterministic grand total
        line_calcs = []
        for l in st.session_state.order_lines:
            is_valid = (
                l.sku is not None
                and l.match_status == "MATCHED"
                and l.converted_quantity is not None
                and l.converted_quantity > 0
                and l.unit_price is not None
                and not l.clarification_required
            )
            lc = calculate_line_item(
                sku=l.sku,
                ordering_quantity=l.converted_quantity,
                ordering_unit=l.ordering_unit,
                unit_price_inr=l.unit_price,
                match_status=l.match_status,
                conversion_status="VALID" if is_valid else "INVALID",
            )
            line_calcs.append(lc)

        summary = calculate_grand_total(line_calcs)
        order_total_val = summary.grand_total_inr or Decimal("0")

        # Payment Option: Pay Now vs Pay Later
        payment_choice = st.radio(
            "Select Payment Option:",
            options=["💵 Pay Now (Immediate Settlement)", "💳 Pay Later (Credit Period Request)"],
            index=0,
            horizontal=True,
            help="Choose 'Pay Now' for immediate cash/UPI/NetBanking without credit check, or 'Pay Later' to request distributor credit.",
            key=f"payment_choice_{st.session_state.order_version}",
        )

        assigned_payment_terms = "Pay Now (Immediate Settlement)"
        credit_eval: Optional[CreditAssessment] = None
        can_proceed_payment = True

        # IF PAY NOW:
        if payment_choice == "💵 Pay Now (Immediate Settlement)":
            st.info(
                "✅ **Payment Option Selected: Pay Now (Immediate Settlement)**\n\n"
                "- **No credit check required.** Payment will be settled immediately via UPI, NetBanking, Card, or Cash on Delivery.\n"
                f"- **Total Amount to Pay Now:** **₹ {order_total_val:,.2f}**\n"
                "- **Approved Terms:** `Immediate Settlement (0-Day Credit)`"
            )
            assigned_payment_terms = "Pay Now (Immediate Settlement / COD)"

        # IF PAY LATER:
        else:
            credit_eval = credit_predictor.assess_store_credit(st.session_state.selected_store_id, order_total_val)

            with st.container(border=True):
                st.markdown(f"#### 🔍 Credit Assessment for {st.session_state.selected_store_id}")

                # 1. Check Guardrail 1: Transaction Ceiling Guardrail (> ₹30,000) - RED
                if credit_eval.exceeds_30k_order_threshold:
                    st.error(
                        f"### 🛑 Policy Guardrail 1 Failed: High-Value Transaction Ceiling (₹30,000 Limit)\n\n"
                        f"This order total is **₹ {order_total_val:,.2f}**, which exceeds the maximum allowable credit ceiling of **₹ 30,000.00**.\n\n"
                        f"👉 **Mandatory Rule:** **Transactions above ₹30,000 are not allowed for credit and must be done as Cash on Delivery (COD) / Full Upfront Payment.**\n\n"
                        f"**Approved Payment Terms:** `Cash on Delivery (COD) / Upfront Payment Required`"
                    )
                    assigned_payment_terms = "COD / Upfront Payment Required (> ₹30k Order Ceiling)"
                    can_proceed_payment = True

                # 2. Check Guardrail 2: Overdue Balance Guardrail (> ₹10,000) - RED
                elif credit_eval.exceeds_10k_threshold:
                    st.error(
                        f"### 🛑 Policy Guardrail 2 Failed: Overdue Balance Ceiling (₹10,000 Limit)\n\n"
                        f"This Kirana store currently has an **unpaid overdue balance of ₹ {credit_eval.overdue_balance_inr:,.2f}**, "
                        f"which exceeds the maximum policy threshold of **₹ 10,000.00**.\n\n"
                        f"👉 **Mandatory Rule:** **The store must finish / clear the last credit that they are left with before another credit period can be granted.**\n\n"
                        f"**Approved Payment Terms:** `Cash on Delivery (COD) / Full Upfront Payment Required`"
                    )
                    assigned_payment_terms = "COD / Full Upfront Payment Required (Overdue > ₹10k)"
                    can_proceed_payment = True

                # 3. Check Guardrail 3: ML Predictive Credit Risk Guardrail (COD Recommended) - YELLOW
                elif not credit_eval.model_credit_granted:
                    st.warning(
                        f"### ⚠️ Policy Guardrail 3 Failed: Predictive Credit Risk (COD Mandated)\n\n"
                        f"Static limits passed (Order: **₹ {order_total_val:,.2f}** $\le$ ₹30,000 | Overdue: **₹ {credit_eval.overdue_balance_inr:,.2f}** $\le$ ₹10,000), "
                        f"**BUT Guardrail 3 (ML Creditworthiness) FAILED.**\n\n"
                        f"The **{credit_eval.best_model_name} model** predicts elevated risk of default based on the store's track record.\n\n"
                        f"👉 **Mandatory Guardrail Action:** **Pay Later / Credit period is denied. The store must pay via Cash on Delivery (COD) / Prepaid.**\n\n"
                        f"**Mandated Payment Terms:** `Cash on Delivery (COD) / Prepaid Required`"
                    )

                    pred_c1, pred_c2 = st.columns(2)
                    with pred_c1:
                        status_badge = "🟡 GUARDRAIL 3 FAILED: COD REQUIRED"
                        st.markdown(f"**Guardrail Status:** ### {status_badge}")
                        st.markdown(f"**Assigned Terms:** `{credit_eval.recommended_payment_terms}`")
                        st.caption(f"Evaluated by: **{credit_eval.best_model_name}** | Risk Confidence: **{(1 - credit_eval.model_probability)*100:.1f}%**")

                    with pred_c2:
                        st.markdown("**Historical Payment Attributes:**")
                        st.write(f"- On-Time Payment Rate: **{credit_eval.on_time_payment_rate*100:.1f}%**")
                        st.write(f"- Average Days Late: **{credit_eval.average_days_late:.1f} days**")
                        st.write(f"- Current Overdue Balance: **₹ {credit_eval.overdue_balance_inr:,.2f}**")
                        st.write(f"- Approved Credit Limit: **₹ {credit_eval.credit_limit_inr:,.2f}**")

                    st.warning(f"⚠️ **Guardrail Recommendation:** {credit_eval.explanation}")
                    assigned_payment_terms = "Cash on Delivery (COD) / Prepaid Required (Guardrail 3 Failed)"
                    can_proceed_payment = True

                # 4. All 3 Guardrails Passed - GREEN
                else:
                    st.success(
                        f"### 🟢 All 3 Credit Policy Guardrails Passed\n\n"
                        f"- ✅ Guardrail 1 (Order Ceiling): **₹ {order_total_val:,.2f}** $\le$ ₹30,000\n"
                        f"- ✅ Guardrail 2 (Overdue Balance): **₹ {credit_eval.overdue_balance_inr:,.2f}** $\le$ ₹10,000\n"
                        f"- ✅ Guardrail 3 (ML Creditworthiness): **Passed** (Low risk, Approved for {credit_eval.recommended_credit_days}-Day Credit)"
                    )

                    pred_c1, pred_c2 = st.columns(2)
                    with pred_c1:
                        status_badge = "🟢 ALL GUARDRAILS PASSED (CREDIT GRANTED)"
                        st.markdown(f"**Guardrail Status:** ### {status_badge}")
                        st.markdown(f"**Approved Terms:** `{credit_eval.recommended_payment_terms}`")
                        st.caption(f"Evaluated by: **{credit_eval.best_model_name}** | Confidence: **{credit_eval.model_probability*100:.1f}%**")

                    with pred_c2:
                        st.markdown("**Historical Payment Attributes:**")
                        st.write(f"- On-Time Payment Rate: **{credit_eval.on_time_payment_rate*100:.1f}%**")
                        st.write(f"- Average Days Late: **{credit_eval.average_days_late:.1f} days**")
                        st.write(f"- Current Overdue Balance: **₹ {credit_eval.overdue_balance_inr:,.2f}**")
                        st.write(f"- Approved Credit Limit: **₹ {credit_eval.credit_limit_inr:,.2f}**")

                    st.info(f"💡 **Decision Explanation:** {credit_eval.explanation}")
                    assigned_payment_terms = credit_eval.recommended_payment_terms

                # ML Model Comparison Breakdown
                with st.expander("📊 View ML Model Benchmark Comparison (Logistic Regression vs Decision Tree)"):
                    lr_m = credit_predictor.lr_metrics
                    dt_m = credit_predictor.dt_metrics
                    if lr_m and dt_m:
                        df_metrics = pd.DataFrame([
                            {
                                "Model": lr_m.model_name,
                                "Accuracy": f"{lr_m.accuracy*100:.1f}%",
                                "Precision": f"{lr_m.precision*100:.1f}%",
                                "Recall": f"{lr_m.recall*100:.1f}%",
                                "F1-Score": f"{lr_m.f1_score*100:.1f}%",
                                "Status": "⭐ Selected Best Model" if credit_predictor.best_model_name == lr_m.model_name else "Evaluated Baseline",
                            },
                            {
                                "Model": dt_m.model_name,
                                "Accuracy": f"{dt_m.accuracy*100:.1f}%",
                                "Precision": f"{dt_m.precision*100:.1f}%",
                                "Recall": f"{dt_m.recall*100:.1f}%",
                                "F1-Score": f"{dt_m.f1_score*100:.1f}%",
                                "Status": "⭐ Selected Best Model" if credit_predictor.best_model_name == dt_m.model_name else "Evaluated Baseline",
                            },
                        ])
                        st.dataframe(df_metrics, use_container_width=True, hide_index=True)

        # Summary KPIs
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        m_c1.metric("Total Line Items", summary.line_count)
        m_c2.metric("Verified Valid Lines", summary.valid_lines_count)
        m_c3.metric(
            "Unresolved Lines",
            summary.unresolved_lines_count,
            delta=None if summary.unresolved_lines_count == 0 else f"{summary.unresolved_lines_count} require resolution",
            delta_color="inverse",
        )
        m_c4.metric(
            "Exact Grand Total",
            f"₹ {order_total_val:,.2f}" if summary.grand_total_inr is not None else "Blocked",
        )

        if not summary.can_confirm:
            st.error("⚠️ **Order confirmation blocked:**")
            for msg in summary.summary_messages:
                st.write(f"- {msg}")
            st.info("💡 To unblock confirmation, resolve spelling/candidate clarifications above or click **🗑️ Drop Item** on any unresolved items.")
        else:
            st.success("✅ **All line items verified against catalogue.** Order draft is ready for affirmative human confirmation.")

            # Explicit Confirmation Checkbox (ALWAYS starts UNTICKED for every new order)
            confirm_key = f"explicit_confirm_checkbox_{st.session_state.order_version}"
            checkbox_label = (
                f"☑️ **I have reviewed all {summary.valid_lines_count} line items, verified quantities and prices, "
                f"and explicitly authorized payment mode: `{assigned_payment_terms}` (Total: ₹ {order_total_val:,.2f}).**"
            )
            affirmation_checked = st.checkbox(
                checkbox_label,
                value=False,
                key=confirm_key,
                help="Affirmative confirmation is strictly required before simulated dispatch.",
            )

            dispatch_c1, dispatch_c2 = st.columns([2, 3])
            with dispatch_c1:
                submit_button = st.button(
                    "🚀 Confirm & Simulated Dispatch",
                    type="primary",
                    disabled=not (summary.can_confirm and affirmation_checked and can_proceed_payment),
                    use_container_width=True,
                    help="Enabled only after affirmative confirmation checkbox is checked.",
                )

                if submit_button:
                    dispatch_id = f"DISPATCH-{uuid.uuid4().hex[:8].upper()}"

                    # Snapshot all lines (including added recommendations) so they show in final order!
                    final_items_snapshot = [
                        {
                            "Line": l.line_id,
                            "Product": l.canonical_product_name or l.product_mention,
                            "SKU": l.sku or "N/A",
                            "Quantity": f"{l.converted_quantity} {l.ordering_unit}",
                            "Unit Price": f"₹ {l.unit_price:,.2f}" if l.unit_price else "—",
                            "Line Total": f"₹ {l.line_total:,.2f}" if l.line_total else "—",
                        }
                        for l in st.session_state.order_lines
                    ]

                    submission_event = st.session_state.audit_logger.log(
                        event_type="SIMULATED_ORDER_SUBMISSION",
                        status="Simulated submission complete - no real order sent",
                        details=(
                            f"Order confirmed for {st.session_state.selected_store_id}. "
                            f"Dispatch ID: {dispatch_id}. Total: ₹ {order_total_val:,.2f}. "
                            f"Items: {len(final_items_snapshot)}. Payment Terms: {assigned_payment_terms}."
                        ),
                        message_id=st.session_state.order_meta.get("message_id"),
                        store_id=st.session_state.selected_store_id,
                    )

                    st.session_state.submission_completed = True
                    st.session_state.trigger_coin_animation = True
                    st.session_state.last_submission_details = {
                        "dispatch_id": dispatch_id,
                        "timestamp": submission_event.timestamp,
                        "store_id": st.session_state.selected_store_id,
                        "store_name": stores_dict.get(st.session_state.selected_store_id, {}).get("Store_Name", ""),
                        "grand_total": order_total_val,
                        "items_count": len(final_items_snapshot),
                        "items": final_items_snapshot,
                        "payment_choice": payment_choice,
                        "payment_terms": assigned_payment_terms,
                    }
                    st.session_state.order_lines = []
                    # Bump order version so next order confirmation checkbox is unticked
                    st.session_state.order_version += 1
                    st.rerun()

        # Display submission completed receipt
        if st.session_state.submission_completed and st.session_state.last_submission_details:
            sub = st.session_state.last_submission_details
            with st.container(border=True):
                st.success(
                    f"### 🎉 Simulated Dispatch Complete!\n\n"
                    f"- **Dispatch Reference:** `{sub['dispatch_id']}`\n"
                    f"- **Kirana Outlet:** {sub['store_id']} - **{sub['store_name']}**\n"
                    f"- **Total Order Value:** **₹ {sub['grand_total']:,.2f}** ({sub['items_count']} line items)\n"
                    f"- **Payment Method Selected:** `{sub['payment_choice']}`\n"
                    f"- **Authorized Terms:** `{sub['payment_terms']}`\n"
                    f"- **Timestamp:** {sub['timestamp']}"
                )

                st.markdown("##### 📦 Itemized Final Order Receipt:")
                df_final_receipt = pd.DataFrame(sub["items"])
                st.dataframe(df_final_receipt, use_container_width=True, hide_index=True)
                st.caption("*(Audit event logged. Simulated submission only — no external ERP or physical inventory modified.)*")


# -------------------------------------------------------------
# TAB 2: CREDIT ML MODEL & GUARDRAILS
# -------------------------------------------------------------
elif view_selection == "💳 Credit ML Model & Guardrails":
    st.header("Payment Credit Predictive Modeling & Policy Guardrails")
    st.markdown(
        "Demonstrates predictive classification of Kirana payment credit eligibility on 300 store profiles "
        "using categorical dummy variables, comparing **Logistic Regression** vs **Decision Tree** models, "
        "and enforcing two credit guardrails:\n"
        "1. **₹10,000 Overdue Balance Guardrail**\n"
        "2. **₹30,000 Transaction Ceiling Guardrail**"
    )

    cp = credit_predictor
    lr_m = cp.lr_metrics
    dt_m = cp.dt_metrics

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.subheader("1. Logistic Regression Model")
        st.caption("Regularized L2 Logistic Regression with feature standardization.")
        if lr_m:
            st.metric("Accuracy", f"{lr_m.accuracy*100:.1f}%")
            st.metric("Precision", f"{lr_m.precision*100:.1f}%")
            st.metric("Recall", f"{lr_m.recall*100:.1f}%")
            st.metric("F1-Score", f"{lr_m.f1_score*100:.1f}%")

    with col_m2:
        st.subheader("2. Decision Tree Classifier")
        st.caption("Cart Decision Tree with max depth = 4.")
        if dt_m:
            st.metric("Accuracy", f"{dt_m.accuracy*100:.1f}%")
            st.metric("Precision", f"{dt_m.precision*100:.1f}%")
            st.metric("Recall", f"{dt_m.recall*100:.1f}%")
            st.metric("F1-Score", f"{dt_m.f1_score*100:.1f}%")

    st.success(f"⭐ **Best Model Selected by F1-Score:** **{cp.best_model_name}**")

    st.divider()

    st.subheader("🛡️ Policy Guardrails Summary")
    st.markdown("""
    - **Guardrail 1 (₹10,000 Overdue Balance Ceiling - Red Flag 🛑):** If any Kirana store has an unpaid overdue balance strictly exceeding **₹ 10,000** (e.g. `STORE012`), credit is blocked. The store must clear pending credit before new credit can be granted.
    - **Guardrail 2 (₹30,000 Transaction Ceiling - Red Flag 🛑):** Any order transaction total exceeding **₹ 30,000** is strictly prohibited from credit and must be paid via COD / Upfront Payment.
    - **ML Predictive Credit Risk (Yellow Flag ⚠️):** Stores with poor repayment behavior (e.g. `STORE008`, `STORE009`, `STORE010`) are evaluated by the **Logistic Regression** model. When the model predicts elevated default risk, credit is denied and **Cash on Delivery (COD)** is mandated with a yellow warning flag.
    """)

    # Interactive Store Credit Checker
    st.subheader("🔎 Store-by-Store Credit Check")
    test_store_id = st.selectbox(
        "Select Store to inspect credit decision:",
        options=store_options,
        index=current_store_idx,
        format_func=format_store_label,
        key="inspect_store_credit_box",
    )
    test_order_val = st.number_input("Hypothetical Order Value (₹)", value=15000.0, step=1000.0)
    assessment = cp.assess_store_credit(test_store_id, Decimal(str(test_order_val)))

    if assessment.exceeds_30k_order_threshold:
        st.error(f"🛑 **Guardrail 1 Failed (Order Ceiling > ₹30,000 - Red Flag):** {assessment.guardrail_message}")
    elif assessment.exceeds_10k_threshold:
        st.error(f"🛑 **Guardrail 2 Failed (Overdue Balance > ₹10,000 - Red Flag):** {assessment.guardrail_message}")
    elif not assessment.model_credit_granted:
        st.warning(f"⚠️ **Guardrail 3 Failed (ML Credit Risk $\\rightarrow$ COD Mandated - Yellow Flag):** `{assessment.recommended_payment_terms}`\n\n- {assessment.explanation}")
    else:
        st.success(f"🟢 **All 3 Guardrails Passed (Credit Approved):** `{assessment.recommended_payment_terms}`\n\n- {assessment.explanation}")


# -------------------------------------------------------------
# TAB 3: REORDER MODEL BENCHMARK
# -------------------------------------------------------------
elif view_selection == "📊 Reorder Model Benchmark":
    st.header("Predictive Reorder Recommender Evaluation")
    st.markdown("Evaluates transparent baseline heuristic vs statistical classifier on time-aware historical orders (`Reorder_Features.csv`).")

    with st.spinner("Calculating benchmark metrics on time-aware split..."):
        base_m, model_m, fin = predictor.evaluate_benchmark()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Simple Baseline Heuristic")
        st.caption("Rule: Ordered in >= 2 of prior 3 store replenishment cycles.")
        st.metric("Baseline Precision", f"{base_m.precision * 100:.1f}%")
        st.metric("Baseline Recall", f"{base_m.recall * 100:.1f}%")
        st.metric("Baseline F1 Score", f"{base_m.f1_score * 100:.1f}%")
        st.write(f"- True Positives: **{base_m.true_positives}**")
        st.write(f"- False Positives: **{base_m.false_positives}** (nuisance nudges)")
        st.write(f"- False Negatives: **{base_m.false_negatives}** (missed sales)")

    with col2:
        st.subheader("2. Calibrated Ranking Classifier")
        st.caption("Supervised logistic features with time-aware train/test evaluation.")
        st.metric("Model Precision", f"{model_m.precision * 100:.1f}%", delta=f"{(model_m.precision - base_m.precision)*100:+.1f}%")
        st.metric("Model Recall", f"{model_m.recall * 100:.1f}%", delta=f"{(model_m.recall - base_m.recall)*100:+.1f}%")
        st.metric("Model F1 Score", f"{model_m.f1_score * 100:.1f}%", delta=f"{(model_m.f1_score - base_m.f1_score)*100:+.1f}%")
        st.write(f"- True Positives: **{model_m.true_positives}**")
        st.write(f"- False Positives: **{model_m.false_positives}**")
        st.write(f"- False Negatives: **{model_m.false_negatives}**")


# -------------------------------------------------------------
# TAB 4: AUDIT LOG
# -------------------------------------------------------------
elif view_selection == "📜 Session Audit Log":
    st.header("Session Audit Log & Governance")
    st.caption("Tamper-evident audit trail of all order parses, clarifications, edits, nudge decisions, and simulated dispatches.")

    events = st.session_state.audit_logger.get_events()
    if not events:
        st.info("No audit events logged in current session.")
    else:
        df_audit = pd.DataFrame([e.to_dict() for e in events])
        st.dataframe(df_audit, use_container_width=True, hide_index=True)

        csv_data = df_audit.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export Audit Log (CSV)",
            data=csv_data,
            file_name="kirana_order_assistant_audit_log.csv",
            mime="text/csv",
        )


# -------------------------------------------------------------
# TAB 5: PRODUCT CATALOGUE & RULES
# -------------------------------------------------------------
elif view_selection == "📚 Product Catalogue & Rules":
    st.header("Authoritative Product Catalogue & Grounded Rules")
    st.caption("Distributor single source of truth (24 active SKUs). Catalogue facts are authoritative and immutable.")

    cat_rows = []
    for sku, p in catalogue.products.items():
        cat_rows.append({
            "SKU": p.sku,
            "Product Name": p.product_name,
            "Brand": p.brand,
            "Category": p.category,
            "Order Unit": p.order_unit,
            "Unit Price (INR)": f"₹ {p.unit_price_inr:,.2f}" if p.unit_price_inr else "N/A",
            "Units/Case (Peti/Dhaba)": p.units_per_case,
            "Aliases & Synonyms": ", ".join(p.aliases),
        })
    df_cat = pd.DataFrame(cat_rows)
    st.dataframe(df_cat, use_container_width=True, hide_index=True)

    st.subheader("Verified Unit Conversion Rules")
    st.markdown("""
    - **`carton` / `dhaba` / `dhabba` / `peti` / `case`:**
      Recognized as full carton packing in vocabulary and converted using explicit master case multipliers (e.g. 1 dhaba/peti Coke = 48 bottles; 1 dhaba/peti Fortune Oil 1L = 12 bottles).
    - **`STORE003` Special Agreement:** 1 dhaba/peti Coke = 48 bottles.
    - **Informal / Unverified units (e.g. unverified `dabba`):** Strictly blocked for human clarification.
    """)
