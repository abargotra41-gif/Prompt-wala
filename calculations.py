"""
calculations.py
Deterministic Monetary and Line Total Calculations Using Exact Decimal Arithmetic.
Never relies on floating point math or LLM-generated calculations.
"""

from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Optional


@dataclass
class LineCalculation:
    sku: Optional[str]
    quantity: Optional[Decimal]
    ordering_unit: Optional[str]
    unit_price_inr: Optional[Decimal]
    price_basis: Optional[str]
    line_total_inr: Optional[Decimal]
    is_valid: bool
    blocking_reasons: List[str]


@dataclass
class OrderSummary:
    line_count: int
    valid_lines_count: int
    unresolved_lines_count: int
    grand_total_inr: Optional[Decimal]
    can_confirm: bool
    summary_messages: List[str]


def calculate_line_item(
    sku: Optional[str],
    ordering_quantity: Optional[Decimal],
    ordering_unit: Optional[str],
    unit_price_inr: Optional[Decimal],
    match_status: str,
    conversion_status: str,
) -> LineCalculation:
    """
    Deterministic calculation for a single line item.
    Enforces that SKU exists, price is available, and quantity is strictly positive.
    """
    reasons = []

    if match_status != "MATCHED" or not sku:
        reasons.append("Product not identified or pending clarification.")

    if unit_price_inr is None:
        reasons.append("Catalogue price unavailable.")

    if ordering_quantity is None:
        reasons.append("Ordering quantity not resolved.")
    elif ordering_quantity <= Decimal("0"):
        reasons.append("Ordering quantity must be positive.")

    if conversion_status != "VALID":
        reasons.append(f"Unit conversion issue ({conversion_status}).")

    if reasons:
        return LineCalculation(
            sku=sku,
            quantity=ordering_quantity,
            ordering_unit=ordering_unit,
            unit_price_inr=unit_price_inr,
            price_basis=f"Per {ordering_unit}" if ordering_unit else None,
            line_total_inr=None,
            is_valid=False,
            blocking_reasons=reasons,
        )

    # Safe deterministic multiplication
    total = (ordering_quantity * unit_price_inr).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    return LineCalculation(
        sku=sku,
        quantity=ordering_quantity,
        ordering_unit=ordering_unit,
        unit_price_inr=unit_price_inr,
        price_basis=f"Per {ordering_unit}",
        line_total_inr=total,
        is_valid=True,
        blocking_reasons=[],
    )


def calculate_grand_total(line_calculations: List[LineCalculation]) -> OrderSummary:
    """
    Deterministic grand total aggregation.
    Blocks order confirmation if any line has an unresolved issue or missing price.
    """
    if not line_calculations:
        return OrderSummary(
            line_count=0,
            valid_lines_count=0,
            unresolved_lines_count=0,
            grand_total_inr=Decimal("0.00"),
            can_confirm=False,
            summary_messages=["Order contains no line items."],
        )

    total = Decimal("0.00")
    valid_count = 0
    unresolved_count = 0
    all_reasons = []

    for idx, calc in enumerate(line_calculations, start=1):
        if calc.is_valid and calc.line_total_inr is not None:
            total += calc.line_total_inr
            valid_count += 1
        else:
            unresolved_count += 1
            for r in calc.blocking_reasons:
                all_reasons.append(f"Line {idx}: {r}")

    can_confirm = (unresolved_count == 0 and valid_count > 0)
    summary_msgs = []
    if can_confirm:
        summary_msgs.append(f"All {valid_count} line items verified against catalogue.")
    else:
        summary_msgs.append(f"{unresolved_count} item(s) require resolution before order can be confirmed.")
        summary_msgs.extend(all_reasons[:5])

    return OrderSummary(
        line_count=len(line_calculations),
        valid_lines_count=valid_count,
        unresolved_lines_count=unresolved_count,
        grand_total_inr=total if can_confirm else None,
        can_confirm=can_confirm,
        summary_messages=summary_msgs,
    )

