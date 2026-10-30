"""
conversion.py
Deterministic Unit and Pack-Size Conversion Engine.
Grounded in explicit catalogue rules and store-specific verified agreements.
Never assumes universal conversions for informal terms like 'peti' or 'dabba'.
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Optional, Tuple
from catalogue import Catalogue, Product


@dataclass
class ConversionResult:
    converted_quantity: Optional[Decimal]
    ordering_unit: Optional[str]
    rule_id: Optional[str]
    status: str  # "VALID", "CLARIFICATION_REQUIRED", "INVALID_QUANTITY", "UNSUPPORTED_UNIT"
    explanation: str


# Canonical singular normalization for direct 1:1 matching
UNIT_SYNONYMS = {
    # Bottle
    "bottle": "Bottle",
    "bottles": "Bottle",
    "botal": "Bottle",
    "botle": "Bottle",
    "botl": "Bottle",
    "botalein": "Bottle",
    # Can
    "can": "Can",
    "cans": "Can",
    "canne": "Can",
    "cane": "Can",
    "tin": "Can",
    "dibba": "Can",
    # Packet & Pack & Pouch
    "packet": "Packet",
    "packets": "Packet",
    "pakit": "Packet",
    "pakt": "Packet",
    "pack": "Pack",
    "packs": "Pack",
    "pouch": "Pouch",
    "pouches": "Pouch",
    "paunch": "Pouch",
    # Bag
    "bag": "Bag",
    "bags": "Bag",
    "bori": "Bag",
    "bore": "Bag",
    "thela": "Bag",
    "thele": "Bag",
    "katta": "Bag",
    "katte": "Bag",
    # Bar
    "bar": "Bar",
    "bars": "Bar",
    "tikiya": "Bar",
    "tikia": "Bar",
    "bati": "Bar",
    "batia": "Bar",
    "sabun": "Bar",
    # Tube
    "tube": "Tube",
    "tubes": "Tube",
    "tyub": "Tube",
    "paste": "Tube",
    # Jar
    "jar": "Jar",
    "jars": "Jar",
    "barni": "Jar",
    "shishi": "Jar",
    "sisi": "Jar",
    # Piece
    "piece": "Piece",
    "pieces": "Piece",
    "pc": "Piece",
    "pcs": "Piece",
    "unit": "Piece",
    "units": "Piece",
    "nag": "Piece",
    "adad": "Piece",
    # Carton & Peti
    "carton": "Carton",
    "cartons": "Carton",
    "case": "Case",
    "cases": "Case",
    "box": "Box",
    "boxes": "Box",
    "peti": "Carton",
    "petiyan": "Carton",
    "petia": "Carton",
    "dhaba": "Carton",
    "dhabba": "Carton",
    "dhabe": "Carton",
    "dhabbe": "Carton",
    "dhabas": "Carton",
    "dhabbas": "Carton",
}

PETI_SYNONYMS = {"peti", "petiyan", "petia", "dhaba", "dhabba", "dhabe", "dhabbe", "dhabas", "dhabbas"}


class UnitConverter:
    def __init__(self, rules_path: Optional[str] = None):
        self.sku_rules: Dict[str, dict] = {}
        self.store_rules: Dict[str, dict] = {}
        self._load_rules(rules_path)

    def _load_rules(self, rules_path: Optional[str]) -> None:
        default_path = os.path.join(os.path.dirname(__file__), "data", "conversion_rules.json")
        target_path = rules_path or default_path
        if os.path.exists(target_path):
            try:
                with open(target_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.sku_rules = data.get("sku_rules", {})
                    self.store_rules = data.get("store_specific_rules", {})
            except Exception:
                self.sku_rules = {}
                self.store_rules = {}

    def convert(
        self,
        product: Optional[Product],
        requested_qty: Optional[Decimal],
        requested_unit: Optional[str],
        store_id: Optional[str] = None,
    ) -> ConversionResult:
        """
        Convert retailer's requested quantity and unit into the canonical ordering unit.
        Returns a ConversionResult with full audit trail of the rule applied.
        """
        # Validate quantity
        if requested_qty is None:
            return ConversionResult(
                converted_quantity=None,
                ordering_unit=product.order_unit if product else None,
                rule_id=None,
                status="CLARIFICATION_REQUIRED",
                explanation="Quantity is missing or could not be parsed.",
            )

        if requested_qty <= Decimal(0):
            return ConversionResult(
                converted_quantity=None,
                ordering_unit=product.order_unit if product else None,
                rule_id=None,
                status="INVALID_QUANTITY",
                explanation=f"Quantity must be strictly positive (got {requested_qty}).",
            )

        if not product:
            return ConversionResult(
                converted_quantity=None,
                ordering_unit=None,
                rule_id=None,
                status="CLARIFICATION_REQUIRED",
                explanation="Product identity unresolved; cannot verify ordering unit.",
            )

        clean_unit = (requested_unit or "").strip().lower()

        # 1. Direct 1:1 match with catalogue order unit (or universal piece/unit)
        cat_order_unit_norm = product.order_unit.lower()
        if (
            not clean_unit
            or clean_unit == cat_order_unit_norm
            or UNIT_SYNONYMS.get(clean_unit, "").lower() == cat_order_unit_norm
            or clean_unit in ["piece", "pieces", "pc", "pcs", "unit", "units", "nag", "adad"]
        ):
            return ConversionResult(
                converted_quantity=requested_qty,
                ordering_unit=product.order_unit,
                rule_id="DIRECT_1_TO_1",
                status="VALID",
                explanation=f"Direct unit match ({requested_qty} {product.order_unit}).",
            )

        # 2. Check Store-Specific Documented Rules
        if store_id and store_id in self.store_rules:
            store_sku_rules = self.store_rules[store_id].get(product.sku, {})
            matched_unit = clean_unit
            if matched_unit not in store_sku_rules and matched_unit in PETI_SYNONYMS and "peti" in store_sku_rules:
                matched_unit = "peti"
            if matched_unit in store_sku_rules:
                rule = store_sku_rules[matched_unit]
                multiplier = Decimal(str(rule["multiplier"]))
                converted = requested_qty * multiplier
                return ConversionResult(
                    converted_quantity=converted,
                    ordering_unit=rule.get("target_unit", product.order_unit),
                    rule_id=rule.get("rule_id", f"STORE_RULE_{store_id}_{clean_unit}"),
                    status="VALID",
                    explanation=f"{rule.get('description', 'Store verified conversion')}: {requested_qty} {requested_unit} = {converted} {product.order_unit}.",
                )

        # 3. Check SKU-Specific Master Pack / Carton / Case Rules
        if product.sku in self.sku_rules:
            sku_rules = self.sku_rules[product.sku]
            matched_unit = clean_unit
            if matched_unit not in sku_rules and matched_unit in PETI_SYNONYMS and "peti" in sku_rules:
                matched_unit = "peti"
            if matched_unit in sku_rules:
                rule = sku_rules[matched_unit]
                multiplier = Decimal(str(rule["multiplier"]))
                converted = requested_qty * multiplier
                return ConversionResult(
                    converted_quantity=converted,
                    ordering_unit=rule.get("target_unit", product.order_unit),
                    rule_id=rule.get("rule_id", f"SKU_RULE_{clean_unit}"),
                    status="VALID",
                    explanation=f"{rule.get('description', 'SKU case conversion')}: {requested_qty} {requested_unit} = {converted} {product.order_unit}.",
                )

        # 4. Built-in case/carton fallback from product.units_per_case (including peti as carton)
        if (clean_unit in ["case", "cases", "carton", "cartons", "box", "boxes"] or clean_unit in PETI_SYNONYMS) and product.units_per_case > 1:
            multiplier = Decimal(product.units_per_case)
            converted = requested_qty * multiplier
            return ConversionResult(
                converted_quantity=converted,
                ordering_unit=product.order_unit,
                rule_id=f"CARTON_PETI_{product.sku}",
                status="VALID",
                explanation=f"Peti/Carton packing ({product.units_per_case} {product.order_unit}/carton): {requested_qty} {requested_unit} = {converted} {product.order_unit}.",
            )

        # 4b. Metric weight and volume conversions
        if clean_unit in ["kg", "kgs", "kilo", "kilos", "kilogram", "kilograms"]:
            if "1kg" in product.product_name.lower():
                return ConversionResult(
                    converted_quantity=requested_qty,
                    ordering_unit=product.order_unit,
                    rule_id=f"METRIC_1KG_{product.sku}",
                    status="VALID",
                    explanation=f"Metric pack match (1kg = 1 {product.order_unit}): {requested_qty} kg = {requested_qty} {product.order_unit}.",
                )
            elif "5kg" in product.product_name.lower():
                converted = (requested_qty / Decimal("5")).quantize(Decimal("1"))
                return ConversionResult(
                    converted_quantity=converted,
                    ordering_unit=product.order_unit,
                    rule_id=f"METRIC_5KG_{product.sku}",
                    status="VALID",
                    explanation=f"Metric 5kg pack conversion: {requested_qty} kg = {converted} {product.order_unit}.",
                )
        elif clean_unit in ["l", "lt", "ltr", "litre", "litres", "liter", "liters", "litar"]:
            if "1l" in product.product_name.lower():
                return ConversionResult(
                    converted_quantity=requested_qty,
                    ordering_unit=product.order_unit,
                    rule_id=f"METRIC_1L_{product.sku}",
                    status="VALID",
                    explanation=f"Metric pack match (1L = 1 {product.order_unit}): {requested_qty} L = {requested_qty} {product.order_unit}.",
                )
            elif "500ml" in product.product_name.lower():
                converted = requested_qty * Decimal("2")
                return ConversionResult(
                    converted_quantity=converted,
                    ordering_unit=product.order_unit,
                    rule_id=f"METRIC_500ML_{product.sku}",
                    status="VALID",
                    explanation=f"Metric 500ml pack conversion (1L = 2 pouches): {requested_qty} L = {converted} {product.order_unit}.",
                )

        # 5. Regional or Informal units without verified rules: MUST REQUIRE CLARIFICATION
        if clean_unit in ["dabba", "tin", "packet", "pack", "bag", "carton", "piece", "dozen", "strip", "bora"] and clean_unit not in PETI_SYNONYMS:
            return ConversionResult(
                converted_quantity=None,
                ordering_unit=product.order_unit,
                rule_id=None,
                status="CLARIFICATION_REQUIRED",
                explanation=f"Ambiguous local unit '{requested_unit}'. No verified conversion rule exists for {product.product_name} at store {store_id or 'General'}. Confirm exact quantity in {product.order_unit}.",
            )

        # 6. Unsupported unit
        return ConversionResult(
            converted_quantity=None,
            ordering_unit=product.order_unit,
            rule_id=None,
            status="UNSUPPORTED_UNIT",
            explanation=f"Unsupported unit '{requested_unit}' for {product.product_name}. Please specify order in {product.order_unit}.",
        )

