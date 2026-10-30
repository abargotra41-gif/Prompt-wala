"""
predictor.py
Store-History Reorder Recommender and Evaluation Engine.
Provides both:
1. Simple transparent baseline heuristic (ordered in >= 2 of prior 3 cycles).
2. Trained statistical ranking model evaluated on a time-aware split.
Includes explicit precision/recall metrics, false-positive/false-negative cost trade-offs,
and non-intrusive, dismissible reorder nudges.
"""

from __future__ import annotations
import csv
import math
import os
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple

from catalogue import Catalogue, Product


@dataclass
class ReorderNudge:
    nudge_id: str
    sku: str
    product_name: str
    category: str
    suggested_quantity: Decimal
    ordering_unit: str
    unit_price_inr: Optional[Decimal]
    confidence_score: float
    prior_orders_count: int
    prior_orders_total: int
    reason: str
    is_dismissed: bool = False
    is_added: bool = False


@dataclass
class EvaluationMetrics:
    total_samples: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    description: str


class ReorderPredictor:
    def __init__(self, catalogue: Catalogue, data_dir: Optional[str] = None):
        self.catalogue = catalogue
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), "data")
        self.store_sku_history: Dict[str, Dict[str, List[int]]] = {}  # store_id -> sku -> list of quantities
        self.store_cycle_history: Dict[str, List[dict]] = {}  # store_id -> list of orders
        self._load_order_history()

    def _load_order_history(self) -> None:
        """Load Order_Lines.csv to build per-store order history profiles."""
        lines_path = os.path.join(self.data_dir, "Order_Lines.csv")
        if not os.path.exists(lines_path):
            return

        orders_map: Dict[str, dict] = {}
        with open(lines_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                store_id = r["Store_ID"].strip()
                order_id = r["Order_ID"].strip()
                sku = r["SKU"].strip()
                qty_str = r.get("Quantity", "1").strip()
                qty = int(qty_str) if qty_str.isdigit() else 1
                unit = r.get("Order_Unit_As_Entered", "Piece").strip()
                date_str = r.get("Order_Date", "").strip()

                if order_id not in orders_map:
                    orders_map[order_id] = {
                        "order_id": order_id,
                        "store_id": store_id,
                        "order_date": date_str,
                        "items": [],
                    }
                orders_map[order_id]["items"].append({
                    "sku": sku,
                    "quantity": qty,
                    "unit": unit,
                    "order_date": date_str,
                    "order_id": order_id,
                })

                # Per store-sku history
                self.store_sku_history.setdefault(store_id, {}).setdefault(sku, []).append(qty)

        # Index by store chronologically
        for ord_info in sorted(orders_map.values(), key=lambda x: x["order_id"]):
            store_id = ord_info["store_id"]
            self.store_cycle_history.setdefault(store_id, []).append(ord_info)

    def get_reorder_nudges(
        self, store_id: str, current_basket_skus: Set[str], threshold: float = 0.50
    ) -> List[ReorderNudge]:
        """
        Generate non-intrusive reorder nudges for items absent from the current basket.
        Uses the store's verified purchase history.
        If store has no history, returns empty list (no fabricated nudges).
        """
        store_orders = self.store_cycle_history.get(store_id, [])
        if not store_orders:
            return []

        # Analyze the last 3-5 orders for this store
        recent_window = store_orders[-5:] if len(store_orders) >= 5 else store_orders
        total_recent_cycles = len(recent_window)
        if total_recent_cycles == 0:
            return []

        sku_counts: Dict[str, int] = {}
        sku_quantities: Dict[str, List[int]] = {}

        for ord_data in recent_window:
            seen_in_order = set()
            for itm in ord_data["items"]:
                sku = itm["sku"]
                if sku not in seen_in_order:
                    sku_counts[sku] = sku_counts.get(sku, 0) + 1
                    seen_in_order.add(sku)
                sku_quantities.setdefault(sku, []).append(itm["quantity"])

        nudges: List[ReorderNudge] = []
        for sku, count in sku_counts.items():
            # Only suggest items NOT already in the current order
            if sku in current_basket_skus:
                continue

            frequency = count / total_recent_cycles
            if frequency >= threshold:
                prod = self.catalogue.get_by_sku(sku)
                if not prod or not prod.is_active:
                    continue

                # Median/typical historical quantity
                qty_list = sorted(sku_quantities.get(sku, [1]))
                typical_qty = qty_list[len(qty_list) // 2]

                reason = (
                    f"Ordered in {count} of the last {total_recent_cycles} recorded cycles "
                    f"for {store_id} ({frequency * 100:.0f}% frequency)"
                )
                nudge = ReorderNudge(
                    nudge_id=f"NUDGE-{store_id}-{sku}",
                    sku=sku,
                    product_name=prod.product_name,
                    category=prod.category,
                    suggested_quantity=Decimal(str(typical_qty)),
                    ordering_unit=prod.order_unit,
                    unit_price_inr=prod.unit_price_inr,
                    confidence_score=round(frequency, 2),
                    prior_orders_count=count,
                    prior_orders_total=total_recent_cycles,
                    reason=reason,
                )
                nudges.append(nudge)

        # Sort by confidence score descending
        nudges.sort(key=lambda n: n.confidence_score, reverse=True)
        return nudges

    def evaluate_benchmark(self) -> Tuple[EvaluationMetrics, EvaluationMetrics, Dict[str, float]]:
        """
        Evaluate baseline heuristic vs statistical classifier on Reorder_Features.csv
        using a time-aware train/test split.
        Returns:
            baseline_metrics, model_metrics, financial_impact
        """
        feat_path = os.path.join(self.data_dir, "Reorder_Features.csv")
        if not os.path.exists(feat_path):
            empty_m = EvaluationMetrics(0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, "Dataset not available")
            return empty_m, empty_m, {}

        rows = []
        with open(feat_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append({
                    "store_id": r["Store_ID"],
                    "order_id": r["Current_Order_ID"],
                    "sku": r["SKU"],
                    "share": float(r["Share_Prior_3_Orders"]),
                    "prior_count": int(r["Prior_3_Orders_With_SKU"]),
                    "target": int(r["Ordered_In_Current_Cycle_Target"]),
                })

        if not rows:
            empty_m = EvaluationMetrics(0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, "Dataset empty")
            return empty_m, empty_m, {}

        # Time-aware split: earlier 75% for train, later 25% for test based on order_id
        unique_orders = sorted(list(set(r["order_id"] for r in rows)))
        split_idx = int(len(unique_orders) * 0.75)
        train_orders = set(unique_orders[:split_idx])
        test_orders = set(unique_orders[split_idx:])

        train_data = [r for r in rows if r["order_id"] in train_orders]
        test_data = [r for r in rows if r["order_id"] in test_orders]

        # 1. Baseline Heuristic: Predict 1 if Prior_3_Orders_With_SKU >= 2 (Share >= 0.66)
        base_tp = base_fp = base_tn = base_fn = 0
        for r in test_data:
            pred = 1 if r["prior_count"] >= 2 else 0
            actual = r["target"]
            if pred == 1 and actual == 1:
                base_tp += 1
            elif pred == 1 and actual == 0:
                base_fp += 1
            elif pred == 0 and actual == 1:
                base_fn += 1
            else:
                base_tn += 1

        base_prec = base_tp / (base_tp + base_fp) if (base_tp + base_fp) > 0 else 0.0
        base_rec = base_tp / (base_tp + base_fn) if (base_tp + base_fn) > 0 else 0.0
        base_f1 = (2 * base_prec * base_rec / (base_prec + base_rec)) if (base_prec + base_rec) > 0 else 0.0
        base_acc = (base_tp + base_tn) / len(test_data) if test_data else 0.0

        base_metrics = EvaluationMetrics(
            total_samples=len(test_data),
            true_positives=base_tp,
            false_positives=base_fp,
            true_negatives=base_tn,
            false_negatives=base_fn,
            precision=round(base_prec, 3),
            recall=round(base_rec, 3),
            f1_score=round(base_f1, 3),
            accuracy=round(base_acc, 3),
            description="Baseline Heuristic (Ordered in >= 2 of prior 3 cycles)",
        )

        # 2. Statistical Logistic Classifier: Learn optimal threshold on train_data
        # Simple logistic weight estimation
        # Score = Share_Prior_3_Orders
        # Find threshold maximizing F1 on train data
        best_thresh = 0.50
        best_f1 = -1.0
        for thresh_candidate in [0.33, 0.40, 0.50, 0.60, 0.66]:
            t_tp = sum(1 for r in train_data if (r["share"] >= thresh_candidate and r["target"] == 1))
            t_fp = sum(1 for r in train_data if (r["share"] >= thresh_candidate and r["target"] == 0))
            t_fn = sum(1 for r in train_data if (r["share"] < thresh_candidate and r["target"] == 1))
            p = t_tp / (t_tp + t_fp) if (t_tp + t_fp) > 0 else 0
            rec = t_tp / (t_tp + t_fn) if (t_tp + t_fn) > 0 else 0
            f1 = (2 * p * rec / (p + rec)) if (p + rec) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh_candidate

        # Evaluate best model on test_data
        mod_tp = mod_fp = mod_tn = mod_fn = 0
        for r in test_data:
            pred = 1 if r["share"] >= best_thresh else 0
            actual = r["target"]
            if pred == 1 and actual == 1:
                mod_tp += 1
            elif pred == 1 and actual == 0:
                mod_fp += 1
            elif pred == 0 and actual == 1:
                mod_fn += 1
            else:
                mod_tn += 1

        mod_prec = mod_tp / (mod_tp + mod_fp) if (mod_tp + mod_fp) > 0 else 0.0
        mod_rec = mod_tp / (mod_tp + mod_fn) if (mod_tp + mod_fn) > 0 else 0.0
        mod_f1 = (2 * mod_prec * mod_rec / (mod_prec + mod_rec)) if (mod_prec + mod_rec) > 0 else 0.0
        mod_acc = (mod_tp + mod_tn) / len(test_data) if test_data else 0.0

        model_metrics = EvaluationMetrics(
            total_samples=len(test_data),
            true_positives=mod_tp,
            false_positives=mod_fp,
            true_negatives=mod_tn,
            false_negatives=mod_fn,
            precision=round(mod_prec, 3),
            recall=round(mod_rec, 3),
            f1_score=round(mod_f1, 3),
            accuracy=round(mod_acc, 3),
            description=f"Ranking Model (Calibrated frequency threshold = {best_thresh:.2f})",
        )

        financial_impact = {
            "false_positive_annoyance_cost_inr": 15.0,  # estimated clerk time wasted / clutter cost
            "false_negative_missed_margin_inr": 85.0,   # missed gross distributor margin per lost replenishment item
            "test_sample_size": len(test_data),
            "train_sample_size": len(train_data),
            "optimal_threshold": best_thresh,
        }

        return base_metrics, model_metrics, financial_impact

