"""
demand_intelligence.py
Retail Demand Estimation Platform — Decision Intelligence Layer
Drop this file in the same folder as app.py and add the route below.
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
# MAIN ENGINE
# ─────────────────────────────────────────────────────────────

class DemandIntelligenceEngine:

    # ── volatility / trend thresholds ──────────────────────
    VOL_LOW  = 15   # CV % below this → LOW volatility
    VOL_MED  = 35   # CV % below this → MODERATE volatility
    SLOPE_WEAK   = 1.0   # % per month
    SLOPE_STRONG = 3.0

    # ── public entry point ──────────────────────────────────
    def analyze(self, history_values, history_labels,
                forecast_values, forecast_lower, forecast_upper,
                mape, demand_score, growth_percent,
                category_revenues=None):
        """
        Parameters
        ----------
        history_values  : list of floats  (monthly revenue, chronological)
        history_labels  : list of str     (YYYY-MM)
        forecast_values : list of floats  (next N months)
        forecast_lower  : list of floats
        forecast_upper  : list of floats
        mape            : float
        demand_score    : float (0-100, your existing DI score)
        growth_percent  : float
        category_revenues : dict {category_name: total_revenue}  (optional)
        """

        hist = np.array(history_values, dtype=float)
        fore = np.array(forecast_values, dtype=float)
        lower = np.array(forecast_lower, dtype=float)
        upper = np.array(forecast_upper, dtype=float)

        risk   = self._risk_profile(hist, fore)
        fi     = self._forecast_intel(fore, lower, upper, mape, demand_score, growth_percent)
        recs   = self._recommendations(risk, fi, category_revenues)
        cat_risks = self._category_risks(category_revenues)
        alerts = self._alerts(risk, fi, cat_risks)
        summary = self._executive_summary(risk, fi, recs)
        benchmarks = self._benchmarks(hist, history_labels, fore, forecast_values)

        return {
            "risk_profile": risk,
            "forecast_intel": fi,
            "recommendations": recs,
            "category_risks": cat_risks,
            "executive_summary": summary,
            "alerts": alerts,
            "kpi_benchmarks": benchmarks,
        }

    # ── Risk Profile ─────────────────────────────────────────
    def _risk_profile(self, hist, fore):
        mean = hist.mean() if hist.mean() != 0 else 1
        cv = (hist.std() / mean) * 100

        if cv < self.VOL_LOW:
            vol_class = "LOW"
        elif cv < self.VOL_MED:
            vol_class = "MODERATE"
        else:
            vol_class = "HIGH"

        # Trend: linear regression on last 6 months
        recent = hist[-6:] if len(hist) >= 6 else hist
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0] if len(recent) >= 3 else 0
        slope_pct = (slope / recent.mean() * 100) if recent.mean() != 0 else 0

        if slope_pct > self.SLOPE_STRONG:
            trend_dir, trend_str = "UPWARD", "STRONG"
        elif slope_pct > self.SLOPE_WEAK:
            trend_dir, trend_str = "UPWARD", "MODERATE"
        elif slope_pct < -self.SLOPE_STRONG:
            trend_dir, trend_str = "DOWNWARD", "STRONG"
        elif slope_pct < -self.SLOPE_WEAK:
            trend_dir, trend_str = "DOWNWARD", "MODERATE"
        else:
            trend_dir, trend_str = "FLAT", "WEAK"

        # Momentum: compare first half vs second half slope of recent window
        momentum = "STABLE"
        if len(recent) >= 4:
            mid = len(recent) // 2
            s1 = np.polyfit(np.arange(mid), recent[:mid], 1)[0]
            s2 = np.polyfit(np.arange(mid), recent[mid:], 1)[0]
            if s2 > s1 * 1.15:
                momentum = "ACCELERATING"
            elif s2 < s1 * 0.85:
                momentum = "DECELERATING"

        # Seasonality
        seasonality = False
        season_strength = 0.0
        if len(hist) >= 12:
            # Group by month position
            monthly_avgs = {}
            for i, v in enumerate(hist):
                m = i % 12
                monthly_avgs.setdefault(m, []).append(v)
            avgs = [np.mean(v) for v in monthly_avgs.values()]
            rng = (max(avgs) - min(avgs)) / mean
            seasonality = rng > 0.20
            season_strength = round(min(float(rng), 1.0), 3)

        # Composite risk score
        risk_score = self._compute_risk_score(cv, slope_pct, momentum, season_strength)

        if risk_score >= 70:
            risk_level = "CRITICAL"
        elif risk_score >= 50:
            risk_level = "HIGH"
        elif risk_score >= 30:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"

        return {
            "risk_level": risk_level,
            "risk_score": round(float(risk_score), 1),
            "volatility_class": vol_class,
            "volatility_cv": round(float(cv), 1),
            "trend_direction": trend_dir,
            "trend_strength": trend_str,
            "trend_slope_pct": round(float(slope_pct), 2),
            "momentum": momentum,
            "seasonality_detected": seasonality,
            "seasonality_strength": season_strength,
            "demand_stability_score": round(max(0.0, 100.0 - float(risk_score)), 1),
        }

    def _compute_risk_score(self, cv, slope_pct, momentum, season_strength):
        score = 0.0
        score += min(cv / self.VOL_MED * 40, 40)
        if slope_pct < 0:
            score += min(abs(slope_pct) / 5.0 * 30, 30)
        else:
            score -= min(slope_pct / 5.0 * 10, 10)
        if momentum == "DECELERATING":
            score += 15
        elif momentum == "ACCELERATING":
            score -= 5
        score += season_strength * 15
        return max(0.0, min(100.0, score))

    # ── Forecast Intelligence ─────────────────────────────────
    def _forecast_intel(self, fore, lower, upper, mape, demand_score, growth_pct):
        # confidence band width as % of forecast
        with np.errstate(divide="ignore", invalid="ignore"):
            band_pct = np.where(fore != 0, (upper - lower) / fore * 100, 0)
        avg_band = float(np.mean(band_pct))

        if mape <= 10:
            reliability = "HIGH"
        elif mape <= 20:
            reliability = "MODERATE"
        else:
            reliability = "LOW"

        if demand_score >= 80:
            outlook = "Strong Growth"
        elif demand_score >= 60:
            outlook = "Stable"
        elif demand_score >= 40:
            outlook = "Moderate Risk"
        else:
            outlook = "Critical Risk"

        peak_idx = int(np.argmax(fore))
        trough_idx = int(np.argmin(fore))

        return {
            "growth_pct": round(float(growth_pct), 2),
            "mape": round(float(mape), 2),
            "di_score": round(float(demand_score), 1),
            "outlook": outlook,
            "confidence_band_pct": round(avg_band, 1),
            "forecast_reliability": reliability,
            "peak_forecast_idx": peak_idx,
            "trough_forecast_idx": trough_idx,
        }

    # ── Recommendations ───────────────────────────────────────
    def _recommendations(self, risk, fi, category_revenues):
        recs = []
        cv  = risk["volatility_cv"]
        vol = risk["volatility_class"]
        td  = risk["trend_direction"]
        ts  = risk["trend_strength"]
        mom = risk["momentum"]
        sp  = risk["trend_slope_pct"]
        sea = risk["seasonality_detected"]
        ss  = risk["seasonality_strength"]
        rel = fi["forecast_reliability"]
        mape = fi["mape"]
        gp  = fi["growth_pct"]
        band = fi["confidence_band_pct"]

        # INVENTORY
        if vol == "HIGH":
            recs.append({
                "category": "INVENTORY", "priority": "CRITICAL",
                "icon": "📦",
                "title": "Increase Safety Stock Buffers",
                "action": "Raise safety stock levels by 25–35% above standard reorder points across top-revenue SKUs immediately.",
                "rationale": f"Demand volatility is HIGH (Coefficient of Variation = {cv}%). Combined with unpredictable demand spikes, current stock levels carry significant stockout risk.",
                "metric_trigger": f"CV: {cv}% (threshold: >35%)",
                "expected_impact": "Reduce stockout incidents by ~40% and protect 8–12% of at-risk revenue."
            })
        elif vol == "MODERATE":
            recs.append({
                "category": "INVENTORY", "priority": "HIGH",
                "icon": "📦",
                "title": "Switch to Dynamic Reorder Points",
                "action": "Replace fixed reorder points with rolling 3-month demand-adjusted triggers, reviewed monthly.",
                "rationale": f"Moderate volatility (CV = {cv}%) means static reorder points create periodic over/understocking cycles.",
                "metric_trigger": f"CV: {cv}% (threshold: 15–35%)",
                "expected_impact": "Reduce carrying costs by 10–15% while maintaining service levels above 95%."
            })
        else:
            recs.append({
                "category": "INVENTORY", "priority": "MEDIUM",
                "icon": "📦",
                "title": "Optimize Inventory Turns via Lean Replenishment",
                "action": "Reduce order quantities by 15% and increase order frequency to improve cash-flow without stockout risk.",
                "rationale": f"Low demand volatility (CV = {cv}%) creates a safe window to reduce holding costs.",
                "metric_trigger": f"CV: {cv}% (threshold: <15%)",
                "expected_impact": "Improve inventory turnover ratio by 0.5–1.0x and free up working capital."
            })

        # MARKETING / TREND
        if td == "DOWNWARD" and ts in ("STRONG", "MODERATE"):
            recs.append({
                "category": "MARKETING", "priority": "CRITICAL",
                "icon": "📣",
                "title": "Launch Demand Stimulation Campaign",
                "action": "Deploy targeted promotions (10–15% discount) on declining categories. Activate loyalty program incentives for lapsed customers within 14 days.",
                "rationale": f"Revenue is declining at {abs(sp)}%/month. Without intervention, revenue will contract ~{round(abs(sp)*3,1)}% over the next 3 months.",
                "metric_trigger": f"Monthly trend slope: {sp}%/month",
                "expected_impact": "Arrest decline and recover 50–70% of projected revenue loss within 60 days."
            })
        elif td == "UPWARD" and ts == "STRONG":
            recs.append({
                "category": "MARKETING", "priority": "HIGH",
                "icon": "📣",
                "title": "Amplify Growth Momentum",
                "action": "Increase marketing spend by 20% on top-performing categories. Expand product assortment in high-growth segments.",
                "rationale": f"Revenue is growing strongly at +{sp}%/month — the optimal window to acquire customers at lowest CAC.",
                "metric_trigger": f"Monthly trend slope: +{sp}%/month",
                "expected_impact": "Amplify natural growth by 15–25% and improve market share in trending segments."
            })

        # MOMENTUM
        if mom == "DECELERATING":
            recs.append({
                "category": "OPERATIONS", "priority": "HIGH",
                "icon": "⚙️",
                "title": "Diagnose Root Cause of Decelerating Growth",
                "action": "Conduct SKU-level performance review. Audit pricing competitiveness, supplier lead times, and top-category margins this week.",
                "rationale": "Growth is decelerating — recent months underperform earlier periods even if overall trend is positive. This is an early warning of potential trend reversal.",
                "metric_trigger": "Momentum: DECELERATING (second-half monthly slope < first-half slope)",
                "expected_impact": "Identify 2–3 corrective levers to restore momentum within 45 days."
            })

        # FORECAST RELIABILITY
        if rel == "LOW":
            recs.append({
                "category": "RISK", "priority": "HIGH",
                "icon": "🛡️",
                "title": "Build Flexible Supply Commitments",
                "action": "Negotiate purchase orders with 30% volume optionality. Avoid long-term inventory lock-in until forecast accuracy improves.",
                "rationale": f"Model error is high (MAPE: {mape}%). Rigid supply commitments under this uncertainty carry significant overstock/stockout risk.",
                "metric_trigger": f"MAPE: {mape}% (threshold: >20%)",
                "expected_impact": "Reduce inventory risk exposure by 25–35% during high-uncertainty periods."
            })
        elif rel == "HIGH" and gp > 5:
            recs.append({
                "category": "OPERATIONS", "priority": "MEDIUM",
                "icon": "⚙️",
                "title": "Pre-Position Inventory for Forecasted Peak",
                "action": f"Pre-order 15–20% above current stock to prepare for the peak forecast month. Execute purchasing within the next 3 weeks.",
                "rationale": f"High forecast confidence (MAPE: {mape}%) with +{gp}% growth projection means this is a high-confidence opportunity to pre-stock.",
                "metric_trigger": f"MAPE: {mape}%, Growth: +{gp}%",
                "expected_impact": "Capture 100% of forecasted demand uplift with minimal stockout risk."
            })

        # SEASONALITY
        if sea:
            recs.append({
                "category": "PRICING", "priority": "MEDIUM",
                "icon": "💰",
                "title": "Implement Seasonal Dynamic Pricing",
                "action": "Apply +5–8% price markup in peak demand months and –8–12% promotional discounts during trough periods.",
                "rationale": f"Seasonality detected with strength score of {round(ss*100)}%. Flat pricing across seasonal cycles leaves margin on the table.",
                "metric_trigger": f"Seasonality strength: {round(ss*100,1)}%",
                "expected_impact": "Improve gross margin by 3–6% annually through demand-responsive pricing."
            })

        # WIDE CONFIDENCE BANDS
        if band > 30:
            recs.append({
                "category": "RISK", "priority": "MEDIUM",
                "icon": "🛡️",
                "title": "Activate Scenario-Based Planning",
                "action": "Create 3 operational plans (Base / Bull / Bear) from forecast confidence bounds. Set inventory action triggers for each scenario.",
                "rationale": f"Forecast confidence interval spans {band}% of predicted revenue, indicating structural demand uncertainty.",
                "metric_trigger": f"Confidence band width: {band}%",
                "expected_impact": "Cut reaction time to demand shocks by 50% through pre-planned operational responses."
            })

        # Sort by priority
        order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        recs.sort(key=lambda r: order.get(r["priority"], 4))
        return recs[:6]

    # ── Category Risk ─────────────────────────────────────────
    def _category_risks(self, category_revenues):
        if not category_revenues:
            return []
        total = sum(category_revenues.values()) or 1
        result = []
        for cat, rev in category_revenues.items():
            share = round(rev / total * 100, 1)
            # Without time-series per category we assign heuristic risk
            risk_score = round(max(10, 60 - share), 1) if share < 20 else round(max(10, 40 - share * 0.5), 1)
            result.append({
                "category": cat,
                "revenue_share": share,
                "risk_score": risk_score,
                "recommendation": (
                    "Core revenue driver — invest to protect and grow."
                    if share > 25
                    else "Monitor closely; diversify if declining."
                )
            })
        result.sort(key=lambda x: x["revenue_share"], reverse=True)
        return result[:8]

    # ── Alerts ────────────────────────────────────────────────
    def _alerts(self, risk, fi, cat_risks):
        alerts = []
        rl  = risk["risk_level"]
        cv  = risk["volatility_cv"]
        vol = risk["volatility_class"]
        mom = risk["momentum"]
        sea = risk["seasonality_detected"]
        mape = fi["mape"]
        gp   = fi["growth_pct"]
        rel  = fi["forecast_reliability"]
        rs   = risk["risk_score"]

        if rl == "CRITICAL":
            alerts.append({"level": "critical", "icon": "🚨",
                "message": f"Critical demand risk detected (score: {rs}/100). Immediate strategic action required."})
        if mape > 20:
            alerts.append({"level": "warning", "icon": "⚠️",
                "message": f"Low forecast accuracy (MAPE: {mape}%). Treat projections as directional, not precise."})
        if vol == "HIGH":
            alerts.append({"level": "warning", "icon": "📊",
                "message": f"High demand volatility (CV: {cv}%). Safety stock buffers are critical to prevent stockouts."})
        if mom == "DECELERATING":
            alerts.append({"level": "warning", "icon": "📉",
                "message": "Growth momentum is decelerating. Early intervention can prevent a full trend reversal."})
        if gp > 10 and rel == "HIGH":
            alerts.append({"level": "success", "icon": "🚀",
                "message": f"Strong, high-confidence growth forecast (+{gp}%). Capitalize now — pre-position inventory."})
        if sea:
            alerts.append({"level": "info", "icon": "🗓️",
                "message": "Seasonal demand patterns detected. Align inventory purchases and promotions with seasonal cycles."})
        return alerts

    # ── Executive Summary ─────────────────────────────────────
    def _executive_summary(self, risk, fi, recs):
        td  = risk["trend_direction"]
        sp  = risk["trend_slope_pct"]
        cv  = risk["volatility_cv"]
        vol = risk["volatility_class"]
        mom = risk["momentum"].lower()
        rl  = risk["risk_level"]
        rs  = risk["risk_score"]
        gp  = fi["growth_pct"]
        mape = fi["mape"]
        rel  = fi["forecast_reliability"].lower()

        direction_text = {
            "UPWARD":   f"growing at +{sp}%/month",
            "DOWNWARD": f"declining at {abs(sp)}%/month",
            "FLAT":     "moving sideways with no clear direction"
        }.get(td, "stable")

        critical = sum(1 for r in recs if r["priority"] == "CRITICAL")
        high     = sum(1 for r in recs if r["priority"] == "HIGH")

        summary = (
            f"Demand is currently {direction_text} with {vol.lower()} volatility "
            f"(CV: {cv}%) and {mom} momentum. "
            f"The 3-month forecast projects {'+' if gp > 0 else ''}{gp}% revenue change "
            f"backed by {rel} model confidence (MAPE: {mape}%). "
            f"Overall demand risk is classified as {rl} (score: {rs}/100). "
        )
        if critical:
            summary += f"{critical} critical action{'s' if critical > 1 else ''} require immediate attention. "
        if high:
            summary += f"{high} high-priority recommendation{'s' if high > 1 else ''} should be actioned within 30 days."

        return summary

    # ── Benchmarks ────────────────────────────────────────────
    def _benchmarks(self, hist, history_labels, fore, forecast_values):
        changes = np.diff(hist) / hist[:-1] * 100
        avg_growth = float(np.mean(changes)) if len(changes) else 0

        peak_idx   = int(np.argmax(hist))
        trough_idx = int(np.argmin(hist))

        peak_label   = history_labels[peak_idx]   if history_labels else "N/A"
        trough_label = history_labels[trough_idx] if history_labels else "N/A"

        return {
            "avg_monthly_growth_pct": round(avg_growth, 2),
            "peak_revenue":    round(float(hist.max()), 2),
            "trough_revenue":  round(float(hist.min()), 2),
            "revenue_range":   round(float(hist.max() - hist.min()), 2),
            "peak_month":      peak_label,
            "trough_month":    trough_label,
            "forecast_peak":   round(float(max(forecast_values)), 2),
            "forecast_trough": round(float(min(forecast_values)), 2),
            "months_analyzed": len(hist),
        }
