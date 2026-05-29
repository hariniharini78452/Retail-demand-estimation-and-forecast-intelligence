import os
import json
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import numpy as np
from prophet import Prophet
from demand_intelligence import DemandIntelligenceEngine

engine = DemandIntelligenceEngine()

app = Flask(__name__)
app.secret_key = "super_secret_key_change_this"

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def detect_columns(df):
    mapping = {"date": None, "price": None, "quantity": None, "category": None}

    date_kw     = ["date","time","day","month","year","period","timestamp","order_date","sale_date","trans","invoice","purchased","created","dt"]
    price_kw    = ["price","amount","revenue","sales","total","value","cost","income","earning","turnover","gross","net","sale_amount","unit_price","rate","charge","fee","payment","spend"]
    quantity_kw = ["qty","quantity","units","count","volume","sold","pieces","num_items","number","ordered","demand","items"]
    category_kw = ["category","product","item","model","type","brand","segment","class","group","department","name","sku","description","sub_category","subcategory","region","store","division"]

    for col in df.columns:
        cl = col.lower().replace(" ", "_")

        if mapping["date"] is None and any(k in cl for k in date_kw):
            try:
                sample = pd.to_datetime(df[col].dropna().head(10), errors="coerce")
                if sample.notna().sum() >= 3:
                    mapping["date"] = col
                    continue
            except:
                pass

        if mapping["price"] is None and any(k in cl for k in price_kw):
            if pd.to_numeric(df[col], errors="coerce").notna().sum() > len(df) * 0.5:
                mapping["price"] = col
                continue

        if mapping["quantity"] is None and any(k in cl for k in quantity_kw):
            if pd.to_numeric(df[col], errors="coerce").notna().sum() > len(df) * 0.5:
                mapping["quantity"] = col
                continue

        if mapping["category"] is None and any(k in cl for k in category_kw):
            if df[col].dtype == object or df[col].nunique() < 200:
                mapping["category"] = col
                continue

    if mapping["date"] is None:
        for col in df.columns:
            try:
                sample = pd.to_datetime(df[col].dropna().head(20), errors="coerce")
                if sample.notna().sum() >= 5:
                    mapping["date"] = col
                    break
            except:
                pass

    if mapping["price"] is None:
        for col in df.columns:
            if col in [mapping["date"], mapping["quantity"]]:
                continue
            nums = pd.to_numeric(df[col], errors="coerce")
            if nums.notna().sum() > len(df) * 0.5 and nums.mean() > 0:
                mapping["price"] = col
                break

    if mapping["category"] is None:
        for col in df.columns:
            if col in [mapping["date"], mapping["price"], mapping["quantity"]]:
                continue
            if df[col].dtype == object and 1 < df[col].nunique() < 200:
                mapping["category"] = col
                break

    return mapping


def calc_revenue(df, mapping):
    price_col    = mapping.get("price")
    quantity_col = mapping.get("quantity")

    if price_col:
        df[price_col] = pd.to_numeric(df[price_col], errors="coerce").fillna(0)
    if quantity_col:
        df[quantity_col] = pd.to_numeric(df[quantity_col], errors="coerce").fillna(0)

    if price_col:
        cl = price_col.lower().replace(" ", "_")
        is_total = any(k in cl for k in ["revenue","sales","total","income","turnover",
                                          "gross","net","earning","payment","spend","amount"])
        if is_total:
            # Column is already a total — use directly
            df["revenue"] = df[price_col]
        elif quantity_col:
            # Unit price × quantity
            df["revenue"] = df[price_col] * df[quantity_col]
        else:
            df["revenue"] = df[price_col]
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            best = max(num_cols, key=lambda c: df[c].mean() if df[c].mean() > 0 else 0)
            df["revenue"] = pd.to_numeric(df[best], errors="coerce").fillna(0)
        else:
            df["revenue"] = 0
    return df


def safe_jsonify(data):
    def converter(obj):
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj): return None
            return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)): return None
        return str(obj)
    return app.response_class(response=json.dumps(data, default=converter), mimetype="application/json")


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    if "uploaded_file" not in session: return redirect(url_for("home"))
    return render_template("dashboard.html")

@app.route("/report")
def report():
    if "uploaded_file" not in session: return redirect(url_for("home"))
    return render_template("report.html")

@app.route("/forecast")
def forecast():
    if "uploaded_file" not in session: return redirect(url_for("home"))
    return render_template("forecast.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files: return "No file part"
    file = request.files["file"]
    if file.filename == "": return "No selected file"
    if file:
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)
        import time
        session["upload_start"] = time.time()
        session["uploaded_file"] = filepath
        df = pd.read_csv(filepath)
        mapping = detect_columns(df)
        session["column_mapping"] = mapping

        # Save to upload history
        from datetime import datetime
        history = session.get("upload_history", [])
        import time
        end_time = time.time()
        start_time = session.pop("upload_start", end_time)
        duration = round(end_time - start_time, 1)

        history.insert(0, {
            "filename":    file.filename,
            "rows":        len(df),
            "columns":     len(df.columns),
            "missing":     int(df.isnull().sum().sum()),
            "duplicates":  int(df.duplicated().sum()),
            "uploaded_at": datetime.now().strftime("%d %b %Y, %I:%M %p"),
            "duration_s":  duration,
            "detected": {
                "date":     mapping.get("date")     or "Not detected",
                "price":    mapping.get("price")    or "Not detected",
                "quantity": mapping.get("quantity") or "Not detected",
                "category": mapping.get("category") or "Not detected",
            }
        })
        session["upload_history"] = history[:10]  # keep last 10
        session.modified = True
        return redirect(url_for("dashboard"))


@app.route("/api/column-mapping")
def column_mapping_debug():
    if "uploaded_file" not in session: return jsonify({"error": "No dataset uploaded"}), 400
    df = pd.read_csv(session["uploaded_file"])
    return jsonify({
        "detected": session.get("column_mapping", {}),
        "all_columns": df.columns.tolist(),
        "dtypes": {col: str(df[col].dtype) for col in df.columns}
    })


@app.route("/api/kpis")
def get_kpis():
    if "uploaded_file" not in session: return jsonify({"error": "No dataset uploaded"}), 400
    try:
        mapping = session.get("column_mapping", {})
        df = calc_revenue(pd.read_csv(session["uploaded_file"]), mapping)
        qty_col = mapping.get("quantity"); cat_col = mapping.get("category")
        return jsonify({
            "transactions": len(df),
            "quantity":     int(df[qty_col].sum()) if qty_col else 0,
            "revenue":      round(float(df["revenue"].sum()), 2),
            "categories":   int(df[cat_col].nunique()) if cat_col else 0
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/categories")
def get_categories():
    if "uploaded_file" not in session: return jsonify({"error": "No dataset uploaded"}), 400
    df = pd.read_csv(session["uploaded_file"])
    cat_col = session.get("column_mapping", {}).get("category")
    if not cat_col: return jsonify({"categories": []})
    return jsonify({"categories": sorted(df[cat_col].dropna().unique().tolist())})


@app.route("/api/dashboard")
def filtered_dashboard():
    if "uploaded_file" not in session: return jsonify({"error": "No dataset uploaded"}), 400
    mapping = session.get("column_mapping", {})
    df = pd.read_csv(session["uploaded_file"])
    date_col = mapping.get("date"); cat_col = mapping.get("category"); qty_col = mapping.get("quantity")
    if date_col: df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = calc_revenue(df, mapping)
    start = request.args.get("start"); end = request.args.get("end"); cat = request.args.get("category")
    if start and end and date_col: df = df[(df[date_col] >= start) & (df[date_col] <= end)]
    if cat and cat != "all" and cat_col: df = df[df[cat_col] == cat]
    return jsonify({
        "transactions": int(len(df)),
        "quantity":     int(df[qty_col].sum()) if qty_col else 0,
        "revenue":      round(float(df["revenue"].sum()), 2),
        "categories":   int(df[cat_col].nunique()) if cat_col else 0
    })


@app.route("/api/chart/monthly")
def monthly_chart():
    if "uploaded_file" not in session: return jsonify({"error": "No dataset uploaded"}), 400
    mapping = session.get("column_mapping", {})
    df = pd.read_csv(session["uploaded_file"])
    date_col = mapping.get("date"); cat_col = mapping.get("category")
    if not date_col: return jsonify({"labels": [], "values": []})
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = calc_revenue(df, mapping)
    start = request.args.get("start"); end = request.args.get("end"); cat = request.args.get("category")
    if start and end: df = df[(df[date_col] >= start) & (df[date_col] <= end)]
    if cat and cat != "all" and cat_col: df = df[df[cat_col] == cat]
    df["month"] = df[date_col].dt.to_period("M")
    grouped = df.groupby("month")["revenue"].sum()
    return jsonify({"labels": [str(m) for m in grouped.index], "values": grouped.values.tolist()})


@app.route("/api/chart/category")
def category_chart():
    if "uploaded_file" not in session: return jsonify({"error": "No dataset uploaded"}), 400
    mapping = session.get("column_mapping", {})
    df = pd.read_csv(session["uploaded_file"])
    date_col = mapping.get("date"); cat_col = mapping.get("category")
    if not cat_col: return jsonify({"labels": [], "values": []})
    if date_col: df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = calc_revenue(df, mapping)
    start = request.args.get("start"); end = request.args.get("end"); cat = request.args.get("category")
    if start and end and date_col: df = df[(df[date_col] >= start) & (df[date_col] <= end)]
    if cat and cat != "all": df = df[df[cat_col] == cat]
    grouped = df.groupby(cat_col)["revenue"].sum().sort_values(ascending=False)
    return jsonify({"labels": grouped.index.tolist(), "values": grouped.values.tolist()})


@app.route("/api/insights")
def insights():
    if "uploaded_file" not in session: return jsonify({"error": "No dataset uploaded"}), 400
    mapping = session.get("column_mapping", {})
    df = pd.read_csv(session["uploaded_file"])
    date_col = mapping.get("date"); cat_col = mapping.get("category")
    if date_col: df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = calc_revenue(df, mapping)
    start = request.args.get("start"); end = request.args.get("end"); cat = request.args.get("category")
    if start and end and date_col: df = df[(df[date_col] >= start) & (df[date_col] <= end)]
    if cat and cat != "all" and cat_col: df = df[df[cat_col] == cat]
    top5 = bottom5 = []
    if cat_col:
        grouped = df.groupby(cat_col)["revenue"].sum().sort_values(ascending=False)
        top5 = grouped.head(5); bottom5 = grouped.tail(5)
    peak_month = "N/A"
    if date_col:
        df["month"] = df[date_col].dt.to_period("M")
        mg = df.groupby("month")["revenue"].sum()
        if not mg.empty: peak_month = str(mg.idxmax())
    return jsonify({
        "top5":       top5.index.tolist() if cat_col else [],
        "bottom5":    bottom5.index.tolist() if cat_col else [],
        "peak_month": peak_month
    })


@app.route("/api/raw-data")
def raw_data():
    if "uploaded_file" not in session: return jsonify({"error": "No dataset uploaded"}), 400
    df = pd.read_csv(session["uploaded_file"])
    total = len(df); df = df.head(1000)
    df = df.where(pd.notnull(df), "")
    return jsonify({"columns": df.columns.tolist(), "rows": df.values.tolist(), "total": total, "showing": len(df)})


@app.route("/api/data-health")
def data_health():
    if "uploaded_file" not in session: return jsonify({"error": "No dataset uploaded"}), 400
    df = pd.read_csv(session["uploaded_file"])
    return jsonify({"rows": len(df), "columns": len(df.columns),
                    "missing_values": int(df.isnull().sum().sum()), "duplicates": int(df.duplicated().sum())})


@app.route("/api/forecast")
def forecast_api():
    if "uploaded_file" not in session: return jsonify({"error": "No dataset uploaded"}), 400
    mapping  = session.get("column_mapping", {})
    df       = pd.read_csv(session["uploaded_file"])
    date_col = mapping.get("date")

    if not date_col:
        return jsonify({"error": "No date column detected. Check /api/column-mapping for details."}), 400

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = calc_revenue(df, mapping)
    df["month"] = df[date_col].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby("month")["revenue"].sum().reset_index()

    if len(monthly) < 3:
        return jsonify({"error": f"Only {len(monthly)} month(s) of data found. Minimum 3 required."}), 400

    # Remove months with zero revenue (breaks Prophet MAPE calculation)
    monthly = monthly[monthly["revenue"] > 0]

    if len(monthly) < 3:
        return jsonify({"error": f"Only {len(monthly)} non-zero month(s) found. Minimum 3 required."}), 400

    prophet_df = monthly.rename(columns={"month": "ds", "revenue": "y"})
    periods    = int(request.args.get("months", 3))

    test_size = min(6, max(1, len(prophet_df) // 4))
    train = prophet_df.iloc[:-test_size]
    test  = prophet_df.iloc[-test_size:]
    if len(train) < 2: train = prophet_df; test = prophet_df.iloc[-1:]

    model = Prophet()
    model.fit(train)
    ft = model.make_future_dataframe(periods=test_size, freq="ME")
    fp = model.predict(ft)
    mape = np.mean(np.abs((test["y"].values - fp.tail(test_size)["yhat"].values) / np.where(test["y"].values == 0, 1, test["y"].values))) * 100
    confidence_level = "High" if mape < 10 else "Moderate" if mape < 20 else "Low"

    mf = Prophet(); mf.fit(prophet_df)
    fc = mf.predict(mf.make_future_dataframe(periods=periods, freq="ME"))
    fr = fc.tail(periods)

    hv = prophet_df["y"].tolist(); hl = prophet_df["ds"].dt.strftime("%Y-%m").tolist()
    fv = fr["yhat"].tolist(); fl = fr["ds"].dt.strftime("%Y-%m").tolist()
    fu = fr["yhat_upper"].tolist(); flo = fr["yhat_lower"].tolist()

    # Growth: compare avg forecast vs avg of last N actual months (where N = periods)
    non_zero_hv = [v for v in hv if v > 0]
    recent_actual = non_zero_hv[-periods:] if len(non_zero_hv) >= periods else non_zero_hv
    la  = float(np.mean(recent_actual)) if recent_actual else hv[-1]
    ff  = float(np.mean(fv))  # avg of all forecast months
    gp  = ((ff - la) / la) * 100 if la != 0 else 0
    tl  = "Increasing" if gp > 3 else "Declining" if gp < -3 else "Stable"

    # MAPE and demand score also reflect forecast period
    gs  = max(min(50 + gp, 100), 0)
    rs  = max(0, 100 - mape * 2)
    bw  = np.mean(np.array(fu) - np.array(flo))
    ss  = max(0, 100 - (bw / la * 100 if la != 0 else 100))
    ds  = round(float(0.4*gs + 0.4*rs + 0.2*ss), 2)
    dst = "Strong" if ds >= 80 else "Stable" if ds >= 60 else "Moderate Risk" if ds >= 40 else "Critical Risk"

    return jsonify({
        "history_labels": hl, "history_values": hv,
        "forecast_labels": fl, "forecast_values": fv,
        "forecast_lower": flo, "forecast_upper": fu,
        "growth_percent": round(float(gp), 2), "trend_label": tl,
        "mape": round(float(mape), 2), "confidence_level": confidence_level,
        "latest_month": hl[-1],
        "ai_explanation": (f"The {periods}-month forecast indicates a {tl.lower()} revenue trend "
                           f"with projected average growth of {round(gp,2)}% compared to recent actuals. "
                           f"Model reliability is {confidence_level.lower()} (MAPE: {round(mape,2)}%). "
                           f"Demand Intelligence Score: {ds}/100 — {dst}."),
        "demand_score": ds, "demand_status": dst,
    })


@app.route("/api/demand-intelligence")
def demand_intelligence_api():
    if "uploaded_file" not in session: return jsonify({"error": "No dataset uploaded"}), 400
    try:
        mapping  = session.get("column_mapping", {})
        df       = pd.read_csv(session["uploaded_file"])
        date_col = mapping.get("date"); cat_col = mapping.get("category")

        if not date_col:
            return jsonify({"error": "Date column not detected"}), 400

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        df = calc_revenue(df, mapping)
        df = df[df["revenue"] > 0]
        df["_month"] = df[date_col].dt.to_period("M").dt.to_timestamp()
        monthly = df.groupby("_month")["revenue"].sum()

        if len(monthly) < 3:
            return jsonify({"error": f"Only {len(monthly)} month(s) found. Minimum 3 required."}), 400

        months = int(request.args.get("months", 3))
        pf = monthly.reset_index().rename(columns={"_month": "ds", "revenue": "y"})
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.fit(pf)
        future = m.make_future_dataframe(periods=months, freq="MS")
        fc = m.predict(future)
        fr = fc.tail(months)

        fv = fr["yhat"].tolist(); flo = fr["yhat_lower"].tolist(); fu = fr["yhat_upper"].tolist()
        av = pf["y"].values; pv = fc.head(len(pf))["yhat"].values
        mape = round(float(np.mean(np.abs((av - pv) / np.where(av == 0, 1, av))) * 100), 2)
        ra = float(monthly.iloc[-3:].mean()) if len(monthly) >= 3 else float(monthly.mean())
        fa = float(np.mean(fv))
        gp = round(((fa - ra) / ra) * 100, 2) if ra != 0 else 0.0
        di = round(max(0, min(100, 100 - mape * 2 + (10 if gp > 0 else 0))), 2)
        cat_data = df.groupby(cat_col)["revenue"].sum().to_dict() if cat_col else None

        report = engine.analyze(
            history_values=monthly.tolist(), history_labels=[str(d)[:7] for d in monthly.index],
            forecast_values=fv, forecast_lower=flo, forecast_upper=fu,
            mape=mape, demand_score=di, growth_percent=gp, category_revenues=cat_data
        )
        return safe_jsonify(report)
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/api/revenue-debug")
def revenue_debug():
    if "uploaded_file" not in session:
        return jsonify({"error": "No dataset uploaded"}), 400
    mapping = session.get("column_mapping", {})
    df = pd.read_csv(session["uploaded_file"])
    df = calc_revenue(df, mapping)
    date_col = mapping.get("date")
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["month"] = df[date_col].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby("month")["revenue"].sum().reset_index()
    return jsonify({
        "mapping": mapping,
        "revenue_sample": df["revenue"].head(5).tolist(),
        "revenue_mean": float(df["revenue"].mean()),
        "revenue_sum": float(df["revenue"].sum()),
        "monthly_sample": monthly.head(5).to_dict(orient="records"),
        "monthly_count": len(monthly)
    })



@app.route("/api/dataset-info")
def dataset_info():
    if "uploaded_file" not in session:
        return jsonify({"loaded": False})
    try:
        filepath = session["uploaded_file"]
        mapping  = session.get("column_mapping", {})
        df       = pd.read_csv(filepath)
        return jsonify({
            "loaded":    True,
            "filename":  os.path.basename(filepath),
            "rows":      len(df),
            "columns":   len(df.columns),
            "missing":   int(df.isnull().sum().sum()),
            "duplicates":int(df.duplicated().sum()),
            "detected": {
                "date":     mapping.get("date")     or "Not detected",
                "price":    mapping.get("price")    or "Not detected",
                "quantity": mapping.get("quantity") or "Not detected",
                "category": mapping.get("category") or "Not detected",
            }
        })
    except Exception as e:
        return jsonify({"loaded": False, "error": str(e)})


@app.route("/api/upload-history")
def upload_history():
    history = session.get("upload_history", [])
    return jsonify({"history": history})



@app.route("/history")
def history_page():
    return render_template("history.html")


@app.route("/demo")
def load_demo():
    demo_path = os.path.join(os.path.dirname(__file__), "data", "demo_sales.csv")
    if not os.path.exists(demo_path):
        return redirect(url_for("home"))
    session["uploaded_file"] = demo_path
    df = pd.read_csv(demo_path)
    mapping = detect_columns(df)
    session["column_mapping"] = mapping

    from datetime import datetime
    import time
    history = session.get("upload_history", [])
    history.insert(0, {
        "filename":    "demo_sales.csv (Demo)",
        "rows":        len(df),
        "columns":     len(df.columns),
        "missing":     0,
        "duplicates":  0,
        "uploaded_at": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "duration_s":  0.1,
        "detected": {
            "date":     mapping.get("date")     or "Not detected",
            "price":    mapping.get("price")    or "Not detected",
            "quantity": mapping.get("quantity") or "Not detected",
            "category": mapping.get("category") or "Not detected",
        }
    })
    session["upload_history"] = history[:10]
    session.modified = True
    return redirect(url_for("dashboard"))


@app.route("/api/export-pdf")
def export_pdf():
    if "uploaded_file" not in session:
        return jsonify({"error": "No dataset uploaded"}), 400
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
        import io
        from datetime import datetime

        mapping  = session.get("column_mapping", {})
        df       = pd.read_csv(session["uploaded_file"])
        filename = os.path.basename(session["uploaded_file"])
        df       = calc_revenue(df, mapping)
        date_col = mapping.get("date")

        # Get forecast data
        fc_data = None
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df_fc = df.dropna(subset=[date_col])
            df_fc["month"] = df_fc[date_col].dt.to_period("M").dt.to_timestamp()
            monthly = df_fc.groupby("month")["revenue"].sum().reset_index()
            monthly = monthly[monthly["revenue"] > 0]
            if len(monthly) >= 3:
                from prophet import Prophet
                pf = monthly.rename(columns={"month":"ds","revenue":"y"})
                m = Prophet(); m.fit(pf)
                fc = m.predict(m.make_future_dataframe(periods=3, freq="ME"))
                fr = fc.tail(3)
                fc_data = {
                    "labels": fr["ds"].dt.strftime("%b %Y").tolist(),
                    "values": [round(v,2) for v in fr["yhat"].tolist()],
                    "total_revenue": round(float(df["revenue"].sum()), 2),
                    "monthly_count": len(monthly)
                }

        # Build PDF
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
            leftMargin=2*cm, rightMargin=2*cm,
            topMargin=2*cm, bottomMargin=2*cm)

        styles = getSampleStyleSheet()
        accent = colors.HexColor("#3b82f6")
        dark   = colors.HexColor("#0d1117")
        gray   = colors.HexColor("#64748b")
        light  = colors.HexColor("#f1f5f9")

        def style(name, **kw):
            s = ParagraphStyle(name, parent=styles["Normal"], **kw)
            return s

        title_style   = style("T", fontSize=22, textColor=dark, fontName="Helvetica-Bold", spaceAfter=4)
        sub_style     = style("S", fontSize=11, textColor=gray, spaceAfter=2)
        heading_style = style("H", fontSize=13, textColor=accent, fontName="Helvetica-Bold", spaceBefore=16, spaceAfter=8)
        body_style    = style("B", fontSize=10, textColor=colors.HexColor("#334155"), leading=16)
        small_style   = style("Sm", fontSize=9, textColor=gray)

        story = []

        # Header
        story.append(Paragraph("RetailIQ", style("Logo", fontSize=10, textColor=accent, fontName="Helvetica-Bold")))
        story.append(Spacer(1, 4))
        story.append(Paragraph("Retail Demand Estimation Platform", title_style))
        story.append(Spacer(1, 4))
        story.append(Paragraph("Forecast Report", style("FR", fontSize=13, textColor=gray, fontName="Helvetica-Bold")))
        story.append(Paragraph(f"Generated on {datetime.now().strftime('%d %B %Y')} at {datetime.now().strftime('%I:%M %p')}", sub_style))
        story.append(HRFlowable(width="100%", thickness=1, color=accent, spaceAfter=16))

        # Dataset Info
        story.append(Paragraph("Dataset Overview", heading_style))
        ds_data = [
            ["Filename", filename],
            ["Total Records", f"{len(df):,}"],
            ["Total Columns", str(len(df.columns))],
            ["Missing Values", str(int(df.isnull().sum().sum()))],
            ["Duplicate Rows", str(int(df.duplicated().sum()))],
            ["Date Column",     mapping.get("date") or "—"],
            ["Revenue Column",  mapping.get("price") or "—"],
            ["Quantity Column", mapping.get("quantity") or "—"],
            ["Category Column", mapping.get("category") or "—"],
        ]
        ds_table = Table(ds_data, colWidths=[5*cm, 11*cm])
        ds_table.setStyle(TableStyle([
            ('FONTNAME',  (0,0), (0,-1), 'Helvetica-Bold'),
            ('FONTSIZE',  (0,0), (-1,-1), 10),
            ('TEXTCOLOR', (0,0), (0,-1), colors.HexColor("#334155")),
            ('TEXTCOLOR', (1,0), (1,-1), colors.HexColor("#0f172a")),
            ('BACKGROUND',(0,0), (-1,0), light),
            ('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.white, light]),
            ('GRID',      (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
            ('PADDING',   (0,0), (-1,-1), 6),
        ]))
        story.append(ds_table)

        # Revenue Summary
        story.append(Paragraph("Revenue Summary", heading_style))
        total_rev = round(float(df["revenue"].sum()), 2)
        avg_rev   = round(float(df["revenue"].mean()), 2)
        story.append(Paragraph(f"Total Revenue: Rs {total_rev:,.2f}", body_style))
        story.append(Paragraph(f"Average Revenue per Transaction: Rs {avg_rev:,.2f}", body_style))

        # Forecast
        if fc_data:
            story.append(Paragraph("3-Month Revenue Forecast", heading_style))
            story.append(Paragraph(
                f"The following forecast was generated using Facebook Prophet on {fc_data['monthly_count']} months of historical data.",
                body_style))
            story.append(Spacer(1, 8))
            fc_table_data = [["Month", "Forecasted Revenue"]] + [
                [l, f"Rs {v:,.2f}"] for l, v in zip(fc_data["labels"], fc_data["values"])
            ]
            fc_table = Table(fc_table_data, colWidths=[8*cm, 8*cm])
            fc_table.setStyle(TableStyle([
                ('BACKGROUND',  (0,0), (-1,0), accent),
                ('TEXTCOLOR',   (0,0), (-1,0), colors.white),
                ('FONTNAME',    (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE',    (0,0), (-1,-1), 10),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, light]),
                ('GRID',        (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
                ('PADDING',     (0,0), (-1,-1), 8),
                ('ALIGN',       (1,0), (1,-1), 'RIGHT'),
            ]))
            story.append(fc_table)

        # Footer
        story.append(Spacer(1, 20))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0")))
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"Generated by RetailIQ · {datetime.now().strftime('%d %B %Y')}", small_style))

        doc.build(story)
        buf.seek(0)

        from flask import send_file
        return send_file(
            buf, as_attachment=True,
            download_name=f"RetailIQ_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mimetype="application/pdf"
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)