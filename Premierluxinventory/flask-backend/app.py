from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import os
import google.generativeai as genai
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime, timezone, timedelta
from flask_socketio import SocketIO
import threading
import time

# ---------- Flask + CORS ----------

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
socketio = SocketIO(app, cors_allowed_origins="*")

# ---------- Gemini setup ----------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
print("GEMINI_API_KEY in Flask:", GEMINI_API_KEY)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("Gemini API key not set! Some AI routes will fail.")

# ---------- MongoDB setup ----------

MONGO_URI = "mongodb+srv://dbirolliverhernandez_db_user:yqHWCWJwNxKofjHs@cluster0.bgmzgav.mongodb.net/?appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client["premierlux"]

ai_dashboard_collection = db["ai_dashboard"]
inventory_collection = db["inventory"]
branches_collection = db["branches"]
batches_collection = db["batches"]
consumption_collection = db["consumption"]
suppliers_collection = db["suppliers"]

# ---------- Root ----------

@app.route("/")
def home():
    return "Flask backend is running!"

# ---------- INVENTORY CRUD + QUERIES ----------

@app.route("/api/inventory", methods=["GET"])
def get_inventory():
    items = list(inventory_collection.find({}, {"_id": 0}))
    return jsonify(items)

@app.route("/api/inventory", methods=["POST"])
def add_inventory():
    data = request.json or {}
    inventory_collection.insert_one(data)
    return jsonify({"message": "Item added"}), 201

@app.route("/api/inventory/<string:item_name>", methods=["DELETE"])
def delete_inventory(item_name):
    inventory_collection.delete_one({"name": item_name})
    return jsonify({"message": "Item deleted"})

@app.route("/api/inventory/branch/<string:branch_name>", methods=["GET"])
def get_inventory_by_branch(branch_name):
    items = list(inventory_collection.find({"branch": branch_name}, {"_id": 0}))
    return jsonify(items)

@app.route("/api/inventory/lowstock", methods=["GET"])
def get_low_stock():
    items = list(
        inventory_collection.find(
            {"$expr": {"$lte": ["$quantity", "$reorder_level"]}},
            {"_id": 0},
        )
    )
    return jsonify(items)

@app.route("/api/inventory/<name>/adjust", methods=["POST"])
def adjust_inventory(name):
    data = request.json or {}
    branch = data.get("branch")
    delta = int(data.get("delta", 0))

    if delta == 0:
        return jsonify({"error": "non-zero delta required"}), 400

    query = {"name": name}
    if branch:
        query["branch"] = branch

    inv = inventory_collection.find_one(query)
    if not inv:
        return jsonify({"error": "item not found"}), 404

    new_qty = max(0, int(inv.get("quantity", 0)) + delta)
    inventory_collection.update_one(
        {"_id": inv["_id"]},
        {"$set": {"quantity": new_qty}},
    )

    # log this change as a movement event for analytics
    consumption_collection.insert_one({
        "name": name,
        "date": datetime.utcnow(),
        "quantity_used": abs(delta),
        "direction": "out" if delta < 0 else "in",
        "branch": branch or inv.get("branch"),
    })

    return jsonify({"status": "ok", "quantity": new_qty})

# ---------- BATCHES ----------

@app.route("/api/batches", methods=["POST"])
def create_batch():
    data = request.get_json(force=True)

    if not data or "item_name" not in data or "branch" not in data:
        return jsonify({"error": "item_name and branch are required"}), 400

    batch_doc = {
        "item_name": data.get("item_name"),
        "sku": data.get("sku"),
        "branch": data.get("branch"),
        "current_stock": data.get("current_stock", 0),
        "monthly_usage": data.get("monthly_usage", 0),
        "price": data.get("price", 0),
        "reorder_level": data.get("reorder_level", 0),
        "batch_number": data.get("batch_number") or None,
        "lot_number": data.get("lot_number") or None,
        "mfg_date": data.get("mfg_date") or None,
        "exp_date": data.get("exp_date") or None,
        "supplier_batch": data.get("supplier_batch") or None,
        "qr_code_id": data.get("qr_code_id") or None,
    }

    result = batches_collection.insert_one(batch_doc)
    batch_doc["_id"] = str(result.inserted_id)

    # update inventory totals for this item + branch
    inventory_collection.update_one(
        {"name": batch_doc["item_name"], "branch": batch_doc["branch"]},
        {
            "$setOnInsert": {
                "reorder_level": batch_doc["reorder_level"],
                "price": batch_doc["price"],
            },
            "$inc": {"quantity": batch_doc["current_stock"]},
        },
        upsert=True,
    )

    return jsonify({"status": "ok", "batch": batch_doc}), 201

# ---------- AI: SIMPLE FORECASTING ----------

@app.route("/api/forecast/<item_name>", methods=["GET"])
def forecast_item(item_name):
    try:
        history = list(
            consumption_collection.find(
                {"name": item_name},
                {"_id": 0, "date": 1, "quantity_used": 1},
            )
        )

        if not history:
            return jsonify(
                {
                    "item": item_name,
                    "message": "No consumption history found for this item.",
                    "forecast": [],
                }
            ), 200

        history_sorted = sorted(
            history,
            key=lambda x: datetime.strptime(x["date"], "%Y-%m-%d"),
        )
        y = np.array([h["quantity_used"] for h in history_sorted], dtype=float)

        if len(y) < 3:
            avg = float(np.mean(y))
            forecast_values = [avg] * 7
        else:
            model = ExponentialSmoothing(y, trend=None, seasonal=None)
            model_fit = model.fit(optimized=True)
            forecast_values = model_fit.forecast(7).tolist()

        return jsonify(
            {
                "item": item_name,
                "history_points": len(y),
                "forecast_horizon_days": 7,
                "daily_forecast": forecast_values,
            }
        ), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------- AI DASHBOARD ----------

@app.route("/api/ai/dashboard", methods=["GET"])
def get_ai_dashboard():
    doc = ai_dashboard_collection.find_one({"_id": "summary"}) or {}
    if doc:
        doc["updated_at"] = doc.get("updated_at")
    return jsonify({
        "summary_text": doc.get("summary_text", ""),
        "risk_text": doc.get("risk_text", ""),
        "updated_at": doc.get("updated_at"),
    }), 200

@app.route("/api/ai/dashboard/refresh", methods=["POST"])
def refresh_ai_dashboard():
    items = list(inventory_collection.find({}, {"_id": 0}))
    total_items = len(items)
    total_cost = float(sum((i.get("price", 0) or 0) * (i.get("quantity", 0) or 0) for i in items))
    low_stock = [i for i in items if i.get("quantity", 0) <= i.get("reorder_level", 0)]

    inventory_brief = {
        "total_items": total_items,
        "total_cost": total_cost,
        "low_stock_count": len(low_stock),
        "low_stock_names": [i.get("name") for i in low_stock][:10],
    }

    if total_items == 0:
        summary_text = (
            "No items in inventory yet. Add your first batch to start tracking stock and cost."
        )
        risk_text = (
            "Main risk is missing data. Add core consumables and instruments first to build history."
        )
    else:
        low_names_str = ", ".join(inventory_brief["low_stock_names"]) or "none"
        summary_text = (
            f"Inventory has {total_items} items with an estimated total cost of ₱{total_cost:.2f}. "
            f"{len(low_stock)} items are at or below reorder level."
        )
        risk_text = (
            f"Focus on low stock items ({low_names_str}). "
            "Plan purchase orders in the next 7 days to avoid stockouts and spread costs."
        )

    now_iso = datetime.now(timezone.utc).isoformat()

    ai_dashboard_collection.update_one(
        {"_id": "summary"},
        {
            "$set": {
                "summary_text": summary_text,
                "risk_text": risk_text,
                "updated_at": now_iso,
                "total_items": total_items,
                "total_cost": total_cost,
                "low_stock_count": len(low_stock),
            }
        },
        upsert=True,
    )

    return jsonify({
        "summary_text": summary_text,
        "risk_text": risk_text,
        "updated_at": now_iso,
        "total_items": total_items,
        "total_cost": total_cost,
        "low_stock_count": len(low_stock),
    }), 200

# ---------- CHAT ----------

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "message is required"}), 400

    if not GEMINI_API_KEY:
        return jsonify(
            {
                "type": "info",
                "text": "Gemini API key is not configured. Ask your admin to set GEMINI_API_KEY.",
            }
        ), 200

    inventory = list(inventory_collection.find({}, {"_id": 0}).limit(50))
    lower_msg = user_message.lower()

    if "low stock" in lower_msg or "reorder" in lower_msg:
        low_stock_items = list(
            inventory_collection.find(
                {"$expr": {"$lte": ["$quantity", "$reorder_level"]}},
                {"_id": 0},
            )
        )
        return jsonify(
            {
                "type": "low_stock_summary",
                "text": f"Found {len(low_stock_items)} low stock or reorder items.",
                "items": low_stock_items,
            }
        ), 200

    try:
        genai_reply = genai.generate_content(
            model="gemini-1.0-pro", contents=user_message
        )
        answer_text = (
            genai_reply.text if hasattr(genai_reply, "text") else str(genai_reply)
        )
        return jsonify({"type": "llm_answer", "text": answer_text}), 200
    except Exception as e:
        return jsonify(
            {"type": "error", "text": "Gemini call failed.", "details": str(e)}
        ), 500

# ---------- BRANCHES ----------

@app.route("/api/branches", methods=["GET"])
def get_branches():
    branches = list(branches_collection.find({}, {"_id": 0}))
    return jsonify(branches)

@app.route("/api/branches", methods=["POST"])
def add_branch():
    data = request.json or {}
    if not data.get("name"):
        return jsonify({"error": "Branch name is required"}), 400

    doc = {
        "name": data["name"],
        "address": data.get("address", ""),
        "manager": data.get("manager", ""),
    }
    branches_collection.insert_one(doc)
    return jsonify({"message": "Branch added"}), 201

# ---------- KPI ENDPOINTS ----------

@app.route("/api/low-stock-count", methods=["GET"])
def api_low_stock_count():
    try:
        count = inventory_collection.count_documents({
            "$expr": {"$lte": ["$quantity", "$reorder_level"]}
        })
        return jsonify({"count": int(count)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/total-inventory", methods=["GET"])
def api_total_inventory():
    try:
        cursor = inventory_collection.find({}, {"price": 1, "quantity": 1})
        total_value = 0.0
        for doc in cursor:
            price = float(doc.get("price") or 0)
            qty = float(doc.get("quantity") or 0)
            total_value += price * qty
        return jsonify({"value": total_value}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/branches-count", methods=["GET"])
def api_branches_count():
    try:
        count = branches_collection.count_documents({})
        return jsonify({"count": int(count)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------- SUPPLIERS ----------

@app.route("/api/suppliers", methods=["GET"])
def get_suppliers():
    suppliers = list(suppliers_collection.find({}, {"_id": 0}))
    return jsonify(suppliers), 200

@app.route("/api/suppliers", methods=["POST"])
def add_supplier():
    data = request.json or {}
    if not data.get("name"):
        return jsonify({"error": "Supplier name is required"}), 400

    doc = {
        "name": data["name"],
        "contact": data.get("contact", ""),
        "lead_time_days": data.get("lead_time_days", 0),
    }
    suppliers_collection.insert_one(doc)
    return jsonify({"message": "Supplier added"}), 201

# ---------- ALERTS ----------

@app.post("/api/alerts/<alert_id>/acknowledge")
def acknowledge_alert(alert_id):
    user_id = request.headers.get("X-User-Id", "demo-admin")
    user_name = request.headers.get("X-User-Name", "Demo Admin")

    doc = {
        "alert_id": alert_id,
        "user_id": user_id,
        "user_name": user_name,
        "acknowledged_at": datetime.utcnow().isoformat() + "Z",
    }

    db.alert_acknowledgements.insert_one(doc)
    return jsonify({"status": "ok"})

@app.get("/api/alerts")
def get_alerts():
    user_id = request.headers.get("X-User-Id", "demo-admin")

    alerts = []
    low_by_branch = {}

    items = list(inventory_collection.find({}))

    for item in items:
        name = item.get("name")
        branch = item.get("branch", "Main branch")
        qty = item.get("quantity", 0)
        reorder = item.get("reorder_level", 0)
        expiry = item.get("expiry_date")

        if reorder and qty <= reorder:
            alerts.append({
                "id": f"low-stock-{branch}-{name}",
                "type": "low_stock",
                "severity": "high",
                "title": f"Low stock: {name} – {branch}",
                "description": f"Current stock {qty}, below reorder point {reorder}.",
                "branch": branch,
            })
            low_by_branch[branch] = low_by_branch.get(branch, 0) + 1

        if expiry and expiry_within_days(expiry, 30):
            alerts.append({
                "id": f"expiry-{branch}-{name}",
                "type": "expiry_risk",
                "severity": "medium",
                "title": f"Expiry soon: {name} – {branch}",
                "description": "Batch expiring within 30 days.",
                "branch": branch,
            })

    for branch, count in low_by_branch.items():
        if count >= 3:
            alerts.append({
                "id": f"branch-low-{branch}",
                "type": "branch_low_stock",
                "severity": "high",
                "title": f"Branch alert: {branch} has {count} low‑stock items",
                "description": "Review this branch inventory and create replenishment orders.",
                "branch": branch,
            })

    acked_ids = {
        doc["alert_id"]
        for doc in db.alert_acknowledgements.find(
            {"user_id": user_id}, {"alert_id": 1, "_id": 0}
        )
    }

    visible_alerts = [a for a in alerts if a["id"] not in acked_ids]
    return jsonify(visible_alerts)

def expiry_within_days(expiry_value, days):
    if not expiry_value:
        return False
    if isinstance(expiry_value, datetime):
        expiry_dt = expiry_value
    else:
        try:
            expiry_dt = datetime.fromisoformat(str(expiry_value))
        except ValueError:
            return False
    return expiry_dt <= datetime.utcnow() + timedelta(days=days)

# ---------- REPLENISHMENT ----------

@app.get("/api/replenishment/recommendations")
def get_replenishment_recommendations():
    items = list(inventory_collection.find({}))

    recommendations = []

    for item in items:
        name = item.get("name")
        branch = item.get("branch", "Main")
        qty = item.get("quantity", 0)
        reorder = item.get("reorder_level", 0)

        avg_daily_usage = item.get("avg_daily_usage", 1)
        lead_time_days = item.get("lead_time_days", 7)
        safety_stock = item.get("safety_stock", reorder or 0)

        reorder_point = avg_daily_usage * lead_time_days + safety_stock

        trigger_level = max(reorder, reorder_point)
        if qty <= trigger_level:
            target_stock = avg_daily_usage * (lead_time_days + 7) + safety_stock
            suggested_qty = max(int(target_stock - qty), 0)

            if suggested_qty > 0:
                recommendations.append({
                    "name": name,
                    "branch": branch,
                    "current_quantity": qty,
                    "reorder_level": reorder,
                    "reorder_point": reorder_point,
                    "suggested_order_qty": suggested_qty,
                })

    return jsonify(recommendations)

# ---------- ANALYTICS REST ENDPOINTS ----------

@app.get("/analytics/overview")
def analytics_overview():
    new_items = inventory_collection.count_documents({
        "created_at": {"$gte": datetime.now() - timedelta(days=7)}
    })

    batches_7d = batches_collection.count_documents({
        "mfg_date": {"$gte": datetime.now() - timedelta(days=7)}
    })

    total_items = inventory_collection.count_documents({})
    branches = branches_collection.count_documents({})

    return jsonify({
        "new_items": new_items,
        "batches_7d": batches_7d,
        "total_items": total_items,
        "branches": branches,
    })

@app.get("/analytics/movement")
def analytics_movement():
    today = datetime.now()
    start_date = today - timedelta(days=6)

    labels = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    stock_in = [0] * 7
    stock_out = [0] * 7
    low_stock = [0] * 7

    batches = list(batches_collection.find({}))
    for b in batches:
        mfg = b.get("mfg_date")
        if not mfg:
            continue
        if isinstance(mfg, str):
            mfg = datetime.fromisoformat(mfg)
        if mfg < start_date:
            continue
        day = mfg.weekday()
        stock_in[day] += b.get("current_stock", 0)

    usage = list(consumption_collection.find({}))
    for u in usage:
        used_date = u.get("date")
        if not used_date:
            continue
        if isinstance(used_date, str):
            used_date = datetime.fromisoformat(used_date)
        if used_date < start_date:
            continue
        day = used_date.weekday()
        qty = u.get("quantity_used", 0)
        if u.get("direction") == "in":
            stock_in[day] += qty
        else:
            stock_out[day] += qty

    low_stock_count = inventory_collection.count_documents({
        "$expr": {"$lte": ["$quantity", "$reorder_level"]}
    })
    low_stock = [low_stock_count] * 7

    return jsonify({
        "labels": labels,
        "stock_in": stock_in,
        "stock_out": stock_out,
        "low_stock": low_stock,
    })

@app.get("/analytics/category")
def analytics_category():
    return jsonify(list(inventory_collection.aggregate([
        {"$match": {"category": {"$ne": None}}},
        {"$group": {"_id": "$category", "total": {"$sum": "$quantity"}}},
    ])))

@app.get("/analytics/low-stock")
def analytics_low_stock():
    return jsonify(list(inventory_collection.find(
        {"$expr": {"$lte": ["$quantity", "$reorder_level"]}},
        {"_id": 0, "name": 1, "quantity": 1},
    )))

@app.get("/analytics/top-products")
def analytics_top_products():
    return jsonify(list(consumption_collection.aggregate([
        {"$group": {"_id": "$name", "used": {"$sum": "$quantity_used"}}},
        {"$sort": {"used": -1}},
        {"$limit": 5},
    ])))

@app.route("/api/analytics/branch-stock", methods=["GET"])
def analytics_branch_stock():
    pipeline = [
        {"$group": {
            "_id": "$branch",
            "total_qty": {"$sum": "$quantity"},
        }},
        {"$sort": {"_id": 1}},
    ]
    results = list(inventory_collection.aggregate(pipeline))
    labels = [r["_id"] or "Unassigned" for r in results]
    values = [r["total_qty"] for r in results]
    return jsonify({"labels": labels, "values": values}), 200

# ---------- SOCKET ANALYTICS BROADCASTER ----------

def build_analytics_payload():
    # ---------- Overview ----------
    new_items = inventory_collection.count_documents({
        "created_at": {"$gte": datetime.now() - timedelta(days=7)}
    })
    batches_7d = batches_collection.count_documents({
        "mfg_date": {"$gte": datetime.now() - timedelta(days=7)}
    })
    total_items = inventory_collection.count_documents({})
    branches = branches_collection.count_documents({})

    overview = {
        "new_items": new_items,
        "batches_7d": batches_7d,
        "total_items": total_items,
        "branches": branches,
    }

    # ---------- WEEKLY movement (for small bar chart) ----------
    today = datetime.now()
    start_week = today - timedelta(days=6)
    week_labels = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    week_in = [0] * 7
    week_out = [0] * 7

    # stock in from batches (last 7 days)
    batches = list(batches_collection.find({"mfg_date": {"$gte": start_week}}))
    for b in batches:
        d = b.get("mfg_date")
        if isinstance(d, str):
            d = datetime.fromisoformat(d)
        if d < start_week:
            continue
        day = d.weekday()  # 0=Mon..6=Sun
        week_in[day] += b.get("current_stock", b.get("quantity", 0) or 0)

    # stock in/out from consumption (last 7 days)
    usage = list(consumption_collection.find({"date": {"$gte": start_week}}))
    for u in usage:
        d = u.get("date")
        if isinstance(d, str):
            d = datetime.fromisoformat(d)
        if d < start_week:
            continue
        day = d.weekday()
        qty = u.get("quantity_used", 1)
        if u.get("direction") == "in":
            week_in[day] += qty
        else:
            week_out[day] += qty

    weekly_movement = {
        "labels": week_labels,
        "stock_in": week_in,
        "stock_out": week_out,
    }

    # ---------- MONTHLY movement (for big line chart, last 12 months) ----------
    now = datetime.now()
    months = []
    for i in range(11, -1, -1):
        first_of_month = (now.replace(day=1) - timedelta(days=30 * i))
        months.append((first_of_month.year, first_of_month.month))

    month_labels = [
        datetime(y, m, 1).strftime("%b %Y")
        for (y, m) in months
    ]
    month_in = [0] * 12
    month_out = [0] * 12

    oldest_year, oldest_month = months[0]
    since = datetime(oldest_year, oldest_month, 1)

    # use the same movements but aggregated per month
    usage_all = list(consumption_collection.find({"date": {"$gte": since}}))
    for u in usage_all:
        d = u.get("date")
        if isinstance(d, str):
            d = datetime.fromisoformat(d)
        ym = (d.year, d.month)
        if ym not in months:
            continue
        idx = months.index(ym)
        qty = u.get("quantity_used", 1)
        if u.get("direction") == "in":
            month_in[idx] += qty
        else:
            month_out[idx] += qty

    monthly_movement = {
        "labels": month_labels,
        "stock_in": month_in,
        "stock_out": month_out,
    }

    # ---------- Low stock table & top products ----------
    low_stock_rows = list(inventory_collection.find(
        {"$expr": {"$lte": ["$quantity", "$reorder_level"]}},
        {"_id": 0, "name": 1, "quantity": 1},
    ))

    top_products = list(consumption_collection.aggregate([
        {"$group": {"_id": "$name", "used": {"$sum": "$quantity_used"}}},
        {"$sort": {"used": -1}},
        {"$limit": 5},
    ]))

    return {
        "overview": overview,
        "movement": weekly_movement,          # weekly for small chart
        "movement_monthly": monthly_movement, # monthly for big chart
        "low_stock": low_stock_rows,
        "top_products": top_products,
    }


def analytics_broadcaster():
    while True:
        try:
            payload = build_analytics_payload()
            print("SOCKET ANALYTICS PAYLOAD:", payload)
            socketio.emit("analytics_update", payload, namespace="/analytics")
        except Exception as e:
            print("Analytics broadcaster error:", e)
        time.sleep(5)


@socketio.on("connect", namespace="/analytics")
def analytics_connect():
    print("Client connected to analytics")


# ---------- Run server ----------

if __name__ == "__main__":
    t = threading.Thread(target=analytics_broadcaster, daemon=True)
    t.start()
    socketio.run(app, debug=True)
