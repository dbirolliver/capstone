from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from bson import ObjectId
import os
import google.generativeai as genai
from datetime import datetime
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime, timezone 
from datetime import datetime, timedelta

# ---------- Flask + CORS ----------

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

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
    branch = data.get("branch")          # may be None / empty
    delta = int(data.get("delta", 0))

    if delta == 0:
        return jsonify({"error": "non-zero delta required"}), 400

    # Build query: by name, and branch only if provided
    query = {"name": name}
    if branch:
        query["branch"] = branch

    inv = inventory_collection.find_one(query)
    if not inv:
        return jsonify({"error": "item not found"}), 404

    new_qty = max(0, int(inv.get("quantity", 0)) + delta)
    inventory_collection.update_one(
        {"_id": inv["_id"]},
        {"$set": {"quantity": new_qty}}
    )

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

    # insert batch
    result = batches_collection.insert_one(batch_doc)
    batch_doc["_id"] = str(result.inserted_id)

    # also update inventory totals for this item + branch
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
    
    from datetime import datetime, timezone

@app.route("/api/ai/dashboard", methods=["GET"])
def get_ai_dashboard():
    doc = ai_dashboard_collection.find_one({"_id": "summary"}) or {}
    if doc:
        doc["updated_at"] = doc.get("updated_at")
    return jsonify({
        "summary_text": doc.get("summary_text", ""),
        "risk_text": doc.get("risk_text", ""),
        "updated_at": doc.get("updated_at")  # ISO string or None
    }), 200


@app.route("/api/ai/dashboard/refresh", methods=["POST"])
def refresh_ai_dashboard():
    # 1) Gather simple stats from inventory
    items = list(inventory_collection.find({}, {"_id": 0}))
    total_items = len(items)
    total_cost = float(sum((i.get("price", 0) or 0) * (i.get("quantity", 0) or 0) for i in items))
    low_stock = [i for i in items if i.get("quantity", 0) <= i.get("reorder_level", 0)]

    # 2) Build a compact text summary for Gemini
    inventory_brief = {
        "total_items": total_items,
        "total_cost": total_cost,
        "low_stock_count": len(low_stock),
        "low_stock_names": [i.get("name") for i in low_stock][:10],
    }

    # Simple rule-based AI-style texts (no external API)
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



# ---------- DEMO GEMINI CHAT ENDPOINT ----------

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

# ---------- KPI ENDPOINTS (Dashboard cards) ----------

@app.route("/api/low-stock-count", methods=["GET"])
def api_low_stock_count():
    """
    Count items where quantity <= reorder_level.
    """
    try:
        count = inventory_collection.count_documents({
            "$expr": {"$lte": ["$quantity", "$reorder_level"]}
        })
        return jsonify({"count": int(count)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/total-inventory", methods=["GET"])
def api_total_inventory():
    """
    Return the total inventory value = sum(price * quantity) over all items.
    """
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
    """
    Count all branches.
    """
    try:
        count = branches_collection.count_documents({})
        return jsonify({"count": int(count)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def expiry_within_days(expiry_value, days):
    """Return True if expiry_value is within <days> days from now."""
    if not expiry_value:
        return False

    # If expiry is already a datetime
    if isinstance(expiry_value, datetime):
        expiry_dt = expiry_value
    else:
        # If you stored it as ISO string like "2025-12-31"
        try:
            expiry_dt = datetime.fromisoformat(str(expiry_value))
        except ValueError:
            return False

    return expiry_dt <= datetime.utcnow() + timedelta(days=days)



suppliers_collection = db["suppliers"]

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
    suppliers_collection.insert_one(doc)  # auto-creates collection if missing [web:100][web:111]
    return jsonify({"message": "Supplier added"}), 201



@app.post("/api/alerts/<alert_id>/acknowledge")
def acknowledge_alert(alert_id):
    # TODO: replace with your real auth/current user
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

    # 1) Build alerts from inventory
    items = list(inventory_collection.find({}))  # adjust to your collection name

    for item in items:
        name = item.get("name")
        branch = item.get("branch", "Main branch")
        qty = item.get("quantity", 0)
        reorder = item.get("reorder_level", 0)
        expiry = item.get("expiry_date")  # whatever field you use

        # Low stock
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

        # Expiring soon
        if expiry and expiry_within_days(expiry, 30):
            alerts.append({
                "id": f"expiry-{branch}-{name}",
                "type": "expiry_risk",
                "severity": "medium",
                "title": f"Expiry soon: {name} – {branch}",
                "description": "Batch expiring within 30 days.",
                "branch": branch,
            })

    # Branch-level low-stock summary
    for branch, count in low_by_branch.items():
        if count >= 3:  # threshold
            alerts.append({
                "id": f"branch-low-{branch}",
                "type": "branch_low_stock",
                "severity": "high",
                "title": f"Branch alert: {branch} has {count} low‑stock items",
                "description": "Review this branch inventory and create replenishment orders.",
                "branch": branch,
            })

    # 2) Filter out alerts already acknowledged by this user
    acked_ids = {
        doc["alert_id"]
        for doc in db.alert_acknowledgements.find(
            {"user_id": user_id},
            {"alert_id": 1, "_id": 0}
        )
    }

    visible_alerts = [a for a in alerts if a["id"] not in acked_ids]
    return jsonify(visible_alerts)


@app.get("/api/replenishment/recommendations")
def get_replenishment_recommendations():
    items = list(inventory_collection.find({}))  # your inventory collection

    recommendations = []

    for item in items:
        name = item.get("name")
        branch = item.get("branch", "Main")
        qty = item.get("quantity", 0)
        reorder = item.get("reorder_level", 0)

        # optional extra fields
        avg_daily_usage = item.get("avg_daily_usage", 1)      # fallback 1
        lead_time_days = item.get("lead_time_days", 7)        # fallback 7
        safety_stock = item.get("safety_stock", reorder or 0)

        # basic reorder point formula
        reorder_point = avg_daily_usage * lead_time_days + safety_stock

        # trigger recommendation if at/below either configured reorder or ROP
        trigger_level = max(reorder, reorder_point)
        if qty <= trigger_level:
            # target stock for next period
            target_stock = avg_daily_usage * (lead_time_days + 7) + safety_stock
            suggested_qty = max(int(target_stock - qty), 0)

            if suggested_qty > 0:
                recommendations.append({
                    "name": name,
                    "branch": branch,
                    "current_quantity": qty,
                    "reorder_level": reorder,
                    "reorder_point": reorder_point,
                    "suggested_order_qty": suggested_qty
                })

    return jsonify(recommendations)


# ---------- Run server ----------

if __name__ == "__main__":
    app.run(debug=True)
