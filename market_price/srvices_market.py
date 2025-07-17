from flask import Blueprint, jsonify, request
import pandas as pd

bp = Blueprint("market", __name__, url_prefix="/api/market")

DF_P = pd.read_csv("market_price/data/prices.csv", parse_dates=["arrival_date"])
DF_A = pd.read_csv("market_price/data/area.csv")
DF_PR = pd.read_csv("market_price/data/production.csv")

@bp.route("/prices", methods=["GET"])
def get_prices():
    commodity = request.args.get("commodity")
    market = request.args.get("market")
    days = int(request.args.get("days", 30))
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    df = DF_P
    if commodity: df = df[df["commodity_name"] == commodity]
    if market: df = df[df["market"] == market]
    df = df[df["arrival_date"] >= cutoff]
    return jsonify(df.to_dict(orient="records"))

@bp.route("/area", methods=["GET"])
def get_area():
    commodity = request.args.get("commodity")
    district = request.args.get("district")
    df = DF_A
    if commodity: df = df[df["commodity_name"] == commodity]
    if district: df = df[df["district"] == district]
    return jsonify(df.to_dict(orient="records"))

@bp.route("/production", methods=["GET"])
def get_production():
    commodity = request.args.get("commodity")
    district = request.args.get("district")
    df = DF_PR
    if commodity: df = df[df["commodity_name"] == commodity]
    if district: df = df[df["district"] == district]
    return jsonify(df.to_dict(orient="records"))
