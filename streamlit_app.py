import os
import json
import math
import re
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime, timezone
from io import BytesIO

import requests
import pandas as pd
import numpy as np
import streamlit as st

try:
    import openpyxl
    HAS_OPENPYXL = True
    # Debug: Show in sidebar if openpyxl is available
except ImportError:
    HAS_OPENPYXL = False
    # Debug: Show in sidebar if openpyxl is NOT available
if HAS_OPENPYXL:
    from openpyxl.utils import get_column_letter

# -----------------------------
# Config
# -----------------------------
DEFAULT_HOST = os.getenv("DERIBIT_HOST", "https://test.deribit.com")

def get_config_path():
    """Find config.json in multiple possible locations"""
    possible_paths = [
        Path("./config.json"),  # Same directory as app.py (Linux/Ubuntu)
        Path("config.json"),  # Current working directory
        Path(r"I:\myDocuments\Gitlab\deribit_position\config.json"),  # Original Windows path
        Path.home() / "deribit_position" / "config.json",  # User home directory
        Path("/etc/deribit-dashboard/config.json"),  # System config (Linux)
    ]

    for path in possible_paths:
        if path.exists():
            return path

    # If none found, return the first one (same directory) for error message
    return possible_paths[0]

CONFIG_PATH = get_config_path()

def load_config() -> Dict[str, Any]:
    """
    Streamlit-friendly config loader:
    Priority:
      1) st.secrets (Streamlit Cloud)
      2) env vars
      3) local config.json
    """
    # 1) Streamlit secrets
    try:
        if "deribit" in st.secrets:
            d = st.secrets["deribit"]
            return {
                "host": d.get("host", DEFAULT_HOST),
                "client_id": d["client_id"],
                "client_secret": d["client_secret"],
                "subs": d.get("subs", []),
            }
    except Exception:
        # st.secrets not available or not configured
        pass

    # 2) Environment variables
    client_id = os.getenv("DERIBIT_CLIENT_ID", "").strip()
    client_secret = os.getenv("DERIBIT_CLIENT_SECRET", "").strip()
    subs_raw = os.getenv("DERIBIT_SUBS_JSON", "").strip()  # e.g. '[{"alias":"A","id":123}]'
    if client_id and client_secret and subs_raw:
        return {
            "host": DEFAULT_HOST,
            "client_id": client_id,
            "client_secret": client_secret,
            "subs": json.loads(subs_raw),
        }

    # 3) Local config.json (dev)
    cfg_path = CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Missing {cfg_path}. Provide Streamlit secrets [deribit] or env vars "
            "DERIBIT_CLIENT_ID / DERIBIT_CLIENT_SECRET / DERIBIT_SUBS_JSON, "
            "or a local config.json."
        )
    with open(cfg_path, "r", encoding="utf-8") as f:
        c = json.load(f)

    return {
        "host": c.get("host", DEFAULT_HOST),
        "client_id": str(c.get("client_id", "")).strip(),
        "client_secret": str(c.get("client_secret", "")).strip(),
        "subs": c.get("subs", []),
    }

def clean_for_json(obj):
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_for_json(x) for x in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (np.float64, np.float32)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    return obj

# -----------------------------
# Deribit calls (reused from your app)
# -----------------------------
def get_token(config: Dict[str, Any]) -> str:
    r = requests.get(
        f"{config['host']}/api/v2/public/auth",
        params={
            "grant_type": "client_credentials",
            "client_id": config["client_id"],
            "client_secret": config["client_secret"],
        },
        headers={"Accept": "application/json"},
        timeout=20,
    )
    j = r.json()
    if "error" in j:
        raise RuntimeError(f"auth failed: {j['error']}")
    return j["result"]["access_token"]

def fetch_positions(token: str, config: Dict[str, Any], sub_id: int, alias: str,
                    currency: str = "any", kind: str | None = None) -> pd.DataFrame:
    params = {"currency": currency}
    if kind:
        params["kind"] = kind
    if sub_id is not None:
        params["subaccount_id"] = sub_id

    r = requests.get(
        f"{config['host']}/api/v2/private/get_positions",
        params=params,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        timeout=20,
    )
    j = r.json()
    if "error" in j:
        return pd.DataFrame()

    rows = j.get("result", [])
    if not rows:
        return pd.DataFrame()

    df = pd.json_normalize(rows)

    # instrument parsing
    if "instrument_name" in df.columns:
        parts = df["instrument_name"].astype(str).str.split("-", n=3, expand=True)
        while parts.shape[1] < 4:
            parts[parts.shape[1]] = pd.NA
        df["UNDERLYING"] = parts[0]
        df["EXPIRY"] = parts[1]
        df["STRIKE"] = pd.to_numeric(parts[2], errors="coerce")
        df["OPT_TYPE"] = parts[3]

    # time to expiry
    if "EXPIRY" in df.columns:
        def parse_expiry(x):
            try:
                dt = datetime.strptime(x, "%d%b%y")
                return dt.replace(tzinfo=timezone.utc)
            except:
                return None

        expiry_dates = df["EXPIRY"].astype(str).apply(parse_expiry)
        now = datetime.now(timezone.utc)
        df["Time to Expiry"] = expiry_dates.apply(lambda d: (d - now).days if d else pd.NA)

    # Make numeric columns
    numeric_cols = [
        "size", "mark_price", "index_price", "average_price",
        "delta", "gamma", "vega", "theta",
        "realized_profit_loss", "floating_profit_loss", "total_profit_loss",
        "initial_margin", "maintenance_margin", "open_orders_margin"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # mark_vs_avg_pct
    if {"mark_price", "average_price"}.issubset(df.columns):
        mask_opt = df.get("kind", pd.Series(index=df.index)).eq("option")
        safe = mask_opt & df["average_price"].ne(0)
        df.loc[safe, "mark_vs_avg_pct"] = (
            df.loc[safe, "mark_price"] / df.loc[safe, "average_price"] * 100
        ).round(2)
    if "mark_vs_avg_pct" not in df.columns:
        df["mark_vs_avg_pct"] = pd.NA

    # value logic (same as yours)
    df["value"] = pd.NA
    if {"UNDERLYING", "size", "mark_price"}.issubset(df.columns):
        if "index_price" not in df.columns:
            df["index_price"] = pd.NA

        kind_series = df.get("kind", pd.Series(index=df.index, dtype="object")).astype(str)
        mask_future = kind_series.eq("future")
        underlying_series = df["UNDERLYING"].astype(str)
        mask_usdc = underlying_series.str.endswith("_USDC")

        mask_usdc_valid = (~mask_future) & mask_usdc
        df.loc[mask_usdc_valid, "value"] = df.loc[mask_usdc_valid, "size"] * df.loc[mask_usdc_valid, "mark_price"]

        mask_non_usdc_valid = (~mask_future) & (~mask_usdc)
        df.loc[mask_non_usdc_valid, "value"] = (
            df.loc[mask_non_usdc_valid, "size"]
            * df.loc[mask_non_usdc_valid, "mark_price"]
            * df.loc[mask_non_usdc_valid, "index_price"]
        )

    df["value"] = pd.to_numeric(df["value"], errors="coerce").round(0)

    if {"value", "Time to Expiry"}.issubset(df.columns):
        df["value per day"] = pd.NA
        mask = pd.to_numeric(df["Time to Expiry"], errors="coerce").gt(0)
        df.loc[mask, "value per day"] = df.loc[mask, "value"] / df.loc[mask, "Time to Expiry"]
        df["value per day"] = pd.to_numeric(df["value per day"], errors="coerce").round(1)

    df["value per month"] = pd.to_numeric(df.get("value per day"), errors="coerce") * 30
    df["value per month"] = pd.to_numeric(df["value per month"], errors="coerce").round(1)

    # HKD conversion (your 7.8)
    for col in ["value", "value per day", "value per month"]:
        if col in df.columns:
            df[f"{col}_HKD"] = pd.to_numeric(df[col], errors="coerce") * 7.8

    df.insert(0, "ALIAS", alias)
    df.insert(1, "SUB_ID", sub_id)

    # Column order (remove mark_vs_avg_pct and HKD columns)
    keep = [
        "ALIAS","SUB_ID",
        "instrument_name","UNDERLYING","EXPIRY","Time to Expiry","STRIKE","OPT_TYPE",
        "kind","direction","size","size_currency",
        "average_price","mark_price","index_price",
        "value","value per day","value per month",
        "delta","gamma","vega","theta",
        "realized_profit_loss","floating_profit_loss","total_profit_loss",
        "initial_margin","maintenance_margin","open_orders_margin",
        "timestamp",
    ]

    cols = [c for c in keep if c in df.columns]

    # Sort by value desc (since mark_vs_avg_pct is removed)
    if "value" in df.columns:
        df = df.sort_values("value", ascending=False, na_position="last")

    return df[cols] if cols else df

@st.cache_data(ttl=20)
def load_all_positions(currency: str, kind: str | None) -> tuple[pd.DataFrame, list[str], dict]:
    config = load_config()
    token = get_token(config)

    frames = []
    errors = []
    sub_accounts = {}

    for sub in config["subs"]:
        alias = sub.get("alias", "Unknown")
        sub_id = sub.get("id")
        try:
            df = fetch_positions(token, config, sub_id=sub_id, alias=alias, currency=currency, kind=kind)
            if not df.empty:
                frames.append(df)
                # Store individual sub-account data
                sub_accounts[alias] = df.copy()
            else:
                sub_accounts[alias] = pd.DataFrame()
        except Exception as e:
            errors.append(f"{alias}: {e}")
            sub_accounts[alias] = pd.DataFrame()

    if frames:
        combined = pd.concat(frames, ignore_index=True)
    else:
        combined = pd.DataFrame()

    return combined, errors, sub_accounts

def create_excel_download(df: pd.DataFrame, filename: str) -> tuple[bytes, str]:
    """Create Excel file for download"""
    if not HAS_OPENPYXL:
        # Fallback to CSV if openpyxl not available
        return df.to_csv(index=False).encode("utf-8"), f"{filename}.csv"

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Positions', index=False)
    buffer.seek(0)
    return buffer.getvalue(), f"{filename}.xlsx"

def _sanitize_sheet_name(name: str) -> str:
    # Excel sheet name rules: max 31 chars and no []:*?/\
    cleaned = re.sub(r"[\[\]:*?/\\]", "_", name).strip()
    if not cleaned:
        cleaned = "Sheet"
    return cleaned[:31]

def _unique_sheet_name(base: str, existing: set[str]) -> str:
    if base not in existing:
        return base
    i = 1
    while True:
        suffix = f"_{i}"
        max_len = 31 - len(suffix)
        trimmed = base[:max_len] if max_len > 0 else "Sheet"
        candidate = f"{trimmed}{suffix}"
        if candidate not in existing:
            return candidate
        i += 1

def create_excel_workbook(dataframes: Dict[str, pd.DataFrame], filename: str) -> tuple[bytes, str]:
    """Create a multi-sheet Excel file for download"""
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        used_names: set[str] = set()
        for sheet_title, frame in dataframes.items():
            base = _sanitize_sheet_name(str(sheet_title))
            sheet_name = _unique_sheet_name(base, used_names)
            used_names.add(sheet_name)
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            # Auto-adjust column widths based on values and header
            for col_idx, col_name in enumerate(frame.columns, start=1):
                series = frame[col_name].astype(str)
                max_len = max([len(str(col_name))] + [len(v) for v in series.tolist()])
                worksheet.column_dimensions[get_column_letter(col_idx)].width = min(max_len + 2, 60)
    buffer.seek(0)
    return buffer.getvalue(), f"{filename}.xlsx"

def create_json_download(data: dict, filename: str) -> tuple[bytes, str]:
    """Create JSON file for download"""
    json_str = json.dumps(clean_for_json(data), indent=2, default=str)
    return json_str.encode("utf-8"), f"{filename}.json"

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Deribit Positions Dashboard", layout="wide")

st.title("Deribit Positions Dashboard")

with st.sidebar:
    st.header("Filters")
    currency = st.selectbox("Currency", ["any", "BTC", "ETH"], index=0)
    kind = st.selectbox("Kind", ["(all)", "option", "future"], index=0)
    kind_val = None if kind == "(all)" else kind
    refresh = st.button("Refresh")
    download_slot = st.empty()
    
    # Debug info
    st.markdown("---")
    st.caption(f"Excel export: {'✅ Available' if HAS_OPENPYXL else '❌ CSV fallback'}")

if refresh:
    st.cache_data.clear()

try:
    with st.spinner("Loading position data..."):
        df, errors, sub_accounts = load_all_positions(currency=currency, kind=kind_val)
except FileNotFoundError as e:
    st.error(f"❌ Configuration file not found: {e}")
    st.info("💡 Please ensure config.json exists in the project directory or configure Streamlit secrets.")
    st.stop()
except ValueError as e:
    st.error(f"❌ Configuration error: {e}")
    st.info("💡 Check that your config.json contains valid client_id, client_secret, and subs array.")
    st.stop()
except requests.RequestException as e:
    st.error(f"❌ Network/API error: {e}")
    st.info("💡 Check your internet connection and Deribit API status.")
    st.stop()
except Exception as e:
    st.error(f"❌ Unexpected error: {e}")
    st.info("💡 Please check the application logs for more details.")
    st.stop()

if errors:
    with st.expander("⚠️ Sub-account Errors", expanded=False):
        st.warning("Some sub-accounts failed to load:")
        for err in errors:
            st.error(f"• {err}")
        st.info("💡 This may be due to invalid sub-account IDs or API permissions.")

if df.empty:
    st.info("📭 No positions found matching the current filters.")
    if not errors:
        st.success("✅ API connection successful - no open positions.")
else:
    if HAS_OPENPYXL:
        workbook_frames = {"All Positions": df}
        for alias, sub_df in sub_accounts.items():
            workbook_frames[alias] = sub_df
        date_stamp = datetime.now().strftime("%Y%m%d")
        excel_bytes, excel_filename = create_excel_workbook(
            workbook_frames, f"deribit_positions_{date_stamp}"
        )
        with download_slot.container():
            st.download_button(
                "📥 Download Excel",
                excel_bytes,
                file_name=excel_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    else:
        with download_slot.container():
            st.info("Excel workbook export requires openpyxl.")

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total positions", int(len(df)))
    with col2:
        total_value = df['value'].sum() if 'value' in df.columns else 0
        st.metric("Total value", f"${total_value:,.0f}")
    with col3:
        total_value_hkd = df['value_HKD'].sum() if 'value_HKD' in df.columns else 0
        st.metric("Total value (HKD)", f"HKD ${total_value_hkd:,.0f}")

    # Create tabs for combined view and individual sub-accounts
    tab_names = ["All Positions"] + [alias for alias in sub_accounts.keys()]
    tabs = st.tabs(tab_names)

    # Combined view tab
    with tabs[0]:
        st.dataframe(df, use_container_width=True, height=650)

        # Export options
        col1, col2, col3 = st.columns(3)
        with col1:
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📄 CSV", csv, file_name="deribit_positions.csv", mime="text/csv")

        with col2:
            excel_data, excel_filename = create_excel_download(df, "deribit_positions")
            mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if HAS_OPENPYXL else "text/csv"
            st.download_button("📊 Excel", excel_data, file_name=excel_filename, mime=mime_type)

        with col3:
            json_data = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "total_positions": len(df),
                    "filters": {"currency": currency, "kind": kind_val}
                },
                "positions": clean_for_json(df.to_dict('records'))
            }
            json_bytes, json_filename = create_json_download(json_data, "deribit_positions")
            st.download_button("🔧 JSON", json_bytes, file_name=json_filename, mime="application/json")

    # Individual sub-account tabs
    for i, (alias, sub_df) in enumerate(sub_accounts.items(), 1):
        with tabs[i]:
            if sub_df.empty:
                st.info(f"No positions for {alias}")
            else:
                st.metric(f"{alias} positions", int(len(sub_df)))
                st.dataframe(sub_df, use_container_width=True, height=650)

                # Export options for sub-account
                col1, col2, col3 = st.columns(3)
                with col1:
                    csv = sub_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        f"📄 {alias} CSV",
                        csv,
                        file_name=f"deribit_positions_{alias}.csv",
                        mime="text/csv",
                        key=f"csv_{alias}"
                    )

                with col2:
                    excel_data, excel_filename = create_excel_download(sub_df, f"deribit_positions_{alias}")
                    mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if HAS_OPENPYXL else "text/csv"
                    st.download_button(
                        f"📊 {alias} Excel",
                        excel_data,
                        file_name=excel_filename,
                        mime=mime_type,
                        key=f"excel_{alias}"
                    )

                with col3:
                    json_data = {
                        "metadata": {
                            "timestamp": datetime.now().isoformat(),
                            "account": alias,
                            "positions_count": len(sub_df),
                            "filters": {"currency": currency, "kind": kind_val}
                        },
                        "positions": clean_for_json(sub_df.to_dict('records'))
                    }
                    json_bytes, json_filename = create_json_download(json_data, f"deribit_positions_{alias}")
                    st.download_button(
                        f"🔧 {alias} JSON",
                        json_bytes,
                        file_name=json_filename,
                        mime="application/json",
                        key=f"json_{alias}"
                    )