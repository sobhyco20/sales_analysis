# -*- coding: utf-8 -*-
# ==============================================
# 📊 Streamlit — لوحة تحليل المبيعات (نسخة مستقرة مع حفظ دائم)
# ==============================================

import datetime
from io import BytesIO
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# -------------------------------
# إعدادات الصفحة
# -------------------------------
st.set_page_config(page_title="تحليل المبيعات", page_icon="📊", layout="wide")


# -------------------------------
# أدوات عامة
# -------------------------------
def _clean_col_name(name: str) -> str:
    """توحيد اسم العمود (إزالة الفراغات الإضافية + تحويل مسافات متعددة لمسافة واحدة)."""
    return re.sub(r"\s+", " ", str(name)).strip()


def _coerce_numeric(s: pd.Series) -> pd.Series:
    """تحويل أي سلسلة رقمية إلى float بأمان (يراعي وجود فواصل وأحرف مخفية)."""
    if s.dtype.kind in ("i", "u", "f"):
        return s.astype(float)
    return (
        s.astype(str)
        .str.replace("\u200f", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace("\t", "", regex=False)
        .replace(["", "None", "nan", "NaN"], np.nan)
        .astype(float)
    )


# قاموس الأسماء البديلة للأعمدة (بعد تنظيفها بـ _clean_col_name ثم lower)
FIELD_ALIASES: Dict[str, List[str]] = {
    "invoice": ["invoice#", "الفاتورة", "رقم الفاتورة"],
    "customer": ["customer name", "العميل", "اسم العميل"],
    "customer_group": ["customer group", "مجموعة العملاء"],
    "date": ["trans date", "date", "التاريخ"],
    "period": ["period", "الفترة"],
    "item": ["item name", "اسم الصنف", "الصنف"],
    "item_code": ["item#", "كود الصنف"],
    "qty": ["qty|kg", "qty", "quantity", "الكمية"],
    "unit": ["unit", "الوحدة"],
    "sales_total": ["total in sar", "net sales", "إجمالي المبيعات", "المبيعات"],
    "cost_total": ["total cost", "إجمالي التكلفة", "التكلفة"],
    "unit_cost": ["unit cost", "تكلفة الوحدة"],
    "product_type": ["product type", "نوع المنتج"],
    "item_group": ["item group", "مجموعة الصنف"],
    "voucher": ["voucher #", "رقم السند"],
    "so": ["so#"],
    "cust_no": ["cust #", "customer no", "رقم العميل"],
    "ojaimi_num": ["ojaimi num"],
    "rep": ["sales representative", "sales rep", "المندوب"],
}


def _resolve_col(df: pd.DataFrame, key: str) -> Optional[str]:
    """إرجاع اسم العمود الفعلي في الداتا فريم بحسب المسمّى المنطقي (key)."""
    candidates = FIELD_ALIASES.get(key, [])
    norm = { _clean_col_name(c).lower(): c for c in df.columns }
    for alias in candidates:
        a = _clean_col_name(alias).lower()
        if a in norm:
            return norm[a]
    return None


def _detect_all_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """يحاول اكتشاف كل الأعمدة ذات الصلة مرة واحدة."""
    return {k: _resolve_col(df, k) for k in FIELD_ALIASES.keys()}


def _ensure_date_month_year(df: pd.DataFrame, cols: Dict[str, Optional[str]]) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    """تحويل عمود التاريخ إن وجد + إنشاء __year__ و __month_num__."""
    df = df.copy()
    date_col = cols.get("date")
    if date_col and date_col in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df["__year__"] = df[date_col].dt.year
        df["__month_num__"] = df[date_col].dt.month
    else:
        df["__year__"] = np.nan
        df["__month_num__"] = np.nan
    return df, cols


def _add_derived_columns(df: pd.DataFrame, cols: Dict[str, Optional[str]]) -> pd.DataFrame:
    """إنشاء الأعمدة الداخلية اللازمة للتجميع والتحليل."""
    df = df.copy()

    qty_col = cols.get("qty")
    sales_col = cols.get("sales_total")
    cost_col = cols.get("cost_total")
    unit_cost_col = cols.get("unit_cost")

    # الكمية
    if qty_col and qty_col in df.columns:
        df["__qty__"] = _coerce_numeric(df[qty_col])
    else:
        df["__qty__"] = 0.0

    # إجمالي المبيعات
    if sales_col and sales_col in df.columns:
        df["__sales_total__"] = _coerce_numeric(df[sales_col])
    else:
        df["__sales_total__"] = 0.0

    # إجمالي التكلفة
    if cost_col and cost_col in df.columns:
        df["__cost_total__"] = _coerce_numeric(df[cost_col])
    else:
        if unit_cost_col and unit_cost_col in df.columns and qty_col and qty_col in df.columns:
            df["__cost_total__"] = _coerce_numeric(df[qty_col]) * _coerce_numeric(df[unit_cost_col])
        else:
            df["__cost_total__"] = 0.0

    # الربح والمشتقات
    df["الربح"] = df["__sales_total__"] - df["__cost_total__"]
    with np.errstate(divide="ignore", invalid="ignore"):
        df["نسبة الربح %"] = np.where(df["__sales_total__"] != 0, (df["الربح"] / df["__sales_total__"]) * 100.0, np.nan)
        df["متوسط سعر البيع الفعلي"] = np.where(df["__qty__"] != 0, df["__sales_total__"] / df["__qty__"], np.nan)

    return df


# -------------------------------
# حفظ/تحميل الملف المرفوع (دائم)
# -------------------------------
@st.cache_data(show_spinner=False)
def _read_excel_first_sheet(file_bytes: bytes) -> pd.DataFrame:
    """قراءة أول شيت من ملف إكسل (بايتس)."""
    return pd.read_excel(BytesIO(file_bytes), engine="openpyxl")


def _save_bytes_to_disk(data: bytes, suggested_name: str) -> Path:
    uploads_dir = Path("uploaded_files")
    uploads_dir.mkdir(exist_ok=True)
    safe_stem = re.sub(r'[<>:"/\\|?*]+', "_", Path(suggested_name).stem or "uploaded")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = uploads_dir / f"{safe_stem}_{ts}.xlsx"
    with open(save_path, "wb") as f:
        f.write(data)
    return save_path


def _find_latest_saved_file() -> Optional[Path]:
    uploads_dir = Path("uploaded_files")
    if not uploads_dir.exists():
        return None
    files = sorted(uploads_dir.glob("*.xlsx"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _load_data() -> Optional[pd.DataFrame]:
    """يرفع/يحمل الملف مع إبقاءه محفوظًا محليًا واستعادته تلقائيًا بعد إعادة التحميل."""
    with st.sidebar:
        st.markdown("### 📁 ملف البيانات")
        file = st.file_uploader("ارفع ملف Excel", type=["xlsx", "xls"], key="file_uploader_main")

        # 1) لو رفع ملف الآن: خزّنه في القرص وفي الجلسة
        if file is not None:
            try:
                data = file.read()
                # حفظ دائم على القرص
                saved_path = _save_bytes_to_disk(data, file.name)
                st.session_state["last_file_path"] = str(saved_path)
                # قراءة وإرجاع
                df = _read_excel_first_sheet(data)
                df.columns = [_clean_col_name(c) for c in df.columns]
                st.session_state["df_cached"] = df  # حفظ داخل الجلسة لتسريع rerun
                return df
            except Exception:
                pass

        # 2) لا يوجد رفع الآن، لو عندنا داتا محفوظة في الجلسة، ارجعها
        if "df_cached" in st.session_state and isinstance(st.session_state["df_cached"], pd.DataFrame):
            return st.session_state["df_cached"]

        # 3) لا يوجد في الجلسة — نحاول تحميل آخر ملف محفوظ على القرص
        latest = _find_latest_saved_file()
        if latest is not None and latest.exists():
            try:
                data = latest.read_bytes()
                df = _read_excel_first_sheet(data)
                df.columns = [_clean_col_name(c) for c in df.columns]
                st.session_state["last_file_path"] = str(latest)
                st.session_state["df_cached"] = df
                st.info(f"تم تحميل آخر ملف محفوظ تلقائيًا: {latest.name}")
                return df
            except Exception:
                return None

        # 4) لا شيء
        return None


# -------------------------------
# تبويب: مقارنة عميل بين سنتين
# -------------------------------
def customer_year_compare_tab(df: pd.DataFrame):
    st.subheader("👤 العميل — مقارنة سنتين", divider="rainbow")

    cols = _detect_all_columns(df)
    df, cols = _ensure_date_month_year(df, cols)
    df = _add_derived_columns(df, cols)

    # أسماء الأشهر
    month_names_ar = [
        "01-يناير", "02-فبراير", "03-مارس", "04-أبريل", "05-مايو", "06-يونيو",
        "07-يوليو", "08-أغسطس", "09-سبتمبر", "10-أكتوبر", "11-نوفمبر", "12-ديسمبر",
    ]

    # فلاتر أعلى الصفحة
    c1, c2, c3 = st.columns([2, 1, 1])

    # قائمة العملاء
    customers = []
    if cols.get("customer") and cols["customer"] in df.columns:
        customers = sorted(df[cols["customer"]].dropna().astype(str).unique())

    # تهيئة session_state الافتراضية
    st.session_state.setdefault("chosen_customer", "— اختر —")
    st.session_state.setdefault("years_selected", [])
    st.session_state.setdefault("months_selected", month_names_ar)

    # اختيار العميل (قابل للبحث)
    chosen_customer = c1.selectbox(
        "🧑‍🤝‍🧑 اختر العميل",
        options=["— اختر —"] + customers,
        index=(["— اختر —"] + customers).index(st.session_state["chosen_customer"])
              if st.session_state["chosen_customer"] in (["— اختر —"] + customers) else 0,
        key="chosen_customer"
    )

    # السنين
    years = sorted([int(y) for y in df["__year__"].dropna().unique()]) if "__year__" in df.columns else []
    default_years = years[-2:] if len(years) >= 2 else years
    if not st.session_state["years_selected"]:
        st.session_state["years_selected"] = default_years

    selected_years = c2.multiselect("📅 سنتان للمقارنة", options=years,
                                    default=st.session_state["years_selected"],
                                    max_selections=2, key="years_selected")

    # الشهور
    selected_months_labels = c3.multiselect("🗓️ الشهور", options=month_names_ar,
                                            default=st.session_state["months_selected"],
                                            key="months_selected")
    selected_months = [lbl.split("-")[0] for lbl in selected_months_labels]

    # شروط أساسية
    if chosen_customer == "— اختر —" or len(selected_years) != 2:
        st.info("اختر عميلًا وحدد سنتين لعرض المقارنة.")
        return

    y1, y2 = sorted(selected_years)
    inv_col = cols.get("invoice")
    item_col = cols.get("item")
    date_col = cols.get("date")

    # تطبيق الفلتر
    mask = pd.Series(True, index=df.index)
    if cols.get("customer"):
        mask &= df[cols["customer"]].astype(str).eq(chosen_customer)
    mask &= df["__year__"].isin([y1, y2])
    dff = df.loc[mask].copy()

    # أعمدة مساعدة للشهور
    if date_col and date_col in dff.columns:
        dff["__month_num__"] = pd.to_datetime(dff[date_col]).dt.month.fillna(0).astype(int)
    else:
        dff["__month_num__"] = 0

    # إجمالي الفواتير حسب السنة
    if inv_col and inv_col in dff.columns:
        invoices_by_year = dff.groupby("__year__")[inv_col].nunique().reindex([y1, y2]).fillna(0).astype(int)
    else:
        invoices_by_year = pd.Series({y1: np.nan, y2: np.nan})

    # مقارنة الأصناف
    has_items = bool(item_col and item_col in dff.columns)

    def _agg_item(df_year: pd.DataFrame) -> pd.DataFrame:
        if not has_items:
            return pd.DataFrame()
        agg = df_year.groupby(item_col, dropna=False).agg({
            "__qty__": "sum",
            "__sales_total__": "sum",
            "__cost_total__": "sum",
            "الربح": "sum",
        }).reset_index()
        return agg

    dff_y1 = dff[dff["__year__"] == y1]
    dff_y2 = dff[dff["__year__"] == y2]

    items_y1 = _agg_item(dff_y1)
    items_y2 = _agg_item(dff_y2)

    comp_items = pd.DataFrame()
    if has_items:
        comp_items = pd.merge(
            items_y1.add_suffix(f"_{y1}"),
            items_y2.add_suffix(f"_{y2}"),
            left_on=f"{item_col}_{y1}",
            right_on=f"{item_col}_{y2}",
            how="outer",
        )
        first_nonnull = comp_items[[f"{item_col}_{y1}", f"{item_col}_{y2}"]].bfill(axis=1).ffill(axis=1).iloc[:, 0]
        comp_items["الصنف"] = first_nonnull
        comp_items.drop(columns=[f"{item_col}_{y1}", f"{item_col}_{y2}"], inplace=True)
        num_cols = comp_items.select_dtypes(include=["number"]).columns
        comp_items[num_cols] = comp_items[num_cols].fillna(0.0)

    # ربحية الأصناف
    def _profit_by_item(df_year: pd.DataFrame) -> pd.DataFrame:
        if not has_items:
            return pd.DataFrame()
        piv = df_year.groupby(item_col, dropna=False).agg({
            "__qty__": "sum",
            "__sales_total__": "sum",
            "__cost_total__": "sum",
        }).reset_index()
        piv["الربح"] = piv["__sales_total__"] - piv["__cost_total__"]
        with np.errstate(divide="ignore", invalid="ignore"):
            piv["نسبة الربح %"] = np.where(
                piv["__sales_total__"] != 0,
                (piv["الربح"] / piv["__sales_total__"]) * 100.0,
                np.nan
            )
        piv.rename(columns={"__qty__": "الكمية", "__sales_total__": "إجمالي المبيعات", "__cost_total__": "إجمالي التكلفة"}, inplace=True)
        return piv.sort_values("الربح", ascending=False)

    prof_y1 = _profit_by_item(dff_y1)
    prof_y2 = _profit_by_item(dff_y2)

    # مصفوفة شهرية
    def _monthly_matrix(df_year: pd.DataFrame, year_label: int) -> pd.DataFrame:
        g = df_year.groupby("__month_num__").agg({
            "__qty__": "sum",
            "__sales_total__": "sum",
            "__cost_total__": "sum",
        }).reindex(range(1, 12 + 1)).reset_index()
        month_names_ar = [
            "01-يناير", "02-فبراير", "03-مارس", "04-أبريل", "05-مايو", "06-يونيو",
            "07-يوليو", "08-أغسطس", "09-سبتمبر", "10-أكتوبر", "11-نوفمبر", "12-ديسمبر",
        ]
        g["الشهر"] = g["__month_num__"].map(lambda m: month_names_ar[m-1] if 1 <= m <= 12 else "-")
        g.drop(columns=["__month_num__"], inplace=True)
        g["الربح"] = g["__sales_total__"] - g["__cost_total__"]
        with np.errstate(divide="ignore", invalid="ignore"):
            g["نسبة الربح %"] = np.where(
                g["__sales_total__"] != 0,
                (g["الربح"] / g["__sales_total__"]) * 100.0,
                np.nan
            )
        # فلترة الشهور
        g["__m__"] = g["الشهر"].str[:2]
        g = g[g["__m__"].isin(selected_months)].drop(columns=["__m__"])
        # إعادة تسمية
        g.rename(columns={
            "__qty__": f"الكمية {year_label}",
            "__sales_total__": f"المبيعات {year_label}",
            "__cost_total__": f"التكلفة {year_label}",
            "الربح": f"الربح {year_label}",
            "نسبة الربح %": f"% الربح {year_label}",
        }, inplace=True)
        return g

    mat_y1 = _monthly_matrix(dff_y1, y1)
    mat_y2 = _monthly_matrix(dff_y2, y2)
    monthly = pd.merge(mat_y1, mat_y2, on="الشهر", how="outer")

    # مجاميع
    def _totals(df_year: pd.DataFrame) -> Dict[str, float]:
        sales = float(df_year["__sales_total__"].sum())
        cost = float(df_year["__cost_total__"].sum())
        profit = sales - cost
        margin = (profit / sales * 100.0) if sales else np.nan
        invs = df_year[inv_col].nunique() if inv_col and inv_col in df_year.columns else np.nan
        return {"sales": sales, "cost": cost, "profit": profit, "margin": margin, "invoices": invs}

    t1 = _totals(dff_y1)
    t2 = _totals(dff_y2)

    # بطاقة باسم العميل المختار (كبيرة)
    if chosen_customer and chosen_customer != "— اختر —":
        st.markdown(
            f"""
            <div style='
                background: linear-gradient(90deg,#2b5876,#4e4376);
                color:#fff;
                padding: 16px 20px;
                border-radius: 12px;
                text-align: center;
                font-size: 26px;
                font-weight: 800;
                letter-spacing: .5px;
                margin-bottom: 18px;
            '>
                🧑‍💼 {chosen_customer}
            </div>
            """,
            unsafe_allow_html=True
        )

    # صفّان: كل سنة في صف (مبيعات + عدد فواتير + الربح)
    st.markdown("### 📌 نظرة سريعة حسب السنة")
    r1c1, r1c2, r1c3 = st.columns(3)
    r1c1.metric(f"إجمالي المبيعات {y1}", f"{t1['sales']:,.0f}")
    r1c2.metric(f"عدد فواتير {y1}", f"{t1['invoices'] if pd.notna(t1['invoices']) else '-'}")
    r1c3.metric(f"الربح {y1}", f"{t1['profit']:,.0f}")

    r2c1, r2c2, r2c3 = st.columns(3)
    r2c1.metric(f"إجمالي المبيعات {y2}", f"{t2['sales']:,.0f}")
    r2c2.metric(f"عدد فواتير {y2}", f"{t2['invoices'] if pd.notna(t2['invoices']) else '-'}")
    r2c3.metric(f"الربح {y2}", f"{t2['profit']:,.0f}")

    # الجداول
    st.markdown("### 🗓️ مقارنة شهرية")
    disp_monthly = monthly.copy()
    num_cols = disp_monthly.select_dtypes(include=["number"]).columns.tolist()
    for c in num_cols:
        if c.startswith("% الربح "):
            disp_monthly[c] = disp_monthly[c].apply(lambda x: "" if pd.isna(x) else f"{x:,.0f}%")
        else:
            disp_monthly[c] = disp_monthly[c].apply(lambda x: "" if pd.isna(x) else f"{x:,.0f}")
    st.dataframe(disp_monthly, use_container_width=True, hide_index=True)

    if has_items and not comp_items.empty:
        st.markdown("### 🛒 الأصناف — مقارنة بين السنتين")
        comp_view = comp_items.copy()
        num_cols = comp_view.select_dtypes(include=["number"]).columns.tolist()
        for c in num_cols:
            comp_view[c] = comp_view[c].apply(lambda x: f"{x:,.0f}")
        st.dataframe(comp_view, use_container_width=True, hide_index=True)

        st.markdown(f"### 💰 ربحية الأصناف في {y1}")
        st.dataframe(prof_y1, use_container_width=True, hide_index=True)
        st.markdown(f"### 💰 ربحية الأصناف في {y2}")
        st.dataframe(prof_y2, use_container_width=True, hide_index=True)

    # رسم
    try:
        import plotly.express as px
        melt_cols = [c for c in monthly.columns if c.startswith("المبيعات ")]
        if len(melt_cols) == 2:
            plot_df = monthly.melt(id_vars=["الشهر"], value_vars=melt_cols, var_name="السنة", value_name="المبيعات")
            plot_df["السنة"] = plot_df["السنة"].str.replace("المبيعات ", "")
            fig = px.bar(plot_df, x="الشهر", y="المبيعات", color="السنة", barmode="group")
            fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass

    # تنزيل Excel
    st.markdown("---")
    out = BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        summary_df = pd.DataFrame({
            "السنة": [y1, y2],
            "إجمالي المبيعات": [t1['sales'], t2['sales']],
            "إجمالي التكلفة": [t1['cost'], t2['cost']],
            "إجمالي الربح": [t1['profit'], t2['profit']],
            "نسبة الربح %": [t1['margin'], t2['margin']],
            "عدد الفواتير": [t1['invoices'], t2['invoices']],
        })
        summary_df.to_excel(writer, index=False, sheet_name="Summary")
        monthly.to_excel(writer, index=False, sheet_name="Monthly")
        if has_items:
            comp_items.to_excel(writer, index=False, sheet_name="Items_Compare")
            prof_y1.to_excel(writer, index=False, sheet_name=f"Profit_{y1}")
            prof_y2.to_excel(writer, index=False, sheet_name=f"Profit_{y2}")

        wb = writer.book
        for sh in ["Summary", "Monthly", "Items_Compare", f"Profit_{y1}", f"Profit_{y2}"]:
            if sh in writer.sheets:
                ws = writer.sheets[sh]
                ws.set_column(0, 0, 22)
                ws.set_column(1, 40, 18)

    st.download_button(
        label="⬇️ تنزيل مقارنة العميل (Excel)",
        data=out.getvalue(),
        file_name=f"customer_compare_{chosen_customer}_{y1}_{y2}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# -------------------------------
# تبويب: Pivot شامل
# -------------------------------
DEFAULT_DIMENSIONS = [
    ("الفاتورة", "invoice"),
    ("العميل", "customer"),
    ("الصنف", "item"),
    ("الشهر", "month"),
    ("الفترة", "period"),
    ("المجموعة", "item_group"),
    ("نوع المنتج", "product_type"),
    ("المندوب", "rep"),
]


def pivot_tab(df: pd.DataFrame):
    st.subheader("📈 Pivot شامل", divider="rainbow")

    cols = _detect_all_columns(df)
    df, cols = _ensure_date_month_year(df, cols)
    df = _add_derived_columns(df, cols)

    # YearMonth للعرض
    if "__year__" in df.columns and "__month_num__" in df.columns:
        df["YearMonth"] = pd.to_datetime(
            dict(year=df["__year__"], month=df["__month_num__"], day=1),
            errors="coerce"
        ).dt.to_period("M").astype(str)
    else:
        df["YearMonth"] = np.nan

    with st.sidebar:
        st.markdown("### ⚙️ فلاتر pivot")

        date_from = date_to = None
        date_col = cols.get("date")
        if date_col and date_col in df.columns:
            try:
                min_d = pd.to_datetime(df[date_col]).min()
                max_d = pd.to_datetime(df[date_col]).max()
                default_start = min_d.date() if pd.notna(min_d) else datetime.date.today()
                default_end = max_d.date() if pd.notna(max_d) else datetime.date.today()
                dr = st.date_input("الفترة الزمنية", value=(default_start, default_end), key="pivot_date")
                if isinstance(dr, tuple) and len(dr) == 2:
                    date_from, date_to = dr
                else:
                    date_from = dr
                    date_to = dr
            except Exception:
                date_from = date_to = None

        def _multiselect_if(col_key: str, label: str, key_suffix: str):
            ck = cols.get(col_key)
            if ck and ck in df.columns:
                options = sorted([x for x in df[ck].dropna().astype(str).unique()])
                return st.multiselect(label, options, key=f"{col_key}_ms_{key_suffix}")
            return []

        sel_customers = _multiselect_if("customer", "العملاء", "cust")
        sel_items = _multiselect_if("item", "الأصناف", "item")
        sel_groups = _multiselect_if("item_group", "مجموعات الأصناف", "grp")
        sel_types = _multiselect_if("product_type", "أنواع المنتج", "type")
        sel_reps = _multiselect_if("rep", "المندوبون", "rep")
        sel_customer_groups = _multiselect_if("customer_group", "مجموعات العملاء", "cgrp")

        st.markdown("---")
        st.markdown("### 🧭 أبعاد التجميع")
        dim_labels = [name for name, key in DEFAULT_DIMENSIONS if cols.get(key)]
        dim_keys = [key for name, key in DEFAULT_DIMENSIONS if cols.get(key)]
        default_dim = [dim_labels[0]] if dim_labels else []
        selected_dims_labels = st.multiselect("اختر الأبعاد", dim_labels, default=default_dim, key="pivot_dims")
        selected_dims_keys = [dim_keys[dim_labels.index(lbl)] for lbl in selected_dims_labels] if selected_dims_labels else []

        sort_by_profit = st.checkbox("ترتيب تنازلي حسب الربح", value=True, key="pivot_sort")
        top_n = st.number_input("أعلى N سجل", min_value=5, max_value=1000, value=50, step=5, key="pivot_topn")

    # تطبيق الفلاتر
    mask = pd.Series(True, index=df.index)
    if date_col and date_from and date_to:
        s = pd.to_datetime(df[date_col]).dt.date
        mask &= (s >= date_from) & (s <= date_to)
    if sel_customers and cols.get("customer"):
        mask &= df[cols["customer"]].astype(str).isin(sel_customers)
    if sel_items and cols.get("item"):
        mask &= df[cols["item"]].astype(str).isin(sel_items)
    if sel_groups and cols.get("item_group"):
        mask &= df[cols["item_group"]].astype(str).isin(sel_groups)
    if sel_types and cols.get("product_type"):
        mask &= df[cols["product_type"]].astype(str).isin(sel_types)
    if sel_reps and cols.get("rep"):
        mask &= df[cols["rep"]].astype(str).isin(sel_reps)
    if sel_customer_groups and cols.get("customer_group"):
        mask &= df[cols["customer_group"]].astype(str).isin(sel_customer_groups)

    dff = df.loc[mask].copy()

    value_cols = [c for c in ["__qty__", "__sales_total__", "__cost_total__", "الربح", "نسبة الربح %", "متوسط سعر البيع الفعلي"] if c in dff.columns]

    group_cols = []
    for key in selected_dims_keys:
        if key == "month":
            group_cols.append("YearMonth")
        else:
            ck = cols.get(key)
            if ck and ck in dff.columns:
                group_cols.append(ck)

    if group_cols:
        agg_map = {
            "__qty__": "sum",
            "__sales_total__": "sum",
            "__cost_total__": "sum",
            "الربح": "sum",
            "نسبة الربح %": "mean",
            "متوسط سعر البيع الفعلي": "mean",
        }
        agg_map = {k: v for k, v in agg_map.items() if k in dff.columns}
        piv = dff.groupby(group_cols, dropna=False).agg(agg_map).reset_index()
    else:
        piv = dff[value_cols].sum(numeric_only=True).to_frame().T if value_cols else pd.DataFrame()

    rename_map = {"__qty__": "الكمية", "__sales_total__": "إجمالي المبيعات", "__cost_total__": "إجمالي التكلفة"}
    piv.rename(columns=rename_map, inplace=True)

    if {"إجمالي المبيعات", "إجمالي التكلفة"}.issubset(piv.columns):
        if "الربح" not in piv.columns:
            piv["الربح"] = 0.0
        piv["الربح"] = piv["إجمالي المبيعات"] - piv["إجمالي التكلفة"]
        with np.errstate(divide="ignore", invalid="ignore"):
            piv["نسبة الربح %"] = np.where(
                piv["إجمالي المبيعات"] != 0,
                (piv["الربح"] / piv["إجمالي المبيعات"]) * 100.0,
                np.nan,
            )

    if sort_by_profit and "الربح" in piv.columns:
        piv = piv.sort_values(by="الربح", ascending=False)

    if top_n and top_n > 0:
        piv = piv.head(int(top_n))

    st.markdown("#### 📈 النتائج الجدولية")
    display_piv = piv.copy()
    if "نسبة الربح %" in display_piv.columns:
        display_piv["نسبة الربح %"] = display_piv["نسبة الربح %"].apply(lambda x: "" if pd.isna(x) else f"{x:,.0f}%")
    numeric_cols = [c for c in display_piv.select_dtypes(include=["number"]).columns if c != "نسبة الربح %"]
    for c in numeric_cols:
        display_piv[c] = display_piv[c].apply(lambda x: "" if pd.isna(x) else f"{x:,.0f}")
    st.dataframe(display_piv, use_container_width=True, hide_index=True)

    st.markdown("---")
    totals = {
        "إجمالي المبيعات": float(piv["إجمالي المبيعات"].sum()) if "إجمالي المبيعات" in piv.columns else np.nan,
        "إجمالي التكلفة": float(piv["إجمالي التكلفة"].sum()) if "إجمالي التكلفة" in piv.columns else np.nan,
        "إجمالي الربح": float(piv["الربح"].sum()) if "الربح" in piv.columns else np.nan,
    }
    if not np.isnan(totals["إجمالي المبيعات"]) and totals["إجمالي المبيعات"] != 0:
        totals["نسبة الربح %"] = (totals["إجمالي الربح"] / totals["إجمالي المبيعات"]) * 100.0
    else:
        totals["نسبة الربح %"] = np.nan

    c1, c2, c3, _ = st.columns(4)
    c1.metric("إجمالي المبيعات", f"{totals['إجمالي المبيعات']:,.0f}" if not np.isnan(totals["إجمالي المبيعات"]) else "-")
    c2.metric("إجمالي التكلفة", f"{totals['إجمالي التكلفة']:,.0f}" if not np.isnan(totals["إجمالي التكلفة"]) else "-")
    c3.metric("إجمالي الربح", f"{totals['إجمالي الربح']:,.0f}" if not np.isnan(totals["إجمالي الربح"]) else "-")

    st.markdown("---")
    out = BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        piv.to_excel(writer, index=False, sheet_name="Pivot")
        totals_df = pd.DataFrame([totals])
        totals_df.to_excel(writer, index=False, sheet_name="Totals")

        wb = writer.book
        ws_piv = writer.sheets["Pivot"]
        ws_tot = writer.sheets["Totals"]
        num_fmt = wb.add_format({"num_format": "#,##0"})
        pct_fmt = wb.add_format({"num_format": '#,##0"%"'})

        for col in piv.columns:
            col_idx = piv.columns.get_loc(col)
            if col == "نسبة الربح %":
                ws_piv.set_column(col_idx, col_idx, 18, pct_fmt)
            elif piv[col].dtype.kind in "ifu":
                ws_piv.set_column(col_idx, col_idx, 18, num_fmt)
            else:
                ws_piv.set_column(col_idx, col_idx, 18)

        for col in totals_df.columns:
            col_idx = totals_df.columns.get_loc(col)
            if col == "نسبة الربح %":
                ws_tot.set_column(col_idx, col_idx, 18, pct_fmt)
            else:
                ws_tot.set_column(col_idx, col_idx, 18, num_fmt)

    st.download_button(
        label="⬇️ تنزيل النتائج (Excel)",
        data=out.getvalue(),
        file_name="sales_pivot_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )




def products_analysis_tab(df):
    st.header("📦 تحليل المنتجات والمجموعات")

    # فلتر العملاء
    customers = st.multiselect("اختر العملاء", df["العميل"].unique())

    # فلتر المجموعات
    groups = st.multiselect("اختر المجموعات", df["المجموعة"].unique())

    filtered_df = df.copy()
    if customers:
        filtered_df = filtered_df[filtered_df["العميل"].isin(customers)]
    if groups:
        filtered_df = filtered_df[filtered_df["المجموعة"].isin(groups)]

    total_sales = filtered_df["المبيعات"].sum()
    st.metric("إجمالي المبيعات", f"{total_sales:,.2f}")

    st.dataframe(filtered_df)



# -------------------------------
# Main
# -------------------------------
def main():
    st.title("📊 لوحة تحليل المبيعات")

    df = _load_data()
    if df is None or df.empty:
        st.info("الرجاء رفع ملف Excel يحتوي بيانات المبيعات لبدء التحليل.")
        return

    tabs = st.tabs([
        "👤 العميل (مقارنة سنتين)",
        "📈 Pivot شامل",
        "📦 تحليل المنتجات والمجموعات"
    ])
    
    with tabs[0]:
        customer_year_compare_tab(df)
    with tabs[1]:
        pivot_tab(df)
    with tabs[2]:
        products_analysis_tab(df)  # دالة جديدة نكتبها لتحليل المنتجات


if __name__ == "__main__":
    main()



