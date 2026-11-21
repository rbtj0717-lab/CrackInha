# main.py — 극지식물 개체별 생육 차이 분석 Streamlit 대시보드
import io
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ─────────────────────────────────────────────
# 한글 폰트 (Streamlit CSS + Plotly)
# ─────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)

st.set_page_config(page_title="극지식물 생육 분석", layout="wide")

# ─────────────────────────────────────────────
# CSV 읽기 (인코딩 시도: utf-8-sig → cp949)
# ─────────────────────────────────────────────
@st.cache_data
def read_csv(file):
    """업로드된 파일을 utf-8-sig / cp949 순서로 시도하여 읽는다."""
    if file is None:
        return pd.DataFrame()
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


# ─────────────────────────────────────────────
# 열 이름 표준화 (KeyError 방지, 필수 함수)
# ─────────────────────────────────────────────
def std_cols(df):
    """학교마다 다른 열 이름을 표준 이름으로 통일하고 문자열을 숫자로 변환한다."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"

    df = df.rename(columns=rename)

    # 숫자 변환: 문자열이 섞인 열을 숫자로
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c].astype(str).str.replace(",", "").str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
            if num.notna().mean() >= 0.5:
                df[c] = num

    # 생중량 열이 없으면 지상부 질량 + 지하부 질량으로 계산
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]

    # 시간 열 파싱
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = pd.to_datetime(parts[0], format="%Y%m%d") + pd.to_timedelta(
                parts[1].astype(int), unit="h"
            )
        df["time"] = parsed

    return df


# ─────────────────────────────────────────────
# 추세선 helper (np.polyfit 사용, statsmodels 불필요)
# ─────────────────────────────────────────────
def add_trendline(fig, x, y, color="gray", dash="dash"):
    """산점도 fig에 단순 선형 추세선을 추가한다."""
    valid = pd.concat([x, y], axis=1).dropna()
    if len(valid) < 2:
        return
    coeffs = np.polyfit(valid.iloc[:, 0], valid.iloc[:, 1], 1)
    slope, intercept = coeffs[0], coeffs[1]
    x_range = np.linspace(valid.iloc[:, 0].min(), valid.iloc[:, 0].max(), 100)
    y_pred = slope * x_range + intercept
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_pred,
            mode="lines",
            line=dict(color=color, dash=dash),
            name=f"추세선 (r = {valid.corr().iloc[0, 1]:.2f})",
        )
    )


# ─────────────────────────────────────────────
# 데이터 로딩
# ─────────────────────────────────────────────
st.sidebar.title("📂 데이터 업로드")

env_file = st.sidebar.file_uploader("환경 CSV 업로드", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 CSV 업로드", type="csv")

# 생육 측정 항목 선택
growth_cols_default = ["잎 수", "지상부 길이", "지하부 길이", "생중량"]
available_growth_cols = [c for c in growth_cols_default]
selected_growth_cols = st.sidebar.multiselect(
    "생육 측정 항목 선택",
    options=available_growth_cols,
    default=available_growth_cols,
)

# 학교명 추출 (파일명에서, 하드코딩 금지)
def extract_school_name(filename):
    """파일명에서 학교명을 추출한다 (예: '인하부고_환경데이터.csv' → '인하부고')."""
    if filename is None:
        return "우리 학교"
    base = os.path.splitext(os.path.basename(filename))[0]
    # 첫 번째 '_' 또는 '_환경' / '_생육' 이전 부분
    for sep in ("_환경", "_생육", "_"):
        if sep in base:
            return base.split(sep)[0]
    return base


env_df = read_csv(env_file)
growth_df = read_csv(growth_file)

# CSV 읽기 직후 반드시 std_cols 적용
env_df = std_cols(env_df)
growth_df = std_cols(growth_df)

school_name = extract_school_name(env_file.name if env_file else (growth_file.name if growth_file else None))

st.title(f"🌱 {school_name} 극지식물 개체별 생육 차이 분석")

tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])

# ═══════════════════════════════════════════════
# Tab 1: 실험 개요
# ═══════════════════════════════════════════════
with tab1:
    try:
        st.header("📖 실험 개요")

        # 연구 배경 및 목적
        with st.expander("🔍 연구 배경 및 목적"):
            st.write("""
**극지식물(Extremophyte)** 은 건조·고온·염분 등 불리한 환경에서도 생존하는 식물입니다.
이 대시보드에서는 우리 학교 극지식물 개체별로 **생중량(가볍게 느꼈을 때의 무게, g 단위)** 이 얼마나 다른지,
그리고 **잎 수, 지상부 길이(땅 위로 나온 부분 길이), 지하부 길이(뿌리 길이)** 같은 측정 항목이
생중량과 어떤 관계가 있는지 살펴봅니다.

- **평균**: 여러 값을 더해 개수로 나눈 값. 데이터 전체의 중심 경향을 나타냅니다.
- **표준편차**: 값들이 평균에서 얼마나 흩어져 있는지 나타내는 지표. 클수록 개체 간 차이가 큽니다.
- **CV(변동계수)**: 표준편차를 평균으로 나눈 값. 평균이 다른 집단끼리 퍼짐 정도를 비교할 때 씁니다.
- **상관계수(r)**: 두 변수가 함께 어떻게 변하는지 −1에서 1 사이 숫자로 나타냅니다. 1에 가까울수록 함께 증가, −1에 가까울수록 한쪽이 증가할 때 다른 쪽은 감소합니다.
            """)

        # 데이터 요약 표
        st.subheader("📋 데이터 요약")

        # 개체 수 (생육 데이터)
        if not growth_df.empty and "개체번호" in growth_df.columns:
            n_individuals = growth_df["개체번호"].nunique()
        else:
            n_individuals = "데이터 없음"

        # 측정 기간 (환경 데이터)
        if not env_df.empty and "time" in env_df.columns:
            time_min = env_df["time"].min()
            time_max = env_df["time"].max()
            period_str = f"{time_min.strftime('%Y-%m-%d')} ~ {time_max.strftime('%Y-%m-%d')}"
        else:
            period_str = "데이터 없음"

        # 측정 항목 표시
        env_items = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]
        growth_items = [c for c in selected_growth_cols if c in growth_df.columns]

        summary_rows = []
        summary_rows.append(["총 개체 수", str(n_individuals)])
        summary_rows.append(["측정 기간", period_str])
        summary_rows.append(["환경 측정 항목", ", ".join(env_items) if env_items else "없음"])
        summary_rows.append(["생육 측정 항목", ", ".join(growth_items) if growth_items else "없음"])

        summary_df = pd.DataFrame(summary_rows, columns=["항목", "내용"])
        st.dataframe(summary_df, width="stretch", use_container_width=False)

        # 주요 지표 카드 4개
        st.subheader("📊 주요 지표")

        col1, col2, col3, col4 = st.columns(4)

        # 총 개체수
        if isinstance(n_individuals, int):
            col1.metric("총 개체수", f"{n_individuals}개체")
        else:
            col1.metric("총 개체수", "데이터 없음")

        # 평균 온도
        if "temperature" in env_df.columns:
            temp_vals = env_df["temperature"].dropna()
            if len(temp_vals) > 0:
                col2.metric("평균 온도", f"{temp_vals.mean():.1f}°C")
            else:
                col2.metric("평균 온도", "데이터 없음")
        else:
            col2.metric("평균 온도", "데이터 없음")

        # 평균 습도
        if "humidity" in env_df.columns:
            hum_vals = env_df["humidity"].dropna()
            if len(hum_vals) > 0:
                col3.metric("평균 습도", f"{hum_vals.mean():.1f}%")
            else:
                col3.metric("평균 습도", "데이터 없음")
        else:
            col3.metric("평균 습도", "데이터 없음")

        # 평균 생중량
        if "생중량" in growth_df.columns:
            gw_vals = growth_df["생중량"].dropna()
            if len(gw_vals) > 0:
                col4.metric("평균 생중량", f"{gw_vals.mean():.2f}g")
            else:
                col4.metric("평균 생중량", "데이터 없음")
        else:
            col4.metric("평균 생중량", "데이터 없음")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ═══════════════════════════════════════════════
# Tab 2: 환경 데이터
# ═══════════════════════════════════════════════
with tab2:
    try:
        st.header("🌡️ 환경 데이터")

        if env_df.empty:
            st.info("📤 사이드바에서 환경 CSV를 업로드해주세요.")
        else:
            # 환경 요약 표 (평균·최소·최대)
            st.subheader("📈 환경 요약 (평균 · 최소 · 최대)")

            env_numeric_cols = [
                c for c in ["temperature", "humidity", "ph", "ec"]
                if c in env_df.columns and pd.api.types.is_numeric_dtype(env_df[c])
            ]

            if env_numeric_cols:
                summary_data = []
                for c in env_numeric_cols:
                    s = env_df[c].dropna()
                    if len(s) > 0:
                        summary_data.append(
                            [c, f"{s.mean():.2f}", f"{s.min():.2f}", f"{s.max():.2f}", str(len(s))]
                        )
                summary_env_df = pd.DataFrame(
                    summary_data, columns=["측정 항목", "평균", "최소", "최대", "데이터 수"]
                )
                st.dataframe(summary_env_df, width="stretch", use_container_width=False)
            else:
                st.info("표시할 숫자 환경 열이 없습니다.")

            # 날짜별 시계열 (일별 평균 꺾은선)
            st.subheader("📅 날짜별 환경 변화 (일별 평균)")

            if "time" in env_df.columns:
                env_df_sorted = env_df.dropna(subset=["time"]).sort_values("time")
                env_df_sorted["day"] = env_df_sorted["time"].dt.floor("D")

                # 온도 변화
                if "temperature" in env_df_sorted.columns:
                    daily_temp = (
                        env_df_sorted.groupby("day")["temperature"]
                        .mean()
                        .dropna()
                        .reset_index()
                    )
                    if len(daily_temp) > 0:
                        fig_temp = px.line(
                            daily_temp,
                            x="day",
                            y="temperature",
                            markers=True,
                            title="일별 평균 온도 변화",
                            labels={"day": "날짜", "temperature": "온도 (°C)"},
                        )
                        fig_temp.update_layout(
                            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                            xaxis_tickformat="%m-%d",
                        )
                        st.plotly_chart(fig_temp, width="stretch")

                # 습도 변화
                if "humidity" in env_df_sorted.columns:
                    daily_hum = (
                        env_df_sorted.groupby("day")["humidity"]
                        .mean()
                        .dropna()
                        .reset_index()
                    )
                    if len(daily_hum) > 0:
                        fig_hum = px.line(
                            daily_hum,
                            x="day",
                            y="humidity",
                            markers=True,
                            title="일별 평균 습도 변화",
                            labels={"day": "날짜", "humidity": "습도 (%)"},
                        )
                        fig_hum.update_layout(
                            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                            xaxis_tickformat="%m-%d",
                        )
                        st.plotly_chart(fig_hum, width="stretch")

                # pH·EC 변화 (열이 있을 때만)
                has_ph = "ph" in env_df_sorted.columns
                has_ec = "ec" in env_df_sorted.columns

                if has_ph or has_ec:
                    fig_pe = make_subplots(
                        rows=1, cols=2, subplot_titles=("일별 평균 pH", "일별 평균 EC")
                    )

                    if has_ph:
                        daily_ph = (
                            env_df_sorted.groupby("day")["ph"]
                            .mean()
                            .dropna()
                            .reset_index()
                        )
                        if len(daily_ph) > 0:
                            fig_pe.add_trace(
                                go.Scatter(
                                    x=daily_ph["day"],
                                    y=daily_ph["ph"],
                                    mode="lines+markers",
                                    name="pH",
                                ),
                                row=1,
                                col=1,
                            )

                    if has_ec:
                        daily_ec = (
                            env_df_sorted.groupby("day")["ec"]
                            .mean()
                            .dropna()
                            .reset_index()
                        )
                        if len(daily_ec) > 0:
                            fig_pe.add_trace(
                                go.Scatter(
                                    x=daily_ec["day"],
                                    y=daily_ec["ec"],
                                    mode="lines+markers",
                                    name="EC",
                                ),
                                row=1,
                                col=2,
                            )

                    fig_pe.update_layout(
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        showlegend=False,
                    )
                    fig_pe.update_xaxes(tickformat="%m-%d")
                    st.plotly_chart(fig_pe, width="stretch")
                else:
                    st.info("pH·EC 열이 없어 이 그래프는 표시하지 않았습니다.")
            else:
                st.info("time 열이 없어 시계열을 표시할 수 없습니다.")

            # Expander: 원본 테이블 + CSV 다운로드
            with st.expander("📂 환경 데이터 원본 테이블 + CSV 다운로드"):
                st.dataframe(env_df, width="stretch", use_container_width=False)

                # CSV 다운로드 (utf-8-sig 인코딩)
                csv_bytes = env_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="⬇️ 환경 데이터 CSV 다운로드",
                    data=csv_bytes,
                    file_name=f"{school_name}_환경데이터_download.csv",
                    mime="text/csv",
                )

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ═══════════════════════════════════════════════
# Tab 3: 생육 결과
# ═══════════════════════════════════════════════
with tab3:
    try:
        st.header("📊 생육 결과")

        if growth_df.empty:
            st.info("📤 사이드바에서 생육 결과 CSV를 업로드해주세요.")
        else:
            # ── 핵심 결과 카드 ──
            st.subheader("🥇 핵심 결과")

            if "생중량" in growth_df.columns:
                gw = growth_df["생중량"].dropna()
                if len(gw) >= 1:
                    mean_gw = gw.mean()
                    max_idx = gw.idxmax()
                    min_idx = gw.idxmin()

                    # 개체번호는 원래 값 그대로 표시 (소수점 제거)
                    max_id = growth_df.loc[max_idx, "개체번호"]
                    min_id = growth_df.loc[min_idx, "개체번호"]
                    # 개체번호가 수치형이면 정수 형식으로
                    if pd.api.types.is_numeric_dtype(growth_df["개체번호"]):
                        try:
                            max_id = int(max_id)
                            min_id = int(min_id)
                        except (ValueError, TypeError):
                            pass

                    hc1, hc2, hc3 = st.columns(3)
                    hc1.metric("평균 생중량", f"{mean_gw:.2f} g")
                    hc2.metric("최대 생중량", f"{gw.max():.2f} g (개체 {max_id})")
                    hc3.metric("최소 생중량", f"{gw.min():.2f} g (개체 {min_id})")

                    # 설명
                    with st.expander("ℹ️ 지표 설명"):
                        st.write(
                            f"""
- **평균 생중량**: {len(gw)}개체의 생중량 평균값입니다.
- **최대/최소 생중량**: 가장 무거운 개체와 가장 가벼운 개체를 나타냅니다.
- 개체번호는 우리 학교 개체 고유 번호이며, 숫자(1, 2, …) 또는 문자(A1, A2, …)일 수 있습니다.
                            """
                        )
                else:
                    st.info("생중량 데이터가 없습니다.")
            else:
                st.info("생중량 열이 없어 핵심 결과 카드를 표시할 수 없습니다.")

            # ── 개체별 생육 비교 (막대 그래프) ──
            st.subheader("📊 개체별 생육 비교")

            if "개체번호" in growth_df.columns and "생중량" in growth_df.columns:
                gw_df = growth_df[["개체번호", "생중량"]].dropna().sort_values("생중량", ascending=False)

                if len(gw_df) > 0:
                    # 상위 25% / 하위 25% 구분
                    n = len(gw_df)
                    q75 = gw_df["생중량"].quantile(0.75)
                    q25 = gw_df["생중량"].quantile(0.25)

                    gw_df["group"] = "중간"
                    gw_df.loc[gw_df["생중량"] >= q75, "group"] = "상위 25%"
                    gw_df.loc[gw_df["생중량"] <= q25, "group"] = "하위 25%"

                    # 색상으로 구분
                    color_map = {"상위 25%": "#2ecc71", "중간": "#3498db", "하위 25%": "#e74c3c"}
                    colors = [color_map.get(g, "#3498db") for g in gw_df["group"]]

                    fig_bar = go.Figure()
                    fig_bar.add_trace(
                        go.Bar(
                            x=gw_df["개체번호"].astype(str),
                            y=gw_df["생중량"],
                            marker_color=colors,
                            hovertext=gw_df["group"],
                        )
                    )
                    fig_bar.update_layout(
                        title="개체별 생중량 (초록=상위 25%, 빨강=하위 25%)",
                        xaxis_title="개체번호",
                        yaxis_title="생중량 (g)",
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        height=400,
                    )
                    st.plotly_chart(fig_bar, width="stretch")

                    # 설명
                    with st.expander("ℹ️ 상위·하위 25%란?"):
                        st.write(
                            """
상위 25%는 생중량이 가장 큰 1/4 집단, 하위 25%는 가장 작은 1/4 집단을 뜻합니다.
이 경계는 3사분위수(Q3)와 1사분위수(Q1)로 정했습니다.
                        """
                        )

                    st.info(f"⭐ 상위 25% 경계: {q75:.2f}g 이상  |  하위 25% 경계: {q25:.2f}g 이하")
                else:
                    st.info("생중량 데이터가 없습니다.")
            else:
                st.info("개체번호 또는 생중량 열이 없어 막대 그래프를 표시할 수 없습니다.")

            # ── 추가 생육 지표 막대 그래프 (선택 항목) ──
            if selected_growth_cols:
                other_cols = [c for c in selected_growth_cols if c in growth_df.columns and c != "생중량"]
                if other_cols and "개체번호" in growth_df.columns:
                    st.subheader("🌿 다른 생육 지표 비교")

                    fig_other = make_subplots(
                        rows=len(other_cols),
                        cols=1,
                        subplot_titles=other_cols,
                        shared_x_axis=True,
                    )

                    for i, col_name in enumerate(other_cols, start=1):
                        sub = growth_df[["개체번호", col_name]].dropna().sort_values(col_name, ascending=False)
                        if len(sub) > 0:
                            fig_other.add_trace(
                                go.Bar(
                                    x=sub["개체번호"].astype(str),
                                    y=sub[col_name],
                                    showlegend=False,
                                ),
                                row=i,
                                col=1,
                            )

                    fig_other.update_layout(
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        height=200 * len(other_cols),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_other, width="stretch")

            # ── 생중량 분포 (박스플롯) ──
            st.subheader("📦 생중량 분포")

            if "생중량" in growth_df.columns:
                gw_box = growth_df[["생중량"]].dropna()
                if len(gw_box) >= 2:
                    fig_box = go.Figure()
                    fig_box.add_trace(
                        go.Box(
                            y=gw_box["생중량"],
                            name="생중량",
                            fillcolor="#3498db",
                        )
                    )
                    fig_box.update_layout(
                        title="개체별 생중량 분포 (박스플롯)",
                        yaxis_title="생중량 (g)",
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        height=350,
                    )
                    st.plotly_chart(fig_box, width="stretch")

                    # 통계 요약
                    gw_vals = gw_box["생중량"]
                    summary_stats = {
                        "평균": f"{gw_vals.mean():.2f} g",
                        "표준편차": f"{gw_vals.std():.2f} g",
                        "CV(변동계수)": f"{gw_vals.std() / gw_vals.mean():.2%}",
                        "최소값": f"{gw_vals.min():.2f} g",
                        "최대값": f"{gw_vals.max():.2f} g",
                        "사분위수 Q1": f"{gw_vals.quantile(0.25):.2f} g",
                        "중앙값 Q2": f"{gw_vals.median():.2f} g",
                        "사분위수 Q3": f"{gw_vals.quantile(0.75):.2f} g",
                    }
                    st.write("### 📐 생중량 통계 요약")
                    stats_df = pd.DataFrame(
                        list(summary_stats.items()), columns=["통계량", "값"]
                    )
                    st.dataframe(stats_df, width="stretch", use_container_width=False)

                    with st.expander("ℹ️ 표준편차·CV란?"):
                        st.write(
                            """
- **표준편차**: 값들이 평균에서 얼마나 떨어져 있는지 나타냅니다. 값이 클수록 개체마다 생중량 차이가 큽니다.
- **CV(변동계수)**: (표준편차 ÷ 평균) × 100%. 평균이 다른 집단끼리 비교할 때 유용한 상대 퍼짐 지표입니다.
                        """
                        )
                else:
                    st.info("생중량 데이터가 2개 미만이라 분포를 표시할 수 없습니다.")
            else:
                st.info("생중량 열이 없어 분포를 표시할 수 없습니다.")

            # ── 상관관계 분석 (산점도 2개 + 추세선) ──
            st.subheader("🔗 상관관계 분석")

            corr_pairs = [
                ("잎 수", "생중량"),
                ("지상부 길이", "생중량"),
            ]

            for x_col, y_col in corr_pairs:
                if x_col in growth_df.columns and y_col in growth_df.columns:
                    valid = growth_df[[x_col, y_col]].dropna()
                    if len(valid) >= 2:
                        corr_val = valid[x_col].corr(valid[y_col])

                        fig_scatter = px.scatter(
                            valid,
                            x=x_col,
                            y=y_col,
                            title=f"{x_col} vs {y_col}",
                            labels={x_col: x_col, y_col: f"{y_col} (g)"},
                        )

                        # 추세선 추가 (np.polyfit)
                        add_trendline(fig_scatter, valid[x_col], valid[y_col])

                        fig_scatter.update_layout(
                            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                            height=350,
                        )
                        st.plotly_chart(fig_scatter, width="stretch")

                        st.write(
                            f"**상관계수 r = {corr_val:.3f}** "
                            f"({'강한 양의 상관' if corr_val > 0.7 else '중간 양의 상관' if corr_val > 0.4 else '약한 상관' if corr_val > 0 else '약한 음의 상관' if corr_val > -0.4 else '중간 음의 상관' if corr_val > -0.7 else '강한 음의 상관'})"
                        )

                        with st.expander("ℹ️ 상관계수(r) 해석"):
                            st.write(
                                """
- r이 1에 가까우면 두 값이 함께 증가 (양의 상관)
- r이 −1에 가까우면 한쪽이 증가할 때 다른 쪽은 감소 (음의 상관)
- r이 0에 가까우면 두 값 사이에 뚜렷한 선형 관계가 없음
- **주의**: 상관은 인과관계가 아닙니다. 관계가 보인다고 해서 하나가 다른 하나의 원인이라고 단정할 수 없습니다.
                            """
                            )
                    else:
                        st.info(f"{x_col}과 {y_col} 데이터가 2개 미만이라 상관계수를 구할 수 없습니다.")
                else:
                    st.info(f"{x_col} 또는 {y_col} 열이 없어 이 산점도를 표시할 수 없습니다.")

            # ── Expander: 생육 원본 + CSV 다운로드 ──
            with st.expander("📂 생육 데이터 원본 테이블 + CSV 다운로드"):
                st.dataframe(growth_df, width="stretch", use_container_width=False)
                csv_bytes = growth_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="⬇️ 생육 데이터 CSV 다운로드",
                    data=csv_bytes,
                    file_name=f"{school_name}_생육결과_download.csv",
                    mime="text/csv",
                )

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
