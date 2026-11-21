# main.py
import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats


# ─────────────────────────────────────────────
# 페이지 설정 (가장 먼저 호출)
# ─────────────────────────────────────────────
st.set_page_config(page_title="바이오트래커", layout="wide")

# 한글 폰트 CSS
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


# ─────────────────────────────────────────────
# CSV 읽기 함수 (인코딩 자동 처리)
# ─────────────────────────────────────────────
def read_csv(file):
    """uploaded file → DataFrame (utf-8-sig → cp949)"""
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


# ─────────────────────────────────────────────
# 열 이름 표준화 함수 (주어진 예시 그대로)
# ─────────────────────────────────────────────
def std_cols(df):
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

    # 숫자 변환 (글자 열은 계산에서 제외)
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c]
                .astype(str)
                .str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
            if num.notna().mean() >= 0.5:
                df[c] = num

    # 생중량 계산 (없을 때)
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]

    # 시간 파싱
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
# 학교명 추출 (파일명 앞부분에서 읽기)
# ─────────────────────────────────────────────
def extract_school_name(filename):
    """파일명에서 학교명 추정 (확장자·_제거)"""
    if filename is None:
        return "학교명 미상"
    base = os.path.splitext(filename)[0]  # 확장자 제거
    name = base.split("_")[0].strip()
    if name:
        return name
    return "학교명 미상"


# ─────────────────────────────────────────────
# @st.cache_data 로 CSV 로딩 (한 번만 읽기)
# ─────────────────────────────────────────────
@st.cache_data
def load_and_standardize(file, role):
    """사이드바에서 받은 파일을 읽고 std_cols 적용"""
    if file is None:
        return pd.DataFrame()
    with st.spinner(f"{role} CSV 읽는 중…"):
        df = read_csv(file)
        if df.empty:
            return pd.DataFrame()
        df = std_cols(df)
    return df


# ─────────────────────────────────────────────
# 안전한 데이터 확인 헬퍼
# ─────────────────────────────────────────────
def safe_col(df, col):
    """열이 있으면Series, 없으면 None 반환"""
    if col in df.columns:
        return df[col]
    return None


def safe_dropna_series(series):
    """Series에서 결측치 제거, 남은 값이 없으면 None"""
    if series is None:
        return None
    cleaned = series.dropna()
    if len(cleaned) < 2:
        return None
    return cleaned


def corr_with_info(x, y):
    """두 Series의 피어슨 상관계수 계산 (2개 미만이면 None)"""
    xn = safe_dropna_series(x)
    yn = safe_dropna_series(y)
    if xn is None or yn is None:
        return None
    df2 = pd.concat([xn, yn], axis=1).dropna()
    if len(df2) < 2:
        return None
    r, _p = stats.pearsonr(df2.iloc[:, 0], df2.iloc[:, 1])
    return r


# ─────────────────────────────────────────────
# 추세선 추가 헬퍼 (np.polyfit 직접 사용)
# ─────────────────────────────────────────────
def add_trendline(fig, x, y, xaxis, yaxis, color="red"):
    """산점도 fig에 추세선(선형회귀) 추가"""
    xn = x.dropna()
    yn = y.dropna()
    df2 = pd.concat([xn, yn], axis=1).dropna()
    if len(df2) < 2:
        return
    xx = df2.iloc[:, 0].values
    yy = df2.iloc[:, 1].values
    slope, intercept = np.polyfit(xx, yy, 1)
    x_line = np.array([xx.min(), xx.max()])
    y_line = slope * x_line + intercept
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            line=dict(color=color, width=2, dash="dash"),
            name="추세선",
            showlegend=True,
        ),
        row=yaxis if isinstance(yaxis, int) else 1,
        col=xaxis if isinstance(xaxis, int) else 1,
    )


# ─────────────────────────────────────────────
# 지표 설명 (고등학생용)
# ─────────────────────────────────────────────
EXPLANATIONS = {
    "평균": "모든 값을 더해 개수로 나눈 값 — 중심 경향성",
    "표준편차": "값이 평균에서 얼마나 흩어져 있는지 나타내는 지표",
    "상관계수(r)": "−1에서 +1 사이 값, 0에 가까우면 관계 약함, ±1에 가까울수록 강한 직선 관계",
    "추세선": "산점도에서 데이터의 전체적인 방향을 직선 하나로 나타낸 선 (np.polyfit 사용)",
}


# ─────────────────────────────────────────────
# 사이드바 구성
# ─────────────────────────────────────────────
st.sidebar.title("📂 데이터 업로드")
env_file = st.sidebar.file_uploader("환경 CSV", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 CSV", type="csv")

# 생육 측정 항목 선택 (생육 CSV 업로드 시에만 옵션 표시)
growth_cols = []
if growth_file is not None:
    growth_raw = read_csv(growth_file)
    growth_raw = std_cols(growth_raw)
    growth_cols = [
        c
        for c in ["지상부 길이", "지하부 길이", "생중량", "잎 수", "지상부 질량", "지하부 질량"]
        if c in growth_raw.columns
    ]

selected_growth_items = st.sidebar.multiselect(
    "📊 생육 측정 항목 선택 (분석 대상)",
    options=growth_cols,
    default=growth_cols[:3] if len(growth_cols) >= 3 else growth_cols,
)

# 로딩 상태 표시
if env_file or growth_file:
    st.sidebar.success("✅ 파일 업로드 완료 — 탭에서 확인하세요.")


# ─────────────────────────────────────────────
# 데이터 로드 (캐시 적용)
# ─────────────────────────────────────────────
env_df = load_and_standardize(env_file, "환경") if env_file else pd.DataFrame()
growth_df = load_and_standardize(growth_file, "생육") if growth_file else pd.DataFrame()

school_name_env = extract_school_name(env_file.name) if env_file else "미상"
school_name_grow = extract_school_name(growth_file.name) if growth_file else "미상"


# ─────────────────────────────────────────────
# 탭 구성
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔬 연구 목적 및 질문", "📖 연구 내용", "📊 데이터 및 분석", "📈 그래프"]
)


# ═════════════════════════════════════════════
# 탭 1: 연구 목적 및 연구 질문
# ═════════════════════════════════════════════
with tab1:
    try:
        st.header("🔬 연구 목적 및 연구 질문")

        st.subheader("연구 배경")
        st.write(
            "극지방 식물(극지식물)은 낮은 온도, 적은 수분, 강한 바람 등 "
            "극한 환경에서 살아남기 위해 독특한 생장 전략을 가지고 있습니다. "
            "같은 종이라도 개체마다 지상부(줄기·잎)와 지하부(뿌리)의 생육이 다르게 나타나며, "
            "이러한 차이가 전체 생체 무게(생중량)에 어떻게 반영되는지 알아보는 것이 중요합니다."
        )

        st.subheader("핵심 용어 설명")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**평균 (mean)**")
            st.info(EXPLANATIONS["평균"])
            st.markdown("**표준편차 (standard deviation)**")
            st.info(EXPLANATIONS["표준편차"])
        with col_b:
            st.markdown("**상관계수 (correlation coefficient, r)**")
            st.info(EXPLANATIONS["상관계수(r)"])
            st.markdown("**추세선 (trend line)**")
            st.info(EXPLANATIONS["추세선"])

        st.subheader("연구 질문")
        st.markdown(
            """1. **지상부 길이와 생중량은 어떤 관계가 있을까?**  
   지상부가 길수록 전체 생체 무게(생중량)도 늘어날까?

2. **지하부 길이와 생중량은 어떤 관계가 있을까?**  
   뿌리가 길수록 생중량이 더 많이 나갈까?

3. **지상부와 지하부 중 생중량에 더 큰 영향을 미치는 요인은 무엇일까?**  
   두 상관계수(r)의 크기를 비교하여 확인한다."""
        )

        st.subheader("데이터 출처")
        if env_file:
            st.write(f"• 환경 데이터: **{school_name_env}** 학교 CSV")
        else:
            st.info("환경 데이터를 업로드하면 학교명이 표시됩니다.")
        if growth_file:
            st.write(f"• 생육 결과 데이터: **{school_name_grow}** 학교 CSV")
        else:
            st.info("생육 결과 데이터를 업로드하면 학교명이 표시됩니다.")

        if not env_file and not growth_file:
            st.info("☝️ 사이드바에서 CSV 파일 2개를 업로드해주세요.")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ═════════════════════════════════════════════
# 탭 2: 연구 내용 소개
# ═════════════════════════════════════════════
with tab2:
    try:
        st.header("📖 연구 내용 소개")

        st.subheader("연구 설계")
        st.write(
            "본 연구는 극지식물 개체의 **지상부 길이(mm)**, **지하부 길이(mm)**, "
            "**생중량(g)** 을 측정하여 세 변수 간의 관계를 살펴보는 관찰 연구입니다."
        )

        st.subheader("사용 데이터")
        env_cols_info = []
        if not env_df.empty:
            env_cols_info = [c for c in env_df.columns if c != "time"]
        grow_cols_info = []
        if not growth_df.empty:
            grow_cols_info = [c for c in growth_df.columns if c != "개체번호"]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**환경 데이터 열**")
            if env_cols_info:
                st.write(", ".join(env_cols_info))
            else:
                st.info("환경 CSV를 업로드하면 열 목록이 표시됩니다.")
        with col2:
            st.markdown("**생육 결과 데이터 열**")
            if grow_cols_info:
                st.write(", ".join(grow_cols_info))
            else:
                st.info("생육 CSV를 업로드하면 열 목록이 표시됩니다.")

        st.subheader("분석 방법")
        st.markdown(
            """- **상관계수(r)** 로 두 변수 간의 직선 관계 강도와 방향을 확인
- **산점도(scatter plot)** 로 두 변수의 분포와 추세를 시각화
- **추세선(trend line)** 을 np.polyfit으로 직접 계산하여 산점도에 표시
- **히스토그램** 으로 생중량의 분포를 확인
- **상위·하위 개체 확인** 으로 생육 차이가 큰 개체를 식별"""
        )

        st.subheader("주의 사항")
        st.markdown(
            """- 각 학교의 CSV는 열 이름이 다를 수 있어, 분석 전 **열 이름 표준화**를 거친다.
- **개체번호**는 분석 항목이 아니라 개체 구분용 식별자이므로 그래프·상관 계산에서 제외한다.
- 센서가 측정하지 못한 빈 기간(결측치)은 계산 전 **dropna()** 로 제외한다.
- 생중량(g) 열이 없으면 **지상부 질량 + 지하부 질량** 으로 자동 계산한다."""
        )

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ═════════════════════════════════════════════
# 탭 3: 데이터 원본 표 + 분석
# ═════════════════════════════════════════════
with tab3:
    try:
        st.header("📊 데이터 원본 및 분석 결과")

        # ── 환경 데이터 원본 ──
        st.subheader("환경 데이터 원본")
        if env_df.empty:
            st.info("☝️ 사이드바에서 환경 CSV를 업로드하세요.")
        else:
            st.write(f"**{school_name_env}** 학교의 환경 데이터")
            env_display_cols = [c for c in env_df.columns]
            st.dataframe(env_df[env_display_cols], width="stretch", use_container_width=False)
            st.caption(f"행 수: {len(env_df)} · 열 수: {len(env_df.columns)}")

            # 환경 데이터 기초 통계 (숫자 열만)
            env_num_cols = [
                c for c in ["temperature", "humidity", "ph", "ec", "water_temp", "co2"]
                if c in env_df.columns
            ]
            if env_num_cols:
                st.subheader("환경 데이터 기초 통계")
                env_stats = env_df[env_num_cols].describe().round(2)
                st.dataframe(env_stats, width="stretch", use_container_width=False)
            else:
                st.info("표시할 숫자 열이 없습니다.")

        st.divider()

        # ── 생육 결과 데이터 원본 ──
        st.subheader("생육 결과 데이터 원본")
        if growth_df.empty:
            st.info("☝️ 사이드바에서 생육 결과 CSV를 업로드하세요.")
        else:
            st.write(f"**{school_name_grow}** 학교의 생육 결과 데이터")
            grow_display_cols = [c for c in growth_df.columns]
            st.dataframe(growth_df[grow_display_cols], width="stretch", use_container_width=False)
            st.caption(f"개체 수: {len(growth_df)} · 열 수: {len(growth_df.columns)}")

            # 생육 기초 통계
            grow_num_cols = [
                c
                for c in [
                    "잎 수",
                    "지상부 길이",
                    "지하부 길이",
                    "생중량",
                    "지상부 질량",
                    "지하부 질량",
                ]
                if c in growth_df.columns
            ]
            if grow_num_cols:
                st.subheader("생육 데이터 기초 통계")
                grow_stats = growth_df[grow_num_cols].describe().round(2)
                st.dataframe(grow_stats, width="stretch", use_container_width=False)

                # 분석 카드가 선택한 측정 항목만 표시되도록 필터
                if selected_growth_items:
                    analyze_cols = [c for c in selected_growth_items if c in growth_df.columns]
                else:
                    analyze_cols = grow_num_cols

                if analyze_cols:
                    st.subheader("선택 항목 기초 통계")
                    sel_stats = growth_df[analyze_cols].describe().round(2)
                    st.dataframe(sel_stats, width="stretch", use_container_width=False)

            # ── 분석 섹션 ──
            st.divider()
            st.subheader("📌 연구 질문 분석")

            # 지상부 길이 & 생중량 상관계수
            if "지상부 길이" in growth_df.columns and "생중량" in growth_df.columns:
                x_shoot = growth_df["지상부 길이"]
                y_bio = growth_df["생중량"]
                r_shoot = corr_with_info(x_shoot, y_bio)
                if r_shoot is not None:
                    st.markdown("**① 지상부 길이 ↔ 생중량 상관계수**")
                    st.metric(
                        "상관계수 r",
                        f"{r_shoot:.3f}",
                        delta=None,
                        delta_color="off",
                    )
                    st.info(
                        f"r = {r_shoot:.3f} → "
                        + (
                            "강한 양의 상관관계" if r_shoot > 0.7
                            else "중간 양의 상관관계" if r_shoot > 0.4
                            else "약한 양의 상관관계" if r_shoot > 0
                            else "음의 상관관계"
                        )
                    )
                else:
                    st.info("지상부 길이와 생중량의 상관계수를 계산할 데이터가 부족합니다.")

            # 지하부 길이 & 생중량 상관계수
            if "지하부 길이" in growth_df.columns and "생중량" in growth_df.columns:
                x_root = growth_df["지하부 길이"]
                y_bio = growth_df["생중량"]
                r_root = corr_with_info(x_root, y_bio)
                if r_root is not None:
                    st.markdown("**② 지하부 길이 ↔ 생중량 상관계수**")
                    st.metric(
                        "상관계수 r",
                        f"{r_root:.3f}",
                        delta=None,
                        delta_color="off",
                    )
                    st.info(
                        f"r = {r_root:.3f} → "
                        + (
                            "강한 양의 상관관계" if r_root > 0.7
                            else "중간 양의 상관관계" if r_root > 0.4
                            else "약한 양의 상관관계" if r_root > 0
                            else "음의 상관관계"
                        )
                    )
                else:
                    st.info("지하부 길이와 생중량의 상관계수를 계산할 데이터가 부족합니다.")

            # 지상부 vs 지하부 중 더 큰 영향 요인 비교
            if (
                "지상부 길이" in growth_df.columns
                and "지하부 길이" in growth_df.columns
                and "생중량" in growth_df.columns
            ):
                x_shoot = growth_df["지상부 길이"]
                x_root = growth_df["지하부 길이"]
                y_bio = growth_df["생중량"]
                r_shoot = corr_with_info(x_shoot, y_bio)
                r_root = corr_with_info(x_root, y_bio)
                if r_shoot is not None and r_root is not None:
                    st.markdown("**③ 지상부 vs 지하부 — 생중량에 더 큰 영향 요인**")
                    bigger = "지상부 길이" if abs(r_shoot) >= abs(r_root) else "지하부 길이"
                    smaller = "지하부 길이" if bigger == "지상부 길이" else "지상부 길이"
                    r_big = max(r_shoot, r_root, key=abs)
                    r_sml = min(r_shoot, r_root, key=abs)
                    st.write(
                        f"- **{bigger}** 의 r = {r_big:.3f} (절대값 기준 더 큼) → 생중량에 더 큰 영향을 미치는 요인"
                    )
                    st.write(f"- **{smaller}** 의 r = {r_sml:.3f}")

                    diff = abs(r_shoot) - abs(r_root)
                    st.info(
                        f"두 상관계수의 절대값 차이: {abs(diff):.3f} — "
                        + ("의미 있는 차이" if abs(diff) > 0.15 else "큰 차이 없음")
                    )
                else:
                    st.info("비교할 데이터가 충분하지 않습니다.")

            # ── CSV 다운로드 버튼 ──
            st.divider()
            st.subheader("📥 분석 결과 CSV 다운로드")
            if not growth_df.empty:
                download_df = growth_df.copy()
                csv_bytes = download_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="생육 결과 데이터 다운로드 (utf-8-sig)",
                    data=csv_bytes,
                    file_name=f"{school_name_grow}_생육결과_다운로드.csv",
                    mime="text/csv",
                )
            if not env_df.empty:
                download_env = env_df.copy()
                csv_env_bytes = download_env.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="환경 데이터 다운로드 (utf-8-sig)",
                    data=csv_env_bytes,
                    file_name=f"{school_name_env}_환경데이터_다운로드.csv",
                    mime="text/csv",
                )

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ═════════════════════════════════════════════
# 탭 4: 그래프
# ═════════════════════════════════════════════
with tab4:
    try:
        st.header("📈 생육 데이터 그래프")

        if growth_df.empty:
            st.info("☝️ 사이드바에서 생육 결과 CSV를 업로드하면 그래프가 표시됩니다.")
        else:
            has_shoot = "지상부 길이" in growth_df.columns
            has_root = "지하부 길이" in growth_df.columns
            has_bio = "생중량" in growth_df.columns

            # ── 그래프 1: 생중량 히스토그램 ──
            if has_bio:
                fig1 = go.Figure()
                fig1.add_trace(
                    go.Histogram(
                        x=growth_df["생중량"].dropna(),
                        nbinsx=max(10, min(30, len(growth_df) // 3)),
                        marker_color="forestgreen",
                        opacity=0.75,
                    )
                )
                fig1.update_layout(
                    title="생중량(g) 분포 히스토그램",
                    xaxis_title="생중량 (g)",
                    yaxis_title="빈도 (개체 수)",
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                    bargap=0.1,
                )
                st.plotly_chart(fig1, width="stretch")
                st.caption("히스토그램: 생중량 값의 분포를 막대그래프로 나타냄. "
                           "막대가 어느 구간에 몰려 있는지 보면 개체의 전형적인 생중량 범위를 알 수 있다.")

            # ── 그래프 2: 지상부 길이 vs 생중량 산점도 + 추세선 ──
            if has_shoot and has_bio:
                fig2 = go.Figure()
                df_clean = growth_df[["지상부 길이", "생중량"]].dropna()
                if len(df_clean) >= 2:
                    fig2.add_trace(
                        go.Scatter(
                            x=df_clean["지상부 길이"],
                            y=df_clean["생중량"],
                            mode="markers",
                            marker=dict(size=7, color="seagreen", opacity=0.75),
                            name="개체",
                            hovertemplate="지상부 길이: %{x:.1f} mm<br>생중량: %{y:.2f} g<extra></extra>",
                        )
                    )
                    slope, intercept = np.polyfit(
                        df_clean["지상부 길이"], df_clean["생중량"], 1
                    )
                    x_line = np.array([df_clean["지상부 길이"].min(), df_clean["지상부 길이"].max()])
                    y_line = slope * x_line + intercept
                    fig2.add_trace(
                        go.Scatter(
                            x=x_line,
                            y=y_line,
                            mode="lines",
                            line=dict(color="crimson", width=2, dash="dash"),
                            name=f"추세선 (기울기={slope:.3f})",
                        )
                    )

                    r_val = corr_with_info(df_clean["지상부 길이"], df_clean["생중량"])
                    if r_val is not None:
                        fig2.add_annotation(
                            xref="paper",
                            yref="paper",
                            x=0.03,
                            y=0.97,
                            text=f"r = {r_val:.3f}",
                            showarrow=False,
                            bgcolor="white",
                            bordercolor="gray",
                            borderwidth=1,
                            font=dict(size=13),
                        )

                fig2.update_layout(
                    title="지상부 길이(mm) vs 생중량(g) 산점도",
                    xaxis_title="지상부 길이 (mm)",
                    yaxis_title="생중량 (g)",
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                )
                st.plotly_chart(fig2, width="stretch")
                st.caption("산점도: 각 점이 하나의 개체를 나타냄. "
                           "추세선(빨간 점선)은 np.polyfit로 계산한 직선. "
                           "r 값이 1에 가까울수록 지상부가 길수록 생중량이 늘어나는 경향이 뚜렷함.")

            # ── 그래프 3: 지하부 길이 vs 생중량 산점도 + 추세선 ──
            if has_root and has_bio:
                fig3 = go.Figure()
                df_clean3 = growth_df[["지하부 길이", "생중량"]].dropna()
                if len(df_clean3) >= 2:
                    fig3.add_trace(
                        go.Scatter(
                            x=df_clean3["지하부 길이"],
                            y=df_clean3["생중량"],
                            mode="markers",
                            marker=dict(size=7, color="saddlebrown", opacity=0.75),
                            name="개체",
                            hovertemplate="지하부 길이: %{x:.1f} mm<br>생중량: %{y:.2f} g<extra></extra>",
                        )
                    )
                    slope3, intercept3 = np.polyfit(
                        df_clean3["지하부 길이"], df_clean3["생중량"], 1
                    )
                    x_line3 = np.array([df_clean3["지하부 길이"].min(), df_clean3["지하부 길이"].max()])
                    y_line3 = slope3 * x_line3 + intercept3
                    fig3.add_trace(
                        go.Scatter(
                            x=x_line3,
                            y=y_line3,
                            mode="lines",
                            line=dict(color="teal", width=2, dash="dash"),
                            name=f"추세선 (기울기={slope3:.3f})",
                        )
                    )

                    r_val3 = corr_with_info(df_clean3["지하부 길이"], df_clean3["생중량"])
                    if r_val3 is not None:
                        fig3.add_annotation(
                            xref="paper",
                            yref="paper",
                            x=0.03,
                            y=0.97,
                            text=f"r = {r_val3:.3f}",
                            showarrow=False,
                            bgcolor="white",
                            bordercolor="gray",
                            borderwidth=1,
                            font=dict(size=13),
                        )

                fig3.update_layout(
                    title="지하부 길이(mm) vs 생중량(g) 산점도",
                    xaxis_title="지하부 길이 (mm)",
                    yaxis_title="생중량 (g)",
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                )
                st.plotly_chart(fig3, width="stretch")
                st.caption("산점도: 뿌리가 길수록 생중량이 어떻게 변하는지 확인. "
                           "지하부 길이-생중량 r 값과 지상부 길이-생중량 r 값을 비교하면 어느 쪽이 생중량에 더 큰 영향을 주는지 판단할 수 있다.")

            # ── 그래프 4: 지상부 + 지하부 길이 → 생중량 통합 산점도 ──
            if has_shoot and has_root and has_bio:
                fig4 = make_subplots(
                    rows=1,
                    cols=2,
                    subplot_titles=("지상부 길이 → 생중량", "지하부 길이 → 생중량"),
                )

                df_combined = growth_df[["지상부 길이", "지하부 길이", "생중량"]].dropna()
                if len(df_combined) >= 2:
                    # 왼쪽: 지상부
                    fig4.add_trace(
                        go.Scatter(
                            x=df_combined["지상부 길이"],
                            y=df_combined["생중량"],
                            mode="markers",
                            marker=dict(size=7, color="seagreen", opacity=0.75),
                            name="지상부",
                            hovertemplate="지상부: %{x:.1f} mm<br>생중량: %{y:.2f} g<extra></extra>",
                        ),
                        row=1,
                        col=1,
                    )
                    # 오른쪽: 지하부
                    fig4.add_trace(
                        go.Scatter(
                            x=df_combined["지하부 길이"],
                            y=df_combined["생중량"],
                            mode="markers",
                            marker=dict(size=7, color="saddlebrown", opacity=0.75),
                            name="지하부",
                            hovertemplate="지하부: %{x:.1f} mm<br>생중량: %{y:.2f} g<extra></extra>",
                        ),
                        row=1,
                        col=2,
                    )

                    # 추세선 (왼쪽 - 지상부)
                    sl_s, int_s = np.polyfit(
                        df_combined["지상부 길이"], df_combined["생중량"], 1
                    )
                    x_s = np.array([df_combined["지상부 길이"].min(), df_combined["지상부 길이"].max()])
                    y_s = sl_s * x_s + int_s
                    fig4.add_trace(
                        go.Scatter(
                            x=x_s,
                            y=y_s,
                            mode="lines",
                            line=dict(color="crimson", width=2, dash="dash"),
                            name="지상부 추세선",
                            showlegend=False,
                        ),
                        row=1,
                        col=1,
                    )

                    # 추세선 (오른쪽 - 지하부)
                    sl_r, int_r = np.polyfit(
                        df_combined["지하부 길이"], df_combined["생중량"], 1
                    )
                    x_r = np.array([df_combined["지하부 길이"].min(), df_combined["지하부 길이"].max()])
                    y_r = sl_r * x_r + int_r
                    fig4.add_trace(
                        go.Scatter(
                            x=x_r,
                            y=y_r,
                            mode="lines",
                            line=dict(color="teal", width=2, dash="dash"),
                            name="지하부 추세선",
                            showlegend=False,
                        ),
                        row=1,
                        col=2,
                    )

                fig4.update_layout(
                    title="지상부·지하부 길이와 생중량의 관계 (한 화면에 비교)",
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                    height=500,
                )
                fig4.update_xaxes(title_text="길이 (mm)", row=1, col=1)
                fig4.update_xaxes(title_text="길이 (mm)", row=1, col=2)
                fig4.update_yaxes(title_text="생중량 (g)", row=1, col=1)
                fig4.update_yaxes(title_text="생중량 (g)", row=1, col=2)

                st.plotly_chart(fig4, width="stretch")
                st.caption("두 그래프를 한 화면에 나란히 배치. "
                           "왼쪽은 지상부 길이(녹색 점), 오른쪽은 지하부 길이(갈색 점)를 x축으로, "
                           "y축은 둘 다 생중량(g). "
                           "추세선의 기울기가 더 가파른 쪽이 생중량 증가에 더 큰 영향을 주는 요인이다.")

            # ── 상위·하위 개체 강조 (생중량 기준) ──
            if has_bio:
                st.subheader("🏆 생중량 상위·하위 개체")
                bio_series = growth_df["생중량"].dropna()
                if len(bio_series) >= 3:
                    top_n = min(3, len(bio_series))
                    bottom_n = min(3, len(bio_series))
                    top_ids = bio_series.nlargest(top_n)
                    bottom_ids = bio_series.nsmallest(bottom_n)

                    col_t, col_b = st.columns(2)
                    with col_t:
                        st.markdown("**생중량 상위 개체**")
                        for val, idx in top_ids.items():
                            orig_idx = growth_df["생중량"].dropna().index.get_loc(idx)
                            raw_id = growth_df.iloc[orig_idx]["개체번호"] if "개체번호" in growth_df.columns else f"#{orig_idx+1}"
                            st.metric(
                                label=f"개체 {raw_id}",
                                value=f"{val:.2f} g",
                            )
                    with col_b:
                        st.markdown("**생중량 하위 개체**")
                        for val, idx in bottom_ids.items():
                            orig_idx = growth_df["생중량"].dropna().index.get_loc(idx)
                            raw_id = growth_df.iloc[orig_idx]["개체번호"] if "개체번호" in growth_df.columns else f"#{orig_idx+1}"
                            st.metric(
                                label=f"개체 {raw_id}",
                                value=f"{val:.2f} g",
                            )
                else:
                    st.info("개체 수가 3개 미만이라 상위·하위 개체를 표시할 수 없습니다.")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ─────────────────────────────────────────────
# 하단 안내
# ─────────────────────────────────────────────
st.divider()
st.caption(
    "바이오트래커 — 극지식물 개체별 생육 차이 분석 대시보드  |  "
    "CSV 인코딩: utf-8-sig / cp949 자동 처리  |  "
    "모든 그래프는 Plotly, 데이터는 pandas로 처리"
)
