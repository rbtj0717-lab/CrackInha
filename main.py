# main.py — 바이오트래커: 극지식물 개체별 생육 차이 분석
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

st.set_page_config(page_title="바이오트래커", layout="wide")

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
            name="추세선",
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
    for sep in ("_환경", "_생육", "_"):
        if sep in base:
            return base.split(sep)[0]
    return base


env_df = read_csv(env_file)
growth_df = read_csv(growth_file)

# CSV 읽기 직후 반드시 std_cols 적용
env_df = std_cols(env_df)
growth_df = std_cols(growth_df)

school_name = extract_school_name(
    env_file.name if env_file else (growth_file.name if growth_file else None)
)

st.title(f"🧬 바이오트래커 — {school_name} 극지식물 생육 분석")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📋 연구 목적",
        "🔬 연구 내용",
        "📄 데이터 및 분석",
        "📊 분포와 관계 시각화",
    ]
)

# ═══════════════════════════════════════════════
# 탭1: 연구의 목적 및 연구 질문
# ═══════════════════════════════════════════════
with tab1:
    try:
        st.header("📋 연구 목적 및 연구 질문")

        st.subheader("🎯 연구 배경")
        st.write(
            """
**극지식물(Extremophyte)** 은 건조, 고온, 염분 등 불리한 환경에서도 살아남는 식물입니다.
이 연구에서는 우리 학교 극지식물 **개체별 생육 차이**를 분석해서,
어떤 개체가 더 잘 자랐는지, 그리고 **지상부 길이(땅 위로 나온 부분)** 와 **지하부 길이(뿌리)** 가
**생중량(가볍게 쟀을 때의 무게)** 과 어떤 관계가 있는지 살펴봅니다.
            """
        )

        st.subheader("❓ 연구 질문")
        st.write("""
**1. 식물의 생중량은 지상부 길이와 연관이 있을까?**
- 지상부(줄기·잎)가 길수록 전체 무게도 무거울 것이라고 예상할 수 있습니다.

**2. 식물의 생중량은 지하부 길이와 연관이 있을까?**
- 뿌리(지하부)가 길면 영양분과 물을 더 많이 흡수해서 생중량이 클 가능성이 있습니다.

**3. 지상부와 지하부 중 어떤 요인이 생중량에 더 큰 영향을 미칠까?**
- 두 관계를 비교해 보고, 어느 쪽의 상관계수(r)가 더 큰지 살펴봅니다.
""")

        with st.expander("📐 통계 용어 설명"):
            st.write(
                """
- **평균**: 모든 값을 더해 개수로 나눈 값. 데이터 전체의 대표값입니다.
- **표준편차**: 값들이 평균에서 얼마나 흩어져 있는지 나타내는 지표. 클수록 개체 간 차이가 큽니다.
- **상관계수(r)**: 두 변수가 함께 변하는 정도를 −1에서 1 사이 숫자로 나타냅니다.
  - 1에 가까울수록 함께 증가 (양의 상관)
  - −1에 가까울수록 한쪽 증가 시 다른 쪽은 감소 (음의 상관)
  - 0에 가까우면 뚜렷한 선형 관계가 없음
- **히스토그램**: 값의 분포를 막대 그래프로 나타낸 것. 어떤 값대에 개체가 몰려 있는지 볼 수 있습니다.
- **산점도**: 두 변수를 각각 x축·y축에 놓고 점으로 찍은 그래프. 두 값의 관계를 눈으로 확인할 수 있습니다.
                """
            )

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ═══════════════════════════════════════════════
# 탭2: 연구 내용 소개
# ═══════════════════════════════════════════════
with tab2:
    try:
        st.header("🔬 연구 내용 소개")

        st.subheader("📐 분석 방법")

        st.write("""
이 연구에서는 다음 세 가지 분석을 진행했습니다.

**① 생중량 분포 확인**
- 모든 개체의 생중량을 히스토그램으로 그려서, 무게가 어떤 범위에 모여 있는지 확인합니다.
- 평균, 표준편차, CV(변동계수)도 함께 계산합니다.

**② 지상부 길이 vs 생중량 관계 분석**
- 산점도에 추세선을 함께 그려서 두 변수의 관계를 시각화합니다.
- 상관계수(r)를 계산해 관계의 강도를 숫자로 나타냅니다.

**③ 지하부 길이 vs 생중량 관계 분석**
- 위와 동일한 방법으로 지하부 길이와 생중량의 관계를 분석합니다.
- 두 상관계수를 비교해 어느 요인이 생중량에 더 큰 영향을 주는지 판단합니다.
""")

        st.subheader("📊 데이터 개요")
        if not growth_df.empty and "생중량" in growth_df.columns:
            gw = growth_df["생중량"].dropna()
            n = len(gw)
            if n > 0:
                st.write(
                    f"""
- 총 개체 수: **{n}개체**
- 생중량 범위: **{gw.min():.2f}g ~ {gw.max():.2f}g**
- 평균 생중량: **{gw.mean():.2f}g**
- 표준편차: **{gw.std():.2f}g**
- CV(변동계수): **{gw.std()/gw.mean()*100:.1f}%**
                """
                )
        else:
            st.info("생중량 데이터가 없습니다. 사이드바에서 생육 CSV를 업로드해주세요.")

        with st.expander("ℹ️ 변동계수(CV)란?"):
            st.write(
                """
CV(변동계수) = (표준편차 ÷ 평균) × 100%
평균이 다른 집단끼리 비교할 때 유용한 상대 퍼짐 지표입니다.
값이 클수록 개체 간 차이가 크다는 뜻입니다.
            """
            )

        # 지상부/지하부 길이 데이터 유무 안내
        has_shoot = "지상부 길이" in growth_df.columns
        has_root = "지하부 길이" in growth_df.columns

        if has_shoot or has_root:
            st.subheader("🌿 측정 항목 현황")
            if has_shoot:
                shoot_vals = growth_df["지상부 길이"].dropna()
                st.write(
                    f"- 지상부 길이: {len(shoot_vals)}개체 측정됨 "
                    f"(범위: {shoot_vals.min():.1f}mm ~ {shoot_vals.max():.1f}mm)"
                )
            else:
                st.write("- 지상부 길이: 데이터에 없음")
            if has_root:
                root_vals = growth_df["지하부 길이"].dropna()
                st.write(
                    f"- 지하부 길이: {len(root_vals)}개체 측정됨 "
                    f"(범위: {root_vals.min():.1f}mm ~ {root_vals.max():.1f}mm)"
                )
            else:
                st.write("- 지하부 길이: 데이터에 없음")
        else:
            st.info("지상부 길이·지하부 길이 열이 모두 없습니다.")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ═══════════════════════════════════════════════
# 탭3: 환경 데이터 및 생육결과 데이터 원본 표 + 분석
# ═══════════════════════════════════════════════
with tab3:
    try:
        st.header("📄 데이터 및 분석")

        # ── 환경 데이터 원본 및 요약 ──
        st.subheader("🌡️ 환경 데이터")

        if env_df.empty:
            st.info("📤 사이드바에서 환경 CSV를 업로드해주세요.")
        else:
            # 환경 요약 표
            env_numeric_cols = [
                c for c in ["temperature", "humidity", "ph", "ec"]
                if c in env_df.columns and pd.api.types.is_numeric_dtype(env_df[c])
            ]

            if env_numeric_cols:
                st.write("### 환경 요약 (평균 · 최소 · 최대)")
                summary_rows = []
                for c in env_numeric_cols:
                    s = env_df[c].dropna()
                    if len(s) > 0:
                        summary_rows.append(
                            [c, f"{s.mean():.2f}", f"{s.min():.2f}", f"{s.max():.2f}", str(len(s))]
                        )
                summary_env_df = pd.DataFrame(
                    summary_rows, columns=["측정 항목", "평균", "최소", "최대", "데이터 수"]
                )
                st.dataframe(summary_env_df, width="stretch", use_container_width=False)
            else:
                st.info("표시할 숫자 환경 열이 없습니다.")

            # 환경 데이터 원본 테이블 + 다운로드
            with st.expander("📂 환경 데이터 원본 테이블 + CSV 다운로드"):
                st.dataframe(env_df, width="stretch", use_container_width=False)
                csv_bytes = env_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="⬇️ 환경 데이터 CSV 다운로드",
                    data=csv_bytes,
                    file_name=f"{school_name}_환경데이터_download.csv",
                    mime="text/csv",
                )

        # ── 생육 결과 데이터 원본 및 분석 ──
        st.subheader("🌱 생육 결과 데이터")

        if growth_df.empty:
            st.info("📤 사이드바에서 생육 결과 CSV를 업로드해주세요.")
        else:
            # 생육 데이터 원본 테이블 + 다운로드
            with st.expander("📂 생육 결과 데이터 원본 테이블 + CSV 다운로드"):
                st.dataframe(growth_df, width="stretch", use_container_width=False)
                csv_bytes = growth_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="⬇️ 생육 데이터 CSV 다운로드",
                    data=csv_bytes,
                    file_name=f"{school_name}_생육결과_download.csv",
                    mime="text/csv",
                )

            # 핵심 분석 결과
            st.write("### 📊 연구 질문에 따른 분석 결과")

            # 분석1: 지상부 길이 vs 생중량
            if "지상부 길이" in growth_df.columns and "생중량" in growth_df.columns:
                valid_shoot = growth_df[["지상부 길이", "생중량"]].dropna()
                if len(valid_shoot) >= 2:
                    r_shoot = valid_shoot["지상부 길이"].corr(valid_shoot["생중량"])
                    st.write("**① 지상부 길이 ↔ 생중량**")
                    strength = (
                        "강한 양의 상관"
                        if r_shoot > 0.7
                        else "중간 양의 상관"
                        if r_shoot > 0.4
                        else "약한 상관"
                        if r_shoot > 0
                        else "약한 음의 상관"
                        if r_shoot > -0.4
                        else "중간 음의 상관"
                        if r_shoot > -0.7
                        else "강한 음의 상관"
                    )
                    st.write(
                        f"- 상관계수 r = **{r_shoot:.3f}** ({strength})"
                    )
                    st.write(f"- 데이터 수: {len(valid_shoot)}개체")
                else:
                    st.info(
                        "지상부 길이와 생중량 데이터가 2개 미만이라 상관계수를 구할 수 없습니다."
                    )
            else:
                st.info(
                    "지상부 길이 또는 생중량 열이 없어 이 분석을 수행할 수 없습니다."
                )

            # 분석2: 지하부 길이 vs 생중량
            if "지하부 길이" in growth_df.columns and "생중량" in growth_df.columns:
                valid_root = growth_df[["지하부 길이", "생중량"]].dropna()
                if len(valid_root) >= 2:
                    r_root = valid_root["지하부 길이"].corr(valid_root["생중량"])
                    st.write("**② 지하부 길이 ↔ 생중량**")
                    strength = (
                        "강한 양의 상관"
                        if r_root > 0.7
                        else "중간 양의 상관"
                        if r_root > 0.4
                        else "약한 상관"
                        if r_root > 0
                        else "약한 음의 상관"
                        if r_root > -0.4
                        else "중간 음의 상관"
                        if r_root > -0.7
                        else "강한 음의 상관"
                    )
                    st.write(
                        f"- 상관계수 r = **{r_root:.3f}** ({strength})"
                    )
                    st.write(f"- 데이터 수: {len(valid_root)}개체")
                else:
                    st.info(
                        "지하부 길이와 생중량 데이터가 2개 미만이라 상관계수를 구할 수 없습니다."
                    )
            else:
                st.info(
                    "지하부 길이 또는 생중량 열이 없어 이 분석을 수행할 수 없습니다."
                )

            # 분석3: 어느 쪽이 더 큰 영향?
            if (
                "지상부 길이" in growth_df.columns
                and "지하부 길이" in growth_df.columns
                and "생중량" in growth_df.columns
            ):
                valid_both = growth_df[["지상부 길이", "지하부 길이", "생중량"]].dropna()
                if len(valid_both) >= 2:
                    r_shoot2 = valid_both["지상부 길이"].corr(valid_both["생중량"])
                    r_root2 = valid_both["지하부 길이"].corr(valid_both["생중량"])

                    st.write(
                        "**③ 지상부 vs 지하부 — 생중량에 더 큰 영향을 미치는 요인**"
                    )
                    abs_shoot = abs(r_shoot2)
                    abs_root = abs(r_root2)

                    if abs_shoot > abs_root:
                        diff_val = abs_shoot - abs_root
                        st.write(
                            f"""
- 지상부 길이의 |r| = **{abs_shoot:.3f}**
- 지하부 길이의 |r| = **{abs_root:.3f}**
- → **지상부 길이**가 생중량과 더 강한 관계({diff_val: .3f} 차이)를 보였습니다.
                        """
                        )
                    elif abs_root > abs_shoot:
                        diff_val = abs_root - abs_shoot
                        st.write(
                            f"""
- 지상부 길이의 |r| = **{abs_shoot:.3f}**
- 지하부 길이의 |r| = **{abs_root:.3f}**
- → **지하부 길이**가 생중량과 더 강한 관계({diff_val: .3f} 차이)를 보였습니다.
                        """
                        )
                    else:
                        st.write(
                            f"""
- 지상부 길이의 |r| = **{abs_shoot:.3f}**
- 지하부 길이의 |r| = **{abs_root:.3f}**
- → 두 요인의 관계가 비슷하게 강했습니다.
                        """
                        )

                    with st.expander("ℹ️ 해석 시 주의점"):
                        st.write(
                            """
- 상관계수는 **선형 관계**만 나타냅니다. 곡선 관계나 다른 복잡한 관계는 놓칠 수 있습니다.
- 상관이 강하다고 해서 **인과관계**가 있다고 단정할 수 없습니다.
- 데이터 수가 적으면 상관계수가 불안정할 수 있습니다.
                            """
                        )
                else:
                    st.info(
                        "세 열 모두 데이터가 2개 미만이라 비교 분석을 할 수 없습니다."
                    )

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ═══════════════════════════════════════════════
# 탭4: 히스토그램 + 산점도 시각화
# ═══════════════════════════════════════════════
with tab4:
    try:
        st.header("📊 분포와 관계 시각화")

        if growth_df.empty:
            st.info("📤 사이드바에서 생육 결과 CSV를 업로드해주세요.")
        else:
            # ── 생중량 히스토그램 ──
            if "생중량" in growth_df.columns:
                gw_vals = growth_df["생중량"].dropna()
                if len(gw_vals) > 0:
                    fig_gw_hist = px.histogram(
                        gw_vals,
                        x=gw_vals,
                        nbins=min(20, max(5, len(gw_vals) // 3)),
                        title="생중량 분포 (히스토그램)",
                        labels={"value": "생중량 (g)", "count": "개체 수"},
                    )
                    fig_gw_hist.update_layout(
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        xaxis_title="생중량 (g)",
                        yaxis_title="개체 수",
                        bargap=0.1,
                    )
                    st.plotly_chart(fig_gw_hist, width="stretch")
                else:
                    st.info("생중량 데이터가 없습니다.")
            else:
                st.info("생중량 열이 없어 히스토그램을 표시할 수 없습니다.")

            # ── 지상부 길이 히스토그램 ──
            if "지상부 길이" in growth_df.columns:
                shoot_vals = growth_df["지상부 길이"].dropna()
                if len(shoot_vals) > 0:
                    fig_shoot_hist = px.histogram(
                        shoot_vals,
                        x=shoot_vals,
                        nbins=min(20, max(5, len(shoot_vals) // 3)),
                        title="지상부 길이 분포 (히스토그램)",
                        labels={"value": "지상부 길이 (mm)", "count": "개체 수"},
                    )
                    fig_shoot_hist.update_layout(
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        xaxis_title="지상부 길이 (mm)",
                        yaxis_title="개체 수",
                        bargap=0.1,
                    )
                    st.plotly_chart(fig_shoot_hist, width="stretch")
                else:
                    st.info("지상부 길이 데이터가 없습니다.")
            else:
                st.info("지상부 길이 열이 없어 히스토그램을 표시할 수 없습니다.")

            # ── 지하부 길이 히스토그램 ──
            if "지하부 길이" in growth_df.columns:
                root_vals = growth_df["지하부 길이"].dropna()
                if len(root_vals) > 0:
                    fig_root_hist = px.histogram(
                        root_vals,
                        x=root_vals,
                        nbins=min(20, max(5, len(root_vals) // 3)),
                        title="지하부 길이 분포 (히스토그램)",
                        labels={"value": "지하부 길이 (mm)", "count": "개체 수"},
                    )
                    fig_root_hist.update_layout(
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        xaxis_title="지하부 길이 (mm)",
                        yaxis_title="개체 수",
                        bargap=0.1,
                    )
                    st.plotly_chart(fig_root_hist, width="stretch")
                else:
                    st.info("지하부 길이 데이터가 없습니다.")
            else:
                st.info("지하부 길이 열이 없어 히스토그램을 표시할 수 없습니다.")

            # ── 산점도: 지상부 길이 vs 생중량 ──
            if "지상부 길이" in growth_df.columns and "생중량" in growth_df.columns:
                valid_shoot2 = growth_df[["지상부 길이", "생중량"]].dropna()
                if len(valid_shoot2) >= 2:
                    fig_shoot_scatter = px.scatter(
                        valid_shoot2,
                        x="지상부 길이",
                        y="생중량",
                        title="지상부 길이 vs 생중량 (산점도)",
                        labels={
                            "지상부 길이": "지상부 길이 (mm)",
                            "생중량": "생중량 (g)",
                        },
                    )
                    add_trendline(
                        fig_shoot_scatter, valid_shoot2["지상부 길이"], valid_shoot2["생중량"]
                    )
                    fig_shoot_scatter.update_layout(
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        xaxis_title="지상부 길이 (mm)",
                        yaxis_title="생중량 (g)",
                        height=400,
                    )
                    st.plotly_chart(fig_shoot_scatter, width="stretch")
                else:
                    st.info(
                        "지상부 길이 vs 생중량 데이터가 2개 미만이라 산점도를 표시할 수 없습니다."
                    )
            else:
                st.info(
                    "지상부 길이 또는 생중량 열이 없어 산점도를 표시할 수 없습니다."
                )

            # ── 산점도: 지하부 길이 vs 생중량 (다른 색) ──
            if "지하부 길이" in growth_df.columns and "생중량" in growth_df.columns:
                valid_root2 = growth_df[["지하부 길이", "생중량"]].dropna()
                if len(valid_root2) >= 2:
                    fig_root_scatter = px.scatter(
                        valid_root2,
                        x="지하부 길이",
                        y="생중량",
                        title="지하부 길이 vs 생중량 (산점도)",
                        color="생중량",  # 생중량 값에 따라 다른 색깔
                        color_continuous_scale="Viridis",
                        labels={
                            "지하부 길이": "지하부 길이 (mm)",
                            "생중량": "생중량 (g)",
                        },
                    )
                    add_trendline(
                        fig_root_scatter, valid_root2["지하부 길이"], valid_root2["생중량"]
                    )
                    fig_root_scatter.update_layout(
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        xaxis_title="지하부 길이 (mm)",
                        yaxis_title="생중량 (g)",
                        height=400,
                    )
                    st.plotly_chart(fig_root_scatter, width="stretch")
                else:
                    st.info(
                        "지하부 길이 vs 생중량 데이터가 2개 미만이라 산점도를 표시할 수 없습니다."
                    )
            else:
                st.info(
                    "지하부 길이 또는 생중량 열이 없어 산점도를 표시할 수 없습니다."
                )

            # ── 두 산점도 나란히 비교 (서브플롯) ──
            has_shoot3 = (
                "지상부 길이" in growth_df.columns and "생중량" in growth_df.columns
            )
            has_root3 = (
                "지하부 길이" in growth_df.columns and "생중량" in growth_df.columns
            )

            if has_shoot3 and has_root3:
                valid_s = growth_df[["지상부 길이", "생중량"]].dropna()
                valid_r = growth_df[["지하부 길이", "생중량"]].dropna()

                if len(valid_s) >= 2 and len(valid_r) >= 2:
                    fig_compare = make_subplots(
                        rows=1,
                        cols=2,
                        subplot_titles=("지상부 길이 vs 생중량", "지하부 길이 vs 생중량"),
                    )

                    fig_compare.add_trace(
                        go.Scatter(
                            x=valid_s["지상부 길이"],
                            y=valid_s["생중량"],
                            mode="markers",
                            marker=dict(color="#2ecc71", size=8),
                            name="지상부",
                        ),
                        row=1,
                        col=1,
                    )
                    add_trendline(
                        fig_compare, valid_s["지상부 길이"], valid_s["생중량"], color="#27ae60"
                    )

                    fig_compare.add_trace(
                        go.Scatter(
                            x=valid_r["지하부 길이"],
                            y=valid_r["생중량"],
                            mode="markers",
                            marker=dict(color="#3498db", size=8),
                            name="지하부",
                        ),
                        row=1,
                        col=2,
                    )
                    add_trendline(
                        fig_compare, valid_r["지하부 길이"], valid_r["생중량"], color="#2980b9"
                    )

                    fig_compare.update_layout(
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        height=420,
                        showlegend=False,
                    )
                    fig_compare.update_xaxes(title_text="길이 (mm)")
                    fig_compare.update_yaxes(title_text="생중량 (g)")
                    st.plotly_chart(fig_compare, width="stretch")

                    st.info(
                        "⭐ 위 그래프에서 초록색(지상부)과 파란색(지하부) 추세선의 기울기를 비교해보세요. "
                        "더 가파른 쪽이 생중량에 더 큰 영향을 미치는 요인입니다."
                    )

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
