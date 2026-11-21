import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ─────────────────────────────────────────────
# 데이터 로딩
# ─────────────────────────────────────────────
@st.cache_data
def read_csv(file):
    """utf-8-sig → cp949 순서로 시도, 실패 시 빈 DataFrame 반환"""
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


def std_cols(df):
    """학교마다 다른 열 이름을 표준 이름으로 통일"""
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

    # 숫자 열이 아닌 측정 항목은 숫자로 변환 시도
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c].astype(str).str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
            if num.notna().mean() >= 0.5:
                df[c] = num

    # 생중량이 없고 지상부+지하부 질량이 있으면 계산
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]

    # time 파싱 (날짜_시각 형식도 처리)
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = (
                pd.to_datetime(parts[0], format="%Y%m%d")
                + pd.to_timedelta(parts[1].astype(int), unit="h")
            )
        df["time"] = parsed

    return df


# ─────────────────────────────────────────────
# 공통 스타일
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

st.set_page_config(page_title="극지식물 생육 분석 대시보드", layout="wide")
st.title("🌱 우리 학교 극지식물 개체별 생육 차이 분석")
st.caption("CSV 파일을 업로드하면 개체별 생육 차이와 환경 조건을 한눈에 비교할 수 있어요.")


# ─────────────────────────────────────────────
# 사이드바
# ─────────────────────────────────────────────
st.sidebar.header("📁 데이터 업로드")

env_file = st.sidebar.file_uploader("환경 데이터 CSV", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 데이터 CSV", type="csv")

env_df = None
growth_df = None

if env_file is not None:
    with st.spinner("환경 데이터를 읽는 중…"):
        raw_env = read_csv(env_file)
        if not raw_env.empty:
            env_df = std_cols(raw_env)

if growth_file is not None:
    with st.spinner("생육 데이터를 읽는 중…"):
        raw_growth = read_csv(growth_file)
        if not raw_growth.empty:
            growth_df = std_cols(raw_growth)

# 학교명 추출 (파일명 앞부분)
school_name = "우리 학교"
if env_file is not None:
    fn = os.path.basename(env_file.name)
    school_name = fn.split("_")[0].strip() or school_name
elif growth_file is not None:
    fn = os.path.basename(growth_file.name)
    school_name = fn.split("_")[0].strip() or school_name


# ─────────────────────────────────────────────
# 탭 구성
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])


# ═════════════════════════════════════════════
#  보조 함수
# ═════════════════════════════════════════════
def safe_mean(series):
    """결측치 제거 후 평균 반환 (값이 2개 미만이면 None)"""
    s = series.dropna()
    if len(s) < 2:
        return None
    return float(s.mean())


def plot_bar_subplots(g, cols_spec, colors_map):
    """개체별 수평 막대 그래프 여러 개를 서브플롯으로 표시"""
    n = len(cols_spec)
    rows_n = (n + 1) // 2
    fig = make_subplots(
        rows=rows_n,
        cols=2 if n > 1 else 1,
        subplot_titles=[f"평균 {name} (개체별)" for name in cols_spec],
        horizontal_spacing=0.12,
    )
    for i, col_name in enumerate(cols_spec):
        d = g[["개체번호", col_name]].dropna().sort_values(col_name, ascending=True)
        if len(d) < 2:
            continue
        labels = [str(x) for x in d["개체번호"]]
        vals = d[col_name].tolist()
        base = "#64B5F6"
        top_c = colors_map.get("top", "#2E7D32")
        bot_c = colors_map.get("bot", "#B71C1C")
        bar_colors = [base] * len(vals)
        if len(vals) >= 4:
            q75 = int(np.ceil(len(vals) * 0.75)) - 1
            q25 = int(np.floor(len(vals) * 0.25)) - 1
            for j in range(q75, len(vals)):
                bar_colors[j] = top_c
            for j in range(0, q25 + 1):
                bar_colors[j] = bot_c
        r = i // 2 + 1
        c_col = i % 2 + 1
        fig.add_trace(
            go.Bar(
                y=labels,
                x=vals,
                orientation="h",
                marker_color=bar_colors,
                marker_line_color="white",
                name=col_name,
            ),
            row=r,
            col=c_col,
        )
    fig.update_layout(
        height=400 + 160 * max(rows_n - 1, 0),
        showlegend=False,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        margin=dict(l=100, r=30, t=50, b=40),
    )
    return fig


# ═════════════════════════════════════════════
# Tab 1 — 실험 개요
# ═════════════════════════════════════════════
with tab1:
    try:
        st.header(f"🧪 {school_name} 극지식물 실험 개요")

        st.markdown("""
**극지환경 모의 시설**에서 재배한 극지식물의 개체별 생육 차이를 분석해요.

- **연구 배경**: 같은 환경에서도 개체마다 생장 속도와 최종 크기가 달라요. 그 차이를 숫자로 비교해 봐요.
- **분석 목표**:
  1. 개체별 생중량 분포와 차이 (평균·표준편차·CV)
  2. 잎 수·지상부 길이와 생중량의 관계 (상관계수·산점도)
  3. 생중량 상위 25% vs 하위 25% 개체 비교
""")

        if growth_df is not None and not growth_df.empty:
            grow_cols = [
                c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량"]
                if c in growth_df.columns
            ]
            n_individuals = len(growth_df)

            period_str = "—"
            if env_df is not None and "time" in env_df.columns:
                t_min = env_df["time"].dropna().min()
                t_max = env_df["time"].dropna().max()
                if pd.notna(t_min) and pd.notna(t_max):
                    period_str = f"{t_min.strftime('%Y-%m-%d')} ~ {t_max.strftime('%Y-%m-%d')}"

            rows = []
            rows.append({
                "항목": "총 개체 수",
                "값": f"{n_individuals}개체",
                "설명": "생육 결과를 측정한 극지식물 개체 수",
            })
            rows.append({
                "항목": "측정 기간",
                "값": period_str,
                "설명": "환경 센서 기록 기간",
            })
            rows.append({
                "항목": "생육 측정 항목",
                "값": ", ".join(grow_cols) if grow_cols else "없음",
                "설명": "잎 수, 지상부·지하부 길이, 생중량 등",
            })
            st.subheader("📋 데이터 요약")
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        # 주요 지표 카드 4개
        st.subheader("📊 주요 지표")
        col1, col2, col3, col4 = st.columns(4)

        if growth_df is not None and not growth_df.empty:
            col1.metric("총 개체 수", f"{len(growth_df)}개체")
        else:
            col1.metric("총 개체 수", "—")

        if env_df is not None and "temperature" in env_df.columns:
            m = safe_mean(env_df["temperature"])
            col2.metric("평균 온도", f"{m:.1f}℃" if m is not None else "—")
        else:
            col2.metric("평균 온도", "—")

        if env_df is not None and "humidity" in env_df.columns:
            m = safe_mean(env_df["humidity"])
            col3.metric("평균 습도", f"{m:.1f}%" if m is not None else "—")
        else:
            col3.metric("평균 습도", "—")

        if growth_df is not None and "생중량" in growth_df.columns:
            m = safe_mean(growth_df["생중량"])
            col4.metric("평균 생중량", f"{m:.2f}g" if m is not None else "—")
        else:
            col4.metric("평균 생중량", "—")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ═════════════════════════════════════════════
# Tab 2 — 환경 데이터
# ═════════════════════════════════════════════
with tab2:
    try:
        if env_df is None or env_df.empty:
            st.info("🌡️ 환경 데이터 CSV를 사이드바에서 업로드해 주세요.")
        else:
            st.header(f"🌡️ {school_name} 환경 데이터")

            # 환경 요약 표
            env_num_cols = [
                c for c in ["temperature", "humidity", "ph", "ec", "water_temp", "co2"]
                if c in env_df.columns
            ]
            if env_num_cols:
                s = env_df[env_num_cols].dropna()
                if len(s) >= 2:
                    unit_map = {
                        "temperature": "℃", "humidity": "%",
                        "ph": "", "ec": "mS/cm",
                        "water_temp": "℃", "co2": "ppm",
                    }
                    env_summary = pd.DataFrame({
                        "측정 항목": list(env_num_cols),
                        "평균": [round(float(s[c].mean()), 2) for c in env_num_cols],
                        "최소": [round(float(s[c].min()), 2) for c in env_num_cols],
                        "최대": [round(float(s[c].max()), 2) for c in env_num_cols],
                        "단위": [unit_map.get(c, "") for c in env_num_cols],
                    })
                    st.subheader("📋 환경 요약 (평균 · 최소 · 최대)")
                    st.dataframe(env_summary, width="stretch", hide_index=True)
                else:
                    st.info("환경 데이터가 충분하지 않아 요약을 만들 수 없어요.")

            # 일별 시계열 계산
            env_t = env_df.dropna(subset=["time"]).copy()
            if len(env_t) >= 2:
                env_t["day"] = env_t["time"].dt.floor("D")
                ts_cols = [c for c in ["temperature", "humidity"] if c in env_df.columns]
                ts_extra = [c for c in ["ph", "ec"] if c in env_df.columns]
                day_cols = ts_cols + ts_extra

                if day_cols:
                    day_data = env_t.groupby("day")[day_cols].mean().reset_index()

                    if day_data.shape[0] >= 2:
                        # 온도·습도 서브플롯
                        fig_ts = make_subplots(
                            rows=2, cols=1,
                            shared_xaxes=True,
                            subplot_titles=("일별 평균 온도 변화", "일별 평균 습도 변화"),
                            vertical_spacing=0.12,
                        )
                        if "temperature" in day_data.columns:
                            fig_ts.add_trace(
                                go.Scatter(
                                    x=day_data["day"],
                                    y=day_data["temperature"],
                                    mode="lines+markers",
                                    name="온도",
                                    line=dict(color="#FF6F00", width=2),
                                    marker=dict(size=4),
                                ),
                                row=1, col=1,
                            )
                        if "humidity" in day_data.columns:
                            fig_ts.add_trace(
                                go.Scatter(
                                    x=day_data["day"],
                                    y=day_data["humidity"],
                                    mode="lines+markers",
                                    name="습도",
                                    line=dict(color="#1565C0", width=2),
                                    marker=dict(size=4),
                                ),
                                row=2, col=1,
                            )
                        fig_ts.update_layout(
                            height=460,
                            showlegend=False,
                            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                            margin=dict(l=50, r=30, t=50, b=40),
                        )
                        fig_ts.update_xaxes(title_text="날짜", row=2, col=1)
                        st.plotly_chart(fig_ts, width="stretch")

                        # pH·EC 서브플롯 (열이 있을 때만)
                        if ts_extra:
                            fig_ph = make_subplots(
                                rows=2, cols=1,
                                shared_xaxes=True,
                                subplot_titles=("일별 평균 pH 변화", "일별 평균 EC 변화"),
                                vertical_spacing=0.12,
                            )
                            if "ph" in day_data.columns:
                                fig_ph.add_trace(
                                    go.Scatter(
                                        x=day_data["day"],
                                        y=day_data["ph"],
                                        mode="lines+markers",
                                        name="pH",
                                        line=dict(color="#6A1B9A", width=2),
                                        marker=dict(size=4),
                                    ),
                                    row=1, col=1,
                                )
                            if "ec" in day_data.columns:
                                fig_ph.add_trace(
                                    go.Scatter(
                                        x=day_data["day"],
                                        y=day_data["ec"],
                                        mode="lines+markers",
                                        name="EC",
                                        line=dict(color="#2E7D32", width=2),
                                        marker=dict(size=4),
                                    ),
                                    row=2, col=1,
                                )
                            fig_ph.update_layout(
                                height=400,
                                showlegend=False,
                                font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                                margin=dict(l=50, r=30, t=50, b=40),
                            )
                            fig_ph.update_xaxes(title_text="날짜", row=2, col=1)
                            st.plotly_chart(fig_ph, width="stretch")
                    else:
                        st.info("날짜별 데이터가 충분하지 않아 시계열 그래프를 만들 수 없어요.")

            # 원본 테이블 + CSV 다운로드
            with st.expander("🔍 환경 데이터 원본 테이블 보기"):
                st.dataframe(env_df, width="stretch")
                csv_bytes = env_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="📥 환경 데이터 CSV 다운로드",
                    data=csv_bytes,
                    file_name=f"{school_name}_환경데이터.csv",
                    mime="text/csv",
                )

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ═════════════════════════════════════════════
# Tab 3 — 생육 결과
# ═════════════════════════════════════════════
with tab3:
    try:
        if growth_df is None or growth_df.empty:
            st.info("📊 생육 결과 CSV를 사이드바에서 업로드해 주세요.")
        else:
            st.header(f"📊 {school_name} 생육 결과 분석")
            g = growth_df.dropna(subset=["개체번호"]).copy()

            # ── 🥇 핵심 결과 카드 ──
            st.subheader("🥇 핵심 결과")
            if "생중량" in g.columns:
                gw = g["생중량"].dropna()
                if len(gw) >= 2:
                    mean_gw = float(gw.mean())
                    max_row = g.loc[gw.idxmax()]
                    min_row = g.loc[gw.idxmin()]
                    std_gw = float(gw.std())
                    cv_gw = (std_gw / mean_gw * 100) if mean_gw > 0 else 0.0

                    c1, c2, c3 = st.columns(3)
                    c1.metric("평균 생중량", f"{mean_gw:.2f}g")
                    c2.metric("최대 생중량", f"{float(gw.max()):.2f}g",
                              f"개체 {max_row['개체번호']}")
                    c3.metric("최소 생중량", f"{float(gw.min()):.2f}g",
                              f"개체 {min_row['개체번호']}")
                    st.info(
                        f"📐 변동계수(CV) = {cv_gw:.1f}%   "
                        f"(표준편차 ÷ 평균 × 100 — 값이 클수록 개체 간 차이가 크다는 뜻)"
                    )

            # ── 개체별 생육 비교 (막대 그래프 서브플롯) ──
            st.subheader("📈 개체별 생육 비교")
            bar_cols = [
                "생중량",
                "잎 수",
                "지상부 길이",
                "지하부 길이",
            ]
            bar_available = [c for c in bar_cols if c in g.columns]
            if bar_available:
                fig_bar = plot_bar_subplots(
                    g,
                    bar_available,
                    {"top": "#2E7D32", "bot": "#B71C1C"},
                )
                st.plotly_chart(fig_bar, width="stretch")
                st.caption(
                    "🟦 파란색: 중간 개체   🟩 초록색: 상위 25%   🟥 빨간색: 하위 25%"
                )
            else:
                st.info("생육 측정 항목이 없어 막대 그래프를 만들 수 없어요.")

            # ── 생중량 분포 (박스플롯) ──
            st.subheader("📦 생중량 분포")
            if "생중량" in g.columns:
                gw = g["생중량"].dropna()
                if len(gw) >= 3:
                    fig_box = go.Figure()
                    fig_box.add_trace(
                        go.Box(
                            y=gw.tolist(),
                            name="생중량(g)",
                            fillcolor="#26A69A",
                            line_color="#00695C",
                            boxpoints="outliers",
                            marker=dict(color="#00695C", size=6),
                        )
                    )
                    fig_box.add_hline(
                        y=float(gw.mean()),
                        line_dash="dot",
                        line_color="#EF5350",
                        annotation_text="평균",
                        annotation_position="right",
                    )
                    fig_box.update_layout(
                        title="생중량 분포 (박스플롯)",
                        yaxis_title="생중량 (g)",
                        height=340,
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        margin=dict(l=50, r=30, t=50, b=40),
                    )
                    st.plotly_chart(fig_box, width="stretch")
                else:
                    st.info("생중량 데이터가 충분하지 않아 박스플롯을 만들 수 없어요.")
            else:
                st.info("생중량 열이 없어 박스플롯을 만들 수 없어요.")

            # ── 상관관계 분석 (산점도 + 추세선) ──
            st.subheader("🔗 상관관계 분석 (산점도 + 추세선)")
            corr_pairs = [
                ("잎 수", "생중량"),
                ("지상부 길이", "생중량"),
            ]
            corr_available = [
                (x, y) for x, y in corr_pairs
                if x in g.columns and y in g.columns
            ]

            if corr_available:
                for x_col, y_col in corr_available:
                    xd = g[[x_col, y_col]].dropna()
                    if len(xd) >= 3:
                        r = float(np.corrcoef(xd[x_col], xd[y_col])[0, 1])
                        fig_scatter = go.Figure()
                        fig_scatter.add_trace(
                            go.Scatter(
                                x=xd[x_col],
                                y=xd[y_col],
                                mode="markers",
                                marker=dict(
                                    size=9,
                                    color="#1565C0",
                                    opacity=0.7,
                                    line=dict(width=1, color="white"),
                                ),
                                name="개체",
                                hovertemplate=(
                                    f"{x_col}: %{{x:.1f}}<br>"
                                    f"{y_col}: %{{y:.2f}}<extra></extra>"
                                ),
                            )
                        )
                        xs = xd[x_col].dropna()
                        ys = xd[y_col].dropna()
                        if len(xs) >= 2:
                            coeffs = np.polyfit(xs, ys, 1)
                            xs_sorted = np.sort(xs)
                            ys_fit = np.polyval(coeffs, xs_sorted)
                            fig_scatter.add_trace(
                                go.Scatter(
                                    x=xs_sorted,
                                    y=ys_fit,
                                    mode="lines",
                                    line=dict(color="#EF5350", width=2, dash="dot"),
                                    name=f"추세선 (r = {r:.2f})",
                                )
                            )
                        fig_scatter.update_layout(
                            title=f"{x_col} vs {y_col}  (상관계수 r = {r:.2f})",
                            xaxis_title=x_col,
                            yaxis_title=y_col,
                            height=380,
                            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                            margin=dict(l=50, r=30, t=50, b=40),
                        )
                        st.plotly_chart(fig_scatter, width="stretch")
                        strength = (
                            "강한" if abs(r) >= 0.7
                            else ("중간" if abs(r) >= 0.4 else "약한")
                        )
                        direction = "양" if r > 0 else "음"
                        st.caption(
                            f"📐 r = {r:.2f} → {direction}의 {strength} 상관관계  "
                            f"(1에 가까울수록 함께 증가, -1에 가까울수록 함께 감소, "
                            f"0에 가까우면 관계가 약함)"
                        )
                    else:
                        st.info(
                            f"{x_col}과 {y_col} 데이터가 충분하지 않아 산점도를 만들 수 없어요."
                        )
            else:
                st.info("상관관계 분석에 필요한 생육 항목이 없어요.")

            # ── 원본 테이블 + CSV 다운로드 ──
            with st.expander("🔍 생육 결과 데이터 원본 테이블 보기"):
                st.dataframe(g, width="stretch")
                csv_bytes = g.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="📥 생육 결과 CSV 다운로드",
                    data=csv_bytes,
                    file_name=f"{school_name}_생육결과데이터.csv",
                    mime="text/csv",
                )

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
