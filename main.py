import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@st.cache_data
def read_csv(file):
    data = file.getvalue()
    for enc in ('utf-8-sig', 'cp949'):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


def std_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(' ', '')
        if k in ('time', 'categories', '시간', '날짜', 'datetime', 'date'):
            rename[c] = 'time'
        elif k in ('temperature', '온도'):
            rename[c] = 'temperature'
        elif k in ('humidity', '습도'):
            rename[c] = 'humidity'
        elif k == 'ph':
            rename[c] = 'ph'
        elif k == 'ec':
            rename[c] = 'ec'
        elif '생중량' in k:
            rename[c] = '생중량'
        elif '잎' in k:
            rename[c] = '잎 수'
        elif '지상부' in k and '길이' in k:
            rename[c] = '지상부 길이'
        elif '지하부' in k and '길이' in k:
            rename[c] = '지하부 길이'
        elif '지상부' in k and ('질량' in k or '무게' in k):
            rename[c] = '지상부 질량'
        elif '지하부' in k and ('질량' in k or '무게' in k):
            rename[c] = '지하부 질량'
        elif '개체' in k:
            rename[c] = '개체번호'
    df = df.rename(columns=rename)
    for c in df.columns:
        if c in ('time', '개체번호'):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c].astype(str).str.replace(',', '')
                .str.extract(r'(-?\d+\.?\d*)')[0],
                errors='coerce',
            )
            if num.notna().mean() >= 0.5:
                df[c] = num
    if '생중량' not in df.columns and {'지상부 질량', '지하부 질량'} <= set(df.columns):
        df['생중량'] = df['지상부 질량'] + df['지하부 질량']
    if 'time' in df.columns:
        t = df['time'].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors='coerce', format='mixed')
        bad = parsed.isna() & t.str.match(r'^\d{8}_\d{1,2}$')
        if bad.any():
            parts = t[bad].str.split('_', expand=True)
            parsed[bad] = (
                pd.to_datetime(parts[0], format='%Y%m%d')
                + pd.to_timedelta(parts[1].astype(int), unit='h')
            )
        df['time'] = parsed
    return df


st.markdown(
    '''<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>''',
    unsafe_allow_html=True,
)

st.set_page_config(page_title='나도수영 지상-지하부 길이 분석', layout='wide')
st.title('나도수영의 지상-지하부 길이의 의미')

st.sidebar.header('데이터 업로드')
env_file = st.sidebar.file_uploader('환경 데이터 CSV', type='csv')
growth_file = st.sidebar.file_uploader('생육 결과 CSV', type='csv')

env_df = None
growth_df = None

if env_file is not None:
    with st.spinner('환경 데이터 읽는 중...'):
        raw = read_csv(env_file)
        if not raw.empty:
            env_df = std_cols(raw)

if growth_file is not None:
    with st.spinner('생육 결과 데이터 읽는 중...'):
        raw = read_csv(growth_file)
        if not raw.empty:
            growth_df = std_cols(raw)

school_name = '우리 학교'
if env_file is not None:
    fn = os.path.basename(env_file.name)
    school_name = fn.split('_')[0].strip() or school_name
elif growth_file is not None:
    fn = os.path.basename(growth_file.name)
    school_name = fn.split('_')[0].strip() or school_name


tab1, tab2, tab3 = st.tabs(
    [
        '나도수영 사진 및 설명',
        '지상-지하부 평균 비율 (원그래프)',
        '개체별 지상-지하부 길이 (누적 막대그래프)',
    ]
)


# ---- Tab 1: 나도수영 사진 및 설명 ----
with tab1:
    try:
        st.header(f'{school_name} 나도수영(Elatine) 사진 및 설명')

        st.info(
            '사진 파일을 보려면 아래 업로더에 사진을 올려 주세요. '
            '(사이드바 CSV 업로드와는 별개입니다.)'
        )
        img_file = st.file_uploader(
            '나도수영 사진 (jpg, png)', type=['jpg', 'jpeg', 'png']
        )
        if img_file is not None:
            st.image(img_file, caption=f'{school_name} 나도수영', use_container_width=True)

        st.markdown('''
**나도수영( Elatine spp.)** 에 대하여

- **분류**: 일부 분류 체계에서는 부처꽃과(Lythraceae)로, 독립 과(나도수영과 Elatinaceae)로 다루기도 하는 여러해살이 수생식물
- **서식지**: 논, 습지, 연못 가장자리 등 얕은 물이나 젖은 땅
- **형태 특징**:
  - 잎은 마주나기 또는 돌려나기, 작고 좁은 바늘 모양~타원형
  - 꽃은 매우 작고(보통 수 mm), 연한 분홍빛 또는 흰빛
  - 열매는 삭과로, 익으면 여러 개의 작은 종자로 터짐
- **생장 전략**:
  - **지상부 길이**: 수면 위 잎과 줄기를 포함한 광합성 기관의 길이, 광 포착 능력에 영향
  - **지하부 길이**: 뿌리줄기와 뿌리의 발달 정도, 고정력, 영양분 저장, 양분 흡수 능력
  - 같은 조건에서도 개체마다 지상부, 지하부 성장 비중이 다를 수 있음(유전적 변이 + 미세환경 차이)

[이번 분석의 질문]
"지상부 길이와 지하부 길이는 어떤 비율로 나타나며, 개체마다 그 비율은 어떻게 다를까?"

- 원그래프: 전체 집단의 평균 지상부, 지하부 길이 비율
- 누적 막대그래프: 개체별로 지상부, 지하부 길이를 쌓아 올려, 전체 길이와 비율을 한눈에 비교하고 가설을 검증
''')
    except Exception as e:
        st.warning(f'경고: 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}')


# ---- Tab 2: 원그래프 (지상-지하부 평균 비율) ----
with tab2:
    try:
        if growth_df is None or growth_df.empty:
            st.info(
                '지상-지하부 평균 비율을 보려면 생육 결과 CSV를 '
                '사이드바에서 업로드해 주세요.'
            )
        else:
            g = growth_df.dropna(subset=['개체번호']).copy()

            if '지상부 길이' not in g.columns or '지하부 길이' not in g.columns:
                st.info(
                    '지상부 길이와 지하부 길이 열이 모두 있어야 '
                    '원그래프를 만들 수 있어요.'
                )
            else:
                sa = g['지상부 길이'].dropna()
                sb = g['지하부 길이'].dropna()

                if len(sa) < 2 or len(sb) < 2:
                    st.info('지상부-지하부 길이 데이터가 충분하지 않아요.')
                else:
                    mean_sa = float(sa.mean())
                    mean_sb = float(sb.mean())
                    total_mean = mean_sa + mean_sb
                    pct_sa = (mean_sa / total_mean * 100) if total_mean > 0 else 0.0
                    pct_sb = (mean_sb / total_mean * 100) if total_mean > 0 else 0.0

                    st.subheader('전체 개체 평균 지상-지하부 길이 비율')
                    st.caption(
                        '※ 각 개체의 지상부 길이와 지하부 길이를 각각 평균 낸 뒤 '
                        '그 평균값들의 비율을 원그래프로 나타냈어요. '
                        '(개체별 비율의 평균이 아니라, 평균 길이의 비율이에요.)'
                    )

                    col_a, col_b = st.columns(2)
                    col_a.metric('평균 지상부 길이', f'{mean_sa:.1f} mm')
                    col_b.metric('평균 지하부 길이', f'{mean_sb:.1f} mm')

                    fig_pie = go.Figure()
                    fig_pie.add_trace(
                        go.Pie(
                            labels=['지상부 길이', '지하부 길이'],
                            values=[mean_sa, mean_sb],
                            hole=0.45,
                            marker=dict(
                                colors=['#7CB342', '#546E7A'],
                                line=dict(color='white', width=2),
                            ),
                            textinfo='label+percent',
                            hovertemplate=(
                                '<b>%{label}</b><br>'
                                '평균 길이: %{value:.1f} mm<br>'
                                '비율: %{percent:.1f}%<extra></extra>'
                            ),
                            pull=[0.08, 0.0],
                        )
                    )
                    fig_pie.update_layout(
                        title='전체 개체 평균 지상부-지하부 길이 비율',
                        height=400,
                        font=dict(
                            family='Malgun Gothic, Apple SD Gothic Neo, sans-serif'
                        ),
                        margin=dict(l=40, r=30, t=60, b=40),
                    )
                    st.plotly_chart(fig_pie, width='stretch')

                    ind_ratios = []
                    for _, row in g.iterrows():
                        a_val = row.get('지상부 길이')
                        b_val = row.get('지하부 길이')
                        if pd.notna(a_val) and pd.notna(b_val) and (a_val + b_val) > 0:
                            ind_ratios.append(float(a_val) / float(a_val + b_val) * 100)

                    if len(ind_ratios) >= 2:
                        arr = pd.Series(ind_ratios)
                        st.info(
                            '개체별 지상부 비율 분포: '
                            f'평균 {float(arr.mean()):.1f}%  '
                            f'범위 {float(arr.min()):.1f}% ~ {float(arr.max()):.1f}%  '
                            f'(표준편차 {float(arr.std()):.1f}%p)'
                        )
                    else:
                        st.info('개체별 비율 요약은 데이터가 부족해 만들 수 없어요.')

                    with st.expander('생육 데이터 원본 + CSV 다운로드'):
                        st.dataframe(g, width='stretch')
                        csv_bytes = g.to_csv(index=False).encode('utf-8-sig')
                        st.download_button(
                            label='생육 결과 CSV 다운로드',
                            data=csv_bytes,
                            file_name=f'{school_name}_생육결과데이터.csv',
                            mime='text/csv',
                        )
    except Exception as e:
        st.warning(f'경고: 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}')


# ---- Tab 3: 누적 막대그래프 (개체별 지상-지하부 길이) ----
with tab3:
    try:
        if growth_df is None or growth_df.empty:
            st.info(
                '개체별 지상-지하부 길이를 보려면 생육 결과 CSV를 '
                '사이드바에서 업로드해 주세요.'
            )
        else:
            g = growth_df.dropna(subset=['개체번호']).copy()

            need_cols = ['지상부 길이', '지하부 길이']
            available = [c for c in need_cols if c in g.columns]

            if available != need_cols:
                st.info(
                    '지상부 길이 및 지하부 길이 열이 모두 필요해요. '
                    f'현재 찾은 열: {available}'
                )
            else:
                df_plot = g[['개체번호', '지상부 길이', '지하부 길이']].dropna()
                if len(df_plot) < 2:
                    st.info('길이 데이터가 충분하지 않아요.')
                else:
                    df_plot = df_plot.sort_values(
                        '지상부 길이', ascending=True
                    ).reset_index(drop=True)

                    labels = [str(x) for x in df_plot['개체번호']]
                    top_n = len(df_plot)
                    q_idx_75 = int(np.ceil(top_n * 0.75)) - 1
                    q_idx_25 = int(np.floor(top_n * 0.25)) - 1

                    fig_stack = go.Figure()
                    fig_stack.add_trace(
                        go.Bar(
                            x=labels,
                            y=df_plot['지상부 길이'].tolist(),
                            name='지상부 길이 (mm)',
                            marker=dict(color='#7CB342', line=dict(width=0)),
                            hovertemplate=(
                                '<b>개체 %{x}</b><br>'
                                '지상부: %{y:.0f} mm<extra></extra>'
                            ),
                        )
                    )
                    fig_stack.add_trace(
                        go.Bar(
                            x=labels,
                            y=df_plot['지하부 길이'].tolist(),
                            name='지하부 길이 (mm)',
                            marker=dict(color='#546E7A', line=dict(width=0)),
                            base=df_plot['지상부 길이'].tolist(),
                            hovertemplate=(
                                '<b>개체 %{x}</b><br>'
                                '지하부: %{y:.0f} mm<br>'
                                '전체(지상+지하): %{customdata:.0f} mm<extra></extra>'
                            ),
                            customdata=(
                                df_plot['지상부 길이'].tolist()
                                + df_plot['지하부 길이'].tolist()
                            ),
                        )
                    )

                    fig_stack.update_layout(
                        barmode='stack',
                        title='개체별 지상부-지하부 길이 (쌓은 막대그래프)',
                        xaxis_title='개체번호',
                        yaxis_title='길이 (mm)',
                        height=460,
                        font=dict(
                            family='Malgun Gothic, Apple SD Gothic Neo, sans-serif'
                        ),
                        legend=dict(
                            orientation='h',
                            yanchor='bottom',
                            y=1.02,
                            xanchor='center',
                            x=0.5,
                        ),
                        margin=dict(l=50, r=30, t=60, b=40),
                    )
                    fig_stack.update_xaxes(tickmode='linear')
                    st.plotly_chart(fig_stack, width='stretch')

                    total = (df_plot['지상부 길이'] + df_plot['지하부 길이']).dropna()
                    st.caption(
                        f'※ 총 {len(df_plot)}개체의 전체 길이(지상 + 지하) 평균: '
                        f'{float(total.mean()):.1f} mm, '
                        f'범위 {float(total.min()):.0f} ~ {float(total.max()):.0f} mm'
                    )
                    st.caption(
                        '초록 막대: 지상부 길이, 회색 막대: 지하부 길이, '
                        '막대를 쌓아 올렸으므로 막대 꼭대기 = 지상 + 지하 전체 길이'
                    )

                    if top_n >= 4:
                        top_ids = [
                            str(x) for x in df_plot.iloc[q_idx_75:]['개체번호'].tolist()
                        ]
                        bot_ids = [
                            str(x) for x in df_plot.iloc[: q_idx_25 + 1]['개체번호'].tolist()
                        ]
                        st.info(
                            '지상부 길이 기준 상위 25% 개체: '
                            + ', '.join(top_ids[-5:])
                            + ' 등  '
                            + '(상위군은 지상부가 길어 광합성 면적이 넓을 가능성이 높아요)'
                        )
                        st.info(
                            '지상부 길이 기준 하위 25% 개체: '
                            + ', '.join(bot_ids[:5])
                            + ' 등  '
                            + '(하위군은 지하부 비율이 상대적으로 높을 수 있어요)'
                        )

                    with st.expander('생육 데이터 원본 + CSV 다운로드'):
                        st.dataframe(g, width='stretch')
                        csv_bytes = g.to_csv(index=False).encode('utf-8-sig')
                        st.download_button(
                            label='생육 결과 CSV 다운로드',
                            data=csv_bytes,
                            file_name=f'{school_name}_생육결과데이터.csv',
                            mime='text/csv',
                        )
    except Exception as e:
        st.warning(f'경고: 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}')
