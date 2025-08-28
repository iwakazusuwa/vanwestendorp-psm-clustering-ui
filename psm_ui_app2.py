# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# cd C:\Users\user\iwaiwa\0801_クラスタと予測UI★\2_Zenn

# %% [markdown]
# streamlit run 1.py

# %% [markdown]
# streamlit run 0_psm_ui_app2.py

# %% [markdown]
# ＃OKです

# %%
# ------------------------
# 0. 必要ライブラリ
# ------------------------
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from scipy.interpolate import interp1d

# ------------------------
# 1. グローバル列名
# ------------------------
GLB_AGE = '年齢'
GLB_GEN = '性別'
GLB_JOB = '職業'
GLB_CHAR = 'キャラ傾向'
GLB_FREQ = '購買頻度'
GLB_STYLE = '購入スタイル'
GLB_IMPORT = '重要視すること'
GLB_SNS = 'SNS利用時間'
GLB_AVG = '平均購入単価'

# ------------------------
# 2. Streamlit設定
# ------------------------
st.set_page_config(layout="wide")
st.title("💴 Van Westendorp PSM + クラスタリング分析アプリ")

# ------------------------
# 3. 関数定義
# ------------------------
@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

def find_intersection(y1, y2, x):
    diff = np.array(y1) - np.array(y2)
    sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_change) == 0:
        return None
    i = sign_change[0]
    try:
        f = interp1d(diff[i:i+2], x[i:i+2])
        return float(f(0))
    except:
        return None

def apply_filters(df):
    """セッションステートに基づきデータをフィルタリング"""
    filtered = df.copy()

    # 数値スライダー系
    if GLB_AGE in df.columns and "selected_age_range" in st.session_state:
        filtered = filtered[filtered[GLB_AGE].between(*st.session_state["selected_age_range"])]
    if GLB_SNS in df.columns and "selected_sns" in st.session_state:
        filtered = filtered[filtered[GLB_SNS].between(*st.session_state["selected_sns"])]
    if GLB_AVG in df.columns and "selected_average_bands" in st.session_state:
        filtered = filtered[filtered[GLB_AVG].between(*st.session_state["selected_average_bands"])]

    # カテゴリマルチセレクト系
    for col, key in [(GLB_GEN,"selected_gender"), (GLB_JOB,"selected_jobs"),
                     (GLB_FREQ,"selected_frequency"), (GLB_CHAR,"selected_character"),
                     (GLB_IMPORT,"selected_importance")]:
        if col in df.columns and key in st.session_state:
            filtered = filtered[filtered[col].isin(st.session_state[key])]

    # 購入スタイル
    if GLB_STYLE in df.columns:
        style_options = df[GLB_STYLE].dropna().unique().tolist()
        selected_style = [s for s in style_options if st.session_state.get(f"selected_style_{s}", True)]
        if selected_style:
            filtered = filtered[filtered[GLB_STYLE].isin(selected_style)]

    return filtered

# ------------------------
# 4. CSVアップロード
# ------------------------
uploaded_file = st.file_uploader("📂 CSVファイルをアップロード", type=["csv"])
if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.markdown("---")
    st.markdown("#### 🔍 絞り込みフィルター")

    # ------------------------
    # 5.全フィルター解除ボタン
    # ------------------------
    if st.button("🚿 全フィルター解除"):
        if GLB_AGE in df.columns:
            st.session_state["selected_age_range"] = (int(df[GLB_AGE].min()), int(df[GLB_AGE].max()))
        if GLB_SNS in df.columns:
            st.session_state["selected_sns"] = (int(df[GLB_SNS].min()), int(df[GLB_SNS].max()))
        if GLB_AVG in df.columns:
            st.session_state["selected_average_bands"] = (int(df[GLB_AVG].min()), int(df[GLB_AVG].max()))
        for col, key in [(GLB_GEN,"selected_gender"), (GLB_JOB,"selected_jobs"),
                         (GLB_FREQ,"selected_frequency"), (GLB_CHAR,"selected_character"),
                         (GLB_IMPORT,"selected_importance")]:
            if col in df.columns:
                st.session_state[key] = df[col].dropna().unique().tolist()
        if GLB_STYLE in df.columns:
            for s in df[GLB_STYLE].dropna().unique().tolist():
                st.session_state[f"selected_style_{s}"] = True
        #st.experimental_rerun()

    # ------------------------
    # 6. フィルターUI
    # ------------------------
    col1, col2, col3 = st.columns(3)

    # col1: 年齢・性別・職業・平均購入単価・SNS
    with col1:
        if GLB_AGE in df.columns:
            st.session_state.setdefault("selected_age_range", (int(df[GLB_AGE].min()), int(df[GLB_AGE].max())))
            st.session_state["selected_age_range"] = st.slider(f"🔍 {GLB_AGE}",
                                                               int(df[GLB_AGE].min()),
                                                               int(df[GLB_AGE].max()),
                                                               st.session_state["selected_age_range"])
        if GLB_GEN in df.columns:
            st.session_state.setdefault("selected_gender", df[GLB_GEN].dropna().unique().tolist())
            st.session_state["selected_gender"] = st.multiselect(f"🔍 {GLB_GEN}",
                                                                 df[GLB_GEN].dropna().unique().tolist(),
                                                                 default=st.session_state["selected_gender"])
        if GLB_JOB in df.columns:
            st.session_state.setdefault("selected_jobs", df[GLB_JOB].dropna().unique().tolist())
            st.session_state["selected_jobs"] = st.multiselect(f"🔍 {GLB_JOB}",
                                                               df[GLB_JOB].dropna().unique().tolist(),
                                                               default=st.session_state["selected_jobs"])
        if GLB_AVG in df.columns:
            st.session_state.setdefault("selected_average_bands", (int(df[GLB_AVG].min()), int(df[GLB_AVG].max())))
            st.session_state["selected_average_bands"] = st.slider(f"🔍 {GLB_AVG}",
                                                                    int(df[GLB_AVG].min()),
                                                                    int(df[GLB_AVG].max()),
                                                                    st.session_state["selected_average_bands"])
        if GLB_SNS in df.columns:
            st.session_state.setdefault("selected_sns", (int(df[GLB_SNS].min()), int(df[GLB_SNS].max())))
            st.session_state["selected_sns"] = st.slider(f"🔍 {GLB_SNS}",
                                                          int(df[GLB_SNS].min()),
                                                          int(df[GLB_SNS].max()),
                                                          st.session_state["selected_sns"])

    # col2: キャラ傾向・重要視すること・購買頻度
    with col2:
        for col, key, label in [(GLB_CHAR,"selected_character",GLB_CHAR),
                                (GLB_IMPORT,"selected_importance",GLB_IMPORT),
                                (GLB_FREQ,"selected_frequency",GLB_FREQ)]:
            if col in df.columns:
                st.session_state.setdefault(key, df[col].dropna().unique().tolist())
                st.session_state[key] = st.multiselect(f"🔍 {label}", df[col].dropna().unique().tolist(),
                                                       default=st.session_state[key])


    # %%
    # col3: 購入スタイル
    with col3:
        if GLB_STYLE in df.columns:
            style_options = df[GLB_STYLE].dropna().unique().tolist()
            st.markdown(f"🔍 {GLB_STYLE}")
            for s in style_options:
                st.session_state.setdefault(f"selected_style_{s}", True)
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ 全て選択", key="style_all_select"):
                    for s in style_options: st.session_state[f"selected_style_{s}"] = True
            with c2:
                if st.button("❌ 全て解除", key="style_all_clear"):
                    for s in style_options: st.session_state[f"selected_style_{s}"] = False
            # 個別チェックボックス
            selected_style = []
            for s in style_options:
                key_name = f"selected_style_{s}"
                checked = st.checkbox(s, key=key_name)
                if checked:
                    selected_style.append(s)


    # ------------------------
    # 7.フィルター適用
    # ------------------------
    st.markdown("---")
    filtered_df = apply_filters(df)
    st.markdown(f"#### <フィルター後の対象者数: {len(filtered_df)} 人>")

    # ------------------------
    # 8.KMeansクラスタリング設定
    # ------------------------
    st.markdown("### 🧩 クラスタリング設定")
    candidate_features = [GLB_AGE, GLB_GEN,GLB_JOB,GLB_FREQ,GLB_AVG, GLB_STYLE, GLB_IMPORT]
     #candidate_features = ['年齢', '性別','職業','購買頻度','平均購入単価', '購入スタイル', '重要視すること']
    selected_features = st.multiselect("クラスタリングに使う属性を選択してください",
                                       candidate_features, default=candidate_features)

    if len(selected_features) == 0:
        st.warning("少なくとも1つ以上の特徴量を選択してください。")
    else:
        cluster_count = st.slider("クラスタ数 (K)", 2, 10, 3)
        X = filtered_df[selected_features].copy()
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        X_scaled = StandardScaler().fit_transform(X)
        kmeans = KMeans(n_clusters=cluster_count, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        filtered_df['cluster'] = clusters

        # クラスタ概要表示
        st.markdown(f"#### クラスタリング結果（K={cluster_count}）")
        st.write(filtered_df[['ID'] + selected_features + ['cluster']])

        # クラスタ別サマリ
        num_clusters = filtered_df['cluster'].nunique()
        num_features = [GLB_AGE, GLB_SNS, GLB_AVG]
        cat_features = [GLB_GEN, GLB_JOB, GLB_FREQ,GLB_IMPORT,GLB_STYLE, GLB_CHAR]
        #num_features = ['年齢', 'SNS利用時間', '平均購入単価']
        #at_features = ['性別', '職業', '購買頻度',"重要視すること","購入スタイル", 'キャラ傾向']

        rows = []
        for c in range(num_clusters):
            cluster_df = filtered_df[filtered_df['cluster']==c]
            row = {"クラスタ":c, "人数":len(cluster_df)}
            for f in num_features:
                if f in cluster_df.columns:
                    row[f"{f}平均"] = round(cluster_df[f].mean(),2)
            for f in cat_features:
                if f in cluster_df.columns:
                    top_val = cluster_df[f].value_counts(normalize=True).idxmax()
                    top_ratio = cluster_df[f].value_counts(normalize=True).max()
                    row[f"{f}（最多）"] = f"{top_val} ({top_ratio:.1%})"
            rows.append(row)
        summary_df = pd.DataFrame(rows)
        st.dataframe(summary_df)


        # %%
        # ------------------------
        # 9.クラスタ別 PSM分析
        # ------------------------
        st.markdown("---")
        st.markdown("#### 📊 クラスタ別 Van Westendorp PSM分析")

        show_lines = st.checkbox("📊 指標の補助線とラベルを表示/非表示", value=True, key="show_lines_checkbox")
               
        labels = ['too_cheap', 'cheap', 'expensive', 'too_expensive']
        brands = sorted(set(col.split('_')[0] for col in df.columns if any(lbl in col for lbl in labels)))
        cluster_tabs = st.tabs([f"Cluster {i}" for i in range(cluster_count)])

        for cluster_id, tab in zip(range(cluster_count), cluster_tabs):
            with tab:
                st.markdown(f"##### Cluster {cluster_id} の分析")
                df_cluster = filtered_df[filtered_df['cluster']==cluster_id]
                results = []

                for brand in brands:
                    brand_cols = [f"{brand}_{lbl}" for lbl in labels if f"{brand}_{lbl}" in df_cluster.columns]
                    df_brand = df_cluster[df_cluster[brand_cols].notnull().any(axis=1)]
                    if df_brand.empty:
                        st.warning(f"{brand} のデータがありません。")
                        continue

                    price_data = {label: df_brand[f"{brand}_{label}"].dropna().astype(int).values
                                  for label in labels if f"{brand}_{label}" in df_brand.columns}
                    valid_arrays = [arr for arr in price_data.values() if len(arr) > 0]
                    if not valid_arrays:
                        st.warning("有効な価格データがありません。")
                        continue

                    all_prices = np.arange(min(np.concatenate(valid_arrays)),
                                           max(np.concatenate(valid_arrays))+1000, 100)
                    n = len(df_brand)
                    cumulative = {
                        'too_cheap':[np.sum(price_data.get('too_cheap',[])>=p)/n for p in all_prices],
                        'cheap':[np.sum(price_data.get('cheap',[])>=p)/n for p in all_prices],
                        'expensive':[np.sum(price_data.get('expensive',[])<=p)/n for p in all_prices],
                        'too_expensive':[np.sum(price_data.get('too_expensive',[])<=p)/n for p in all_prices]
                    }

                    opp = find_intersection(cumulative['cheap'], cumulative['expensive'], all_prices)
                    idp = find_intersection(cumulative['too_cheap'], cumulative['too_expensive'], all_prices)
                    pme = find_intersection(cumulative['cheap'], cumulative['too_expensive'], all_prices)
                    pmc = find_intersection(cumulative['expensive'], cumulative['too_cheap'], all_prices)

                    # グラフ生成
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=all_prices, y=np.array(cumulative['too_cheap'])*100,
                                             name='Too Cheap', line=dict(color='blue')))
                    fig.add_trace(go.Scatter(x=all_prices, y=np.array(cumulative['cheap'])*100,
                                             name='Cheap', line=dict(color='green')))
                    fig.add_trace(go.Scatter(x=all_prices, y=np.array(cumulative['expensive'])*100,
                                             name='Expensive', line=dict(color='orange')))
                    fig.add_trace(go.Scatter(x=all_prices, y=np.array(cumulative['too_expensive'])*100,
                                             name='Too Expensive', line=dict(color='red')))

                    for val, name, color in zip([opp,idp,pme,pmc],
                                                ['OPP','IDP','PME','PMC'],
                                                ['purple','black','magenta','cyan']):
                        if val and show_lines:  # ←ここでチェック
                            fig.add_vline(x=val, line_dash='dash', line_color=color)
                            fig.add_annotation(x=val, y=50, text=name, showarrow=False, textangle=90,
                                               font=dict(color=color,size=12), bgcolor='white')

                    fig.update_layout(title=f"{brand} - PSM分析",
                                      xaxis_title="価格（円)",
                                      yaxis_title="累積比率（%）",
                                      height=400,
                                      hovermode="x unified",
                                      xaxis=dict(tickformat='d'))

                    # 表示
                    col_plot, col_info = st.columns([3,1])
                    with col_plot:
                        st.plotly_chart(fig, use_container_width=True)
                    with col_info:
                        st.markdown(f"**{brand} の該当人数：{df_brand.shape[0]}人**")
                        st.write(f"📌 OPP（最適価格）: {round(opp) if opp else '計算不可'} 円")
                        st.write(f"📌 IDP（無関心価格）: {round(idp) if idp else '計算不可'} 円")
                        st.write(f"📌 PMC（下限）: {round(pmc) if pmc else '計算不可'} 円")
                        st.write(f"📌 PME（上限）: {round(pme) if pme else '計算不可'} 円")

                    results.append({"ブランド":brand,"該当人数":df_brand.shape[0],
                                    "最適価格（OPP）":round(opp) if opp else None,
                                    "無関心価格（IDP）":round(idp) if idp else None,
                                    "価格受容範囲下限（PMC）":round(pmc) if pmc else None,
                                    "価格受容範囲上限（PME）":round(pme) if pme else None})

                # クラスタ内全ブランド表
                if results:
                    st.markdown("---")
                    st.markdown("#### クラスタ内ブランド別 PSM指標一覧")
                    df_result = pd.DataFrame(results)
                    st.dataframe(df_result.style.format({col:"{:.0f}" for col in df_result.columns if col!="ブランド"}))

        # ------------------------
        # 10.フィルター前ブランド別PSM集計（全体）
        # ------------------------
        results_before_filter = []
        for brand in brands:
            brand_cols = [f"{brand}_{lbl}" for lbl in labels if f"{brand}_{lbl}" in df.columns]
            df_brand = df[df[brand_cols].notnull().any(axis=1)]
            if df_brand.empty:
                continue

            price_data = {label: df_brand[f"{brand}_{label}"].dropna().astype(int).values
                          for label in labels if f"{brand}_{label}" in df_brand.columns}
            valid_arrays = [arr for arr in price_data.values() if len(arr) > 0]
            if not valid_arrays:
                continue

            all_prices = np.arange(min(np.concatenate(valid_arrays)),
                                   max(np.concatenate(valid_arrays))+1000, 100)
            n = len(df_brand)
            cumulative = {
                'too_cheap':[np.sum(price_data.get('too_cheap',[])>=p)/n for p in all_prices],
                'cheap':[np.sum(price_data.get('cheap',[])>=p)/n for p in all_prices],
                'expensive':[np.sum(price_data.get('expensive',[])<=p)/n for p in all_prices],
                'too_expensive':[np.sum(price_data.get('too_expensive',[])<=p)/n for p in all_prices]
            }

            opp = find_intersection(cumulative['cheap'], cumulative['expensive'], all_prices)
            idp = find_intersection(cumulative['too_cheap'], cumulative['too_expensive'], all_prices)
            pme = find_intersection(cumulative['cheap'], cumulative['too_expensive'], all_prices)
            pmc = find_intersection(cumulative['expensive'], cumulative['too_cheap'], all_prices)

            results_before_filter.append({"ブランド":brand,"OPP":opp,"IDP":idp,"PMC":pmc,"PME":pme})

        st.markdown("---")
        with st.expander("📋 フィルター前 ブランド別 PSM 指標一覧（全体集計）", expanded=False):
            st.markdown(f"**全体調査人数：{len(df)}人**")
            summary_df_before = pd.DataFrame(results_before_filter)
            st.dataframe(summary_df_before.style.format({col:"{:.0f}" for col in summary_df_before.columns if col!="ブランド"}))


# %%

# %%

# %%

# %%

# %% [markdown]
# 内容に不備あり
#
#
# # ------------------------
# # 0. 必要ライブラリ
# # ------------------------
# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.cluster import KMeans
# import plotly.graph_objects as go
# from scipy.interpolate import interp1d
#
# # ------------------------
# # グローバル列名設定
# # ------------------------
# glb_age = '年齢'
# glb_gen = '性別'
# glb_wok = '職業'
# glb_cha = 'キャラ傾向'
# glb_pur = '購買頻度'
# glb_sty = '購入スタイル'
# glb_imp = '重要視すること'
# glb_sns = 'SNS利用時間'
# glb_ave = '平均購入単価'
#
# # ------------------------
# # Streamlit UI設定
# # ------------------------
# st.set_page_config(layout="wide")
# st.title("💴 Van Westendorp PSM + クラスタリング分析アプリ")
#
# # ------------------------
# # 関数定義
# # ------------------------
# @st.cache_data
# def load_data(uploaded_file):
#     return pd.read_csv(uploaded_file)
#
# def find_intersection(y1, y2, x):
#     diff = np.array(y1) - np.array(y2)
#     sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
#     if len(sign_change) == 0:
#         return None
#     i = sign_change[0]
#     try:
#         f = interp1d(diff[i:i+2], x[i:i+2])
#         return float(f(0))
#     except:
#         return None
#
# def apply_filters(df):
#     filtered = df.copy()
#
#     # 数値スライダー系
#     if glb_age in df.columns and "selected_age_range" in st.session_state:
#         filtered = filtered[filtered[glb_age].between(*st.session_state["selected_age_range"])]
#     if glb_sns in df.columns and "selected_sns" in st.session_state:
#         filtered = filtered[filtered[glb_sns].between(*st.session_state["selected_sns"])]
#     if glb_ave in df.columns and "selected_average_bands" in st.session_state:
#         filtered = filtered[filtered[glb_ave].between(*st.session_state["selected_average_bands"])]
#
#     # カテゴリマルチセレクト系
#     if glb_gen in df.columns and "selected_gender" in st.session_state:
#         filtered = filtered[filtered[glb_gen].isin(st.session_state["selected_gender"])]
#     if glb_wok in df.columns and "selected_jobs" in st.session_state:
#         filtered = filtered[filtered[glb_wok].isin(st.session_state["selected_jobs"])]
#     if glb_pur in df.columns and "selected_frequency" in st.session_state:
#         filtered = filtered[filtered[glb_pur].isin(st.session_state["selected_frequency"])]
#     if glb_cha in df.columns and "selected_character" in st.session_state:
#         filtered = filtered[filtered[glb_cha].isin(st.session_state["selected_character"])]
#     if glb_imp in df.columns and "selected_importance" in st.session_state:
#         filtered = filtered[filtered[glb_imp].isin(st.session_state["selected_importance"])]
#
#     # 購入スタイル
#     if glb_sty in df.columns:
#         style_options = df[glb_sty].dropna().unique().tolist()
#         selected_style = [s for s in style_options if st.session_state.get(f"selected_style_{s}", True)]
#         if selected_style:
#             filtered = filtered[filtered[glb_sty].isin(selected_style)]
#
#     return filtered
#
# # ------------------------
# # CSVアップロード
# # ------------------------
# uploaded_file = st.file_uploader("📂 CSVファイルをアップロード", type=["csv"])
#
# if uploaded_file is not None:
#     df = load_data(uploaded_file)
#     st.markdown("#### 🔍 絞り込みフィルター")
#
#     # 全フィルター解除ボタン
#     if st.button("🚿 全フィルター解除"):
#         if glb_age in df.columns:
#             st.session_state["selected_age_range"] = (int(df[glb_age].min()), int(df[glb_age].max()))
#         if glb_sns in df.columns:
#             st.session_state["selected_sns"] = (int(df[glb_sns].min()), int(df[glb_sns].max()))
#         if glb_ave in df.columns:
#             st.session_state["selected_average_bands"] = (int(df[glb_ave].min()), int(df[glb_ave].max()))
#         if glb_gen in df.columns:
#             st.session_state["selected_gender"] = df[glb_gen].dropna().unique().tolist()
#         if glb_wok in df.columns:
#             st.session_state["selected_jobs"] = df[glb_wok].dropna().unique().tolist()
#         if glb_pur in df.columns:
#             st.session_state["selected_frequency"] = df[glb_pur].dropna().unique().tolist()
#         if glb_cha in df.columns:
#             st.session_state["selected_character"] = df[glb_cha].dropna().unique().tolist()
#         if glb_imp in df.columns:
#             st.session_state["selected_importance"] = df[glb_imp].dropna().unique().tolist()
#         if glb_sty in df.columns:
#             for s in df[glb_sty].dropna().unique().tolist():
#                 st.session_state[f"selected_style_{s}"] = True
#         st.experimental_rerun()
#
#     # ------------------------
#     # フィルターUI
#     # ------------------------
#     col1, col2, col3 = st.columns(3)
#
#     # col1: 年齢・性別・職業・平均購入単価・SNS
#     with col1:
#         if glb_age in df.columns:
#             st.session_state.setdefault("selected_age_range", (int(df[glb_age].min()), int(df[glb_age].max())))
#             st.session_state["selected_age_range"] = st.slider(f"🔍 {glb_age}", int(df[glb_age].min()), int(df[glb_age].max()), st.session_state["selected_age_range"])
#         if glb_gen in df.columns:
#             st.session_state.setdefault("selected_gender", df[glb_gen].dropna().unique().tolist())
#             st.session_state["selected_gender"] = st.multiselect(f"🔍 {glb_gen}", df[glb_gen].dropna().unique().tolist(), default=st.session_state["selected_gender"])
#         if glb_wok in df.columns:
#             st.session_state.setdefault("selected_jobs", df[glb_wok].dropna().unique().tolist())
#             st.session_state["selected_jobs"] = st.multiselect(f"🔍 {glb_wok}", df[glb_wok].dropna().unique().tolist(), default=st.session_state["selected_jobs"])
#         if glb_ave in df.columns:
#             st.session_state.setdefault("selected_average_bands", (int(df[glb_ave].min()), int(df[glb_ave].max())))
#             st.session_state["selected_average_bands"] = st.slider(f"🔍 {glb_ave}", int(df[glb_ave].min()), int(df[glb_ave].max()), st.session_state["selected_average_bands"])
#         if glb_sns in df.columns:
#             st.session_state.setdefault("selected_sns", (int(df[glb_sns].min()), int(df[glb_sns].max())))
#             st.session_state["selected_sns"] = st.slider(f"🔍 {glb_sns}", int(df[glb_sns].min()), int(df[glb_sns].max()), st.session_state["selected_sns"])
#
#     # col2: キャラ傾向・重要視すること・購買頻度
#     with col2:
#         if glb_cha in df.columns:
#             st.session_state.setdefault("selected_character", df[glb_cha].dropna().unique().tolist())
#             st.session_state["selected_character"] = st.multiselect(f"🔍 {glb_cha}", df[glb_cha].dropna().unique().tolist(), default=st.session_state["selected_character"])
#         if glb_imp in df.columns:
#             st.session_state.setdefault("selected_importance", df[glb_imp].dropna().unique().tolist())
#             st.session_state["selected_importance"] = st.multiselect(f"🔍 {glb_imp}", df[glb_imp].dropna().unique().tolist(), default=st.session_state["selected_importance"])
#         if glb_pur in df.columns:
#             st.session_state.setdefault("selected_frequency", df[glb_pur].dropna().unique().tolist())
#             st.session_state["selected_frequency"] = st.multiselect(f"🔍 {glb_pur}", df[glb_pur].dropna().unique().tolist(), default=st.session_state["selected_frequency"])
#
#     # col3: 購入スタイル
#     with col3:
#         if glb_sty in df.columns:
#             style_options = df[glb_sty].dropna().unique().tolist()
#             st.markdown(f'🔍 {glb_sty}')
#         
#             # セッションステート初期化
#             for s in style_options:
#                 key_name = f"selected_style_{s}"
#                 if key_name not in st.session_state:
#                     st.session_state[key_name] = True  # 初期は全選択状態
#         
#             colA, colB = st.columns(2)
#             with colA:
#                 if st.button("✅ 全て選択"):
#                     for s in style_options:
#                         st.session_state[f"selected_style_{s}"] = True
#             with colB:
#                 if st.button("❌ 全て解除"):
#                     for s in style_options:
#                         st.session_state[f"selected_style_{s}"] = False
#         
#             # チェックボックス
#             selected_style = []
#             for s in style_options:
#                 key_name = f"selected_style_{s}"
#                 checked = st.checkbox(s, key=key_name)  # value= は不要
#                 if checked:
#                     selected_style.append(s)
#         else:
#             selected_style = None
#
#     # ------------------------
#     # フィルタ適用
#     # ------------------------
#     filtered_df = apply_filters(df)
#     st.markdown(f"#### <フィルター後の対象者数: {len(filtered_df)} 人>")
#
#     # ------------------------
#     # KMeansクラスタリング
#     # ------------------------
#     st.markdown("### 🧩 クラスタリング設定")
#     candidate_features = [glb_age, glb_gen,glb_wok,glb_pur,glb_ave, glb_sty, glb_imp]
#     selected_features = st.multiselect("クラスタリングに使う属性を選択してください", candidate_features, default=candidate_features)
#
#     if selected_features and len(filtered_df) > 0:
#         cluster_count = st.slider("クラスタ数 (K)", 2, 10, 3)
#         X = filtered_df[selected_features].copy()
#         for col in X.columns:
#             if X[col].dtype == 'object' or X[col].dtype.name == 'category':
#                 le = LabelEncoder()
#                 X[col] = le.fit_transform(X[col].astype(str))
#         X_scaled = StandardScaler().fit_transform(X)
#         kmeans = KMeans(n_clusters=cluster_count, random_state=42)
#         clusters = kmeans.fit_predict(X_scaled)
#         filtered_df['cluster'] = clusters
#
#         # クラスタ概要
#         st.markdown(f"#### クラスタリング結果（K={cluster_count}）")
#         st.write(filtered_df[['ID'] + selected_features + ['cluster']])
#
#         # クラスタ別サマリ
#         num_clusters = filtered_df['cluster'].nunique()
#         num_features = [glb_age, glb_sns, glb_ave]
#         cat_features = [glb_gen, glb_wok, glb_pur,glb_imp,glb_sty, glb_cha]
#         rows = []
#         for c in range(num_clusters):
#             cluster_df = filtered_df[filtered_df['cluster']==c]
#             row = {"クラスタ":c, "人数":len(cluster_df)}
#             for f in num_features:
#                 if f in cluster_df.columns:
#                     row[f"{f}平均"] = round(cluster_df[f].mean(),2)
#             for f in cat_features:
#                 if f in cluster_df.columns:
#                     top_val = cluster_df[f].value_counts(normalize=True).idxmax()
#                     top_ratio = cluster_df[f].value_counts(normalize=True).max()
#                     row[f"{f}（最多）"] = f"{top_val} ({top_ratio:.1%})"
#             rows.append(row)
#         st.dataframe(pd.DataFrame(rows))
#
#         # ------------------------
#         # クラスタ別 PSM分析
#         # ------------------------
#         st.markdown("#### 📊 クラスタ別 Van Westendorp PSM分析")
#         labels = ['too_cheap', 'cheap', 'expensive', 'too_expensive']
#         brands = sorted(set(col.split('_')[0] for col in df.columns if any(lbl in col for lbl in labels)))
#         cluster_tabs = st.tabs([f"Cluster {i}" for i in range(cluster_count)])
#
#         for cluster_id, tab in zip(range(cluster_count), cluster_tabs):
#             with tab:
#                 st.markdown(f"##### Cluster {cluster_id} の分析")
#                 df_cluster = filtered_df[filtered_df['cluster']==cluster_id]
#                 results = []
#                 for brand in brands:
#                     brand_cols = [f"{brand}_{lbl}" for lbl in labels if f"{brand}_{lbl}" in df_cluster.columns]
#                     df_brand = df_cluster[df_cluster[brand_cols].notnull().any(axis=1)]
#                     if df_brand.empty:
#                         continue
#                     price_data = {label: df_brand[f"{brand}_{label}"].dropna().astype(int).values for label in labels if f"{brand}_{label}" in df_brand.columns}
#                     valid_arrays = [arr for arr in price_data.values() if len(arr)>0]
#                     if not valid_arrays: continue
#                     all_prices = np.arange(min(np.concatenate(valid_arrays)), max(np.concatenate(valid_arrays))+1000, 100)
#                     n = len(df_brand)
#                     cumulative = {
#                         'too_cheap':[np.sum(price_data.get('too_cheap',[])>=p)/n for p in all_prices],
#                         'cheap':[np.sum(price_data.get('cheap',[])>=p)/n for p in all_prices],
#                         'expensive':[np.sum(price_data.get('expensive',[])<=p)/n for p in all_prices],
#                         'too_expensive':[np.sum(price_data.get('too_expensive',[])<=p)/n for p in all_prices]
#                     }
#                     opp = find_intersection(cumulative['cheap'], cumulative['expensive'], all_prices)
#                     idp = find_intersection(cumulative['too_cheap'], cumulative['too_expensive'], all_prices)
#                     pme = find_intersection(cumulative['cheap'], cumulative['too_expensive'], all_prices)
#                     pmc = find_intersection(cumulative['expensive'], cumulative['too_cheap'], all_prices)
#
#                     # グラフ描画
#                     fig = go.Figure()
#                     fig.add_trace(go.Scatter(x=all_prices, y=np.array(cumulative['too_cheap'])*100, name='Too Cheap', line=dict(color='blue')))
#                     fig.add_trace(go.Scatter(x=all_prices, y=np.array(cumulative['cheap'])*100, name='Cheap', line=dict(color='green')))
#                     fig.add_trace(go.Scatter(x=all_prices, y=np.array(cumulative['expensive'])*100, name='Expensive', line=dict(color='orange')))
#                     fig.add_trace(go.Scatter(x=all_prices, y=np.array(cumulative['too_expensive'])*100, name='Too Expensive', line=dict(color='red')))
#                     for val, name, color in zip([opp,idp,pme,pmc], ['OPP','IDP','PME','PMC'], ['purple','black','magenta','cyan']):
#                         if val:
#                             fig.add_vline(x=val, line_dash='dash', line_color=color)
#                             fig.add_annotation(x=val, y=50, text=name, showarrow=False, textangle=90, font=dict(color=color,size=12), bgcolor='white')
#                     fig.update_layout(title=f"{brand} - PSM分析", xaxis_title="価格（円）", yaxis_title="累積比率（%）", height=400, hovermode="x unified", xaxis=dict(tickformat='d'))
#                     st.plotly_chart(fig, use_container_width=True)
#
#                     results.append({"ブランド":brand,"該当人数":df_brand.shape[0],"OPP":round(opp) if opp else None,"IDP":round(idp) if idp else None,"PMC":round(pmc) if pmc else None,"PME":round(pme) if pme else None})
#
#                 if results:
#                     st.markdown("#### クラスタ内ブランド別 PSM指標一覧")
#                     df_result = pd.DataFrame(results)
#                     st.dataframe(df_result.style.format({col:"{:.0f}" for col in ["OPP","IDP","PMC","PME"]}))
# else:
#     st.info("CSVファイルをアップロードしてください。")
