# vanwestendorp-psm-clustering-ui

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)  
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-orange)](https://streamlit.io/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Van Westendorpの価格感度モデル（PSM）をPython＋Streamlitで再現し、さらに**属性別クラスタリング分析**を組み合わせたインタラクティブアプリです。  
アップロードした調査データをもとに、ブランド別・クラスタ別の心理的価格閾値（OPP、IDP、PMC、PME）を可視化できます。

---

## 📌 特徴

- CSVをアップロードするだけで簡単に分析可能  
- ブランドごとの価格感度をグラフで直感的に把握  
- 属性情報（年齢、性別、職業、購買頻度、購入スタイルなど）に基づくクラスタリング分析  
- クラスタ別にターゲットごとの価格感度を比較可能  
- 補助線やラベル表示で、最適価格や価格受容範囲を確認可能

---

## 📌 必要環境

- Python 3.9 以上
- 必要ライブラリ：
  ```bash
  pip install streamlit pandas numpy scipy plotly scikit-learn
  ```

## 📌 使い方

1. ターミナル（PowerShellなど）でアプリを保存したフォルダに移動：
  ```bash
  cd path\to\your\folder
  ```

2. アプリを起動：
  ```bash
  streamlit run psm_clustering_ui_app.py
  ```
3. ブラウザが自動で開き、CSVファイルのアップロード画面が表示されます
4. CSVをアップロードすると、ブランド別・クラスタ別のPSM分析結果をインタラクティブに確認可能です

## 📌 CSVフォーマット例

- 前回記事の `sample_data.csv` と同様の形式  
- ブランドごとの価格項目（Too Cheap / Cheap / Expensive / Too Expensive）と属性情報（年齢、性別、職業、購買頻度、購入スタイルなど）が必要

---

## 📌 主な機能

### 1. クラスタリング＋属性別分析
- 属性情報をもとに回答者をクラスタリング（KMeans）  
- クラスタごとの特徴（年齢平均、性別構成など）を確認可能  
- 「どのターゲットがどの価格帯に敏感か」を把握

### 2. ブランド別価格分析（PSM）
- 折れ線グラフで **Too Cheap / Cheap / Expensive / Too Expensive** を表示  
- 交点から以下の価格指標を算出：
  - **OPP**：最適価格  
  - **IDP**：無関心価格  
  - **PMC**：価格受容下限  
  - **PME**：価格受容上限

### 3. フィルター機能
- 年齢、性別、職業、購買頻度、SNS利用時間、購入スタイルなどで絞り込み可能  
- 絞り込み後の対象者数も表示  
- クラスタリング対象もフィルター条件に応じて動的に変更可能

### 4. インタラクティブグラフ
- Plotlyによる折れ線グラフ  
- 補助線とラベルで心理的価格閾値を可視化  
- クラスタごとのタブ切替で比較可能

### 5. 全体集計との比較
- フィルター前・全体データでのブランド別PSM指標も表示  
- クラスタ別分析との違いを確認可能

---

## 📌 まとめ

- クラスタリング＋PSM分析により、単なる数値ではなく「どんな人がどんな価格を好むか」が具体的に把握可能  
- ターゲットごとの価格戦略やマーケティング施策に活用できる  
- 小さいクラスタの場合は参考値として扱い、解釈に注意

---

## 📌 貢献方法

- バグ報告や機能追加の提案は Issues を通じて  
- コード改善や新機能の追加は Pull Request を作成  
- ドキュメント改善や翻訳も歓迎

---

## 📌 LICENSE

MIT License（詳細はLICENSEファイルをご参照ください）

### 開発者： iwakazusuwa(Swatchp)
