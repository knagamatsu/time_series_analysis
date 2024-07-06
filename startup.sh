#!/bin/bash

# pagesディレクトリとPythonファイルの作成
mkdir -p pages
touch pages/{data_loading.py,data_preprocessing.py,time_series_analysis.py,modeling.py,prediction.py}

# メインのapp.pyファイルの作成
touch app.py

# requirements.txtの作成
cat << EOF > requirements.txt
streamlit
pandas
numpy<2.0
matplotlib
scikit-learn
statsmodels
prophet
EOF

# README.mdの作成
cat << EOF > README.md
# 化学工場時系列データ解析アプリケーション

このStreamlitアプリケーションは、化学工場の時系列データを解析するためのツールです。

## 機能

1. データ読み込み
2. データ前処理
3. 時系列分析
4. モデリング
5. 予測

## インストール

\`\`\`
pip install -r requirements.txt
\`\`\`

## 使用方法

\`\`\`
streamlit run app.py
\`\`\`

詳細は各ファイルのコメントを参照してください。
EOF

# ディレクトリ構造の表示
echo "プロジェクト構造:"
tree -L 2 --dirsfirst

echo -e "\nrequirements.txt の内容:"
cat requirements.txt

echo -e "\nプロジェクト構造を作成しました"