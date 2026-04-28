import pandas as pd
from pathlib import Path

REPORT_PATH = Path("reports/report.html")

def generate_html_report(df_results: pd.DataFrame):
    df_sorted = df_results.sort_values(by="f1", ascending=False)

    html = f"""
    <html>
    <head>
        <title>Model Comparison Report</title>
    </head>
    <body>
        <h1>Model Comparison Report</h1>

        <h2>Top Model</h2>
        {df_sorted.head(1).to_html(index=False)}

        <h2>All Results</h2>
        {df_sorted.to_html(index=False)}

    </body>
    </html>
    """

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w") as f:
        f.write(html)

    print(f"📄 Report saved to {REPORT_PATH}")