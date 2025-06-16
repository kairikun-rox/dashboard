from flask import Flask, render_template, request
import io
import json
import plotly.express as px
import plotly.utils
from data_processing import process_data, BusinessAnalyzer

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    metrics = None
    graphJSON = None
    if request.method == 'POST':
        sales = request.files.get('sales')
        material = request.files.get('material')
        if sales and material:
            sales_bytes = sales.read()
            material_bytes = material.read()
            sales_df, daily_sales, sy, sm = process_data(sales_bytes, sales.filename, 'sales')
            mat_df, daily_mat, my, mm = process_data(material_bytes, material.filename, 'material')
            if sales_df is not None and mat_df is not None:
                analyzer = BusinessAnalyzer(sales_df, daily_sales, mat_df, daily_mat)
                metrics = analyzer.get_summary_metrics()
                fig = px.line(daily_sales, x='日付', y='総売上額', title='日別売上推移')
                graphJSON = fig.to_json()
    return render_template('index.html', metrics=metrics, graphJSON=graphJSON)

if __name__ == '__main__':
    app.run(debug=True)
