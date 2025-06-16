# Dashboard

Sample Streamlit dashboard with basic data analytics. This version includes a
dark mode toggle, a modern admin‑style theme and extra CSS animations for a richer
interactive interface.

## Requirements
The application relies on several Python packages:

- **Streamlit** – web framework used to build the dashboard UI.
- **Pandas** – data manipulation and analytics.
- **Plotly** – interactive charts and graphs.
- **NumPy** – numerical operations.
- **SciPy** – statistical utilities.
- **Scikit‑learn** – clustering algorithms used in the analysis.

All required libraries are listed in `requirements.txt`.

## Quick start
Install the dependencies and run the dashboard:

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

## Customization
A simple dark‑mode toggle is built into the app. You can tweak the colours and
animations by editing `style.css`. The file contains the CSS rules applied at
startup, including the dark‑mode overrides.
