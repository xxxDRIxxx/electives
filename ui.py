from ipywidgets import VBox, HBox, Dropdown, Output
from IPython.display import display, clear_output
import matplotlib.pyplot as plt

def start_ui(df):
    metric_dropdown = Dropdown(options=['DIED', 'RECOVERED'], description='Metric:')
    regions = ['All'] + sorted(df['RegionRes'].dropna().unique())
    region_dropdown = Dropdown(options=regions, description='Region:')
    output = Output()

    def plot(metric='DIED', region='All'):
        with output:
            clear_output()
            df_filtered = df if region == 'All' else df[df['RegionRes'] == region]
            agg = df_filtered[df_filtered['RemovalType'] == metric].groupby(['AgeGroup', 'Sex']).size().unstack().fillna(0)
            agg.plot(kind='bar', figsize=(10,6), title=f'{metric} by Age Group and Sex in {region}')
            plt.grid()
            plt.tight_layout()
            plt.show()

    def on_change(change):
        plot(metric_dropdown.value, region_dropdown.value)

    metric_dropdown.observe(on_change, names='value')
    region_dropdown.observe(on_change, names='value')

    display(VBox([HBox([metric_dropdown, region_dropdown]), output]))
    plot()
