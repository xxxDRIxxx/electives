import plotly.express as px
import folium
import matplotlib.pyplot as plt
import seaborn as sns

def plot_map(geo_df):
    fig = px.scatter_geo(
        geo_df, lat='latitude', lon='longitude',
        size='DIED', color='ProvRes', hover_name='ProvRes',
        title='COVID-19 Deaths by Province (Philippines)'
    )
    fig.update_layout(geo=dict(scope='asia', projection_scale=7))
    fig.show()

def plot_time_series(df_clean):
    df = df_clean[df_clean['RemovalType'] == 'RECOVERED'].copy()
    df['DateRecover'] = pd.to_datetime(df['DateRecover'], errors='coerce')
    df = df.dropna(subset=['DateRecover'])
    recoveries = df.groupby('DateRecover').size().reset_index(name='recoveries')
    recoveries['cumulative'] = recoveries['recoveries'].cumsum()

    fig = px.line(recoveries, x='DateRecover', y='cumulative', title='Cumulative Recoveries Over Time')
    fig.show()

def plot_age_sex_region(df_clean):
    df = df_clean[df_clean['RemovalType'].isin(['DIED', 'RECOVERED'])]
    pivot = df.groupby(['AgeGroup', 'Sex', 'RegionRes', 'RemovalType']).size().unstack(fill_value=0).reset_index()

    grouped = pivot.groupby(['AgeGroup', 'Sex'])['DIED'].sum().unstack().fillna(0)
    grouped.plot(kind='bar', figsize=(10, 6), title='Deaths by Age Group and Sex')
    plt.ylabel("Deaths")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
