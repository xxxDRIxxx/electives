from data_preprocessing import load_and_clean_data, geocode_provinces
from visualization import plot_map, plot_time_series, plot_age_sex_region
from modeling import train_random_forest, evaluate_model
from ui import start_ui

if __name__ == "__main__":
    print("Loading and cleaning data...")
    df_clean = load_and_clean_data()

    print("Geocoding provinces...")
    geo_df = geocode_provinces(df_clean)

    print("Plotting interactive maps...")
    plot_map(geo_df)

    print("Generating time series plot...")
    plot_time_series(df_clean)

    print("Plotting bar plots...")
    plot_age_sex_region(df_clean)

    print("Training and evaluating model...")
    train_random_forest(df_clean)
    evaluate_model()

    print("Launching interactive UI...")
    start_ui(df_clean)
