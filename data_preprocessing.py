import pandas as pd
import glob
import time
from geopy.geocoders import Nominatim

def load_and_clean_data(folder_path='/content/drive/MyDrive/covid_data'):
    csv_files = glob.glob(folder_path + '/*.csv')
    df = pd.concat([pd.read_csv(file, low_memory=False) for file in csv_files], ignore_index=True)
    df.drop(columns=['DateSpecimen', 'DateResultRelease', 'DateDied', 'BarangayRes', 'BarangayPSGC', 'DateOnset', 'Pregnanttab'], inplace=True)
    df['RemovalType'] = df['RemovalType'].astype(str).str.strip().str.upper()
    return df.dropna(subset=['RemovalType', 'ProvRes'])

def geocode_provinces(df):
    geolocator = Nominatim(user_agent="geo_locator_ph")
    city_counts = df.groupby(['ProvRes', 'RemovalType']).size().unstack(fill_value=0).reset_index()
    city_coordinates = []

    for city in city_counts['ProvRes']:
        try:
            location = geolocator.geocode(f"{city}, Philippines", timeout=10)
            city_coordinates.append({"ProvRes": city, "latitude": location.latitude if location else None, "longitude": location.longitude if location else None})
        except:
            city_coordinates.append({"ProvRes": city, "latitude": None, "longitude": None})
        time.sleep(1)

    df_coords = pd.DataFrame(city_coordinates)
    return pd.merge(city_counts, df_coords, on='ProvRes').dropna(subset=['latitude', 'longitude'])
