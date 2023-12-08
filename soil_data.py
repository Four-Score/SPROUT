import requests
def get_soil_data(polygon_id, api_key):
    api_url = f"http://api.agromonitoring.com/agro/1.0/soil?polyid={polygon_id}&appid={api_key}"
    response = requests.get(api_url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching soil data: {response.text}")
        return None

def process_soil_data(soil_data):
    if soil_data:
        # Convert temperatures from Kelvin to Celsius
        temperature_surface_c = soil_data['t0'] - 273.15
        temperature_10cm_c = soil_data['t10'] - 273.15
        # Convert moisture to percentage
        moisture_percentage = soil_data['moisture'] * 100

        processed_data = {
            'Temperature at surface (°C)': round(temperature_surface_c, 2),
            'Temperature at 10cm depth (°C)': round(temperature_10cm_c, 2),
            'Soil moisture (%)': round(moisture_percentage, 2)
        }

        return processed_data
    else:
        return {'error':'No soil data available'}
