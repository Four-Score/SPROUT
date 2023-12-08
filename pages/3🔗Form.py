import streamlit as st
from utils import save_plant_data_to_string, embed_info, update_user_info

def main():
    st.title('AgriOptimize: Potted Plant Profiles')

    # Initialize session state variables if they don't exist
    if 'plants_data' not in st.session_state:
        st.session_state['plants_data'] = [{}]
    if 'current_plant' not in st.session_state:
        st.session_state['current_plant'] = 0

    # Function to check if required fields are filled
    def are_required_fields_filled(plant_data):
        return all(plant_data.get(field) for field in ['name', 'age'])

    # Function to display the form for a given plant
    def display_form(index):
        default_data = st.session_state['plants_data'][index]
        with st.form(key=f'form_{index}'):
            st.subheader(f'Plant {index + 1} Profile')

            # Plant Profile Fields (Required fields: Name, Age)
            name = st.text_input('Plant Name (Common/Botanical)', value=default_data.get('name', ''))
            age = st.text_input('Plant Age or Size (e.g., 2 years, 15 cm)', value=default_data.get('age', ''))
            date_acquired = st.date_input('Date Acquired', value=default_data.get('date_acquired', None))

            # Environmental Conditions Fields
            location = st.text_input('Plant\'s Location (e.g., indoor, balcony)', value=default_data.get('location', ''))
            light = st.text_input('Light Exposure (e.g., low, medium, high)', value=default_data.get('light', ''))
            temperature = st.text_input('Average Temperature Range (e.g., 18-25°C)', value=default_data.get('temperature', ''))

            # Care History Fields
            watering_freq = st.text_input('Watering Frequency (e.g., twice a week)', value=default_data.get('watering_freq', ''))

            # Save button
            if st.form_submit_button('Save Plant Information'):
                st.session_state['plants_data'][index] = {
                    'name': name,
                    'age': age,
                    'date_acquired': date_acquired,
                    'location': location,
                    'light': light,
                    'average temperature range': temperature,
                    'watering_freq': watering_freq
                }
                st.experimental_rerun()
                return True
            return False

    # Display the form for the current plant
    display_form(st.session_state['current_plant'])

    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state['current_plant'] > 0 and st.button('Previous Plant'):
            st.session_state['current_plant'] -= 1
            st.experimental_rerun()

    with col2:
        current_data = st.session_state['plants_data'][st.session_state['current_plant']]
        if are_required_fields_filled(current_data) and st.button('Next Plant'):
            if st.session_state['current_plant'] + 1 == len(st.session_state['plants_data']):
                st.session_state['plants_data'].append({})
            st.session_state['current_plant'] += 1
            st.experimental_rerun()

    # Submit button
    if st.button('Submit'):
        st.success('Plant data saved successfully!')
        for plant in st.session_state['plants_data']:
            st.json(plant)
        # Replace `st.session_state['plants_data']` with your actual session state variable
        plants_info = save_plant_data_to_string(st.session_state['plants_data'])
        print(plants_info)
        print(type(plants_info))

        user_id = st.session_state.user_id
        if user_id:
            with st.spinner("Processing form..."):
                tempData = {
                    "id": user_id,
                    "user_info": plants_info,
                    "info_vector": embed_info(plants_info)
                }
                update_user_info(tempData)
        else:
            st.error("Please sign up or log in first.")

if __name__ == "__main__":
    main()
