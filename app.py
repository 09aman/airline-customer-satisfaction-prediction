
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# 1. Load the trained model and preprocessing files
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "svm_customer_satisfaction.pkl"
SCALER_PATH = BASE_DIR / "customer_satisfaction_scalar.pkl"
COLUMNS_PATH = BASE_DIR / "customer_satisfaction_columns.pkl"

st.set_page_config(
    page_title="Airline Customer Satisfaction Prediction",
    page_icon="✈️",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    expected_columns = joblib.load(COLUMNS_PATH)

    return model, scaler, expected_columns


# Check whether all required files exist
required_files = [MODEL_PATH, SCALER_PATH, COLUMNS_PATH]

missing_files = [
    path.name for path in required_files if not path.exists()
]

if missing_files:
    st.error(
        "The following required files were not found:\n\n"
        + "\n".join(f"- {name}" for name in missing_files)
    )
    st.stop()

model, scaler, expected_columns = load_artifacts()


# --------------------------------------------------
# 2. Application title and description
# --------------------------------------------------

st.title("✈️ Airline Customer Satisfaction Prediction")

st.markdown(
    """
    Enter passenger information and rate the airline's services
    to predict whether the passenger is satisfied.

    **Rating scale:** 0 = Lowest, 5 = Highest
    """
)


# --------------------------------------------------
# 3. Passenger information and service ratings
# --------------------------------------------------

with st.form("satisfaction_form"):

    st.subheader("Passenger and Travel Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=35,
        )

        flight_distance = st.number_input(
            "Flight Distance",
            min_value=1,
            max_value=20000,
            value=1000,
            step=50,
        )

        customer_type = st.selectbox(
            "Customer Type",
            ["Loyal Customer", "Disloyal Customer"],
        )

        travel_type = st.selectbox(
            "Type of Travel",
            ["Business Travel", "Personal Travel"],
        )

        travel_class = st.selectbox(
            "Travel Class",
            ["Business", "Economy (Eco)", "Economy Plus"],
        )

    with col2:
        st.subheader("Booking and Boarding")

        wifi = st.select_slider(
            "Inflight Wi-Fi Service",
            options=[0, 1, 2, 3, 4, 5],
            value=3,
        )

        time_convenience = st.select_slider(
            "Departure/Arrival Time Convenience",
            options=[0, 1, 2, 3, 4, 5],
            value=3,
        )

        online_booking = st.select_slider(
            "Ease of Online Booking",
            options=[0, 1, 2, 3, 4, 5],
            value=3,
        )

        gate_location = st.select_slider(
            "Gate Location",
            options=[0, 1, 2, 3, 4, 5],
            value=3,
        )

        online_boarding = st.select_slider(
            "Online Boarding",
            options=[0, 1, 2, 3, 4, 5],
            value=3,
        )

        checkin = st.select_slider(
            "Check-in Service",
            options=[0, 1, 2, 3, 4, 5],
            value=3,
        )

    with col3:
        st.subheader("Inflight Services")

        food = st.select_slider(
            "Food and Drink",
            options=[0, 1, 2, 3, 4, 5],
            value=3,
        )

        seat_comfort = st.select_slider(
            "Seat Comfort",
            options=[0, 1, 2, 3, 4, 5],
            value=3,
        )

        entertainment = st.select_slider(
            "Inflight Entertainment",
            options=[0, 1, 2, 3, 4, 5],
            value=3,
        )

        onboard_service = st.select_slider(
            "On-board Service",
            options=[0, 1, 2, 3, 4, 5],
            value=3,
        )

        leg_room = st.select_slider(
            "Leg Room Service",
            options=[0, 1, 2, 3, 4, 5],
            value=3,
        )

        baggage = st.select_slider(
            "Baggage Handling",
            options=[0, 1, 2, 3, 4, 5],
            value=3,
        )

        inflight_service = st.select_slider(
            "Inflight Service",
            options=[0, 1, 2, 3, 4, 5],
            value=3,
        )

        cleanliness = st.select_slider(
            "Cleanliness",
            options=[0, 1, 2, 3, 4, 5],
            value=3,
        )

    submitted = st.form_submit_button(
        "Predict Customer Satisfaction",
        type="primary",
        use_container_width=True,
    )


# --------------------------------------------------
# 4. Preprocess the input and make a prediction
# --------------------------------------------------

if submitted:

    # Match the encodings used in the notebook
    class_map = {
        "Business": 0,
        "Economy (Eco)": 1,
        "Economy Plus": 2,
    }

    customer_map = {
        "Loyal Customer": 1,
        "Disloyal Customer": 0,
    }

    travel_map = {
        "Business Travel": 1,
        "Personal Travel": 0,
    }

    # Create input features using the notebook's column names
    values = {
        "is_loyal": customer_map[customer_type],
        "Age": age,
        "is_bussiness_travel": travel_map[travel_type],
        "Class": class_map[travel_class],
        "Flight Distance": flight_distance,
        "Inflight wifi service": wifi,
        "Departure/Arrival time convenient": time_convenience,
        "Ease of Online booking": online_booking,
        "Gate location": gate_location,
        "Food and drink": food,
        "Online boarding": online_boarding,
        "Seat comfort": seat_comfort,
        "Inflight entertainment": entertainment,
        "On-board service": onboard_service,
        "Leg room service": leg_room,
        "Baggage handling": baggage,
        "Checkin service": checkin,
        "Inflight service": inflight_service,
        "Cleanliness": cleanliness,
    }

    # Arrange features in exactly the order used during training
    input_df = pd.DataFrame(
        [values],
        columns=expected_columns,
    )

    # Apply the same scaling used in the training notebook
    numeric_columns = [
        "Age",
        "Class",
        "Flight Distance",
        "Inflight wifi service",
        "Departure/Arrival time convenient",
        "Ease of Online booking",
        "Gate location",
        "Food and drink",
        "Online boarding",
        "Seat comfort",
        "Inflight entertainment",
        "On-board service",
        "Leg room service",
        "Baggage handling",
        "Checkin service",
        "Inflight service",
        "Cleanliness",
    ]

    input_df[numeric_columns] = scaler.transform(
        input_df[numeric_columns]
    )

    # Predict satisfaction
    prediction = int(model.predict(input_df)[0])

    # Display the result
    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("Prediction: Satisfied 😊")
    else:
        st.warning("Prediction: Neutral or Dissatisfied")

    st.caption(
        "This prediction is generated by a machine learning model "
        "and does not guarantee the passenger's actual satisfaction."
    )
