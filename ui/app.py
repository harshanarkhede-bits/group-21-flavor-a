import sys
from pathlib import Path

import streamlit as st
import requests
from datetime import date, time

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT))


from src.serving.locations import ALLOWED_LOCATIONS





# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000/predict"


# ============================================================
# ALLOWED LOCATIONS
# ============================================================

LOCATIONS = ALLOWED_LOCATIONS


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ETA Prediction",
    page_icon="🚗",
    layout="centered"
)


# ============================================================
# PAGE TITLE
# ============================================================

st.title("🚗 ETA Prediction")

st.write(
    "Enter your pickup and drop details to estimate "
    "the expected delivery/trip time."
)


# ============================================================
# LOCATION INPUTS
# ============================================================

st.subheader("Trip Details")

pickup_location = st.selectbox(
    "Pickup Location",
    options=LOCATIONS,
    index=None,
    placeholder="Select pickup location"
)


# ------------------------------------------------------------
# Drop location
# ------------------------------------------------------------

drop_location = st.selectbox(
    "Drop Location",
    options=LOCATIONS,
    index=None,
    placeholder="Select drop location"
)


# ============================================================
# DATE AND TIME
# ============================================================

pickup_date = st.date_input(
    "Pickup Date",
    value=None,
    min_value=date(2000, 1, 1),
    max_value=date(2030, 12, 31)
    )


pickup_time = st.time_input(
    "Pickup Time",
    value=None
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button(
    "Predict ETA",
    type="primary",
    use_container_width=True
):

    # --------------------------------------------------------
    # Validation 1: pickup location
    # --------------------------------------------------------

    if pickup_location is None:

        st.error(
            "Please select a pickup location."
        )

        st.stop()


    # --------------------------------------------------------
    # Validation 2: drop location
    # --------------------------------------------------------

    if drop_location is None:

        st.error(
            "Please select a drop location."
        )

        st.stop()


    # --------------------------------------------------------
    # Validation 3: same pickup and drop
    # --------------------------------------------------------

    if pickup_location == drop_location:

        st.error(
            "Pickup and drop locations cannot be the same."
        )

        st.stop()


    # --------------------------------------------------------
    # Validation 4: date
    # --------------------------------------------------------

    if pickup_date is None:

        st.error(
            "Please select a pickup date."
        )

        st.stop()


    # --------------------------------------------------------
    # Validation 5: time
    # --------------------------------------------------------

    if pickup_time is None:

        st.error(
            "Please select a pickup time."
        )

        st.stop()


    # ========================================================
    # CONVERT DATE AND TIME TO API FORMAT
    # ========================================================

    pickup_date_str = pickup_date.strftime(
        "%Y-%m-%d"
    )

    pickup_time_str = pickup_time.strftime(
        "%H:%M"
    )


    # ========================================================
    # CREATE API REQUEST
    # ========================================================

    payload = {

        "pickup_location": pickup_location,

        "drop_location": drop_location,

        "pickup_date": pickup_date_str,

        "pickup_time": pickup_time_str,
    }


    # ========================================================
    # CALL FASTAPI
    # ========================================================

    try:

        with st.spinner(
            "Calculating ETA..."
        ):

            response = requests.post(
                API_URL,
                json=payload,
                timeout=10
            )


        # ----------------------------------------------------
        # API validation
        # ----------------------------------------------------

        if response.status_code == 200:

            result = response.json()

            
            # =================================================
            # DISPLAY RESULT
            # =================================================

            st.success(
                "ETA calculated successfully!"
            )


            # -------------------------------------------------
            # Extract ETA
            # -------------------------------------------------

            prediction = result.get("prediction", {})

            eta = prediction.get("eta_minutes")


            if eta is not None:

                st.metric(
                    label="Estimated ETA",
                    value=f"{float(eta):.1f} minutes"
                )

            else:

                st.warning(
                    "The API responded successfully, but no ETA was found."
                )

                st.json(result)


            
    # ========================================================
    # CONNECTION ERROR
    # ========================================================

    except requests.exceptions.ConnectionError:

        st.error(
            "Unable to connect to the ETA API. "
            "Please make sure FastAPI is running."
        )


    # ========================================================
    # TIMEOUT
    # ========================================================

    except requests.exceptions.Timeout:

        st.error(
            "The ETA API took too long to respond."
        )


    # ========================================================
    # OTHER ERRORS
    # ========================================================

    except Exception as e:

        st.error(
            f"Unexpected error: {str(e)}"
        )