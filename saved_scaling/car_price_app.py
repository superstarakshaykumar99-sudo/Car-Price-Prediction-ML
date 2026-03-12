from tkinter import *
import pickle
import os

# ── Resolve absolute paths so the app works no matter where it's launched from ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # CarPricePrediction/
MODEL_PATH  = os.path.join(BASE_DIR, "saved_models", "RandomForestRegressor.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "saved_scaling", "scaler.pkl")

def format_value(value):
    if value >= 10000000:
        return f"{value/10000000:.2f} Cr"
    elif value >= 100000:
        return f"{value/100000:.2f} Lakhs"
    else:
        return f"{value:.2f}"


# Load the trained model
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Load the scaler used during model training
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

# Global variables to hold radio btn values — initialised to defaults matching the radio buttons
seller_selected_value = "Dealer"
fuel_selected_value   = "Petrol"
transmission_selected_value = "Manual"


def pred_price():
    try:
        input_values = []

        # ── Numeric Inputs ──────────────────────────────────────────────────────
        try:
            input_values.append(int(vehicle_age_entry.get()))
        except ValueError:
            price_label.config(text="⚠ Vehicle Age must be a whole number.", fg="red")
            return

        try:
            input_values.append(int(km_driven_entry.get()))
        except ValueError:
            price_label.config(text="⚠ KM Driven must be a whole number.", fg="red")
            return

        try:
            input_values.append(float(mileage_entry.get()))
        except ValueError:
            price_label.config(text="⚠ Mileage must be a number.", fg="red")
            return

        try:
            input_values.append(int(engine_entry.get()))
        except ValueError:
            price_label.config(text="⚠ Engine (CC) must be a whole number.", fg="red")
            return

        try:
            input_values.append(float(max_power_entry.get()))
        except ValueError:
            price_label.config(text="⚠ Max Power must be a number.", fg="red")
            return

        try:
            seat_val = int(seats_entry.get())
            if not (2 <= seat_val <= 10):
                price_label.config(text="⚠ Seats must be between 2 and 10.", fg="red")
                return
            input_values.append(seat_val)
        except ValueError:
            price_label.config(text="⚠ Seats must be a whole number.", fg="red")
            return

        # ── Seller Type Encoding (3 categories) ─────────────────────────────────
        seller_encoding = {
            "Dealer":          [1, 0, 0],
            "Individual":      [0, 1, 0],
            "Trustmark Dealer":[0, 0, 1],
        }
        input_values.extend(seller_encoding.get(seller_selected_value, [1, 0, 0]))

        # ── Fuel Type Encoding (5 categories) ───────────────────────────────────
        fuel_dict = {
            "CNG":      [1, 0, 0, 0, 0],
            "Diesel":   [0, 1, 0, 0, 0],
            "Electric": [0, 0, 1, 0, 0],
            "LPG":      [0, 0, 0, 1, 0],
            "Petrol":   [0, 0, 0, 0, 1],
        }
        input_values.extend(fuel_dict.get(fuel_selected_value, [0, 0, 0, 0, 1]))

        # ── Transmission Encoding (2 categories) ────────────────────────────────
        transmission_encoding = {
            "Automatic": [1, 0],
            "Manual":    [0, 1],
        }
        input_values.extend(transmission_encoding.get(transmission_selected_value, [0, 1]))

        # ── Predict ─────────────────────────────────────────────────────────────
        if len(input_values) != 16:
            price_label.config(
                text=f"⚠ Input size mismatch: expected 16, got {len(input_values)}.", fg="red"
            )
            return

        input_scaled = scaler.transform([input_values])
        prediction   = model.predict(input_scaled)[0]
        prediction_formatted = format_value(prediction)

        price_label.config(
            text=f"Predicted Price: ₹ {prediction_formatted}", fg="white"
        )
        print(f"Predicted price: ₹ {prediction_formatted}")

    except Exception as e:
        price_label.config(text=f"⚠ Error: {e}", fg="red")
        print(f"Error: {e}")


# ── GUI Setup ────────────────────────────────────────────────────────────────────
root = Tk()
root.geometry("1080x720")
root.title("Car Price Predictor")
root.config(bg="black")

title_label = Label(
    root, text="Car Price Predictor",
    bg="black", fg="green", font=("Arial", 30, "bold")
)
title_label.pack(pady=30)


def create_labeled_entry(parent, label_text, padx=30):
    frame = Frame(parent, bg="black")
    frame.pack()
    Label(frame, text=label_text, bg="black", fg="white",
          font=("Arial", 20, "bold")).pack(side=LEFT, padx=padx)
    entry = Entry(frame, font=("Arial", 15, "bold"), bd=5, relief=RAISED)
    entry.pack(side=LEFT)
    return entry


# ── Entry Widgets ────────────────────────────────────────────────────────────────
# Note: Car Name is shown for reference only — the ML model does not use it
car_name_entry   = create_labeled_entry(root, "Car Name",    30)
vehicle_age_entry = create_labeled_entry(root, "Vehicle Age", 19)
km_driven_entry  = create_labeled_entry(root, "KM Driven",  30)
mileage_entry    = create_labeled_entry(root, "Mileage",     50)
engine_entry     = create_labeled_entry(root, "Engine (CC)", 30)
max_power_entry  = create_labeled_entry(root, "Max Power",   30)
seats_entry      = create_labeled_entry(root, "Seats",       60)

# ── Radio Buttons: Seller Type ───────────────────────────────────────────────────
seller_type_frame = Frame(root, bg="black")
seller_type_frame.pack()
Label(seller_type_frame, text="Seller Type", bg="black", fg="white",
      font=("Arial", 20, "bold")).pack(side=LEFT, padx=60)

selected_seller = StringVar(root, value="Dealer")


def on_seller_selected():
    global seller_selected_value
    seller_selected_value = selected_seller.get()


for text, value in {"Dealer": "Dealer", "Individual": "Individual",
                    "Trustmark Dealer": "Trustmark Dealer"}.items():
    Radiobutton(seller_type_frame, text=text, variable=selected_seller, value=value,
                font=("Arial", 10, "bold"), command=on_seller_selected).pack(side=LEFT, ipady=5)

# ── Radio Buttons: Fuel Type ────────────────────────────────────────────────────
fuel_type_frame = Frame(root, bg="black")
fuel_type_frame.pack()
Label(fuel_type_frame, text="Fuel Type", bg="black", fg="white",
      font=("Arial", 20, "bold")).pack(side=LEFT, padx=80)

selected_fuel = StringVar(root, value="Petrol")


def on_fuel_selected():
    global fuel_selected_value
    fuel_selected_value = selected_fuel.get()


for text, value in {"CNG": "CNG", "Diesel": "Diesel", "Electric": "Electric",
                    "LPG": "LPG", "Petrol": "Petrol"}.items():
    Radiobutton(fuel_type_frame, text=text, variable=selected_fuel, value=value,
                font=("Arial", 10, "bold"), command=on_fuel_selected).pack(side=LEFT, ipady=5)

# ── Radio Buttons: Transmission Type ────────────────────────────────────────────
transmission_frame = Frame(root, bg="black")
transmission_frame.pack()
Label(transmission_frame, text="Transmission Type", bg="black", fg="white",
      font=("Arial", 20, "bold")).pack(side=LEFT, padx=10)

selected_transmission = StringVar(root, value="Manual")


def on_transmission_selected():
    global transmission_selected_value
    transmission_selected_value = selected_transmission.get()


for text, value in {"Automatic": "Automatic", "Manual": "Manual"}.items():
    Radiobutton(transmission_frame, text=text, variable=selected_transmission, value=value,
                font=("Arial", 10, "bold"), command=on_transmission_selected).pack(side=LEFT, ipady=5)

# ── Predict Button ───────────────────────────────────────────────────────────────
pred_btn = Button(root, text="Predict Price", command=pred_price,
                  bg="green", fg="white", font=("Arial", 20, "bold"))
pred_btn.pack(pady=30)

# ── Price Output Label ───────────────────────────────────────────────────────────
price_label = Label(root, text="Predicted Price: ₹ 0",
                    bg="black", fg="white", font=("Arial", 20, "bold"))
price_label.pack()

root.mainloop()
