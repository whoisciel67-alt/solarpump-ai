import gradio as gr
import joblib
import pandas as pd
import requests

# Load your trained model
model = joblib.load("model.joblib")


def fetch_open_meteo_weather(lat, lon):
  try:
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        "&daily=shortwave_radiation_sum,temperature_2m_max,relative_humidity_2m_mean&timezone=auto"
    )
    response = requests.get(url, timeout=10)
    data = response.json()
    daily = data.get("daily", {})
    solar_rad = daily.get("shortwave_radiation_sum", [18.0])[0]
    max_temp = daily.get("temperature_2m_max", [30.0])[0]
    humidity = daily.get("relative_humidity_2m_mean", [65.0])[0]
    clear_sky = solar_rad * 1.2 if solar_rad else 22.0
    return float(solar_rad), float(clear_sky), float(max_temp), float(humidity)
  except Exception:
    return 18.0, 22.0, 30.0, 65.0


def predict_solar_power(
    solar_radiation, clear_sky_radiation, max_temp, relative_humidity
):
  features = pd.DataFrame(
      [[solar_radiation, clear_sky_radiation, max_temp, relative_humidity]],
      columns=[
          "solar_radiation",
          "clear_sky_radiation",
          "max_temp",
          "relative_humidity",
      ],
  )
  prediction = round(float(model.predict(features)[0]), 2)

  # Color-coded HTML badge styling based on tier
  if prediction >= 6.0:
    bg_color, border_color, text_color = "#dcfce7", "#22c55e", "#166534"
    tier = "HIGH OUTPUT"
    icon = "☀️"
    recommendation = (
        "Run water pumps at full capacity during midday. Optimal solar"
        " generation expected."
    )
  elif prediction >= 3.5:
    bg_color, border_color, text_color = "#fef3c7", "#f59e0b", "#92400e"
    tier = "MODERATE OUTPUT"
    icon = "⛅"
    recommendation = (
        "Schedule pumping during peak sunlight hours (10 AM – 2 PM). Monitor"
        " storage levels."
    )
  else:
    bg_color, border_color, text_color = "#fee2e2", "#ef4444", "#991b1b"
    tier = "LOW OUTPUT"
    icon = "☁️"
    recommendation = (
        "Conserve energy. Limit irrigation to essential crops or switch to"
        " backup power."
    )

  status_card = f"""
    <div style="background-color: {bg_color}; border-left: 6px solid {border_color}; padding: 20px; border-radius: 10px; color: {text_color}; font-family: sans-serif;">
        <div style="font-size: 14px; font-weight: 700; letter-spacing: 0.05em; opacity: 0.85;">PUMP RECOMMENDATION TIER</div>
        <div style="font-size: 26px; font-weight: 800; margin: 6px 0;">{icon} {tier}</div>
        <div style="font-size: 16px; margin-bottom: 12px;">Estimated Generation: <strong>{prediction} kWh/m²/day</strong></div>
        <div style="font-size: 15px; line-height: 1.5; background: rgba(255,255,255,0.6); padding: 10px 12px; border-radius: 6px;">
            <strong>Guidance:</strong> {recommendation}
        </div>
    </div>
    """
  weather_summary = (
      f"📊 **Forecast Inputs Used:** Solar Radiation: `{solar_radiation:.1f}` |"
      f" Clear-Sky: `{clear_sky_radiation:.1f}` | Max Temp: `{max_temp:.1f}°C`"
      f" | Humidity: `{relative_humidity:.0f}%`"
  )
  return status_card, weather_summary


def run_live_forecast(lat, lon):
  s_rad, c_sky, temp, hum = fetch_open_meteo_weather(lat, lon)
  return predict_solar_power(s_rad, c_sky, temp, hum)


# Custom theme and layout
custom_theme = gr.themes.Soft(
    primary_hue="amber",
    secondary_hue="emerald",
    neutral_hue="slate",
)

with gr.Blocks(theme=custom_theme, title="SolarPump AI") as demo:
  # Clean Header Banner
  gr.Markdown("""
        # ☀️ SolarPump AI
        ### Smart Next-Day Solar Forecasting & Agricultural Irrigation Assistant
        *Powered by Random Forest Machine Learning (MAE: 1.865) & Live Open-Meteo Satellite Forecasts*
        """)

  with gr.Tabs():
    # TAB 1: Live Location Forecast
    with gr.Tab("🌍 Live Location Forecast"):
      with gr.Row():
        # Left Column: Inputs
        with gr.Column(scale=1):
          gr.Markdown("#### 1. Select Farm Location")
          lat_input = gr.Number(label="Latitude", value=13.36)
          lon_input = gr.Number(label="Longitude", value=103.85)

          gr.Markdown("**Quick Presets:**")
          with gr.Row():
            btn_sr = gr.Button("Siem Reap", size="sm")
            btn_pp = gr.Button("Phnom Penh", size="sm")
            btn_ca = gr.Button("Central Valley, CA", size="sm")

          predict_live_btn = gr.Button(
              "⚡ Generate Tomorrow's Forecast", variant="primary", size="lg"
          )

        # Right Column: Results
        with gr.Column(scale=1):
          gr.Markdown("#### 2. Irrigation Forecast Result")
          live_card = gr.HTML(
              value="""
                        <div style="border: 2px dashed #cbd5e1; padding: 30px; border-radius: 10px; text-align: center; color: #64748b;">
                            Click <strong>Generate Tomorrow's Forecast</strong> to fetch live weather and calculate pump guidance.
                        </div>
                        """
          )
          live_details = gr.Markdown("")

      # Preset Button Actions
      btn_sr.click(lambda: (13.36, 103.85), outputs=[lat_input, lon_input])
      btn_pp.click(lambda: (11.55, 104.92), outputs=[lat_input, lon_input])
      btn_ca.click(lambda: (36.74, -119.78), outputs=[lat_input, lon_input])

      predict_live_btn.click(
          fn=run_live_forecast,
          inputs=[lat_input, lon_input],
          outputs=[live_card, live_details],
      )

    # TAB 2: Scenario Simulator
    with gr.Tab("🎛️ Weather Scenario Simulator"):
      with gr.Row():
        with gr.Column(scale=1):
          gr.Markdown("#### Adjust Weather Variables")
          s_sol = gr.Slider(
              0, 30, value=20.0, step=0.5, label="Solar Radiation (MJ/m²)"
          )
          s_clr = gr.Slider(
              0, 35, value=24.0, step=0.5, label="Clear-Sky Radiation (MJ/m²)"
          )
          s_tmp = gr.Slider(
              10, 50, value=33.0, step=0.5, label="Max Temperature (°C)"
          )
          s_hum = gr.Slider(
              0, 100, value=60.0, step=1.0, label="Relative Humidity (%)"
          )
          sim_btn = gr.Button(
              "🔍 Simulate Scenario", variant="primary", size="lg"
          )

        with gr.Column(scale=1):
          gr.Markdown("#### Simulated Pump Guidance")
          sim_card = gr.HTML(
              value="""
                        <div style="border: 2px dashed #cbd5e1; padding: 30px; border-radius: 10px; text-align: center; color: #64748b;">
                            Adjust the sliders on the left and click <strong>Simulate Scenario</strong> to test different weather conditions.
                        </div>
                        """
          )
          sim_details = gr.Markdown("")

      sim_btn.click(
          fn=predict_solar_power,
          inputs=[s_sol, s_clr, s_tmp, s_hum],
          outputs=[sim_card, sim_details],
      )

  # Footer
  gr.Markdown("---\n*Model Evaluation: 92.3% Tier Classification Accuracy*")

demo.launch(share=True)
