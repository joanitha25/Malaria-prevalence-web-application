import streamlit as st
import sys
import joblib
import json
import numpy as np
import pandas as pd
import ee
import os
import base64
import folium
import streamlit.components.v1 as components
from branca.element import Template, MacroElement

# ==============================================================================
# 0. CONFIGURATION SETUP (Must be the absolute first Streamlit execution)
# ==============================================================================
st.set_page_config(page_title="Malaria Prevalence Prediction", layout="wide")

# ==============================================================================
# 1. MODEL LOADING (Cached)
# ==============================================================================
@st.cache_resource
def load_ml_pipeline():
    return joblib.load("best_rf_reduced_model.joblib")

model_pipeline = load_ml_pipeline()

# ==============================================================================
# 2. EARTH ENGINE AUTHENTICATION SETUP
# ==============================================================================
try:
    b64_credentials = st.secrets["EARTHENGINE_CREDENTIALS_BASE64"]
    decoded_bytes = base64.b64decode(b64_credentials)
    credentials_dict = json.loads(decoded_bytes.decode("utf-8"))
    
    service_account_email = credentials_dict["client_email"]
    
    with open("ee_credentials.json", "w") as f:
        json.dump(credentials_dict, f)
        
    credentials = ee.ServiceAccountCredentials(service_account_email, "ee_credentials.json")
    ee.Initialize(credentials)
    
except Exception as e:
    st.error(f"Earth Engine Authentication Failed. Error details: {e}")

# ==========================================
# 3. Main Interface Header Setup
# ==========================================
st.set_page_config(page_title="Malaria Prevalence Prediction", layout="wide")

st.title("A Web Application for Malaria Prevalence Prediction")

# Initialize session state variables with default values
if "map_ready" not in st.session_state:
    st.session_state.map_ready = False
    st.session_state.smoothed_prediction_30m = None
    st.session_state.pixel_data = None
    st.session_state.aoi = None
    st.session_state.target_year = 2026
    st.session_state.target_district = "Karagwe"

# Explicit Navigation System
current_view = st.radio(
    label="Navigation Menu",
    options=["About the Application", "Malaria Prevalence Prediction Workspace"],
    horizontal=True,
    label_visibility="collapsed"
)

st.write("---")

# ==========================================
# View 1: About Panel
# ==========================================
if current_view == "About the Application":
    st.header("About the Application")
    st.write(
        """
        This web application provides an automated platform for predicting the *Plasmodium falciparum* parasite 
        rate for children between 2 and 10 years (**PfPR2-10**) using satellite-derived environmental variables 
        and a trained Random Forest machine learning model. It integrates Google Earth Engine (GEE) to 
        automatically retrieve environmental predictors, including the Normalized Difference Water Index (NDWI), 
        Normalized Difference Moisture Index (NDMI), land surface temperature (LST), rainfall, elevation, 
        and distance to water bodies, for the selected district and year.
        
        ### Operational Timeline Capabilities:
        * **Historical Observation (2020–2025):** Extracts observed, completed annual satellite data layers.
        * **Hybrid Active Prediction (2026):** Dynamically blends actual in-situ observations recorded during the current year with historical baseline trends to capture real-time anomalies.
        * **Baseline Projection Framework (2027):** Simulates a mid-term risk map using a stable multi-year climatology background to isolate persistent structural transmission hot spots.
        """
    )
    st.warning(
        """
        **Important note:**
        The model generates predictions at the original model resolution (5 km). The displayed 30 m map is 
        produced through resampling for visualization purposes only and should not be interpreted as 
        increasing the spatial resolution or predictive accuracy of the model.
        """
    )

# ==========================================
# View 2: Prediction Workspace
# ==========================================
elif current_view == "Malaria Prevalence Prediction Workspace":
    st.header("Malaria Prevalence Prediction Workspace")
    
    def reset_map_state():
        st.session_state.map_ready = False

    # Updated selectbox encompassing historical baseline, 2026 hybrid, and 2027 projection
    available_years = [2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027]
    
    target_year = st.selectbox(
        "Select Target Surveillance / Projection Year", 
        available_years, 
        index=available_years.index(st.session_state.target_year),
        on_change=reset_map_state
    )
    target_district = st.selectbox(
        "Select Target District", 
        ["Karagwe", "Kyerwa"], 
        index=["Karagwe", "Kyerwa"].index(st.session_state.target_district),
        on_change=reset_map_state
    )
    
    st.session_state.target_year = target_year
    st.session_state.target_district = target_district

    if st.button("Run Predictions"):
        current_year = 2026
        
        # Configure UI Spinner Messaging
        if st.session_state.target_year == current_year:
            spinner_msg = "Compiling 2026 hybrid observation-climatology blending pipeline..."
        elif st.session_state.target_year > current_year:
            spinner_msg = "Simulating 2027 projection framework from structural baseline climatology..."
        else:
            spinner_msg = "Extracting historical observed climate diagnostics from GEE..."
            
        with st.spinner(spinner_msg):
            
            # Define Geographic Spatial Boundaries
            districts = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level2")
            aoi = districts.filter(ee.Filter.eq("ADM2_NAME", st.session_state.target_district))
            aoi_geometry = aoi.geometry()
            
            commonCRS = "EPSG:4326"
            fineScale = 30
            pfprScale = 5000
            commonProjection = ee.Projection(commonCRS).atScale(fineScale)

            # Establish the historical reference window baseline (2020 through end of 2025)
            base_start = ee.Date.fromYMD(2020, 1, 1)
            base_end = ee.Date.fromYMD(2026, 1, 1)
            
            years_list = ee.List([2020, 2021, 2022, 2023, 2024, 2025])
            
            def get_annual_rain(y):
                start = ee.Date.fromYMD(y, 1, 1)
                end = ee.Date.fromYMD(ee.Number(y).add(1), 1, 1)
                return ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")\
                    .filterBounds(aoi_geometry).filterDate(start, end).select("precipitation").sum()

            # ------------------------------------------------------------
            # DYNAMIC TEMPORAL ROUTING PIPELINE
            # ------------------------------------------------------------
            if st.session_state.target_year == current_year:
                # HYBRID APPROACH: Real 2026 metrics up to today combined with historical baselines
                real_start = ee.Date.fromYMD(current_year, 1, 1)
                real_end = ee.Date('2026-07-10')  # Hardcoded deployment date threshold
                
                s2_real = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
                    .filterBounds(aoi_geometry).filterDate(real_start, real_end)\
                    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)).select(["B3", "B8", "B11"])
                
                lst_real = ee.ImageCollection("MODIS/061/MOD11A1")\
                    .filterBounds(aoi_geometry).filterDate(real_start, real_end).select("LST_Day_1km")
                
                rain_real = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")\
                    .filterBounds(aoi_geometry).filterDate(real_start, real_end).select("precipitation").sum()
                
                # Fetch reference baseline background assets
                s2_base = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
                    .filterBounds(aoi_geometry).filterDate(base_start, base_end)\
                    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)).select(["B3", "B8", "B11"])
                
                lst_base = ee.ImageCollection("MODIS/061/MOD11A1")\
                    .filterBounds(aoi_geometry).filterDate(base_start, base_end).select("LST_Day_1km")
                
                rain_base = ee.ImageCollection(years_list.map(get_annual_rain)).mean()

                # Merge active observations with baseline background context layers
                s2 = ee.ImageCollection([s2_real.median(), s2_base.median()]).mean().clip(aoi_geometry)
                lst_raw = ee.ImageCollection([lst_real.mean(), lst_base.mean()]).mean().clip(aoi_geometry)
                
                # Rain composite: Active accumulation + a 50% fraction of typical remaining variance
                rainfall = rain_real.add(rain_base.multiply(0.5)).rename("Rainfall").clip(aoi_geometry)

            elif st.session_state.target_year > current_year:
                # PURE CLIMATOLOGY PROJECTION (Target Year 2027)
                s2Collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
                    .filterBounds(aoi_geometry).filterDate(base_start, base_end)\
                    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)).select(["B3", "B8", "B11"])
                s2 = s2Collection.median().clip(aoi_geometry)
                
                lst_collection = ee.ImageCollection("MODIS/061/MOD11A1")\
                    .filterBounds(aoi_geometry).filterDate(base_start, base_end).select("LST_Day_1km")
                lst_raw = lst_collection.mean().clip(aoi_geometry)
                
                annual_rain_collection = ee.ImageCollection(years_list.map(get_annual_rain))
                rainfall = annual_rain_collection.mean().rename("Rainfall").clip(aoi_geometry)
                
            else:
                # STANDARD HISTORICAL RECORD TRACKING (2020 - 2025)
                start_date = ee.Date.fromYMD(st.session_state.target_year, 1, 1)
                end_date = ee.Date.fromYMD(st.session_state.target_year + 1, 1, 1)
                
                s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
                    .filterBounds(aoi_geometry).filterDate(start_date, end_date)\
                    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)).select(["B3", "B8", "B11"])\
                    .median().clip(aoi_geometry)
                    
                lst_raw = ee.ImageCollection("MODIS/061/MOD11A1")\
                    .filterBounds(aoi_geometry).filterDate(start_date, end_date).select("LST_Day_1km")\
                    .mean().clip(aoi_geometry)
                    
                rainfall = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")\
                    .filterBounds(aoi_geometry).filterDate(start_date, end_date).select("precipitation")\
                    .sum().rename("Rainfall").clip(aoi_geometry)

            # Feature Calculation Logic
            ndwi = s2.normalizedDifference(["B3", "B8"]).rename("NDWI")
            ndmi = s2.normalizedDifference(["B8", "B11"]).rename("NDMI")
            lst = lst_raw.multiply(0.02).subtract(273.15).rename("LST")
            elevation = ee.Image("USGS/SRTMGL1_003").select("elevation").rename("Elevation").clip(aoi_geometry)
            
            water = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(50).clip(aoi_geometry)
            water5km = water.reproject(crs=commonCRS, scale=pfprScale).unmask(0)
            distWater5km = water5km.fastDistanceTransform(256, "pixels", "squared_euclidean").sqrt().multiply(pfprScale).rename("DistWater").clip(aoi_geometry)

            # MEMORY FIX: Bounding Boxes (Removes the .getInfo() segmentation fault vector)
            if st.session_state.target_district == "Kyerwa":
                min_lon, max_lon, min_lat, max_lat = 30.39, 31.05, -1.58, -1.02
            else:  # Karagwe Default
                min_lon, max_lon, min_lat, max_lat = 30.45, 31.42, -2.03, -1.29
            
            step = pfprScale / 111000.0
            grid_points = []
            lon_iter = min_lon
            while lon_iter <= max_lon:
                lat_iter = min_lat
                while lat_iter <= max_lat:
                    grid_points.append(ee.Feature(ee.Geometry.Point([lon_iter, lat_iter])))
                    lat_iter += step
                lon_iter += step
                
            grid_samples = ee.FeatureCollection(grid_points).filterBounds(aoi)

            band_names = ["NDWI", "NDMI", "LST", "Rainfall", "Elevation", "DistWater"]
            raw_stack = ee.Image.cat([ndwi, ndmi, lst, rainfall, elevation, distWater5km]).toFloat()

            pixel_samples = raw_stack.reduceRegions(
                collection=grid_samples, reducer=ee.Reducer.mean(),
                scale=fineScale, crs=commonCRS, tileScale=4
            ).getInfo()

            features_list = []
            for feat in pixel_samples["features"]:
                props = feat["properties"]
                if "geometry" in feat and feat["geometry"] is not None:
                    coords = feat["geometry"]["coordinates"]
                    props["longitude"], props["latitude"] = coords[0], coords[1]
                    features_list.append(props)
            
            pixel_data = pd.DataFrame(features_list) if features_list else pd.DataFrame()

            if not pixel_data.empty:
                valid_cols = band_names + ["longitude", "latitude"]
                pixel_data = pixel_data[[col for col in pixel_data.columns if col in valid_cols]].dropna(subset=band_names)
                
                X_pixels = pixel_data[band_names]
                pixel_data["predicted_PfPR"] = model_pipeline.predict(X_pixels)
                
                # REFACTOR STABILITY FIX: Pure Native ee.Feature assembly without geemap dependencies
                ee_features = []
                for _, row in pixel_data.iterrows():
                    geom = ee.Geometry.Point([row["longitude"], row["latitude"]])
                    feat = ee.Feature(geom, {"predicted_PfPR": float(row["predicted_PfPR"])})
                    ee_features.append(feat)
                
                predicted_gee_layer = ee.FeatureCollection(ee_features)
                grid_projection = ee.Projection(commonCRS).atScale(pfprScale)
                
                prediction_raster_5km = predicted_gee_layer.reduceToImage(
                    properties=["predicted_PfPR"], reducer=ee.Reducer.first()
                ).reproject(crs=grid_projection)
                
                smoothed_prediction_30m = prediction_raster_5km.resample('bilinear').reproject(crs=commonProjection).clip(aoi_geometry)
                
                st.session_state.pixel_data = pixel_data
                st.session_state.smoothed_prediction_30m = smoothed_prediction_30m
                st.session_state.aoi = aoi
                st.session_state.map_ready = True

    # Visual Output Canvas
    if st.session_state.map_ready:
        st.write("---")
        if st.session_state.target_year == 2026:
            st.info("📊 Displaying active hybrid model: Integrates real-time 2026 observations with predictive baseline models.")
        elif st.session_state.target_year == 2027:
            st.warning("🔮 Displaying projected structural malaria risk framework modeled for the 2027 climatology horizon.")
        else:
            st.success(f"✅ Successfully processed {st.session_state.target_district} District for {st.session_state.target_year}!")
        
        st.write("### 💾 Export Spatial Products:")
        try:
            raw_download_url = st.session_state.smoothed_prediction_30m.reproject(
                crs="EPSG:4326", scale=5000
            ).getDownloadURL({
                'name': f'PfPR_Output_{st.session_state.target_district}_{st.session_state.target_year}_5km',
                'scale': 5000, 'crs': 'EPSG:4326', 'filePerBand': False
            })
            st.markdown(f"[📥 Download Native 5km Model Raster (.tiff)]({raw_download_url})")
        except Exception:
            st.info("Download link generation timed out on remote GEE servers.")

        st.write("### Interactive Map Display:")
        
        map_center = [-1.30, 30.85] if st.session_state.target_district == "Kyerwa" else [-1.59, 31.21]
        map_zoom = 10 if st.session_state.target_district == "Kyerwa" else 9

        f_map = folium.Map(location=map_center, zoom_start=map_zoom, control_scale=True)
        
        aoi_map_id = ee.Image().paint(st.session_state.aoi, 0, 2).getMapId()
        folium.TileLayer(
            tiles=aoi_map_id['tile_fetcher'].url_format, attr='Google Earth Engine',
            name=f'{st.session_state.target_district} Border', overlay=True
        ).add_to(f_map)
        
        min_val = float(st.session_state.pixel_data["predicted_PfPR"].min())
        max_val = float(st.session_state.pixel_data["predicted_PfPR"].max())
        
        high_contrast_palette = ['#3288bd', '#99d594', '#e6f598', '#fee08b', '#fc8d59', '#d53e4f']
        vis_params = {'min': min_val, 'max': max_val, 'palette': high_contrast_palette}
        
        prediction_map_id = st.session_state.smoothed_prediction_30m.getMapId(vis_params)
        folium.TileLayer(
            tiles=prediction_map_id['tile_fetcher'].url_format, attr='Google Earth Engine',
            name=f'Predicted PfPR ({st.session_state.target_year})', overlay=True, opacity=0.85
        ).add_to(f_map)
        
        v_min, v_max = f"{min_val:.1f}%", f"{max_val:.1f}%"
        css_gradient = ", ".join(high_contrast_palette)

        legend_template = f"""
        {{% macro html(this, kwargs) %}}
        <div id='maplegend' class='maplegend' style='position: absolute; z-index:9999; border:2px solid #bbb; background-color:rgba(255, 255, 255, 0.95); border-radius:8px; padding: 12px 15px; font-size:13px; right: 20px; bottom: 30px; width: 280px; font-family: "Source Sans Pro", sans-serif; box-shadow: 0 0 15px rgba(0,0,0,0.2);'>
          <div class='legend-title' style='font-weight: bold; margin-bottom: 8px; text-align: center; color: #333;'>Malaria Prevalence (PfPR2-10)</div>
          <div class='gradient-bar' style='background: linear-gradient(to right, {css_gradient}) !important; width: 100%; height: 18px; border-radius: 4px; border: 1px solid #777;'></div>
          <div class='legend-labels' style='margin-top: 5px; font-weight: 600; color: #444; display: flex; justify-content: space-between;'>
            <span>Low ({v_min})</span><span>High ({v_max})</span>
          </div>
        </div>
        <div id='export-container' style='position: absolute; z-index:9999; top: 10px; left: 50px;'>
          <button onclick="window.print()" style='padding: 6px 12px; background: white; border: 2px solid #ccc; border-radius: 4px; cursor: pointer; font-weight: bold; font-family: "Source Sans Pro", sans-serif; font-size: 12px;'>📷 Save Map View</button>
        </div>
        {{% endmacro %}}
        """
        macro = MacroElement()
        macro._template = Template(legend_template)
        f_map.add_child(macro)

        folium.LayerControl().add_to(f_map)
        components.html(f_map._repr_html_(), height=650, scrolling=True)