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
from streamlit_folium import st_folium  # Imported for interactive point selection

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

# ==============================================================================
# 3. MAIN INTERFACE HEADER SETUP
# ==============================================================================
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
        """
    )

# ==========================================
# View 2: Prediction Workspace
# ==========================================
elif current_view == "Malaria Prevalence Prediction Workspace":
    st.header("Malaria Prevalence Prediction Workspace")
    
    def reset_map_state():
        st.session_state.map_ready = False

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
        spinner_msg = "Extracting spatial diagnostics from Google Earth Engine..."
            
        with st.spinner(spinner_msg):
            
            # Define Geographic Spatial Boundaries
            districts = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level2")
            aoi = districts.filter(ee.Filter.eq("ADM2_NAME", st.session_state.target_district))
            aoi_geometry = aoi.geometry()
            
            commonCRS = "EPSG:4326"
            fineScale = 30
            pfprScale = 5000
            commonProjection = ee.Projection(commonCRS).atScale(fineScale)

            # Establish historical baseline dates
            base_start = ee.Date.fromYMD(2020, 1, 1)
            base_end = ee.Date.fromYMD(2026, 1, 1)
            years_list = ee.List([2020, 2021, 2022, 2023, 2024, 2025])
            
            def get_annual_rain(y):
                start = ee.Date.fromYMD(y, 1, 1)
                end = ee.Date.fromYMD(ee.Number(y).add(1), 1, 1)
                return ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")\
                    .filterBounds(aoi_geometry).filterDate(start, end).select("precipitation").sum()

            # ------------------------------------------------------------
            # TEMPORAL DISPATCH ROUTER
            # ------------------------------------------------------------
            if st.session_state.target_year == current_year:
                real_start = ee.Date.fromYMD(current_year, 1, 1)
                real_end = ee.Date('2026-07-10')
                
                s2_real = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
                    .filterBounds(aoi_geometry).filterDate(real_start, real_end)\
                    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)).select(["B3", "B8", "B11"])
                lst_real = ee.ImageCollection("MODIS/061/MOD11A1")\
                    .filterBounds(aoi_geometry).filterDate(real_start, real_end).select("LST_Day_1km")
                rain_real = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")\
                    .filterBounds(aoi_geometry).filterDate(real_start, real_end).select("precipitation").sum()
                
                s2_base = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
                    .filterBounds(aoi_geometry).filterDate(base_start, base_end)\
                    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)).select(["B3", "B8", "B11"])
                lst_base = ee.ImageCollection("MODIS/061/MOD11A1")\
                    .filterBounds(aoi_geometry).filterDate(base_start, base_end).select("LST_Day_1km")
                rain_base = ee.ImageCollection(years_list.map(get_annual_rain)).mean()

                s2 = ee.ImageCollection([s2_real.median(), s2_base.median()]).mean().clip(aoi_geometry)
                lst_raw = ee.ImageCollection([lst_real.mean(), lst_base.mean()]).mean().clip(aoi_geometry)
                rainfall = rain_real.add(rain_base.multiply(0.5)).rename("Rainfall").clip(aoi_geometry)

            elif st.session_state.target_year > current_year:
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

            # Feature Layer Synthesizer
            ndwi = s2.normalizedDifference(["B3", "B8"]).rename("NDWI")
            ndmi = s2.normalizedDifference(["B8", "B11"]).rename("NDMI")
            lst = lst_raw.multiply(0.02).subtract(273.15).rename("LST")
            elevation = ee.Image("USGS/SRTMGL1_003").select("elevation").rename("Elevation").clip(aoi_geometry)
            
            water = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(50).clip(aoi_geometry)
            water5km = water.reproject(crs=commonCRS, scale=pfprScale).unmask(0)
            distWater5km = water5km.fastDistanceTransform(256, "pixels", "squared_euclidean").sqrt().multiply(pfprScale).rename("DistWater").clip(aoi_geometry)

            band_names = ["NDWI", "NDMI", "LST", "Rainfall", "Elevation", "DistWater"]
            raw_stack = ee.Image.cat([ndwi, ndmi, lst, rainfall, elevation, distWater5km]).toFloat()

            # ==================================================================
            # ROBUST SERVER-SIDE SAMPLING 
            # ==================================================================
            lonlat_image = ee.Image.pixelLonLat().clip(aoi_geometry)
            data_and_coords = raw_stack.addBands(lonlat_image)
            
            sampling_features = data_and_coords.sample(
                region=aoi_geometry,
                scale=4500,  
                projection=commonCRS,
                factor=None,
                numPixels=None,
                geometries=True,
                tileScale=4
            ).getInfo()

            features_list = []
            for feat in sampling_features["features"]:
                props = feat["properties"]
                if "longitude" in props and "latitude" in props:
                    features_list.append(props)
            
            pixel_data = pd.DataFrame(features_list) if features_list else pd.DataFrame()

            if not pixel_data.empty:
                pixel_data = pixel_data.dropna(subset=band_names)
                X_pixels = pixel_data[band_names]
                pixel_data["predicted_PfPR"] = model_pipeline.predict(X_pixels)
                
                ee_features = []
                for _, row in pixel_data.iterrows():
                    geom = ee.Geometry.Point([row["longitude"], row["latitude"]])
                    feat = ee.Feature(geom, {"predicted_PfPR": float(row["predicted_PfPR"])})
                    ee_features.append(feat)
                
                predicted_gee_layer = ee.FeatureCollection(ee_features)
                grid_projection = ee.Projection(commonCRS).atScale(pfprScale)
                
                prediction_raster_5km = predicted_gee_layer.reduceToImage(
                    properties=["predicted_PfPR"], reducer=ee.Reducer.mean()
                ).reproject(crs=grid_projection)
                
                smoothed_prediction_30m = prediction_raster_5km.resample('bilinear')\
                    .reproject(crs=commonProjection)\
                    .clip(aoi_geometry)
                
                # Cache parameters to local session memory for interface extraction
                st.session_state.raw_stack_bands = raw_stack
                st.session_state.band_names_list = band_names
                st.session_state.common_crs_str = commonCRS
                st.session_state.pixel_data = pixel_data
                st.session_state.smoothed_prediction_30m = smoothed_prediction_30m
                st.session_state.aoi = aoi
                st.session_state.map_ready = True

    # ==============================================================================
    # Rendering Screen Elements (Inside Workspace View)
    # ==============================================================================
    if st.session_state.map_ready:
        st.write("---")
        st.success(f"✅ Full predictive grid generated successfully for {st.session_state.target_district} ({st.session_state.target_year})!")
        
        # 💾 Export Spatial Products Code block
        st.write("### 💾 Export Spatial Products:")
        try:
            raw_download_url = st.session_state.smoothed_prediction_30m.getDownloadURL({
                'name': f'PfPR_Output_{st.session_state.target_district}_{st.session_state.target_year}',
                'scale': 5000, 
                'crs': 'EPSG:4326', 
                'filePerBand': False
            })
            st.markdown(f"[📥 Download Native 5km Model Raster (.tiff)]({raw_download_url})")
        except Exception:
            st.info("Download link generation timed out on remote GEE servers.")
            
        st.write("---")
        st.write("### 🗺️ Interactive Analysis Workspace")
        st.caption("💡 Click anywhere inside the mapped district border to inspect local environmental predictors and model output.")

        # Create a side-by-side split layout panel
        col1, col2 = st.columns([3, 1.5])

        with col1:
            # 🗺️ Build the Base Folium Map Asset
            map_center = [-1.30, 30.39] if st.session_state.target_district == "Kyerwa" else [-1.59, 31.05]
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
            
            folium.LayerControl().add_to(f_map)

            # NOTE: MacroElement removed here to prevent MarshallComponentException!
            # Render map with bi-directional click tracking component active
            map_data = st_folium(f_map, width="100%", height=650, key="interactive_workspace_map")

        with col2:
            # Render the Legend beautifully as a native Streamlit component right above the Inspector
            v_min, v_max = f"{min_val:.1f}%", f"{max_val:.1f}%"
            css_gradient = ", ".join(high_contrast_palette)
            
            st.markdown(
                f"""
                <div style='border:1px solid #ddd; background-color:#f9f9f9; border-radius:8px; padding: 12px 15px; font-size:14px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>
                  <div style='font-weight: bold; margin-bottom: 8px; text-align: center; color: #333;'>Malaria Prevalence (PfPR2-10)</div>
                  <div style='background: linear-gradient(to right, {css_gradient}); width: 100%; height: 18px; border-radius: 4px; border: 1px solid #777;'></div>
                  <div style='margin-top: 5px; font-weight: 600; color: #444; display: flex; justify-content: space-between;'>
                    <span>Low ({v_min})</span><span>High ({v_max})</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.subheader("📍 Point Inspector")
            
            # Catch the click parameters returned from the map interface element
            if map_data and map_data.get("last_clicked"):
                clicked_lat = map_data["last_clicked"]["lat"]
                clicked_lng = map_data["last_clicked"]["lng"]
                
                st.info(f"**Selected Coordinates:**\n* **Lat:** `{clicked_lat:.5f}`\n* **Lon:** `{clicked_lng:.5f}`")
                
                with st.spinner("Extracting pixel attributes from GEE layers..."):
                    try:
                        inspect_point = ee.Geometry.Point([clicked_lng, clicked_lat])
                        
                        # Sample raw data bands at clicked coordinate space location
                        point_sample = st.session_state.raw_stack_bands.sample(
                            region=inspect_point,
                            scale=30,
                            projection=st.session_state.common_crs_str,
                            geometries=False
                        ).getInfo()
                        
                        if point_sample and len(point_sample['features']) > 0:
                            extracted_props = point_sample['features'][0]['properties']
                            
                            # Align variables with prediction pipeline structure rules
                            input_df = pd.DataFrame([extracted_props])[st.session_state.band_names_list]
                            
                            # Run local target area custom prediction matrix calculation
                            point_prediction = model_pipeline.predict(input_df)[0]
                            
                            st.metric(
                                label="Predicted Malaria Prevalence (PfPR2-10)", 
                                value=f"{point_prediction:.2f}%"
                            )
                            
                            st.markdown("---")
                            st.markdown("**Environmental Predictor Metrics:**")
                            st.markdown(f"💧 **NDWI:** `{extracted_props.get('NDWI', 0):.4f}`")
                            st.markdown(f"🌿 **NDMI:** `{extracted_props.get('NDMI', 0):.4f}`")
                            st.markdown(f"🌡️ **LST:** `{extracted_props.get('LST', 0):.2f} °C`")
                            st.markdown(f"🌧️ **Rainfall:** `{extracted_props.get('Rainfall', 0):.1f} mm`")
                            st.markdown(f"⛰️ **Elevation:** `{extracted_props.get('Elevation', 0):.1f} m`")
                            st.markdown(f"🌊 **Distance to Water:** `{extracted_props.get('DistWater', 0)/1000:.2f} km`")
                        else:
                            st.warning("⚠️ Selected location is outside available coverage or administrative borders.")
                    except Exception as err:
                        st.error(f"Error extracting point attributes: {err}")
            else:
                st.write("Click any location inside the district bounds to view raw model inputs and custom coordinates.")