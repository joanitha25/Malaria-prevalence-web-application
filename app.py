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

# Inject custom CSS to enhance the overall visual hierarchy, font layout, and cards
st.markdown("""
    <style>
    .reportview-container {
        background-color: #f8f9fa;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #eef2f6;
        margin-bottom: 10px;
    }
    .instruction-card {
        background-color: #f1f3f5;
        padding: 20px;
        border-radius: 8px;
        border-left: 5px solid #2b6cb0;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

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
    st.session_state.map_raster = None
    st.session_state.clicked_point_data = None

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
    
    st.write("### 🗺️ Target District: Karagwe")
    about_map = folium.Map(location=[-1.59, 31.05], zoom_start=9, control_scale=True)
    try:
        districts = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level2")
        aoi_karagwe = districts.filter(ee.Filter.eq("ADM2_NAME", "Karagwe"))
        aoi_map_id = ee.Image().paint(aoi_karagwe, 0, 2).getMapId()
        folium.TileLayer(
            tiles=aoi_map_id['tile_fetcher'].url_format, attr='Google Earth Engine',
            name='Karagwe Boundary Layer', overlay=True
        ).add_to(about_map)
    except Exception:
        pass
    components.html(about_map._repr_html_(), height=400, scrolling=False)

# ==========================================
# View 2: Prediction Workspace
# ==========================================
elif current_view == "Malaria Prevalence Prediction Workspace":
    st.header("Malaria Prevalence Prediction Workspace")
    
    # ------------------------------------------------------------
    # 🔄 STATE BRIDGE ENGINE (Forces rerun when query parameters change)
    # ------------------------------------------------------------
    query_params = st.query_params
    if "click_data" in query_params:
        try:
            # Decode the base64 payload from the map click
            raw_json = base64.b64decode(query_params["click_data"]).decode("utf-8")
            new_click_data = json.loads(raw_json)
            
            # Check if this is a fresh click (prevents infinite rerun loops)
            if (st.session_state.clicked_point_data is None or 
                st.session_state.clicked_point_data.get("latitude") != new_click_data.get("latitude") or
                st.session_state.clicked_point_data.get("longitude") != new_click_data.get("longitude")):
                
                st.session_state.clicked_point_data = new_click_data
                # Force an immediate rerun so Streamlit updates the UI instantly
                st.rerun()
        except Exception as e:
            pass
    
    # Model Achievement Metrics Column Layout
    st.write("### 📊 Predictive Model Achieved Metrics")
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.metric(label="Mean Absolute Error (MAE)", value="0.592", delta="Good Precision")
    with m_col2:
        st.metric(label="Root Mean Squared Error (RMSE)", value="0.848", delta="Low Prediction Error")
    with m_col3:
        st.metric(label="R-squared (R² Coefficient)", value="84.7%", delta="Strong Variance Explained")

    st.markdown("The underlying machine learning model is trained on spatial covariate extractions from public databases, showing highly robust performance profiles across variable ecosystems.")
    st.write("---")

    def reset_map_state():
        st.session_state.map_ready = False
        st.session_state.clicked_point_data = None

    available_years = [2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027]
    
    target_year = st.selectbox(
        "Select Target Surveillance / Projection Year", 
        available_years, 
        index=available_years.index(st.session_state.target_year),
        on_change=reset_map_state
    )
    
    st.session_state.target_year = target_year
    st.session_state.target_district = "Karagwe"

    if st.button("Run Predictions", type="primary"):
        current_year = 2026
        spinner_msg = f"Extracting spatial diagnostics from Google Earth Engine for {st.session_state.target_district}..."
            
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
            
            # ------------------------------------------------------------
            # PERSONAL USER ASSETS PATH ROUTING MAP (Validation check)
            # ------------------------------------------------------------
            user_map_assets = {
                2020: "projects/ee-joanithakaijage/assets/MAP_PfPR2_10_2020",
                2021: "projects/ee-joanithakaijage/assets/MAP_PfPR2_10_2021",
                2022: "projects/ee-joanithakaijage/assets/MAP_PfPR2_10_2022",
                2023: "projects/ee-joanithakaijage/assets/MAP_PfPR2_10_2023",
                2024: "projects/ee-joanithakaijage/assets/MAP_PfPR2_10_2024"
            }
            
            # Map Raster references are strictly excluded for target projection years 2025-2027
            if st.session_state.target_year < 2025:
                map_target_year = min(st.session_state.target_year, 2024)
                selected_asset_path = user_map_assets[map_target_year]
                map_raster = ee.Image(selected_asset_path).select([0]).rename("MAP_PfPR").clip(aoi_geometry)
                raw_stack = ee.Image.cat([ndwi, ndmi, lst, rainfall, elevation, distWater5km, map_raster]).toFloat()
            else:
                map_raster = None
                raw_stack = ee.Image.cat([ndwi, ndmi, lst, rainfall, elevation, distWater5km]).toFloat()

            # ==================================================================
            # ROBUST SERVER-SIDE SAMPLING
            # ==================================================================
            lonlat_image = ee.Image.pixelLonLat().clip(aoi_geometry)
            data_and_coords = raw_stack.addBands(lonlat_image)
            
            sampling_features = data_and_coords.sample(
                region=aoi_geometry,
                scale=3000, 
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
                
                if "MAP_PfPR" in pixel_data.columns:
                    if pixel_data["MAP_PfPR"].max() <= 1.0:
                        pixel_data["MAP_PfPR"] = pixel_data["MAP_PfPR"] * 100.0
                
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
                
                st.session_state.pixel_data = pixel_data
                st.session_state.smoothed_prediction_30m = smoothed_prediction_30m
                st.session_state.map_raster = map_raster
                st.session_state.aoi = aoi
                st.session_state.map_ready = True

    # ==========================================
    # Rendering Interface (Map & Visualizations)
    # ==========================================
    if st.session_state.map_ready:
        st.write("---")
        
        st.subheader("🗺️ Interactive Map Display")
        
        # Interactive Map Display
        map_center = [-1.59, 31.05]
        map_zoom = 9

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
        
        # Layer 1: Model Machine Learning Predictions
        prediction_map_id = st.session_state.smoothed_prediction_30m.getMapId(vis_params)
        folium.TileLayer(
            tiles=prediction_map_id['tile_fetcher'].url_format, attr='Google Earth Engine',
            name=f'Predicted PfPR ({st.session_state.target_year})', overlay=True, opacity=0.85
        ).add_to(f_map)
        
        # Layer 2: Malaria Atlas Project Reference Layer (Only load for < 2025)
        if st.session_state.map_raster is not None:
            map_layer_id = st.session_state.map_raster.getMapId(vis_params) # Re-uses unified parameters
            folium.TileLayer(
                tiles=map_layer_id['tile_fetcher'].url_format, attr='Malaria Atlas Project User Asset',
                name=f'MAP Asset Baseline ({min(st.session_state.target_year, 2024)})', overlay=True, opacity=0.65
            ).add_to(f_map)
        
        # ------------------------------------------------------------
        # JAVASCRIPT DIRECT PARAM ON-CLICK HANDLER
        # ------------------------------------------------------------
        json_data_payload = st.session_state.pixel_data.to_json(orient="records")
        
        # Safe string replacement avoiding brace interpolation syntax issues
        click_macro_template = """
        {% macro html(this, kwargs) %}
        <script>
            document.addEventListener("DOMContentLoaded", function() {
                setTimeout(function() {
                    var map_instance = Object.values(window).find(val => val instanceof L.Map);
                    if (!map_instance) return;
                    
                    var spatial_grid_dataset = REPLACE_JSON_DATA_PAYLOAD;
                    
                    map_instance.on('click', function(e) {
                        var click_lat = e.latlng.lat;
                        var click_lon = e.latlng.lng;
                        
                        var nearest_pixel = null;
                        var minimal_distance = Infinity;
                        
                        for (var i = 0; i < spatial_grid_dataset.length; i++) {
                            var p = spatial_grid_dataset[i];
                            var d = Math.sqrt(Math.pow(p.latitude - click_lat, 2) + Math.pow(p.longitude - click_lon, 2));
                            if (d < minimal_distance) {
                                minimal_distance = d;
                                nearest_pixel = p;
                            }
                        }
                        
                        if (nearest_pixel && minimal_distance < 0.06) {
                            var baseline_val_str = nearest_pixel.MAP_PfPR ? nearest_pixel.MAP_PfPR.toFixed(2) + "%" : "N/A (Projection Period)";
                            var content = `
                                <div style="font-family: 'Source Sans Pro', sans-serif; font-size:12px; width:220px;">
                                    <h4 style="margin:2px 0; color:#2b6cb0;">🎯 Coordinates Inspector</h4>
                                    <hr style="margin:4px 0;"/>
                                    <b>Predicted PfPR2-10:</b> ${nearest_pixel.predicted_PfPR.toFixed(2)}%<br/>
                                    <b>MAP Baseline PfPR:</b> ${baseline_val_str}<br/>
                                    <b>LST Temp:</b> ${nearest_pixel.LST.toFixed(2)} °C<br/>
                                    <b>Rainfall:</b> ${nearest_pixel.Rainfall.toFixed(2)} mm<br/>
                                    <b>Elevation:</b> ${nearest_pixel.Elevation.toFixed(1)} m<br/>
                                    <b>Dist to Water:</b> ${nearest_pixel.DistWater.toFixed(1)} m<br/>
                                    <b>NDWI Index:</b> ${nearest_pixel.NDWI.toFixed(4)}<br/>
                                    <b>NDMI Index:</b> ${nearest_pixel.NDMI.toFixed(4)}
                                </div>
                            `;
                            
                            L.popup()
                             .setLatLng(e.latlng)
                             .setContent(content)
                             .openOn(map_instance);

                            // Construct state payload and dispatch query update variables directly back to Streamlit URL state
                            var payload = {
                                latitude: nearest_pixel.latitude,
                                longitude: nearest_pixel.longitude,
                                predicted_PfPR: nearest_pixel.predicted_PfPR,
                                MAP_PfPR: nearest_pixel.MAP_PfPR || null,
                                LST: nearest_pixel.LST,
                                Rainfall: nearest_pixel.Rainfall,
                                Elevation: nearest_pixel.Elevation,
                                DistWater: nearest_pixel.DistWater,
                                NDWI: nearest_pixel.NDWI,
                                NDMI: nearest_pixel.NDMI
                            };
                            
                            var b64_payload = btoa(JSON.stringify(payload));
                            window.parent.postMessage({
                                type: 'streamlit:set_query_params',
                                queryParams: { click_data: b64_payload }
                            }, '*');
                        }
                    });
                }, 1500);
            });
        </script>
        {% endmacro %}
        """.replace("REPLACE_JSON_DATA_PAYLOAD", json_data_payload)
        
        click_macro = MacroElement()
        click_macro._template = Template(click_macro_template)
        f_map.add_child(click_macro)

        # Legend Layout Elements Construction
        v_min, v_max = f"{min_val:.1f}%", f"{max_val:.1f}%"
        css_gradient = ", ".join(high_contrast_palette)

        # Removed '.format()' and safely injected standard values via sequential '.replace()'
        legend_template = """
        {% macro html(this, kwargs) %}
        <div id='maplegend' class='maplegend' style='position: absolute; z-index:9999; border:2px solid #bbb; background-color:rgba(255, 255, 255, 0.95); border-radius:8px; padding: 12px 15px; font-size:13px; right: 20px; bottom: 30px; width: 280px; font-family: "Source Sans Pro", sans-serif; box-shadow: 0 0 15px rgba(0,0,0,0.2);'>
          <div class='legend-title' style='font-weight: bold; margin-bottom: 8px; text-align: center; color: #333;'>Malaria Prevalence (PfPR2-10)</div>
          <div class='gradient-bar' style='background: linear-gradient(to right, REPLACE_GRADIENT_PALETTE) !important; background-image: linear-gradient(to right, REPLACE_GRADIENT_PALETTE) !important; width: 100%; height: 18px; border-radius: 4px; border: 1px solid #777;'></div>
          <div class='legend-labels' style='margin-top: 5px; font-weight: 600; color: #444; display: flex; justify-content: space-between;'>
            <span>Low (REPLACE_V_MIN)</span><span>High (REPLACE_V_MAX)</span>
          </div>
        </div>
        {% endmacro %}
        """
        
        legend_template = (
            legend_template
            .replace("REPLACE_GRADIENT_PALETTE", css_gradient)
            .replace("REPLACE_V_MIN", v_min)
            .replace("REPLACE_V_MAX", v_max)
        )
        
        legend_macro = MacroElement()
        legend_macro._template = Template(legend_template)
        f_map.add_child(legend_macro)

        folium.LayerControl().add_to(f_map)
        components.html(f_map._repr_html_(), height=600, scrolling=True)

        # 💾 Export Spatial Products Code block
        st.write("---")
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

    # =========================================================
    # How to Use & Guidelines (Scoped strictly inside workspace view)
    # =========================================================
    st.write("---")
    st.write("## 📖 How To Use & Model Interpretation Panel")

    with st.expander("ℹ️ How to Use This Module", expanded=True):
        st.markdown(f"""
        1. **Select Target Projection Year:** Choose the specific year (between 2020 and 2027) for projection. Current selected year is **{st.session_state.target_year}**.
        2. **Run Predictions:** Click the "Run Predictions" button to fetch the dynamic Earth Engine products and evaluate the spatial Random Forest model.
        3. **Interact with the Map:** Use the layers panel, pan, or zoom into the high-resolution layers. Use the **Scale Bar** on the bottom-left of the map to interpret predictive values.
        4. **Extract Values:** Click any point on the map to instantly retrieve localized environmental predictors and model calculations inside the map popup window.
        """)