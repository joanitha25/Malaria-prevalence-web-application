import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import ee
import geemap
import os
import base64

# ==========================================
# 1. Earth Engine Authentication Setup
# ==========================================

try:
    # 1. Pull the clean, flat base64 sequence string from secrets
    b64_credentials = st.secrets["EARTHENGINE_CREDENTIALS_BASE64"]
    
    # 2. Decode the Base64 bytes back into a standard string layout
    decoded_bytes = base64.b64decode(b64_credentials)
    credentials_dict = json.loads(decoded_bytes.decode("utf-8"))
    
    service_account_email = credentials_dict["client_email"]
    
    # 3. Generate the local file mapping for the environment container
    with open("ee_credentials.json", "w") as f:
        json.dump(credentials_dict, f)
        
    # 4. Execute structural validation sequence
    credentials = ee.ServiceAccountCredentials(service_account_email, "ee_credentials.json")
    ee.Initialize(credentials)
    
except Exception as e:
    st.error(f"Earth Engine Authentication Failed. Error details: {e}")

# ==========================================
# 2. Model Loading (Cached)
# ==========================================
@st.cache_resource
def load_ml_pipeline():
    return joblib.load("best_rf_reduced_model.joblib")

model_pipeline = load_ml_pipeline()

# ==========================================
# 3. Main Interface Header Setup
# ==========================================
st.set_page_config(page_title="Malaria Prevalence Prediction", layout="wide")

# App Main Title
st.title("A Web Application for Malaria Prevalence Prediction")

# Initialize session state variables so data persists across tab switches
if "map_ready" not in st.session_state:
    st.session_state.map_ready = False
    st.session_state.smoothed_prediction_30m = None
    st.session_state.pixel_data = None
    st.session_state.aoi = None
    st.session_state.target_year = None
    st.session_state.target_district = None

# Create interactive tabs right below the main title instead of using a sidebar
about_tab, prediction_tab = st.tabs(["About the Application", "Malaria Prevalence Prediction"])

# ==========================================
# Tab 1: About the Application Panel
# ==========================================
with about_tab:
    st.header("About the Application")
    st.write(
        """
        This web application provides an automated platform for predicting the *Plasmodium falciparum* parasite 
        rate for children between 2 and 10 years (**PfPR2-10**) using satellite-derived environmental variables 
        and a trained Random Forest machine learning model. It integrates Google Earth Engine (GEE) to 
        automatically retrieve environmental predictors, including the Normalized Difference Water Index (NDWI), 
        Normalized Difference Moisture Index (NDMI), land surface temperature (LST), rainfall, elevation, 
        and distance to water bodies, for the selected district and year.
        
        Users can select the target district (Karagwe or Kyerwa) and surveillance year, after which the 
        application automatically extracts the required environmental data, generates malaria prevalence predictions, 
        and visualizes the results as an interactive map. The predicted PfPR2-10 values are displayed using a 
        continuous colour scale, from lower predicted malaria prevalence to higher predicted malaria prevalence. 
        Once the prediction pipeline is executed, a download link to retrieve the native model outputs as a GeoTIFF 
        (.tiff) file is provided within the mapping workspace for further spatial analysis.
        
        The prediction model was developed using satellite-derived environmental variables and validated using 
        spatial cross-validation to ensure reliable prediction across different geographical locations. The 
        generated malaria prevalence map is intended to support disease surveillance, environmental risk assessment, 
        and evidence-based decision-making by identifying areas that may require targeted malaria control interventions.
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
# Tab 2: Prediction Pipeline Workspace
# ==========================================
with prediction_tab:
    st.header("Malaria Prevalence Prediction Workspace")
    
    # Keep input fields clearly separated at the top of the workspace tab context
    target_year = st.selectbox("Select Target Surveillance Year", [2020, 2021, 2022, 2023, 2024, 2025])
    target_district = st.selectbox("Select Target District", ["Karagwe", "Kyerwa"])
    
    if st.button("Run Predictions"):
        with st.spinner("Extracting environmental indicators from GEE and executing pipeline..."):
            
            # ------------------------------------------
            # 4. Define Geographic Spatial Boundaries
            # ------------------------------------------
            districts = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level2")
            aoi = districts.filter(ee.Filter.eq("ADM2_NAME", target_district))
            aoi_geometry = aoi.geometry()
            
            start_date = ee.Date.fromYMD(target_year, 1, 1)
            end_date = ee.Date.fromYMD(target_year + 1, 1, 1)
            
            commonCRS = "EPSG:4326"
            fineScale = 30
            pfprScale = 5000
            commonProjection = ee.Projection(commonCRS).atScale(fineScale)

            # ------------------------------------------
            # 5. Extract Predictor Variables from GEE
            # ------------------------------------------
            s2Collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
                .filterBounds(aoi_geometry)\
                .filterDate(start_date, end_date)\
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))\
                .select(["B3", "B8", "B11"])
            
            s2 = s2Collection.median().clip(aoi_geometry)
            
            ndwi = s2.normalizedDifference(["B3", "B8"]).rename("NDWI")
            ndmi = s2.normalizedDifference(["B8", "B11"]).rename("NDMI")
            
            lst = ee.ImageCollection("MODIS/061/MOD11A1")\
                .filterBounds(aoi_geometry)\
                .filterDate(start_date, end_date)\
                .select("LST_Day_1km")\
                .mean()\
                .multiply(0.02).subtract(273.15).rename("LST").clip(aoi_geometry)
                
            rainfall = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")\
                .filterBounds(aoi_geometry)\
                .filterDate(start_date, end_date)\
                .select("precipitation")\
                .sum().rename("Rainfall").clip(aoi_geometry)
                
            elevation = ee.Image("USGS/SRTMGL1_003").select("elevation").rename("Elevation").clip(aoi_geometry)
            
            water = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(50).clip(aoi_geometry)
            water5km = water.reproject(crs=commonCRS, scale=pfprScale).unmask(0)
            distWater5km = water5km.fastDistanceTransform(256, "pixels", "squared_euclidean").sqrt().multiply(pfprScale).rename("DistWater").clip(aoi_geometry)

            # ------------------------------------------
            # 6. Create clean 5km Grid Points for Sampling
            # ------------------------------------------
            bounds = aoi_geometry.bounds().getInfo()['coordinates'][0]
            lons = [pt[0] for pt in bounds]
            lats = [pt[1] for pt in bounds]
            min_lon, max_lon = min(lons), max(lons)
            min_lat, max_lat = min(lats), max(lats)
            
            step = pfprScale / 111000.0
            grid_points = []
            lon_iter = min_lon
            while lon_iter <= max_lon:
                lat_iter = min_lat
                while lat_iter <= max_lat:
                    grid_points.append(ee.Feature(ee.Geometry.Point([lon_iter, lat_iter])))
                    lat_iter += step
                lat_iter += step
                
            raw_grid_collection = ee.FeatureCollection(grid_points)
            grid_samples = raw_grid_collection.filterBounds(aoi)

            # ------------------------------------------
            # 7. Extract Predictor Variables Safely (Point-Based Reduction)
            # ------------------------------------------
            band_names = ["NDWI", "NDMI", "LST", "Rainfall", "Elevation", "DistWater"]
            raw_stack = ee.Image.cat([ndwi, ndmi, lst, rainfall, elevation, distWater5km]).toFloat()

            pixel_samples = raw_stack.reduceRegions(
                collection=grid_samples,
                reducer=ee.Reducer.mean(),
                scale=fineScale,
                crs=commonCRS,
                tileScale=4
            ).getInfo()

            features_list = []
            for feat in pixel_samples["features"]:
                props = feat["properties"]
                if "geometry" in feat and feat["geometry"] is not None:
                    coords = feat["geometry"]["coordinates"]
                    props["longitude"] = coords[0]
                    props["latitude"] = coords[1]
                    features_list.append(props)
            
            if not features_list:
                pixel_data = pd.DataFrame()
            else:
                pixel_data = pd.DataFrame(features_list)

            if not pixel_data.empty:
                valid_cols = band_names + ["longitude", "latitude"]
                pixel_data = pixel_data[[col for col in pixel_data.columns if col in valid_cols]]
                pixel_data = pixel_data.dropna(subset=band_names)
                
                # ------------------------------------------
                # 8. Compute Model Predictions
                # ------------------------------------------
                X_pixels = pixel_data[band_names]
                pixel_data["predicted_PfPR"] = model_pipeline.predict(X_pixels)
                
                # ------------------------------------------
                # 9. Re-upload Predictions to GEE and Rasterize Cleanly
                # ------------------------------------------
                predicted_gee_layer = geemap.pandas_to_ee(pixel_data[["longitude", "latitude", "predicted_PfPR"]], latitude="latitude", longitude="longitude")
                grid_projection = ee.Projection(commonCRS).atScale(pfprScale)
                
                prediction_raster_5km = predicted_gee_layer.reduceToImage(
                    properties=["predicted_PfPR"], 
                    reducer=ee.Reducer.first()
                ).reproject(crs=grid_projection)
                
                smoothed_prediction_30m = prediction_raster_5km.resample('bilinear').reproject(crs=commonProjection).clip(aoi_geometry)
                
                # Assign to state parameters to persist rendering cross tabs
                st.session_state.pixel_data = pixel_data
                st.session_state.smoothed_prediction_30m = smoothed_prediction_30m
                st.session_state.aoi = aoi
                st.session_state.target_year = target_year
                st.session_state.target_district = target_district
                st.session_state.map_ready = True

    # Map display system strictly retained inside the prediction tab context block
    if st.session_state.map_ready:
        st.write("---")
        st.success(f"Successfully processed {st.session_state.target_district} District for {st.session_state.target_year}!")
        
        # Display Download option strictly at 5km model scale
        st.write("### 💾 Export Spatial Products:")
        try:
            raw_download_url = st.session_state.smoothed_prediction_30m.reproject(
                crs="EPSG:4326", 
                scale=5000
            ).getDownloadURL({
                'name': f'PfPR_{st.session_state.target_district}_{st.session_state.target_year}_5km',
                'scale': 5000,
                'crs': 'EPSG:4326',
                'filePerBand': False
            })
            st.markdown(f"[📥 Download Native 5km Model Raster (.tiff)]({raw_download_url})")
        except Exception as export_error:
            st.info("Download link generation timed out on remote GEE servers. Use the print tool below if this persists.")

        st.write("### Interactive Map Display:")
        
        import folium
        import streamlit.components.v1 as components
        from branca.element import Template, MacroElement

        if st.session_state.target_district == "Kyerwa":
            map_center = [-1.30, 30.85]
            map_zoom = 10
        else:
            map_center = [-1.59, 31.21]
            map_zoom = 9

        f_map = folium.Map(location=map_center, zoom_start=map_zoom, control_scale=True)
        
        aoi_map_id = ee.Image().paint(st.session_state.aoi, 0, 2).getMapId()
        folium.TileLayer(
            tiles=aoi_map_id['tile_fetcher'].url_format,
            attr='Google Earth Engine',
            name=f'{st.session_state.target_district} Border',
            overlay=True,
            control=True
        ).add_to(f_map)
        
        min_val = float(st.session_state.pixel_data["predicted_PfPR"].min())
        max_val = float(st.session_state.pixel_data["predicted_PfPR"].max())
        
        high_contrast_palette = ['#3288bd', '#99d594', '#e6f598', '#fee08b', '#fc8d59', '#d53e4f']
        vis_params = {
            'min': min_val,
            'max': max_val,
            'palette': high_contrast_palette
        }
        
        prediction_map_id = st.session_state.smoothed_prediction_30m.getMapId(vis_params)
        folium.TileLayer(
            tiles=prediction_map_id['tile_fetcher'].url_format,
            attr='Google Earth Engine',
            name=f'Predicted PfPR ({st.session_state.target_year})',
            overlay=True,
            control=True,
            opacity=0.85
        ).add_to(f_map)
        
        v_min = f"{min_val:.1f}%"
        v_max = f"{max_val:.1f}%"
        css_gradient = ", ".join(high_contrast_palette)

        legend_template = f"""
        {{% macro html(this, kwargs) %}}
        <div id='maplegend' class='maplegend' 
            style='position: absolute; z-index:9999; border:2px solid #bbb; background-color:rgba(255, 255, 255, 0.95);
            border-radius:8px; padding: 12px 15px; font-size:13px; right: 20px; bottom: 30px; width: 280px;
            font-family: "Source Sans Pro", sans-serif; box-shadow: 0 0 15px rgba(0,0,0,0.2);
            print-color-adjust: exact; -webkit-print-color-adjust: exact;'>
          
          <div class='legend-title' style='font-weight: bold; margin-bottom: 8px; text-align: center; color: #333;'>
            Malaria Prevalence (PfPR2-10)
          </div>
          
          <div class='gradient-bar' style='
            background: linear-gradient(to right, {css_gradient}) !important; 
            width: 100%; height: 18px; border-radius: 4px; border: 1px solid #777;
            print-color-adjust: exact; -webkit-print-color-adjust: exact;'>
          </div>
          
          <div class='legend-labels' style='margin-top: 5px; font-weight: 600; color: #444; display: flex; justify-content: space-between;'>
            <span>Low ({v_min})</span>
            <span>High ({v_max})</span>
          </div>
        </div>

        <div id='export-container' style='position: absolute; z-index:9999; top: 10px; left: 50px;'>
          <button onclick="window.print()" style='padding: 6px 12px; background: white; border: 2px solid #ccc; 
            border-radius: 4px; cursor: pointer; font-weight: bold; font-family: "Source Sans Pro", sans-serif; font-size: 12px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);'>
              📷 Save Map View
          </button>
        </div>
        {{% endmacro %}}
        """
        macro = MacroElement()
        macro._template = Template(legend_template)
        f_map.add_child(macro)

        folium.LayerControl().add_to(f_map)
        map_html = f_map._repr_html_()
        components.html(map_html, height=650, scrolling=True)