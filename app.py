import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import ee
import geemap
import os

# ==========================================
# 1. Earth Engine Authentication Setup
# ==========================================

try:
    # Streamlit reads the TOML table directly as a native Python dictionary
    credentials_dict = dict(st.secrets["EARTHENGINE_TOKEN"])
    
    # Advanced sanitization: handles both literal text escapes and true control codes
    raw_key = credentials_dict["private_key"]
    clean_key = raw_key.replace('\\n', '\n').replace('\n', '\n')
    credentials_dict["private_key"] = clean_key
    
    service_account_email = credentials_dict["client_email"]
    
    # Generate the credentials file cache locally on the runtime engine
    with open("ee_credentials.json", "w") as f:
        json.dump(credentials_dict, f)
        
    # Execute structural authentication sequence
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
# 3. User Interface Layout
# ==========================================
st.title("Automated GEE Malaria Surveillance Platform")
st.write("Extracting automated environmental indices for Karagwe District, Tanzania.")

# User inputs the target year for the surveillance update
target_year = st.selectbox("Select Target Surveillance Year", [2020, 2021, 2022, 2023, 2024, 2025])

if st.button("Generate 30m Visual Risk Map Profile"):
    with st.spinner(f"Extracting GEE layers and evaluating Random Forest model for {target_year}..."):
        
        # ------------------------------------------
        # 4. Define Geographic Spatial Boundaries
        # ------------------------------------------
        districts = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level2")
        aoi = districts.filter(ee.Filter.eq("ADM2_NAME", "Karagwe"))
        
        start_date = ee.Date.fromYMD(target_year, 1, 1)
        end_date = ee.Date.fromYMD(target_year + 1, 1, 1)
        
        commonCRS = "EPSG:4326"
        fineScale = 30
        pfprScale = 5000
        commonProjection = ee.Projection(commonCRS).atScale(fineScale)

        # ------------------------------------------
        # 5. Extract Predictor Variables from GEE
        # ------------------------------------------
        # Sentinel-2 Imagery
        s2Collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
            .filterBounds(aoi)\
            .filterDate(start_date, end_date)\
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
        
        s2 = s2Collection.median().clip(aoi)
        
        # Calculate indices matching your 6-feature requirements
        ndwi = s2.normalizedDifference(["B3", "B8"]).rename("NDWI")
        ndmi = s2.normalizedDifference(["B8", "B11"]).rename("NDMI")
        
        # MODIS LST (°C)
        lst = ee.ImageCollection("MODIS/061/MOD11A1")\
            .filterBounds(aoi)\
            .filterDate(start_date, end_date)\
            .select("LST_Day_1km")\
            .mean()\
            .multiply(0.02).subtract(273.15).rename("LST")
            
        # CHIRPS Rainfall (mm)
        rainfall = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")\
            .filterBounds(aoi)\
            .filterDate(start_date, end_date)\
            .select("precipitation")\
            .sum().rename("Rainfall")
            
        # SRTM Elevation
        elevation = ee.Image("USGS/SRTMGL1_003").select("elevation").rename("Elevation")
        
        # Distance to Water
        water = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(50).clip(aoi).unmask(0)
        distWater = water.fastDistanceTransform(30, "pixels", "squared_euclidean").sqrt().multiply(30).rename("DistWater")

        # ------------------------------------------
        # 6. Aggregate to 5km Model Resolution for Math
        # ------------------------------------------
        meanStack = ee.Image.cat([ndwi, ndmi, lst, rainfall, elevation]).toFloat().clip(aoi).reproject(crs=commonCRS, scale=fineScale)
        meanStack5km = meanStack.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=65535).reproject(crs=commonCRS, scale=pfprScale)
        
        distWater5km = distWater.toFloat().reproject(crs=commonCRS, scale=fineScale).reduceResolution(reducer=ee.Reducer.min(), maxPixels=65535).reproject(crs=commonCRS, scale=pfprScale).rename("DistWater")
        
        fullStack5km = meanStack5km.addBands(distWater5km).clip(aoi)

        # ------------------------------------------
        # 7. Extract Aggregated Pixels to Python Dataframe Safely
        # ------------------------------------------
        band_names = ["NDWI", "NDMI", "LST", "Rainfall", "Elevation", "DistWater"]

        # Add coordinate bands directly to the aggregated 5km stack
        pixel_grid = fullStack5km.addBands(ee.Image.pixelLonLat())

        # Extract data matching the exact 5km pixels directly
        pixel_samples = pixel_grid.reduceRegion(
            reducer=ee.Reducer.toList(),
            geometry=aoi.geometry(),
            scale=pfprScale,
            crs=commonCRS,
            maxPixels=1e6
        ).getInfo()

        # Convert the dictionary arrays directly into a structured Pandas DataFrame
        pixel_data = pd.DataFrame({
            "longitude": pixel_samples["longitude"],
            "latitude": pixel_samples["latitude"],
            "NDWI": pixel_samples["NDWI"],
            "NDMI": pixel_samples["NDMI"],
            "LST": pixel_samples["LST"],
            "Rainfall": pixel_samples["Rainfall"],
            "Elevation": pixel_samples["Elevation"],
            "DistWater": pixel_samples["DistWater"]
        })

        # Drop any rows that fall outside the exact vector boundary mask of Karagwe
        pixel_data = pixel_data.dropna(subset=band_names)
            
        # ------------------------------------------
        # 8. Compute Model Predictions
        # ------------------------------------------
        if pixel_data.empty:
            st.error("No pixel data could be extracted from GEE bounds.")
        else:
            # Map input array features directly
            X_pixels = pixel_data[band_names]
            
            # The pipeline scales and predicts PfPR using the loaded joblib model
            pixel_data["predicted_PfPR"] = model_pipeline.predict(X_pixels)
            
            # ------------------------------------------
            # 9. Re-upload Predictions to GEE and Downscale to 30m
            # ------------------------------------------
            # Convert the predicted column back into an Earth Engine Image object
            predicted_gee_layer = geemap.pandas_to_ee(pixel_data[["longitude", "latitude", "predicted_PfPR"]], latitude="latitude", longitude="longitude")
            
            # Rasterize the predicted points back into a 5km pixel grid canvas
            prediction_raster_5km = predicted_gee_layer.reduceToImage(properties=["predicted_PfPR"], reducer=ee.Reducer.first()).rename("PfPR_Prediction")
            
            # The Resolution Trick: Force the 5km model array to smoothly scale to 30m using Bilinear Resampling
            smoothed_prediction_30m = prediction_raster_5km.resample('bilinear').reproject(crs=commonCRS, scale=fineScale).clip(aoi)

            # ------------------------------------------
            # 10. Display Map on Interactive Streamlit Dashboard
            # ------------------------------------------
            st.success(f"Successfully processed {target_year} model pipeline!")
            st.write("### Interactive 30m Smoothed Risk Map:")
            
            # Generate a Leaflet map window via geemap
            Map = geemap.Map(center=[-1.59, 31.21], zoom=9)
            
            # Define visualization color gradients (Green to Red for low to high parasite risk)
            vis_params = {
                'min': float(pixel_data["predicted_PfPR"].min()),
                'max': float(pixel_data["predicted_PfPR"].max()),
                'palette': ['#1a9850', '#91cf60', '#d9ef8b', '#fee08b', '#fc8d59', '#d73027']
            }
            
            # Add layers to the map profile view
            Map.addLayer(aoi, {'color': 'black'}, 'Karagwe Border', True, 0.4)
            Map.addLayer(smoothed_prediction_30m, vis_params, f'Predicted PfPR ({target_year}) - 30m Smooth')
            Map.add_colorbar(vis_params, label="Parasite Rate Prediction (%)")
            
            # Render map to the Streamlit page canvas
            Map.to_streamlit(height=600)