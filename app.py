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
        # 5. Extract Predictor Variables from GEE (Strictly Bounded)
        # ------------------------------------------
        # Get the geometry definition out early to optimize filtering
        aoi_geometry = aoi.geometry()

        # Sentinel-2 Imagery (Pre-clip collections to minimize server overhead)
        s2Collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
            .filterBounds(aoi_geometry)\
            .filterDate(start_date, end_date)\
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))\
            .select(["B3", "B8", "B11"])
        
        s2 = s2Collection.median().clip(aoi_geometry)
        
        # Calculate indices matching your 6-feature requirements
        ndwi = s2.normalizedDifference(["B3", "B8"]).rename("NDWI")
        ndmi = s2.normalizedDifference(["B8", "B11"]).rename("NDMI")
        
        # MODIS LST (°C) - Filter bounds explicitly
        lst = ee.ImageCollection("MODIS/061/MOD11A1")\
            .filterBounds(aoi_geometry)\
            .filterDate(start_date, end_date)\
            .select("LST_Day_1km")\
            .mean()\
            .multiply(0.02).subtract(273.15).rename("LST").clip(aoi_geometry)
            
        # CHIRPS Rainfall (mm) - Filter bounds explicitly
        rainfall = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")\
            .filterBounds(aoi_geometry)\
            .filterDate(start_date, end_date)\
            .select("precipitation")\
            .sum().rename("Rainfall").clip(aoi_geometry)
            
        # SRTM Elevation - Clip immediately
        elevation = ee.Image("USGS/SRTMGL1_003").select("elevation").rename("Elevation").clip(aoi_geometry)
        
        # Calculate water mask directly at 5km target scale to prevent timeout
        water = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(50).clip(aoi_geometry)
        water5km = water.reproject(crs=commonCRS, scale=pfprScale).unmask(0)
        distWater5km = water5km.fastDistanceTransform(256, "pixels", "squared_euclidean").sqrt().multiply(pfprScale).rename("DistWater").clip(aoi_geometry)

        # ------------------------------------------
        # 6. Create clean 5km Grid Points for Sampling
        # ------------------------------------------
        # Instead of reducing the image, we generate a clean coordinate grid at 5km resolution
        lonLat = ee.Image.pixelLonLat().reproject(crs=commonCRS, scale=pfprScale)
        
        # Extract the center points of the 5km grid cells within Karagwe
        grid_samples = lonLat.sample(
            region=aoi_geometry,
            scale=pfprScale,
            projection=commonCRS,
            factor=None,
            numPixels=None,
            seed=42,
            dropNulls=True,
            tileScale=2
        )

        # ------------------------------------------
        # 7. Extract Predictor Variables Safely (Point-Based Reduction)
        # ------------------------------------------
        band_names = ["NDWI", "NDMI", "LST", "Rainfall", "Elevation", "DistWater"]
        
        # Combine all our raw source layers directly
        raw_stack = ee.Image.cat([ndwi, ndmi, lst, rainfall, elevation, distWater5km]).toFloat()

        # Reduce the regions strictly around our pre-calculated 5km points
        pixel_samples = raw_stack.reduceRegions(
            collection=grid_samples,
            reducer=ee.Reducer.mean(),
            scale=fineScale,
            crs=commonCRS,
            tileScale=4
        ).getInfo()

        # Extract features into the Pandas DataFrame structure
        features_list = [feat["properties"] for feat in pixel_samples["features"]]
        
        if not features_list:
            pixel_data = pd.DataFrame()
        else:
            pixel_data = pd.DataFrame(features_list)
            
            # FIX: pixelLonLat saves coordinates as 'longitude' and 'latitude' inside properties!
            # If for some reason the keys are named differently by GEE, we map them safely:
            if "longitude" not in pixel_data.columns:
                if "lon" in pixel_data.columns:
                    pixel_data = pixel_data.rename(columns={"lon": "longitude", "lat": "latitude"})
                else:
                    st.error("Coordinate properties were not found in the extracted Earth Engine feature stack.")

        # Clean the dataset profile
        if not pixel_data.empty:
            # Keep only columns matching the model features and spatial coordinates
            valid_cols = band_names + ["longitude", "latitude"]
            pixel_data = pixel_data[[col for col in pixel_data.columns if col in valid_cols]]
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