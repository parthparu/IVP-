# License Plate Detection System with Supabase

This project has been updated to use Supabase instead of CSV files for data storage and retrieval.

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Supabase Setup

1. Create a new project at [supabase.com](https://supabase.com)
2. Go to your project's SQL Editor
3. Run the SQL schema from `database_schema.sql` to create the required tables
4. Go to Settings > API to get your project URL and anon key

### 3. Environment Configuration

1. Copy `env_example.txt` to `.env`
2. Fill in your Supabase credentials:
```
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_anon_key
```

### 4. Usage

1. **Run Detection**: `python main.py`
   - Processes video and stores results in Supabase
   - Requires `sample.mp4` video file and `models/license_plate_detector.pt` model

2. **Interpolate Missing Data**: `python add_missing_data.py`
   - Fills in missing frames using linear interpolation
   - Updates the Supabase database with interpolated results

3. **Visualize Results**: `python visualize.py`
   - Reads data from Supabase
   - Creates annotated video `out.mp4` with bounding boxes and license plate text

## Database Schema

The system uses a single table `detection_results` with the following columns:
- `id`: Primary key (auto-generated)
- `frame_nmr`: Frame number
- `car_id`: Vehicle tracking ID
- `car_bbox`: Vehicle bounding box coordinates
- `license_plate_bbox`: License plate bounding box coordinates
- `license_plate_bbox_score`: Detection confidence score
- `license_number`: Extracted license plate text
- `license_number_score`: OCR confidence score
- `created_at`: Timestamp (auto-generated)
- `updated_at`: Timestamp (auto-updated)

## Key Changes Made

- **util.py**: Replaced `write_csv()` with `write_to_supabase()`
- **main.py**: Updated to use Supabase for data storage
- **visualize.py**: Reads data from Supabase instead of CSV
- **add_missing_data.py**: Works with Supabase for interpolation
- **requirements.txt**: Added Supabase and python-dotenv dependencies

## Benefits of Using Supabase

- **Real-time**: Data is immediately available across all components
- **Scalable**: Handles large datasets efficiently
- **Queryable**: Easy to filter and analyze results
- **Persistent**: Data survives system restarts
- **Collaborative**: Multiple users can access the same data

