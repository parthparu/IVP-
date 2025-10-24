import ast
import os
import cv2
import numpy as np
import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Supabase client
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  #-- top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  #-- bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  #-- top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  #-- bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img


def get_results_from_supabase():
    """
    Fetch detection results from Supabase database.
    
    Returns:
        pd.DataFrame: DataFrame containing the detection results.
    """
    try:
        response = supabase.table('detection_results').select('*').order('frame_nmr').order('car_id').execute()
        
        if response.data:
            df = pd.DataFrame(response.data)
            print(f"Successfully fetched {len(df)} records from Supabase")
            return df
        else:
            print("No data found in Supabase. Run main.py first.")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error fetching data from Supabase: {e}")
        return pd.DataFrame()


results = get_results_from_supabase()

if results.empty:
    print("Exiting script because no data was fetched from Supabase.")
    exit()

# load video
video_path = 'sample_short.mp4'
cap = cv2.VideoCapture(video_path)

# --- ✅ FIX: Changed the FourCC codec to 'avc1' for H.264 ---
fourcc = cv2.VideoWriter_fourcc(*'avc1')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./out1.mp4', fourcc, fps, (width, height))

license_plate = {}
for car_id in np.unique(results['car_id']):
    max_score = np.amax(results[results['car_id'] == car_id]['license_number_score'])
    best_entry = results[(results['car_id'] == car_id) & (results['license_number_score'] == max_score)]
    
    if not best_entry.empty:
        license_plate[car_id] = {
            'license_crop': None,
            'license_plate_number': best_entry['license_number'].iloc[0]
        }
        
        frame_to_read = best_entry['frame_nmr'].iloc[0]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_read)
        ret, frame = cap.read()

        if ret:
            bbox_str = best_entry['license_plate_bbox'].iloc[0]
            # Clean up the string before parsing
            clean_bbox_str = bbox_str.replace('[', '').replace(']', '').replace(',', ' ').strip()
            bbox_list = [float(i) for i in clean_bbox_str.split()]
            x1, y1, x2, y2 = bbox_list

            license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
            if license_crop.size > 0 and (y2 - y1) > 0:
                 license_crop = cv2.resize(license_crop, (int((x2 - x1) * 200 / (y2 - y1)), 200))
                 license_plate[car_id]['license_crop'] = license_crop


frame_nmr = -1
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# read frames
ret = True
while ret:
    ret, frame = cap.read()
    frame_nmr += 1
    if ret:
        df_ = results[results['frame_nmr'] == frame_nmr]
        for row_indx in range(len(df_)):
            car_id = df_.iloc[row_indx]['car_id']
            
            # Draw car bbox
            car_bbox_str = df_.iloc[row_indx]['car_bbox'].replace('[', '').replace(']', '').replace(',', ' ').strip()
            car_bbox_list = [float(i) for i in car_bbox_str.split()]
            car_x1, car_y1, car_x2, car_y2 = car_bbox_list
            draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 15,
                        line_length_x=200, line_length_y=200)

            # Draw license plate bbox
            lp_bbox_str = df_.iloc[row_indx]['license_plate_bbox'].replace('[', '').replace(']', '').replace(',', ' ').strip()
            lp_bbox_list = [float(i) for i in lp_bbox_str.split()]
            x1, y1, x2, y2 = lp_bbox_list
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)

            if car_id in license_plate and license_plate[car_id]['license_crop'] is not None:
                license_crop = license_plate[car_id]['license_crop']
                H, W, _ = license_crop.shape

                try:
                    # Position for license plate crop
                    y_start_crop = int(car_y1) - H - 20
                    y_end_crop = int(car_y1) - 20
                    x_start_crop = int((car_x1 + car_x2 - W) / 2)
                    x_end_crop = x_start_crop + W

                    # Position for text background
                    y_start_text_bg = y_start_crop - 80
                    y_end_text_bg = y_start_crop

                    if y_start_text_bg > 0 and x_start_crop > 0:
                        # Draw license plate crop
                        frame[y_start_crop:y_end_crop, x_start_crop:x_end_crop, :] = license_crop
                        # Draw white background for text
                        frame[y_start_text_bg:y_end_text_bg, x_start_crop:x_end_crop, :] = (255, 255, 255)

                        (text_width, text_height), _ = cv2.getTextSize(
                            license_plate[car_id]['license_plate_number'], cv2.FONT_HERSHEY_SIMPLEX, 1.5, 6)
                        
                        # Center text on the white background
                        text_x = x_start_crop + int((W - text_width) / 2)
                        text_y = y_start_text_bg + int((80 + text_height) / 2)
                        
                        cv2.putText(frame, license_plate[car_id]['license_plate_number'],
                                    (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 6)
                except Exception as e:
                    # This will catch errors if the overlay goes out of the frame boundaries
                    # print(f"Could not draw overlay for car_id {car_id}: {e}")
                    pass

        # Write the frame to the output video
        out.write(frame)

out.release()
cap.release()
cv2.destroyAllWindows()

print("✅ Visualization complete. The video 'out.mp4' has been saved.")