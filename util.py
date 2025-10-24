import string
import easyocr
import os
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Supabase client
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}


def write_to_supabase(results):
    """
    Write the results to Supabase database.

    Args:
        results (dict): Dictionary containing the results.
    """
    try:
        data_to_insert = []

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                car_data = results[frame_nmr][car_id]
                if ('car' in car_data and 
                    'license_plate' in car_data and 
                    'text' in car_data['license_plate']):

                    car_bbox = car_data['car']['bbox']
                    license_plate_bbox = car_data['license_plate']['bbox']
                    license_plate_text = car_data['license_plate']['text']
                    bbox_score = car_data['license_plate']['bbox_score']
                    text_score = car_data['license_plate']['text_score']

                    car_bbox_str = f"[{car_bbox[0]} {car_bbox[1]} {car_bbox[2]} {car_bbox[3]}]"
                    license_plate_bbox_str = f"[{license_plate_bbox[0]} {license_plate_bbox[1]} {license_plate_bbox[2]} {license_plate_bbox[3]}]"

                    data_to_insert.append({
                        'frame_nmr': int(frame_nmr),
                        'car_id': int(car_id),
                        'car_bbox': car_bbox_str,
                        'license_plate_bbox': license_plate_bbox_str,
                        'license_plate_bbox_score': float(bbox_score),
                        'license_number': license_plate_text,
                        'license_number_score': float(text_score)
                    })

        if data_to_insert:
            response = supabase.table('detection_results').insert(data_to_insert).execute()
            print(f"Successfully inserted {len(data_to_insert)} records to Supabase")
            print("Response from Supabase:", response)
            return response
        else:
            print("No valid data to insert")
            return None

    except Exception as e:
        print(f"Error writing to Supabase: {e}")
        return None



def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    if len(text) != 7:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
       (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
        return True
    else:
        return False


def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_


def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """

    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace(' ', '')

        if license_complies_format(text):
            return format_license(text), score

    return None, None


def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1
