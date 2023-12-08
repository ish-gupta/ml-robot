from PIL import Image
import pandas as pd
import os
from pathlib import Path

def split_and_update_data(folder_path):
    # Create output folder for split images
    output_folder = os.path.join(folder_path + "_YES")
    os.makedirs(output_folder, exist_ok=True)

    # Load the original data.csv
    original_data_path = os.path.join(folder_path, "data0.csv")
    original_data = pd.read_csv(original_data_path)

    # Iterate through images and split them
    for index, row in original_data.iterrows():
        image_name = row['image name']
        image_path = os.path.join(folder_path, image_name)
        original_image = Image.open(image_path)

        # Split the image vertically
        width, height = original_image.size
        left_image = original_image.crop((0, 0, width // 2, height))
        right_image = original_image.crop((width // 2, 0, width, height))

        # Save the split images
        left_image_path = os.path.join(output_folder, f"left_{image_name}")
        right_image_path = os.path.join(output_folder, f"right_{image_name}")

        left_image.save(left_image_path)
        right_image.save(right_image_path)

        # Update the data DataFrame with new entries for left and right images
        original_data.at[index, 'image name'] = f"left_{image_name}"
        new_data = pd.DataFrame([{'image name': f"right_{image_name}", 'linear_speed_x': row['linear_speed_x'], 'angular_speed_z': row['angular_speed_z']}])
        original_data = pd.concat([original_data, new_data], ignore_index=True)


    # Save the updated DataFrame to a new data.csv file
    updated_data_path = os.path.join(output_folder, "data.csv")
    original_data.to_csv(updated_data_path, index=False)

    print("Split and updated data successfully!")

if __name__ == "__main__":
    folder_path = "C:/Users/ishit/Documents/ROSbot_data_collection/datasets/training_data1" 
    split_and_update_data(folder_path)
