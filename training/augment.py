import cv2
import os
import pandas as pd
import numpy as np

def augment_brightness(image, factor=1.5):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def augment_flip(image):
    return cv2.flip(image, 1)  # 1 for horizontal flip

def augment_blur(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)

def apply_augmentation(image_path, linear_speed_x, angular_speed_z, output_folder):
    image = cv2.imread(image_path)
    
    # Check if augmentation is needed based on angular_speed_z
    if angular_speed_z < -0.016 or angular_speed_z > 0.025:
        # Augmentation 1: Brightness change
        augmented_image1 = augment_brightness(image)
        save_path1 = os.path.join(output_folder, os.path.basename(image_path).replace('.jpg', '_brightness.jpg'))
        cv2.imwrite(save_path1, augmented_image1)
        
        # Augmentation 2: Flipping
        augmented_image2 = augment_flip(image)
        save_path2 = os.path.join(output_folder, os.path.basename(image_path).replace('.jpg', '_flip.jpg'))
        cv2.imwrite(save_path2, augmented_image2)
        
        # Augmentation 3: Blurring
        augmented_image3 = augment_blur(image)
        save_path3 = os.path.join(output_folder, os.path.basename(image_path).replace('.jpg', '_blur.jpg'))
        cv2.imwrite(save_path3, augmented_image3)
        
        # Update CSV file
        # df.at[df['image name'] == os.path.basename(image_path), 'angular_speed_z'] = 0  # Reset original image's angular_speed_z

        # Update CSV for augmented images
        df_augmented = pd.DataFrame({'image name': [os.path.basename(save_path1), os.path.basename(save_path2), os.path.basename(save_path3)],
                                     'linear_speed_x': [linear_speed_x, linear_speed_x, linear_speed_x],
                                     'angular_speed_z': [angular_speed_z, -angular_speed_z, angular_speed_z]})

        # Concatenate to the original DataFrame
        # df_augmented = pd.concat([df_original, df_augmented], ignore_index=True)
        df_augmented.to_csv(os.path.join(output_folder, 'data_augmented.csv'), index=False)

# Read the original CSV file
csv_file_path = 'data.csv'
df_original = pd.read_csv(csv_file_path)

# Specify the output folder for augmented images
output_folder = 'output_folder'
os.makedirs(output_folder, exist_ok=True)

# Iterate through images in the folder
image_folder_path = "C:/Users/ishit/Documents/ROSbot_data_collection/datasets/temp_YES"
original_samples = len(df_original)

for image_file in os.listdir(image_folder_path):
    if image_file.endswith('.jpg'):
        image_path = os.path.join(image_folder_path, image_file)
        linear_speed_x = df_original.loc[df_original['image name'] == image_file, 'linear_speed_x'].values[0]
        angular_speed_z = df_original.loc[df_original['image name'] == image_file, 'angular_speed_z'].values[0]
        
        apply_augmentation(image_path, linear_speed_x, angular_speed_z, output_folder)

# Read the augmented CSV file
csv_augmented_file_path = os.path.join(output_folder, 'data_augmented.csv')
df_augmented = pd.read_csv(csv_augmented_file_path)

# Print the number of samples
new_samples = len(df_augmented) - original_samples
print(f"Number of original samples: {original_samples}")
print(f"Number of new samples after augmentation: {new_samples}")