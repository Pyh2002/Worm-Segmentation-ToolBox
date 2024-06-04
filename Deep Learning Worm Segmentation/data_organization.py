import os
import cv2
import random
import shutil

def convert_to_binary(input_folder, output_folder):
    """ Converts the mask images to binary images

    Args:
        input_folder: (string) directory of masks to be converted to binary
        output_folder: (string) location for binary masks output
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            # Read the image
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            # Convert the image to binary
            _, binary_img = cv2.threshold(img, 2, 255, cv2.THRESH_BINARY)

            # Write the binary image to output folder
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, binary_img)

            print(f"{filename} converted to binary successfully.")

def move_random_images(image_folder, mask_folder,output_folder_masks,output_folder_images, num_images):
    """Select random image, mask pairs and move them to the output folders
        Image and masks should have the exact same filename

    Args:
        image_folder: (string) directory with images to be randomly selected and moved
        mask_folder: (string) directory with corresponding masks to be randomly selected and moved
        output_folder_masks: (string) output directory for the randomly selected masks
        output_folder_images: (string) output directory for the randomly selected images
        num_images: (int) number of randomly selected images
    """
    # Get list of common filenames
    image_filenames = [filename for filename in os.listdir(image_folder) if filename.endswith(".png")]
    mask_filenames = [filename for filename in os.listdir(mask_folder) if filename.endswith(".png")]
    common_filenames = list(set(image_filenames) & set(mask_filenames))

    # Select random filenames
    random_filenames = random.sample(common_filenames, min(num_images, len(common_filenames)))

    # Move selected images and masks to output folder
    for filename in random_filenames:
        image_src = os.path.join(image_folder, filename)
        mask_src = os.path.join(mask_folder, filename)
        image_dst = os.path.join(output_folder_images, filename)
        mask_dst = os.path.join(output_folder_masks, filename)
        shutil.move(image_src, image_dst)
        shutil.move(mask_src, mask_dst)
        print(f"Moved {filename}")

def create_training_val_test(images,masks,output,val_num,test_num):
    """ Creates folders and splits image and mask pairs into validation and test folders

    Args:
        images: (string) directory for the images to be split between training, validation, and testing
        masks:  (string) directory for the masks to be split between training, validation, and testing
        output: (string) the output directory for the training, validation, and testing directories
        val_num: (int) number of image, mask pairs to be sent to validation directories
        test_num: (int) number of image, mask pairs to be sent to test directories
    """
    # Create valuation directory
    convert_to_binary(masks,masks)
    val_output = os.path.join(output, "Val")
    val_output_masks = os.path.join(val_output, "Masks")
    val_output_images = os.path.join(val_output, "Images")
    if not os.path.exists(val_output_masks):
        os.mkdir(val_output_masks)
    if not os.path.exists(val_output_images):
        os.mkdir(val_output_images)
    # Move random images to valuation
    move_random_images(images,masks,val_output_masks,val_output_images,val_num)
    # Create test directory
    test_output = os.path.join(output, "Test")
    test_output_masks = os.path.join(test_output, "Masks")
    test_output_images = os.path.join(test_output, "Images")
    if not os.path.exists(test_output_masks):
        os.mkdir(test_output_masks)
    if not os.path.exists(test_output_images):
        os.mkdir(test_output_images)
    # Move random images to test
    move_random_images(images, masks, test_output_masks, test_output_images, test_num)
    # Create train directory
    train_output = os.path.join(output, "Train")
    train_output_masks = os.path.join(train_output, "Masks")
    train_output_images = os.path.join(train_output, "Images")
    if not os.path.exists(train_output_masks):
        os.mkdir(train_output_masks)
    if not os.path.exists(train_output_images):
        os.mkdir(train_output_images)

    # Move random images to test
    image_filenames = [filename for filename in os.listdir(images) if filename.endswith(".png")]
    mask_filenames = [filename for filename in os.listdir(masks) if filename.endswith(".png")]
    common_filenames = list(set(image_filenames) & set(mask_filenames))
    for filename in common_filenames:
        image_src = os.path.join(images, filename)
        mask_src = os.path.join(masks, filename)
        image_dst = os.path.join(train_output_images, filename)
        mask_dst = os.path.join(train_output_masks, filename)
        shutil.move(image_src, image_dst)
        shutil.move(mask_src, mask_dst)
        print(f"Moved {filename}")

if __name__ == '__main__':
    create_training_val_test("TrainingData/PNGImages","TrainingData/SegmentationClass","Training",77,0)