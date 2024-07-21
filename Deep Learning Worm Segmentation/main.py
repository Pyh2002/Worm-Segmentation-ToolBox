import os
import cv2
import torch
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
import numpy as np

if __name__ == '__main__':
    # Make sure this neural network object is the exact same as the one in the train_segmentation.py
    NNetwork = smp.DeepLabV3Plus(
        encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        in_channels=1,
        classes=1,  # model output channels (number of classes in your dataset)
    )
    NNetwork = NNetwork.to(torch.device('cuda'))  # Send to the GPU
    # Load the state dictionary from training
    NNetwork.load_state_dict(torch.load('./best.pth'))
    NNetwork.eval()  # Put the model in evaluation mode
    #  Input frames transforms to match input from training
    transform = transforms.Compose([
        # Make sure the starting point for the input data is a PIL image
        transforms.ToPILImage(),
        # transforms.Resize((1024, 1024)),  # Change the input image size if needed
        transforms.ToTensor()  # Convert to a tensor
    ])

    video_folder_path = "./videos"  # Directory for the input videos
    #  Loop through each video in the video directory
    for file_name in os.listdir(video_folder_path):
        # Check if the file is an .avi video
        if file_name.endswith('.avi'):
            # Construct the full path to the video file
            video_path = os.path.join(video_folder_path, file_name)
            # Create Cv2 video object
            cap = cv2.VideoCapture(video_path)
            # Create Cv2 video writer object. 14.225 is the frame rate of the output video
            result = cv2.VideoWriter('./results/'+file_name,  # Output filepath
                                     cv2.VideoWriter_fourcc(*'MJPG'),
                                     14.225, (1920, 1440))
            # Loop through the video frames
            while cap.isOpened():
                # Read a frame from the video
                success, frame = cap.read()  # Extract current frame of the video

                if success:  # Make sure video opened correctly
                    # Convert the frame to greyscale
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    #  Apply transforms to the frame and send it to the GPU
                    input_tensor = transform(frame).unsqueeze(
                        0).to(torch.device('cuda'))
                    #  Make sure that the model is not modified while getting predictions
                    with torch.no_grad():
                        # Feed input tensor into model and get output prediction
                        output = NNetwork(input_tensor)
                    #  Send the prediction to the cpu and convert it to a numpy array
                    predicted_mask = output.squeeze().cpu().numpy()

                    # Threshold for uncertain pixels in the model
                    threshold = 0.1
                    # Apply the threshold to the prediction array and convert from float to 0-255 range
                    predicted_mask = (
                        predicted_mask > threshold).astype(np.uint8)*255

                    # Resize predicted mask to match the original frame size if needed
                    predicted_mask = cv2.resize(
                        predicted_mask, (frame.shape[1], frame.shape[0]))

                    # Apply the prediction as a mask on the original frame if needed
                    masked_frame = cv2.bitwise_and(
                        frame, frame, mask=predicted_mask)
                    # cv2.imshow(file_name, predicted_mask)  # Display the annotated frame if needed

                    # Convert the greyscale image to RGB to that it can write as a .avi
                    predicted_mask = cv2.cvtColor(
                        predicted_mask, cv2.COLOR_GRAY2RGB)
                    # Write the frame of the video to the output
                    result.write(predicted_mask.astype('uint8'))

                    # Break the loop if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                else:
                    # Break the loop if the end of the video is reached
                    break

            # Release the video capture objects and close the display window
            cap.release()
            result.release()
            # Print to console when a video is complete
            print(f"{file_name} Complete")
        cv2.destroyAllWindows()
