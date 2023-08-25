import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
from PIL import Image
import numpy as np
from glob import glob


def find_object(base, obj, result):
    # Create SIFT feature detector
    sift = cv2.xfeatures2d.SIFT_create()
    
    # Detect keypoints and compute descriptors for base and object images
    keypts_base, descr_base = sift.detectAndCompute(base["image"], None)
    keypts_obj, descr_obj = sift.detectAndCompute(obj["image"], None)

    # Create and configure brute-force k-nearest neighbors matcher for feature matching
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Perform k-nn matching and filter using Lowe's ratio test
    matches = flann.knnMatch(descr_obj, descr_base, k=2)
    matches = [[i] for i, j in matches if i.distance < 0.75 * j.distance]

    if len(matches) > 10:
        # Extract corresponding points for homography calculation
        src_pts = np.float32([keypts_obj[m[0].queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypts_base[m[0].trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Find homography transformation using RANSAC
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Define corners of the object image
        h, w, d = obj["image"].shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        
        # Transform object corners to match perspective in the base image
        dst = np.int32(cv2.perspectiveTransform(pts, M))
        
        # Find the top point among transformed corners for text placement
        matching_top_point = tuple(dst[np.argmin([x[0][1] for x in dst])][0])

        # Draw the matching polygon and add object name as text
        result = cv2.polylines(result, [dst], True, (100, 250, 200), 5, cv2.LINE_AA)
        cv2.putText(
            result,
            obj["name"],
            matching_top_point,
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            0.7,
            (255, 255, 255),
            1,
        )
        matching_img = cv2.drawMatchesKnn(
            obj["image"], keypts_obj, base["image"], keypts_base, matches, None, flags=2
        )
        return True, result, matching_img
    else:
        # Print a message if insufficient matches are found
        print("Could not find any satisfying matches for {}".format(obj["name"]))
        return False, None, None

       

def predict(image1, image2):
    # Convert input images to numpy arrays
    np_image1 = np.array(image1)
    np_image2 = np.array(image2)
    
    # Create a dictionary to represent the base image (shelf)
    base = {"name": "shelf", "image": np_image1}
    
    # Create a  dictionaries for object to be detected
    obj = {"name": "product", "image": np_image2}
    
    # Call the find_object function to detect the object in the base image
    # The function returns a status value, resulting image and a visualization of matching keypoints, 
    val, result, matching_img = find_object(base, obj, base["image"])
        
    # Return the resulting image with drawn matches and the visualization (matching_img)
    return result, matching_img



# Get a list of file paths for query examples from the "queries" directory
queries_examples = sorted(glob("assets/queries/*"))

# Get a list of file paths for target examples from the "targets" directory
target_examples = sorted(glob("assets/targets/*"))


#----------------------------------Gradio Integration-------------------------------------------#
block = gr.Blocks()

with block:
    with gr.Row():
        gr.Markdown("# Target Query Matching")

    with gr.Row():
        with gr.Tab("Matching"):
            with gr.Column():
                with gr.Row(scale=40):
                    input_image1 = gr.inputs.Image(type="pil", label="Query Image")
                    gr.Examples(
                        label="Query Examples",
                        examples=queries_examples,
                        fn=predict,
                        inputs=[input_image1],
                        outputs=[],
                        cache_examples=False,
                    )
                with gr.Row(scale=40):
                    input_image2 = gr.inputs.Image(type="pil", label="Target Image")
                    gr.Examples(
                        label="Target Examples",
                        examples=target_examples,
                        fn=predict,
                        inputs=[input_image2],
                        outputs=[],
                        cache_examples=False,
                    )
                with gr.Column(scale=20):
                    run_button = gr.Button(label="Run")
            with gr.Column():
                gallery = gr.Gallery(
                    label="Matched images",
                    show_label=False,
                    object_fit="contain",
                    height="auto",
                    preview=True,
                ).style(grid=[2], height="auto", width="auto", preview=True)

        val = run_button.click(
            fn=predict, inputs=[input_image1, input_image2], outputs=[gallery]
        )

block.launch()
