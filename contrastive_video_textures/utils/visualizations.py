import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image


def overlay_cmap_image(img, heatmap, cmap="jet", alpha=0.3):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img, mode="RGB")

    if isinstance(heatmap, np.ndarray):
        colorize = plt.get_cmap(cmap)

        # Normalize
        heatmap = heatmap - np.min(heatmap)
        heatmap = heatmap / np.max(heatmap)
        heatmap = colorize(heatmap, bytes=True)
        heatmap = Image.fromarray(heatmap[:, :, :3], mode="RGB")

    # Resize the heatmap to cover whole img. Map and Image should be of the same size
    heatmap = heatmap.resize((img.size[0], img.size[1]))

    # Display final overlayed output
    result = Image.blend(img, heatmap, alpha)
    result_t = torch.tensor(np.asarray(result)).permute(2, 0, 1)

    return result_t


def generate_html(
    results_dir,
    gt_paragraphs,
    gen_paragraphs,
    image_paths,
    epoch,
    repeat,
    batch_size,
    learning_rate,
    loss=0,
    image_size=300,
):
    # generate a html for each epoch to visualize everything
    html_filename = "res_par_epoch%d_rep%d_bs%d_lr_%f.html" % (
        epoch,
        repeat,
        batch_size,
        learning_rate,
    )
    html_filename = os.path.join(results_dir, html_filename)
    with open(html_filename, "w") as f:
        # write meta
        f.write(
            "<!DOCTYPE html>\n<html lang='en'>\n<head>\n<meta charset='UTF-8'>\n<title>\n</title>\n</head>\n<body>\n"
        )
        headline = "<h1> Paragraph Generation</h1>"
        f.write(headline)
        # write loss
        statline = "<h2> Validation Loss: %.4f</h2>" % (loss)
        f.write(statline)
        # write image, generated paragraph and gt_paragraph
        f.write("<h2>Validation Results</h2>")
        for idx, image_path in enumerate(image_paths):
            image_file = os.path.join("../../data/images/", image_path)
            imageline = "<img id='%s' height='%d' src='%s'> &nbsp &nbsp" % (
                image_path,
                image_size,
                image_file,
            )
            f.write(imageline)

            gen_paragraph = (
                "<p><b>Generated paragraph</b> %s </p>\n" % gen_paragraphs[idx]
            )
            f.write(gen_paragraph)

            gt_par = ". ".join(gt_paragraphs[image_path.split(".")[0]]["paragraph"])
            gt_paragraph = "<p><b>GT paragraph</b> %s </p>\n" % gt_par
            f.write(gt_paragraph)
        # finished
        f.write("</body>\n</html>\n")
