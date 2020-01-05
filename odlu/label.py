import os
from typing import List
from absl import app, flags
import numpy as np
import matplotlib.pyplot as plt
from gui import GUI, AnnotatedImage, Category

flags.DEFINE_string(
    "input_image_dir", "../data/images", "Location of the image files to label."
)

flags.DEFINE_string(
    "input_labels_dir",
    "../data/labels.txt",
    "Path to the file containing the category labels.",
)

flags.DEFINE_string(
    "output_annotations_dir",
    "../data/annotations",
    "Directory to store the output annotations",
)

flags.DEFINE_string(
    "manifest_file", "../data/manifest.txt", "Location of the annotated image manifest"
)


def create_output_dir(dir_name) -> bool:
    if not os.path.isdir(dir_name) or not os.path.exists(dir_name):
        print("Creating output directory: %s" % dir_name)
        try:
            os.makedirs(dir_name)
        except OSError:
            print("Creation of the directory %s failed" % dir_name)
            return False
        else:
            print("Successfully created the directory %s " % dir_name)
            return True
    else:
        print("Output directory exists.")
        return True


def save_annotations(annotatedImages: List[AnnotatedImage]) -> None:
    # create a manifest file if it does not exit
    if not os.path.isfile(flags.FLAGS.manifest_file):
        open(flags.FLAGS.manifest_file, "a").close()

    with open(flags.FLAGS.manifest_file, "a") as manifest:
        for image in annotatedImages:
            image.write_to_pascal_voc()
            manifest.write(
                "%s,%s\n"
                % (os.path.basename(image.image_path), image.get_pascal_voc_filename(),)
            )


def main(unused_argv):

    fig = plt.figure()
    gui = GUI(fig)

    if not os.path.isfile(flags.FLAGS.input_labels_dir):
        print("Invalid category labels path.")
        return

    # read in the category labels
    category_labels = open(flags.FLAGS.input_labels_dir).read().splitlines()

    if len(category_labels) == 0:
        print("No label categories found")
        return

    category_colors = plt.get_cmap("hsv")(np.linspace(0, 0.9, len(category_labels)))

    for index, (name, color) in enumerate(zip(category_labels, category_colors)):
        gui.add_category(Category(name, tuple(color), str(index)))

    if not os.path.isdir(flags.FLAGS.input_image_dir):
        print("Invalid input image directory")
        return

    manifest_images = set()
    if os.path.isfile(flags.FLAGS.manifest_file):
        with open(flags.FLAGS.manifest_file, "r") as manifest:
            for line in manifest:
                manifest_images.add(line.split(",")[0].rstrip())

    # read in the names of the images to label
    for image_file in os.listdir(flags.FLAGS.input_image_dir):
        if (
            image_file.endswith(".jpg")
            and os.path.basename(image_file) not in manifest_images
        ):
            gui.add_image(
                AnnotatedImage(
                    os.path.join(flags.FLAGS.input_image_dir, image_file),
                    flags.FLAGS.output_annotations_dir,
                )
            )

    if len(gui.images) == 0:
        print("No input images found")
        return

    if not create_output_dir(flags.FLAGS.output_annotations_dir):
        print("Cannot create output annotations directory.")
        return

    annotated_images = gui.show()
    save_annotations(annotated_images)


if __name__ == "__main__":
    app.run(main)
