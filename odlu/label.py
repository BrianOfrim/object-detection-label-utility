import os
import pathlib
import re
import time
from typing import List

from absl import app, flags
import numpy as np
import matplotlib.pyplot as plt

import s3_utilities
from gui import GUI, AnnotatedImage, Category

flags.DEFINE_string(
    "label_file_path",
    "../data/labels.txt",
    "Path to the file containing the category labels.",
)

flags.DEFINE_string(
    "local_image_dir", "../data/images", "Local directory of the image files to label."
)

flags.DEFINE_string(
    "local_annotation_dir",
    "../data/annotations",
    "Local directory of the image annotations",
)

flags.DEFINE_string(
    "local_manifest_dir",
    "../data/manifests/",
    "Local directory of the annotated image manifests",
)

flags.DEFINE_string(
    "s3_bucket_name", None, "S3 bucket to retrieve images from and upload manifest to."
)

flags.DEFINE_string("s3_image_dir", "data/images/", "Prefix of the s3 image objects.")

flags.DEFINE_string(
    "s3_annotations_dir",
    "data/annotations/",
    "Prefix of the s3 image annotation objects",
)
flags.DEFINE_string(
    "s3_manifest_dir",
    "data/manifests/",
    "Prefix of the s3 image annotation manifest objects",
)

flags.DEFINE_string("image_file_type", "jpg", "File type of the image files")

flags.DEFINE_string("annotation_file_type", "xml", "File type of the annotation files")

flags.DEFINE_string("manifest_file_type", "txt", "File type of the manifest files")


def get_files_from_dir(dir_path: str, file_type: str = None) -> List[str]:
    if not os.path.isdir(dir_path):
        return []
    file_paths = [f for f in os.listdir(dir_path) if os.path.isfile(join(dir_path, f))]
    if file_type is not None:
        file_paths = [f for f in file_paths if f.lower().endswith(file_type.lower())]
    return file_paths


# def manifest_file_sort(manifest_file): int:
#     if re.fi


def get_newest_manifest() -> str:
    manifest_files = get_files_from_dir(flags.FLAGS.local_manifest_dir)
    manifest_files = [
        f for f in manifest_files if f.lower().endswith(flags.FLAGS.manifest_file_type)
    ]
    if len(manifest_files) == 0:
        return None
    newest_manifest_file = sorted(
        manifest_files, key=lambda f: int(re.findall("[0-9]+", f)[0]), reverse=True
    )


def save_annotations(annotatedImages: List[AnnotatedImage]) -> None:
    # create a manifest file if it does not exit
    if not os.path.isfile(flags.FLAGS.manifest_file_path):
        open(flags.FLAGS.manifest_file_path, "a").close()

    with open(flags.FLAGS.manifest_file_path, "a") as manifest:
        for image in annotatedImages:
            image.write_to_pascal_voc()
            manifest.write(
                "%s,%s\n"
                % (os.path.basename(image.image_path), image.get_pascal_voc_filename(),)
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


def main(unused_argv):

    start_time = time.time()

    fig = plt.figure()
    gui = GUI(fig)

    use_s3 = True if flags.FLAGS.s3_bucket_name is not None else False

    if use_s3:
        if not s3_utilities.s3_bucket_exists(flags.FLAGS.s3_bucket_name):
            use_s3 = False
            print(
                "Bucket: %s either does not exist or you do not have access to it"
                % flags.FLAGS.s3_bucket_name
            )
        else:
            print(
                "Bucket: %s exists and you have access to it"
                % flags.FLAGS.s3_bucket_name
            )

    if use_s3:
        # Download new images from s3
        s3_images = s3_utilities.s3_get_object_names_from_dir(
            flags.FLAGS.s3_bucket_name,
            flags.FLAGS.s3_image_dir,
            flags.FLAGS.image_file_type,
        )
        s3_utilities.s3_download_files(
            flags.FLAGS.s3_bucket_name, s3_images, flags.FLAGS.local_image_dir
        )

        # Download any nest annotation files from s3
        s3_annotations = s3_utilities.s3_get_object_names_from_dir(
            flags.FLAGS.s3_bucket_name,
            flags.FLAGS.s3_annotation_dir,
            flags.FLAGS.annotation_file_type,
        )

        s3_utilities.s3_download_files(
            flags.FLAGS.s3_bucket_name,
            s3_annotations,
            flags.FLAGS.local_annotation_dir,
        )

        # Download any new manifests files from s3
        s3_manifests = s3_utilities.s3_get_object_names_from_dir(
            flags.FLAGS.s3_bucket_name, flags.FLAGS.s3_manifest_dir,
        )

        s3_utilities.s3_download_files(
            flags.FLAGS.s3_bucket_name, s3_manifests, flags.FLAGS.local_manifest_dir
        )

    if not os.path.isfile(flags.FLAGS.label_file_path):
        print("Invalid category labels path.")
        return

    # read in the category labels
    category_labels = open(flags.FLAGS.label_file_path).read().splitlines()

    if len(category_labels) == 0:
        print("No label categories found")
        return

    category_colors = plt.get_cmap("hsv")(np.linspace(0, 0.9, len(category_labels)))

    for index, (name, color) in enumerate(zip(category_labels, category_colors)):
        gui.add_category(Category(name, tuple(color), str(index)))

    if not os.path.isdir(flags.FLAGS.local_image_dir):
        print("Invalid input image directory")
        return

    manifest_images = set()
    if os.path.isfile(flags.FLAGS.manifest_file):
        with open(flags.FLAGS.manifest_file, "r") as manifest:
            for line in manifest:
                manifest_images.add(line.split(",")[0].rstrip())

    # read in the names of the images to label
    for image_file in os.listdir(flags.FLAGS.local_image_dir):
        if (
            image_file.endswith(flags.FLAGS.image_file_type)
            and os.path.basename(image_file) not in manifest_images
        ):
            gui.add_image(
                AnnotatedImage(
                    os.path.join(flags.FLAGS.local_image_dir, image_file),
                    flags.FLAGS.local_annotation_dir,
                )
            )

    if len(gui.images) == 0:
        print("No input images found")
        return

    if not create_output_dir(flags.FLAGS.local_annotation_dir):
        print("Cannot create output annotations directory.")
        return

    annotated_images = gui.show()
    save_annotations(annotated_images)


if __name__ == "__main__":
    app.run(main)
