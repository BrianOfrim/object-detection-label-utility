import os
import pathlib
import re
import time
from typing import List
import shutil

from absl import app, flags
import numpy as np
import matplotlib.pyplot as plt

import s3_util
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
    "s3_annotation_dir",
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
    file_paths = [
        f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))
    ]
    if file_type is not None:
        file_paths = [f for f in file_paths if f.lower().endswith(file_type.lower())]
    return file_paths


def manifest_file_sort(manifest_file) -> int:
    match = re.match("[0-9]+", manifest_file)
    if not match:
        return 0
    return int(match[0])


def get_newest_manifest_path() -> str:
    manifest_files = get_files_from_dir(flags.FLAGS.local_manifest_dir)
    manifest_files = [
        f for f in manifest_files if f.lower().endswith(flags.FLAGS.manifest_file_type)
    ]
    if len(manifest_files) == 0:
        return None
    newest_manifest_file = sorted(manifest_files, key=manifest_file_sort, reverse=True)[
        0
    ]
    return os.path.join(flags.FLAGS.local_manifest_dir, newest_manifest_file)


def save_outputs(
    annotatedImages: List[AnnotatedImage],
    previous_manifest_path: str,
    start_time: int,
    use_s3: bool,
) -> None:
    # create a new manifest file
    new_manifest_path = os.path.join(
        flags.FLAGS.local_manifest_dir,
        "%i-manifest.%s" % (start_time, flags.FLAGS.manifest_file_type),
    )
    if previous_manifest_path is not None:
        shutil.copyfile(previous_manifest_path, new_manifest_path)
    else:
        open(new_manifest_path, "a").close()

    new_annotation_filepaths = []
    with open(new_manifest_path, "a") as manifest:
        for image in annotatedImages:
            annotation_filepath = image.write_to_pascal_voc()
            image_filename = os.path.basename(image.image_path)
            annotation_filename = (
                os.path.basename(annotation_filepath)
                if annotation_filepath is not None
                else "Invalid"
            )
            if annotation_filepath is not None:
                new_annotation_filepaths.append(annotation_filepath)
            manifest.write("%s,%s\n" % (image_filename, annotation_filename,))
    if use_s3:
        s3_util.upload_files(
            flags.FLAGS.s3_bucket_name,
            new_annotation_filepaths,
            flags.FLAGS.s3_annotation_dir,
        )
        s3_util.upload_files(
            flags.FLAGS.s3_bucket_name,
            [new_manifest_path],
            flags.FLAGS.s3_manifest_dir,
        )
        # ensure that all images have been uploaded
        s3_util.upload_files(
            flags.FLAGS.s3_bucket_name,
            [image.image_path for image in annotatedImages if image.valid],
            flags.FLAGS.s3_image_dir,
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
        if not s3_util.s3_bucket_exists(flags.FLAGS.s3_bucket_name):
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
        s3_images = s3_util.s3_get_object_names_from_dir(
            flags.FLAGS.s3_bucket_name,
            flags.FLAGS.s3_image_dir,
            flags.FLAGS.image_file_type,
        )
        s3_util.s3_download_files(
            flags.FLAGS.s3_bucket_name, s3_images, flags.FLAGS.local_image_dir
        )

        # Download any nest annotation files from s3
        s3_annotations = s3_util.s3_get_object_names_from_dir(
            flags.FLAGS.s3_bucket_name,
            flags.FLAGS.s3_annotation_dir,
            flags.FLAGS.annotation_file_type,
        )

        s3_util.s3_download_files(
            flags.FLAGS.s3_bucket_name,
            s3_annotations,
            flags.FLAGS.local_annotation_dir,
        )

        # Download any new manifests files from s3
        s3_manifests = s3_util.s3_get_object_names_from_dir(
            flags.FLAGS.s3_bucket_name, flags.FLAGS.s3_manifest_dir,
        )

        s3_util.s3_download_files(
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

    previous_manifest_file = get_newest_manifest_path()
    manifest_images = set()
    if previous_manifest_file is not None:
        with open(previous_manifest_file, "r") as manifest:
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
    save_outputs(annotated_images, previous_manifest_file, start_time, use_s3)


if __name__ == "__main__":
    app.run(main)
