import math
import os
from dataclasses import dataclass
from typing import List, Dict

import matplotlib

matplotlib.use("TKAgg")
import matplotlib.image as mpimg
import matplotlib.lines as mlines
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from absl import app, flags
from matplotlib.widgets import Button
from pascal_voc_writer import Writer
from PIL import Image

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

# GLOBAL Constants
INVALID_IMAGE_COLOR = "#f58d42"
SELECTED_CATEGORY_COLOR = "#42f545"


@dataclass
class BBoxCorner:
    x: int
    y: int


@dataclass
class BBox:
    corner1: BBoxCorner
    corner2: BBoxCorner
    category: str


class AnnotatedImage:
    def __init__(self, path: str, bboxes: List[BBox]):
        self.path = path
        self.bboxes = bboxes
        self.valid = True

    def _get_pascal_voc_filename(self) -> str:
        return self.path.split("/")[-1].split(".")[0] + ".xml"

    def write_to_pascal_voc(self) -> None:
        if len(self.bboxes) == 0 or not self.valid:
            return
        width, height = Image.open(self.path).size
        writer = Writer(self.path, width, height)
        for bbox in self.bboxes:
            writer.addObject(
                bbox.category,
                bbox.corner1.x,
                bbox.corner1.y,
                bbox.corner2.x,
                bbox.corner2.y,
            )
        writer.save(
            os.path.join(
                flags.FLAGS.output_annotations_dir, self._get_pascal_voc_filename()
            )
        )


class Category:
    def __init__(self, name, color, keyboard_string):
        self.name = name
        self.color = color
        self.keyboard_string = keyboard_string
        self.ax = None
        self.button = None

    def select(self):
        if self.ax is not None:
            self.ax.spines["left"].set_linewidth(2)
            self.ax.spines["right"].set_linewidth(2)
            self.ax.spines["top"].set_linewidth(2)
            self.ax.spines["bottom"].set_linewidth(2)

            self.ax.spines["bottom"].set_color(SELECTED_CATEGORY_COLOR)
            self.ax.spines["top"].set_color(SELECTED_CATEGORY_COLOR)
            self.ax.spines["left"].set_color(SELECTED_CATEGORY_COLOR)
            self.ax.spines["right"].set_color(SELECTED_CATEGORY_COLOR)

    def deselect(self):
        if self.ax is not None:
            self.ax.spines["left"].set_linewidth(None)
            self.ax.spines["right"].set_linewidth(None)
            self.ax.spines["top"].set_linewidth(None)
            self.ax.spines["bottom"].set_linewidth(None)

            self.ax.spines["bottom"].set_color("black")
            self.ax.spines["top"].set_color("black")
            self.ax.spines["left"].set_color("black")
            self.ax.spines["right"].set_color("black")


class GUI:
    def __init__(self, fig):
        self.fig = fig
        self.categories: Dict[str, Category] = dict()
        self.current_category = None
        self.images: List[AnnotatedImage] = []
        self.image_index = 0
        self.fig.canvas.set_window_title("Label")
        self.image_ax = self.fig.add_axes([0.075, 0.25, 0.85, 0.65])
        self.invalid_ax = self.fig.add_axes([0.7, 0.01, 0.085, 0.075])
        self.prev_ax = self.fig.add_axes([0.8, 0.01, 0.085, 0.075])
        self.next_ax = self.fig.add_axes([0.9, 0.01, 0.085, 0.075])
        self.invalid_button = Button(self.invalid_ax, "Invalid", color="#f58d42")
        self.prev_button = Button(self.prev_ax, "Prev")
        self.next_button = Button(self.next_ax, "Next")

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_keypress)

    def show(self) -> None:
        # Display the first image
        self._display_image(self.images[self.image_index].path)
        # Select the first category as default
        self.current_category = next(iter(self.categories))
        self.categories[self.current_category].select()
        plt.show()
        print("Closed window")
        for i in range(self.image_index + 1):
            self.images[i].write_to_pascal_voc()

    def add_category(self, category: Category) -> None:
        category.ax = self.fig.add_axes(
            [(len(self.categories) * 0.12) + 0.05, 0.1, 0.11, 0.075]
        )
        category.button = Button(category.ax, category.name, color=category.color)
        self.categories[category.name] = category

    def add_image(self, image: AnnotatedImage) -> None:
        self.images.append(image)

    def _remove_incomplete_boxes(self, bboxes) -> None:
        for bbox in bboxes:
            if bbox.corner2 is None:
                bboxes.remove(bbox)

    def _display_image(self, path) -> None:
        img = Image.open(path)
        self.image_ax.imshow(img)
        self.image_ax.set_title(path.split("/")[-1])
        self.fig.canvas.draw()

    def _next_image(self, event) -> None:
        self._remove_incomplete_boxes(self.images[self.image_index].bboxes)
        self._clear_all_lines()
        self.image_index += 1
        self._display_image(self.images[self.image_index].path)
        self._draw_bounding_boxes(self.images[self.image_index].bboxes)
        self._draw_image_border()

    def _prev_image(self, event) -> None:
        self._clear_all_lines()
        if self.image_index != 0:
            self._remove_incomplete_boxes(self.images[self.image_index].bboxes)
            self.image_index -= 1
            self._display_image(self.images[self.image_index].path)
            self._draw_bounding_boxes(self.images[self.image_index].bboxes)
            self._draw_image_border()

    def _format_corners(self, bbox) -> None:
        x_min = min(bbox.corner1.x, bbox.corner2.x)
        y_min = min(bbox.corner1.y, bbox.corner2.y)
        x_max = max(bbox.corner1.x, bbox.corner2.x)
        y_max = max(bbox.corner1.y, bbox.corner2.y)
        bbox.corner1.x = x_min
        bbox.corner1.y = y_min
        bbox.corner2.x = x_max
        bbox.corner2.y = y_max

    def _clear_all_lines(self) -> None:
        [l.remove() for l in reversed(self.image_ax.lines)]

    def _draw_corner_1_lines(self, corner) -> None:
        x_min, x_max = self.image_ax.get_xbound()
        y_min, y_max = self.image_ax.get_ybound()
        v_line = mlines.Line2D([corner.x, corner.x], [y_min, y_max], linestyle="dashed")
        h_line = mlines.Line2D([x_min, x_max], [corner.y, corner.y], linestyle="dashed")
        self.image_ax.add_line(v_line)
        self.image_ax.add_line(h_line)
        self.fig.canvas.draw()

    def _draw_bounding_boxes(self, bboxes) -> None:
        # clear all current boxes
        [p.remove() for p in reversed(self.image_ax.patches)]
        # redraw the boxes
        for bbox in bboxes:
            if bbox.corner2 is None:
                continue
            height = bbox.corner2.y - bbox.corner1.y
            width = bbox.corner2.x - bbox.corner1.x
            lower_left = (bbox.corner1.x, bbox.corner1.y)
            color = self.categories[bbox.category].color
            rect = patches.Rectangle(
                lower_left,
                width,
                height,
                linewidth=2,
                edgecolor=color,
                facecolor="none",
            )
            self.image_ax.add_patch(rect)
        self.fig.canvas.draw()

    def _handle_bbox_entry(self, event) -> None:
        bboxes = self.images[self.image_index].bboxes
        if len(bboxes) > 0 and bboxes[-1].corner2 is None:
            self._clear_all_lines()
            bboxes[-1].corner2 = BBoxCorner(
                math.floor(event.xdata), math.floor(event.ydata)
            )
            self._format_corners(bboxes[-1])
            self._draw_bounding_boxes(bboxes)
        else:
            bboxes.append(
                BBox(
                    BBoxCorner(math.floor(event.xdata), math.floor(event.ydata)),
                    None,
                    self.current_category,
                )
            )
            self._draw_corner_1_lines(bboxes[-1].corner1)

    def _draw_invalid_image_border(self) -> None:
        print("invalid boarder")
        self.image_ax.spines["left"].set_linewidth(5)
        self.image_ax.spines["right"].set_linewidth(5)
        self.image_ax.spines["top"].set_linewidth(5)
        self.image_ax.spines["bottom"].set_linewidth(5)

        self.image_ax.spines["bottom"].set_color(INVALID_IMAGE_COLOR)
        self.image_ax.spines["top"].set_color(INVALID_IMAGE_COLOR)
        self.image_ax.spines["left"].set_color(INVALID_IMAGE_COLOR)
        self.image_ax.spines["right"].set_color(INVALID_IMAGE_COLOR)

    def _draw_valid_image_border(self) -> None:
        print("Valid boarder")
        self.image_ax.spines["left"].set_linewidth(None)
        self.image_ax.spines["right"].set_linewidth(None)
        self.image_ax.spines["top"].set_linewidth(None)
        self.image_ax.spines["bottom"].set_linewidth(None)

        self.image_ax.spines["bottom"].set_color("black")
        self.image_ax.spines["top"].set_color("black")
        self.image_ax.spines["left"].set_color("black")
        self.image_ax.spines["right"].set_color("black")

    def _draw_image_border(self):
        if not self.images[self.image_index].valid:
            self._draw_invalid_image_border()
        else:
            self._draw_valid_image_border()

    def _toggle_image_validation(self, event) -> None:
        self.images[self.image_index].valid = not self.images[self.image_index].valid
        self.images[self.image_index].bboxes.clear()
        self._draw_bounding_boxes(self.images[self.image_index].bboxes)
        self._draw_image_border()

    def _on_click(self, event) -> None:
        # verify that the click was inbounds for an axes
        if event.xdata is None or event.ydata is None or event.inaxes is None:
            print("Invalid selection.")
        elif event.inaxes == self.image_ax:
            self._handle_bbox_entry(event)
        elif event.inaxes == self.next_ax:
            self._next_image(event)
        elif event.inaxes == self.prev_ax:
            self._prev_image(event)
        elif event.inaxes == self.invalid_ax:
            self._toggle_image_validation(event)
        else:
            for category_name, category in self.categories.items():
                if event.inaxes == category.ax:
                    self.current_category = category_name
                    # Deselect all buttons
                    [c.deselect() for _, c in self.categories.items()]
                    # Select the clicked button
                    category.select()
                    break
        self.fig.canvas.draw()

    def _on_keypress(self, event) -> None:
        print("press", event.key)
        if event.key == "d":
            self._next_image(event)
        elif event.key == "a":
            self._prev_image(event)
        elif event.key == "w" or event.key == "escape":
            self._clear_all_lines()
            if len(self.images[self.image_index].bboxes) == 0:
                print("No more bounding boxes to clear")
            elif self.images[self.image_index].bboxes[-1].corner2 is None:
                print("Remove corner 1 guidelines")
                self._remove_incomplete_boxes(self.images[self.image_index].bboxes)
            else:
                print("Remove corner 2")
                self.images[self.image_index].bboxes[-1].corner2 = None
                self._draw_corner_1_lines(
                    self.images[self.image_index].bboxes[-1].corner1
                )
            self._draw_bounding_boxes(self.images[self.image_index].bboxes)

        for category_name, category in self.categories.items():
            if event.key == category.keyboard_string:
                self.current_category = category_name
                print("Current category: %s" % self.current_category)
                [c.deselect() for _, c in self.categories.items()]
                category.select()
        # Redraw the figure
        self.fig.canvas.draw()


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

    # read in the names of the images to label
    for image_file in os.listdir(flags.FLAGS.input_image_dir):
        if image_file.endswith(".jpg"):
            gui.add_image(
                AnnotatedImage(
                    os.path.join(flags.FLAGS.input_image_dir, image_file), []
                )
            )

    if len(gui.images) == 0:
        print("No input images found")
        return

    if not create_output_dir(flags.FLAGS.output_annotations_dir):
        print("Cannot create output annotations directory.")
        return

    gui.show()


if __name__ == "__main__":
    app.run(main)
