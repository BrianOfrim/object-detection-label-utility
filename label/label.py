import os
import sys
from dataclasses import dataclass
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import matplotlib.image as mpimg
from matplotlib.widgets import Button
from absl import app
from absl import flags
from PIL import Image
from pascal_voc_writer import Writer

@dataclass
class BBoxCorner:
    x: int
    y: int

@dataclass
class BBox:
    corner1: BBoxCorner
    corner2: BBoxCorner
    category_index: int 

@dataclass
class AnnotatedImage:
    path: str
    boundingBoxes: list

class Category:
    def __init__(self, name, color, keyboard_string):
        self.name = name
        self.color = color
        self.keyboard_string = keyboard_string
        self.ax = None
        self.button = None
    def select(self):
        print('Button pushed: %s' % self.name)
        if(self.ax is not None):
            self.ax.spines['left'].set_linewidth(2)
            self.ax.spines['right'].set_linewidth(2)
            self.ax.spines['top'].set_linewidth(2)
            self.ax.spines['bottom'].set_linewidth(2)
            
            self.ax.spines['bottom'].set_color('#42f545')
            self.ax.spines['top'].set_color('#42f545')
            self.ax.spines['left'].set_color('#42f545')
            self.ax.spines['right'].set_color('#42f545')

    def deselect(self):
         if(self.ax is not None):
            self.ax.spines['left'].set_linewidth(None)
            self.ax.spines['right'].set_linewidth(None)
            self.ax.spines['top'].set_linewidth(None)
            self.ax.spines['bottom'].set_linewidth(None)
            
            self.ax.spines['bottom'].set_color('black')
            self.ax.spines['top'].set_color('black')
            self.ax.spines['left'].set_color('black')
            self.ax.spines['right'].set_color('black')
       
item_categories = []

flags.DEFINE_string(
    'input_image_dir',
    '../data/images',
    'Location of the image files to lable.'
)

flags.DEFINE_string(
    'input_labels_dir',
    '../data/labels.txt',
    'Path to the file containing the category labels.'
)

flags.DEFINE_string(
    'output_annotations_dir',
    '../data/annotations',
    'Directory to store the output annotations'
)

current_category_index = 0
input_images = []
current_image_index = 0

fig = plt.figure()
fig.canvas.set_window_title('Label')

im_ax = plt.axes([0.075, 0.15, 0.85, 0.75])

def get_base_filename(path):
    file_name = path.split('/')[-1]
    return file_name.split('.')[0]

def create_output_dir(dir_name) -> bool:
    if(not os.path.isdir(dir_name) or not os.path.exists(dir_name)):
        print('Creating output directory: %s' % dir_name)
        try:
            os.makedirs(dir_name)
        except OSError:
            print ("Creation of the directory %s failed" % dir_name)
            return False
        else:
            print ("Successfully created the directory %s " % dir_name)
            return True
    else:
        print('Output directory exists.')
        return True

def next_image(event):
    global current_image_index
    print('Next')
    remove_incomplete_boxes(input_images[current_image_index].boundingBoxes)
    clear_all_lines()
    current_image_index += 1
    display_image(input_images[current_image_index].path)
    draw_bounding_boxes(input_images[current_image_index].boundingBoxes) 

def prev_image(event):
    global current_image_index
    clear_all_lines()
    if (current_image_index == 0):
        print('Already at the start')
    else:
        print('Previous')
        remove_incomplete_boxes(input_images[current_image_index].boundingBoxes)
        current_image_index -= 1
        display_image(input_images[current_image_index].path)
        draw_bounding_boxes(input_images[current_image_index].boundingBoxes) 

def draw_bounding_boxes(bboxes):
    # clear all current boxes
    [p.remove() for p in reversed(im_ax.patches)] 
    
    # redraw the boxes
    for bbox in bboxes:
        height = bbox.corner2.y - bbox.corner1.y 
        width = bbox.corner2.x - bbox.corner1.x
        lower_left = (bbox.corner1.x, bbox.corner1.y)
        item_categories[bbox.category_index].color
        color = item_categories[bbox.category_index].color
        rect = patches.Rectangle(lower_left, width, height,
                linewidth=2,edgecolor=color,facecolor='none')
        im_ax.add_patch(rect)
    
    fig.canvas.draw()

def remove_incomplete_boxes(bboxes):
    # Clear last bbox if it is incomplete
    for bbox in bboxes:
        if bbox.corner2 is None:
            bboxes.remove(bbox)

def display_image(path):
    # Load and show the new image
    img = mpimg.imread(path)
    im_ax.imshow(img)
    # Update the title
    im_ax.set_title(path.split('/')[-1])
    # Redraw the figure
    fig.canvas.draw()

def clear_all_lines():
     [l.remove() for l in reversed(im_ax.lines)]

def draw_corner_1_lines(bboxCorner):
    xmin, xmax = im_ax.get_xbound()
    ymin, ymax = im_ax.get_ybound()
    
    vline = mlines.Line2D([bboxCorner.x ,bboxCorner.x],
            [ymin,ymax], linestyle='dashed')
    hline = mlines.Line2D([xmin,xmax], 
            [bboxCorner.y,bboxCorner.y], linestyle='dashed')

    im_ax.add_line(vline)
    im_ax.add_line(hline)

    fig.canvas.draw()

def format_corners(bbox):
    xmin = min(bbox.corner1.x, bbox.corner2.x)
    ymin = min(bbox.corner1.y, bbox.corner2.y)
    xmax = max(bbox.corner1.x, bbox.corner2.x)
    ymax = max(bbox.corner1.y, bbox.corner2.y)
    bbox.corner1.x = xmin
    bbox.corner1.y = ymin
    bbox.corner2.x = xmax
    bbox.corner2.y = ymax

def handle_bbox_entry(event):
    # get the current bounding box list
    bboxes = input_images[current_image_index].boundingBoxes
    if(len(bboxes) > 0 and bboxes[-1].corner2 is None):
        clear_all_lines()
        bboxes[-1].corner2 = BBoxCorner(math.floor(event.xdata),
            math.floor(event.ydata))
        format_corners(bboxes[-1])
        draw_bounding_boxes(bboxes)    
    else:
        bboxes.append(BBox(BBoxCorner(
            math.floor(event.xdata), math.floor(event.ydata)), None, 
                current_category_index))
        draw_corner_1_lines(bboxes[-1].corner1)

def keypress(event):
    global current_category_index
    print('press', event.key)
    if event.key == 'd':
        next_image(event)
    elif event.key == 'a':
        prev_image(event)
    elif event.key == 'w' or event.key == 'escape':
        clear_all_lines()
        if(len(input_images[current_image_index].boundingBoxes) == 0):
            print('No more bounding boxes to clear')
        elif input_images[current_image_index].boundingBoxes[-1].corner2 is None:
            print('Remove corner 1 guidelines')
        else:
            print('Remove latest bounding box')
            input_images[current_image_index].boundingBoxes.pop()
        remove_incomplete_boxes(input_images[current_image_index].boundingBoxes)
        draw_bounding_boxes(input_images[current_image_index].boundingBoxes) 

    for category_index, category in enumerate(item_categories):
        if event.key == category.keyboard_string:
            current_category_index = category_index
            print('Current category: %s' % 
                    item_categories[current_category_index].name)
    # Redraw the figure
    fig.canvas.draw()

def handle_click(event):
    global current_category_index
    # verify that the click was inbounds for an axes
    if event.xdata is None or event.ydata is None or event.inaxes is None:
        print('Invalid selection.')
        return

    if event.inaxes == im_ax:
        handle_bbox_entry(event)
        fig.canvas.draw()
        return
    
    for category_index, category in enumerate(item_categories):
        if event.inaxes == category.ax:
            # Deselect all buttons
            [c.deselect() for c in item_categories]
            # Select the clicked button
            category.select()
            current_category_index = category_index
            return

def on_click(event):
    handle_click(event)
    # Redraw the figure
    fig.canvas.draw()

def main(unused_argv):

    if not os.path.isfile(flags.FLAGS.input_labels_dir):
        print('Invalid category labels path.')
        return

    # read in the category labels
    category_labels = open(flags.FLAGS.input_labels_dir).read().splitlines()

    if (len(category_labels) == 0):
        print('No label categories found')
        return

    category_colors = plt.get_cmap('hsv')(
        np.linspace(0, 0.9, len(category_labels)))

    for index, (name, color) in enumerate(zip(category_labels, category_colors)):
        item_categories.append(Category(name, tuple(color), str(index)))    

    if not os.path.isdir(flags.FLAGS.input_image_dir):
        print('Invalid input image directory')
        return

    # read in the names of the images to label
    for image_file in os.listdir(flags.FLAGS.input_image_dir):
        if image_file.endswith(".jpg"):
            input_images.append(AnnotatedImage(
                os.path.join(flags.FLAGS.input_image_dir, image_file), []))

    if (len(input_images) == 0):
        print('No input images found')
        return

    if not create_output_dir(flags.FLAGS.output_annotations_dir):
        print('Cannot create output annotations directory.')
        return
 
    axprev = plt.axes([0.8, 0.01, 0.085, 0.075])
    axnext = plt.axes([0.9, 0.01, 0.085, 0.075])
    bprev = Button(axprev, 'Prev')
    bnext = Button(axnext, 'Next')

    cid_onclick= fig.canvas.mpl_connect('button_press_event', on_click)
    cid_keypress = fig.canvas.mpl_connect('key_press_event', keypress)
    
    bprev.on_clicked(prev_image)
    bnext.on_clicked(next_image)
 
    for item_index, category_item in enumerate(item_categories):
        category_item.ax = plt.axes([(item_index * 0.11) + 0.05, 0.01, 0.1, 0.075])
        category_item.button = Button(category_item.ax, category_item.name,
                color=category_item.color)

    # Display the first image
    display_image(input_images[current_image_index].path)
    # Select the first category as default
    item_categories[current_category_index].select()
    plt.show()
    print('Window closed')

    for i in range(current_image_index + 1):
        if len(input_images[i].boundingBoxes) == 0:
            continue
        path = input_images[i].path
        width, height = Image.open(path).size
        print(path, ' Width: ', width, ' Height: ', height)
        writer = Writer(path, width, height)
        for bbox in input_images[i].boundingBoxes:
            writer.addObject(item_categories[bbox.category_index].name,
                bbox.corner1.x, bbox.corner1.y, bbox.corner2.x, bbox.corner2.y)
            print('\t', bbox)
        print(os.path.join(flags.FLAGS.output_annotations_dir,
            get_base_filename(path) + '.xml'))   

if __name__ == "__main__":
  app.run(main)
