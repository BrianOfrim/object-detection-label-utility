import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import numpy as np
import matplotlib.image as mpimg
from dataclasses import dataclass
import math
from absl import app
from absl import flags


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

@dataclass
class Category:
    data_name: str
    color: tuple
    keyboard_string: str

item_categories = [
    Category('trash', (23, 171, 245), '0'),
    Category('aluminum', (237, 181, 14), '1'),
    Category('compost', (219, 56, 210), '2'),
    Category('glass', (255, 74, 164), '3'),
    Category('paper', (230, 245, 24), '4'),
    Category('plastic', (24, 230, 245), '5'),
]

flags.DEFINE_string(
    'input_image_dir',
    '../data/images',
    'Location of the image files to lable.'
)

current_category_index = 0
input_images = []
current_image_index = 0


def draw_bounding_boxes(bboxes):
    # clear all current boxes
    [p.remove() for p in reversed(plt.gca().patches)]
    
    # redraw the boxes
    for bbox in bboxes:
        height = bbox.corner2.y - bbox.corner1.y 
        width = bbox.corner2.x - bbox.corner1.x
        lower_left = (bbox.corner1.x, bbox.corner1.y)
        item_categories[bbox.category_index].color
        color = tuple(x / 255.0 for x in item_categories[bbox.category_index].color)
        rect = patches.Rectangle(lower_left, width, height,
                linewidth=2,edgecolor=color,facecolor='none')
        plt.gca().add_patch(rect)
    
    plt.gcf().canvas.draw()

def removeInclompleteBoxes(bboxes):
    # Clear last bbox if it is incomplete
    for bbox in bboxes:
        if bbox.corner2 is None:
            bboxes.remove(bbox)

def displayImage(path):
    # Load and show the new image
    img = mpimg.imread(path)
    plt.imshow(img)
    # Update the title
    plt.title(path.split('/')[-1])
    # Redraw the figure
    plt.gcf().canvas.draw()

def clearAllLines():
     [l.remove() for l in reversed(plt.gca().lines)]


def drawCorner1Lines(bboxCorner):
    ax = plt.gca()
    
    xmin, xmax = ax.get_xbound()
    ymin, ymax = ax.get_ybound()
    
    vLine = mlines.Line2D([bboxCorner.x ,bboxCorner.x],
            [ymin,ymax], linestyle='dashed')
    hLine = mlines.Line2D([xmin,xmax], 
            [bboxCorner.y,bboxCorner.y], linestyle='dashed')

    ax.add_line(vLine)
    ax.add_line(hLine)

    plt.gcf().canvas.draw()


def keypress(event):
    global current_image_index
    global current_category_index
    print('press', event.key)
    if event.key == 'd':
        print('Next')
        removeInclompleteBoxes(input_images[current_image_index].boundingBoxes)
        clearAllLines()
        current_image_index += 1
        displayImage(input_images[current_image_index].path)
        draw_bounding_boxes(input_images[current_image_index].boundingBoxes) 
    elif event.key == 'a':
        clearAllLines()
        if (current_image_index == 0):
            print('Already at the start')
        else:
            print('Previous')
            removeInclompleteBoxes(input_images[current_image_index].boundingBoxes)
            current_image_index -= 1
            displayImage(input_images[current_image_index].path)
            draw_bounding_boxes(input_images[current_image_index].boundingBoxes) 
    elif event.key == 'w' or event.key == 'escape':
        clearAllLines()
        if(len(input_images[current_image_index].boundingBoxes) == 0):
            print('No more bounding boxes to clear')
        elif input_images[current_image_index].boundingBoxes[-1].corner2 is None:
            print('Remove corner 1 guidelines')
        else:
            print('Remove latest bounding box')
            input_images[current_image_index].boundingBoxes.pop()
        removeInclompleteBoxes(input_images[current_image_index].boundingBoxes)
        draw_bounding_boxes(input_images[current_image_index].boundingBoxes) 

    for category_index, catagory in enumerate(item_categories):
        if event.key == catagory.keyboard_string:
            current_category_index = category_index
            print('Current catagory: %s' % 
                    item_categories[current_category_index].data_name)
    # Redraw the figure
    plt.gcf().canvas.draw()

def onclick(event):
    # verify that the click was inbounds
    if event.xdata is None or event.ydata is None:
        print('Invalid box corner')
        return

    # get the current bounding box list
    bboxes = input_images[current_image_index].boundingBoxes
    if(len(bboxes) > 0 and bboxes[-1].corner2 is None):
        clearAllLines()
        bboxes[-1].corner2 = BBoxCorner(math.floor(event.xdata),
            math.floor(event.ydata))
        draw_bounding_boxes(bboxes)    
    else:
        bboxes.append(BBox(BBoxCorner(
            math.floor(event.xdata), math.floor(event.ydata)), None, current_category_index))
        drawCorner1Lines(bboxes[-1].corner1)

def main(unused_argv):
    fig, ax = plt.subplots()
    fig.canvas.set_window_title('Label')

    # read in the names of the images to label
    for image_file in os.listdir(flags.FLAGS.input_image_dir):
        if image_file.endswith(".jpg"):
            input_images.append(AnnotatedImage(
                os.path.join( flags.FLAGS.input_image_dir, image_file), []))

    cid_onclick= fig.canvas.mpl_connect('button_press_event', onclick)
    cid_keypress = fig.canvas.mpl_connect('key_press_event', keypress)
    
    if(len(input_images) > 0):
        displayImage(input_images[current_image_index].path)
        plt.show()
        print('Window closed')
    else:
        print('No input images to label were found') 

if __name__ == "__main__":
  app.run(main)
