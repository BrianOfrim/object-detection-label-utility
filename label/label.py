import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import matplotlib.image as mpimg
from matplotlib.widgets import Button
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

class Category:
    def __init__(self, data_name, color, keyboard_string):
        self.data_name = data_name
        self.color = color
        self.keyboard_string = keyboard_string
        self.ax = None
        self.button = None
    def selected(self, event):
        print('Button pushed: %s' % self.data_name)
        if(self.ax is not None):
            self.ax.spines['left'].set_linewidth(2)
            self.ax.spines['right'].set_linewidth(2)
            self.ax.spines['top'].set_linewidth(2)
            self.ax.spines['bottom'].set_linewidth(2)
            
            self.ax.spines['bottom'].set_color('green')
            self.ax.spines['top'].set_color('green')
            self.ax.spines['left'].set_color('green')
            self.ax.spines['right'].set_color('green')

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

fig = plt.figure()
fig.canvas.set_window_title('Label')

im_ax = plt.axes([0.075, 0.15, 0.85, 0.75])

def normalize_color(rgbColor):
    return tuple(x / 255.0 for x in rgbColor)


def next_image(event):
    global current_image_index
    print('Next')
    removeInclompleteBoxes(input_images[current_image_index].boundingBoxes)
    clearAllLines()
    current_image_index += 1
    displayImage(input_images[current_image_index].path)
    draw_bounding_boxes(input_images[current_image_index].boundingBoxes) 

def prev_image(event):
    global current_image_index
    clearAllLines()
    if (current_image_index == 0):
        print('Already at the start')
    else:
        print('Previous')
        removeInclompleteBoxes(input_images[current_image_index].boundingBoxes)
        current_image_index -= 1
        displayImage(input_images[current_image_index].path)
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
        color = normalize_color(item_categories[bbox.category_index].color)
        rect = patches.Rectangle(lower_left, width, height,
                linewidth=2,edgecolor=color,facecolor='none')
        im_ax.add_patch(rect)
    
    fig.canvas.draw()

def removeInclompleteBoxes(bboxes):
    # Clear last bbox if it is incomplete
    for bbox in bboxes:
        if bbox.corner2 is None:
            bboxes.remove(bbox)

def displayImage(path):
    # Load and show the new image
    img = mpimg.imread(path)
    im_ax.imshow(img)
    # Update the title
    im_ax.set_title(path.split('/')[-1])
    # Redraw the figure
    fig.canvas.draw()

def clearAllLines():
     [l.remove() for l in reversed(im_ax.lines)]

def drawCorner1Lines(bboxCorner):
    xmin, xmax = im_ax.get_xbound()
    ymin, ymax = im_ax.get_ybound()
    
    vLine = mlines.Line2D([bboxCorner.x ,bboxCorner.x],
            [ymin,ymax], linestyle='dashed')
    hLine = mlines.Line2D([xmin,xmax], 
            [bboxCorner.y,bboxCorner.y], linestyle='dashed')

    im_ax.add_line(vLine)
    im_ax.add_line(hLine)

    fig.canvas.draw()


def keypress(event):
    global current_category_index
    print('press', event.key)
    if event.key == 'd':
        next_image(event)
    elif event.key == 'a':
        prev_image(event)
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

    for category_index, category in enumerate(item_categories):
        if event.key == category.keyboard_string:
            current_category_index = category_index
            print('Current category: %s' % 
                    item_categories[current_category_index].data_name)
    # Redraw the figure
    fig.canvas.draw()

def onclick(event):
    
    # verify that the click was inbounds
    if event.inaxes is None or event.inaxes != im_ax or\
            event.xdata is None or event.ydata is None:
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
    # read in the names of the images to label
    for image_file in os.listdir(flags.FLAGS.input_image_dir):
        if image_file.endswith(".jpg"):
            input_images.append(AnnotatedImage(
                os.path.join(flags.FLAGS.input_image_dir, image_file), []))

    axprev = plt.axes([0.8, 0.01, 0.085, 0.075])
    axnext = plt.axes([0.9, 0.01, 0.085, 0.075])
    bprev = Button(axprev, 'Prev')
    bnext = Button(axnext, 'Next')

    cid_onclick= fig.canvas.mpl_connect('button_press_event', onclick)
    cid_keypress = fig.canvas.mpl_connect('key_press_event', keypress)
    
    bprev.on_clicked(prev_image)
    bnext.on_clicked(next_image)
 
    for item_index, category_item in enumerate(item_categories):
        category_item.ax = plt.axes([(item_index * 0.11) + 0.05, 0.01, 0.1, 0.075])
        category_item.button = Button(category_item.ax, category_item.data_name,
                color=normalize_color(category_item.color))
        category_item.button.on_clicked(category_item.selected)
        
        category_item.ax.spines['left'].set_linewidth(None)
        category_item.ax.spines['right'].set_linewidth(None)
        category_item.ax.spines['top'].set_linewidth(None)
        category_item.ax.spines['bottom'].set_linewidth(None)

    
    if(len(input_images) > 0):
        displayImage(input_images[current_image_index].path)
        plt.show()
        print('Window closed')
    else:
        print('No input images to label were found') 

if __name__ == "__main__":
  app.run(main)
