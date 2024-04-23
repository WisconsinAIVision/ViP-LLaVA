from PIL import Image, ImageDraw
import random, math
import numpy as np
import random
from shapely.ops import unary_union
from shapely.geometry import Point, Polygon
from scipy.stats import multivariate_normal


color_pool = {
    'red': (255, 0, 0),
    'lime': (0, 255, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'fuchsia': (255, 0, 255),
    'aqua': (0, 255, 255),
    'orange': (255, 165, 0),
    'purple': (128, 0, 128),
    'gold': (255, 215, 0),
}


words_shape ={
    "rectangle": ["within", "rectangle"], 
    "ellipse": ["within", "ellipse"],
    "triangle": ["with", "triangle"],
    "point": ["at", "point"], 
    "scribble" : ["with", "scribble"], 
    "mask contour": ["with", "mask contour"],
    "mask": ["with", "mask"],
    "arrow": ["pointed to by", "arrow"],
 }




def draw_arrow(draw, bbox_coord, outline_color, line_width, max_arrow_length=100, max_image_size=336, image_size_anchor = 336):
    left, top, right, bottom = bbox_coord
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    
    # Arrow length related to the bounding box size
    bounding_box_side_length = min(right - left, bottom - top)
    arrow_length = random.uniform(0.8 * bounding_box_side_length, max_arrow_length)
    
    # Randomize the arrow angle
    angle = random.uniform(0, 2 * math.pi)
    center_x += random.uniform(-0.25, 0.25) * (right - left)
    center_y += random.uniform(-0.25, 0.25) * (bottom - top)
    
    # Arrowhead size related to arrow length
    arrow_head_size = max(random.uniform(0.2, 0.5) * arrow_length, int(6 * max_image_size / image_size_anchor))
    
    # Recalculate the arrow end to ensure it connects properly with the arrowhead
    arrow_end_x = center_x + (arrow_length - arrow_head_size) * math.cos(angle)
    arrow_end_y = center_y + (arrow_length - arrow_head_size) * math.sin(angle)
    
    if random.random() < 0.5:
        # Draw with a "wobble" to mimic human drawing
        mid_x = (center_x + arrow_end_x) / 2 + random.uniform(-5, 5) * int(max_image_size / image_size_anchor)
        mid_y = (center_y + arrow_end_y) / 2 + random.uniform(-5, 5) * int(max_image_size / image_size_anchor)
        draw.line([(center_x, center_y), (mid_x, mid_y), (arrow_end_x, arrow_end_y)], fill=outline_color, width=line_width)
    else:
        # Draw the arrow line
        draw.line([(center_x, center_y), (arrow_end_x, arrow_end_y)], fill=outline_color, width=line_width)
    arrow_end_x = center_x
    arrow_end_y = center_y
    # Draw the arrow head
    if random.random() < 0.5:
        draw.polygon([
            (arrow_end_x + arrow_head_size * math.cos(angle + math.pi / 3),
            arrow_end_y + arrow_head_size * math.sin(angle + math.pi / 3)),
            (arrow_end_x, arrow_end_y),
            (arrow_end_x + arrow_head_size * math.cos(angle - math.pi / 3),
            arrow_end_y + arrow_head_size * math.sin(angle - math.pi / 3))
        ], fill=outline_color)
    else:
        draw.line([
            (arrow_end_x + arrow_head_size * math.cos(angle + math.pi / 3),
            arrow_end_y + arrow_head_size * math.sin(angle + math.pi / 3)),
            (arrow_end_x, arrow_end_y),
            (arrow_end_x + arrow_head_size * math.cos(angle - math.pi / 3),
            arrow_end_y + arrow_head_size * math.sin(angle - math.pi / 3))
        ], fill=outline_color, width=line_width)
    



def draw_rectangle(draw, bbox_coord, outline_color, width):
    left, top, right, bottom = bbox_coord
    draw.rectangle([(left, top), (right, bottom)], outline=outline_color, width=width)


def draw_ellipse(draw, bbox_coord, mask_polygon, outline_color, width, size_ratio=1, aspect_ratio = 1.0):
    if mask_polygon!= None:
        minx, miny, maxx, maxy = mask_polygon.bounds
    else:
        minx, miny, maxx, maxy = bbox_coord
    
    # Calculate the center of the bounding box
    center_x = (maxx + minx) / 2
    center_y = (maxy + miny) / 2
    
    # Calculate the dimensions of the new bounding box
    new_width = (maxx - minx) * size_ratio * aspect_ratio
    new_height = (maxy - miny) * size_ratio / aspect_ratio
    
    # Calculate the new minx, miny, maxx, maxy based on the new dimensions
    minx = center_x - new_width / 2
    miny = center_y - new_height / 2
    maxx = center_x + new_width / 2
    maxy = center_y + new_height / 2
    
    # Draw the ellipse
    bbox = [minx, miny, maxx, maxy]
    draw.ellipse(bbox, outline=outline_color, width=width)
    
    


def is_max_angle_less_than_150(points):
    for i in range(3):
        p1 = np.array(points[i])
        p2 = np.array(points[(i + 1) % 3])
        p3 = np.array(points[(i + 2) % 3])
        
        a = np.linalg.norm(p3 - p2)
        b = np.linalg.norm(p1 - p3)
        c = np.linalg.norm(p1 - p2)
        
        # Calculate angle at p2 using cosine rule
        angle_at_p2 = np.degrees(np.arccos((a**2 + c**2 - b**2) / (2 * a * c)))
        
        if angle_at_p2 > 150:
            return False
    return True



def get_random_point_within_bbox(bbox):
    left, top, right, bottom = bbox
    x = np.random.uniform(left, right)
    y = np.random.uniform(top, bottom)
    return x, y


def get_random_point_within_polygon(polygon):
    minx, miny, maxx, maxy = polygon.bounds
    trial_num = 0
    while True:
        if  trial_num<50:
            x = np.random.uniform(minx, maxx)
            y = np.random.uniform(miny, maxy)
            point = Point(x, y)
            if polygon.contains(point):
                return x, y
            trial_num += 1
        else:
            x = np.random.uniform(minx, maxx)
            y = np.random.uniform(miny, maxy)
            return x, y
        
        
      

def draw_rounded_triangle(draw, bbox_coord, mask_polygon, outline_color, width):
    while True:
        points = []
        for _ in range(3):
            if mask_polygon!= None:
                point = get_random_point_within_polygon(mask_polygon)
            else:
                point = get_random_point_within_bbox(bbox_coord)
            points.append(point)
        if is_max_angle_less_than_150(points):
            break
    draw.line([points[0], points[1], points[2], points[0]], fill=outline_color, width=width, joint='curve')



def draw_point(draw, bbox_coord, mask_polygon, outline_color=(255,0,0), radius=3, aspect_ratio=1.0):
    # Calculate the center and covariance matrix for multivariate normal distribution
    if mask_polygon!= None:
        minx, miny, maxx, maxy = mask_polygon.bounds
    else:
        minx, miny, maxx, maxy = bbox_coord
    mean = [(maxx + minx) / 2, (maxy + miny) / 2]
    cov = [[(maxx - minx) / 8, 0], [0, (maxy - miny) / 8]]

    # Initialize counter for fail-safe mechanism
    counter = 0

    # Generate a random central point within the mask using a normal distribution
    max_tries = 10
    while True:
        cx, cy = multivariate_normal.rvs(mean=mean, cov=cov)
        center_point = Point(cx, cy)
        if mask_polygon.contains(center_point):
            break
        counter += 1
        if counter >= max_tries:
            cx, cy = multivariate_normal.rvs(mean=mean, cov=cov)
            center_point = Point(cx, cy)
            # print("Failed to find a point within the polygon after {} tries".format(max_tries))
            break
    
    x_radius = radius * aspect_ratio
    y_radius = radius / aspect_ratio
    bbox = [cx - x_radius, cy - y_radius, cx + x_radius, cy + y_radius]
    
    # Draw the ellipse and fill it with color
    draw.ellipse(bbox, outline=outline_color, fill= outline_color )
    
    


def draw_scribble(draw, bbox_coord, mask_polygon, outline_color=(255, 0, 0), width=3,  max_image_size=336, image_size_anchor = 336):
            
    prev_point = None  # Initialize prev_point outside the loop
    if mask_polygon!= None:
        p0 = get_random_point_within_polygon(mask_polygon)
        p1 = get_random_point_within_polygon(mask_polygon)
        p2 = get_random_point_within_polygon(mask_polygon)
        p3 = get_random_point_within_polygon(mask_polygon)
    else:
        p0 = get_random_point_within_bbox(bbox_coord)
        p1 = get_random_point_within_bbox(bbox_coord)
        p2 = get_random_point_within_bbox(bbox_coord)
        p3 = get_random_point_within_bbox(bbox_coord)
    
    for t in np.linspace(0, 1, int(1000* max_image_size/image_size_anchor)):
        x = (1 - t)**3 * p0[0] + 3 * (1 - t)**2 * t * p1[0] + 3 * (1 - t) * t**2 * p2[0] + t**3 * p3[0]
        y = (1 - t)**3 * p0[1] + 3 * (1 - t)**2 * t * p1[1] + 3 * (1 - t) * t**2 * p2[1] + t**3 * p3[1]
        
        current_point = (x, y)
        if prev_point:
            draw.line([prev_point, current_point], fill=outline_color, width=width)
            
        prev_point = current_point  # Update prev_point to the current ending point
    
    



            
            

def draw_mask_contour(draw, bbox_coord, segmentation_coords, color="red", width=1, ):
    if segmentation_coords == None:
          segmentation_coords = [[bbox_coord[0], bbox_coord[1], bbox_coord[0], bbox_coord[3], 
                                bbox_coord[2], bbox_coord[3], bbox_coord[2], bbox_coord[1]]]
    for segment in segmentation_coords:
        coords = [(segment[i], segment[i + 1]) for i in range(0, len(segment), 2)]
        for dx in range(-width, width + 1):
            for dy in range(-width, width + 1):
                shifted_coords = [(x + dx, y + dy) for x, y in coords]
                draw.polygon(shifted_coords, outline=color)
            

def draw_mask(draw, bbox_coord,  segmentation_coords, color="red", width=1,   ):
    if segmentation_coords == None:
          segmentation_coords = [[bbox_coord[0], bbox_coord[1], bbox_coord[0], bbox_coord[3], 
                                bbox_coord[2], bbox_coord[3], bbox_coord[2], bbox_coord[1]]]
    for segment in segmentation_coords:
        coords = [(segment[i], segment[i + 1]) for i in range(0, len(segment), 2)]
        draw.polygon(coords, outline= None, fill=color, width=width)
        


def image_blending(image, shape = 'rectangle', bbox_coord = None, segmentation = None, image_size_anchor = 336, rgb_value = None, visual_prompt_style = '', alpha = None, width = None):
      image = image.convert("RGB")
      img_width, img_height = image.size
      max_image_size = max(img_width, img_height)
      visual_prompt_img = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
      visual_prompt_img_canvas = ImageDraw.Draw(visual_prompt_img)
    #   color, rgb_value = random.choice(list(color_pool.items()))
      if alpha == None:
            alpha =  random.randint(96, 255) if shape != 'mask' else  random.randint(48, 128) 
      color_alpha = rgb_value + (alpha,)
      if segmentation != None:
            try:
                polygons = []
                for segmentation_coord in segmentation:
                    mask_polygon = Polygon([(segmentation_coord[i], segmentation_coord[i+1]) for i in range(0, len(segmentation_coord), 2)])
                    polygons.append(mask_polygon)
                mask_polygon = random.choice(polygons)
                try: 
                    all_polygons_union = unary_union(polygons)
                except:
                    all_polygons_union = None
                    # print('Error in all_polygons_union')
            except:
                mask_polygon = None
                # print('Error in Polygon Generation')
                
                
                
      else:
            all_polygons_union = mask_polygon = None
        
        # draw shapes
      if shape == 'rectangle':
            line_width =  max( int(3 *max_image_size/image_size_anchor), 1) if visual_prompt_style == 'constant' else max(random.randint( int(2 *max_image_size/image_size_anchor), int(8 * max_image_size/image_size_anchor)), 1)
            line_width =  max( int(width *max_image_size/image_size_anchor), 1) if width != None else line_width
            draw_rectangle(visual_prompt_img_canvas, bbox_coord, color_alpha, line_width)
      elif shape == 'ellipse':
            line_width =  max(random.randint( int(2 *max_image_size/image_size_anchor), int(8 * max_image_size/image_size_anchor)), 1)
            line_width =  max( int(width *max_image_size/image_size_anchor), 1) if width != None else line_width
            size_ratio = random.uniform(1, 1.5) # 
            draw_ellipse(visual_prompt_img_canvas, bbox_coord, all_polygons_union, color_alpha, line_width, size_ratio = size_ratio)  
      elif shape == 'arrow':
            line_width = max(random.randint(int(1 * max_image_size / image_size_anchor), int(6 * max_image_size / image_size_anchor)), 1)
            line_width =  max( int(width *max_image_size/image_size_anchor), 1) if width != None else line_width
            max_arrow_length= max( int(50 * max_image_size/image_size_anchor), 1)
            draw_arrow(visual_prompt_img_canvas, bbox_coord, color_alpha, line_width , max_image_size=max_image_size, max_arrow_length = max_arrow_length, image_size_anchor = image_size_anchor)
      elif shape == 'triangle':
            line_width =  max(random.randint(int(2 *  max_image_size/image_size_anchor), int(8 * max_image_size/image_size_anchor)), 1)
            line_width =  max( int(width *max_image_size/image_size_anchor), 1) if width != None else line_width
            draw_rounded_triangle(visual_prompt_img_canvas, bbox_coord, all_polygons_union, color_alpha, line_width)
      elif shape == 'point':
            radius =   max( int(8 * max_image_size/image_size_anchor), 1) if visual_prompt_style == 'constant' else  max(random.randint(int(5 * max_image_size/image_size_anchor),  int(20 *max_image_size/image_size_anchor)), 1)
            aspect_ratio =1 if random.random()<0.5 or  visual_prompt_style == 'constant' else random.uniform(0.5, 2.0)
            draw_point(visual_prompt_img_canvas, bbox_coord, mask_polygon, color_alpha, radius, aspect_ratio)
      elif shape == 'scribble':
            line_width =  max(random.randint(int(2 * max_image_size/image_size_anchor), int(12 * max_image_size/image_size_anchor)), 1)
            line_width =  max( int(width *max_image_size/image_size_anchor), 1) if width != None else line_width
            draw_scribble(visual_prompt_img_canvas, bbox_coord, mask_polygon, color_alpha, line_width, max_image_size=max_image_size, image_size_anchor = image_size_anchor)
      elif shape == 'mask':
            line_width = random.randint( int(0 *max_image_size/image_size_anchor), int(2 * max_image_size/image_size_anchor))
            line_width =  max( int(width *max_image_size/image_size_anchor), 1) if width != None else line_width
            draw_mask(visual_prompt_img_canvas, bbox_coord, segmentation, color_alpha, line_width)       
      elif shape == 'mask contour':
            line_width =  max(random.randint( int(1 *max_image_size/image_size_anchor), int(2 * max_image_size/image_size_anchor)), 1)
            line_width =  max( int(width *max_image_size/image_size_anchor), 1) if width != None else line_width
            draw_mask_contour(visual_prompt_img_canvas, bbox_coord, segmentation, color_alpha, line_width)
 
      image = image.convert("RGBA")
      image = Image.alpha_composite(image, visual_prompt_img)
      image = image.convert("RGB")
      return image