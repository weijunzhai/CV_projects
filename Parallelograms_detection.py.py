# Para_detection.py
# mainly responsible for detecting parallelograms on images

from PIL import Image
import math


# given two point in the image coordinate, return the distance of the two points
def point_distance(point1, point2):
    return  int(math.sqrt((point1[0]-point2[0])*(point1[0]-point2[0]) + (point1[1]-point2[1])*(point1[1]-point2[1])))


# given an point of the image, detect whether there is pixel nearby.
# scale here is the 8 path distance of the detected point
def is_pixel_nearby(image, point, scale):
    j = point[0]
    i = point[1]
    for y in range(i-scale, i+scale+1):
        for x in range(j-scale, j+scale+1):
            color = image.getpixel((x, y))
            if color[0] == 0:
                return True
    return False


# given points in the parameter_space,
# draw the cross points of the lines that represent as points in the parameter space
def draw_cross_point(image, parameter_space, quantize_cita, quantize_p):
    points = cross_point(image, parameter_space, quantize_cita, quantize_p)
    print(points)
    size = image.size
    size_x = size[0]
    size_y = size[1]
    for item in points:
        for i in range(size_y):
            for j in range(size_x):
                if abs(j - item[0]) <= 2 and abs(i - item[1]) <= 2:
                    image.putpixel((j, i), (255, 0, 0))

    return image


# check whether a point is in a given line represent as a point in the parameter space
# threshold here is the the distance we can tolerate
def is_point_in_line(point, image, p_cita, quantize_cita, quantize_p, threshold):
    size = image.size
    size_x = size[0]
    size_y = size[1]
    p_coordinate = int((size_x + size_y) / quantize_p)
    cita = float(p_cita[0] * quantize_cita * math.pi)/180.0
    p = p_cita[1] - p_coordinate
    j = point[0]
    i = point[1]
    if abs(p*quantize_p - int(j*math.cos(cita) + i*math.sin(cita))) <= threshold:
        return True
    return False


# draw_lines based on parameter space given
def draw_line(image, parameter_space, quantize_cita, quantize_p, scale):
    final_image = Image.new(image.mode, image.size, color=(255, 255, 255))
    size = image.size
    size_x = size[0]
    size_y = size[1]
    p_coordinate = int((size_x + size_y) / quantize_p)
    # scale is used for pixel detection in the function is_pixel_near_by
    for item in parameter_space:
        for i in range(scale, size_y-scale):
            for j in range(scale, size_x-scale):
                angle = float(item[0] * quantize_cita * math.pi)/180.0
                # check whether the pixel (j, i) is in the line given or not
                if item[1] == int((j*math.cos(angle) + i*math.sin(angle))/quantize_p + p_coordinate):
                    if is_pixel_nearby(image, (j, i), scale):
                        final_image.putpixel((j, i), (0, 0, 0))
    return final_image


# edges_detection returns the edge-detection image that generate from the input image
# we use sobel operater to deal with the image given
def edges_detection(image):
    # generate two new images, the gray_image is the gray-image of the input image
    gray_image = Image.new(image.mode, image.size, color=0)
    # the final_image is the image that shows the edge of the input image
    final_image = Image.new(image.mode, image.size, color=0)
    size = image.size
    size_x = size[0]
    size_y = size[1]
    angle = [[-1 for col in range(size_x)] for row in range(size_y)]
    # generate the gray_image by putting the gray value of the input image into the gray_image
    for i in range(size_y):
        for j in range(size_x):
            color = image.getpixel((j, i))
            gray = int((30*color[0] + 59*color[1] + 11*color[2])/100)
            gray_image.putpixel((j, i), (gray, gray, gray))

    # edge detection by sobel operater
    for i in range(1, size_y-2):
        for j in range(1, size_x-2):
            # generate the soble operater
            color00 = gray_image.getpixel((j-1, i-1))
            color01 = gray_image.getpixel((j, i-1))
            color02 = gray_image.getpixel((j+1, i-1))
            color10 = gray_image.getpixel((j-1, i))
            color12 = gray_image.getpixel((j+1, i))
            color20 = gray_image.getpixel((j-1, i+1))
            color21 = gray_image.getpixel((j, i+1))
            color22 = gray_image.getpixel((j+1, i+1))
            # edge detection by correlation and normalize the value between [0, 255]
            gx = (color02[0] + 2*color12[0] + color22[0] - color00[0] - 2*color10[0] - color20[0])/4
            gy = (color00[0] + 2*color01[0] + color02[0] - color20[0] - 2*color21[0] - color22[0])/4
            # calculate the edge magnitude of the image and normalize
            value = int(math.sqrt(gx*gx + gy*gy)/math.sqrt(2))
            # create the edge image pixel by pixel
            final_image.putpixel((j, i), (value, value, value))
    # final_image = non_maximum_suppression(final_image, angle)
    return final_image


# responsible for transforming the provided image to a binary edge_map image by the provide threshold
def edge_map(image, threshold):
    final_image = Image.new(image.mode, image.size, color=(255, 255, 255))
    size = image.size
    size_x = size[0]
    size_y = size[1]
    # choosing a threshold to produce a binary edge map
    for i in range(1, size_y-1):
        for j in range(1, size_x-1):
            color = image.getpixel((j, i))
            if color[0] > threshold:
                final_image.putpixel((j, i), (0, 0, 0))

    return final_image


# using hough transform to transfter the pixel in the edge map image into the parameter space
# line_threshold will decide which point in the parameter space will be returned
def hough(image, quantize_cita, quantize_p, line_threshold):
    size = image.size
    size_x = size[0]
    size_y = size[1]
    p_coordinate = int((size_x + size_y)/quantize_p)
    cita_coordinate = int(360/quantize_cita)
    # initialize the parameter space
    parameter_space = [[0 for col in range(cita_coordinate)] for row in range(2 * p_coordinate)]
    for i in range(size_y):
        for j in range(size_x):
            color = image.getpixel((j, i))
            if color[0] == 0:
                # hough tramsform based on the pixel in the image space
                for cita in range(cita_coordinate):
                    angle = float(cita * quantize_cita * math.pi)/180.0
                    p_value = int((j * math.cos(angle) + i * math.sin(angle))/quantize_p + p_coordinate)
                    # print(i, j, cita, angle, math.sin(angle), p_value)
                    parameter_space[p_value][cita] += 1

    line_in_parameter_space = []
    # filter the parameter space base on the threshold given
    for i in range(p_coordinate):
        for j in range(cita_coordinate):
            if parameter_space[i][j] > line_threshold:
                line_in_parameter_space.append((j, i, parameter_space[i][j]))
    line_in_parameter_space.sort(key=lambda x: x[0])

    return line_in_parameter_space


# optimize the points in parameter_space, return more accurate poins in parameter space
# by mergining the similar lines based on the given optimize_cita and optimize_p parameter
def optimize_parameter_space(parameter_space, optimize_cita, optimize_p):
    final_list = []
    for i in range(len(parameter_space)):
        item1 = parameter_space[i]
        max_count = item1[2]
        for j in range(len(parameter_space)):
            item2 = parameter_space[j]
            # if two point in parameter_space have similar cita and similar p, then they are similar,
            # remove the point with the less count on the cumulator
            if abs(item1[0] - item2[0]) <= optimize_cita and abs(item1[1] - item2[1]) <= optimize_p:
                if item1[2] == item2[2]:
                    parameter_space[j] = (item2[0], item2[1], item2[2]+1)
                if item1[2] < item2[2]:
                    max_count = item2[2]
                    break
        if max_count == item1[2]:
            final_list.append(item1)

    return final_list


# from the given list of lines, choose pair of paralleled lines by selecting lines with similar cita
# but with different p values
def pair_paralleled_lines(parameter_space, pair_cita, pair_p, pair_count):
    pairs = []
    for i in range(len(parameter_space)-1):
        for j in range(i+1, len(parameter_space)):
            line1 = parameter_space[i]
            line2 = parameter_space[j]
            cita1 = line1[0]
            cita2 = line2[0]
            p1 = line1[1]
            p2 = line2[1]
            count1 = line1[2]
            count2 = line2[2]
            # if the cita of the two lines are similar, then they could by paralleled, and also,
            # to avoid the situation that they could lay on the same edge,
            # the two lime should be a certain distance away
            if (abs(cita1-cita2) <= pair_cita or abs(cita1+180-cita2) <= pair_cita) and abs(p1-p2) >= pair_p and abs(count1-count2) <= pair_count:
                pairs.append((line1, line2))
    return pairs


# function given the 4 lines in parameter_space, and will return cross points of those paralleled lines
def cross_point(image, parameter_space, quantize_cita, quantize_p):
    size = image.size
    size_x = size[0]
    size_y = size[1]
    p_coordinate = int((size_x + size_y) / quantize_p)
    point = []
    for x in range(0, len(parameter_space)-1):
        for y in range(x+1, len(parameter_space)):
            item1 = parameter_space[x]
            item2 = parameter_space[y]
            if abs(item1[0] - item2[0]) <= 15:
                continue
            p1 = item1[1] - p_coordinate
            p2 = item2[1] - p_coordinate
            cita1 = item1[0] * quantize_cita * math.pi / 180.0
            cita2 = item2[0] * quantize_cita * math.pi / 180.0
            # calculating the cross point of the given pair of paralleled lines
            j = int((p1 / math.sin(cita1) - p2 / math.sin(cita2)) / (
            math.cos(cita1) / math.sin(cita1) - math.cos(cita2) / math.sin(cita2)))
            i = int(p1 / math.sin(cita1) - j * (math.cos(cita1) / math.sin(cita1)))
            # if the cross point of two lines is out of the image, then coutinue to checking other points
            if j >= size_x or i >= size_y or i <= 0 or j <= 0:
                continue
            if (j, i) not in point:
                point.append((j, i))
    # two pair of paralleled lines should return 4 cross points, if the cross points are less than 4,
    # return an empty list
    if not len(point) == 4:
        return []
    return point


# return cross point in other formate
def cross_lines(image, parameter_space, quantize_cita, quantize_p):
    size = image.size
    size_x = size[0]
    size_y = size[1]
    p_coordinate = int((size_x + size_y) / quantize_p)
    point = []
    temp = []
    for x in range(0, 2):
        for y in range(2, 4):
            item1 = parameter_space[x]
            item2 = parameter_space[y]
            if abs(item1[0] - item2[0]) <= 15/quantize_cita:
                continue
            p1 = item1[1] - p_coordinate
            p2 = item2[1] - p_coordinate
            cita1 = item1[0] * quantize_cita * math.pi / 180.0
            cita2 = item2[0] * quantize_cita * math.pi / 180.0
            # j = int((p1 / math.sin(cita1) - p2 / math.sin(cita2)) / (
            # math.cos(cita1) / math.sin(cita1) - math.cos(cita2) / math.sin(cita2)))
            j = int((p1*math.sin(cita2) - p2*math.sin(cita1))/(math.cos(cita1)*math.sin(cita2) - math.cos(cita2)*math.sin(cita1)))
            i = 0
            if abs(cita1 - math.pi)*180.0/math.pi <= 5:
                i = int(p2 / math.sin(cita2) - j * (math.cos(cita2) / math.sin(cita2)))
            else:
                i = int(p1 / math.sin(cita1) - j * (math.cos(cita1) / math.sin(cita1)))
            if j >= size_x or i >= size_y or i <= 0 or j <= 0:
                continue
            if (j, i) not in temp:
                temp.append((j, i))
                point.append(((j, i), item1, item2))
    if not len(point) == 4:
        return []
    return point


# detect 4 lines based the cross point, and return 4 lines in a list.
# lines will be stored in a data structure as (star_pixel, end_pixel, P_cita)
def paralleled_lines(image, parameter_space, quantize_cita, quantize_p):
    size = image.size
    lines = []
    points = cross_lines(image, parameter_space, quantize_cita, quantize_p)

    if len(points) == 0:
        return lines
    for i in range(len(points)-1):
        for j in range(i+1, len(points)):
            # based on the given cross point, store the lines in certain data structure
            for item in parameter_space:
                if item in points[i] and item in points[j]:
                    point1 = points[i]
                    point2 = points[j]
                    lines.append((point1[0], point2[0], item))
    return lines


# given 4 lines, determine whether the four lines given is a paragrams or not.
# by cheching the number of pixels that are background and the number of pixels are foreground on the line
# given the threshold, if the ratio of the forground pixel is larger than the threshold, then the lines can be composed
# as a parallelogram
def is_parallelograms(lines, image, quantize_cita, quantize_p, detected_para_threshold):
    size = image.size
    size_x = size[0]
    size_y = size[1]
    fore_count = 0
    back_count = 0
    # scan every given line
    for line in lines:
        distance = point_distance(line[0], line[1])
        # scan every pixel in the image
        for i in range(2, size_y-2):
            for j in range(2, size_x-2):
                # checking whether a pixel is on the line or not
                if point_distance((j, i), line[0]) + point_distance((j, i), line[1]) == distance \
                        and is_point_in_line((j, i), image, line[2], quantize_cita, quantize_p, 5):
                    # checking whether a pixel is foreground or not
                    if is_pixel_nearby(image, (j,i), 0):
                        fore_count += 1
                    else:
                        back_count += 1

    if fore_count == 0 or fore_count+back_count <= 50:
        return False
    if back_count == 0:
        if fore_count > 0:
            return True
        else:
            return False
        print(lines)
        print("forecount: ", fore_count, " backcount:", back_count)
    # checking whether the lines can compose a parallelogram or not
    if float(fore_count)/float(fore_count + back_count) >= detected_para_threshold:
        return True
    return False


# draw the parallelograms based on the lines given.
def draw_parallelograms(lines, image, quantize_cita, quantize_p):
    size = image.size
    size_x = size[0]
    size_y = size[1]
    for line in lines:
        distance = point_distance(line[0], line[1])
        parameter = line[2]
        # scan the whole image
        for i in range(size_y):
            for j in range(size_x):
                # if the pixel is on the line, make it red
                if point_distance((j, i), line[0]) + point_distance((j, i), line[1]) == distance \
                        and is_point_in_line((j, i), image, parameter, quantize_cita, quantize_p, 1):
                    image.putpixel((j, i), (255, 0, 0))
    return image


# extract points from the lines structure(point1, point2, p_cita)
def get_cross_points(lines):
    points = []
    for line in lines:
        if line[0] not in points:
            points.append(line[0])
        if line[1] not in points:
            points.append(line[1])
    return points


# mark the parallelograms on the original iamge based on the parallelograms marked edge map image.
# the parameter thick is used to decide how thick the lines of the paragram will be marked
def thicken_lines(original_image, image, thick):
    final_image = image.copy()
    size = image.size
    size_x = size[0]
    size_y = size[1]
    for i in range(size_y):
        for j in range(size_x):
            color = image.getpixel((j, i))
            if color == (255, 0, 0):
                for y in range(i-thick, i+thick):
                    for x in range(j-thick, j+thick):
                        original_image.putpixel((x, y), (255, 0, 0))

    return original_image


# image 1c
# gray_scale_threshold
print("For TestImage1c")
gray_scale_threshold = 50
# quantize_cita is used to determine how the cita coordinate should be quantized.
quantize_cita = 1
# quantize_p is used to determine how the p coordinate should be quantized.
quantize_p = 1
# point_count is used in the parameter_list function, determining whether a p-cita value should be a line or not
point_count = 140
# detected_para_threshold using in the is_parallelograms function,
# to detemine the threshold whether the parallelogram is real or not
detected_para_threshold = 0.15
image1 = Image.open("TestImage1c.jpg")
# gradient_magnitude_image is the result image of after the convolution with sobel operator
gradient_magnitude_image = edges_detection(image1)
gradient_magnitude_image.save("1c_gradient_magnitude_image.jpg", "jpeg")
# edge_map_image is the edge map of the gradient magnitude image
edge_map_image = edge_map(gradient_magnitude_image, gray_scale_threshold)
edge_map_image.save("1c_edge_map_image.jpg", "jpeg")
# parameter_space is the list of the coordinates in parameter space that could be a direct line in the edge_map
parameter_space = hough(edge_map_image, quantize_cita, quantize_p, point_count)
# print(parameter_space)
# after optimize the list, we eliminate similar lines
parameter_space = optimize_parameter_space(parameter_space, 5, 15)
# print(parameter_space)
# print out the cross_point of the detected paralled lines
print(cross_point(edge_map_image, parameter_space, quantize_cita, quantize_p))
# lines is a list of structure lines. every item in lines contain two cross point and the parameter in parameter space
lines = paralleled_lines(edge_map_image, parameter_space, quantize_cita, quantize_p)
# save image if we detected parallelograms
if is_parallelograms(lines, edge_map_image, quantize_cita, quantize_p, detected_para_threshold):
    detected_para = draw_parallelograms(lines, edge_map_image, quantize_cita, quantize_p)
    detected_para.save("1c_detected_para.jpg", "jpeg")
    final = thicken_lines(image1, edge_map_image, 2)
    final.save("1c_final.jpg", "jpeg")
    final.show()


# image TestImage2c.jpg
# gray_scale_threshold
print("For TestImage2c")
gray_scale_threshold = 30
# quantize_cita is used to determine how the cita coordinate should be quantized.
quantize_cita = 1
# quantize_p is used to determine how the p coordinate should be quantized.
quantize_p = 1
# point_count is used in the parameter_list function, determining whether a p-cita value should be a line or not
point_count = 70
# detected_para_threshold using in the is_parallelograms function,
# to detemine the threshold whether the parallelogram is real or not
detected_para_threshold = 0.20
image1 = Image.open("TestImage2c.jpg")
# gradient_magnitude_image is the result image of after the convolution with sobel operator
gradient_magnitude_image = edges_detection(image1)
gradient_magnitude_image.save("2c_gradient_magnitude_image.jpg", "jpeg")
# edge_map_image is the edge map of the gradient magnitude image
edge_map_image = edge_map(gradient_magnitude_image, gray_scale_threshold)
edge_map_image.save("2c_edge_map_image.jpg", "jpeg")
# parameter_space is the list of the coordinates in parameter space that could be a direct line in the edge_map
parameter_space = hough(edge_map_image, quantize_cita, quantize_p, point_count)
parameter_space.sort(key=lambda x: x[2])
# # # after optimize the list, we eliminate similar lines
optimize_cita = 20
optimize_p = 60
parameter_space.sort(key=lambda x: x[0])
parameter_space = optimize_parameter_space(parameter_space, optimize_cita, optimize_p)
parameter_space.sort(key=lambda x: x[0])
parameter_space = optimize_parameter_space(parameter_space, optimize_cita, optimize_p)
scale = 0
temp = draw_line(edge_map_image, parameter_space, quantize_cita, quantize_p, scale)

# # # lines is a list of structure lines. every item in lines contain two cross point and the parameter in parameter space
lines = pair_paralleled_lines(parameter_space, 15, 60, 30)
for i in range(len(lines)-1):
    for j in range(i+1, len(lines)):
        line1 = lines[i]
        line2 = lines[j]
        parameter = []
        parameter.append(line1[0])
        parameter.append(line1[1])
        parameter.append(line2[0])
        parameter.append(line2[1])
        temp_lines = paralleled_lines(temp, parameter, quantize_cita, quantize_p)
        if len(temp_lines) < 4:
            continue
        if is_parallelograms(temp_lines, edge_map_image, quantize_cita, quantize_p, detected_para_threshold):
            print(get_cross_points(temp_lines))
            temp = draw_parallelograms(temp_lines, edge_map_image, quantize_cita, quantize_p)
temp.save("2c_parallelograms_marked.jpg", "jpeg")
final_image = thicken_lines(image1, temp, 2)
final_image.save("2c_final_image.jpg", "jpeg")
final_image.show()


# image TestImage3.jpg
print("For TestImage3")
# gray_scale_threshold
gray_scale_threshold = 15
# quantize_cita is used to determine how the cita coordinate should be quantized.
quantize_cita = 1
# quantize_p is used to determine how the p coordinate should be quantized.
quantize_p = 1
# point_count is used in the parameter_list function, determining whether a p-cita value should be a line or not
point_count = 80
# detected_para_threshold using in the is_parallelograms function,
# to detemine the threshold whether the parallelogram is real or not
detected_para_threshold = 0.22
# create an image object based on the input image
image1 = Image.open("TestImage3.jpg")
# gradient_magnitude_image is the result image of after the convolution with sobel operator
gradient_magnitude_image = edges_detection(image1)
gradient_magnitude_image.save("3_gradient_magnitude_image.jpg", "jpeg")
# edge_map_image is the edge map of the gradient magnitude image
edge_map_image = edge_map(gradient_magnitude_image, gray_scale_threshold)
edge_map_image.save("3_edge_map_image.jpg", "jpeg")
# parameter_space is a list of lines return by hough transform.
# it is a list of the coordinates in parameter space that could be a direct line in the edge_map
parameter_space = hough(edge_map_image, quantize_cita, quantize_p, point_count)
# # # after getting the list of possible direct lines, we eliminate similar lines
# we consider lines with in the cita scale of optimize_cita and p value scale of optimize_p are similar lines
optimize_cita = 5
optimize_p = 10
parameter_space.sort(key=lambda x: x[0])
parameter_space = optimize_parameter_space(parameter_space, optimize_cita, optimize_p)
parameter_space.sort(key=lambda x: x[0])
parameter_space = optimize_parameter_space(parameter_space, optimize_cita, optimize_p)
# scale = 0
# temp = draw_line(edge_map_image, parameter_space, quantize_cita, quantize_p, scale)
# # lines is a list of structure lines. every item in lines contain two cross point and the parameter in parameter space
lines = pair_paralleled_lines(parameter_space, 20, 40, 50)
temp = edge_map_image.copy()
for i in range(len(lines)-1):
    for j in range(i+1, len(lines)):
        # compose two pair of lines into a candidate parallelograms
        line1 = lines[i]
        line2 = lines[j]
        parameter = []
        parameter.append(line1[0])
        parameter.append(line1[1])
        parameter.append(line2[0])
        parameter.append(line2[1])
        temp_lines = paralleled_lines(temp, parameter, quantize_cita, quantize_p)
        if len(temp_lines) < 4:
            continue
        if is_parallelograms(temp_lines, edge_map_image, quantize_cita, quantize_p, detected_para_threshold):
            print(get_cross_points(temp_lines))
            temp = draw_parallelograms(temp_lines, temp, quantize_cita, quantize_p)
temp.show()
temp.save("3_parallelograms_marked.jpg", "jpeg")
final_image = thicken_lines(image1, temp, 2)
final_image.save("3_final_image.jpg", "jpeg")
final_image.show()
