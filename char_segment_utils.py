# Importing required libraries
import cv2
import numpy as np


def sort_contours(contours, method='left-to-right'):
    """
    Sorts contours either top-to-bottom or left-to-right
    """
    # i = 0 for x co-ordinates
    i = 0
    if method == 'top-to-bottom':
        # i = 1 for y co-ordinates
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    # Sorts contour-boundingBox pairs wrt x values if i = 0, y values if i = 1
    (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes), key=lambda b: b[1][i]))
    return contours


def center(contour):
    """
    Returns the centroid of the contour as a tuple (x,y)
    """
    x, y, w, h = cv2.boundingRect(contour)
    return x + w // 2, y + h // 2


def get_average_height(contours):
    """
    Calculates the average of all the heights of the contours
    """
    heights = []
    for contour in contours:
        _, _, _, h = cv2.boundingRect(contour)
        heights.append(h)
    return sum(heights) / len(heights)


def group_similar_centers(centers, threshold):
    """
    Groups centers vertically close to each other within given threshold.
    Requires the input list to be sorted.
    e.g.
    a list of centers with y co-ordinates [22,23,24,54,54,56,57,91,92] with a threshold of 20,
    and the function will return [[22,23,24],[54,54,56,57],[91,92]]
    """
    groups = []
    group = []
    for i in range(1, len(centers)):
        group.append(centers[i - 1])
        if centers[i][1] - centers[i - 1][1] >= threshold:
            # If the vertical distance between centers exceeds threshold, append
            # the group of centers to the 'groups' list and reset the 'group' list
            groups.append(group)
            group = []
    # Add last center to the group list
    group.append(centers[-1])
    groups.append(group)

    sorted_groups = []
    # Sort each group by x co-ordinates
    for group in groups:
        sorted_groups.append(sorted(group, key=lambda c: c[0]))
    return sorted_groups


def separate_lines(contours):
    """
    Puts all the contours that belong on the same line in a list, does the same for other contours.
    Then puts all the lists in a list and returns said list.
    """
    centers = [center(contour) for contour in contours]
    # Dictionary with center as key and contour as value
    c_dict = dict(zip(centers, contours))

    average_height = get_average_height(contours)
    threshold = average_height / 3
    lines = group_similar_centers(centers, threshold)

    contour_lines = []
    for line in lines:
        contour_line = [c_dict[x] for x in line]
        contour_lines.append(contour_line)
    return contour_lines


def find_major_contours(rgb):
    """
    Finds the major contours in an image and returns them as a list
    """
    # B&W version of the image
    small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    # Elliptical Kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # Difference between dilation and erosion of an image, making outlines
    grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)
    # Converts to monochromatic
    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Rectangular Kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    # Dilation followed by erosion, causes close text to get connected
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    _, contours, _ = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contours = sort_contours(contours, 'top-to-bottom')

    major_contours = []
    # Ignore small contours
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 8 and h > 8:
            major_contours.append(contour)

    return major_contours


def get_box_lines(path):
    """
    Puts all the boxes that belong on the same line in a list, does the same for other boxes.
    Then puts all the lists in a list and returns said list.
    Does the same for spaces between the boxes and the heights of the boxes
    """
    rgb = cv2.imread(path)
    major_contours = find_major_contours(rgb)
    contour_lines = separate_lines(major_contours)
    box_lines = []
    # Dictionary to store dimensions of previous box
    prev_dims = {'x': np.inf, 'h': np.inf, 'w': np.inf}
    space_widths_lines = []
    box_height_lines = []
    for contours in contour_lines:
        box_heights = []
        boxes = []
        space_widths = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append(rgb[y:y + h, x:x + w])
            if (x - prev_dims['x'] - prev_dims['w']) > 0:
                # If box isn't overlapping with the previous box
                space_widths.append(x - prev_dims['x'] - prev_dims['w'])
            else:
                space_widths.append(0)
            prev_dims['x'] = x
            prev_dims['w'] = w
            box_heights.append(h)
        space_widths_lines.append(space_widths)
        box_lines.append(boxes)
        box_height_lines.append(box_heights)
    return box_lines, space_widths_lines, box_height_lines


def image_resize(img_to_resize):
    """
    To eliminate noise and emphasize the text
    """
    height, width = img_to_resize.shape[:2]
    img_resized = cv2.resize(img_to_resize, (int((50 / height) * width), 40), interpolation=cv2.INTER_CUBIC)
    return img_resized


def threshold(img_resized):
    """
    Pre-processing to help find contours
    """
    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    ath = grayscale2bw(img_resized, 150)
    ath = cv2.GaussianBlur(ath, (1, 1), 1)
    ath = cv2.bitwise_not(ath)
    return ath


def drawrect(img_resized, ath, prev_w, space_width):
    """
    Returns:
    1. A list of all the letters in the image
    2. Indices where space should be inserted
    3. Width of the last letter (for spacing)
    """
    image1 = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    _, contours, _ = cv2.findContours(ath, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sort_contours(contours)
    letters = []
    spaces = []
    prev_x = 0
    major_contours = []
    # Ignore contours of height less than 20
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h < 20:
            continue
        major_contours.append(contour)

    for contour, i in zip(major_contours, range(len(major_contours))):
        x, y, w, h = cv2.boundingRect(contour)
        letters.append(image1[y:y + h, x:x + w])
        if i != 0:
            # Only for 2nd letter onwards
            space_width = x - prev_x - prev_w
        # If letters are far away
        if np.ceil((w + prev_w) / 3) < space_width:
            spaces.append(i)
        prev_x = x
        prev_w = w
    return letters, spaces, prev_w


def grayscale2bw(img, thresh):
    """
    Converts a grayscale image to monochromatic black and white.
    """
    img = np.copy(img)
    img[img < thresh] = 0
    img[img >= thresh] = 255
    return img


def pad_letter(img, thresh):
    """
    Reshapes the image to be 28x28 by resizing and providing padding where necessary
    """
    img = grayscale2bw(img, thresh) / 255
    rows, cols = img.shape
    if rows > cols:
        img = cv2.resize(img, (0, 0), fx=1.75, fy=1)
        pad = rows - cols
        lpad = pad // 2
        rpad = pad - lpad
        img = cv2.copyMakeBorder(img, top=0, bottom=0, left=lpad, right=rpad, borderType=cv2.BORDER_CONSTANT,
                                 value=[1, 1, 1])
    elif cols > rows:
        pad = cols - rows
        tpad = pad // 2
        bpad = pad - tpad
        img = cv2.copyMakeBorder(img, top=tpad, bottom=bpad, left=0, right=0, borderType=cv2.BORDER_CONSTANT,
                                 value=[1, 1, 1])
    # Resize letter to 20x20 and add mandatory padding of 4 pixels
    img = cv2.resize(img, (20, 20))
    img = cv2.copyMakeBorder(img, top=4, bottom=4, left=4, right=4, borderType=cv2.BORDER_CONSTANT, value=[1, 1, 1])
    return img


"""def num2char(num):
    keys = list(range(0, 34))
    values = [i for i in '23456789ABCDEFGHIJKLMNOPQRSTUVWXYZ']
    dictionary = dict(zip(keys, values))
    return dictionary[num]"""


def predict_letter(letter, thresh, model):
    """
    Predicts the character in the input image
    """
    l = pad_letter(letter, thresh)
    pred = chr(model.predict((1 - l).reshape(1, 28, 28, 1)).argmax() + 65)
    return pred


def resize_space_widths(space_widths_lines, box_height_lines):
    """
    Rescale the space widths to match the rescaled images
    """
    resized_space_widths_lines = []
    for space_widths, box_heights in zip(space_widths_lines, box_height_lines):
        resized_space_widths = []
        for space_width, box_height in zip(space_widths, box_heights):
            resized_space_widths.append(np.ceil(space_width * 50 / box_height))
        resized_space_widths_lines.append(resized_space_widths)
    return resized_space_widths_lines


def get_letters_and_spaces(box_lines, resized_space_widths_lines):
    """
    Get list of lists of letters and spaces inside boxes
    """
    letters_lines = []
    intra_box_spaces_lines = []
    for boxes, space_widths in zip(box_lines, resized_space_widths_lines):
        space_index = 0
        prev_w = np.inf
        intra_box_spaces_line = []
        letters = []
        for box in boxes:
            resized_box = image_resize(box)
            ath = threshold(resized_box)
            letter, spaces, prev_w = drawrect(resized_box, ath, prev_w, space_widths[space_index])
            letters.append(letter)
            space_index += 1
            intra_box_spaces_line.append(spaces)
        intra_box_spaces_lines.append(intra_box_spaces_line)
        letters_lines.append(letters)
    return letters_lines, intra_box_spaces_lines


def predict_text(letters_lines, intra_box_spaces_lines, model):
    """
    Predict the text and print it
    """
    prediction = ''
    for letters_line, intra_box_spaces_line in zip(letters_lines, intra_box_spaces_lines):
        pred_line = []
        for letters, intra_box_spaces in zip(letters_line, intra_box_spaces_line):
            pred_word = []
            for letter in letters:
                pred = predict_letter(letter, 150, model)
                pred_word.append(pred)
            for space in reversed(intra_box_spaces):
                pred_word.insert(space, ' ')
            pred_line.append(''.join(pred_word))
        prediction += ''.join(pred_line) + '\n'
    print(prediction)
