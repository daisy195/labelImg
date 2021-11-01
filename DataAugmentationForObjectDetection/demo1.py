from data_aug.data_aug import *
from data_aug.bbox_util import *
import cv2
from scipy.spatial import distance
import collections

def order_points_old(pts):
    # order points clockwise
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def order_points_new(pts):
    # order points clockwise
    xSorted = pts[np.argsort(pts[:, 0]), :]

    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    (tr, br) = rightMost
    
    return np.array([tl, tr, br, bl], dtype="float32")


def perspective_transform(image, four_points, logo_point):
    four_points_order = order_points_new(four_points)
    idx_top_left = np.argmin(distance.cdist(np.array(four_points_order), np.array([logo_point])).min(axis=1))

    # shift top_left to first
    list_points = collections.deque(list(four_points_order))
    list_points.rotate((-1) * idx_top_left)
    four_points_final = np.float32(list(list_points))
    
    # transform with ratio 300x500
    four_points_new = np.float32([[0, 0], [1000, 0], [1000, 600], [0, 600]])
    M = cv2.getPerspectiveTransform(four_points_final, four_points_new)
    warped = cv2.warpPerspective(image, M, (1000, 600))

    px = (M[0][0]*logo_point[0] + M[0][1]*logo_point[1] + M[0][2]) / ((M[2][0]*logo_point[0] + M[2][1]*logo_point[1] + M[2][2]))
    py = (M[1][0]*logo_point[0] + M[1][1]*logo_point[1] + M[1][2]) / ((M[2][0]*logo_point[0] + M[2][1]*logo_point[1] + M[2][2]))
    logo_after = (int(px), int(py))
    cv2.circle(warped, logo_after, 1, (0,255,0), -1)

    return warped


def four_point_transform(image, pts):
    rect = order_points_new(pts)
    (tl, tr, br, bl) = rect

    # transform with max edge
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


rotate = RandomRotate(10)
shear = RandomShear(0.2)

img = cv2.imread("img.jpeg")
bboxes = np.float32([[68, 557, 97, 586], [83, 56, 112, 85], [853, 51, 882, 80], [
                    851, 540, 880, 569], [166, 110, 297, 240]])
img1 = draw_rect(img, bboxes, color=[0, 255, 0])
cv2.imwrite("img1.jpg", img1)
img, bboxes = rotate(img, bboxes)
img, bboxes = shear(img, bboxes)

img1 = draw_rect(img, bboxes, color=[0, 255, 0])

points = []
for bb in bboxes:
    x = int((bb[0] + bb[2])//2)
    y = int((bb[1] + bb[3])//2)
    img1 = cv2.circle(img1, (x, y), 3, [255, 0, 0], -1)
    points.append((x, y))
cv2.imwrite("out.jpg", img1)
x = int((bboxes[-1][0] + bboxes[-1][2])//2)
y = int((bboxes[-1][1] + bboxes[-1][3])//2)
warped = perspective_transform(img, np.float32(points[:4]), (x, y))


cv2.imwrite("out1.jpg", warped)
