import pandas as np
import numpy as np
import matplotlib.pyplot as plt
import math


class Point2D:
    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = x
        self.y = y

    def get_coordinates(self):
        return [self.x, self.y]

    def __del__(self):
        class_name = self.__class__.__name__
        print(class_name, "destroyed")


class Triangle:
    def __init__(self, a: Point2D, b: Point2D, c: Point2D):
        self.A = a
        self.B = b
        self.C = c

    def _get_triangle_area(self):
        first = self.A.x * (self.B.y - self.C.y)
        second = self.B.x * (self.C.y - self.A.y)
        third = self.C.x * (self.A.y - self.B.y)
        return abs((first+second+third) / 2.0)

    def check_point_inside(self, p: Point2D):
        '''
        check if a given point is inside the traingle

        :param point:
        :return:
        '''
        # Calculate area of triangle ABC
        total_area = self._get_triangle_area()

        # Calculate area of triangle PBC
        sub_area1 = Triangle(a=p, b=self.B, c=self.C)._get_triangle_area()

        # Calculate area of triangle APC
        sub_area2 = Triangle(a=self.A, b=p, c=self.C)._get_triangle_area()

        # Calculate area of triangle ABP
        sub_area3 = Triangle(a=self.A, b=self.B, c=p)._get_triangle_area()

        # Check if all smaller triangles are the same
        if total_area == sub_area1 + sub_area2 + sub_area3:
            return True
        else:
            return False

    def get_coordinates(self):
        return [self.A.get_coordinates(), self.B.get_coordinates(), self.C.get_coordinates()]

    def __del__(self):
        class_name = self.__class__.__name__
        print(class_name, "destroyed")


# def area(x1, y1, x2, y2, x3, y3):
#     return abs((x1 * (y2 - y3) + x2 * (y3 - y1)
#                 + x3 * (y1 - y2)) / 2.0)


# # A function to check whether point P(x, y)
# # lies inside the triangle formed by
# # A(x1, y1), B(x2, y2) and C(x3, y3)
# def isInside(x1, y1, x2, y2, x3, y3, x, y):
#     # Calculate area of triangle ABC
#     A = area(x1, y1, x2, y2, x3, y3)
#
#     # Calculate area of triangle PBC
#     A1 = area(x, y, x2, y2, x3, y3)
#
#     # Calculate area of triangle PAC
#     A2 = area(x1, y1, x, y, x3, y3)
#
#     # Calculate area of triangle PAB
#     A3 = area(x1, y1, x2, y2, x, y)
#
#     # Check if sum of A1, A2 and A3
#     # is same as A
#     if (A == A1 + A2 + A3):
#         return True
#     else:
#         return False

MAX_VALUE = 10


def generate_data(sample_size):
    return np.random.uniform(low=-MAX_VALUE, high=MAX_VALUE, size=[2, sample_size])


def get_errors_count(triangle, X_in):
    err_cnt = 0
    for x in X_in:
        p = Point2D(x[0], x[1])
        if not triangle.check_point_inside(p):
            err_cnt += 1
    err = err_cnt / X_in.shape[0]
    return err_cnt


def plot_hypothesis_results(m, u, v, w):
    # Data generate:
    X = generate_data(sample_size=m)
    R = 3
    label = []
    unlabeled = []

    # Split data to inside and outside of R:
    for x in X.transpose():
        if np.dot(u, x) <= R and np.dot(v, x) <= R and np.dot(w, x) <= R:
            label.append(x)
        else:
            unlabeled.append(x)
    X_in = np.array(label)
    X_out = np.array(unlabeled)

    # R hypothesis:
    '''

    R_hyp = radius of inside circle !!!

    given you have a traingle that is equilateral - take its height, divide by 3 - that is the radius of the circle inside
    '''
    R_hyp = (X_in[:, 1].max() - X_in[:, 1].min()) / 3
    edge_length = (3 ** 0.5) * 2 * R_hyp
    base_x_val = np.sqrt(edge_length ** 2 - 9 * (R_hyp ** 2))

    '''
    top = np.array([0, 2 * R_hyp])
    right = np.array([base_x_val, -R_hyp])
    left = np.array([-base_x_val, -R_hyp])
    triangle = [top, right, left]
    '''
    top = Point2D(0, 2 * R_hyp)
    right = Point2D(base_x_val, -R_hyp)
    left = Point2D(-base_x_val, -R_hyp)
    triangle = Triangle(top, right, left)

    # Count errors
    get_errors_count(triangle, X_in)

    # plots
    print(f'Plotting figure for m {m}')
    plt.figure(0, figsize=(8,8))
    plt.scatter(top.x, top.y, color='black')
    plt.scatter(right.x, right.y, color='black')
    plt.scatter(left.x, left.y, color='black')

    t1 = plt.Polygon(triangle.get_coordinates(), edgecolor='black', facecolor="none")
    plt.gca().add_patch(t1)

    plt.scatter([0], [0], color='blue', label='Center')
    plt.scatter(u[0], u[1], color='orange', label='u')
    plt.scatter(v[0], v[1], color='pink', label='v')
    plt.scatter(w[0], w[1], color='yellow', label='w')
    plt.scatter(X_in[:, 0], X_in[:, 1], color='green', marker='+', label='labeled (in triangle)')
    plt.scatter(X_out[:, 0], X_out[:, 1], color="red", marker='_', label='unlabeled (not in triangle)')
    plt.xlim(-MAX_VALUE - 2, MAX_VALUE + 2)
    plt.ylim(-MAX_VALUE - 2, MAX_VALUE + 2)
    plt.grid()
    plt.title(f'Plotting for m={m} samples')
    plt.legend()
    plt.show()


def main():
    # given vectors:
    u = np.array([np.sqrt(3) / 2, 0.5])
    w = np.array([-np.sqrt(3) / 2, 0.5])
    v = np.array([0, -1])

    # algorithm parameters:
    M_sample_sizes = [100, 1000, 10000]

    for m in M_sample_sizes:
        plot_hypothesis_results(m, u, v, w)

    print('finish')


if __name__ == '__main__':
    main()
