from tests.fuzzy.online.unsupervised.granulation.clip import gaussian


class InputStateVariable(object):
    fuzzy_set_list = []

    def __init__(self, *args):
        self.fuzzy_set_list = args

    def get_fuzzy_sets(self):
        return self.fuzzy_set_list


class Gaussian(object):
    def __init__(self, center, sigma):
        self.center = center
        self.sigma = sigma

    def membership_value(self, input_value):
        return gaussian(input_value, self.center, self.sigma)


class Trapeziums(object):
    def __init__(self, left, left_top, right_top, right):
        self.left = left
        self.right = right
        self.left_top = left_top
        self.right_top = right_top

    def membership_value(self, input_value):
        if (input_value >= self.left_top) and (input_value <= self.right_top):
            membership_value = 1.0
        elif (input_value <= self.left) or (input_value >= self.right):
            membership_value = 0.0
        elif input_value < self.left_top:
            membership_value = (input_value - self.left) / (self.left_top - self.left)
        elif input_value > self.right_top:
            membership_value = (input_value - self.right) / (self.right_top - self.right)
        else:
            membership_value = 0.0
        return membership_value


class Build(object):
    list_of_input_variable = []

    def __init__(self, *args):
        self.list_of_input_variable = args

    def get_input(self):
        return self.list_of_input_variable

    def get_number_of_rules(self):
        number_of_rules = 1
        for input_variable in self.list_of_input_variable:
            number_of_rules = (number_of_rules * self.get_number_of_fuzzy_sets(input_variable))
        return number_of_rules

    def get_number_of_fuzzy_sets(self, input_variable):
        return len(input_variable.get_fuzzy_sets())


# Define membership functions for MountainCar problems
def mountain_car_fis():
    p = InputStateVariable(Trapeziums(-1.2, -1.2, -1.2, -0.775),
                           Trapeziums(-1.2, -0.775, -0.775, -0.35),
                           Trapeziums(-0.775, -0.35, -0.35, 0.075),
                           Trapeziums(-0.35, 0.075, 0.075, 0.5),
                           Trapeziums(0.075, 0.5, 0.5, 0.5))
    v = InputStateVariable(Trapeziums(-0.07, -0.07, -0.07, -0.035),
                           Trapeziums(-0.07, -0.035, -0.035, 0.),
                           Trapeziums(-0.035, 0., 0., 0.035),
                           Trapeziums(0., 0.035, 0.035, 0.07),
                           Trapeziums(0.035, 0.035, 0.035, 0.07))
    fis = Build(p, v)
    return fis
