import math
import warnings
# The model has two parameters w and b.
# There is just one input value x (feature) and one predicted y_hat as output (target).

class LinearRegression:
    def __init__(self, x_train, y_train, w_init=0, b_init=0):
        # Setup fields
        self.x_train = x_train
        self.y_train = y_train
        self.w = w_init
        self.b = b_init
    def model(self, x):
        return self.w * x + self.b
    @property
    def computed_cost(self):
        m = self.x_train.shape[0]
        total_cost = 0
        for i in range(m):
            total_cost += (self.model(self.x_train[i]) - self.y_train[i])**2
        total_cost /= 2*m
        return total_cost
    @property
    def computed_gradient(self):
        dj_w = dj_b = 0
        m = self.x_train.shape[0]
        for i in range(m):
            dj_w += (self.model(self.x_train[i]) - self.y_train[i]) * self.x_train[i]
            dj_b += self.model(self.x_train[i]) - self.y_train[i]
        dj_w /= m
        dj_b /= m
        return dj_w, dj_b
    
    def gradientDescentDataTrain(self, alpha: float, iteration_limit: int):
        warnings.filterwarnings('error', category=RuntimeWarning)
        print(f'Initial w is {self.w} - Initial b is {self.b}')
        dj_w, dj_b = self.computed_gradient
        for i in range(iteration_limit):
            print(f'w: {self.w} b: {self.b} dj_w: {dj_w} dj_b: {dj_b}')
            try: 
                self.w -= alpha*dj_w
                self.b -= alpha*dj_b
                dj_w, dj_b = self.computed_gradient   
            except RuntimeWarning:
                print(f'Training interrupted at {i}º iteration to avoid infinit type operation and, therefore, NaN results.')
                break
        print(f'New w is {self.w} - New b is {self.b}')
    
