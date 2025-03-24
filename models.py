import math

# The model has two parameters w and b.
# There is just one input value x (feature) and one predicted y_hat as output (target).
class LinearRegression:
    def __init__(self, x_train, y_train, w_init=0, b_init=0):
        # Setup fields
        self.x_train = x_train
        self.y_train = y_train
        self.w = w_init
        self.b = b_init
    def model(self, x: float):
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
    def gradientDescentTrain(self, alpha: float, num_iters: int):
        cost_history = []
        param_history = []
        for i in range(num_iters):
            dj_w, dj_b = self.computed_gradient
            self.w -= alpha * dj_w
            self.b -= alpha * dj_b
            if i < 100000:
                cost_history.append(self.computed_cost)
                param_history.append([self.w, self.b])
                if i % math.ceil(num_iters/10) == 0:
                    print(f'Iteração: {i}:  |  J(w,b): {cost_history[-1]:0.2e}  |  w: {self.w}  |  b: {self.b}')
        return cost_history, param_history