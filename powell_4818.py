import sympy as sp
import numpy as np


class opt_tech():
    """
    This is a class for the two main classes of optimization techniques for solving
    1-D unconstraind maximization problems numerically

    ...

    Attributes
    ----------
    fx : sympy class obj
        the objective function you want to find its optimum point
    itr_no : int
        the number of iterations you will go through (default 1)
    prec : int
        the number of decimal places after the decimal point (default 3)
    """

    def __init__(self, fi, itr_no=1, epsilon=0.01):
        """
        The constructor for opt_tech class.

        ...

        Parameters
        ----------
        fi : sympy class obj
            the objective function you want to find its optimum point
        itr_no : int
            the number of iterations you will go through (default 1)
        epsilon : float
            used to help find a suitable search direction Si
        """

        self.fi = fi
        self.itr_no = itr_no
        self.epsilon = epsilon
        self.x = sp.Symbol("x")
        self.y = sp.Symbol("y")
        self.lamda = sp.Symbol("lamda")

    def powells_method(self, X=np.array([0, 0])):
        """
        A direct search Optimization method for solving n-D unconstrained problem.

        ...

        Parameters
        ----------
        X : np.array (vector)
            the initial guess needed to compute the optimum point (default for 2 variables [0, 0])
        """

        print()
        print()
        print("Powelll's Method :")
        print("_________________")

        # initializing variables for our loop
        Z = X

        # find the optimal value for the given number of iterations
        for i in range(self.itr_no):
            # for Cycle #1: S2, S1, S2;
            # we will need to manually initialize S1 & S2
            if i < 3:
                if i % 2:
                    S = np.array([1, 0])  # S1
                else:
                    S = np.array([0, 1])  # S2
            # otherwise, we can calculate it
            else:
                S = X - Z
            print("S{} = {}".format(i+1, S))

            # calculate fi(x, y) at point X
            fi = self.fi.subs({self.x: X[0], self.y: X[1]})

            # calculate fi_+ve & fi_-ve to find the suitable search direction
            fi_postive = self.fi.subs(
                {self.x: X[0]+self.epsilon*S[0], self.y: X[1]+self.epsilon*S[1]})
            fi_negative = self.fi.subs(
                {self.x: X[0]-self.epsilon*S[0], self.y: X[1]-self.epsilon*S[1]})

            print("f{}={}".format(i+1, fi))
            print("f{i}_+ve={:.{prec}f},  f{i}_-ve={:.{prec}f}".format(fi_postive,
                                                                       fi_negative, i=i+1, prec=4))

            # check if we reached our optimum point
            if fi_postive > fi and fi_negative > fi:
                print("Reached Optimum Point at X{} = {}".format(i+1, X))
                break
            # check which direction will make objective function decrease
            elif fi_postive > fi:
                S *= -1
                print("f{} decrease along -ve S{}".format(i+1, i+1))
            else:
                print("f{} decrease along +ve S{}".format(i+1, i+1))

            # to find the approx step length λi* along dir Si
            # sub λ in the objective function
            fi_lamda = self.fi.subs(
                {self.x: X[0]+self.lamda*S[0], self.y: X[1]+self.lamda*S[1]})
            print("f{}_lamda = {}".format(i+1, fi_lamda))

            # get the derivative to find the appropriate step length λ*
            dfi = sp.diff(fi_lamda)
            print("df/d1 = {} = 0".format(dfi))
            lamda = sp.solve(dfi)
            print(f"λ* = {lamda}")
            # calcualte the new approx point
            X = X + lamda * S
            print("X{} = X = {}".format(i+2, X))

            # set our Z (X2) point
            if i == 0:
                Z = X
                print(f"X{i+2} = X = Z = {X}")
            print()
            print()

        print("_________________")

    def steepest_descent(self, X=np.array([0, 0])):
        """
        A (Gradient) Descent search Optimization method for solving n-D unconstrained problem.

        ...

        Parameters
        ----------
        X : np.array (vector)
            the initial guess needed to compute the optimum point (default for 2 variables [0, 0])
        """

        print()
        print()
        print("Steepest Descent Method :")
        print("_______________________")
        print()

        # find the gradient (1st order partial derivative) for our objective function
        gradient_f = [sp.diff(self.fi, self.x), sp.diff(self.fi, self.y)]
        print(f"∇f = {gradient_f}")
        print()

        # find the optimal value for the given number of iterations
        for i in range(self.itr_no):
            # calculate the gradient values at point X
            dx = gradient_f[0].subs({self.x: X[0], self.y: X[1]})
            dy = gradient_f[1].subs({self.x: X[0], self.y: X[1]})
            neg_grad_fi = -1 * np.array([dx, dy])
            print(f"-∇f{i+1} = {neg_grad_fi}")

            # if -∇fi = [0, 0] then we reached out optimal point
            if (neg_grad_fi == np.zeros(2)).all():
                print("Reached Optimum Point at X{} = {}".format(i+1, X))
                break

            # to find the approx step length λi* along dir Si
            # sub λ in the objective function
            fi_lamda = self.fi.subs(
                {self.x: X[0]+self.lamda*neg_grad_fi[0], self.y: X[1]+self.lamda*neg_grad_fi[1]})
            print("f{}_lamda = {}".format(i+1, fi_lamda))

            # get the derivative to find the appropriate step length
            dfi = sp.diff(fi_lamda)
            print("df/dλ = {} = 0".format(dfi))
            lamda = sp.solve(dfi)
            print(f"λ* = {lamda}")

            # calcualte the new approx point
            X = X + lamda * neg_grad_fi
            print("X{} = {}".format(i+2, X))
            print()
        print("_________________")


# the objective function
x = sp.Symbol("x")
y = sp.Symbol("y")
fi = x - y + 2*x**2 + 2*x*y + y**2

# the number of iterations
itr_no = 5

# initializing our optimization techniques class
opt_tech = opt_tech(fi, itr_no)

# our initial guess
X = np.array([0, 0])

# Calling our Powell's Method
opt_tech.powells_method(X)

# the number of iterations
opt_tech.itr_no = 3

# Calling our Steepest Descent Method
opt_tech.steepest_descent(X)
