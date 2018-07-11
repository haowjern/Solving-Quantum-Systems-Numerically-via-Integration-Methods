import numpy as np
import time
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# integrand function
def integrand(z):
    return (1 / (np.pi**(1/4))*np.exp(-z**2/2))**2

###############################################################################

""" 
Trapezoidal Rule 
"""
# f: integrand
# a: lower limit
# b: upper limit
# e: stopping accuracy 
def integ_trap(f, a, b, e):
    midpoint = (a+b)/2
    f_mid = f(midpoint)
    integ_trap.count += 1   # used for counting number of evaluations
    
    h_1 = b - a
    h_2 = h_1 / 2 
    
    est1 = (h_1/2)*(f(a) + f(b))    # estimate 1
    est2 = est1/2 + h_2*f_mid       # estimate 2 
    
    if abs((est2 - est1)/est1) < e:     # check if within stopping accuracy
        return est2
     
    else:   # recursive calling 
        est_a = integ_trap(f, a, a + h_2, e)   
        est_b = integ_trap(f, a + h_2, b, e)
        return (est_a + est_b)  # sum all points using extended Trapezoidal Rule 
        
    # this is the sum of the two end points/2 + sum of the mid points, change the wording!
    
###############################################################################

"""
Simpson's Rule
"""
# f: integrand
# a: lower limit
# b: upper limit
# e: stopping accuracy 
def integ_simp(f, a, b, e):
    f_mid = f((a+b)/2)
    integ_simp.count += 1  # used for counting number of evaluations
    
    h_1 = b - a
    h_2 = h_1 / 2 
    
    est1 = (h_1/2)*(f(a) + f(b)) # estimate 1 
    est2 = est1/2 + h_2*f_mid   # estimate 2 
    simp = (4*est2 - est1)/3    # trick relating simpson's rule and trapezoidal's rule
    
    if abs((simp - est2)/est2) < e:     # check if within stopping accuracy
        return simp
         
    else:   # recursive calls 
        est_a = integ_simp(f, a, a + h_2, e)
        est_b = integ_simp(f, a + h_2, b, e) 
        return (est_a + est_b)  # extended Simpson's rule 
    
###############################################################################
############################### RESULTS #######################################
###############################################################################
        
# Find results for differents specified stopping accuracies!
a = 0
b = 2 

rel_error = [10**(-3), 10**(-4), 10**(-5), 10**(-6)]

for i in range(0, len(rel_error)):
    integ_trap.count = 0    # count number of calls
    integ_simp.count = 0

    trap = integ_trap(integrand, a, b, rel_error[i])
    simp = integ_simp(integrand, a, b, rel_error[i])
    
    print("--------------Relative Error: {}--------------".format(rel_error[i]))
    
    t = time.time()
    print("Trapezoidal Rule: {}".format(trap))
    print("Trapezoidal Rule Error: {}".format(rel_error[i]))
    print("Time Taken: {}".format(time.time() - t))
    print("Number of Calls: {}\n" .format(integ_trap.count))
    
    t = time.time()
    print("Simpson's Rule with Trap: {}".format(simp))   
    print("Simpson's Rule with Trap Error: {}".format(rel_error[i])) 
    print("Time Taken: {}".format(time.time() - t))
    print("Number of Calls: {}\n".format(integ_simp.count))
    
###############################################################################
# Plot between relationship and number of calls required

rel_error = np.linspace(10**(-5), 10**(-3), 100)

trap_list = []
simp_list = []

trap_calls = []
simp_calls = [] 

for i in range(0, len(rel_error)):
    integ_trap.count = 0 
    integ_simp.count = 0
    
    trap = integ_trap(integrand, a, b, rel_error[i])
    simp = integ_simp(integrand, a, b, rel_error[i])

    trap_calls.append(integ_trap.count)
    simp_calls.append(integ_simp.count)
    
plt.plot(rel_error, trap_calls, 'ro', label="Trapezoidal Rule", alpha = 0.3)
plt.plot(rel_error, simp_calls, 'go', label="Simpson's Rule", alpha = 0.3)


# Trapezoidal Fit

# fit function to find values of m, b and c
def fit_trap(x, m, c, b):
    return (m/(x**b) + c)

popt, cov = curve_fit(fit_trap, rel_error, trap_calls) # curve_fit optimises to find m, b and c
[trap_m, trap_c, b1] = popt

x = np.linspace(rel_error[0], rel_error[-1], 1000)
y = fit_trap(x, *popt)
    
plt.plot(x, y, 'y-', label='Trapezoidal Fit')
print("Equation: Trapezoidal Fitting: {:2f}/(x^{:2f}) + {:2f}\n".format(trap_m, b1, trap_c))


# Simpson Fit

# fit fucntion to find values of m, b and c
def fit_simp(x, m, c, b):
    return (m/(x**b) + c)

popt, cov = curve_fit(fit_simp, rel_error, simp_calls)
[simp_m, simp_c, b2] = popt

x = np.linspace(rel_error[0], rel_error[-1], 1000)
y = fit_simp(x, *popt)
    
plt.plot(x, y, 'b-', label='Simpson Fit')
print("Equation: Simpson Fitting: {:2f}/(x^{:2f}) + {:2f}\n".format(simp_m, b2, simp_c))


plt.legend()
plt.xlabel("Stopping Accuracies")
plt.ylabel("Number of Calls")

plt.show()