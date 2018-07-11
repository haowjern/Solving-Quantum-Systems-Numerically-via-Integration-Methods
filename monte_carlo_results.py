import monte_carlo_integration as mc
import matplotlib.pyplot as plt 
import numpy as np
from scipy.optimize import curve_fit


# integrand
def integrand(z):
    return ((1 / np.pi**(1/4))*np.exp(-z**2/2))**2

# sampling pdf
def sampling_pdf(x):
    a = -0.48
    b = 0.98 
    return a*x+b

###############################################################################
# default values 
np.random.seed(1)
lower = 0 
upper = 2 
var_e = 1e-2    # tested to be small enough to not force too much time on our 
                # results, but not too small to be inaccurate
init_samples = 1000
max_samples = 10**6
shift = 10


# adaptive
n_updates = 20    # number of times to repeat loop for
n_samples = 2000 # number of samples in each rectangle 
n = 100 # number of rectangles(grids)
m_multiplier = 10000 # smoothness
alpha = 0.5 # damping

###############################################################################

# Print Values for specified stopping accuracies

error_list = [1e-2, 1e-3]   # stopping accuracies

for i in range(0, len(error_list)):
    est, error, n = mc.flat(integrand, lower, upper, error_list[i], init_samples, max_samples, shift, var_e)
    
    flat_y = n          # number of evaluations
    flat_est = est      # estimate
    flat_error = error  # error in estimate
    
    print("Actual Flat Sampling")
    print("Estimate I: {}".format(flat_est))
    print("Error: {}".format(flat_error))
    print("Relative Accuracy: {}".format(error_list[i])) 
    print("Number of Evaluations: {}\n".format(flat_y)) 
    

for i in range(0, len(error_list)):
    est, error, n = mc.importance(integrand, lower, upper, error_list[i], sampling_pdf, init_samples, max_samples, shift, var_e)
    imp_y = n
    imp_est = est
    imp_error = error

    print("Actual Importance Sampling")
    print("Estimate I: {}".format(imp_est))
    print("Error: {}".format(imp_error))
    print("Relative Accuracy: {}".format(error_list[i])) 
    print("Number of Evaluations: {}\n".format(imp_y)) 
    
# adaptive importance sampling with non proportional sampling pdf
def sampling_pdf_wrong(x):
    a = +0.48   # perviously was negative
    b = 0.02 
    return a*x+b   
    
error_list = [1e-2]
for i in range(0, len(error_list)):
    est, error, n = mc.importance(integrand, lower, upper, error_list[i], sampling_pdf_wrong, init_samples, max_samples, shift, var_e)
    imp_y = n
    imp_est = est
    imp_error = error

    print("Actual Importance Sampling with Non-Proportional Sampling Pdf (Positive Gradient)")
    print("Estimate I: {}".format(imp_est))
    print("Error: {}".format(imp_error))
    print("Relative Accuracy: {}".format(error_list[i])) 
    print("Number of Evaluations: {}\n".format(imp_y)) 
    

# Smaller accuracies takes too long, so should try to extrapolate current data 
# to predict the number of evaluations for smaller accuracies!

###############################################################################
    
# Plot relationship between stopping accuracy and number of evaluations, this can be 
# fitted with a curve to obtain an estimate through the curve for subsequent 
# error estimates!
    
error_list = np.linspace(1e-2, 1e-3, 30) # use these values as this is achievable in a reasonable time frame
var_e = 1e-1 # don't want our condition affecting this, because we assume that 
            # n is large enough for larger numbers for assumption to hold true

est_flat = []
var_flat = []
n_flat = []
est_imp = []
var_imp = []
n_imp = []

for i in range(0, len(error_list)):
    est, var, n = mc.flat(integrand, lower, upper, error_list[i], init_samples, max_samples, shift, var_e)
    est_flat.append(est)
    var_flat.append(var)
    n_flat.append(n)
    
    est, var, n = mc.importance(integrand, lower, upper, error_list[i], sampling_pdf, init_samples, max_samples, shift, var_e)
    est_imp.append(est)
    var_imp.append(var)
    n_imp.append(n)

error_list = np.array(error_list)


# plot flat

plt.figure(1)
plt.plot(error_list, n_flat, 'go', label='Flat Sampling')

# fitting function for flat to find values of m, b1 and c. 
def fit_flat(x, m, c, b1):
    return (m/(x**b1) + c)

popt, cov = curve_fit(fit_flat, error_list, n_flat)
[flat_m, flat_c, b1] = popt

x = np.linspace(error_list[0], error_list[-1], 1000)
y = fit_flat(x, *popt)
    
plt.plot(x, y, 'y-', label='Flat Fit')
plt.legend()


# plot imp

plt.plot(error_list, n_imp, 'bo', label='Importance Sampling')


# fitting function for flat to find values of m, b2 and c. 

def fit_imp(x, m, c, b2):
    return (m/(x**b2) + c)

popt, cov = curve_fit(fit_imp, error_list, n_imp)
[imp_m, imp_c, b2] = popt

x = np.linspace(error_list[0], error_list[-1], 1000)
y = fit_imp(x, *popt)
    
plt.plot(x, y, 'r-', label='Importance Fit')
plt.legend()

plt.xlabel("Relative Accuracies")
plt.ylabel("Number of Evaluations")

print("Equation: Flat Sampling: {:f}/(x^{:f}) + {:f}\n".format(flat_m, b1, flat_c))
print("Equation: Importance Sampling: {:f}/(x^{:f}) + {:f}\n".format(flat_m, b2, flat_c))

###############################################################################

# Use fit to estimate number of calls required for different accuracies 

error_list = [1e-4, 1e-5, 1e-6]
error_list = np.array(error_list)
flat_y = fit_flat(error_list, flat_m, flat_c, b1)
imp_y = fit_imp(error_list, imp_m, imp_c, b2)

for i in range(0, len(error_list)):
    print("Estimation Flat Sampling")
    print("Relative Accuracy: {}".format(error_list[i])) 
    print("Number of Evaluations: {}\n".format(flat_y[i])) 
    
    print("Estimation Importance Sampling")
    print("Relative Accuracy: {}".format(error_list[i])) 
    print("Number of Evaluations: {}\n".format(imp_y[i])) 

plt.show()