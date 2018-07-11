import numpy as np 

##############################################################################
##################### MONTE CARLO WITH FLAT SAMPLING #########################
##############################################################################

# integrand: function (f(z)) to be integrated
# z: z axis value
# sum_f: current total sum of f(z) values
# count: total number of times iterated
# v: volume of integration 
def get_est(integrand, z, sum_f, count, v): 
    if count != 0:                          # so as to not divide by 0 
        mean = sum_f / count                # find mean 
        est = v*mean                        # find integral estimate from lecture notes
        sum_1 = (integrand(z) - mean)**2    
        return est, sum_1
    return 0, 0

# sum_1 is current summation due to integrand and its mean 
# count is number of times iterated so far 
# v is volume of integration
def get_var(sum_1, count, v):
    if count > 1:                           # so as to not divide by 0
        var_f = sum_1 / (count - 1)         # find variance of f 
        var_i = ((v**2) / count) * var_f    # find variance of estimate I 
        return var_i
    return 0


"""
Monte Carlo Integration with Flat Sampling. 

Algorithm performs the above method and obtains an estimate I, est1 and its 
error1, the standard deviation for each iteration. It continously checks for 
each iteration if stopping accuracy condition is satisfied. Once satisfied, 
algorithm takes another independent but same number of samples as the one used 
previously to calculate another estimate I, est2 and its error2. The difference
between error1 and error2 are obtained, and if it is smaller than a stated 
accuracy, e_std, then it will return the final value. 
"""

# monte flat sampling integration algorithm  
# f: integrand
# a: lower limit of integrand
# b: upper limite of integrand 
# e: the stopping accuracy, if relative accuracy is < e, algorithm returns values
# initial_samples: the number of samples to first iterate for 
# (this is used to check if the variances are close enough due to Central Limit Theorem, see report)
# max_samples: the maximum number of samples to use
# shift: the amount of samples that is used for each iteration after the first iteration 
# (which uses initial samples in the first iteration)
# e_std: the accuracy between successive standard deviations of Estimate I's that is desired. 
def flat(f, a, b, e, initial_samples, max_samples, shift, e_std): 
    v = b-a                         # volume of integration
    sum_f1 = 0                      
    sum_f2 = 0
    sum_1 = 0 
    sum_2 = 0 
    samples_est2 = initial_samples
    count1 = 0
    count2 = 0
    
    for k in range(0, max_samples):
        # do for first estimate 
        for i in range(0, initial_samples):
            z1 = np.random.uniform(a,b)
            sum_f1 += f(z1)     # sum values of integrand
            
            est1, sum_temp = get_est(f, z1, sum_f1, count1, v) # get estimate, and current sum_f value
            sum_1 += sum_temp
            
            var_i1 = get_var(sum_1, count1, v)  # get variance
            error_est1 = np.sqrt(var_i1)    # get standard deviation
                    
            count1 += 1     # number of iterations
            
        if abs(error_est1/ est1) <= e:  # check first if variance of this estimate is small enough
            # do for second estimate
            if count2 != 0: # if estimate 2 has already been calculated initially
                samples_est2 = count1 - count2 # so 2nd estimate doesn't repeat samples 
                
            else:
                samples_est2 += count1 # must be same number of samples as the one used for estimate 1
            
            for i in range(0, samples_est2):    # same method as previously for the calculation of estimate 1
                z2 = np.random.uniform(a,b)
                sum_f2 += f(z2)
                
                est2, sum_temp = get_est(f, z2, sum_f2, count2, v)
                sum_2 += sum_temp
                
                var_i2 = get_var(sum_2, count2, v)
                error_est2 = np.sqrt(var_i2)
            
                count2 += 1
              
            if abs(error_est1 - error_est2) <= e_std: # check estimates are not too far apart
                return est1, error_est1, count1

            else:
                initial_samples = 10000 # value cannot be too low else there might be too many checks for the 2nd estimate, so just use a large value to avoid this problem
        else:
            initial_samples = shift   # continue looping and repeat for shift number of samples 
                
    # if max samples has already been reached, 
    print("Flat: Unable to reach within required relative accuracy.")
    return est1, error_est1, count1


##############################################################################
##################### MONTE CARLO WITH IMPORTANCE SAMPLING ###################
##############################################################################
            
# same as get_est_flat, only difference is implementation of sampling_pdf
# sampling_pdf: the sampling pdf that is used
def get_est_imp(integrand, z, sum_f, count, sampling_pdf):
    if count != 0:
        mean = sum_f / count 
        est = mean
        sum_1 = (integrand(z) / sampling_pdf(z) - mean)**2 
        return est, sum_1 
    return 0, 0

# rejection method that was used in project A
# only differencce is that it forces a value to be returned
def rejection(a, b, g):
    y_i = np.random.uniform(a, b)
    p_i = np.random.uniform(0, 1)
    
    if p_i < g(y_i):
        return y_i
    
    else:   # must return a value from the rejection method 
        return rejection(a, b, g)
    
"""
Monte Carlo Integration with Importance Sampling.
 
Variables are all defined in the same way as 'flat' function, and the only 
difference is the use of sampling_pdf, g.  

"""

# g: sampling pdf 
def importance(f, a, b, e, g, initial_samples, max_samples, shift, e_var): 
    sum_f1 = 0
    sum_f2 = 0
    sum_1 = 0 
    sum_2 = 0
    samples_est2 = initial_samples
    count1 = 0
    count2 = 0
    v = 1 # v actually depends on sampling pdf, but use 1 so that we can use the same variance function as above
    
    for k in range(0, max_samples):     # uses two independent samples
        # do for first estimate 
        for i in range(0, initial_samples):
            y_i = rejection(a, b, g)
            sum_f1 += f(y_i)/g(y_i)     # this is why v = 1, as sum_f1 already takes into account of the sampling_pdf
            
            est1, sum_temp = get_est_imp(f, y_i, sum_f1, count1, g)
            sum_1 += sum_temp
            
            var_i1 = get_var(sum_1, count1, v)
            error_est1 = np.sqrt(var_i1)
                    
            count1 += 1
            
        if abs(error_est1/ est1) <= e:  # check first if variance of this estimate is small enough
            # do for second estimate
            if count2 != 0: # to use initial samples_est2   
                samples_est2 = count1 - count2 # so 2nd estimate doesn't repeat samples
            
            else:
                samples_est2 += count1

            for i in range(0, samples_est2):
                y_i = rejection(a, b, g)
                sum_f2 += f(y_i)/g(y_i)
                
                est2, sum_temp = get_est_imp(f, y_i, sum_f2, count2, g)
                sum_2 += sum_temp
                
                var_i2 = get_var(sum_2, count2, v)
                error_est2 = np.sqrt(var_i2)
                        
                count2 += 1
                  
            if abs(error_est1 - error_est2) <= e_var:# check estimates are not too far apart
                return est1, error_est1, count1

            else:
                initial_samples = 10000
        else:
            initial_samples = shift   # continue looping and repeat for shift samples 
                
    print("Imp: Unable to reach within required relative accuracy.")
    return est1, error_est1, count1