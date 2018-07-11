import numpy as np 
import matplotlib.pyplot as plt 
import copy 
from monte_carlo_integration import get_est
from monte_carlo_integration import get_var 

##############################################################################
##################### MONTE CARLO WITH ADAPTIVE SAMPLING #####################
##############################################################################

# this function is similar to the one in monte_carlo_integration for flat sampling, only that it returns an estimate and an error within a specific range, using some number of samples
# f: integrand 
# a: lower limit 
# b: upper limit
# samples: number of samples 
def integ_monte_flat(f, a, b, samples):
    sum_f1 = 0
    sum_1 = 0
    v = b-a
    est1 = 0 
    count1 = 0
    
    for k in range(0, samples):     # uses two independent samples
        # do for first estimate 
        z1 = np.random.uniform(a,b)
        sum_f1 += f(z1)
        
        est1, sum_temp = get_est(f, z1, sum_f1, count1, v)
        sum_1 += sum_temp
        
        var_i1 = get_var(sum_1, count1, v)
        error_est1 = np.sqrt(var_i1)
        
        count1 += 1
        
    return est1, error_est1
        
"""
Monte Carlo with Adaptive Sampling. 

"""
# f: integrand
# a: lower limit
# b: upper limit
# n_updates: number of iterations
# n_samples: number of samples in each rectangle 
# n_regions: number of rectangles(grids)
# K: multiplier - smoothness of distrbution term
# alpha: damping, speed of convergence term
# e: stopping accuracy 

def adaptive(f, a, b, n_updates, n_samples, n_regions, K, alpha, e):
      
    est_list = []   # list of estimates I obtained with each iteration
    error_list = []   # list of respective errors
    
    # intially divide the whole integrand into n_regions, with n_regions + 1 x axis points (named as increment_list)
    delta_x = (b-a)/n_regions
    increment_list = [a]
    for i in range(0, n_regions):
        increment_list.append(increment_list[i] + delta_x)
        
    count = 0 
        
    # main loop     
    for k in range(0, n_updates):   # repeat for n_updates total iterations
             
        # calculate estimates using monte carlo flat sampling for each rectangle 
        est_list_2 = [] # multiple estimates due to multiple rectangles 
        error_list_2 = []
        for i in range(0, n_regions):
            # obtain estimate and error for each rectangle 
            est, error = (integ_monte_flat(f, increment_list[i], increment_list[i+1], n_samples))
            est_list_2.append(est)
            error_list_2.append(error)
            count += 1
            
        est = sum(est_list_2)     # sum estimates from range a to b
        error_list_2 = np.array(error_list_2)   # turn into array so can perform arithmetric operations 
        error = np.sqrt(sum(error_list_2**2)) # add in quadrature
        
        # add results to a list to later make an average best estimate 
        est_list.append(est)
        error_list.append(error)
         
        #################### to make it adaptive #############################
        numerator_list = []   # numerators for each subregion
        for i in range(0, n_regions):   # iterate for each rectangle
            
            # numerator is estimate in each subregion multiplied by its subregion
            numerator = est_list_2[i] * (increment_list[i+1] - increment_list[i])
            numerator_list.append(numerator)

        # denominator is the sum of every contribution 
        denominator = sum(numerator_list)
        
        # find m subincrements
        # m tells you how many steps there are required in between regions 
        # e.g. for the 1st region, if m = 4, 
        # means that there are 4 steps in the region that should be taken into account for 
        # for example, if n_regions = 5 rectangles, from 0 to 2 
        # now there are only 5 steps, x = 0, 0.4, 0.8, 1.2, 1.6, 2.0  (increment_list)
        # m is calculated to be 4, 2, 2, 1, 1 for each region 
        # x is now 0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2, 1.6, 2.0 (sub_increment_list)
        # now have to reshape choose points from sub_increment_list to force total regions back to n_region 
        # hence sum(m) / n_regions == 10 / 5 == 2
        # now our x values are 0, 0.2, 0.4, 0.8, 1.2, 2.0
        m = []
        for i in range(0, len(numerator_list)):
            
            # this is like weighing, ratio is greater when numerator_list[i] is big 
            # which depends on the integrand being big (due to monte carlo method)
            
            ratio = numerator_list[i] / denominator
            
            # K is for smoothness 
            # (1 / nplog(ratio))**alpha is for damping, so that the distribution doesn't converge so fast 
            temp = int(np.ceil(K * ((ratio - 1)*(1/np.log(ratio)))**alpha))
            m.append(temp)
        
        # increments divided by respective m to get subincrements, and append to sub_increment_list 
        sub_increment_list = [a]
        for i in range(0, len(m)):
            shift = (increment_list[i+1] - increment_list[i]) / m[i]
            
            total_shift = copy.copy(increment_list[i])
            for j in range(0, m[i]):
                total_shift += shift 
                sub_increment_list.append(total_shift)
                
                
        # from sub_increment_list, amalgamate and reestablish n_regions increments again
        
        d = (len(sub_increment_list)-1) / (len(increment_list)-1)  
        
        # d might be a floating point, so do the following to estimate how many times to round down and round up
        floor = int(np.floor(d)) # rounds down 
        ceil = int(np.ceil(d)) # roudns up  
        
        diff1 = ceil - d 
        diff2 = d - floor 

        # want to find, for everytime i add either (round down / round up) number
        # how much did i lose?  (e.g. this will be in decimal, e.g. 4.2 - 4 = 0.2), lose 0.2 each time I add 4
        # how many times will I need to lose to get 1? 
        # therefore, 4 * 0.2 = 1 
        # hence, I should add 3 times of (value = 4), and one time of (value = 5), so that on average
        # I get 4.2 
        if diff1 + diff1 == diff1: # if diff1 is 0  (means also diff2 is 0)
            xth_time = 1 
            every_xth_time = int(d) 
            before_xth_time = int(d)
        
        else: 
            if diff1 > diff2: # closer to floor 
                xth_time = int(np.round(1 / diff2)) # find number of times it differs 
                
                before_xth_time = floor
                every_xth_time = ceil 
                
            else: # closer to ceiling 
                xth_time = int(np.round(1 / diff1)) # find number of times it differs 
                
                before_xth_time = ceil
                every_xth_time = floor 
            
        div = []
        for i in range(1, len(increment_list)):
            if i % xth_time == 0: # every xth time 
                div.append(every_xth_time)
                
            else:
                div.append(before_xth_time)
           
        # make sure sum(div) is (len(sub_increment_list) - 1)
        # so can call to the last index
        for i in range(0, len(div)): 
            if div[-1-i] > 1:
                if sum(div) > (len(sub_increment_list)-1):
                    div[-1-i] -= 1 
                    
                elif sum(div) < (len(sub_increment_list)-1):
                    div[-1-i] += 1
                    
                else:
                    break

        # choose every div[i]th division
        index_shift = 0 
        for i in range(0, len(increment_list) - 1):
            increment_list[i] = sub_increment_list[index_shift]
            index_shift += div[i]

        if isinstance(e, float):    # check if error is specified, else do it for n_updates
            # return result and error
            sum_inv_std = 0
            sum_est = 0
        
            # this is for weightage, estimates with larger variances will contribute 
            # less to the final result 
            for i in range(0, k+1):
                sum_inv_std += 1 / error_list[i]  
                sum_est += est_list[i] / error_list[i]   
        
            est_avg = sum_est / sum_inv_std
            
            sum_std = 0
            
            for i in range(0, k+1):
                sum_std += (est_list[i] - est_avg)**2
                
            if k >= 1:
                error_avg = sum_std / k
                error = np.sqrt(error_avg)
            else:
                error = error_list[k]

            if (error / est_avg) < e:
                return est_avg, error, est_list, error_list, count
                
    # return result and error
    sum_inv_std = 0
    sum_est = 0
    
    # this is for weightage, estimates with larger variances will contribute 
    # less to the final result 
    for i in range(0, n_updates):
        sum_inv_std += 1 / error_list[i]  
        sum_est += est_list[i] / error_list[i]   

    est_avg = sum_est / sum_inv_std
    
    sum_std = 0
    
    for i in range(0, n_updates):
        sum_std += (est_list[i] - est_avg)**2
        
    error_avg = sum_std / n_updates
    error = np.sqrt(error_avg)
    
    return est_avg, error, est_list, error_list, count

###############################################################################
# This is just to illustrate plots and how they become adaptive
  
def adaptive_with_plot(f, a, b, n_updates, n_samples, n_regions, K, alpha):
      
    est_list = []   # list of estimates I obtained with each iteration
    error_list = []   # list of respective errors
    
    # intially divide the whole integrand into n_regions, with n_regions + 1 x axis points (named as increment_list)
    delta_x = (b-a)/n_regions
    increment_list = [a]
    for i in range(0, n_regions):
        increment_list.append(increment_list[i] + delta_x)
        
    count = 0 
        
    # main loop     
    for k in range(0, n_updates):   # repeat for n_updates total iterations
        
        # for plotting purposes to illustrate the adaptive distrbution 
        if k == 0 or k == 1 or k == 2 or k == 3 or k == int(n_updates/2) or k == n_updates -1:
            prob = []
            for i in range(0, len(increment_list) - 1):
                prob.append((1/(increment_list[i+1] - increment_list[i])) / n_regions)
            prob.append(prob[-1]) # len of prob_list[i] == len of increment_list[i]
            
            phi = []
            for i in range(0, len(increment_list)):
                phi.append(f(increment_list[i]))
            
            plt.figure(k)
            plt.plot(increment_list, prob, 'ro', label="Adaptive PDF")
            plt.plot(increment_list, phi, 'bo', label="Integrand PDF")
            plt.xlabel("z")
            plt.ylabel("PDF(z)")
            plt.legend()
            plt.show()
         
        # calculate estimates using monte carlo flat sampling for each rectangle 
        est_list_2 = [] # multiple estimates due to multiple rectangles 
        error_list_2 = []
        for i in range(0, n_regions):
            # obtain estimate and error for each rectangle 
            est, error = (integ_monte_flat(f, increment_list[i], increment_list[i+1], n_samples))
            est_list_2.append(est)
            error_list_2.append(error)
            
        count += n_regions*n_samples
            
        est = sum(est_list_2)     # sum estimates from range a to b
        error_list_2 = np.array(error_list_2)   # turn into array so can perform arithmetric operations 
        error = np.sqrt(sum(error_list_2**2)) # add in quadrature
        
        # add results to a list to later make an average best estimate 
        est_list.append(est)
        error_list.append(error)
         
        
        #################### to make it adaptive #############################
        numerator_list = []   # numerators for each subregion
        for i in range(0, n_regions):   # iterate for each rectangle
            
            # numerator is estimate in each subregion multiplied by its subregion
            numerator = est_list_2[i] * (increment_list[i+1] - increment_list[i])
            numerator_list.append(numerator)

        # denominator is the sum of every contribution 
        denominator = sum(numerator_list)
        
        # find m subincrements
        # m tells you how many steps there are required in between regions 
        # e.g. for the 1st region, if m = 4, 
        # means that there are 4 steps in the region that should be taken into account for 
        # for example, if n_regions = 5 rectangles, from 0 to 2 
        # now there are only 5 steps, x = 0, 0.4, 0.8, 1.2, 1.6, 2.0  (increment_list)
        # m is calculated to be 4, 2, 2, 1, 1 for each region 
        # x is now 0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2, 1.6, 2.0 (sub_increment_list)
        # now have to reshape choose points from sub_increment_list to force total regions back to n_region 
        # hence sum(m) / n_regions == 10 / 5 == 2
        # now our x values are 0, 0.2, 0.4, 0.8, 1.2, 2.0
        m = []
        for i in range(0, len(numerator_list)):
            
            # this is like weighing, ratio is greater when numerator_list[i] is big 
            # which depends on the integrand being big (due to monte carlo method)
            
            ratio = numerator_list[i] / denominator
            
            # K is for smoothness 
            # (1 / nplog(ratio))**alpha is for damping, so that the distribution doesn't converge so fast 
            temp = int(np.ceil(K * ((ratio - 1)*(1/np.log(ratio)))**alpha))
            m.append(temp)
        
        # increments divided by respective m to get subincrements, and append to sub_increment_list 
        sub_increment_list = [a]
        for i in range(0, len(m)):
            shift = (increment_list[i+1] - increment_list[i]) / m[i]
            
            total_shift = copy.copy(increment_list[i])
            for j in range(0, m[i]):
                total_shift += shift 
                sub_increment_list.append(total_shift)
                
                
        # from sub_increment_list, amalgamate and reestablish n_regions increments again
        
        d = (len(sub_increment_list)-1) / (len(increment_list)-1)  
        
        # d might be a floating point, so do the following to estimate how many times to round down and round up
        floor = int(np.floor(d)) # rounds down 
        ceil = int(np.ceil(d)) # roudns up  
        
        diff1 = ceil - d 
        diff2 = d - floor 

        # want to find, for everytime i add either (round down / round up) number
        # how much did i lose?  (e.g. this will be in decimal, e.g. 4.2 - 4 = 0.2), lose 0.2 each time I add 4
        # how many times will I need to lose to get 1? 
        # therefore, 4 * 0.2 = 1 
        # hence, I should add 3 times of (value = 4), and one time of (value = 5), so that on average
        # I get 4.2 
        if diff1 + diff1 == diff1: # if diff1 is 0  (means also diff2 is 0)
            xth_time = 1 
            every_xth_time = int(d) 
            before_xth_time = int(d)
        
        else: 
            if diff1 > diff2: # closer to floor 
                xth_time = int(np.round(1 / diff2)) # find number of times it differs 
                
                before_xth_time = floor
                every_xth_time = ceil 
                
            else: # closer to ceiling 
                xth_time = int(np.round(1 / diff1)) # find number of times it differs 
                
                before_xth_time = ceil
                every_xth_time = floor 
            
        div = []
        for i in range(1, len(increment_list)):
            if i % xth_time == 0: # every xth time 
                div.append(every_xth_time)
                
            else:
                div.append(before_xth_time)
           
        # make sure sum(div) is (len(sub_increment_list) - 1)
        # so can call to the last index
        for i in range(0, len(div)): 
            if div[-1-i] > 1:
                if sum(div) > (len(sub_increment_list)-1):
                    div[-1-i] -= 1 
                    
                elif sum(div) < (len(sub_increment_list)-1):
                    div[-1-i] += 1
                    
                else:
                    break

        # choose every div[i]th division
        index_shift = 0 
        for i in range(0, len(increment_list) - 1):
            increment_list[i] = sub_increment_list[index_shift]
            index_shift += div[i]
                
    # return result and error
    sum_inv_std = 0
    sum_est = 0
    
    # this is for weightage, estimates with larger variances will contribute 
    # less to the final result 
    for i in range(0, n_updates):
        sum_inv_std += 1 / error_list[i]  
        sum_est += est_list[i] / error_list[i]   

    est_avg = sum_est / sum_inv_std
    
    sum_std = 0
    
    for i in range(0, n_updates):
        sum_std += (est_list[i] - est_avg)**2
        
    error_avg = sum_std / n_updates
    error = np.sqrt(error_avg)
    
    return est_avg, error, est_list, error_list, count

################# RESULTS ####################################################

lower = 0
upper = 2 
n_updates = 10 
n_samples = 2000
n = 100
m_multiplier = 10000 
alpha = 0.5

def integrand(z):
    #return z**2
    return ((1 / np.pi**(1/4))*np.exp(-z**2/2))**2

error_list = [1e-3, 1e-4, 1e-5, 1e-6]     
for i in range(0, len(error_list)):
    result = adaptive(integrand, lower, upper, n_updates, n_samples, n, m_multiplier, alpha, error_list[i])
    est_a = result[0]
    error_a = result[1]
    est_list_a = result[2]
    var_list_a = result[3]
    iterations_a = result[4] 
    
    
    print("Actual Adaptive Sampling")
    print("Estimate I: {}".format(est_a))
    print("Error: {}".format(error_a))
    print("Relative Accuracy: {}".format(error_list[i])) 
    print("Number of Evaluations: {}\n".format(iterations_a)) 
    
    
###############################################################################
# show adaptive plots
# plots are in the function
    
lower = 0
upper = 2 
n_updates = 10 
n_samples = 2000
n = 100
m_multiplier = 10000 
alpha = 0.5

result = adaptive_with_plot(integrand, lower, upper, n_updates, n_samples, n, m_multiplier, alpha)