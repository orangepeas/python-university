# -*- coding: utf-8 -*-
"""
STEP 1
"""
import numpy as np
from matplotlib import pyplot as plt
#from scipy import optimize #never used it

def chisq(y, sig_y, y_m):
    """
    takes model, data and error vectors and calculates the chi2
    this function is copied from one of the animations
    """
    chi2 = np.sum(((y-y_m)**2.0)/(sig_y**2.0))
    return chi2


def chisq_xandy(y, sig_x, sig_y, y_m, gradient):
    """
    takes model, data and error vectors for x and y and calculates the chi2
    this function is partially copied from one of the animations
    """
    chi2 = np.sum(((y-y_m)**2.0)/((gradient**2)*(sig_x**2)+(sig_y**2.0)))
    return chi2


def CPLR():            
    """
    Plots the cepheid period luminosity relationship and uses brute force to find the form of the 
    relationship between them
    """
    (parallax,errparallax,period,mag,A,errA)=np.loadtxt("MW_Cepheids.dat",comments='#',\
                                                        usecols=[1,2,3,4,5,6],unpack=True)
    
    dpc=1000/parallax
    mu=5*np.log10(dpc)-5+A
    MAG=mag-mu
    logP=np.log10(period)
    logP=logP-0.85         #in order to uncorrelate the gradient and intercept, and set the mean of x to be roughly 0
    errdpc=1000*errparallax/parallax**2
    sigma_M=np.sqrt((5/(dpc*np.log(10))*errdpc)**2+(errA)**2)
    
    plt.scatter(logP,MAG,s=50,c='b',marker='x',zorder=1)
    plt.xlabel('log(Period)')
    plt.ylabel('Absolute Magnitude')
    plt.errorbar(logP,MAG,yerr=sigma_M,linestyle='none',c='c', capsize=5)
        
    gradients = np.arange(0,-5,-0.01)         #setting up gradients and intercepts for bruteforce
    #intercepts = np.arange(2,-3,-0.01),      #this was before uncorrelation
    intercepts = np.arange(0,-5,-0.01)        #for the new uncorrelated logP values
    (chi2,slope,intercept,slope_err,intercept_err,chitest) = \
        bruteforce(gradients,intercepts,logP,MAG,sigma_M)
    print("Minimum chisquared = ",chi2,"±",np.sqrt(2*8),", gradient = ",slope, "intercept = ",intercept)

    yfit=slope*logP+intercept        #the model 
    yfit_high=(slope+slope_err)*logP+intercept+intercept_err #most extreme model within errors above gradient
    yfit_low=(slope-slope_err)*logP+intercept-intercept_err #most extreme model within errors below gradient
    #plots the best fit lines
    plt.plot(logP,yfit,c='r',zorder=2)
    plt.plot(logP,yfit_high,c='b',linestyle=':')
    plt.plot(logP,yfit_low,c='b',linestyle=':')
    plt.fill_between(logP,yfit_low,yfit_high,color='c',alpha=0.2) #nice looking fill
    plt.title("Period-luminosity relationship for Cepheids")
    plt.show()
    
    chi2v = chi2/8
    print("The Reduced chisquared = ",chi2v,"plus or minus",np.sqrt(2/8))
    
    print("Error on slope :", slope_err)
    print("Error on intercept :", intercept_err)
    
    #plotting the contours, most of this is copied from lesson15
    plt.plot(slope,intercept,color='black', marker="*", markersize=25)
    delta_chi2 = 45
    levs = [chi2 + delta_chi2]
    plt.contour(gradients,intercepts,chitest,levs)
    plt.axis([-6,2,-5,2])
    plt.xlabel('Slopes')
    plt.ylabel('Intercepts')
    plt.show()
    
    return(slope,intercept,slope_err,intercept_err)



def bruteforce(gradients,intercepts,x,y,sigma_y): 
    """
    inputted with a list of gradients and intercepts that seem like they might be suitable for the plot,
    it pairs each of them and sees which gives the minimum chisquared
    this section of the code has a decent chunk from animation3 of lesson14, as well as lesson15
    """
    gradients_grid, intercepts_grid = np.meshgrid(gradients, intercepts, indexing='xy')
    best_slope = 1.e5 
    best_intercept = 1.e5
    best_chi2 = 1.e5            #large data that is rejected
    chitest=1.e5+gradients_grid*0.0
    
    i=0
    for m in gradients:
        j=0
        for c in intercepts:
            y_test=m*x+c        
            chitest[j,i]=chisq(y,sigma_y,y_test)
            #print(chitest[j,i])
            if chitest[j,i] < best_chi2:
                best_chi2=chitest[j,i]
                best_slope=m
                best_intercept=c
            j=j+1
        i=i+1
        
    slope_low = best_slope + 100.0
    slope_high = best_slope - 100.0
    intercept_low = best_intercept + 100.0
    intercept_high = best_intercept - 100.0
    
    i = 0
    for m in gradients:
        j = 0
        for b in intercepts:    
            if (chitest[j, i] <= (best_chi2 + 1.0)):
                if (m < slope_low):
                    slope_low = m
                if (m > slope_high):
                    slope_high = m 
                if (b < intercept_low):
                    intercept_low = b
                if (b > intercept_high):
                    intercept_high = b
            j = j+1
        i = i+1
        
    slope_err1 = slope_high - best_slope
    slope_err2 = best_slope - slope_low
    slope_err = (slope_err1+slope_err2)/2
    
    intercept_err1 = intercept_high - best_intercept
    intercept_err2 = best_intercept - intercept_low
    intercept_err = (intercept_err1+intercept_err2)/2    
    
    return(best_chi2, best_slope, best_intercept,slope_err,intercept_err,chitest)        
            

"""
STEP 2 & 3
"""

def galaxydistance():
    """
    Works out the distance to each galaxy, then plots that against the recessional velocity 
    and works out the hubble constant by doing 1 over the gradient
    """
    dpclist=np.zeros(8)      #dpclist is the list of the distance of the galaxies in parsecs
    dpclist_err=np.zeros(8)  #initialises two arrays of length 8
    
    (slope,intercept,slope_err,intercept_err)=CPLR()    #gets the model for absolute magnitude from the cepheid period luminosity graph
    (rvel,A)=np.loadtxt('galaxy_data.dat',comments='#',usecols=[1,2],unpack=True,dtype=float)
    allfiles = ['hst_gal1_cepheids.dat','hst_gal2_cepheids.dat','hst_gal3_cepheids.dat','hst_gal4_cepheids.dat','hst_gal5_cepheids.dat','hst_gal6_cepheids.dat','hst_gal7_cepheids.dat','hst_gal8_cepheids.dat']
    
    
    for i in range (8):
        (logP,mag)=np.loadtxt(allfiles[i],comments='#',usecols=[1,2],unpack=True,dtype=float) #imports logP and apparent magnitude
        logP=logP-0.85               #so that the uncorrelated model works
        MAG=slope*(logP)+intercept   #absolute magnitude calculated with model from  CPLR()
        mu=mag-MAG                   #distance modulus
        dpc=10**((mu-A[i]+5)/5)      #array of distances in parsecs to the stars
        
        #sig_M=np.sqrt((logP*slope_err)**2+(intercept_err)**2)    #sig_M is error on absolute magnitude
        #sig_dpc=(0.2*dpc*np.log(10)*sig_M)                       #error propagation
        
        #dpc_weighted_mean=np.sum(dpc/(sig_dpc)**2)/np.sum(1/(sig_dpc)**2) #inverse variance weighted mean
        dpc_mean=np.mean(dpc)                            #taken instead of weighted mean due to errors disagreeing
        dpclist[i]=dpc_mean                              #galaxy distance is the mean of star distance  
        #dpclist_err[i]=np.sqrt(1/(np.sum((1/sig_dpc)**2)))                #inverse variance weighted mean error
        dpclist_err[i]=np.std(dpc)/np.sqrt(len(dpc))     #standard deviation over square root of number of data points 
        #print("The distance modulus is",mu,", the galaxy number is",i+1,)
        #print("mu in parsecs is",mu_parsec,"for galaxy",i)
    #print('dpc list', dpclist)
    #print('errprs on the dpcs',dpclist_err)       #some print statements that were useful

    dmpclist=dpclist*(10**-6)  #converts to megaparsecs
    sig_y=(dpclist_err*10**-6) #converts errors to megaparsecs
    
    plt.scatter(rvel,dmpclist,s=50,c='b',marker='x',zorder=1)  #plots the data points
    plt.ylabel('Distance (Megaparsecs)')
    plt.xlabel('Recession velocity (km/s)')

    gradients=np.arange(0,0.5,0.00001)      #array of gradients to test graph over
    (chi2,h_slope,h_slope_err,slope_high,slope_low) = bruteforce_gradient(gradients,rvel,dmpclist,sig_y)
    
    """the prefix h_ has been added in order to distinguish between the slope and intercept that CPLR() returned
    and the slope and intercept that the brute force function returns"""
    
    print()
    hubble_c=1/h_slope      #h_slope is the slope of dmpc against rvel, slope for age of the universe
    print("Minimum h_chisquared = ",chi2,", h_gradient = ",h_slope)
    # yfit=h_slope*rvel 
    # yfit_high=(slope_high+h_slope_err)*rvel
    # yfit_low=(slope_low-h_slope_err)*rvel
    # plt.plot(rvel,yfit,c='r')     #plots the model line
    # plt.plot(rvel,yfit_high,c='b',linestyle='--',alpha=0.5)
    # plt.plot(rvel,yfit_low,c='b',linestyle='--',alpha=0.5)
    # plt.errorbar(rvel,dmpclist,yerr=sig_y,linestyle='none',c='r')
    # chi2v=chi2/7
    # print("The Reduced chisquared = ",chi2v,"plus or minus",np.sqrt(2/(len(rvel)-1)))
    
    """the above commented code is before intrinsic dispersion, below is with intrinsic dispersion, 
       with the suffix '_id' meaning intrinsic dispersion has been applied""" 
    sig_id=2.45215                         #the error value unaccounted for in the data
    sig_y_id=np.sqrt(sig_y**2+sig_id**2)   #adding in quadrature
    (chi_id,h_slope_id,h_slope_err_id,slope_high_id,slope_low_id) = bruteforce_gradient(gradients,rvel,dmpclist,sig_y_id)

    dof=(len(dmpclist))-1   #number of degrees of freedom
    chi2v_id = chi_id/dof   #reduced chisquared with intrinsic dispersion (should be very close to equal to 1)
    
    print()
    print("Minimum corrected h_chisquared = ",chi_id,"±",np.sqrt(2*dof))
    print("The corrected reduced chisquared = ",chi2v_id,"±",np.sqrt(2/dof))
    
    sig_x=rvel_errs(h_slope,rvel,dmpclist)   #intrinsic dispersion to obtain errors on 
    plt.errorbar(rvel,dmpclist,yerr=sig_y_id, xerr= sig_x, linestyle='none',c='c', \
                alpha=0.7, capsize=5,)       #plots error bars

        
    yfit=h_slope_id*rvel          #the model  
    yfit_high=(h_slope_id+h_slope_err_id)*rvel  #most extreme model within errors above gradient
    yfit_low=(h_slope_id-h_slope_err_id)*rvel   #most extreme model within errors below gradient
    
    chisq_xy=chisq_xandy(dmpclist, sig_x, sig_y, yfit, h_slope_id)
    print("The chisquared due to errors in x and y = ",chisq_xy,"±",np.sqrt(2*dof)\
          ,"meaning reduced chisquared due to errors in x and y= ",chisq_xy/8,"±",np.sqrt(2/dof))
    """this little section below is so that the dashed lines are actually dashed"""
    yfit_high=np.sort(yfit_high)
    yfit_low=np.sort(yfit_low)
    yfit_high=np.array([yfit_high[0],yfit_high[7]])
    yfit_low=np.array([yfit_low[0],yfit_low[7]])
    rvel_dashed=np.sort(rvel)
    rvel_dashed=np.array([rvel_dashed[0],rvel_dashed[7]])

    plt.plot(rvel,yfit,c='r',zorder=2)     #plots the model line
    plt.plot(rvel_dashed,yfit_high,c='b',linestyle='--',alpha=0.6) #plots most extreme model within errors above gradient
    plt.plot(rvel_dashed,yfit_low,c='b',linestyle='--',alpha=0.6) #plots most extreme model within errors below gradient
    plt.fill_between(rvel_dashed,yfit_low,yfit_high,color='cyan',alpha=0.18) #nice looking fill
    plt.title("The Age of the Universe")
    plt.show()
    
    hubble_c=1/h_slope_id
    print("Corrected h_gradient = ",h_slope_id,"±",h_slope_err_id)
    print("Hubble Constant = ",hubble_c,"±",(h_slope_err_id/h_slope**2))
    ageoftheuniverse(h_slope,h_slope_err_id)
    
def rvel_errs(slope,rvel,dmpclist):
    """
    Works out the errors on the recession velocity of the galaxies (sigma), by pretending they are the y values and 
    the distance to the galaxies are the x values, it then works out sigma such that the reduced chisquared 
    of the data is equal to 1
    """
    sigma_0=1
    yfit=(1/slope)*dmpclist
    chi2min=chisq(rvel,sigma_0,yfit)
    red_chi2min=chi2min/7
    sig_rvel=sigma_0*np.sqrt(red_chi2min)
    chi2min=chisq(rvel,sig_rvel,yfit)
    #red_chi2min=chi2min/7
    #print("rvel reduced chi2: ",red_chi2min)
    return(sig_rvel)



def bruteforce_gradient(gradients,x,y,sigma_y): 
    """
    inputted with a list of gradients that seem like they might be suitable for the plot,
    and sees which gives the minimum chisquared
    this section of the code has a decent chunk from animation3 of lesson14, as well as lesson15
    """
    
    best_slope = 1.e5 
    best_chi2 = 1.e5            #large data that is rejected
    chitest=1.e5+gradients*0.0
    
    i=0
    for m in gradients:
        y_test=m*x        
        chitest[i]=chisq(y,sigma_y,y_test)
        if chitest[i] < best_chi2:
            best_chi2=chitest[i]
            best_slope=m
        i=i+1
        
    slope_low = best_slope + 1.e5
    slope_high = best_slope - 1.e5
    i = 0
    for m in gradients:
        if (chitest[i] <= (best_chi2 + 1.0)):
            if (m < slope_low):
                slope_low = m
            if (m > slope_high):
                slope_high = m
        i = i + 1
        
    slope_err1 = slope_high - best_slope
    #print("slope  error 1 =",slope_err1)
    slope_err2 = best_slope - slope_low
    #print("slope  error 2 =",slope_err2)
    slope_err = (slope_err1+slope_err2)/2
    
    return(best_chi2,best_slope,slope_err,slope_high,slope_low)



"""
STEP  4
"""
def ageoftheuniverse(h_slope,h_slope_err):
    age=(h_slope*(3.086*10**19))/(86400*365)         #converts the age of the universe into years
    age_err=h_slope_err*((3.086*10**19)/(86400*365)) #converts the age of the universe errors into years
    print("Age of the universe is",age*10**-9,"billion years ±",age_err*10**-9)
    
    

galaxydistance()