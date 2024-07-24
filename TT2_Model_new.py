from pycbc.types.timeseries import TimeSeries
from copy import deepcopy

#%run TaylorT2_script.py

try:
    # some versions of pycbc include td_taper in pycbc.waveform
    from pycbc.waveform import td_taper
except:
    # some other versions of pycbc include td_taper in pycbc.waveform.utils
    from pycbc.waveform.utils import td_taper

def taper_signal(signal, beta=5):
    """
    Returns tapered signal between start and start+0.4s.

    Parameters
    ----------

    signal : PyCBC TimeSeries
    beta : The beta parameter to use for the Kaiser window. See scipy.signal.kaiser for details. Default is 5.

    Returns
    -------

    signal_tapered : PyCBC TimeSeries
    """
    signal_length = signal.sample_times[-1] - signal.sample_times[0]
    if signal_length <= 0.4:
        taper_window = signal_length/7
    else:
        taper_window = 0.4
    signal_tapered = td_taper(signal, signal.sample_times[0], signal.sample_times[0]+taper_window, beta=beta)
    return(signal_tapered)

def use_modified_input_params(**input_params):
    # sim_inspiral table format uses alpha, alpha1, alpha2.. for additional parameters
    # hence using alpha as a proxy for eccentricity

    modified_input_params = deepcopy(input_params)
    verbose = modified_input_params.get("verbose", False)
    
    if 'alpha' in input_params:
        eccentricity = float(input_params.get("alpha", 0))
        modified_input_params["eccentricity"] = eccentricity
        if verbose:
                print(f"Using eccentricity from `alpha` column, value = {eccentricity}")
    if 'alpha1' in input_params:
        mean_anomaly = float(input_params.get("alpha1", 0))
        modified_input_params["mean_anomaly"] = mean_anomaly
        if verbose:
                print(f"Using mean_anomaly from `alpha1` column, value = {mean_anomaly}")
    return(modified_input_params)






def TaylorT2_Model(**input_params):
    """
    Returns tapered time domain gravitational polarizations for TT2 waveform model containing only the (l,|m|) = (2,2) mode.

    Parameters
    ----------

    Takes the same parameters as pycbc.waveform.get_td_waveform().
    
    Returns
    -------

    hplus : PyCBC TimeSeries
        The plus-polarization of the waveform in time domain tapered from start to 0.4s.
    hcross : PyCBC TimeSeries
        The cross-polarization of the waveform in time domain tapered from start to 0.4s.
    """


    wf_input_params = use_modified_input_params(**input_params)
    
    hp, hc = get_TT2_Model(**wf_input_params) #, modes_to_use=[(2, 2)])
    hp_ts = TimeSeries(hp, input_params['delta_t'])
    hc_ts = TimeSeries(hc, input_params['delta_t'])
    
    hp_tapered = taper_signal(hp_ts)
    hc_tapered = taper_signal(hc_ts)
    return(hp_tapered, hc_tapered)



import pylab
import h5py
import math
import array
import time
import numpy
from numpy import *
import numpy as np
from pycbc.types import TimeSeries, FrequencySeries
from pycbc.waveform import get_td_waveform, get_fd_waveform
from pycbc.waveform.waveform_modes import get_td_waveform_modes
import lal
from scipy import interpolate
from scipy.interpolate import interp1d
from lal import MSUN_SI, MTSUN_SI, G_SI, PC_SI, C_SI, PI
from tqdm import tqdm#

import matplotlib.pyplot as plt
import pyseobnr
from pyseobnr.generate_waveform import GenerateWaveform

EulerGamma=np.euler_gamma
gamma=np.euler_gamma

# In[2]:


# Eq. (4.17a, 4.17b), Pg. 18, Moore et al (2016)

def epsilon(xi, eta):
    return(( 1 + ( ( -2833/2016 + 197/72 * eta ) * ( xi )**( 2/3 ) + 
                  ( -377/144 * np.pi * xi + ( ( 77006005/24385536 + ( -1143767/145152 * eta + 
 	 43807/10368 * ( eta )**( 2 ) ) ) * ( xi )**( 4/3 ) + ( np.pi * ( 9901567/1451520 + 
 	 -202589/362880 * eta ) * ( xi )**( 5/3 ) + ( xi )**( 2 ) * ( -33320661414619/386266890240 + 
 	 ( 3317/252 * EulerGamma + ( 180721/41472 * ( np.pi )**( 2 ) + ( ( 161339510737/8778792960 + 
 	 3977/2304 * ( np.pi )**( 2 ) ) * eta + ( -359037739/20901888 * ( eta )**( 2 ) + 
      ( 10647791/2239488 * ( eta )**( 3 ) + ( -87419/3780 * np.log( 2 ) + 
 	 ( 26001/1120 * np.log( 3 ) + 3317/504 * np.log( 16 * ( xi )**( 2/3 ) ) ) ) ) ) ) ) ) ) ) ) ) ) ))

# ## Oscillatory terms in phasing $W(\xi_{\phi})$
# Eq. (5.3), Pg. 22, Moore et al (2016)

#
#

def W(xie, xi0, eta, e0, l):
    
    xiphi=xie
    xiphi0=xi0
    
    w=e0 * ( ( xiphi )**( -1 ) * xiphi0 )**( 19/18 ) * np.sin(l) * ( ( ( 2 +
         ( ( 7247/1008 + 161/36 * eta ) * ( xiphi )**( 2/3 ) + ( ( 539690101/12192768 +
         ( -477299/72576 * eta + 30055/5184 * ( eta )**( 2 ) ) ) * ( xiphi )**( 4/3 ) +
         ( ( 2833/1008 + -197/36 * eta ) * ( xiphi0 )**( 2/3 ) + ( ( 20530751/2032128 +
         ( -485773/36288 * eta + -31717/2592 * ( eta )**( 2 ) ) ) * ( xiphi )**( 2/3 ) * ( xiphi0 )**( 2/3 ) +
         ( ( -28850671/12192768 + ( 27565/72576 * eta +
         33811/5184 * ( eta )**( 2 ) ) ) * ( xiphi0 )**( 4/3 ) + 377/72 * np.pi * ( -1 * xiphi +
         xiphi0 ) ) ) ) ) ) ) ) ) #+ ( e0 )**( 2 ) * ( ( xiphi )**( -1 ) * xiphi0 )**( 19/9 ) * np.sin(2*l) * ( ( ( 5/4 +
         #( ( 17083/4032 + 841/144 * eta ) * ( xiphi )**( 2/3 ) + ( ( 176530423/6096384 +
         #( 168745/72576 * eta + 37667/2592 * ( eta )**( 2 ) ) ) * ( xiphi )**( 4/3 ) +
         #( ( 14165/4032 + -985/144 * eta ) * ( xiphi0 )**( 2/3 ) + ( ( 48396139/4064256 +
         #( -491399/72576 * eta + -165677/5184 * ( eta )**( 2 ) ) ) * ( xiphi )**( 2/3 ) * ( xiphi0 )**( 2/3 ) +
         #( ( -5966255/12192768 + ( -331585/36288 * eta +
         #90775/5184 * ( eta )**( 2 ) ) ) * ( xiphi0 )**( 4/3 ) +
         #1885/288 * np.pi * ( -1 * xiphi + xiphi0 ) ) ) ) ) ) ) ) )
    w2=e0 * ( ( xiphi )**( -1 ) * xiphi0 )**( 19/18 ) * np.sin(l) * ( ( ( 2 +
         ( ( 7247/1008 + 161/36 * eta ) * ( xiphi )**( 2/3 ) + ( ( 539690101/12192768 +
         ( -477299/72576 * eta + 30055/5184 * ( eta )**( 2 ) ) ) * ( xiphi )**( 4/3 ) +
         ( ( 2833/1008 + -197/36 * eta ) * ( xiphi0 )**( 2/3 ) + ( ( 20530751/2032128 +
         ( -485773/36288 * eta + -31717/2592 * ( eta )**( 2 ) ) ) * ( xiphi )**( 2/3 ) * ( xiphi0 )**( 2/3 ) +
         ( ( -28850671/12192768 + ( 27565/72576 * eta +
         33811/5184 * ( eta )**( 2 ) ) ) * ( xiphi0 )**( 4/3 ) + 377/72 * np.pi * ( -1 * xiphi +
         xiphi0 ) ) ) ) ) ) ) ) ) + ( e0 )**( 2 ) * ( ( xiphi )**( -1 ) * xiphi0 )**( 19/9 ) * np.sin(2*l) * ( ( ( 5/4 +
         ( ( 17083/4032 + 841/144 * eta ) * ( xiphi )**( 2/3 ) + ( ( 176530423/6096384 +
         ( 168745/72576 * eta + 37667/2592 * ( eta )**( 2 ) ) ) * ( xiphi )**( 4/3 ) +
         ( ( 14165/4032 + -985/144 * eta ) * ( xiphi0 )**( 2/3 ) + ( ( 48396139/4064256 +
         ( -491399/72576 * eta + -165677/5184 * ( eta )**( 2 ) ) ) * ( xiphi )**( 2/3 ) * ( xiphi0 )**( 2/3 ) +
         ( ( -5966255/12192768 + ( -331585/36288 * eta +
         90775/5184 * ( eta )**( 2 ) ) ) * ( xiphi0 )**( 4/3 ) +
         1885/288 * np.pi * ( -1 * xiphi + xiphi0 ) ) ) ) ) ) ) ) )
    return(w2)

# ## Mean Anomaly $l (\xi_{\phi})$
# Eq. (C2), Pg. 41, Moore et al (2016)

def mean_anomaly(xie, xi0, cl, eta,e0):  #e0, ### cl is some constant, find out. At the moment set to lref/l0 as required.
    
    l= cl -1/32 * ( eta )**( -1 ) * ( xie )**( -5/3 ) * ( 1 + ( ( -1325/1008 + 
 	 55/12 * eta ) * ( xie )**( 2/3 ) + ( -10 * np.pi * xie + ( ( -41270555/1016064 + 
 	 ( 20845/1008 * eta + 3085/144 * ( eta )**( 2 ) ) ) * ( xie )**( 4/3 ) + 
 	 ( ( xie )**( 2 ) * ( 15398147061251/18776862720 + ( -1712/21 * EulerGamma + ( -160/3 * ( np.pi )**( 2 ) + 
 	 ( ( -22272871555/12192768 + 6355/96 * ( np.pi )**( 2 ) ) * eta + 
 	 ( 96935/6912 * ( eta )**( 2 ) + ( -127825/5184 * ( eta )**( 3 ) + 
 	 -856/21 * np.log( 16 * ( xie )**( 2/3 ) ) ) ) ) ) ) ) + 
 	 ( -1 * np.pi * ( 1675/2016 + 65/24 * eta ) * ( xie )**( 5/3 ) * np.log( xie ) + 
 	 -785/272 * ( e0 )**( 2 ) * ( ( xie )**( -1 ) * xi0 )**( 19/9 ) * ( 1 + 
 	 ( ( 2833/1008 + -197/36 * eta ) * ( xie )**( 2/3 ) + ( ( 117997/2215584 + 
 	 436441/79128 * eta ) * ( xie )**( 2/3 ) + ( -1114537/141300 * np.pi * xie + ( ( -732350735/68366592 + 
 	 ( 271164331/31334688 * eta + 36339727/2238192 * ( eta )**( 2 ) ) ) * ( xie )**( 4/3 ) + 
 	 ( np.pi * ( 270050729/33827220 + -268652717/9664920 * eta ) * ( xie )**( 5/3 ) + 
 	 ( ( 334285501/2233308672 + ( 151648993/9970128 * eta + 
 	 -85978877/2848608 * ( eta )**( 2 ) ) ) * ( xie )**( 2/3 ) * ( xi0 )**( 2/3 ) + 
      ( np.pi * ( -3157483321/142430400 + 
 	 219563789/5086800 * eta ) * xie * ( xi0 )**( 2/3 ) + ( ( -2074749632255/68913524736 + 
 	 ( 15718279597553/189512193024 * eta + ( -1296099941/752032512 * ( eta )**( 2 ) + 
 	 -7158926219/80574912 * ( eta )**( 3 ) ) ) ) * ( xie )**( 4/3 ) * ( xi0 )**( 2/3 ) + 
      ( 377/72 * np.pi * xi0 + 
 	 ( np.pi * ( 44484869/159522048 + 164538257/5697216 * eta ) * ( xie )**( 2/3 ) * xi0 + 
 	 ( -420180449/10173600 * ( np.pi )**( 2 ) * xie * xi0 + ( ( -1193251/3048192 + ( -66317/9072 * eta + 
 	 18155/1296 * ( eta )**( 2 ) ) ) * ( xi0 )**( 4/3 ) + ( ( -140800038247/6753525424128 + 
 	 ( -614686144279/241197336576 * eta + ( -37877198551/957132288 * ( eta )**( 2 ) + 
 	 7923586355/102549888 * ( eta )**( 3 ) ) ) ) * ( xie )**( 2/3 ) * ( xi0 )**( 4/3 ) + 
 	 ( np.pi * ( 764881/90720 + -949457/22680 * eta ) * ( xi0 )**( 5/3 ) + 
 	 ( ( xie )**( 2 ) * ( -231385908692247049/1061268280934400 + ( 12483797/791280 * EulerGamma + 
 	 ( 365639621/13022208 * ( np.pi )**( 2 ) + ( ( 43054867314787/137827049472 + 
 	 -14711579/1446912 * ( np.pi )**( 2 ) ) * eta + ( 55988213933/1640798208 * ( eta )**( 2 ) + 
 	 ( 5885194385/175799808 * ( eta )**( 3 ) + ( 89383841/2373840 * np.log( 2 ) + 
 	 -26079003/703360 * np.log( 16 * ( xie )**( 2/3 ) ) ) ) ) ) ) ) ) + 
 	 ( xi0 )**( 2 ) * ( 26531900578691/168991764480 + 
 	 ( -3317/126 * EulerGamma + ( 122833/10368 * ( np.pi )**( 2 ) + ( ( 9155185261/548674560 + 
 	 -3977/1152 * ( np.pi )**( 2 ) ) * eta + ( -5732473/1306368 * ( eta )**( 2 ) + 
 	 ( -3090307/139968 * ( eta )**( 3 ) + ( 87419/1890 * np.log( 2 ) + ( -26001/560 * np.log( 3 ) + 
 	 -3317/252 * np.log( 16 * ( xi0 )**( 2/3 ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) 
    return(l)


# ##  Relation between post-Newtonian parameter $x$ and time $t (v, v_0, e_0)$ (from TaylorT2)
# Eq. (6.7a, 6.7b), Pg. 25, Moore et al (2016)

def t_from_x(v, v0, e0, M, eta):
    
    t = -5/256 * M * ( v )**( -8 ) * ( eta )**( -1 ) * ( -1 + ( 32/5 * np.pi * ( v )**( 3 ) + 
 	 ( ( v )**( 2 ) * ( -743/252 + -11/3 * eta ) + ( ( v )**( 5 ) * ( 7729/252 * np.pi + 
 	 -13/3 * np.pi * eta ) + ( ( v )**( 4 ) * ( -3058673/508032 + ( -5429/504 * eta + 
 	 -617/72 * ( eta )**( 2 ) ) ) + ( -1 * np.pi * ( v )**( 7 ) * ( -15419335/127008 + 
 	 ( -75703/756 * eta + 14809/378 * ( eta )**( 2 ) ) ) + ( ( v )**( 6 ) * ( 8/3 * ( -3147553127/8128512 + 
 	 451/32 * ( np.pi )**( 2 ) ) * eta + ( 15211/1728 * ( eta )**( 2 ) + 
 	 ( -25565/1296 * ( eta )**( 3 ) + 8/3 * ( 10052469856691/62589542400 + ( -856/35 * EulerGamma + 
 	 ( -16 * ( np.pi )**( 2 ) + -428/35 * ( 4 * np.log( 2 ) + 2 * np.log( v ) ) ) ) ) ) ) ) + 
 	 157/43 * ( e0 )**( 2 ) * ( v )**( -19/3 ) * ( v0 )**( 19/3 ) * ( 1 + ( 377/72 * np.pi * ( v0 )**( 3 ) + 
 	 ( ( v0 )**( 2 ) * ( 2833/1008 + -197/36 * eta ) + ( ( v0 )**( 5 ) * ( 764881/90720 * np.pi + 
 	 -949457/22680 * np.pi * eta ) + ( ( v )**( 5 ) * ( -166558393/12462660 * np.pi + 
 	 -679533343/28486080 * np.pi * eta ) + ( ( v0 )**( 4 ) * ( -1193251/3048192 + 
 	 ( -66317/9072 * eta + 18155/1296 * ( eta )**( 2 ) ) ) + ( ( v )**( 3 ) * ( -2819123/384336 * np.pi + 
 	 ( -1062809371/27672192 * ( np.pi )**( 2 ) * ( v0 )**( 3 ) + ( v0 )**( 2 ) 
      * ( -7986575459/387410688 * np.pi + 
 	 555367231/13836096 * np.pi * eta ) ) ) + ( ( v )**( 4 ) * ( 955157839/302766336 + 
 	 ( 1419591809/88306848 * eta + ( 91918133/6307632 * ( eta )**( 2 ) + 
 	 ( v0 )**( 2 ) * ( 2705962157887/305188466688 + ( 14910082949515/534079816704 * eta + 
 	 ( -99638367319/2119364352 * ( eta )**( 2 ) + -18107872201/227074752 * ( eta )**( 3 ) ) ) ) ) ) ) + 
 	 ( ( v )**( 2 ) * ( 17592719/5855472 + ( 1103939/209124 * eta + 
 	 ( ( v0 )**( 3 ) * ( 6632455063/421593984 * np.pi + 416185003/15056928 * np.pi * eta ) + 
 	 ( ( v0 )**( 2 ) * ( 49840172927/5902315776 + ( -42288307/26349624 * eta + 
 	 -217475983/7528464 * ( eta )**( 2 ) ) ) + ( v0 )**( 4 ) * ( -20992529539469/17848602906624 + 
 	 ( -15317632466765/637450103808 * eta + ( 8852040931/2529563904 * ( eta )**( 2 ) + 
 	 20042012545/271024704 * ( eta )**( 3 ) ) ) ) ) ) ) ) + 
      ( ( v )**( 6 ) * ( -2604595243207055311/16582316889600000 + ( 31576663/2472750 * EulerGamma + 
 	 ( 924853159/40694400 * ( np.pi )**( 2 ) + ( ( 17598403624381/86141905920 + 
 	 -886789/180864 * ( np.pi )**( 2 ) ) * eta + ( 203247603823/5127494400 * ( eta )**( 2 ) + 
 	 ( 2977215983/109874880 * ( eta )**( 3 ) + ( 226088539/7418250 * np.log( 2 ) + 
 	 ( -65964537/2198000 * np.log( 3 ) + 31576663/4945500 * ( 4 * np.log( 2 ) + 
 	 2 * np.log( v ) ) ) ) ) ) ) ) ) ) + ( v0 )**( 6 ) * ( 26531900578691/168991764480 + 
 	 ( -3317/126 * EulerGamma + ( 122833/10368 * ( np.pi )**( 2 ) + 
 	 ( ( 9155185261/548674560 + -3977/1152 * ( np.pi )**( 2 ) ) * eta + ( -5732473/1306368 * ( eta )**( 2 ) + 
 	 ( -3090307/139968 * ( eta )**( 3 ) + ( 87419/1890 * np.log( 2 ) + ( -26001/560 * np.log( 3 ) + 
 	 -3317/252 * ( 4 * np.log( 2 ) + 2 * np.log( v0 ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) )
    
    return(t)


# In[3]:


A=[-622.279, -330.308, -27.0717, -141.883, 5132.37, 6787.92, -17060.5, -2030.2, -5.38518, 250.49, 1.04693, 6608.44, -18.0212] 


def tshift_Hinsp(q,e,l): 
    return A[0] + A[1]*q + A[2]*q**2 + A[3]*e + A[4]*e**2 + A[5]*e*q + A[6]*(e**2)*q + A[7]*(e)*math.cos(l + A[8]) + A[9]*(e**1)*(q**1)*math.cos(e*l + A[10]) + A[11]*(e**2)*math.cos(l + A[12])

B=[24215.3, -120939.0, 91243.2, -166056.0, 328096.0, 673029.0, -1348170.0, -1033.39, 119.166, 3251.62, -980.368, -5624.0, 681.809] 


def tamp_Hinsp(eta,e,l): 
    return B[0] + B[1]*eta + B[2]*eta**2 + B[3]*e + B[4]*e**2 + B[5]*e*eta + B[6]*(e**2)*eta + B[7]*e*math.cos(l + B[8]) + B[9]*(e**1)*(eta**1)*math.cos(l + B[10]) + B[11]*(e**2)*math.cos(e*l + B[12])


C=[-1877112.1819861324, 8928785.261869295, -5753255.361022231, 13060669.274835777, -25197657.592986427, -52105285.776843265, 104979412.53427844, 764720.9257883099, 290.15273485358364, 181821.50372281516, 61.27144003218318, -1308145.2810294377, 446.12549772201976] 


def tfreq_Hinsp(eta,e,l): 
    return C[0] + C[1]*eta + C[2]*eta**2 + C[3]*e + C[4]*e**2 + C[5]*e*eta + C[6]*(e**2)*eta + C[7]*e*eta*math.cos(l + C[8]) + C[9]*(e)*math.cos(l + C[10]) + C[11]*(e**2)*math.cos(e*l + C[12])


# In[4]:


def sph_harmonics(inc,ell):
    L=ell
    #inc = 10
    theta = inc
    for l in range(L,L+1):

        for m in range(-l,l+1):
            dlm = 0;
            k1 = max([0, m-2]);
            k2 = min([l+m, l-2]);

            #if(m==l or m==l-1):
            for k in range(k1,k2+1):
                A = []; B = []; cosTerm = []; sinTerm = []; dlmTmp = [];

                A = (-1)**k*math.sqrt(math.factorial(l+m)*math.factorial(l-m)*math.factorial(l+2)*math.factorial(l-2));
                B = math.factorial(k)*math.factorial(k-m+2)*math.factorial(l+m-k)*math.factorial(l-k-2);

                cosTerm = pow(math.cos(theta/2), 2*l+m-2*k-2);
                sinTerm = pow(math.sin(theta/2), 2*k-m+2);

                dlmTmp = (A/B)*cosTerm*sinTerm;
                dlm = dlm+dlmTmp

            Ylm = math.sqrt((2*l+1)/(4*math.pi))*dlm
            #print('l:',l,'m:',m,'\t Y_lm:',Ylm)
            if m==ell:
                #globals()['sph' + str(l) + str(m)] = Ylm
                #print('l:',l,'m:',m,'\t Y_lm:',Ylm)
                sphlm = Ylm
            elif m==-ell:
                #globals()['sph' + str(l) + '_' + str(abs(m))] = Ylm
                #print('l:',l,'m:',m,'\t Y_lm:',Ylm)
                sphl_m = Ylm
            else:
                continue
    return sphlm, sphl_m


# In[5]:


def xi(x):
    return x**(3/2)

def xconv(f,M):
    return (PI*M*MTSUN_SI*f)**(2/3)  #22 mode conversion

def fconv(x,M):
    return x**(3/2)/(PI*M*MTSUN_SI)


# In[6]:


# sph_harmonics(0,2)


# In[7]:


def TT2_INSP_Eber22_new(M0,q,e0,l0,flow,inc,d0,delta_t):

    eta=neu=nu=q/(1+q)**2
    G=c=M=d=1
    M2=M/(1+q)
    M1=M2*q
    Delta=math.sqrt(1-(4*neu))
    eta=nu=neu
    gamma=EulerGamma=0.577215664901
    mode2polfac=(5/(64*np.pi))**(1/2)
    
    conv=M*MTSUN_SI
    M_SI=M * MSUN_SI
    D_SI=(10**(6)) * PC_SI * d
    
    xlow = ((M0*MTSUN_SI*math.pi*flow)**(2/3))
    f_low = (xlow**(3/2)/(M*MTSUN_SI*math.pi))
    
    #get_ipython().run_line_magic('run', 'GW_functions.ipynb')
    
    x=xlow
    v=math.sqrt(x)
    
    xie=v**3
    
    xVec = np.arange(xlow,1/5,1./(4*4096))
    vVec = sqrt(xVec)
    tVec_PN  = -t_from_x(sqrt(xVec), vVec[0], e0, M, eta)
    

    tC_NR = 0
    
    x0=xlow
    xi0=x0**(3/2)
    v0=xi0**(1/3)
    

    lp=2
    mp=2
    
    j=0
    h22=[]
    h2_2=[]
    
    v=np.sqrt(xVec)
    v0=math.sqrt(x0)
    xie=v**3
    xi0=v0**3
    l=mean_anomaly(xie, xi0, l0, eta, e0)
    e=e0*np.multiply(np.divide(xi0,xie)**(19/18),np.divide(epsilon(xie, eta),epsilon(xi0, eta)))

    j=j+1
    xi=l   #use xi for amplitude (xie is being used for v**3)
    x=xVec
    psi= mp*New_phase_TT2(x, x0, e0, 0.0, 0.0, 0.0, M, nu, Delta)+mp*W(xie, xi0, eta, e0, l) # New TT2 phase (Omkar)
    
    
    h=amplitude_22(xi,x,nu,Delta,e) #### 22 mode requires additional  eccentricity input
    
    hlm=8*math.sqrt(math.pi/5)*M*neu*xVec*h*((np.e)**(complex(0,-1)*mp*psi/2))/d
    hl_m=8*math.sqrt(math.pi/5)*M*neu*xVec*h*((np.e)**(complex(0,+1)*mp*psi/2))/d

    h22 = hlm
    h2_2 = hl_m
    
    conv_t = M0*MTSUN_SI
    conv_h = G_SI*M0*MSUN_SI/(10**6 * PC_SI * d0)/C_SI/C_SI
    
    sph22, sph2_2 = sph_harmonics(inc,lp)
    
    h = np.multiply(h22,sph22) #+np.multiply(h2_2,sph2_2)
    hp=(np.real(h)) 
    hc=(np.imag(h))
    
    time = tVec_PN - tVec_PN[-1]

    mode2polfac=(5/(64*np.pi))**(1/2)  
    
    hp = np.array(hp) * conv_h
    hc = np.array(hc) * conv_h
    time = tVec_PN * conv_t
    
    
    hp_intrp = interp1d(time, hp, kind='cubic',fill_value='extrapolate')
    hc_intrp = interp1d(time, hc, kind='cubic',fill_value='extrapolate')
    t_intrp = np.arange(time.min(), time.max(), delta_t)
    hp_intrp = hp_intrp(t_intrp)
    hc_intrp = hc_intrp(t_intrp)
    

    return np.array(hp_intrp), np.array(hc_intrp), np.array(t_intrp)


def get_TT2_Model(m,q0,e0,l0,fmin,angle,d,delta_t):
    
    M=m
    M1=q0*M/(1+q0)
    M2=M/(1+q0)
    eta=q0/(1+q0)**2
    M_SI=M*MSUN_SI
    D_SI=(10**(6))*PC_SI*d
    inc = angle
    angle = (np.pi/180)*angle
        
    mode2polfac=4*(5/(64*np.pi))**(1/2)
    
    import time
    t1 = time.time()
    hp, hc, tinsp = TT2_INSP_Eber22_new(M,q0,e0,l0,fmin,angle,d,delta_t)
    t2 = time.time()
    #print(t2-t1)
    
    #Circular IMR
    #sp, sc = get_td_waveform(approximant='SEOBNRv4', mass1=M1, mass2=M2, delta_t=delta_t, f_lower=fmin, distance=d)
    #h22IMR = sp+1j*sc

    wfm_gen = GenerateWaveform({"mass1": M1,"mass2": M2,"spin1x": 0, "spin1y": 0, "spin1z": 0, "spin2x": 0, "spin2y": 0, 
        "spin2z": 0, "deltaT": delta_t, "f22_start": fmin,"distance": d,"inclination": 0,"approximant": "SEOBNRv5HM"})
    # times_eob = sp.sample_times
    times_eob, hlm_eob = wfm_gen.generate_td_modes()
    times_eob = times_eob - times_eob[np.argmax(abs(hlm_eob[(2,2)]))]
    # #plt.plot(times,hlm_eob[(2,2)])
    sp = np.real(hlm_eob[(2,2)]) * mode2polfac
    sc = np.imag(hlm_eob[(2,2)]) * mode2polfac
    
    
    h22IMR = sp-1j*sc
    
    
    
    tshift = -tshift_Hinsp(q0,e0,l0)*M*MTSUN_SI
    

    if e0<0.172:
         tshift = 0.0
    
    tmin = max([tinsp[0]-tshift, times_eob[0]])  #modified this part

    tImr_intrp = np.arange(tmin, times_eob[-1], delta_t)
    tImr = tImr_intrp
    H_intrp = interp1d(times_eob, h22IMR, kind='cubic', fill_value='extrapolate')
    h22Imr = H_intrp(tImr_intrp)
        
    
    #Interpolation Ebersold
    hp_intrp = interp1d(tinsp-tshift, hp, kind='cubic',fill_value='extrapolate')
    hc_intrp = interp1d(tinsp-tshift, hc, kind='cubic',fill_value='extrapolate')
    tEcc_intrp = np.arange(tmin, tinsp[-1]-tshift, delta_t)
    hp_intrp = hp_intrp(tEcc_intrp)
    hc_intrp = hc_intrp(tEcc_intrp)
    tEcc = tEcc_intrp
    hpEcc = hp_intrp
    hcEcc = hc_intrp
    h22Ecc = hpEcc + 1j*hcEcc

    phaseEcc = np.unwrap(np.angle(h22Ecc)*2)/2
    phaseImr = np.unwrap(np.angle(h22Imr)*2)/2
    dphase = phaseEcc[0] - phaseImr[0]
    hp_new = real(h22Ecc * exp(-1j * dphase))
    hc_new = imag(h22Ecc * exp(-1j * dphase))
    
    phase_new = np.unwrap(np.angle(hp_new-1j*hc_new)*2)/2
    
    phaseEcc = phase_new 
    phaseImr = phaseImr 
    h22Ecc_new = (hp_new+1j*hc_new)
    
    if tamp_Hinsp(eta,e0,l0) > 0:
        t_join = -20
    else:
        t_join = tamp_Hinsp(eta,e0,l0)
    
    arg = np.argmin(abs(tEcc-t_join*M*MTSUN_SI))
    Idxjoin = arg
    
    t_amp = tEcc[Idxjoin] - 500*M*MTSUN_SI
    idxstr = np.argmin(abs(tEcc-t_amp))
    
    #Amplitude Model
    amp=[]
    count=0
    length=Idxjoin-idxstr
    
    for i in range(idxstr,Idxjoin):
        amp.append(((length-count)*abs(h22Ecc_new[i])+count*abs(h22Imr[i]))/length)
        count=count+1
    
    t_model=np.concatenate((tEcc[0:Idxjoin],tImr[Idxjoin:len(tImr)]))
    h22amp=np.concatenate((abs(h22Ecc_new[0:idxstr]),amp))
    h22amp_model=np.concatenate((h22amp,abs(h22Imr[Idxjoin:len(h22Imr)])))    
    
    omegaEcc = (M*MTSUN_SI/delta_t)*(np.gradient(phaseEcc))
    omegaImr = (M*MTSUN_SI/delta_t)*(np.gradient(phaseImr))
    
    if tfreq_Hinsp(eta,e0,l0) > 0:
        tjoin0 = -20
    else:
        tjoin0 = tfreq_Hinsp(eta,e0,l0)
        
    tjoin = tjoin0 * M * MTSUN_SI
    fjoin = np.argmin(abs(tEcc-tjoin))
        
    #frequency model
    tstop = min([tEcc[-1],-30*M*MTSUN_SI])
    lst=np.argmin(abs(tEcc-tstop))
    
    indx=lst - fjoin
    a0 = []
    n = indx-1
    k = 0
    for i in range(fjoin,fjoin+indx):
        a0.append(((n-k)*omegaEcc[i]+k*omegaImr[i])/n)
        k=k+1
    
    f1 = np.concatenate((omegaEcc[0:fjoin],a0))
    frequency_model = np.concatenate((f1,omegaImr[fjoin+indx:len(omegaImr)]))
    phase_f_model = np.cumsum(frequency_model)/(M*MTSUN_SI/delta_t)
    phase_f_model = phase_f_model - phase_f_model[0] + phaseEcc[0]
    
    ll = min(len(h22amp_model),len(phase_f_model))
    hp_f_model = h22amp_model[:ll] * np.cos(phase_f_model[:ll])
    hc_f_model = h22amp_model[:ll] * np.sin(phase_f_model[:ll])
    
    ht = (mode2polfac/4)*(((1+math.cos(angle))**2 * (hp_f_model - 1j*hc_f_model)) + ((1-math.cos(angle))**2 * (hp_f_model + 1j*hc_f_model)))
    hplus = np.real(ht)
    hcross = np.imag(ht)
    
    hp_model_TS = TimeSeries(hplus,delta_t)
    hc_model_TS = TimeSeries(hcross,delta_t)
  
    return hp_model_TS, hc_model_TS


# ## Amplitude $(\ell , m)$ = (2, 2)

def amplitude_22(xi, x, nu, Delta, e):
    h = 1 + ( e * ( 1/4 * ( np.e )**( complex( 0,-1 ) * xi ) + 
 	 5/4 * ( np.e )**( complex( 0,1 ) * xi ) ) + 
 	 ( ( e )**( 2 ) * ( -1/2 + 
 	 ( 1/4 * ( np.e )**( complex( 0,-2 ) * xi ) + 
 	 7/4 * ( np.e )**( complex( 0,2 ) * xi ) ) ) + 
 	 ( ( e )**( 3 ) * ( -5/32 * ( np.e )**( complex( 0,-1 ) * xi ) + 
 	 ( -33/32 * ( np.e )**( complex( 0,1 ) * xi ) + 
 	 ( 9/32 * ( np.e )**( complex( 0,-3 ) * xi ) + 
 	 77/32 * ( np.e )**( complex( 0,3 ) * xi ) ) ) ) + 
 	 ( ( e )**( 4 ) * ( -1/8 + 
 	 ( -1/6 * ( np.e )**( complex( 0,-2 ) * xi ) + 
 	 ( -11/6 * ( np.e )**( complex( 0,2 ) * xi ) + 
 	 ( 1/3 * ( np.e )**( complex( 0,-4 ) * xi ) + 
 	 79/24 * ( np.e )**( complex( 0,4 ) * xi ) ) ) ) ) + 
 	 ( ( e )**( 5 ) * ( -47/768 * ( np.e )**( complex( 0,-1 ) * xi ) + 
 	 ( -11/768 * ( np.e )**( complex( 0,1 ) * xi ) + 
 	 ( -117/512 * ( np.e )**( complex( 0,-3 ) * xi ) + 
 	 ( -1585/512 * ( np.e )**( complex( 0,3 ) * xi ) + 
 	 ( 625/1536 * ( np.e )**( complex( 0,-5 ) * xi ) + 
 	 6901/1536 * ( np.e )**( complex( 0,5 ) * xi ) ) ) ) ) ) + 
 	 ( ( e )**( 6 ) * ( -1/16 + 
 	 ( -1/32 * ( np.e )**( complex( 0,-2 ) * xi ) + 
 	 ( 1/3 * ( np.e )**( complex( 0,2 ) * xi ) + 
 	 ( -1/3 * ( np.e )**( complex( 0,-4 ) * xi ) + 
 	 ( -403/80 * ( np.e )**( complex( 0,4 ) * xi ) + 
 	 ( 81/160 * ( np.e )**( complex( 0,-6 ) * xi ) + 
 	 49/8 * ( np.e )**( complex( 0,6 ) * xi ) ) ) ) ) ) ) + 
 	 ( x * ( -107/42 + 
 	 ( 55/42 * nu + 
 	 ( e * ( ( np.e )**( complex( 0,-1 ) * xi ) * ( -257/168 + 
 	 169/168 * nu ) + 
 	 ( np.e )**( complex( 0,1 ) * xi ) * ( -31/24 + 
 	 35/24 * nu ) ) + 
 	 ( ( e )**( 2 ) * ( -221/84 + 
 	 ( 89/84 * nu + 
 	 ( ( np.e )**( complex( 0,-2 ) * xi ) * ( -289/168 + 
 	 181/168 * nu ) + 
 	 ( np.e )**( complex( 0,2 ) * xi ) * ( -71/168 + 
 	 283/168 * nu ) ) ) ) + 
 	 ( ( e )**( 3 ) * ( ( np.e )**( complex( 0,-1 ) * xi ) * ( -1115/1344 + 
 	 895/1344 * nu ) + 
 	 ( ( np.e )**( complex( 0,1 ) * xi ) * ( -685/192 + 
 	 79/64 * nu ) + 
 	 ( ( np.e )**( complex( 0,-3 ) * xi ) * ( -2813/1344 + 
 	 559/448 * nu ) + 
 	 ( np.e )**( complex( 0,3 ) * xi ) * ( 1027/1344 + 
 	 2729/1344 * nu ) ) ) ) + 
 	 ( ( e )**( 4 ) * ( -183/112 + 
 	 ( ( np.e )**( complex( 0,-2 ) * xi ) * ( -19/36 + 
 	 5/9 * nu ) + 
 	 ( 233/336 * nu + 
 	 ( ( np.e )**( complex( 0,2 ) * xi ) * ( -1367/252 + 
 	 82/63 * nu ) + 
 	 ( ( np.e )**( complex( 0,-4 ) * xi ) * ( -379/144 + 
 	 215/144 * nu ) + 
 	 ( np.e )**( complex( 0,4 ) * xi ) * ( 160/63 + 
 	 157/63 * nu ) ) ) ) ) ) + 
 	 ( ( e )**( 5 ) * ( ( np.e )**( complex( 0,-3 ) * xi ) * ( -2815/21504 + 
 	 3137/7168 * nu ) + 
 	 ( ( np.e )**( complex( 0,-1 ) * xi ) * ( -20249/32256 + 
 	 16465/32256 * nu ) + 
 	 ( ( np.e )**( complex( 0,1 ) * xi ) * ( -38305/32256 + 
 	 21661/32256 * nu ) + 
 	 ( ( np.e )**( complex( 0,3 ) * xi ) * ( -183655/21504 + 
 	 29423/21504 * nu ) + 
 	 ( ( np.e )**( complex( 0,-5 ) * xi ) * ( -217369/64512 + 
 	 117265/64512 * nu ) + 
 	 ( np.e )**( complex( 0,5 ) * xi ) * ( 337087/64512 + 
 	 199165/64512 * nu ) ) ) ) ) ) + 
 	 ( e )**( 6 ) * ( -877/672 + 
 	 ( ( np.e )**( complex( 0,-4 ) * xi ) * ( 4873/10080 + 
 	 551/2016 * nu ) + 
 	 ( ( np.e )**( complex( 0,-2 ) * xi ) * ( -295/504 + 
 	 149/336 * nu ) + 
 	 ( ( np.e )**( complex( 0,2 ) * xi ) * ( -1495/4032 + 
 	 1895/4032 * nu ) + 
 	 ( 377/672 * nu + 
 	 ( ( np.e )**( complex( 0,4 ) * xi ) * ( -69071/5040 + 
 	 2389/1680 * nu ) + 
 	 ( ( np.e )**( complex( 0,-6 ) * xi ) * ( -458/105 + 
 	 313/140 * nu ) + 
 	 ( np.e )**( complex( 0,6 ) * xi ) * ( 62233/6720 + 
 	 5167/1344 * nu ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) + 
 	 ( ( x )**( 2 ) * ( -2173/1512 + 
 	 ( -1069/216 * nu + 
 	 ( 2047/1512 * ( nu )**( 2 ) + 
 	 ( ( e )**( 3 ) * ( ( np.e )**( complex( 0,-3 ) * xi ) * ( -35069/1512 + 
 	 ( -459619/48384 * nu + 
 	 -144821/48384 * ( nu )**( 2 ) ) ) + 
 	 ( ( np.e )**( complex( 0,3 ) * xi ) * ( -1432/27 + 
 	 ( 1320281/48384 * nu + 
 	 -1679/6912 * ( nu )**( 2 ) ) ) + 
 	 ( ( np.e )**( complex( 0,1 ) * xi ) * ( -34127/3024 + 
 	 ( -632117/48384 * nu + 
 	 14285/48384 * ( nu )**( 2 ) ) ) + 
 	 ( np.e )**( complex( 0,-1 ) * xi ) * ( -11833/3024 + 
 	 ( -519889/48384 * nu + 
 	 15457/48384 * ( nu )**( 2 ) ) ) ) ) ) + 
 	 ( ( e )**( 2 ) * ( -34399/3024 + 
 	 ( -3271/432 * nu + 
 	 ( 829/3024 * ( nu )**( 2 ) + 
 	 ( ( np.e )**( complex( 0,-2 ) * xi ) * ( -77575/6048 + 
 	 ( -42697/6048 * nu + 
 	 -4523/6048 * ( nu )**( 2 ) ) ) + 
 	 ( np.e )**( complex( 0,2 ) * xi ) * ( -16033/672 + 
 	 ( 14467/2016 * nu + 
 	 209/224 * ( nu )**( 2 ) ) ) ) ) ) ) + 
 	 ( e * ( ( np.e )**( complex( 0,-1 ) * xi ) * ( -4271/756 + 
 	 ( -35131/6048 * nu + 
 	 421/864 * ( nu )**( 2 ) ) ) + 
 	 ( np.e )**( complex( 0,1 ) * xi ) * ( -2155/252 + 
 	 ( -1655/672 * nu + 
 	 371/288 * ( nu )**( 2 ) ) ) ) + 
 	 ( ( e )**( 4 ) * ( -51565/4032 + 
 	 ( -7009/576 * nu + 
 	 ( -605/4032 * ( nu )**( 2 ) + 
 	 ( ( np.e )**( complex( 0,-4 ) * xi ) * ( -348401/9072 + 
 	 ( -61651/4536 * nu + 
 	 -62557/9072 * ( nu )**( 2 ) ) ) + 
 	 ( ( np.e )**( complex( 0,4 ) * xi ) * ( -1258321/12096 + 
 	 ( 86113/1344 * nu + 
 	 -35033/12096 * ( nu )**( 2 ) ) ) + 
 	 ( ( np.e )**( complex( 0,-2 ) * xi ) * ( -21593/6048 + 
 	 ( -59167/6048 * nu + 
 	 6779/6048 * ( nu )**( 2 ) ) ) + 
 	 ( np.e )**( complex( 0,2 ) * xi ) * ( -41117/18144 + 
 	 ( -462659/18144 * nu + 
 	 24575/18144 * ( nu )**( 2 ) ) ) ) ) ) ) ) ) + 
 	 ( ( e )**( 5 ) * ( ( np.e )**( complex( 0,-5 ) * xi ) * (-17610839/290304 + 
 	 ( -46898683/2322432 * nu + 
 	 -31148909/2322432 * ( nu )**( 2 ) ) ) + 
 	 ( ( np.e )**( complex( 0,5 ) * xi ) * ( -677277/3584 + 
 	 ( 98415547/774144 * nu + 
 	 -2079481/258048 * ( nu )**( 2 ) ) ) + 
 	 ( ( np.e )**( complex( 0,1 ) * xi ) * ( -836161/72576 + 
 	 ( -20348767/1161216 * nu + 
 	 -694625/1161216 * ( nu )**( 2 ) ) ) + 
 	 ( ( np.e )**( complex( 0,-1 ) * xi ) * ( -491/189 + 
 	 ( -2110099/129024 * nu + 
 	 -31631/387072 * ( nu )**( 2 ) ) ) + 
 	 ( ( np.e )**( complex( 0,-3 ) * xi ) * ( 8789/13824 + 
 	 ( -5939153/774144 * nu + 
 	 2830337/774144 * ( nu )**( 2 ) ) ) + 
 	 ( np.e )**( complex( 0,3 ) * xi ) * ( 2717333/96768 + 
 	 ( -43930853/774144 * nu + 
 	 3420269/774144 * ( nu )**( 2 ) ) ) ) ) ) ) ) + 
 	 ( e )**( 6 ) * ( -48545/3456 + 
 	 ( -53279/3456 * nu + 
 	 ( -9979/24192 * ( nu )**( 2 ) + 
 	 ( ( np.e )**( complex( 0,-6 ) * xi ) * ( -22510693/241920 + 
 	 ( -1853953/60480 * nu + 
 	 -2900389/120960 * ( nu )**( 2 ) ) ) + 
 	 ( ( np.e )**( complex( 0,6 ) * xi ) * ( -986303/3024 + 
 	 ( 11177365/48384 * nu + 
 	 -4196279/241920 * ( nu )**( 2 ) ) ) + 
 	 ( ( np.e )**( complex( 0,2 ) * xi ) * ( -1092307/72576 + 
 	 ( -2387339/145152 * nu + 
 	 -197119/145152 * ( nu )**( 2 ) ) ) + 
 	 ( ( np.e )**( complex( 0,-2 ) * xi ) * ( -641191/145152 + 
 	 ( -291587/18144 * nu + 
 	 -30157/72576 * ( nu )**( 2 ) ) ) + 
 	 ( ( np.e )**( complex( 0,-4 ) * xi ) * ( 210949/18144 + 
 	 ( -45317/18144 * nu + 
 	 866989/90720 * ( nu )**( 2 ) ) ) + 
 	 ( np.e )**( complex( 0,4 ) * xi ) * ( 36131317/362880 + 
 	 ( -44839757/362880 * nu + 
 	 4173977/362880 * ( nu )**( 2 ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) + 
 	 ( ( x )**( 3/2 ) * ( 2 * np.pi + 
 	 ( e * ( ( np.e )**( complex( 0,1 ) * xi ) * ( 13/4 * np.pi + 
 	 complex( 0,3/2 ) * np.log( 2 ) ) + 
 	 ( np.e )**( complex( 0,-1 ) * xi ) * ( 11/4 * np.pi + 
 	 ( complex( 0,-27/2 ) * np.log( 2 ) + 
 	 complex( 0,27/2 ) * np.log( 3 ) ) ) ) + 
 	 ( ( e )**( 2 ) * ( 2 * np.pi + 
 	 ( ( np.e )**( complex( 0,2 ) * xi ) * ( 5 * np.pi + 
 	 ( complex( 0,-13/252 ) * nu + 
 	 complex( 0,3 ) * np.log( 2 ) ) ) + 
 	 ( complex( 0,-30 ) * np.log( 2 ) + 
 	 ( ( np.e )**( complex( 0,-2 ) * xi ) * ( 4 * np.pi + 
 	 ( complex( 0,59 ) * np.log( 2 ) + 
 	 complex( 0,-27 ) * np.log( 3 ) ) ) + 
 	 complex( 0,27 ) * np.log( 3 ) ) ) ) ) + 
 	 ( ( e )**( 3 ) * ( ( np.e )**( complex( 0,3 ) * xi ) * ( 703/96 * np.pi + 
 	 ( complex( 0,-13/126 ) * nu + 
 	 complex( 0,227/48 ) * np.log( 2 ) ) ) + 
 	 ( ( np.e )**( complex( 0,1 ) * xi ) * ( 59/32 * np.pi + 
 	 ( complex( 0,13/126 ) * nu + 
 	 ( complex( 0,-811/16 ) * np.log( 2 ) + 
 	 complex( 0,351/8 ) * np.log( 3 ) ) ) ) + 
 	 ( ( np.e )**( complex( 0,-1 ) * xi ) * ( 61/32 * np.pi + 
 	 ( complex( 0,2419/16 ) * np.log( 2 ) + 
 	 complex( 0,-1377/16 ) * np.log( 3 ) ) ) + 
 	 ( np.e )**( complex( 0,-3 ) * xi ) * ( 185/32 * np.pi + 
 	 ( complex( 0,-6683/48 ) * np.log( 2 ) + 
 	 ( complex( 0,81/8 ) * np.log( 3 ) + 
 	 complex( 0,3125/48 ) * np.log( 5 ) ) ) ) ) ) ) + 
 	 ( ( e )**( 4 ) * ( 2 * np.pi + 
 	 ( complex( 0,-13/336 ) * nu + 
 	 ( ( np.e )**( complex( 0,4 ) * xi ) * ( 125/12 * np.pi + 
 	 ( complex( 0,-169/1008 ) * nu + 
 	 complex( 0,85/12 ) * np.log( 2 ) ) ) + 
 	 ( complex( 0,527/2 ) * np.log( 2 ) + 
 	 ( ( np.e )**( complex( 0,2 ) * xi ) * ( 11/8 * np.pi + 
 	 ( complex( 0,5/72 ) * nu + 
 	 ( complex( 0,-467/6 ) * np.log( 2 ) + 
 	 complex( 0,531/8 ) * np.log( 3 ) ) ) ) + 
 	 ( complex( 0,-621/4 ) * np.log( 3 ) + 
 	 ( ( np.e )**( complex( 0,-4 ) * xi ) * ( 199/24 * np.pi + 
 	 ( complex( 0,1837/12 ) * np.log( 2 ) + 
 	 ( complex( 0,981/8 ) * np.log( 3 ) + 
 	 complex( 0,-3125/24 ) * np.log( 5 ) ) ) ) + 
 	 ( np.e )**( complex( 0,-2 ) * xi ) * ( 17/12 * np.pi + 
 	 ( complex( 0,-2555/6 ) * np.log( 2 ) + 
 	 ( complex( 0,351/4 ) * np.log( 3 ) + 
 	 complex( 0,3125/24 ) * np.log( 5 ) ) ) ) ) ) ) ) ) ) ) + 
 	 ( ( e )**( 5 ) * ( ( np.e )**( complex( 0,5 ) * xi ) * ( 37257/2560 * np.pi + 
 	 ( complex( 0,-767/3024 ) * nu + 
 	 ( complex( 0,38203/3840 ) * np.log( 2 ) + 
 	 complex( 0,459/1280 ) * np.log( 3 ) ) ) ) + 
 	 ( ( np.e )**( complex( 0,3 ) * xi ) * ( 469/1536 * np.pi + 
 	 ( complex( 0,25/336 ) * nu + 
 	 ( complex( 0,-88579/768 ) * np.log( 2 ) + 
 	 complex( 0,3105/32 ) * np.log( 3 ) ) ) ) + 
 	 ( ( np.e )**( complex( 0,1 ) * xi ) * ( 1465/768 * np.pi + 
 	 ( complex( 0,185/1008 ) * nu + 
 	 ( complex( 0,53185/128 ) * np.log( 2 ) + 
 	 complex( 0,-15993/64 ) * np.log( 3 ) ) ) ) + 
 	 ( ( np.e )**( complex( 0,-1 ) * xi ) * ( 1547/768 * np.pi + 
 	 ( complex( 0,-13/3024 ) * nu + 
 	 ( complex( 0,-301531/384 ) * np.log( 2 ) + 
 	 ( complex( 0,25245/128 ) * np.log( 3 ) + 
 	 complex( 0,40625/192 ) * np.log( 5 ) ) ) ) ) + 
 	 ( ( np.e )**( complex( 0,-3 ) * xi ) * ( 119/512 * np.pi + 
 	 ( complex( 0,517483/768 ) * np.log( 2 ) + 
 	 ( complex( 0,14121/64 ) * np.log( 3 ) + 
 	 complex( 0,-334375/768 ) * np.log( 5 ) ) ) ) + 
 	 ( np.e )**( complex( 0,-5 ) * xi ) * ( 18151/1536 * np.pi + 
 	 ( complex( 0,-334321/1280 ) * np.log( 2 ) + 
 	 ( complex( 0,-3879/16 ) * np.log( 3 ) + 
 	 ( complex( 0,3125/64 ) * np.log( 5 ) + 
 	 complex( 0,823543/3840 ) * np.log( 7 ) ) ) ) ) ) ) ) ) ) + 
 	 ( e )**( 6 ) * ( 1145/576 * np.pi + 
 	 ( complex( 0,-55/504 ) * nu + 
 	 ( complex( 0,-45929/36 ) * np.log( 2 ) + 
 	 ( ( np.e )**( complex( 0,4 ) * xi ) * ( -427/240 * np.pi + 
 	 ( complex( 0,169/1512 ) * nu + 
 	 ( complex( 0,-1667/10 ) * np.log( 2 ) + 
 	 complex( 0,44289/320 ) * np.log( 3 ) ) ) ) + 
 	 ( ( np.e )**( complex( 0,2 ) * xi ) * ( 689/384 * np.pi + 
 	 ( complex( 0,503/4032 ) * nu + 
 	 ( complex( 0,30199/48 ) * np.log( 2 ) + 
 	 complex( 0,-49167/128 ) * np.log( 3 ) ) ) ) + 
 	 ( complex( 0,22329/64 ) * np.log( 3 ) + 
 	 ( ( np.e )**( complex( 0,6 ) * xi ) * ( 115751/5760 * np.pi + 
 	 ( complex( 0,-1495/4032 ) * nu + 
 	 ( complex( 0,1949/144 ) * np.log( 2 ) + 
 	 ( complex( 0,459/640 ) * np.log( 3 ) + 
 	 complex( 0,22/45 ) * np.log( 4 ) ) ) ) ) + 
 	 ( ( np.e )**( complex( 0,-2 ) * xi ) * ( 199/96 * np.pi + 
 	 ( complex( 0,-13/6048 ) * nu + 
 	 ( complex( 0,66655/48 ) * np.log( 2 ) + 
 	 ( complex( 0,18747/64 ) * np.log( 3 ) + 
 	 complex( 0,-303125/384 ) * np.log( 5 ) ) ) ) ) + 
 	 ( complex( 0,184375/576 ) * np.log( 5 ) + 
 	 ( ( np.e )**( complex( 0,-6 ) * xi ) * ( 2681/160 * np.pi + 
 	 ( complex( 0,829897/720 ) * np.log( 2 ) + 
 	 ( complex( 0,58563/640 ) * np.log( 3 ) + 
 	 ( complex( 0,3125/576 ) * np.log( 5 ) + 
 	 complex( 0,-823543/1920 ) * np.log( 7 ) ) ) ) ) + 
 	 ( np.e )**( complex( 0,-4 ) * xi ) * ( -32/15 * np.pi + 
 	 ( complex( 0,-28393/30 ) * np.log( 2 ) + 
 	 ( complex( 0,-538029/640 ) * np.log( 3 ) + 
 	 ( complex( 0,59375/128 ) * np.log( 5 ) + 
 	 complex( 0,823543/1920 ) * np.log( 7 ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) + 
 	 ( ( x )**( 5/2 ) * ( -107/21 * np.pi + 
 	 ( complex( 0,-24 ) * nu + 
 	 ( 34/21 * np.pi * nu + 
 	 ( ( e )**( 2 ) * ( complex( 0,-18 ) + 
 	 ( -103/21 * np.pi + 
 	 ( complex( 0,-83233/630 ) * nu + 
 	 ( 191/42 * np.pi * nu + 
 	 ( complex( 0,-1381/7 ) * np.log( 2 ) + 
 	 ( complex( 0,-15/7 ) * nu * np.log( 2 ) + 
 	 ( ( np.e )**( complex( 0,2 ) * xi ) * ( complex( 0,-9 ) + 
 	 ( 611/42 * np.pi + 
 	 ( complex( 0,-255197/1008 ) * nu + 
 	 ( 37/84 * np.pi * nu + 
 	 ( complex( 0,-865/216 ) * ( nu )**( 2 ) + 
 	 ( complex( 0,683/14 ) * np.log( 2 ) + 
 	 complex( 0,-33/14 ) * nu * np.log( 2 ) ) ) ) ) ) ) + 
 	 ( complex( 0,297/2 ) * np.log( 3 ) + 
 	 ( complex( 0,9/2 ) * nu * np.log( 3 ) + 
 	 ( np.e )**( complex( 0,-2 ) * xi ) * ( complex( 0,-21 ) + 
 	 ( -185/21 * np.pi + 
 	 ( complex( 0,-69703/2520 ) * nu + 
 	 ( 449/84 * np.pi * nu + 
 	 ( complex( 0,9997/42 ) * np.log( 2 ) + 
 	 ( complex( 0,893/42 ) * nu * np.log( 2 ) + 
 	 ( complex( 0,-297/2 ) * np.log( 3 ) + 
 	 complex( 0,-9/2 ) * nu * np.log( 3 ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) + 
 	 ( e * ( ( np.e )**( complex( 0,1 ) * xi ) * ( complex( 0,-9/2 ) + 
 	 ( 229/168 * np.pi + 
 	 ( complex( 0,-14579/140 ) * nu + 
 	 ( 61/42 * np.pi * nu + 
 	 ( complex( 0,473/28 ) * np.log( 2 ) + 
 	 complex( 0,-3/7 ) * nu * np.log( 2 ) ) ) ) ) ) + 
 	 ( np.e )**( complex( 0,-1 ) * xi ) * ( complex( 0,-27/2 ) + 
 	 ( -1081/168 * np.pi + 
 	 ( complex( 0,-1291/180 ) * nu + 
 	 ( 137/42 * np.pi * nu + 
 	 ( complex( 0,-27/4 ) * np.log( 2 ) + 
 	 ( complex( 0,-9 ) * nu * np.log( 2 ) + 
 	 ( complex( 0,27/4 ) * np.log( 3 ) + 
 	 complex( 0,9 ) * nu * np.log( 3 ) ) ) ) ) ) ) ) ) + 
 	 ( ( e )**( 4 ) * ( complex( 0,-63/2 ) + 
 	 ( -33/7 * np.pi + 
 	 ( complex( 0,-6292501/20160 ) * nu + 
 	 ( 157/21 * np.pi * nu + 
 	 ( complex( 0,-5951/2016 ) * ( nu )**( 2 ) + 
 	 ( complex( 0,275803/84 ) * np.log( 2 ) + 
 	 ( complex( 0,-7211/42 ) * nu * np.log( 2 ) + 
 	 ( ( np.e )**( complex( 0,4 ) * xi ) * ( complex( 0,-105/4 ) + 
 	 ( 36341/504 * np.pi + 
 	 ( complex( 0,-18532993/20160 ) * nu + 
 	 ( -4805/1008 * np.pi * nu + 
 	 ( complex( 0,-77779/6048 ) * ( nu )**( 2 ) + 
 	 ( complex( 0,81553/504 ) * np.log( 2 ) + 
 	 complex( 0,-1313/126 ) * nu * np.log( 2 ) ) ) ) ) ) ) + 
 	 ( complex( 0,-113265/56 ) * np.log( 3 ) + 
 	 ( complex( 0,855/7 ) * nu * np.log( 3 ) + 
 	 ( ( np.e )**( complex( 0,2 ) * xi ) * ( complex( 0,-75/2 ) + 
 	 ( -2003/336 * np.pi + 
 	 ( complex( 0,-38756573/30240 ) * nu + 
 	 ( 2705/336 * np.pi * nu + 
 	 ( complex( 0,-5431/3024 ) * ( nu )**( 2 ) + 
 	 ( complex( 0,-259871/252 ) * np.log( 2 ) + 
 	 ( complex( 0,15241/252 ) * nu * np.log( 2 ) + 
 	 ( complex( 0,12573/16 ) * np.log( 3 ) + 
 	 complex( 0,-669/16 ) * nu * np.log( 3 ) ) ) ) ) ) ) ) ) + 
 	 ( ( np.e )**( complex( 0,-2 ) * xi ) * ( complex( 0,-63/2 ) + 
 	 ( -313/72 * np.pi + 
 	 ( complex( 0,799829/5040 ) * nu + 
 	 ( 593/72 * np.pi * nu + 
 	 ( complex( 0,-1218479/252 ) * np.log( 2 ) + 
 	 ( complex( 0,49009/252 ) * nu * np.log( 2 ) + 
 	 ( complex( 0,69039/56 ) * np.log( 3 ) + 
 	 ( complex( 0,-2223/28 ) * nu * np.log( 3 ) + 
 	 ( complex( 0,1269875/1008 ) * np.log( 5 ) + 
 	 complex( 0,-15625/1008 ) * nu * np.log( 5 ) ) ) ) ) ) ) ) ) ) + 
 	 ( np.e )**( complex( 0,-4 ) * xi ) * ( complex( 0,-183/4 ) + 
 	 ( -16481/1008 * np.pi + 
 	 ( complex( 0,-3723263/30240 ) * nu + 
 	 ( 3007/252 * np.pi * nu + 
 	 ( complex( 0,864361/504 ) * np.log( 2 ) + 
 	 ( complex( 0,-4283/126 ) * nu * np.log( 2 ) + 
 	 ( complex( 0,85923/112 ) * np.log( 3 ) + 
 	 ( complex( 0,3135/112 ) * nu * np.log( 3 ) + 
 	 ( complex( 0,-1269875/1008 ) * np.log( 5 ) + 
 	 complex( 0,15625/1008 ) * nu * np.log( 5 ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) + 
 	 ( ( e )**( 3 ) * ( ( np.e )**( complex( 0,3 ) * xi ) * ( complex( 0,-255/16 ) + 
 	 ( 150523/4032 * np.pi + 
 	 ( complex( 0,-5123581/10080 ) * nu + 
 	 ( -395/252 * np.pi * nu + 
 	 ( complex( 0,-1504/189 ) * ( nu )**( 2 ) + 
 	 ( complex( 0,191159/2016 ) * np.log( 2 ) + 
 	 complex( 0,-673/126 ) * nu * np.log( 2 ) ) ) ) ) ) ) + 
 	 ( ( np.e )**( complex( 0,1 ) * xi ) * ( complex( 0,-447/16 ) + 
 	 ( -6325/1344 * np.pi + 
 	 ( complex( 0,-5623151/10080 ) * nu + 
 	 ( 1027/168 * np.pi * nu + 
 	 ( complex( 0,1504/189 ) * ( nu )**( 2 ) + 
 	 ( complex( 0,-353819/672 ) * np.log( 2 ) + 
 	 ( complex( 0,1699/84 ) * nu * np.log( 2 ) + 
 	 ( complex( 0,6345/16 ) * np.log( 3 ) + 
 	 complex( 0,-45/4 ) * nu * np.log( 3 ) ) ) ) ) ) ) ) ) + 
 	 ( ( np.e )**( complex( 0,-1 ) * xi ) * ( complex( 0,-429/16 ) + 
 	 ( -2045/448 * np.pi + 
 	 ( complex( 0,861379/10080 ) * nu + 
 	 ( 263/42 * np.pi * nu + 
 	 ( complex( 0,920711/672 ) * np.log( 2 ) + 
 	 ( complex( 0,-1349/42 ) * nu * np.log( 2 ) + 
 	 ( complex( 0,-188163/224 ) * np.log( 3 ) + 
 	 complex( 0,891/28 ) * nu * np.log( 3 ) ) ) ) ) ) ) ) + 
 	 ( np.e )**( complex( 0,-3 ) * xi ) * ( complex( 0,-501/16 ) + 
 	 ( -16183/1344 * np.pi + 
 	 ( complex( 0,-211109/3360 ) * nu + 
 	 ( 1369/168 * np.pi * nu + 
 	 ( complex( 0,-1963307/2016 ) * np.log( 2 ) + 
 	 ( complex( 0,-4933/252 ) * nu * np.log( 2 ) + 
 	 ( complex( 0,2727/16 ) * np.log( 3 ) + 
 	 ( complex( 0,-27/4 ) * nu * np.log( 3 ) + 
 	 ( complex( 0,613625/2016 ) * np.log( 5 ) + 
 	 complex( 0,3125/126 ) * nu * np.log( 5 ) ) ) ) ) ) ) ) ) ) ) ) ) + 
     ( ( e )**( 5 ) * ( ( np.e )**( complex( 0,5 ) * xi ) * ( complex( 0,-10599/256 ) + 
 	 ( 4383283/35840 * np.pi + 
 	 ( complex( 0,-3781718831/2419200 ) * nu + 
 	 ( -252673/26880 * np.pi * nu + 
 	 ( complex( 0,-175639/9072 ) * ( nu )**( 2 ) + 
 	 ( complex( 0,40412251/161280 ) * np.log( 2 ) + 
 	 ( complex( 0,-651527/40320 ) * nu * np.log( 2 ) + 
 	 ( complex( 0,158841/17920 ) * np.log( 3 ) + 
 	 complex( 0,-10377/4480 ) * nu * np.log( 3 ) ) ) ) ) ) ) ) ) + 
 	 ( ( np.e )**( complex( 0,3 ) * xi ) * ( complex( 0,-12489/256 ) + 
 	 ( -500431/64512 * np.pi + 
 	 ( complex( 0,-168070547/69120 ) * nu + 
 	 ( 167341/16128 * np.pi * nu + 
 	 ( complex( 0,-8587/1008 ) * ( nu )**( 2 ) + 
 	 ( complex( 0,-58139783/32256 ) * np.log( 2 ) + 
 	 ( complex( 0,1043201/8064 ) * nu * np.log( 2 ) + 
 	 ( complex( 0,1377 ) * np.log( 3 ) + 
 	 complex( 0,-1503/16 ) * nu * np.log( 3 ) ) ) ) ) ) ) ) ) + 
 	 ( ( np.e )**( complex( 0,1 ) * xi ) * ( complex( 0,-5147/128 ) + 
 	 ( -219055/32256 * np.pi + 
 	 ( complex( 0,-325429933/241920 ) * nu + 
 	 ( 74539/8064 * np.pi * nu + 
 	 ( complex( 0,85297/3024 ) * ( nu )**( 2 ) + 
 	 ( complex( 0,11217979/1792 ) * np.log( 2 ) + 
 	 ( complex( 0,-569245/1344 ) * nu * np.log( 2 ) + 
 	 ( complex( 0,-3471381/896 ) * np.log( 3 ) + 
 	 complex( 0,15945/56 ) * nu * np.log( 3 ) ) ) ) ) ) ) ) ) + 
 	 ( ( np.e )**( complex( 0,-1 ) * xi ) * ( complex( 0,-5073/128 ) + 
 	 ( -134497/32256 * np.pi + 
 	 ( complex( 0,64447877/241920 ) * nu + 
 	 ( 72731/8064 * np.pi * nu + 
 	 ( complex( 0,-2969/9072 ) * ( nu )**( 2 ) + 
 	 ( complex( 0,-183403471/16128 ) * np.log( 2 ) + 
 	 ( complex( 0,2846033/4032 ) * nu * np.log( 2 ) + 
 	 ( complex( 0,5609547/1792 ) * np.log( 3 ) + 
 	 ( complex( 0,-113625/448 ) * nu * np.log( 3 ) + 
 	 ( complex( 0,22545875/8064 ) * np.log( 5 ) + 
 	 complex( 0,-115625/1008 ) * nu * np.log( 5 ) ) ) ) ) ) ) ) ) ) ) + 
     ( ( np.e )**( complex( 0,-3 ) * xi ) * ( complex( 0,-9075/256 ) + 
 	 ( -73769/21504 * np.pi + 
 	 ( complex( 0,3155465/10752 ) * nu + 
 	 ( 55913/5376 * np.pi * nu + 
 	 ( complex( 0,42181949/4608 ) * np.log( 2 ) + 
 	 ( complex( 0,-584993/1152 ) * nu * np.log( 2 ) + 
 	 ( complex( 0,2026053/896 ) * np.log( 3 ) + 
 	 ( complex( 0,-1503/56 ) * nu * np.log( 3 ) + 
 	 ( complex( 0,-176228375/32256 ) * np.log( 5 ) + 
 	 complex( 0,2084375/8064 ) * nu * np.log( 5 ) ) ) ) ) ) ) ) ) ) + 
 	 ( np.e )**( complex( 0,-5 ) * xi ) * ( complex( 0,-16901/256 ) + 
 	 ( -203251/9216 * np.pi + 
 	 ( complex( 0,-77092807/345600 ) * nu + 
 	 ( 39193/2304 * np.pi * nu + 
 	 ( complex( 0,-143300477/53760 ) * np.log( 2 ) + 
 	 ( complex( 0,325579/13440 ) * nu * np.log( 2 ) + 
 	 ( complex( 0,-1226223/448 ) * np.log( 3 ) + 
 	 ( complex( 0,1761/28 ) * nu * np.log( 3 ) + 
 	 ( complex( 0,919125/896 ) * np.log( 5 ) + 
 	 ( complex( 0,-15625/336 ) * nu * np.log( 5 ) + 
 	 ( complex( 0,38035613/23040 ) * np.log( 7 ) + 
 	 complex( 0,117649/5760 ) * nu * np.log( 7 ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) + 
 	 ( e )**( 6 ) * ( complex( 0,-177/4 ) + 
 	 ( -113761/24192 * np.pi + 
 	 ( complex( 0,-3714541/6720 ) * nu + 
 	 ( 253265/24192 * np.pi * nu + 
 	 ( complex( 0,-5851/432 ) * ( nu )**( 2 ) + 
 	 ( complex( 0,-32693669/1512 ) * np.log( 2 ) + 
 	 ( complex( 0,1218593/756 ) * nu * np.log( 2 ) + 
 	 ( complex( 0,5561757/896 ) * np.log( 3 ) + 
 	 ( complex( 0,-501453/896 ) * nu * np.log( 3 ) + 
 	 ( ( np.e )**( complex( 0,4 ) * xi ) * ( complex( 0,-611/10 ) + 
 	 ( -3783/320 * np.pi + 
 	 ( complex( 0,-2415997141/504000 ) * nu + 
 	 ( 37729/2880 * np.pi * nu + 
 	 ( complex( 0,-131717/9072 ) * ( nu )**( 2 ) + 
 	 ( complex( 0,-3710689/1260 ) * np.log( 2 ) + 
 	 ( complex( 0,395911/1680 ) * nu * np.log( 2 ) + 
 	 ( complex( 0,10026567/4480 ) * np.log( 3 ) + 
 	 complex( 0,-775041/4480 ) * nu * np.log( 3 ) ) ) ) ) ) ) ) ) + 
 	 ( ( np.e )**( complex( 0,2 ) * xi ) * ( complex( 0,-739/16 ) + 
 	 ( -214073/16128 * np.pi + 
 	 ( complex( 0,-21480407/7560 ) * nu + 
 	 ( 186323/16128 * np.pi * nu + 
 	 ( complex( 0,73085/6048 ) * ( nu )**( 2 ) + 
 	 ( complex( 0,21863689/2016 ) * np.log( 2 ) + 
 	 ( complex( 0,-1709585/2016 ) * nu * np.log( 2 ) + 
 	 ( complex( 0,-12073347/1792 ) * np.log( 3 ) + 
 	 complex( 0,999261/1792 ) * nu * np.log( 3 ) ) ) ) ) ) ) ) ) + 
 	 ( ( np.e )**( complex( 0,6 ) * xi ) * ( complex( 0,-5079/80 ) + 
 	 ( 46553861/241920 * np.pi + 
 	 ( complex( 0,-686906611/403200 ) * nu + 
 	 ( -3802027/241920 * np.pi * nu + 
 	 ( complex( 0,-681661/24192 ) * ( nu )**( 2 ) + 
 	 ( complex( 0,11023927/30240 ) * np.log( 2 ) + 
 	 ( complex( 0,-133255/6048 ) * nu * np.log( 2 ) + 
 	 ( complex( 0,190971/8960 ) * np.log( 3 ) + 
 	 ( complex( 0,-44721/8960 ) * nu * np.log( 3 ) + 
 	 ( complex( 0,28387/1890 ) * np.log( 4 ) + 
 	 complex( 0,-3622/945 ) * nu * np.log( 4 ) ) ) ) ) ) ) ) ) ) ) + 
 	 ( complex( 0,124010125/24192 ) * np.log( 5 ) + 
 	 ( complex( 0,-7090625/24192 ) * nu * np.log( 5 ) + 
 	 ( ( np.e )**( complex( 0,-2 ) * xi ) * ( complex( 0,-709/16 ) + 
 	 ( -2429/576 * np.pi + 
 	 ( complex( 0,184865/432 ) * nu + 
 	 ( 6067/576 * np.pi * nu + 
 	 ( complex( 0,-5899/36288 ) * ( nu )**( 2 ) + 
 	 ( complex( 0,2149273/96 ) * np.log( 2 ) + 
 	 ( complex( 0,-474119/288 ) * nu * np.log( 2 ) + 
 	 ( complex( 0,3725739/896 ) * np.log( 3 ) + 
 	 ( complex( 0,-84471/896 ) * nu * np.log( 3 ) + 
 	 ( complex( 0,-9555875/768 ) * np.log( 5 ) + 
 	 complex( 0,1834375/2304 ) * nu * np.log( 5 ) ) ) ) ) ) ) ) ) ) ) +
 	 ( ( np.e )**( complex( 0,-4 ) * xi ) * ( complex( 0,-739/20 ) + 
 	 ( -1123/840 * np.pi + 
 	 ( complex( 0,21076961/37800 ) * nu + 
 	 ( 128299/10080 * np.pi * nu + 
 	 ( complex( 0,-17518111/1260 ) * np.log( 2 ) + 
 	 ( complex( 0,3836549/5040 ) * nu * np.log( 2 ) + 
 	 ( complex( 0,-102979593/8960 ) * np.log( 3 ) + 
 	 ( complex( 0,5374791/8960 ) * nu * np.log( 3 ) + 
 	 ( complex( 0,117328375/16128 ) * np.log( 5 ) + 
 	 ( complex( 0,-890625/1792 ) * nu * np.log( 5 ) + 
 	 ( complex( 0,62741903/11520 ) * np.log( 7 ) + 
 	 complex( 0,-2000033/11520 ) * nu * np.log( 7 ) ) ) ) ) ) ) ) ) ) ) ) + 
 	 ( np.e )**( complex( 0,-6 ) * xi ) * ( complex( 0,-7557/80 ) + 
 	 ( -198661/6720 * np.pi + 
 	 ( complex( 0,-451473527/1008000 ) * nu + 
 	 ( 159743/6720 * np.pi * nu + 
 	 ( complex( 0,51609541/4320 ) * np.log( 2 ) + 
 	 ( complex( 0,-879077/4320 ) * nu * np.log( 2 ) + 
 	 ( complex( 0,3700323/1792 ) * np.log( 3 ) + 
 	 ( complex( 0,-900477/8960 ) * nu * np.log( 3 ) + 
 	 ( complex( 0,1007375/24192 ) * np.log( 5 ) + 
 	 ( complex( 0,-146875/24192 ) * nu * np.log( 5 ) + 
 	 ( complex( 0,-62741903/11520 ) * np.log( 7 ) + 
 	 complex( 0,2000033/11520 ) * nu * np.log( 7 ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) + 
 	 ( x )**( 3 ) * ( 27027409/646800 + 
 	 ( complex( 0,428/105 ) * np.pi + 
 	 ( 2/3 * ( np.pi )**( 2 ) + 
 	 ( -856/105 * gamma + 
 	 ( -278185/33264 * nu + 
 	 ( 41/96 * ( np.pi )**( 2 ) * nu + 
 	 ( -20261/2772 * ( nu )**( 2 ) + 
 	 ( 114635/99792 * ( nu )**( 3 ) + 
 	 ( -1712/105 * np.log( 2 ) + 
 	 ( ( e )**( 5 ) * ( ( np.e )**( complex( 0,-5 ) * xi ) * ( 1023246543653/638668800 + 
 	 ( complex( 0,3521477/23040 ) * np.pi + 
 	 ( 230377/9216 * ( np.pi )**( 2 ) + 
 	 ( -3521477/11520 * gamma + 
 	 ( 1122957263/1064448 * nu + 
 	 ( -1691701/36864 * ( np.pi )**( 2 ) * nu + 
 	 ( -106366543/1596672 * ( nu )**( 2 ) + 
 	 ( -2622530395/153280512 * ( nu )**( 3 ) + 
 	 ( 466953457/403200 * np.log( 2 ) + 
 	 ( complex( 0,-2222607/1280 ) * np.pi * np.log( 2 ) + 
 	 ( -6749741/3840 * ( np.log( 2 ) )**( 2 ) + 
 	 ( 831069/560 * np.log( 3 ) + 
 	 ( complex( 0,-23301/16 ) * np.pi * np.log( 3 ) + 
 	 ( 27/8 * np.log( 2 ) * np.log( 3 ) + 
 	 ( 23301/16 * ( np.log( 3 ) )**( 2 ) + 
 	 ( -334375/1344 * np.log( 5 ) + 
 	 ( complex( 0,15625/64 ) * np.pi * np.log( 5 ) + 
 	 ( 15625/32 * np.log( 2 ) * np.log( 5 ) + 
 	 ( -15625/64 * ( np.log( 5 ) )**( 2 ) + 
 	 ( -88119101/57600 * np.log( 7 ) + 
 	 ( complex( 0,5764801/3840 ) * np.pi * np.log( 7 ) + 
 	 ( 5764801/1920 * np.log( 2 ) * np.log( 7 ) + 
 	 ( -5764801/3840 * ( np.log( 7 ) )**( 2 ) + 
 	 -3521477/23040 * np.log( x ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) + 
 	 ( ( np.e )**( complex( 0,-1 ) * xi ) * ( 5894091113/9676800 + 
 	 ( complex( 0,514349/11520 ) * np.pi + 
 	 ( 33649/4608 * ( np.pi )**( 2 ) + 
 	 ( -514349/5760 * gamma + 
 	 ( -583640093/1451520 * nu + 
 	 ( complex( 0,-29/1512 ) * np.pi * nu + 
 	 ( 451451/73728 * ( np.pi )**( 2 ) * nu + 
 	 ( 2434895/145152 * ( nu )**( 2 ) + 
 	 ( -35629553/6967296 * ( nu )**( 3 ) + 
 	 ( 2719833/896 * np.log( 2 ) + 
 	 ( complex( 0,-1211153/384 ) * np.pi * np.log( 2 ) + 
 	 ( -56081/384 * ( np.log( 2 ) )**( 2 ) + 
 	 ( -540243/896 * np.log( 3 ) + 
 	 ( complex( 0,75735/128 ) * np.pi * np.log( 3 ) + 
 	 ( 75735/64 * np.log( 2 ) * np.log( 3 ) + 
 	 ( -75735/128 * ( np.log( 3 ) )**( 2 ) + 
 	 ( -4346875/4032 * np.log( 5 ) + 
 	 ( complex( 0,203125/192 ) * np.pi * np.log( 5 ) + 
 	 ( 203125/96 * np.log( 2 ) * np.log( 5 ) + 
 	 ( -203125/192 * ( np.log( 5 ) )**( 2 ) + 
 	 -514349/11520 * np.log( x ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) + 
 	 ( ( np.e )**( complex( 0,1 ) * xi ) * ( 999566924003/2235340800 + 
 	 ( complex( 0,3333371/80640 ) * np.pi + 
 	 ( 31069/4608 * ( np.pi )**( 2 ) + 
 	 ( -3324383/40320 * gamma + 
 	 ( -178945301/506880 * nu + 
 	 ( complex( 0,3121/1512 ) * np.pi * nu + 
 	 ( 475313/73728 * ( np.pi )**( 2 ) * nu + 
 	 ( 206198197/6386688 * ( nu )**( 2 ) + 
 	 ( -262791967/76640256 * ( nu )**( 3 ) + 
 	 ( -63658687/40320 * np.log( 2 ) + 
 	 ( complex( 0,177573/128 ) * np.pi * np.log( 2 ) + 
 	 ( 49475/384 * ( np.log( 2 ) )**( 2 ) + 
 	 ( 1711251/2240 * np.log( 3 ) + 
 	 ( complex( 0,-47979/64 ) * np.pi * np.log( 3 ) + 
 	 ( -47979/32 * np.log( 2 ) * np.log( 3 ) + 
 	 ( 47979/64 * ( np.log( 3 ) )**( 2 ) + 
 	 -3324383/80640 * np.log( x ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) + 
 	 ( ( np.e )**( complex( 0,3 ) * xi ) * ( 374231745511/496742400 + 
 	 ( complex( 0,6233713/161280 ) * np.pi + 
 	 ( 6733/1024 * ( np.pi )**( 2 ) + 
 	 ( -720431/8960 * gamma + 
 	 ( -1730857169/2661120 * nu + 
 	 ( complex( 0,-1381/1512 ) * np.pi * nu + 
 	 ( 166337/24576 * ( np.pi )**( 2 ) * nu + 
 	 ( 10069373/266112 * ( nu )**( 2 ) + 
 	 ( -452818469/51093504 * ( nu )**( 3 ) + 
 	 ( 12543289/80640 * np.log( 2 ) + 
 	 ( complex( 0,-237619/768 ) * np.pi * np.log( 2 ) + 
 	 ( -238421/768 * ( np.log( 2 ) )**( 2 ) + 
 	 ( -66447/224 * np.log( 3 ) + 
 	 ( complex( 0,9315/32 ) * np.pi * np.log( 3 ) + 
 	 ( 9315/16 * np.log( 2 ) * np.log( 3 ) + 
 	 ( -9315/32 * ( np.log( 3 ) )**( 2 ) + 
 	 -720431/17920 * np.log( x ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) + 
 	 ( ( np.e )**( complex( 0,-3 ) * xi ) * ( 15182758817/30412800 + 
 	 ( complex( 0,299279/7680 ) * np.pi + 
 	 ( 19579/3072 * ( np.pi )**( 2 ) + 
 	 ( -299279/3840 * gamma + 
 	 ( -5235557539/21288960 * nu + 
 	 ( 2829/512 * ( np.pi )**( 2 ) * nu + 
 	 ( -115834225/4257792 * ( nu )**( 2 ) + 
 	 ( -16136275/1892352 * ( nu )**( 3 ) + 
 	 ( -53595551/16128 * np.log( 2 ) + 
 	 ( complex( 0,2386991/768 ) * np.pi * np.log( 2 ) + 
 	 ( 1059887/768 * ( np.log( 2 ) )**( 2 ) + 
 	 ( -453573/320 * np.log( 3 ) + 
 	 ( complex( 0,89019/64 ) * np.pi * np.log( 3 ) + 
 	 ( -4293/32 * np.log( 2 ) * np.log( 3 ) + 
 	 ( -89019/64 * ( np.log( 3 ) )**( 2 ) + 
 	 ( 35778125/16128 * np.log( 5 ) + 
 	 ( complex( 0,-1671875/768 ) * np.pi * np.log( 5 ) + 
 	 ( -1671875/384 * np.log( 2 ) * np.log( 5 ) + 
 	 ( 1671875/768 * ( np.log( 5 ) )**( 2 ) + 
 	 -299279/7680 * np.log( x ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) + 
 	 ( np.e )**( complex( 0,5 ) * xi ) * ( -745940868881/496742400 + 
 	 ( complex( 0,3387941/89600 ) * np.pi + 
 	 ( 53077/9216 * ( np.pi )**( 2 ) + 
 	 ( -5679239/80640 * gamma + 
 	 ( 14000998499/12773376 * nu + 
 	 ( complex( 0,-1711/1512 ) * np.pi * nu + 
 	 ( -1192157/73728 * ( np.pi )**( 2 ) * nu + 
 	 ( -4649948663/12773376 * ( nu )**( 2 ) + 
 	 ( 1078514177/153280512 * ( nu )**( 3 ) + 
 	 ( -20619649/134400 * np.log( 2 ) + 
 	 ( complex( 0,35449/3840 ) * np.pi * np.log( 2 ) + 
 	 ( 47351/3840 * ( np.log( 2 ) )**( 2 ) + 
 	 ( 49113/44800 * np.log( 3 ) + 
 	 ( complex( 0,1377/1280 ) * np.pi * np.log( 3 ) + 
 	 ( -1377/640 * np.log( 2 ) * np.log( 3 ) + 
 	 ( 1377/1280 * ( np.log( 3 ) )**( 2 ) + 
 	 -5679239/161280 * np.log( x ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) + 
 	 ( ( e )**( 3 ) * ( ( np.e )**( complex( 0,-3 ) * xi ) * ( 1490703301/2822400 + 
 	 ( complex( 0,172163/3360 ) * np.pi + 
 	 ( 1609/192 * ( np.pi )**( 2 ) + 
 	 ( -172163/1680 * gamma + 
 	 ( 1685011/6720 * nu + 
 	 ( -23247/2048 * ( np.pi )**( 2 ) * nu + 
 	 ( -83557/3024 * ( nu )**( 2 ) + 
 	 ( -181117/32256 * ( nu )**( 3 ) + 
 	 ( 2109719/5040 * np.log( 2 ) + 
 	 ( complex( 0,-29371/48 ) * np.pi * np.log( 2 ) + 
 	 ( -4795/48 * ( np.log( 2 ) )**( 2 ) + 
 	 ( -8667/280 * np.log( 3 ) + 
 	 ( complex( 0,243/8 ) * np.pi * np.log( 3 ) + 
 	 ( 243/4 * np.log( 2 ) * np.log( 3 ) + 
 	 ( -243/8 * ( np.log( 3 ) )**( 2 ) + 
 	 ( -334375/1008 * np.log( 5 ) + 
 	 ( complex( 0,15625/48 ) * np.pi * np.log( 5 ) + 
 	 ( 15625/24 * np.log( 2 ) * np.log( 5 ) + 
 	 ( -15625/48 * ( np.log( 5 ) )**( 2 ) + 
 	 -172163/3360 * np.log( x ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) + 
 	 ( ( np.e )**( complex( 0,-1 ) * xi ) * ( 29596324927/93139200 + 
 	 ( complex( 0,86777/3360 ) * np.pi + 
 	 ( 811/192 * ( np.pi )**( 2 ) + 
 	 ( -86777/1680 * gamma + 
 	 ( -31314713/190080 * nu + 
 	 ( 17425/6144 * ( np.pi )**( 2 ) * nu + 
 	 ( -1917163/266112 * ( nu )**( 2 ) + 
 	 ( -7265101/3193344 * ( nu )**( 3 ) + 
 	 ( -351923/560 * np.log( 2 ) + 
 	 ( complex( 0,8245/16 ) * np.pi * np.log( 2 ) + 
 	 ( 53/16 * ( np.log( 2 ) )**( 2 ) + 
 	 ( 147339/560 * np.log( 3 ) + 
 	 ( complex( 0,-4131/16 ) * np.pi * np.log( 3 ) + 
 	 ( -4131/8 * np.log( 2 ) * np.log( 3 ) + 
 	 ( 4131/16 * ( np.log( 3 ) )**( 2 ) + 
 	 -86777/3360 * np.log( x ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) + 
 	 ( ( np.e )**( complex( 0,1 ) * xi ) * ( 7012075729/31046400 + 
 	 ( complex( 0,25359/1120 ) * np.pi + 
 	 ( 237/64 * ( np.pi )**( 2 ) + 
 	 ( -25359/560 * gamma + 
 	 ( -13220077/73920 * nu + 
 	 ( complex( 0,29/63 ) * np.pi * nu + 
 	 ( 7011/2048 * ( np.pi )**( 2 ) * nu + 
 	 ( -9795/2464 * ( nu )**( 2 ) + 
 	 ( -1396459/1064448 * ( nu )**( 3 ) + 
 	 ( 84851/1680 * np.log( 2 ) + 
 	 ( complex( 0,-2215/16 ) * np.pi * np.log( 2 ) + 
 	 ( -2215/16 * ( np.log( 2 ) )**( 2 ) + 
 	 ( -37557/280 * np.log( 3 ) + 
 	 ( complex( 0,1053/8 ) * np.pi * np.log( 3 ) + 
 	 ( 1053/4 * np.log( 2 ) * np.log( 3 ) + 
 	 ( -1053/8 * ( np.log( 3 ) )**( 2 ) + 
 	 -25359/1120 * np.log( x ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) + 
 	 ( np.e )**( complex( 0,3 ) * xi ) * ( -19357323527/93139200 + 
 	 ( complex( 0,177727/10080 ) * np.pi + 
 	 ( 183/64 * ( np.pi )**( 2 ) + 
 	 ( -19581/560 * gamma + 
 	 ( 36062333/443520 * nu + 
 	 ( complex( 0,-29/63 ) * np.pi * nu + 
 	 ( -16933/6144 * ( np.pi )**( 2 ) * nu + 
 	 ( -19613089/266112 * ( nu )**( 2 ) + 
 	 ( 2754901/3193344 * ( nu )**( 3 ) + 
 	 ( -10807/144 * np.log( 2 ) + 
 	 ( complex( 0,227/48 ) * np.pi * np.log( 2 ) + 
 	 ( 241/48 * ( np.log( 2 ) )**( 2 ) + 
 	 -19581/1120 * np.log( x ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) + 
 	 ( e * ( ( np.e )**( complex( 0,-1 ) * xi ) * ( 219775769/1663200 + 
 	 ( complex( 0,749/60 ) * np.pi + 
 	 ( 49/24 * ( np.pi )**( 2 ) + 
 	 ( -749/30 * gamma + 
 	 ( -121717/20790 * nu + 
 	 ( -41/192 * ( np.pi )**( 2 ) * nu + 
 	 ( -86531/8316 * ( nu )**( 2 ) + 
 	 ( -33331/399168 * ( nu )**( 3 ) + 
 	 ( -1819/210 * np.log( 2 ) + 
 	 ( complex( 0,-81/2 ) * np.pi * np.log( 2 ) + 
 	 ( -81/2 * ( np.log( 2 ) )**( 2 ) + 
 	 ( -2889/70 * np.log( 3 ) + 
 	 ( complex( 0,81/2 ) * np.pi * np.log( 3 ) + 
 	 ( 81 * np.log( 2 ) * np.log( 3 ) + 
 	 ( -81/2 * ( np.log( 3 ) )**( 2 ) + 
 	 -749/60 * np.log( x ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) + 
 	 ( np.e )**( complex( 0,1 ) * xi ) * ( 55608313/1058400 + 
 	 ( complex( 0,3103/420 ) * np.pi + 
 	 ( 29/24 * ( np.pi )**( 2 ) + 
 	 ( -3103/210 * gamma + 
 	 ( -199855/3024 * nu + 
 	 ( 41/48 * ( np.pi )**( 2 ) * nu + 
 	 ( -9967/1008 * ( nu )**( 2 ) + 
 	 ( 35579/36288 * ( nu )**( 3 ) + 
 	 ( -6527/210 * np.log( 2 ) + 
 	 ( complex( 0,3/2 ) * np.pi * np.log( 2 ) + 
 	 ( 3/2 * ( np.log( 2 ) )**( 2 ) + 
 	 -3103/420 * np.log( x ) ) ) ) ) ) ) ) ) ) ) ) ) + 
 	 ( ( e )**( 6 ) * ( 1523147009/2661120 + 
 	 ( complex( 0,462347/8640 ) * np.pi + 
 	 ( 35/4 * ( np.pi )**( 2 ) + 
 	 ( -107 * gamma + 
 	 ( -112895995/532224 * nu + 
 	 ( complex( 0,-641/672 ) * np.pi * nu + 
 	 ( 2665/768 * ( np.pi )**( 2 ) * nu + 
 	 ( 4185551/133056 * ( nu )**( 2 ) + 
 	 ( -9383735/1596672 * ( nu )**( 3 ) + 
 	 ( 1337179/270 * np.log( 2 ) + 
 	 ( complex( 0,-162239/32 ) * np.pi * np.log( 2 ) + 
 	 ( -4091/18 * ( np.log( 2 ) )**( 2 ) + 
 	 ( -2389203/2240 * np.log( 3 ) + 
 	 ( complex( 0,66987/64 ) * np.pi * np.log( 3 ) + 
 	 ( 66987/32 * np.log( 2 ) * np.log( 3 ) + 
 	 ( -66987/64 * ( np.log( 3 ) )**( 2 ) + 
 	 ( -19728125/12096 * np.log( 5 ) + 
 	 ( complex( 0,921875/576 ) * np.pi * np.log( 5 ) + 
 	 ( 921875/288 * np.log( 2 ) * np.log( 5 ) + 
 	 ( -921875/576 * ( np.log( 5 ) )**( 2 ) + 
 	 ( ( np.e )**( complex( 0,-6 ) * xi ) * ( 137078526587/51744000 + 
 	 ( complex( 0,4200713/16800 ) * np.pi + 
 	 ( 39259/960 * ( np.pi )**( 2 ) + 
 	 ( -4200713/8400 * gamma + 
 	 ( 50042307101/26611200 * nu + 
 	 ( -3329733/40960 * ( np.pi )**( 2 ) * nu + 
 	 ( -26363437/266112 * ( nu )**( 2 ) + 
 	 ( -133307197/5322240 * ( nu )**( 3 ) + 
 	 ( -150789857/15120 * np.log( 2 ) + 
 	 ( complex( 0,6339593/720 ) * np.pi * np.log( 2 ) + 
 	 ( -6250999/720 * ( np.log( 2 ) )**( 2 ) + 
 	 ( -12506481/22400 * np.log( 3 ) + 
 	 ( complex( 0,350649/640 ) * np.pi * np.log( 3 ) + 
 	 ( 729/320 * np.log( 2 ) * np.log( 3 ) + 
 	 ( -350649/640 * ( np.log( 3 ) )**( 2 ) + 
 	 ( -334375/12096 * np.log( 5 ) + 
 	 ( complex( 0,15625/576 ) * np.pi * np.log( 5 ) + 
 	 ( 15625/288 * np.log( 2 ) * np.log( 5 ) + 
 	 ( -15625/576 * ( np.log( 5 ) )**( 2 ) + 
 	 ( 88119101/28800 * np.log( 7 ) + 
 	 ( complex( 0,-5764801/1920 ) * np.pi * np.log( 7 ) + 
 	 ( -5764801/960 * np.log( 2 ) * np.log( 7 ) + 
 	 ( 5764801/1920 * ( np.log( 7 ) )**( 2 ) + 
 	 -4200713/16800 * np.log( x ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) + 
 	 ( ( np.e )**( complex( 0,-2 ) * xi ) * ( 270820171/352800 + 
 	 ( complex( 0,64093/1120 ) * np.pi + 
 	 ( 599/64 * ( np.pi )**( 2 ) + 
 	 ( -64093/560 * gamma + 
 	 ( -187689179/483840 * nu + 
 	 ( complex( 0,-29/3024 ) * np.pi * nu + 
 	 ( 37351/8192 * ( np.pi )**( 2 ) * nu + 
 	 ( 296771/48384 * ( nu )**( 2 ) + 
 	 ( -290839/41472 * ( nu )**( 3 ) + 
 	 ( -33213121/5040 * np.log( 2 ) + 
 	 ( complex( 0,299621/48 ) * np.pi * np.log( 2 ) + 
 	 ( 36215/16 * ( np.log( 2 ) )**( 2 ) + 
 	 ( -4710033/2240 * np.log( 3 ) + 
 	 ( complex( 0,132057/64 ) * np.pi * np.log( 3 ) + 
 	 ( -19575/32 * np.log( 2 ) * np.log( 3 ) + 
 	 ( -132057/64 * ( np.log( 3 ) )**( 2 ) + 
 	 ( 32434375/8064 * np.log( 5 ) + 
 	 ( complex( 0,-1515625/384 ) * np.pi * np.log( 5 ) + 
 	 ( -1515625/192 * np.log( 2 ) * np.log( 5 ) + 
 	 ( 1515625/384 * ( np.log( 5 ) )**( 2 ) + 
 	 -64093/1120 * np.log( x ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) + 
 	 ( ( np.e )**( complex( 0,4 ) * xi ) * ( 90292317019/58212000 + 
 	 ( complex( 0,4778513/100800 ) * np.pi + 
 	 ( 1027/120 * ( np.pi )**( 2 ) + 
 	 ( -109889/1050 * gamma + 
 	 ( -2491605689/1900800 * nu + 
 	 ( complex( 0,-9223/6048 ) * np.pi * nu + 
 	 ( 361661/30720 * ( np.pi )**( 2 ) * nu + 
 	 ( 85326847/532224 * ( nu )**( 2 ) + 
 	 ( -169240997/7983360 * ( nu )**( 3 ) + 
 	 ( 391834/1575 * np.log( 2 ) + 
 	 ( complex( 0,-70961/160 ) * np.pi * np.log( 2 ) + 
 	 ( -6743/15 * ( np.log( 2 ) )**( 2 ) + 
 	 ( -1197009/2800 * np.log( 3 ) + 
 	 ( complex( 0,132867/320 ) * np.pi * np.log( 3 ) + 
 	 ( 33561/40 * np.log( 2 ) * np.log( 3 ) + 
 	 ( -33561/80 * ( np.log( 3 ) )**( 2 ) + 
 	 -109889/2100 * np.log( x ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) + 
 	 ( ( np.e )**( complex( 0,2 ) * xi ) * ( 308586650569/558835200 + 
 	 ( complex( 0,2041453/40320 ) * np.pi + 
 	 ( 4721/576 * ( np.pi )**( 2 ) + 
 	 ( -505147/5040 * gamma + 
 	 ( -3838426769/7983360 * nu + 
 	 ( complex( 0,13897/24192 ) * np.pi * nu + 
 	 ( 541241/73728 * ( np.pi )**( 2 ) * nu + 
 	 ( 49824815/1596672 * ( nu )**( 2 ) + 
 	 ( -1309163/342144 * ( nu )**( 3 ) + 
 	 ( -11732443/5040 * np.log( 2 ) + 
 	 ( complex( 0,400777/192 ) * np.pi * np.log( 2 ) + 
 	 ( 11887/48 * ( np.log( 2 ) )**( 2 ) + 
 	 ( 5260869/4480 * np.log( 3 ) + 
 	 ( complex( 0,-147501/128 ) * np.pi * np.log( 3 ) + 
 	 ( -147501/64 * np.log( 2 ) * np.log( 3 ) + 
 	 ( 147501/128 * ( np.log( 3 ) )**( 2 ) + 
 	 -505147/10080 * np.log( x ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) + 
 	 ( ( np.e )**( complex( 0,6 ) * xi ) * ( -2826624953189/931392000 + 
 	 ( complex( 0,33207557/604800 ) * np.pi + 
 	 ( 7589/960 * ( np.pi )**( 2 ) + 
 	 ( -812023/8400 * gamma + 
 	 ( 6266140691/2661120 * nu + 
 	 ( complex( 0,-3335/2016 ) * np.pi * nu + 
 	 ( -3735961/122880 * ( np.pi )**( 2 ) * nu + 
 	 ( -474187663/665280 * ( nu )**( 2 ) + 
 	 ( 147111197/7983360 * ( nu )**( 3 ) + 
 	 ( -3235787/15120 * np.log( 2 ) + 
 	 ( complex( 0,6125/576 ) * np.pi * np.log( 2 ) + 
 	 ( 14603/720 * ( np.log( 2 ) )**( 2 ) + 
 	 ( 49113/22400 * np.log( 3 ) + 
 	 ( complex( 0,1377/640 ) * np.pi * np.log( 3 ) + 
 	 ( -1377/320 * np.log( 2 ) * np.log( 3 ) + 
 	 ( 1377/640 * ( np.log( 3 ) )**( 2 ) + 
 	 ( 9416/4725 * np.log( 4 ) + 
 	 ( complex( 0,88/45 ) * np.pi * np.log( 4 ) + 
 	 ( -176/45 * np.log( 2 ) * np.log( 4 ) + 
 	 ( 88/45 * ( np.log( 4 ) )**( 2 ) + 
 	 -812023/16800 * np.log( x ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) + 
 	 ( ( np.e )**( complex( 0,-4 ) * xi ) * ( 623997833051/1397088000 + 
 	 ( complex( 0,192707/6300 ) * np.pi + 
 	 ( 1801/360 * ( np.pi )**( 2 ) + 
 	 ( -192707/3150 * gamma + 
 	 ( -1886302361/3991680 * nu + 
 	 ( 1384037/92160 * ( np.pi )**( 2 ) * nu + 
 	 ( -316527109/7983360 * ( nu )**( 2 ) + 
 	 ( -11054693/748440 * ( nu )**( 3 ) + 
 	 ( 17394241/3150 * np.log( 2 ) + 
 	 ( complex( 0,-33233/6 ) * np.pi * np.log( 2 ) + 
 	 ( -30545/6 * ( np.log( 2 ) )**( 2 ) + 
 	 ( 114979311/22400 * np.log( 3 ) + 
 	 ( complex( 0,-3223719/640 ) * np.pi * np.log( 3 ) + 
 	 ( -891/64 * np.log( 2 ) * np.log( 3 ) + 
 	 ( 3223719/640 * ( np.log( 3 ) )**( 2 ) + 
 	 ( -6353125/2688 * np.log( 5 ) + 
 	 ( complex( 0,296875/128 ) * np.pi * np.log( 5 ) + 
 	 ( 296875/64 * np.log( 2 ) * np.log( 5 ) + 
 	 ( -296875/128 * ( np.log( 5 ) )**( 2 ) + 
 	 ( -88119101/28800 * np.log( 7 ) + 
 	 ( complex( 0,5764801/1920 ) * np.pi * np.log( 7 ) + 
 	 ( 5764801/960 * np.log( 2 ) * np.log( 7 ) + 
 	 ( -5764801/1920 * ( np.log( 7 ) )**( 2 ) + 
 	 -192707/6300 * np.log( x ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) + 
 	 -107/2 * np.log( x ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) + 
 	 ( ( e )**( 4 ) * ( 3006246811/9313920 + 
 	 ( complex( 0,1391/42 ) * np.pi + 
 	 ( 65/12 * ( np.pi )**( 2 ) + 
 	 ( -1391/21 * gamma + 
 	 ( -26431039/266112 * nu + 
 	 ( complex( 0,-29/168 ) * np.pi * nu + 
 	 ( 3239/1536 * ( np.pi )**( 2 ) * nu + 
 	 ( 411151/66528 * ( nu )**( 2 ) + 
 	 ( -2257631/798336 * ( nu )**( 3 ) + 
 	 ( -108712/105 * np.log( 2 ) + 
 	 ( complex( 0,886 ) * np.pi * np.log( 2 ) + 
 	 ( 54 * ( np.log( 2 ) )**( 2 ) + 
 	 ( 66447/140 * np.log( 3 ) + 
 	 ( complex( 0,-1863/4 ) * np.pi * np.log( 3 ) + 
 	 ( -1863/2 * np.log( 2 ) * np.log( 3 ) + 
 	 ( 1863/4 * ( np.log( 3 ) )**( 2 ) + 
 	 ( ( np.e )**( complex( 0,-4 ) * xi ) * ( 131282622323/139708800 + 
 	 ( complex( 0,57031/630 ) * np.pi + 
 	 ( 533/36 * ( np.pi )**( 2 ) + 
 	 ( -57031/315 * gamma + 
 	 ( 548729393/997920 * nu + 
 	 ( -223163/9216 * ( np.pi )**( 2 ) * nu + 
 	 ( -34741549/798336 * ( nu )**( 2 ) + 
 	 ( -6313787/598752 * ( nu )**( 3 ) + 
 	 ( -16799/15 * np.log( 2 ) + 
 	 ( complex( 0,2231/3 ) * np.pi * np.log( 2 ) + 
 	 ( 1655/3 * ( np.log( 2 ) )**( 2 ) + 
 	 ( -29853/40 * np.log( 3 ) + 
 	 ( complex( 0,5859/8 ) * np.pi * np.log( 3 ) + 
 	 ( 27/4 * np.log( 2 ) * np.log( 3 ) + 
 	 ( -5859/8 * ( np.log( 3 ) )**( 2 ) + 
 	 ( 334375/504 * np.log( 5 ) + 
 	 ( complex( 0,-15625/24 ) * np.pi * np.log( 5 ) + 
 	 ( -15625/12 * np.log( 2 ) * np.log( 5 ) + 
 	 ( 15625/24 * ( np.log( 5 ) )**( 2 ) + 
 	 -57031/630 * np.log( x ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) + 
 	 ( ( np.e )**( complex( 0,-2 ) * xi ) * ( 29913154181/69854400 + 
 	 ( complex( 0,43763/1260 ) * np.pi + 
 	 ( 409/72 * ( np.pi )**( 2 ) + 
 	 ( -43763/630 * gamma + 
 	 ( -85085639/498960 * nu + 
 	 ( 12341/4608 * ( np.pi )**( 2 ) * nu + 
 	 ( -1125233/66528 * ( nu )**( 2 ) + 
 	 ( -5732197/1197504 * ( nu )**( 3 ) + 
 	 ( 1033513/630 * np.log( 2 ) + 
 	 ( complex( 0,-10477/6 ) * np.pi * np.log( 2 ) + 
 	 ( -493/6 * ( np.log( 2 ) )**( 2 ) + 
 	 ( -37557/140 * np.log( 3 ) + 
 	 ( complex( 0,1053/4 ) * np.pi * np.log( 3 ) + 
 	 ( 1053/2 * np.log( 2 ) * np.log( 3 ) + 
 	 ( -1053/4 * ( np.log( 3 ) )**( 2 ) + 
 	 ( -334375/504 * np.log( 5 ) + 
 	 ( complex( 0,15625/24 ) * np.pi * np.log( 5 ) + 
 	 ( 15625/12 * np.log( 2 ) * np.log( 5 ) + 
 	 ( -15625/24 * ( np.log( 5 ) )**( 2 ) + 
 	 -43763/1260 * np.log( x ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) + 
 	 ( ( np.e )**( complex( 0,2 ) * xi ) * ( 27122687651/69854400 + 
 	 ( complex( 0,8453/280 ) * np.pi + 
 	 ( 359/72 * ( np.pi )**( 2 ) + 
 	 ( -38413/630 * gamma + 
 	 ( -167972011/498960 * nu + 
 	 ( complex( 0,-473/1512 ) * np.pi * nu + 
 	 ( 20623/4608 * ( np.pi )**( 2 ) * nu + 
 	 ( 57403/199584 * ( nu )**( 2 ) + 
 	 ( -4225115/1197504 * ( nu )**( 3 ) + 
 	 ( 6527/70 * np.log( 2 ) + 
 	 ( complex( 0,-2527/12 ) * np.pi * np.log( 2 ) + 
 	 ( -1267/6 * ( np.log( 2 ) )**( 2 ) + 
 	 ( -56817/280 * np.log( 3 ) + 
 	 ( complex( 0,1593/8 ) * np.pi * np.log( 3 ) + 
 	 ( 1593/4 * np.log( 2 ) * np.log( 3 ) + 
 	 ( -1593/8 * ( np.log( 3 ) )**( 2 ) + 
 	 -38413/1260 * np.log( x ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) + 
 	 ( ( np.e )**( complex( 0,4 ) * xi ) * ( -22662221353/34927200 + 
 	 ( complex( 0,65377/2520 ) * np.pi + 
 	 ( 37/9 * ( np.pi )**( 2 ) + 
 	 ( -15836/315 * gamma + 
 	 ( 1657156441/3991680 * nu + 
 	 ( complex( 0,-377/504 ) * np.pi * nu + 
 	 ( -70315/9216 * ( np.pi )**( 2 ) * nu + 
 	 ( -45937741/266112 * ( nu )**( 2 ) + 
 	 ( 5581397/2395008 * ( nu )**( 3 ) + 
 	 ( -34133/315 * np.log( 2 ) + 
 	 ( complex( 0,85/12 ) * np.pi * np.log( 2 ) + 
 	 ( 23/3 * ( np.log( 2 ) )**( 2 ) + 
 	 -7918/315 * np.log( x ) ) ) ) ) ) ) ) ) ) ) ) ) + 
 	 -1391/42 * np.log( x ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) + 
 	 ( ( e )**( 2 ) * ( 148112893/1058400 + 
 	 ( complex( 0,1712/105 ) * np.pi + 
 	 ( 8/3 * ( np.pi )**( 2 ) + 
 	 ( -3424/105 * gamma + 
 	 ( -35837/1440 * nu + 
 	 ( 205/192 * ( np.pi )**( 2 ) * nu + 
 	 ( -4279/504 * ( nu )**( 2 ) + 
 	 ( -1183/2592 * ( nu )**( 3 ) + 
 	 ( 428/21 * np.log( 2 ) + 
 	 ( complex( 0,-84 ) * np.pi * np.log( 2 ) + 
 	 ( -84 * ( np.log( 2 ) )**( 2 ) + 
 	 ( -2889/35 * np.log( 3 ) + 
 	 ( complex( 0,81 ) * np.pi * np.log( 3 ) + 
 	 ( 162 * np.log( 2 ) * np.log( 3 ) + 
 	 ( -81 * ( np.log( 3 ) )**( 2 ) + 
 	 ( ( np.e )**( complex( 0,-2 ) * xi ) * ( 6487472689/23284800 + 
 	 ( complex( 0,5671/210 ) * np.pi + 
 	 ( 53/12 * ( np.pi )**( 2 ) + 
 	 ( -5671/105 * gamma + 
 	 ( 18175769/221760 * nu + 
 	 ( -779/192 * ( np.pi )**( 2 ) * nu + 
 	 ( -1137721/66528 * ( nu )**( 2 ) + 
 	 ( -883591/399168 * ( nu )**( 3 ) + 
 	 ( -321 * np.log( 2 ) + 
 	 ( complex( 0,209 ) * np.pi * np.log( 2 ) + 
 	 ( -47 * ( np.log( 2 ) )**( 2 ) + 
 	 ( 2889/35 * np.log( 3 ) + 
 	 ( complex( 0,-81 ) * np.pi * np.log( 3 ) + 
 	 ( -162 * np.log( 2 ) * np.log( 3 ) + 
 	 ( 81 * ( np.log( 3 ) )**( 2 ) + 
 	 -5671/210 * np.log( x ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) + 
 	 ( ( np.e )**( complex( 0,2 ) * xi ) * ( -69767507/7761600 + 
 	 ( complex( 0,2461/210 ) * np.pi + 
 	 ( 23/12 * ( np.pi )**( 2 ) + 
 	 ( -2461/105 * gamma + 
 	 ( -33218051/665280 * nu + 
 	 ( complex( 0,-29/126 ) * np.pi * nu + 
 	 ( -41/192 * ( np.pi )**( 2 ) * nu + 
 	 ( -260701/9504 * ( nu )**( 2 ) + 
 	 ( 295943/399168 * ( nu )**( 3 ) + 
 	 ( -749/15 * np.log( 2 ) + 
 	 ( complex( 0,3 ) * np.pi * np.log( 2 ) + 
 	 ( 3 * ( np.log( 2 ) )**( 2 ) + 
 	 -2461/210 * np.log( x ) ) ) ) ) ) ) ) ) ) ) ) ) + 
 	 -1712/105 * np.log( x ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) + 
 	 -428/105 * np.log( x ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) )
    return(h)


def New_phase_TT2(x, x0,e0, SO, chiS, chiA, M, nu, delta):
    PHI_TT2 = -1/32 * M * ( x )**( -5/2 ) * ( nu )**( -1 ) * ( 1 + ( x * ( \
3715/1008 + 55/12 * nu ) + ( ( x )**( 3/2 ) * ( -10 * numpy.pi + ( \
565/24 * SO * delta * chiA + ( 565/24 * SO * chiS + -95/6 * SO * nu * \
chiS ) ) ) + ( ( x )**( 2 ) * ( 15293365/1016064 + ( 3085/144 * ( nu \
)**( 2 ) + ( -405/16 * ( SO )**( 2 ) * ( chiA )**( 2 ) + ( -405/8 * ( \
SO )**( 2 ) * delta * chiA * chiS + ( -405/16 * ( SO )**( 2 ) * ( \
chiS )**( 2 ) + nu * ( 27145/1008 + ( 100 * ( SO )**( 2 ) * ( chiA \
)**( 2 ) + 5/4 * ( SO )**( 2 ) * ( chiS )**( 2 ) ) ) ) ) ) ) ) + ( ( \
x )**( 3 ) * ( 12348611926451/18776862720 + ( -1712/21 * EulerGamma + \
( -160/3 * ( numpy.pi )**( 2 ) + ( -127825/5184 * ( nu )**( 3 ) + ( ( \
195425/384 * ( SO )**( 2 ) + -63845/144 * ( SO )**( 2 ) * ( delta \
)**( 2 ) ) * ( chiA )**( 2 ) + ( 1135/6 * numpy.pi * SO * chiS + ( \
75515/1152 * ( SO )**( 2 ) * ( chiS )**( 2 ) + ( chiA * ( 1135/6 * \
numpy.pi * SO * delta + 75515/576 * ( SO )**( 2 ) * delta * chiS ) + \
( nu * ( -15737765635/12192768 + ( 2255/48 * ( numpy.pi )**( 2 ) + ( \
-683635/336 * ( SO )**( 2 ) * ( chiA )**( 2 ) + ( -130 * numpy.pi * \
SO * chiS + ( -8225/72 * ( SO )**( 2 ) * delta * chiA * chiS + \
-232415/2016 * ( SO )**( 2 ) * ( chiS )**( 2 ) ) ) ) ) ) + ( ( nu \
)**( 2 ) * ( 76055/6912 + ( -120 * ( SO )**( 2 ) * ( chiA )**( 2 ) + \
1255/36 * ( SO )**( 2 ) * ( chiS )**( 2 ) ) ) + ( -3424/21 * \
numpy.log( 2 ) + -856/21 * numpy.log( x ) ) ) ) ) ) ) ) ) ) ) ) + ( ( \
x )**( 5/2 ) * ( 38645/1344 * numpy.pi * numpy.log( x ) + ( \
-732985/4032 * SO * delta * chiA * numpy.log( x ) + ( -732985/4032 * \
SO * chiS * numpy.log( x ) + ( 85/4 * SO * ( nu )**( 2 ) * chiS * \
numpy.log( x ) + nu * ( -65/16 * numpy.pi * numpy.log( x ) + ( -35/4 \
* SO * delta * chiA * numpy.log( x ) + 6065/36 * SO * chiS * \
numpy.log( x ) ) ) ) ) ) ) + -785/272 * ( e0 )**( 2 ) * ( x )**( \
-19/6 ) * ( x0 )**( 19/6 ) * ( 1 + ( x0 * ( 2833/1008 + -197/36 * nu \
) + ( ( x0 )**( 3/2 ) * ( 377/72 * numpy.pi + ( -157/54 * SO * delta \
* chiA + ( -157/54 * SO * chiS + 55/27 * SO * nu * chiS ) ) ) + ( ( x \
)**( 5/2 ) * ( -131697334/8456805 * numpy.pi + ( \
39407176859/405926640 * SO * delta * chiA + ( 39407176859/405926640 * \
SO * chiS + ( -25220180/724869 * SO * ( nu )**( 2 ) * chiS + nu * ( \
-268652717/9664920 * numpy.pi + ( 372333779/14497380 * SO * delta * \
chiA + -6539884139/101481660 * SO * chiS ) ) ) ) ) ) + ( ( x0 )**( \
5/2 ) * ( 764881/90720 * numpy.pi + ( -1279073/38880 * SO * delta * \
chiA + ( -1279073/38880 * SO * chiS + ( -4322/243 * SO * ( nu )**( 2 \
) * chiS + nu * ( -949457/22680 * numpy.pi + ( 242719/9720 * SO * \
delta * chiA + 146807/2430 * SO * chiS ) ) ) ) ) ) + ( ( x0 )**( 2 ) \
* ( -1193251/3048192 + ( 18155/1296 * ( nu )**( 2 ) + ( 191/96 * ( SO \
)**( 2 ) * ( chiA )**( 2 ) + ( 191/48 * ( SO )**( 2 ) * delta * chiA \
* chiS + ( 191/96 * ( SO )**( 2 ) * ( chiS )**( 2 ) + nu * ( \
-66317/9072 + ( -89/12 * ( SO )**( 2 ) * ( chiA )**( 2 ) + -13/24 * ( \
SO )**( 2 ) * ( chiS )**( 2 ) ) ) ) ) ) ) ) + ( x * ( 6955261/2215584 \
+ ( 436441/79128 * nu + ( x0 * ( 19704254413/2233308672 + ( \
-16718633/9970128 * nu + -85978877/2848608 * ( nu )**( 2 ) ) ) + ( ( \
x0 )**( 3/2 ) * ( 2622133397/159522048 * numpy.pi + ( -6955261/762048 \
* SO * delta * chiA + ( -6955261/762048 * SO * chiS + ( \
24004255/2136456 * SO * ( nu )**( 2 ) * chiS + nu * ( \
164538257/5697216 * numpy.pi + ( -436441/27216 * SO * delta * chiA + \
-576757963/59820768 * SO * chiS ) ) ) ) ) ) + ( x0 )**( 2 ) * ( \
-8299372143511/6753525424128 + ( 7923586355/102549888 * ( nu )**( 3 ) \
+ ( 1328454851/212696064 * ( SO )**( 2 ) * ( chiA )**( 2 ) + ( \
1328454851/106348032 * ( SO )**( 2 ) * delta * chiA * chiS + ( \
1328454851/212696064 * ( SO )**( 2 ) * ( chiS )**( 2 ) + ( ( nu )**( \
2 ) * ( 3499644089/957132288 + ( -38843249/949536 * ( SO )**( 2 ) * ( \
chiA )**( 2 ) + -5673733/1899072 * ( SO )**( 2 ) * ( chiS )**( 2 ) ) \
) + nu * ( -6055808184535/241197336576 + ( -654514841/53174016 * ( SO \
)**( 2 ) * ( chiA )**( 2 ) + ( 83360231/3798144 * ( SO )**( 2 ) * \
delta * chiA * chiS + 61637903/6646752 * ( SO )**( 2 ) * ( chiS )**( \
2 ) ) ) ) ) ) ) ) ) ) ) ) ) ) + ( ( x )**( 2 ) * ( \
377620541/107433216 + ( 36339727/2238192 * ( nu )**( 2 ) + ( \
-891871/165792 * ( SO )**( 2 ) * ( chiA )**( 2 ) + ( -891871/82896 * \
( SO )**( 2 ) * delta * chiA * chiS + ( -891871/165792 * ( SO )**( 2 \
) * ( chiS )**( 2 ) + ( nu * ( 561233971/31334688 + ( 430933/20724 * \
( SO )**( 2 ) * ( chiA )**( 2 ) + 30005/41448 * ( SO )**( 2 ) * ( \
chiS )**( 2 ) ) ) + x0 * ( 1069798992653/108292681728 + ( \
-7158926219/80574912 * ( nu )**( 3 ) + ( -2526670543/167118336 * ( SO \
)**( 2 ) * ( chiA )**( 2 ) + ( -2526670543/83559168 * ( SO )**( 2 ) * \
delta * chiA * chiS + ( -2526670543/167118336 * ( SO )**( 2 ) * ( \
chiS )**( 2 ) + ( ( nu )**( 2 ) * ( -39391912661/752032512 + ( \
-84893801/746064 * ( SO )**( 2 ) * ( chiA )**( 2 ) + -5910985/1492128 \
* ( SO )**( 2 ) * ( chiS )**( 2 ) ) ) + nu * ( \
5894683956785/189512193024 + ( 3671556487/41779584 * ( SO )**( 2 ) * \
( chiA )**( 2 ) + ( 175698587/2984256 * ( SO )**( 2 ) * delta * chiA \
* chiS + 657447137/20889792 * ( SO )**( 2 ) * ( chiS )**( 2 ) ) ) ) ) \
) ) ) ) ) ) ) ) ) ) ) + ( ( x )**( 3/2 ) * ( -1114537/141300 * \
numpy.pi + ( 208012/21195 * SO * delta * chiA + ( 208012/21195 * SO * \
chiS + ( -910486/105975 * SO * nu * chiS + ( x0 * ( \
-3157483321/142430400 * numpy.pi + ( 21046357/763020 * SO * delta * \
chiA + ( 21046357/763020 * SO * chiS + ( 89682871/1907550 * SO * ( nu \
)**( 2 ) * chiS + nu * ( 219563789/5086800 * numpy.pi + ( \
-10244591/190755 * SO * delta * chiA + -4158188899/53411400 * SO * \
chiS ) ) ) ) ) ) + ( x0 )**( 3/2 ) * ( -420180449/10173600 * ( \
numpy.pi )**( 2 ) + ( -104006/3645 * ( SO )**( 2 ) * ( delta )**( 2 ) \
* ( chiA )**( 2 ) + ( 567084929/7630200 * numpy.pi * SO * chiS + ( \
-104006/3645 * ( SO )**( 2 ) * ( chiS )**( 2 ) + ( -10015346/572265 * \
( SO )**( 2 ) * ( nu )**( 2 ) * ( chiS )**( 2 ) + ( chiA * ( \
567084929/7630200 * numpy.pi * SO * delta + -208012/3645 * ( SO )**( \
2 ) * delta * chiS ) + nu * ( -116463073/1907550 * numpy.pi * SO * \
chiS + ( 128676451/2861325 * ( SO )**( 2 ) * delta * chiA * chiS + \
128676451/2861325 * ( SO )**( 2 ) * ( chiS )**( 2 ) ) ) ) ) ) ) ) ) ) \
) ) ) ) + ( ( x )**( 3 ) * ( -204814565759250649/1061268280934400 + ( \
12483797/791280 * EulerGamma + ( 365639621/13022208 * ( numpy.pi )**( \
2 ) + ( 5885194385/175799808 * ( nu )**( 3 ) + ( ( \
-24688893025/364621824 * ( SO )**( 2 ) + 356779187/7324992 * ( SO \
)**( 2 ) * ( delta )**( 2 ) ) * ( chiA )**( 2 ) + ( \
-1556012125/19533312 * numpy.pi * SO * chiS + ( \
-62362961449/3281596416 * ( SO )**( 2 ) * ( chiS )**( 2 ) + ( chiA * \
( -1556012125/19533312 * numpy.pi * SO * delta + \
-62362961449/1640798208 * ( SO )**( 2 ) * delta * chiS ) + ( nu * ( \
34787542048195/137827049472 + ( -8764775/1446912 * ( numpy.pi )**( 2 \
) + ( 23282295175/91155456 * ( SO )**( 2 ) * ( chiA )**( 2 ) + ( \
315184913/4883328 * numpy.pi * SO * chiS + ( -2236105693/58599936 * ( \
SO )**( 2 ) * delta * chiA * chiS + -665932109/29299968 * ( SO )**( 2 \
) * ( chiS )**( 2 ) ) ) ) ) ) + ( ( nu )**( 2 ) * ( \
80353703837/1640798208 + ( 40617335/542592 * ( SO )**( 2 ) * ( chiA \
)**( 2 ) + 448905995/29299968 * ( SO )**( 2 ) * ( chiS )**( 2 ) ) ) + \
( 164286623/2373840 * numpy.log( 2 ) + ( -26079003/703360 * \
numpy.log( 3 ) + 12483797/1582560 * numpy.log( x ) ) ) ) ) ) ) ) ) ) \
) ) ) + ( x0 )**( 3 ) * ( 26531900578691/168991764480 + ( -3317/126 * \
EulerGamma + ( 122833/10368 * ( numpy.pi )**( 2 ) + ( -3090307/139968 \
* ( nu )**( 3 ) + ( ( 10898429/290304 * ( SO )**( 2 ) + -110371/11664 \
* ( SO )**( 2 ) * ( delta )**( 2 ) ) * ( chiA )**( 2 ) + ( \
-157043/15552 * numpy.pi * SO * chiS + ( 73362757/2612736 * ( SO )**( \
2 ) * ( chiS )**( 2 ) + ( chiA * ( -157043/15552 * numpy.pi * SO * \
delta + 73362757/1306368 * ( SO )**( 2 ) * delta * chiS ) + ( nu * ( \
9155185261/548674560 + ( -3977/1152 * ( numpy.pi )**( 2 ) + ( \
-11796881/72576 * ( SO )**( 2 ) * ( chiA )**( 2 ) + ( 37663/3888 * \
numpy.pi * SO * chiS + ( -3284705/46656 * ( SO )**( 2 ) * delta * \
chiA * chiS + -18949901/326592 * ( SO )**( 2 ) * ( chiS )**( 2 ) ) ) \
) ) ) + ( ( nu )**( 2 ) * ( -5732473/1306368 + ( 24271/432 * ( SO \
)**( 2 ) * ( chiA )**( 2 ) + 272239/23328 * ( SO )**( 2 ) * ( chiS \
)**( 2 ) ) ) + ( -12091/1890 * numpy.log( 2 ) + ( -26001/560 * \
numpy.log( 3 ) + -3317/252 * numpy.log( x0 ) ) ) ) ) ) ) ) ) ) ) ) ) \
) ) ) ) ) ) ) ) ) ) ) ) ) ) ) )
    return PHI_TT2






# def IMRESIGMAHM_td(**input_params):
#     """
#     Returns tapered time domain gravitational polarizations for IMRESIGMAHM waveform containing all (l,|m|) modes available.

#     Parameters
#     ----------

#     Takes the same parameters as pycbc.waveform.get_td_waveform().
    
#     Returns
#     -------

#     hplus : PyCBC TimeSeries
#         The plus-polarization of the waveform in time domain tapered from start to 0.4s.
#     hcross : PyCBC TimeSeries
#         The cross-polarization of the waveform in time domain tapered from start to 0.4s.
#     """
#     #importing here instead of globally to avoid circular imports
#     from gwnr.waveform import esigma_utils

#     wf_input_params = use_modified_input_params(**input_params)
    
#     hp, hc = esigma_utils.get_imr_esigma_waveform(**wf_input_params)
#     hp_ts = TimeSeries(hp, input_params['delta_t'])
#     hc_ts = TimeSeries(hc, input_params['delta_t'])
    
#     hp_tapered = taper_signal(hp_ts)
#     hc_tapered = taper_signal(hc_ts)
#     return(hp_tapered, hc_tapered)

# def IMRESIGMA_td(**input_params):
#     """
#     Returns tapered time domain gravitational polarizations for IMRESIGMA waveform containing only the (l,|m|) = (2,2) mode.

#     Parameters
#     ----------

#     Takes the same parameters as pycbc.waveform.get_td_waveform().
    
#     Returns
#     -------

#     hplus : PyCBC TimeSeries
#         The plus-polarization of the waveform in time domain tapered from start to 0.4s.
#     hcross : PyCBC TimeSeries
#         The cross-polarization of the waveform in time domain tapered from start to 0.4s.
#     """
#     #importing here instead of globally to avoid circular imports
#     from gwnr.waveform import esigma_utils

#     wf_input_params = use_modified_input_params(**input_params)
    
#     hp, hc = esigma_utils.get_imr_esigma_waveform(**wf_input_params, modes_to_use=[(2, 2)])
#     hp_ts = TimeSeries(hp, input_params['delta_t'])
#     hc_ts = TimeSeries(hc, input_params['delta_t'])
    
#     hp_tapered = taper_signal(hp_ts)
#     hc_tapered = taper_signal(hc_ts)
#     return(hp_tapered, hc_tapered)

# def InspiralESIGMAHM_td(**input_params):
#     """
#     Returns tapered time domain gravitational polarizations for InspiralESIGMAHM waveform containing all (l,|m|) modes available.

#     Parameters
#     ----------

#     Takes the same parameters as pycbc.waveform.get_td_waveform().
    
#     Returns
#     -------

#     hplus : PyCBC TimeSeries
#         The plus-polarization of the waveform in time domain tapered from start to 0.4s.
#     hcross : PyCBC TimeSeries
#         The cross-polarization of the waveform in time domain tapered from start to 0.4s.
#     """
#     #importing here instead of globally to avoid circular imports
#     from gwnr.waveform import esigma_utils

#     wf_input_params = use_modified_input_params(**input_params)
    
#     _, hp, hc = esigma_utils.get_inspiral_esigma_waveform(**wf_input_params)
#     hp_ts = TimeSeries(hp, input_params['delta_t'])
#     hc_ts = TimeSeries(hc, input_params['delta_t'])
    
#     hp_tapered = taper_signal(hp_ts)
#     hc_tapered = taper_signal(hc_ts)
#     return(hp_tapered, hc_tapered)

# def InspiralESIGMA_td(**input_params):
#     """
#     Returns tapered time domain gravitational polarizations for InspiralESIGMA waveform containing only the (l,|m|) = (2,2) mode.

#     Parameters
#     ----------

#     Takes the same parameters as pycbc.waveform.get_td_waveform().
    
#     Returns
#     -------

#     hplus : PyCBC TimeSeries
#         The plus-polarization of the waveform in time domain tapered from start to 0.4s.
#     hcross : PyCBC TimeSeries
#         The cross-polarization of the waveform in time domain tapered from start to 0.4s.
#     """
#     #importing here instead of globally to avoid circular imports
#     from gwnr.waveform import esigma_utils

#     wf_input_params = use_modified_input_params(**input_params)
    
#     _, hp, hc = esigma_utils.get_inspiral_esigma_waveform(**wf_input_params, modes_to_use=[(2, 2)])
#     hp_ts = TimeSeries(hp, input_params['delta_t'])
#     hc_ts = TimeSeries(hc, input_params['delta_t'])
    
#     hp_tapered = taper_signal(hp_ts)
#     hc_tapered = taper_signal(hc_ts)
#     return(hp_tapered, hc_tapered)
