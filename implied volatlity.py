#VL 1217.2 IB EQD trading technical assessment
#SVI Implied Volatility Surface

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import newton, minimize
from datetime import datetime as dt
import re
import matplotlib.pyplot as plt

# Black-Scholes options pricing
def bsm(S, K, T, v, r, q, type):
	"""
	:param v: vol
	:param type: call or put
	:return: Black scholes price estimate, option vega
	"""
	d1 = (np.log(S/K) + (r-q + 0.5*v**2) *T) / (v*np.sqrt(T))
	d2 = d1 - v*np.sqrt(T)
	#vega = S*np.sqrt(T) * ss.norm.cdf(d1)
	if type == "call":
		price = S*np.exp(-q*T) * norm.cdf(d1) - K*np.exp(-r*T) * norm.cdf(d2)
	elif type == "put":
		price = K*np.exp(-r*T) * norm.cdf(-d2) - S*np.exp(-q*T) * norm.cdf(-d1)
	return price

# Implied Volatility via Newton's Method
def imp_vol(S, K, T, r, q, mkt, type):
	"""
	:param mkt: market price of option
	:param type: 'call', 'put'
	:return: v implied vol
	"""
	v = .2  # initial sigma, < results in - warning

	def model(v):
		return bsm(S, K, T, v, r, q, type) - mkt
	return newton(model, v)

# get r, q
def rq(S, K, T, c, p):
	"""
	C(k)-P(k) = S*exp(-q*t) - k*exp(-r*t)
	:return: risk free rate r, div yield q
	"""
	m, b = np.polyfit(K,c-p,1)
	r = -np.log(-m)/T
	q = -np.log(b/S)

	return r, q

# forward at time T
def fwd(S, r, q, T):
	fwd = S * np.exp((r - q) * T)
	return fwd

# load chain data
def load_input(input_file):
	# quotedata header
	# Ticker (desc), Last, net
	# Date @ xx:yy: zone
	# Calls, Last Sale, Net, Bid, Ask, Vol, Open Int, Puts, Last Sale, Net, Bid, Ask, Vol, Open Int
	# option expression as string: e.g. 'SPX1821L2400'

	with open(input_file, 'r') as a:
		S = float(a.readline().split(',')[1])
		t0 = pd.to_datetime(a.readline()[0:11])
			# t0 = a[0].to_string()
			# t0 = dt.strptime(t0, '0 %b %d %Y @ %H:%M ET') # start date t0 "today"
		df = pd.read_csv(input_file, skiprows =2)
		df = df.iloc[:, :-1] # drop NaN col

		# get contract maturity
		row = 0	# assuming chain df contains same maturity
		r1 = df['Calls'].iloc[row].replace(' ','')
		t1 = maturity(row, r1)
		T = (t1-t0).days/365

		df.rename(index=str, columns={"Bid": "Cbid", "Ask": "Cask", "Bid.1": "Pbid","Ask.1": "Pask"}, inplace=True)
		df['K'] = df.Calls.apply(lambda x: x.split(' ')[2])
		df['C'] = (df['Cbid'] + df['Cask'])/2
		df['P'] = (df['Pbid'] + df['Pask'])/2

		cols = ['Cbid','C','Cask','K','Pbid','P','Pask']
		chain = df[cols].apply(pd.to_numeric)

		return S, T, chain

def maturity(row,r1):
	"""
	:param row: row from df
	:param r1: row as string
	:return: t1
	"""
	p = re.compile("(\d{2})(\w{3})(.*\()(\w{3})(\d{2})(\d{2})")
	mon = p.match(r1).group(2)
	yr = str(int(float(p.match(r1).group(1)))+2000)
	day = str(int(float(p.match(r1).group(6))))
	mat = mon+day+yr+'16:00 ET'
	t1 = dt.strptime(mat, '%b%d%Y%H:%M ET')

	return t1

# SVI
def svi(a, b, p, m, s, K, S):
	"""
	args: a -- vert translation (controls overall level of volatility)
	m -- translates the curve horizontally
	s -- controls the local curvature around x=m
	b -- controls the steepness between the two halves
	p -- rotates the curve.
	"""
	x=np.log(K/S)
	return (a + b * (p * (x - m) + np.sqrt( (x - m)**2 + s**2) ) )

def svi_fit(params, Ks, IVs, S):
	svi_var = np.array([svi(*params, k, S) for k in Ks])
	return sum(np.power(svi_var - IVs**2,2))

### Main
S, T, chain = load_input('./quotedata.dat')
r, q = rq(S, np.array(chain.K), T, np.array(chain['C']), np.array(chain['P']))
print('r: %0.4f q: %0.4f' %(r, q))
									# imp_vol(S, K, T, r, q, mkt, type)
chain['cvol'] = chain.apply(lambda x: imp_vol(S, x['K'], T, r, q, x['C'], 'call'), axis=1)
chain['pvol'] = chain.apply(lambda x: imp_vol(S, x['K'], T, r, q, x['P'], 'put'), axis=1)
chain['IV'] = (chain['cvol'] + chain['pvol'])/2
print(chain.head()) # check df

# svi calibration based on parameter bounds, Gatheral
svi_init = [.01, 0.1, 0.1, .1, 0.1]

params = minimize(svi_fit, svi_init,
				args=(np.array(chain['K']), np.array(chain['IV']), S)).x
print('\na,b,p,m,s:',params)
fit = [np.sqrt(svi(*params, ks,S)) for ks in chain['K']]

# error check
delta = pd.Series(chain['IV']-fit)
print('\nMAE: %f' % delta.mad())
print('RMSE: %f' % delta.std())

#test OTM
S1 = 2684.79
v = np.sqrt(svi(*params, S1, S))
theo = bsm(S, S1, T, v, r, q, 'call')
print('\nTheoretical price:\nZ18',S1,'Calls midprice\t',theo)

plt.plot(chain['K'],fit, label='fit: a%5.3f, b=%5.3f,p=%5.3f,m=%5.3f,s=%5.3f' % tuple(params))
plt.xlabel('K')
plt.ylabel('IV')
plt.legend()
plt.show()
