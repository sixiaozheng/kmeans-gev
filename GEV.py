from scipy.stats import genextreme
from scipy.stats import gumbel_r
from scipy.stats import invweibull
from scipy.stats import weibull_max
import matplotlib.pyplot as plt
import  numpy as np
plt.figure(figsize=(18, 6))

plt.subplot(131)
c=0.5
x = np.linspace(genextreme.ppf(0.0001, c),
                genextreme.ppf(0.9999, c), 4000)
plt.plot(x, genextreme.pdf(x, c), 'g')

c=-0.5
x = np.linspace(genextreme.ppf(0.0001, c),
                genextreme.ppf(0.9999, c), 4000)
plt.plot(x, genextreme.pdf(x, c), 'b')

c=0
x = np.linspace(genextreme.ppf(0.0001, c),
                genextreme.ppf(0.9999, c), 4000)
plt.plot(x, genextreme.pdf(x, c), 'r')

plt.axis([-4,4,0,0.5])


plt.subplot(132)
c = 1
x = np.linspace(gumbel_r.ppf(0.001, c),
                gumbel_r.ppf(0.999, c), 5000)
plt.plot(x, gumbel_r.pdf(x,0.5), 'b')

x = np.linspace(invweibull.ppf(0.001, c),
                invweibull.ppf(0.999, c), 5000)
plt.plot(x, invweibull.pdf(x,c), 'r')

x = np.linspace(weibull_max.ppf(0.001, c),
                weibull_max.ppf(0.999, c), 5000)
plt.plot(x, weibull_max.pdf(x,c), 'g')
plt.axis([-8,10,0,1])

plt.subplot(133)
c = 1
x = np.linspace(gumbel_r.ppf(0.001, c),
                gumbel_r.ppf(0.999, c), 5000)
plt.plot(x, gumbel_r.cdf(x,0.5), 'b')

x = np.linspace(invweibull.ppf(0.001, c),
                invweibull.ppf(0.999, c), 5000)
plt.plot(x, invweibull.cdf(x,c), 'r')

x = np.linspace(weibull_max.ppf(0.001, c),
                weibull_max.ppf(0.999, c), 5000)
plt.plot(x, weibull_max.cdf(x,c), 'g')
plt.axis([-8,10,0,1])

plt.show()

