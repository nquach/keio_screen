import numpy as np

#THIS INTERPOLATION IS ONLY FOR 200µL of liquid in 96 well plate well.
#Data from calibration experiment
A200 = [0, 0.0088, .0193, .0405, .0797, .123, .1673, .2128, .2349, .3481]
Aspec = [0, .072, .152, .297, .570, .783, .948, 1.088, 1.199, 1.367]

A200 = np.asarray(A200)
Aspec = np.asarray(Aspec)

#Plate reader OD600 to interpolate into a spectrophotometer reading
to_interp = 0.12

print "Interpolating plate reader OD600 of " + str(to_interp) + ":"
print np.interp(to_interp, A200, Aspec, np.amin(Aspec), np.amax(Aspec))