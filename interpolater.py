import numpy as np

A200 = [0, 0.0088, .0193, .0405, .0797, .123, .1673, .2128, .2349, .3481]
Aspec = [0, .072, .152, .297, .570, .783, .948, 1.088, 1.199, 1.367]

Aspec2 = [.017, .026, .056, .087, .146, .242, .372, .570]
cfu = [9.8e6, 1.15e7, 1.81e7, 2.9e7, 3.27e7, 3.56e7, 4.4e7, 6.2e7]

A200 = np.asarray(A200)
Aspec = np.asarray(Aspec)
Aspec2 = np.asarray(Aspec2)
cfu = np.asarray(cfu)

#Plate reader OD600 to interpolate into a spectrophotometer reading
to_interp = raw_input("Input plate reader OD600: ")
to_interp = float(to_interp)
interp_spec =  np.interp(to_interp, A200, Aspec, np.amin(Aspec), np.amax(Aspec))
interp_cfu = np.interp(interp_spec, Aspec2, cfu, 0, np.amax(cfu))
i = '%0.2E' % interp_cfu
print "Number of cells: " + str(i) + " cfu/mL"