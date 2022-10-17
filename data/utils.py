def read_spectrum(fname):
    spec = []
    for line in open(fname, 'r'):
        spec.append(float(line.rstrip()))
    return np.array(spec)

val = ["02","04","06","08","10","12","14","16","18","20"]

q = -1
for r in val:
    q = q+1
    spec[:,q] = = read_spectrum("SPS_NaITl_Co60_STEEL"+r+"mm.dat")
