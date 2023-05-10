import numpy as np
from sklearn.decomposition import PCA
from tensorly.decomposition import tucker

ap = np.loadtxt("ap_fingerprint.txt")
ava = np.loadtxt("avalon_fingerprint.txt")
erg = np.loadtxt("erg_fingerprint.txt")
morgan = np.loadtxt("morgan_fingerprint.txt")
rdk = np.loadtxt("rdk_fingerprint.txt")
tt = np.loadtxt("tt_fingerprint.txt")

pca = PCA(n_components=50)

print("loading")
reduced_rdk = pca.fit_transform(rdk)
print("loading")
reduced_ap = pca.fit_transform(ap)
print("loading")
reduced_ava = pca.fit_transform(ava)
print("loading")
reduced_morgan = pca.fit_transform(morgan)
print("loading")
reduced_tt = pca.fit_transform(tt)
print("loading")
reduced_erg = pca.fit_transform(erg)

all = np.hstack((reduced_rdk, reduced_ap, reduced_ava, reduced_morgan, reduced_tt, reduced_erg))

np.savetxt("all.txt", all)
