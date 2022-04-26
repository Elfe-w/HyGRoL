from losses import SupConLoss
import torch as th

# define loss with a temperature `temp`
criterion = SupConLoss(temperature=0.2)

# features: [bsz, n_views, f_dim]
# `n_views` is the number of crops from each image
# better be L2 normalized in f_dim dimension
features = th.randn(16,768)

features=features.view(features.shape[0],1,-1)

# labels: [bsz]
labels = th.FloatTensor([1,2,2,3,4,5,6,7,8,8,8,9,10,0,2,1])

# SupContrast
loss = criterion(features, labels)

print('='*80)
import torch
print(loss)