# %%
import numpy as np
import matplotlib.pyplot as plt

base_dir = "/mnt/xutan/lrq/AVDA/modified_FADA"

# %%
accs_avda_s2m = np.load(f'{base_dir}/accs_avda_s2m2.npy')
accs_fada_s2m = np.load(f'{base_dir}/accs_fada_s2m.npy')

# %%
plt.figure(dpi=150)
plt.plot(np.arange(100), accs_avda_s2m[:100], label="Proposed")
plt.plot(np.arange(100), accs_fada_s2m[:100], label="FADA")
plt.legend()
plt.title("Training Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

# %%
losses_avda_s2m = np.load(f'{base_dir}/losses_avda_s2m.npy')
losses_fada_s2m = np.load(f'{base_dir}/losses_fada_s2m.npy')

# %%
plt.plot(np.arange(100), losses_avda_s2m[:100], label="Proposed")
plt.plot(np.arange(100), losses_fada_s2m[:100], label="FADA")
plt.legend()
plt.title("Training Stability")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
