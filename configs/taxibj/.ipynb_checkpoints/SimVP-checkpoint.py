method = 'SimVP'
# model
pre_seq_length = 10
aft_seq_length = 90
total_length = pre_seq_length + aft_seq_length
spatio_kernel_enc = 3
spatio_kernel_dec = 3
# model_type = None
hid_S = 32
hid_T = 256
N_T = 8
N_S = 2
# training
lr = 1e-3
batch_size = 1
drop_path = 0.1
sched = 'cosine'
warmup_epoch = 5
