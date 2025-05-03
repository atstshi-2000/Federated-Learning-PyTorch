import pickle
with open('mnist_cnn_50_Crient[0.5]_iid[1]_Epoch[3]_Batch_s[32]_lr[0.03]__el2n[0_num_users[10]_threshold[0.04].pkl','rb') as f:
        result = pickle.load(f)
        print(result[1])