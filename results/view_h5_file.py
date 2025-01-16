import h5py
import matplotlib.pyplot as plt

file_path = "MNIST_CLDP_test.h5"

with h5py.File(file_path, 'r') as hf:
    rs_test_acc_data = hf['rs_test_acc'][:]
    rs_train_loss_data = hf['rs_train_loss'][:]
    rs_server_acc_data = hf['rs_server_acc'][:]
    rs_server_loss_data = hf['rs_server_loss'][:]
    extra_msg = hf['extra_msg'][()]

    extra_msg = extra_msg.decode('utf-8')
    print("----------------------------------------------")
    print(extra_msg)
    print("rs_server_acc:", rs_server_acc_data)
    print("best_server_acc", max(rs_server_acc_data))
    print("rs_server_loss:", rs_server_loss_data)
    print("----------------------------------------------")

    # print(f"last_server_acc={rs_server_acc_data[-1]}, best_server_acc={max(rs_server_acc_data)}")
    print("last_server_acc={:.4f}, best_server_acc={:.4f}".format(rs_server_acc_data[-1], max(rs_server_acc_data)))

    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    axs[0].plot(rs_test_acc_data, label='rs_test_acc', alpha=0.5, linestyle='--')
    axs[0].plot(rs_server_acc_data, label='rs_server_acc', alpha=0.5, linestyle='--')
    axs[0].set_xlabel('Communication round')
    axs[0].set_ylabel('Value')
    axs[0].set_title(file_path)
    axs[0].legend()

    axs[1].plot(rs_train_loss_data, label='rs_train_loss', alpha=0.5, linestyle='--')
    axs[1].plot(rs_server_loss_data, label='rs_server_loss', alpha=0.5, linestyle='--')
    axs[1].set_xlabel('Communication round')
    axs[1].set_ylabel('Value')
    axs[1].set_title(file_path)
    axs[1].legend()

    plt.tight_layout()

    plt.show()

    # upper_num = 900
    # for i in range(upper_num):
    #     print(i, rs_server_acc_data[i], rs_server_loss_data[i])
