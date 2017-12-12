import matplotlib.pyplot as plt


def get_data(file_name):
    training_steps = []
    total_losses = []
    with open(file_name, 'r') as f:
        for line in f.readlines():
            if line.__contains__('Training Step'):
                # Training Step: 9100  | total loss: 1.53503 | time: 172.716s
                cols = line.split(' ')
                training_steps.append(cols[2])
                total_losses.append(cols[7])
    return training_steps, total_losses


def get_data1(file_name):
    training_steps = []
    total_losses = []
    with open(file_name, 'r') as f:
        for line in f.readlines():
            if line.__contains__('Training Step'):
                # [A[ATraining Step: 10944  | total loss: [1m[32m1.44601[0m[0m | time: 0.855s
                cols = line.split(' ')
                # cols[7] = '\x1b[1m\x1b[32m1.49210\x1b[0m\x1b[0m'
                training_step = cols[2]
                if int(training_step) % 100 == 0:
                    total_loss = cols[7].split('\x1b[')[2].split('m')[1]
                    training_steps.append(training_step)
                    total_losses.append(total_loss)
    return training_steps, total_losses


def plot(datas, colors, labels):
    for data, color, label in zip(datas, colors, labels):
        plt.plot(data[0], data[1], color, label=label)

    plt.xlabel("step")
    plt.ylabel("total loss")
    plt.legend(loc='upper right')
    plt.savefig('graph.png')


t_5_1_data = get_data('tensor5.log')
t_5_2_data = get_data1('tensor5.logg')
t_6_1_data = get_data('tensor6.log')
t_6_2_data = get_data1('tensor6.logg')
tensor5_data = t_5_1_data[0] + t_5_2_data[0], t_5_1_data[1] + t_5_2_data[1]
tensor6_data = t_6_1_data[0] + t_6_2_data[0], t_6_1_data[1] + t_6_2_data[1]


plot([tensor5_data, tensor6_data, get_data1('tensor7.logg')], ['r-', 'b-', 'g-'], ['deeper network', 'shallower network', 'wider network'])
