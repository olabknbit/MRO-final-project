import matplotlib.pyplot as plt


def get_data(file_name):
    training_steps = []
    total_losses = []
    accs = []
    with open(file_name, 'r') as f:
        for line in f.readlines():
            if line.__contains__('Training Step'):
                # Training Step: 9100  | total loss: 1.53503 | time: 172.716s
                cols = line.split(' ')
                training_step = cols[2]
                if int(training_step) % 50 == 0:
                    training_steps.append(cols[2])
                    total_losses.append(cols[7])
            if line.__contains__('acc') and len(accs) < len(training_steps):
                # | Adam | epoch: 001 | loss: 0.00000 - acc: 0.0000 -- iter: 00096/50000
                cols = line.split(' ')
                accs.append(cols[10])
    return training_steps, total_losses, accs


def get_data1(file_name):
    training_steps = []
    total_losses = []
    accs = []
    with open(file_name, 'r') as f:
        for line in f.readlines():
            if line.__contains__('Training Step'):
                # [A[ATraining Step: 10944  | total loss: [1m[32m1.44601[0m[0m | time: 0.855s
                cols = line.split(' ')
                # cols[7] = '\x1b[1m\x1b[32m1.49210\x1b[0m\x1b[0m'
                training_step = cols[2]
                if int(training_step) % 50 == 0:
                    total_loss = cols[7].split('\x1b[')[2].split('m')[1]
                    training_steps.append(training_step)
                    total_losses.append(total_loss)
            if line.__contains__('acc') and len(accs) < len(training_steps):
                # | Adam | epoch: 001 | loss: 0.00000 - acc: 0.0000 -- iter: 00096/50000
                cols = line.split(' ')
                accs.append(cols[10])
    return training_steps, total_losses, accs


def plot(datas, colors, labels, title, loc):
    plt.clf()
    for data, color, label in zip(datas, colors, labels):
        plt.plot(data[0], data[1], color, label=label)

    plt.xlabel("krok")
    plt.ylabel(title)
    plt.legend(loc=loc)
    plt.savefig(title + ".png")


t5_training_steps, t5_total_losses, t5_accs = get_data('tensor5.log')
t5_training_steps1, t5_total_losses1, t5_accs1 = get_data1('tensor5.logg')
t5_training_steps += t5_training_steps1
t5_total_losses += t5_total_losses1
t5_accs += t5_accs1

t6_training_steps, t6_total_losses, t6_accs = get_data('tensor6.log')
t6_training_steps1, t6_total_losses1, t6_accs1 = get_data1('tensor6.logg')
t6_training_steps += t6_training_steps1
t6_total_losses += t6_total_losses1
t6_accs += t6_accs1

t7_training_steps, t7_total_losses, t7_accs = get_data1('tensor7.logg')

plot([(t5_training_steps, t5_accs), (t6_training_steps, t6_accs), (t7_training_steps, t7_accs)],
     ['r-', 'b-', 'g-', 'y-'], ['A', 'B', 'C'], 'precyzja', 'lower right')

plot([(t5_training_steps, t5_total_losses), (t6_training_steps, t6_total_losses), (t7_training_steps, t7_total_losses)],
     ['r-', 'b-', 'g-', 'y-'], ['A', 'B', 'C'], 'total_loss', 'upper right')
