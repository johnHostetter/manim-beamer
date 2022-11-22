import torch

from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


"""
The following is code for training, which was inspired by a mix of the 
[PyTorch training tutorial](https://pytorch.org/tutorials/beginner/introyt/trainingyt.html) 
and the [Offline RL tutorial](https://colab.research.google.com/drive/1oJOYlAIOl9d1JjlutPY66KmfPkwPCgEE?usp=sharing).
"""


def make_target_q_values(model, transition, gamma):
    q_values = model.predict(transition['state']).float()
    q_values_next = model.predict(transition['next state'])
    # to index with action_indices, we need to convert to long data type
    action_indices = transition['action'].long()
    rewards = transition['reward'].float()
    terminals = transition['terminal']
    # detach first and then clone is slightly more efficient than clone first and then detach
    # https://stackoverflow.com/questions/55266154/pytorch-preferred-way-to-copy-a-tensor
    q_values[torch.arange(len(q_values)), action_indices.long()] = rewards.detach().clone()
    max_q_values_next, max_q_values_next_indices = torch.max(q_values_next, dim=1)
    max_q_values_next *= gamma
    q_values[torch.arange(len(q_values)), action_indices] += torch.tensor(
        [0.0 if done else float(max_q_values_next[idx]) for idx, done in enumerate(terminals)])
    return transition['state'], q_values, action_indices


def train_one_epoch(training_loader, model, epoch_index, tb_writer, gamma):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, transition in enumerate(training_loader):
        # Every transition is {'state', 'action', 'reward', 'next state', 'terminal'}

        # Zero your gradients for every batch!
        model.zero_grad()

        states, target_q_values, action_indices = make_target_q_values(model, transition, gamma)

        loss = model.offline_update(transition['state'], target_q_values, action_indices)

        # Gather data and report
        running_loss += loss
        if i % 10 == 0:
            last_loss = running_loss / 10  # loss per batch
            # print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


def offline_q_learning(model, training_dataset, validation_dataset, max_epochs=100, batch_size=64, gamma=0.9):
    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0

    best_vloss = 1_000_000.
    last_avg_vloss = float('inf')
    for epoch in range(max_epochs):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(train_loader, model, epoch_number, writer, gamma)

        # We don't need gradients on to do reporting
        model.train(False)

        running_vloss = 0.0
        for i, validation_transition in enumerate(val_loader):
            _, target_q_values, action_indices = make_target_q_values(model, validation_transition, gamma)
            vloss = model.compute_loss(validation_transition['state'], target_q_values, action_indices)
            running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)

        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch_number + 1)
        writer.flush()

        # if avg_vloss < last_avg_vloss:
        #     last_avg_vloss = avg_vloss
        # else:
        #     # the model is becoming worse, early terminate
        #     return model, [avg_loss], [avg_vloss]

        # Track the best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            # model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            # torch.save(model.state_dict(), model_path)

        epoch_number += 1
    return model, [avg_loss], [avg_vloss]
