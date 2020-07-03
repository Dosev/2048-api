import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

BATCH_SIZE = 4096
LEARNING_RATE = 0.0001

class _2048Dataset(Dataset):
    """2048 dataset."""
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        self.landmarks_frame = pd.read_csv(csv_file,header=None,low_memory=False)
        while self.landmarks_frame.shape[0] % BATCH_SIZE != 0:
            self.landmarks_frame.drop(self.landmarks_frame.index[len(self.landmarks_frame) - 1], inplace=True)
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        label = self.landmarks_frame.iloc[idx, 16]
        tmp_list = [0] * 4
        tmp_list[label] = 1
        label = tmp_list
        label = np.asarray(label)

        boards = self.landmarks_frame.iloc[idx, 0:16].as_matrix().reshape(-1, 4)

        a = []
        for i in range(4):
            b = []
            a.append(b)
        for i in range(4):
            for j in range(4):
                tmp = a
                zero_list = [0] * 16
                zero_list[boards[i][j]] = 1
                tmp[i].append(zero_list)

        board = tmp
        board = np.asarray(board)
        sample = {'board': board, 'label': label}
        return sample

transformed_dataset = _2048Dataset(csv_file='./all_64_256.csv')
#transformed_dataset_test = _2048Dataset(csv_file='./test.csv')
#transformed_dataset_val = _2048Dataset(csv_file='./val.csv')


dataloader = DataLoader(transformed_dataset, batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=0)

#dataloader_test = DataLoader(transformed_dataset_test, batch_size=BATCH_SIZE,
#                        shuffle=False, num_workers=0)

#dataloader_val = DataLoader(transformed_dataset_val, batch_size=BATCH_SIZE,
#                        shuffle=False, num_workers=0)

class NET(nn.Module):
    def __init__(self):
        # Cause there is no "same" padding in Pytorch, so I try to comlete it.
        '''def conv2d_same_padding(input, weight, bias=None, stride=1, dilation=1, groups=1):
            input_rows = input.size(2)
            filter_rows = weight.size(2)
            effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
            out_rows = (input_rows + stride[0] - 1) // stride[0]
            padding_needed = max(0, (out_rows - 1) * stride[0] + effective_filter_size_rows - input_rows)

            padding_rows = max(0, (out_rows - 1) * stride[0] +
                               (filter_rows - 1) * dilation[0] + 1 - input_rows)
            rows_odd = (padding_rows % 2 != 0)
            # same for padding_cols

            if rows_odd or cols_odd:
                input = F.pad(input, [0, int(cols_odd), 0, int(rows_odd)])

            return F.conv2d(input, weight, bias, stride,
                            padding=(padding_rows // 2, padding_cols // 2),
                            dilation=dilation, groups=groups)'''
        #def make_seq_conv(inchannel, outchannel, kernel_size, padding=0, stride=1, relu_inplace=True):
        def make_seq_conv(inchannel, outchannel, kernel_size, padding=0, stride=1, relu_inplace=False):
            # build a conv2d -> bn -> relu sequential
            # by default, Relu is set as inplace=True
            return nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size, padding=padding, stride=stride),
                nn.BatchNorm2d(outchannel),
                nn.ReLU(inplace=relu_inplace)
            )

        #def make_seq_fc(in_features, out_features, drop_ratio=0.5, bias=True, relu_inplace=True):
        def make_seq_fc_relu(in_features, out_features, drop_ratio=0.5, bias=True, relu_inplace=False):
            # build a linear -> bn -> relu -> dropout sequential
            return nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.ReLU(relu_inplace),
                nn.Dropout(drop_ratio)
            )
        def make_seq_fc_softmax(in_features, out_features, drop_ratio=0.5, bias=True):
            # build a linear -> bn -> relu -> dropout sequential
            return nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.Softmax(),
                nn.Dropout(drop_ratio)
            )
        super(NET, self).__init__()
        self.conv41 = nn.Sequential(
            make_seq_conv(4, 128, kernel_size=(4, 1)),
        )
        self.conv14 = nn.Sequential(
            make_seq_conv(4, 128, kernel_size=(1, 4))
        )
        self.conv22 = nn.Sequential(
            make_seq_conv(4, 128, kernel_size=(2, 2))
        )
        self.conv33 = nn.Sequential(
            make_seq_conv(4, 128, kernel_size=(3, 3))
        )
        self.conv44 = nn.Sequential(
            make_seq_conv(4, 128, kernel_size=(4, 4))
        )
        self.fc1 = nn.Sequential(
            make_seq_fc_relu(19712,512)
        )
        self.fc2 = nn.Sequential(
            make_seq_fc_relu(512,128)
        )
        self.fc3 = nn.Sequential(
            make_seq_fc_softmax(128,4)
        )

    def forward(self, x):
        label = x['label']
        x = x['board']
        #print ("board_size" , x.shape)
        x = x.view(BATCH_SIZE,4,4,16)
        x = x.float().cuda()

        conv41 = self.conv41(x)
        #print (conv41.shape)
        conv41 = conv41.view(BATCH_SIZE, -1)
        #print(conv41.shape)
        conv14 = self.conv14(x)
        #print(conv14.shape)
        conv14 = conv14.view(BATCH_SIZE, -1)
        #print(conv14.shape)
        conv22 = self.conv22(x)
        #print(conv22.shape)
        conv22 = conv22.view(BATCH_SIZE, -1)
        #print(conv22.shape)
        conv33 = self.conv33(x)
        #print(conv33.shape)
        conv33 = conv33.view(BATCH_SIZE, -1)
        #print(conv33.shape)
        conv44 = self.conv44(x)
        #print(conv44.shape)
        conv44 = conv44.view(BATCH_SIZE, -1)
        #print(conv44.shape)

        tofc = torch.cat((conv41,conv14,conv22,conv33,conv44),1)

        #print(tofc.shape)
        tofc1 = self.fc1(tofc)
        #print(tofc1.shape)
        tofc2 = self.fc2(tofc1)
        #print(tofc2.shape)
        output = self.fc3(tofc2)
        #print(output.shape)
        #print (output)
        #print (label)

        return  output


net = NET()
#net = nn.DataParallel(net)
net.cuda()
#net.load_state_dict(torch.load('./params3900_64.pkl'))
# Loss and Optimizer
cost = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr=LEARNING_RATE)

# Train the model
epoch = 0
while True:
    epoch = epoch + 1
    num_sample_train = 0
    acc_train = 0.0
    for sample in dataloader:
        sum_loss = 0.0
        label = sample['label'].long()
        label = label.cuda()
        board = sample['board'].float()
        board = board.cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        output = net(sample)
        #print (torch.max(label, 1)[1])
        loss = cost(output, torch.max(label, 1)[1])
        loss.backward()
        optimizer.step()
        for i in range(BATCH_SIZE):
            #print(torch.max(output,1)[1][i])
            #print(torch.max(label, 1)[1][i])
            if torch.max(output,1)[1][i]== torch.max(label, 1)[1][i]:
                acc_train += 1
            #print ("acc_train", acc_train)
        #print ("loss.item", loss.item())
        sum_loss += loss.item()
        num_sample_train = num_sample_train + BATCH_SIZE
        #print ("sum_loss" , sum_loss)
        #print ("num_sample_train" , num_sample_train)
        #print (sum_loss / num_sample_train)
    with open('./naive_res.txt', 'a') as f:
        f.write('[%d] loss_val: %.03f'
                % (epoch + 1,sum_loss))
        f.write('[%d] acc_train: %.03f'
                % (epoch + 1,acc_train / num_sample_train))
        f.write('/n')
    if epoch % 100 == 0:
        torch.save(net.state_dict(), "params" + str(epoch) + ".pkl")
# Save the Trained Model
torch.save(cnn.state_dict(), 'cnn.pkl')
