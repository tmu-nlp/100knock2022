# パラメータの設定
VOCAB_SIZE = len(set(word_id.values())) + 1
EMB_SIZE = 300
PADDING_IDX = len(set(word_id.values()))
OUTPUT_SIZE = 4
OUT_CHANNELS = 100
KERNEL_HEIGHTS = 3
STRIDE = 1
PADDING = 1
LEARNING_RATE = 5e-2
BATCH_SIZE = 64
NUM_EPOCHS = 10

# モデル
model = CNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE,
            OUT_CHANNELS, KERNEL_HEIGHTS, STRIDE, PADDING, emb_weights=weights)

# 損失関数
criterion = nn.CrossEntropyLoss()

# オプティマイザ
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

# cuda
device = torch.device('cuda')

# モデルの学習
log = train_model(dataset_train, dataset_valid, BATCH_SIZE, model, criterion,
                  optimizer, NUM_EPOCHS, collate_fn=Padsequence(PADDING_IDX))

# 損失ログの可視化
plt.plot(np.array(log['train']).T[0], label='train')
plt.plot(np.array(log['valid']).T[0], label='valid')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig('./results/output87_loss.png')
plt.show()

# 正解率ログの可視化
plt.plot(np.array(log['train']).T[1], label='train')
plt.plot(np.array(log['valid']).T[1], label='valid')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.savefig('./results/output87_accuracy.png')
plt.show()

# 正解率の算出
_, acc_train = calculate_loss_and_accuracy(model, dataset_train)
_, acc_test = calculate_loss_and_accuracy(model, dataset_test)
print(f'正解率（学習データ）: {acc_train:.3f}')
print(f'正解率（評価データ）: {acc_test:.3f}')

'''
正解率（学習データ）: 0.743
正解率（評価データ）: 0.702
'''
