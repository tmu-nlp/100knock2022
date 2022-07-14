from torch.nn import functional
import optuna


class textCNN(nn.Module):
    def __init__(self, vocab_size, emb_size, padding_idx, output_size,
                 out_channels, conv_params, drop_rate, emb_weights=None):
        super().__init__()
        if emb_weights != None:
            self.emb = nn.Embedding.from_pretrained(
                emb_weights, padding_idx=padding_idx)
        else:
            self.emb = nn.Embedding(
                vocab_size, emb_size, padding_idx=padding_idx)
        self.convs = nn.ModuleList([nn.Conv2d(1, out_channels, (kernel_height, emb_size), padding=(
            padding, 0)) for kernel_height, padding in conv_params])
        self.drop = nn.Dropout(drop_rate)
        self.fc = nn.Linear(len(conv_params) * out_channels, output_size)

    def forward(self, x):
        emb = self.emb(x).unsqueeze(1)
        conv = [functional.relu(conv(emb)).squeeze(3)
                for _, conv in enumerate(self.convs)]
        max_pool = [functional.max_pool1d(i, i.size(2)) for i in conv]
        max_pool_cat = torch.cat(max_pool, 1)
        out = self.fc(self.drop(max_pool_cat.squeeze(2)))
        return out


def objective(trial):
    # チューニング対象パラメータのセット
    emb_size = int(trial.suggest_discrete_uniform('emb_size', 100, 400, 100))
    out_channels = int(trial.suggest_discrete_uniform(
        'out_channels', 50, 200, 50))
    drop_rate = trial.suggest_discrete_uniform('drop_rate', 0.0, 0.5, 0.1)
    learning_rate = trial.suggest_loguniform('learning_rate', 5e-4, 5e-2)
    momentum = trial.suggest_discrete_uniform('momentum', 0.5, 0.9, 0.1)
    batch_size = int(trial.suggest_discrete_uniform('batch_size', 16, 128, 16))

    # パラメータ
    VOCAB_SIZE = len(set(word_id.values())) + 1
    PADDING_IDX = len(set(word_id.values()))
    OUTPUT_SIZE = 4
    CONV_PARAMS = [[2, 0], [3, 1], [4, 2]]
    NUM_EPOCHS = 30

    # モデル定義
    model = textCNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE,
                    out_channels, CONV_PARAMS, drop_rate, emb_weights=weights)

    # 損失関数
    criterion = nn.CrossEntropyLoss()

    # SGD
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=momentum)

    device = torch.device('cuda')

    # 学習
    log = train_model(dataset_train, dataset_valid, batch_size, model, criterion,
                      optimizer, NUM_EPOCHS, collate_fn=Padsequence(PADDING_IDX), device=device)

    # 損失の算出
    loss_valid, _ = calculate_loss_and_accuracy(
        model, dataset_valid, device, criterion=criterion)

    return loss_valid, log


# 最適化
study = optuna.create_study()
study.optimize(objective, timeout=7200)

# 結果の表示
print('Best trial:')
trial = study.best_trial
print('Value: {:.3f}'.format(trial.value))
print('Params: ')
for key, value in trial.params.items():
    print('\t{}: {}'.format(key, value))
