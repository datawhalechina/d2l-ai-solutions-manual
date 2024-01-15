# 更换更大的预训练词向量 300维
import os
import torch
from torch import nn
from d2l import torch as d2l  # 需要预先下载好d2l包

if torch.cuda.is_available():
    print("CUDA is available!")
else:
    print("CUDA is not available.")


def load_data():
    batch_size = 64
    train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size)  # 加载相关数
    return train_iter, test_iter, vocab


def load_glove_model():
    # 相关GloVe模型地址
    d2l.DATA_HUB['glove.6b.50d'] = (d2l.DATA_URL + 'glove.6B.50d.zip',
                                    '0b8703943ccdb6eb788e6f091b8946e82231bc4d')
    d2l.DATA_HUB['glove.6b.100d'] = (d2l.DATA_URL + 'glove.6B.100d.zip',
                                     'cd43bfb07e44e6f27cbcc7bc9ae3d80284fdaf5a')
    d2l.DATA_HUB['glove.42b.300d'] = (d2l.DATA_URL + 'glove.42B.300d.zip',
                                      'b5116e234e9eb9076672cfeabf5469f3eec904fa')
    d2l.DATA_HUB['wiki.en'] = (d2l.DATA_URL + 'wiki.en.zip',
                               'c1816da3821ae9f43899be655002f6c723e91b88')


# 加载预训练的词向量，这部分与官方保持一致。
class TokenEmbedding:
    """GloVe嵌入"""

    def __init__(self, embedding_name):
        self.idx_to_token, self.idx_to_vec = self._load_embedding(
            embedding_name)
        self.unknown_idx = 0
        self.token_to_idx = {token: idx for idx, token in
                             enumerate(self.idx_to_token)}

    def _load_embedding(self, embedding_name):
        idx_to_token, idx_to_vec = ['<unk>'], []
        data_dir = d2l.download_extract(embedding_name)
        # GloVe网站：https://nlp.stanford.edu/projects/glove/
        # fastText网站：https://fasttext.cc/
        with open(os.path.join(data_dir, 'vec.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                # 跳过标题信息，例如fastText中的首行
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, torch.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [self.token_to_idx.get(token, self.unknown_idx)
                   for token in tokens]
        vecs = self.idx_to_vec[torch.tensor(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)


# 定义循环神经网络，这部分与官方保持一致。
class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens,
                 num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 将bidirectional设置为True以获取双向循环神经网络
        self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers,
                               bidirectional=True)
        self.decoder = nn.Linear(4 * num_hiddens, 2)

    def forward(self, inputs):
        # inputs的形状是（批量大小，时间步数）
        # 因为长短期记忆网络要求其输入的第一个维度是时间维，
        # 所以在获得词元表示之前，输入会被转置。
        # 输出形状为（时间步数，批量大小，词向量维度）
        embeddings = self.embedding(inputs.T)
        self.encoder.flatten_parameters()
        # 返回上一个隐藏层在不同时间步的隐状态，
        # outputs的形状是（时间步数，批量大小，2*隐藏单元数）
        outputs, _ = self.encoder(embeddings)
        # 连结初始和最终时间步的隐状态，作为全连接层的输入，
        # 其形状为（批量大小，4*隐藏单元数）
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)
        outs = self.decoder(encoding)
        return outs


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.LSTM:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])


def train_100d(train_iter, test_iter, vocab):
    embed_size, num_hiddens, num_layers = 100, 100, 2
    net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)
    net.apply(init_weights)
    devices = d2l.try_all_gpus()

    # 官方示例里，词表中的单词加载预训练的100维（需要与embed_size一致）的GloVe嵌入。
    glove_embedding_100 = TokenEmbedding('glove.6b.100d')

    embeds = glove_embedding_100[vocab.idx_to_token]
    net.embedding.weight.data.copy_(embeds)
    net.embedding.weight.requires_grad = False

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 防止部分环境内存报错

    # 训练和评估模型
    lr, num_epochs = 0.01, 5
    net.apply(init_weights)  # 若需多次运行这个cell，需要添加该句语句进行权重刷新
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction="none")
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
                   devices)


# 修改后，我们为词表中的单词加载预训练的300维（需要与embed_size一致）的GloVe嵌入。
def train_300d(train_iter, test_iter, vocab):
    embed_size, num_hiddens, num_layers = 300, 100, 2
    net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)
    net.apply(init_weights)
    devices = d2l.try_all_gpus()

    glove_embedding_300 = TokenEmbedding('glove.42b.300d')

    embeds = glove_embedding_300[vocab.idx_to_token]
    net.embedding.weight.data.copy_(embeds)
    net.embedding.weight.requires_grad = False

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 防止部分环境内存报错

    # 训练和评估模型
    lr, num_epochs = 0.01, 5
    net.apply(init_weights)  # 若需多次运行这个cell，需要添加该句语句进行权重刷新
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction="none")
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
                   devices)


if __name__ == '__main__':
    train_iter, test_iter, vocab = load_data()
    load_glove_model()
    train_100d(train_iter, test_iter, vocab)
    train_300d(train_iter, test_iter, vocab)
