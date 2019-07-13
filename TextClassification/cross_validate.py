from conf import config
from dataloader.Dataloader import load_dataset
from vocab.Vocab import create_vocab, create_wc_vocab
import torch
# from module.bilstm import BiLSTM
from module.bilstm_char import BiLSTM
import numpy as np
from classifier import Classifier


if __name__ == '__main__':
    # 设置随机种子(固定随机值)
    np.random.seed(666)
    torch.manual_seed(6666)
    torch.cuda.manual_seed(1234)

    print('GPU available: ', torch.cuda.is_available())
    print('CuDNN available: ', torch.backends.cudnn.enabled)
    print('GPU number: ', torch.cuda.device_count())

    # 设置参数(数据参数+模型参数)
    data_opts = config.data_path_parse('./conf/data_path.json')
    args = config.arg_parse()

    # args.use_cuda = torch.cuda.is_available()
    # if args.use_cuda and args.enable_cuda:
    #     torch.cuda.set_device(args.cuda)

    if args.enable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda', args.cuda)
    else:
        args.device = torch.device('cpu')

    print(args.device)

    # 加载数据(训练集-学习、开发集-调参、测试集-评估)
    train_data = load_dataset(data_opts['data']['train_data'])
    test_data = load_dataset(data_opts['data']['test_data'])
    print('train_size=%d test_size=%d' % (len(train_data), len(test_data)))

    # 创建词表
    # vocab = create_vocab(data_opts['data']['train_data'])
    vocab, char_vocab = create_wc_vocab(data_opts['data']['train_data'])
    embedding_weights = vocab.get_embedding_weights(data_opts['data']['embedding_path'])
    # vocab.save_vocab(data_opts['model']['save_vocab_path'])

    # 构建分类模型
    # model = TexCNN(args, vocab, embedding_weights).to(args.device)
    model = BiLSTM(args, vocab, char_vocab, embedding_weights).to(args.device)
    classifier = Classifier(model, args, vocab, char_vocab)
    classifier.summary()

    # 10-folds 交叉验证
    classifier.cross_validate(model, train_data, folds=10)

    # 测试
    # classifier.evaluate(test_data)
