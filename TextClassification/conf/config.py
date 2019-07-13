import os
import argparse
import json
import logging

# debug info warning error
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)', level=logging.INFO)


# 语料路径解析
def data_path_parse(path):
    assert os.path.exists(path)
    with open(path, 'r', encoding='utf-8') as fin:
        opts = json.load(fin)  # 读取json文件，写json文件是json.dump()

    # logging.info(opts)
    print(opts)

    return opts


# 参数解析
def arg_parse():
    parser = argparse.ArgumentParser(description="CNN Arguments Configuration")
    # 通用配置
    parser.add_argument('--cuda', type=int, default=-1, help='-1 means train on CPU')
    # parser.add_argument('--use_cuda', type=bool, default=True, help='use GPU or not')
    parser.add_argument('--enable_cuda', type=bool, default=True, help='enable GPU or not')

    # 配置数据参数
    parser.add_argument('-bz', '--batch_size', type=int, default=32, help='the size of per batch')
    parser.add_argument('-ep', '--epochs', type=int, default=12, help='the number of iter')

    # 配置模型参数
    parser.add_argument('-hz', '--hidden_size', type=int, default=200, help='the size of hidden layer')
    parser.add_argument('--nb_layer', type=int, default=2, help='the number of hidden layer')
    parser.add_argument('--embed_dropout', type=float, default=0.5, help='the dropout of embedding layer')
    # *rnn_dropout对结果影响较大
    parser.add_argument('--rnn_dropout', type=float, default=0.0, help='the dropout of recurrent layer')
    parser.add_argument('--linear_dropout', type=float, default=0.5, help='the dropout of linear layer')
    # char embedding
    parser.add_argument('-chz', '--char_hidden_size', type=int, default=200, help='the size of char embedding layer')
    parser.add_argument('--char_embed_dim', type=int, default=50, help='the initialed char embedding size')

    # 配置优化器参数
    parser.add_argument('-lr', '--learning_rate', type=float, default=3e-3, help='leaning rate in training')
    # 权值衰减系数，L2正则化参数
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-7, help='weight decay')

    args = parser.parse_args()

    print(vars(args))

    return args
