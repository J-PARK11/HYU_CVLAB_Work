"""
Configuration for Various Task.
Define independent classes by task.
"""

import argparse

# Video Frame interpolation Transformer
class raft_config():
    """
    args
        >> data_arg
        >> model_arg
        >> learn_arg
        >> option_arg
    """
    def __init__(self):
        self.args_list = []
        self.parser = argparse.ArgumentParser(description='Raft Kernel')
        
        self.root_arg = self.add_argument_group('root')
        self.model_arg = self.add_argument_group('model')
        self.learn_arg = self.add_argument_group('learning')
        self.set_arg = self.add_argument_group('set-up')

        # Root parameters
        self.root_arg.add_argument('--data_root', default='./data/', type=str)
        self.root_arg.add_argument('--out_root', default='./output/temp/', type=str)
        self.root_arg.add_argument('--tensorboard_root', default='./tensorboard/', type=str)
        self.root_arg.add_argument('--checkpoint_root', default='./checkpoint/', type=str)
        
        # Model parameters
        self.model_arg.add_argument('--flow_model', default='raft_large', choices=['raft_large', 'raft_small'])
        self.model_arg.add_argument('--depth_model', default='DPT_Large', choices=['DPT_Large', 'DPT_Hybrid', 'MiDaS_small'])
        self.model_arg.add_argument('--softsplat', default=False, type=bool)

        # Train & Test Parameters
        self.learn_arg.add_argument('--epochs', default=2, type=int)
        self.learn_arg.add_argument('--checkpoint_epoch', default=1, type=int)
        self.learn_arg.add_argument('--crop_size', default=(512, 960))
        self.learn_arg.add_argument('--log_iter', default=1000, type=int)
        self.learn_arg.add_argument('--batch_size', default=6, type=int)
        self.learn_arg.add_argument('--test_batch_size', default=6, type=int)
        self.learn_arg.add_argument('--loss', type=str, default='1*l1')
        self.learn_arg.add_argument("--load_from"  ,type=str , default='checkpoint/model_best.pth')

        # Set-up parameters
        self.set_arg.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
        self.set_arg.add_argument('--use_gpu', default=True, type=bool)
        self.set_arg.add_argument('--gpu', default=0, type=int)
        self.set_arg.add_argument('--seed', default=1123, type=int)
        self.set_arg.add_argument('--num_workers', default=4, type=int)
    
    def add_argument_group(self, name):
        arg = self.parser.add_argument_group(name)
        self.args_list.append(arg)
        return arg

    def feed_args(self):
        self.args, self.unparsed = self.parser.parse_known_args()
        setattr(self.args, 'cuda', True)
        if len(self.unparsed) > 1:
            print("Unparsed args: {}".format(self.unparsed))
        return self.args, self.unparsed

# =========================================================== #

def get_args(task):
    if task == 'raft':
        cfg = raft_config()
        args, unparsed = cfg.feed_args()
    else:
        raise NotImplementedError
    
    print(f'\nConfig: {args}')    # print(f'Unparsed: {unparsed}')
    return args, unparsed

if __name__ == "__main__":
    task = 'raft'
    args, unparsed = get_args(task)



