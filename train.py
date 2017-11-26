from collections import ChainMap
import argparse
import random

from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import model as mod

class ConfigBuilder(object):
    def __init__(self, *default_configs):
        self.default_config = ChainMap(*default_configs)

    def build_argparse(self):
        parser = argparse.ArgumentParser()
        for key, value in self.default_config.items():
            key = "--{}".format(key)
            if isinstance(value, tuple):
                parser.add_argument(key, default=list(value), nargs=len(value), type=type(value[0]))
            elif isinstance(value, list):
                parser.add_argument(key, default=value, nargs="+", type=type(value[0]))
            elif isinstance(value, bool) and not value:
                parser.add_argument(key, action="store_true")
            else:
                parser.add_argument(key, default=value, type=type(value))
        return parser

    def config_from_argparse(self, parser=None):
        if not parser:
            parser = self.build_argparse()
        args = vars(parser.parse_known_args()[0])
        return ChainMap(args, self.default_config)

def print_eval(name, scores, labels, loss, end="\n"):
    batch_size = labels.size(0)
    accuracy = (torch.round(F.sigmoid(scores)).data == labels.data).sum() / (labels.size(1) * batch_size)
    loss = loss.cpu().data.numpy()[0]
    print("{} accuracy: {:>5}, loss: {:<25}".format(name, accuracy, loss), end=end)
    return accuracy

def collate_fn(examples):
    max_len = max(examples, key=lambda x: x[0].size(0))[0].size(0)
    collated_data = []
    collated_labels = []
    for data, label in examples:
        data = torch.cat([data, torch.zeros(max_len - data.size(0), 40)], 0)
        collated_data.append(data)
        collated_labels.append(label)
    return torch.stack(collated_data, 0), torch.stack(collated_labels, 0)

class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.model = mod.ConvRNNModel(config)
        self.criterion = nn.BCEWithLogitsLoss()

        self.train_set, self.dev_set, self.test_set = mod.AIShellDataset.splits(config)
        self.train_loader = data.DataLoader(self.train_set, batch_size=config["batch_size"], shuffle=True, 
            drop_last=True, collate_fn=collate_fn)
        self.dev_loader = data.DataLoader(self.dev_set, batch_size=min(len(self.dev_set), 32), shuffle=False, collate_fn=collate_fn)
        self.test_loader = data.DataLoader(self.test_set, batch_size=min(len(self.test_set), 32), shuffle=False, collate_fn=collate_fn)

        self.step_no = 0
        self.schedule_steps = config["schedule"]
        self.schedule_steps.append(np.inf)
        if config["input_file"]:
            self.model.load(config["input_file"])
        self.model.cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config["lr"][0], momentum=config["momentum"], 
            weight_decay=config["weight_decay"])

    def step(self, train_data, evaluate=False, out=None, name="train"):
        if evaluate:
            self.model.eval()
        else:
            self.model.train()
            self.optimizer.zero_grad()
        model_in, labels = train_data
        labels = Variable(labels, requires_grad=False).cuda()
        model_in = Variable(model_in, requires_grad=False).cuda()

        scores = self.model(model_in).cuda()
        loss = self.criterion(scores, labels)
        if not evaluate:
            loss.backward()
            self.optimizer.step()
        val = print_eval(name + " step #{}".format(self.step_no), scores, labels, loss)
        if out is not None:
            out.append(val)

    def evaluate(self):
        metrics = []
        for test_data in self.test_loader:
            self.step(test_data, evaluate=True, out=metrics, name="test")
        print("final test measure: {}".format(np.mean(metrics)))

    def train(self):
        sched_idx = 0
        max_measure = 0
        config = self.config

        for epoch_idx in range(config["n_epochs"]):
            for batch_idx, train_data in enumerate(self.train_loader):
                self.step(train_data)
                self.step_no += 1

                if self.step_no > self.schedule_steps[sched_idx]:
                    sched_idx += 1
                    print("changing learning rate to {}".format(config["lr"][sched_idx]))
                    self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config["lr"][sched_idx],
                        momentum=config["momentum"], weight_decay=config["weight_decay"])

                if self.step_no % config["dev_every"] == config["dev_every"] - 1:
                    metrics = []
                    for dev_data in self.dev_loader:
                        self.step(dev_data, evaluate=True, out=metrics, name="dev")
                    avg_measure = np.mean(metrics)
                    print("final dev measure: {}".format(avg_measure))
                    if avg_measure > max_measure:
                        print("saving best model...")
                        max_measure = avg_measure
                        self.model.save(config["output_file"])

    def run(self):
        self.train()
        self.evaluate()

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def main():
    global_config = dict(n_epochs=500, lr=[0.001], schedule=[np.inf], batch_size=64, dev_every=200, seed=0,
        input_file="", output_file="output.pt", gpu_no=0, cache_size=32768, momentum=0.9, weight_decay=0.0001)
    builder = ConfigBuilder(
        mod.ConvRNNModel.default_config(),
        mod.AIShellDataset.default_config(),
        global_config)
    config = builder.config_from_argparse()
    set_seed(config["seed"])
    trainer = Trainer(config)
    trainer.run()

if __name__ == "__main__":
    main()