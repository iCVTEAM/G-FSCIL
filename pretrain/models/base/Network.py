import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet18_encoder import *
from models.resnet20_cifar import *
from tqdm import tqdm


def normalize(x):
    norm = x.pow(2).sum(2, keepdim=True).pow(1. / 2)
    out = x.div(norm)
    return out


class MYNET(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode
        self.args = args

        self.resolution = 25
        # self.num_features = 512
        if self.args.dataset in ['cifar100']:
            self.encoder = resnet20()
            self.num_features = 64
        if self.args.dataset in ['mini_imagenet']:
            self.encoder = resnet18(False, args)  # pretrained=False
            self.num_features = 512
        if self.args.dataset == 'cub200':
            self.encoder = resnet18(True,
                                    args)  # pretrained=True follow TOPIC, models for cub is imagenet pre-trained. https://github.com/xyutao/fscil/issues/11#issuecomment-687548790
            self.num_features = 512
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        self.num_classes = self.args.num_classes
        # FRN params
        self.scale = nn.Parameter(torch.FloatTensor([25.0]), requires_grad=True)
        self.is_pretraining = False  # default set as session 0; False for session 1~10

        self.r = nn.Parameter(torch.zeros(2), requires_grad=not self.is_pretraining)

        # old fc not used.
        # self.fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)

        # base prototype

        self.proto = nn.Parameter(torch.randn(self.args.num_classes, self.resolution, self.num_features),
                                  requires_grad=True)

        self.cross_entropy = nn.CrossEntropyLoss()

    def reinit_params(self):
        self.scale = nn.Parameter(torch.FloatTensor([16.0]), requires_grad=True)
        self.is_pretraining = True  # default set as session 0; False for session 1~10
        self.r = nn.Parameter(torch.zeros(2), requires_grad=not self.is_pretraining)
        self.proto = nn.Parameter(torch.randn(self.args.num_classes, self.resolution, self.num_features),
                                  requires_grad=True)

    def get_feature_map(self, x):
        x = self.encoder(x)
        x = self.avgpool(x)
        # resolution b,c,r,r; r = 5
        feature_map = x /np.sqrt(self.num_features) #self.resolution #np.sqrt(self.num_features)

        feature_map = feature_map.view(x.size(0), self.num_features, -1).permute(0, 2, 1).contiguous()
        #normalize(feature_map)
        return feature_map  # N,HW,C



    def update_prototypes(self, dataloader, class_list, session):
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            data = self.encode(data).detach()
        # b,c,r r vectors. mean in the batch dimensions

        new_prototype = []

        for class_index in class_list:
            data_index = (label == class_index).nonzero().squeeze(-1)
            embedding = data[data_index]
            proto = embedding.mean(0)
            # overall mean on the batch dimensions
            new_prototype.append(proto)
            self.proto.data[class_index] = proto
        new_prototype = torch.stack(new_prototype, dim=0)

        if 'ft' in self.args.new_mode:  # further finetune
            self.update_proto_ft(new_prototype, data, label, session)

    def get_recon_dist(self, query, support, alpha, beta, verbose=False):
        # query: way*query_shot*resolution, d
        # support: way, shot*resolution , d             shot =1 i.e. way, resolution, d
        lam = alpha.exp()*(support.size(1) / support.size(2))+1e-6
        rho = beta.exp()

        support_trans = support.permute(0, 2, 1)  # way, num_features, shot * resolution
        support_square = support.matmul(support_trans)

        proj_tmp = support_square + torch.eye(support_square.size(-1)).to(support_square.device) \
            .unsqueeze(0).mul(lam)

        proj_tmp_np = proj_tmp.detach().cpu().numpy()
        proj_tmp_inv_np = np.linalg.inv(proj_tmp_np)
        proj_tmp_inv = torch.tensor(proj_tmp_inv_np).cuda()

        proj_weight = query.matmul(support_trans.matmul(proj_tmp_inv))  # classes, num_features, num_features

        query_proj = proj_weight.matmul(support).mul(rho)  # classes, classes* query_shot * resolution, num_features

        if verbose:
            pass

        dist = (query_proj - query.unsqueeze(0)).pow(2).sum(2).permute(1,
                                                                       0)  # classes_batch* shot(batch) * resolution, classes

        return dist

    def forward(self, x, target=None):
        # target is label
        feature_map = self.get_feature_map(x)
        batch_size = feature_map.size(0)

        # review
        feature_map = feature_map.view(batch_size * self.resolution, self.num_features)
        
        alpha = self.r[0]
        beta = self.r[1]

        recon_dist = self.get_recon_dist(query=feature_map, support=self.proto, alpha= alpha, beta=beta, verbose=False)
        logits = recon_dist.neg().view(batch_size, self.resolution, self.num_classes).mean(1)

        logits = logits * self.scale  # can be controlled

        # _, max_index = torch.max(logits, 1)

        # loss = self.cross_entropy(logits, target)

        return logits

    def proto_measure(self, feature_map, support_proto, current_class_num):
        # target is label
        batch_size = feature_map.size(0)
        #print(feature_map.size())

        # review
        feature_map = feature_map.view(batch_size * self.resolution, self.num_features)
        alpha = self.r[0]
        beta = self.r[1]

        recon_dist = self.get_recon_dist(query=feature_map, support=support_proto,alpha= alpha, beta=beta, verbose=False)
        logits = recon_dist.neg().view(batch_size, self.resolution, current_class_num).mean(1)

        logits = logits * self.scale  # can be controlled

        # _, max_index = torch.max(logits, 1)

        # loss = self.cross_entropy(logits, target)

        return logits

    def frn_finetune_meta(self, dataloader, class_list, session):
        """
        class_list: only contains classes seen during this session. without base classes.
        """
        second = 0
        for batch in dataloader:
            data_tmp, label = [_ for _ in batch]
            # data = self.get_feature_map(data).detach()
            # N,HW,C  
            # only one batch???
            second = second + 1
            if second > 1:
                assert ("Wrong configuration for FRN finetune")
        # sort the labels
        label, lbl_idxes = torch.sort(label, descending=True)
        data = data_tmp.index_select(0, lbl_idxes)

        data = data.cuda()
        label = label.cuda()

        num_samples = data.size(0)
        shot = data.size(0) // len(class_list)

        part_num = 1  # query part number

        indexes = torch.randperm(num_samples)
        s_idxes = indexes[part_num:]
        q_idxes = indexes[:part_num]
        data_query = data[q_idxes, :].view()
        new_proto = self.update_newproto(data, label, class_list)

        if 'ft' in self.args.new_mode:  # further finetune
            self.update_proto_ft(new_proto, data, label, session)
        assert False
        # not finished

    def frn_finetune(self, dataloader, train_transform, test_transform, class_list, session):
        second = 0
        dataloader.dataset.transform = test_transform

        embedding_list = []
        label_list = []
        # data_list=[]
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                data, label = [_.cuda() for _ in batch]
                #model.module.mode = 'encoder'
                embedding = self.get_feature_map(data) # N,HW,C

                embedding_list.append(embedding)
                label_list.append(label)
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        '''
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            data = self.encode(data).detach()
            second = second + 1
            if second > 1:
                assert ("Wrong configuration for FRN finetune")
        '''
        new_proto = self.update_newproto(embedding_list, label_list, class_list)

        #if 'ft' in self.args.new_mode:  # further finetune
        #    assert False
        #dataloader.dataset.transform = train_transform
        self.update_proto_ft(new_proto, dataloader, label, session)

    def update_newproto(self, data, label, class_list):
        new_proto = []
        for class_index in class_list:
            data_index = (label == class_index).nonzero().squeeze(-1)
            embedding = data[data_index]
            proto_tmp = embedding.mean(0)
            new_proto.append(proto_tmp)
            self.proto.data[class_index] = proto_tmp
        new_proto = torch.stack(new_proto, dim=0)
        return new_proto

    def update_proto_ft(self, new_proto, dataloader, label, session):
        new_proto = new_proto.clone().detach()
        new_proto.requires_grad = True
        optimized_parameters = [{'params': new_proto}]
        #optimizer = torch.optim.SGD(optimized_parameters, lr=self.args.lr_new, momentum=0.9, dampening=0.9,
        #                            weight_decay=0)
        optimizer = torch.optim.SGD(optimized_parameters, lr=self.args.lr_new, weight_decay=5e-4)
        current_class_num = self.args.base_class + self.args.way * (session)



        with torch.enable_grad():
            old_proto = self.proto.data[:self.args.base_class + self.args.way * (session - 1), :].detach()
            proto = torch.cat([old_proto, new_proto], dim=0)
            for epoch in tqdm(range(self.args.epochs_new)):
                for i, batch in enumerate(dataloader):
                    data, label = [_.cuda() for _ in batch]
                    embedding = self.get_feature_map(data).detach() # N,HW,C                
                
                    logits = self.proto_measure(embedding, proto,current_class_num)
                    loss = F.cross_entropy(logits, label)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                

        self.proto.data[
        self.args.base_class + self.args.way * (session - 1):self.args.base_class + self.args.way * session, :].copy_(
            new_proto.data)

    ##### original implementation
    def forward_old(self, input):
        if self.mode != 'encoder':
            input = self.forward_metric(input)
            return input
        elif self.mode == 'encoder':
            input = self.encode(input)
            return input
        else:
            raise ValueError('Unknown mode')

    def update_fc(self, dataloader, class_list, session):
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            data = self.encode(data).detach()

        if self.args.not_data_init:
            new_fc = nn.Parameter(
                torch.rand(len(class_list), self.num_features, device="cuda"),
                requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            new_fc = self.update_fc_avg(data, label, class_list)

        if 'ft' in self.args.new_mode:  # further finetune
            self.update_fc_ft(new_fc, data, label, session)

    def update_fc_avg(self, data, label, class_list):
        new_fc = []
        for class_index in class_list:
            data_index = (label == class_index).nonzero().squeeze(-1)
            embedding = data[data_index]
            proto = embedding.mean(0)
            new_fc.append(proto)
            self.fc.weight.data[class_index] = proto
        new_fc = torch.stack(new_fc, dim=0)
        return new_fc

    def get_logits(self, x, fc):
        if 'dot' in self.args.new_mode:
            return F.linear(x, fc)
        elif 'cos' in self.args.new_mode:
            return self.args.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))

    def update_fc_ft(self, new_fc, data, label, session):
        new_fc = new_fc.clone().detach()
        new_fc.requires_grad = True
        optimized_parameters = [{'params': new_fc}]
        optimizer = torch.optim.SGD(optimized_parameters, lr=self.args.lr_new, momentum=0.9, dampening=0.9,
                                    weight_decay=0)

        with torch.enable_grad():
            for epoch in range(self.args.epochs_new):
                old_fc = self.fc.weight[:self.args.base_class + self.args.way * (session - 1), :].detach()
                fc = torch.cat([old_fc, new_fc], dim=0)
                logits = self.get_logits(data, fc)
                loss = F.cross_entropy(logits, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pass

        self.fc.weight.data[
        self.args.base_class + self.args.way * (session - 1):self.args.base_class + self.args.way * session, :].copy_(
            new_fc.data)

    def forward_metric(self, x):
        x = self.encode(x)
        if 'cos' in self.mode:
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x = self.args.temperature * x

        elif 'dot' in self.mode:
            x = self.fc(x)

        return x

    def encode(self, x):
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x
