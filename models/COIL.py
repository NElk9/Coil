import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import IncrementalNet,CosineIncrementalNet,SimpleCosineIncrementalNet
from utils.toolkit import target2onehot, tensor2numpy
import ot
from torch import nn
import copy
EPSILON = 1e-8

epochs = 160
lrate = 0.1
milestones = [80, 120]
lrate_decay = 0.1
batch_size = 128
memory_size = 2000
T = 2



class COIL(BaseLearner):

    def __init__(self, args):
        super().__init__()
        self._network = SimpleCosineIncrementalNet(args['convnet_type'], False) # 余弦分类器网络，分类器采用 “余弦相似度” 替代传统内积，通过归一化特征和权重，增强类别边界的稳定性（适合增量学习）
        self._device = args['device']           # 设备（GPU/CPU）
        self.data_manager=None                  # 数据管理器（后续初始化）
        self.nextperiod_initialization=None     # 新类别分类器的OT初始化权重（基于旧分类器通过OT生成）
        self.sinkhorn_reg=args['sinkhorn']      # Sinkhorn算法的正则化系数（OT求解用）
        self.calibration_term=args['calibration_term']  # 分类器权重校准系数，校准新旧分类器的权重尺度，避免分布差异影响迁移效果
        self.args=args        # 其他参数

    # 前瞻传输（Prospective Transport, PT）,通过after_task和solving_ot函数实现
    # 当前任务结束后准备下一轮初始化
    def after_task(self):
        self.nextperiod_initialization=self.solving_ot()    # 用OT生成下一轮新分类器的初始权重
        self._old_network = self._network.copy().freeze()   # 冻结当前网络作为旧模型（用于后续蒸馏）
        self._known_classes = self._total_classes           # 更新已知类别数量

    # 用OT生成下一轮新分类器的初始权重
    def solving_ot(self):
        with torch.no_grad():
            if self._total_classes==self.data_manager.get_total_classnum():
                print('training over, no more ot solving')
                return None
            each_time_class_num=self.data_manager.get_task_size(1)

            # 提取旧类别和新类别的特征中心（用于计算OT的成本矩阵）
            self._extract_class_means(self.data_manager,0,self._total_classes+each_time_class_num)  # 内部函数：计算每个类别的特征均值（类别中心）
            former_class_means=torch.tensor(self._ot_prototype_means[:self._total_classes])             # 旧类别中心
            next_period_class_means=torch.tensor(self._ot_prototype_means[self._total_classes:self._total_classes+each_time_class_num]) # 新类别中心（从数据中提取）

            # 计算成本矩阵：旧-新类别中心的距离（距离越小，语义关联越强）
            Q_cost_matrix=torch.cdist(former_class_means,next_period_class_means,p=self.args['norm_term'])

            #solving ot
            _mu1_vec=torch.ones(len(former_class_means))/len(former_class_means)*1.0
            _mu2_vec=torch.ones(len(next_period_class_means))/len(former_class_means)*1.0

            # 用Sinkhorn算法求解OT，得到传输计划T（T[i][j]表示旧类别i向新类别j传输的知识比例）
            T=ot.sinkhorn(_mu1_vec,_mu2_vec,Q_cost_matrix,self.sinkhorn_reg) 
            T=torch.tensor(T).float().cuda()

            # 基于传输计划T，从旧分类器权重生成新分类器的初始权重
            transformed_hat_W=torch.mm(T.T,F.normalize(self._network.fc.weight, p=2, dim=1))

            # 校准权重尺度：让新分类器权重的 norm 与旧分类器保持一致（避免尺度差异影响分类）
            oldnorm=(torch.norm(self._network.fc.weight,p=2,dim=1))
            newnorm=(torch.norm(transformed_hat_W*len(former_class_means),p=2,dim=1))
            meannew=torch.mean(newnorm)
            meanold=torch.mean(oldnorm)
            gamma=meanold/meannew
            self.calibration_term=gamma
            # 最终新分类器初始权重（校准后）
            self._ot_new_branch=transformed_hat_W*len(former_class_means)*self.calibration_term
        return transformed_hat_W*len(former_class_means)*self.calibration_term


    # 回溯传输（Retrospective Transport, RT）—— 新知识巩固旧分类器
    def solving_ot_to_old(self):
        current_class_num=self.data_manager.get_task_size(self._cur_task)
        self._extract_class_means_with_memory(self.data_manager,self._known_classes,self._total_classes)
        former_class_means=torch.tensor(self._ot_prototype_means[:self._known_classes])
        next_period_class_means=torch.tensor(self._ot_prototype_means[self._known_classes:self._total_classes])
        Q_cost_matrix=torch.cdist(next_period_class_means,former_class_means,p=self.args['norm_term'])+EPSILON #in case of numerical err
        _mu1_vec=torch.ones(len(former_class_means))/len(former_class_means)*1.
        _mu2_vec=torch.ones(len(next_period_class_means))/len(former_class_means)*1.
        T=ot.sinkhorn(_mu2_vec,_mu1_vec,Q_cost_matrix,self.sinkhorn_reg) 
        T=torch.tensor(T).float().cuda()
        transformed_hat_W=torch.mm(T.T,F.normalize(self._network.fc.weight[-current_class_num:,:], p=2, dim=1))
        return transformed_hat_W*len(former_class_means)*self.calibration_term


    # 准备增量训练的数据集与网络
    def incremental_train(self, data_manager):
        self._cur_task += 1
        # 更新总类别数（已知类别+新任务类别）
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)

        # 扩展网络分类器：用OT生成的初始权重初始化新类别分支
        self._network.update_fc(self._total_classes, self.nextperiod_initialization)
        self.data_manager=data_manager

        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))
        self.lamda = self._known_classes / self._total_classes
        # Loader
        # 构建训练集（新任务数据+旧类别记忆样本）和测试集（所有已见类别）
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train', appendent=self._get_memory())    # 新类别数据+旧类别记忆样本
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')  # 所有已见类别的测试数据
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        
        self._train(self.train_loader, self.test_loader)
        # 维护记忆库：缩减并更新旧类别记忆样本（保留代表性样本）
        self._reduce_exemplar(data_manager, memory_size//self._total_classes)
        self._construct_exemplar(data_manager, memory_size//self._total_classes)

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)      # 旧模型用于知识蒸馏
        # 优化器与学习率调度器
        optimizer = optim.SGD(self._network.parameters(), lr=lrate, momentum=0.9, weight_decay=5e-4)  # 1e-5
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
        self._update_representation(train_loader, test_loader, optimizer, scheduler)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            # 动态调整双向传输损失的权重
            weight_ot_init=max(1.-(epoch/2)**2,0)   # 前瞻传输损失权重（前期高，后期降为0）
            weight_ot_co_tuning=(epoch/epochs)**2.  # 回溯传输损失权重（前期低，后期高）
        
            self._network.train()
            losses = 0.
            correct, total = 0, 0
            
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                output=self._network(inputs)
                logits = output['logits']
                onehots = target2onehot(targets, self._total_classes)

                # 1. 分类损失（交叉熵，学习新类别）
                clf_loss = F.cross_entropy(logits, targets)
                if self._old_network is not None:   # 若存在旧模型（非第一个任务）

                    # 若存在旧模型（非第一个任务）
                    old_logits = self._old_network(inputs)['logits'].detach()               # 旧模型输出（冻结）
                    hat_pai_k = F.softmax(old_logits / T, dim=1)                            # 旧模型的软化概率（T为温度）
                    log_pai_k = F.log_softmax(logits[:, :self._known_classes] / T, dim=1)   # 新模型旧类别分支的对数概率
                    distill_loss = -torch.mean(torch.sum(hat_pai_k * log_pai_k, dim=1))     # KD损失

                    # 3. 前瞻传输损失（PT）：仅在训练初期生效，约束新分类器与OT初始化权重对齐
                    if epoch<1:
                        features=F.normalize(output['features'], p=2, dim=1)
                        current_logit_new=F.log_softmax(logits[:, self._known_classes:] / T, dim=1)
                        new_logit_by_wnew_init_by_ot = F.linear(features, F.normalize(self._ot_new_branch, p=2, dim=1))
                        new_logit_by_wnew_init_by_ot = F.softmax(new_logit_by_wnew_init_by_ot / T, dim=1)
                        new_branch_distill_loss = -torch.mean(torch.sum(current_logit_new * new_logit_by_wnew_init_by_ot, dim=1))

                        # 总损失：分类损失 + KD损失 + PT损失
                        loss=distill_loss * self.lamda + clf_loss * (1 - self.lamda) + 0.001*(weight_ot_init*new_branch_distill_loss)

                    # 4. 回溯传输损失（RT）：训练中后期生效，用新分类器优化旧分类器
                    else:
                        features=F.normalize(output['features'], p=2, dim=1)
                        # 每30个batch更新一次反向OT传输的旧分类器权重
                        if i%30==0:
                            with torch.no_grad():
                                self._ot_old_branch=self.solving_ot_to_old()
                        old_logit_by_wold_init_by_ot = F.linear(features, F.normalize(self._ot_old_branch, p=2, dim=1))
                        old_logit_by_wold_init_by_ot = F.log_softmax(old_logit_by_wold_init_by_ot / T, dim=1)
                        old_branch_distill_loss = -torch.mean(torch.sum(hat_pai_k* old_logit_by_wold_init_by_ot, dim=1))
                        # 总损失：分类损失 + KD损失 + RT损失
                        loss=distill_loss * self.lamda + clf_loss * (1 - self.lamda) + self.args['reg_term']*(weight_ot_co_tuning*old_branch_distill_loss)
                else:
                    loss = clf_loss

                # 反向传播与参数更新
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                # 计算训练准确率
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, epoch+1, epochs, losses/len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)

        logging.info(info)


