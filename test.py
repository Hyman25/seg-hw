import os
from typing import Any
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

class MetricRecorder:
    def __init__(self, thds_for_fm=255, beta_for_fm=1, th_for_acc=0.5, eps=1e-8, use_AP=False, use_SM=False, use_EM=False):
        '''
        :param thds_for_fm: 计算fm时，阈值的个数
        :param beta_for_fm: 计算fm时，beta的值
        :param th_for_acc: 计算acc时，阈值
        :param eps: 防止分母为0
        '''
        self.fm = cal_fm(thds_for_fm, beta_for_fm, eps)
        self.acc = cal_acc(th_for_acc, eps)
        self.sm = cal_sm()
        self.em = cal_em()
        self.ap = cal_ap()
        self.use_AP = use_AP
        self.use_SM = use_SM
        self.use_EM = use_EM

    def update(self, pre, gt):
        pre = pre.reshape(-1)
        gt = gt.reshape(-1)
        self.fm.update(pre, gt)
        self.acc.update(pre, gt)
        if self.use_AP:
            self.ap.update(pre, gt)
        if self.use_SM:
            self.sm.update(pre, gt)
        if self.use_EM:
            self.em.update(pre, gt)

    def show(self, bit_num: int) -> tuple:
        '''
        :param bit_num: 保留小数点后几位
        :return: mae, (maxf, meanf, f, p, r), sm, em
        '''
        f, maxf, meanf, p, r = self.fm.show() # [thds,], [1,], [1,], [thds,], [thds,]
        PRE, REC, FPR, FNR, ACC = self.acc.show() # [1,], [1,], [1,], [1,], [1,]
        maxf, meanf, PRE, REC, FPR, FNR, ACC = self._round_bit_num(
            data=[maxf, meanf, PRE, REC, FPR, FNR, ACC], bit_num=bit_num
        )
        output = {
            'maxf': maxf, 'meanf': meanf, 'f': f, 'p': p, 'r': r,
            'PRE': PRE, 'REC': REC, 'FPR': FPR, 'FNR': FNR, 'ACC': ACC
        }
        if self.use_AP:
            AP = self.ap.show() # [1,]
            AP = self._round_bit_num(data=[AP], bit_num=bit_num)
            output['AP'] = AP
        if self.use_SM:
            sm = self.sm.show()
            sm = self._round_bit_num(data=[sm], bit_num=bit_num)
            output['sm'] = sm
        if self.use_EM:
            em = self.em.show()
            em = self._round_bit_num(data=[em], bit_num=bit_num)
            output['em'] = em
        return output

    def _round_bit_num(self, data: list, bit_num: int):
        results = []
        for d in data:
            results.append(round(d, bit_num))
        return results


class cal_fm(object):
    # Fmeasure(maxFm, meanFm)---Frequency-tuned salient region detection(CVPR 2009)
    # threshold = 2 * pred.mean !!!
    def __init__(self, thds=255, beta=1., eps=1e-8):
        '''
        :param thds: 一共计算多少个阈值
        :param beta: F-measure的beta值
        '''
        self.thds = thds
        self.beta = beta
        self.eps = eps
        self.precision = []
        self.recall = []
        self.meanF = []

    def update(self, pred, gt):
        if gt.max() != 0:
            prediction, recall, Fmeasure_temp = self.cal(pred, gt)
            self.precision.append(prediction)
            self.recall.append(recall)
            self.meanF.append(Fmeasure_temp)

    def cal(self, pred, gt):
        ########################meanF##############################
        threshold = 2 * pred.mean()
        if threshold > 1:
            threshold = 1
        binary = np.zeros_like(pred)
        binary[pred >= threshold] = 1
        hard_gt = np.zeros_like(gt)
        hard_gt[gt > 0.5] = 1
        tp = (binary * hard_gt).sum()
        if tp == 0:
            meanF = 0
        else:
            pre = tp / binary.sum()
            rec = tp / hard_gt.sum()
            meanF = (1 + self.beta**2) * pre * rec / ((self.beta**2) * pre + rec + self.eps)

        ########################maxF##############################
        pred = np.uint8(pred * 255)
        target = pred[gt > 0.5]
        nontarget = pred[gt <= 0.5]
        targetHist, _ = np.histogram(target, bins=range(256))
        nontargetHist, _ = np.histogram(nontarget, bins=range(256))
        targetHist = np.cumsum(np.flip(targetHist, axis=0), axis=0)
        nontargetHist = np.cumsum(np.flip(nontargetHist, axis=0), axis=0)
        precision = targetHist / (targetHist + nontargetHist + self.eps)
        recall = targetHist / np.sum(gt)
        return precision, recall, meanF

    def show(self):
        precision = np.mean(self.precision, axis=0) # [thds,]
        recall = np.mean(self.recall, axis=0) # [thds,]
        fmeasure = (1 + self.beta**2) * precision * recall / ((self.beta**2) * precision + recall + self.eps) # [thds,]
        fmeasure_avg = np.mean(self.meanF, axis=0) # [1,]
        return fmeasure, fmeasure.max(), fmeasure_avg, precision, recall


class cal_acc(object):
    # Accuracy
    def __init__(self, th=0.5, esp=1e-8):
        self.th = th
        self.esp = esp
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0

    def update(self, pred, gt):
        TP, TN, FP, FN = self.cal(pred, gt)
        self.TP += TP
        self.TN += TN
        self.FP += FP
        self.FN += FN

    def cal(self, pred, gt):
        TP = ((pred > self.th) * (gt > self.th)).sum()
        TN = ((pred <= self.th) * (gt <= self.th)).sum()
        FP = ((pred > self.th) * (gt <= self.th)).sum()
        FN = ((pred <= self.th) * (gt > self.th)).sum()
        return TP, TN, FP, FN

    def show(self):
        PRE = self.TP / (self.TP + self.FP + self.esp)
        REC = self.TP / (self.TP + self.FN + self.esp)
        FPR = self.FP / (self.FP + self.TN + self.esp)
        FNR = self.FN / (self.FN + self.TP + self.esp)
        ACC = (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN + self.esp)
        return PRE, REC, FPR, FNR, ACC


class cal_ap(object):
    def __init__(self) -> None:
        self.pair = []

    def update(self, pred, gt):
        self.pair = self.pair + list(zip(pred, gt))

    def show(self):
        data = np.array(sorted(self.pair))[:, 1] # gt [len,]
        data_cumsum = np.cumsum(data)
        rec = data_cumsum / len(data)
        pre = data_cumsum / np.arange(1, len(data) + 1)
        AP = 0.
        for t in np.linspace(0, 1, 11):
            if (rec >= t).any():
                AP += np.max(pre[rec >= t])
        return AP / 11.


class cal_sm(object):
    # Structure-measure: A new way to evaluate foreground maps (ICCV 2017)
    # https://arxiv.org/abs/1708.00786
    # or chinese blog https://www.x-mol.com/paper/1409436008791724032/t
    def __init__(self, alpha=0.5):
        self.prediction = []
        self.alpha = alpha

    def update(self, pred, gt):
        gt = gt > 0.5
        score = self.cal(pred, gt)
        self.prediction.append(score)

    def show(self):
        return np.mean(self.prediction)

    def cal(self, pred, gt):
        y = np.mean(gt)
        if y == 0:
            score = 1 - np.mean(pred)
        elif y == 1:
            score = np.mean(pred)
        else:
            score = self.alpha * self.object(pred, gt) + (1 - self.alpha) * self.region(pred, gt)
        return score

    def object(self, pred, gt):
        fg = pred * gt
        bg = (1 - pred) * (1 - gt)

        u = np.mean(gt)
        return u * self.s_object(fg, gt) + (1 - u) * self.s_object(
            bg, np.logical_not(gt)
        )

    def s_object(self, in1, in2):
        x = np.mean(in1[in2])
        sigma_x = np.std(in1[in2])
        return 2 * x / (pow(x, 2) + 1 + sigma_x + 1e-8)

    def region(self, pred, gt):
        [y, x] = center_of_mass(gt)
        y = int(round(y)) + 1
        x = int(round(x)) + 1
        [gt1, gt2, gt3, gt4, w1, w2, w3, w4] = self.divideGT(gt, x, y)
        pred1, pred2, pred3, pred4 = self.dividePred(pred, x, y)

        score1 = self.ssim(pred1, gt1)
        score2 = self.ssim(pred2, gt2)
        score3 = self.ssim(pred3, gt3)
        score4 = self.ssim(pred4, gt4)

        return w1 * score1 + w2 * score2 + w3 * score3 + w4 * score4

    def divideGT(self, gt, x, y):
        h, w = gt.shape
        area = h * w
        LT = gt[0:y, 0:x]
        RT = gt[0:y, x:w]
        LB = gt[y:h, 0:x]
        RB = gt[y:h, x:w]

        w1 = x * y / area
        w2 = y * (w - x) / area
        w3 = (h - y) * x / area
        w4 = (h - y) * (w - x) / area

        return LT, RT, LB, RB, w1, w2, w3, w4

    def dividePred(self, pred, x, y):
        h, w = pred.shape
        LT = pred[0:y, 0:x]
        RT = pred[0:y, x:w]
        LB = pred[y:h, 0:x]
        RB = pred[y:h, x:w]

        return LT, RT, LB, RB

    def ssim(self, in1, in2):
        in2 = np.float32(in2)
        h, w = in1.shape
        N = h * w

        x = np.mean(in1)
        y = np.mean(in2)
        sigma_x = np.var(in1)
        sigma_y = np.var(in2)
        sigma_xy = np.sum((in1 - x) * (in2 - y)) / (N - 1)

        alpha = 4 * x * y * sigma_xy
        beta = (x * x + y * y) * (sigma_x + sigma_y)

        if alpha != 0:
            score = alpha / (beta + 1e-8)
        elif alpha == 0 and beta == 0:
            score = 1
        else:
            score = 0

        return score


class cal_em(object):
    # Enhanced-alignment Measure for Binary Foreground Map Evaluation (IJCAI 2018)
    def __init__(self):
        self.prediction = []

    def update(self, pred, gt):
        score = self.cal(pred, gt)
        self.prediction.append(score)

    def cal(self, pred, gt):
        th = 2 * pred.mean()
        if th > 1:
            th = 1
        FM = np.zeros(gt.shape)
        FM[pred >= th] = 1
        FM = np.array(FM, dtype=bool)
        GT = np.array(gt, dtype=bool)
        dFM = np.double(FM)
        if sum(sum(np.double(GT))) == 0:
            enhanced_matrix = 1.0 - dFM
        elif sum(sum(np.double(~GT))) == 0:
            enhanced_matrix = dFM
        else:
            dGT = np.double(GT)
            align_matrix = self.AlignmentTerm(dFM, dGT)
            enhanced_matrix = self.EnhancedAlignmentTerm(align_matrix)
        [w, h] = np.shape(GT)
        score = sum(sum(enhanced_matrix)) / (w * h - 1 + 1e-8)
        return score

    def AlignmentTerm(self, dFM, dGT):
        mu_FM = np.mean(dFM)
        mu_GT = np.mean(dGT)
        align_FM = dFM - mu_FM
        align_GT = dGT - mu_GT
        align_Matrix = (
            2.0
            * (align_GT * align_FM)
            / (align_GT * align_GT + align_FM * align_FM + 1e-8)
        )
        return align_Matrix

    def EnhancedAlignmentTerm(self, align_Matrix):
        enhanced = np.power(align_Matrix + 1, 2) / 4
        return enhanced

    def show(self):
        return np.mean(self.prediction)