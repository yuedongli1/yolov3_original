import math
import numpy as np
import mindspore as ms
import mindspore.numpy as mnp
from mindspore import nn, ops, Tensor


@ops.constexpr
def get_tensor(x, dtype=ms.float32):
    return Tensor(x, dtype)


@ops.constexpr(reuse_result=True)
def get_pi(dtype=ms.float32):
    return Tensor(math.pi, dtype)


def bbox_iou(box1, box2, xywh=True, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        x1, y1, w1, h1 = ops.split(box1, 2, 4)
        x2, y2, w2, h2 = ops.split(box2, 2, 4)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = ops.split(box1, 2, 4)
        b2_x1, b2_y1, b2_x2, b2_y2 = ops.split(box2, 2, 4)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

    # Intersection area
    inter = (ops.minimum(b1_x2, b2_x2) - ops.maximum(b1_x1, b2_x1)).clip(0, None) * \
            (ops.minimum(b1_y2, b2_y2) - ops.maximum(b1_y1, b2_y1)).clip(0, None)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union

    return iou  # IoU


class ComputeLoss(nn.Cell):
    # Compute losses
    def __init__(self, model):
        super(ComputeLoss, self).__init__()
        self.ignore_threshold = 0.5
        self.lambda_xy = 2.5
        self.lambda_wh = 2.5
        self.lambda_conf = 1.0
        self.lambda_cls = 1.0

        m = model.model[-1]  # Detect() module
        self.na = m.na  # number of anchors
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss(reduction='mean')

    def construct(self, p, targets):  # predictions, targets
        loss_x = 0.0
        loss_y = 0.0
        loss_w = 0.0
        loss_h = 0.0
        loss_conf = 0.0
        loss_cls = 0.0

        tcls, tx, ty, tw, th, tmasks, noobj_masks, indices, anchors = self.build_targets(p, targets)
        tcls, tx, ty, tw, th, tmasks, noobj_masks, indices, anchors = ops.stop_gradient(tcls), ops.stop_gradient(tx), \
                                               ops.stop_gradient(ty), ops.stop_gradient(tw), ops.stop_gradient(th), \
                                               ops.stop_gradient(tmasks), ops.stop_gradient(noobj_masks), ops.stop_gradient(indices), \
                                               ops.stop_gradient(anchors)

        # Losses
        for layer_index, pi in enumerate(p):  # layer index, layer predictions
            tmask = tmasks[layer_index]
            noobj_mask = noobj_masks[layer_index]
            b, a, gj, gi = ops.split(indices[layer_index] * tmask[None, :], 0, 4)  # image, anchor, gridy, gridx
            b, a, gj, gi = b.view(-1), a.view(-1), gj.view(-1), gi.view(-1)
            tobj = ops.zeros(pi.shape[:4], pi.dtype)  # target obj for no_object
            pobj = ops.Sigmoid()(pi[..., 4])  # pred obj for no_object

            n = b.shape[0]  # number of targets
            _meta_pred = pi[b, a, gj, gi]  # gather from (bs,na,h,w,nc)

            # Get outputs
            px = ops.Sigmoid()(_meta_pred[..., 0])  # Center x
            py = ops.Sigmoid()(_meta_pred[..., 1])  # Center y
            pwh = ops.Exp()(_meta_pred[..., 2:4]) * anchors[layer_index]
            pw = pwh[:, 0]  # Width
            ph = pwh[:, 1]  # Height
            pconf = ops.Sigmoid()(_meta_pred[..., 4])  # Conf
            pcls = ops.Sigmoid()(_meta_pred[..., 5:])  # Cls pred.

            #  losses.
            loss_x += self.bce_loss(px * tmask, tx[layer_index] * tmask)
            loss_y += self.bce_loss(py * tmask, ty[layer_index] * tmask)
            loss_w += self.mse_loss(pw * tmask, tw[layer_index] * tmask)
            loss_h += self.mse_loss(ph * tmask, th[layer_index] * tmask)

            b, a, gj, gi = ops.split(indices[layer_index] * noobj_mask[None, :], 0, 4)  # image, anchor, gridy, gridx
            b, a, gj, gi = b.view(-1), a.view(-1), gj.view(-1), gi.view(-1)
            pobj[b, a, gj, gi] = 0.

            loss_conf += self.bce_loss(pconf * tmask, tmask * 1.0) + 0.5 * self.bce_loss(pobj, tobj)

            t = ops.fill(pcls.dtype, pcls.shape, 0.)  # targets
            t[mnp.arange(n), tcls[layer_index]] = 1.
            loss_cls += self.bce_loss(pcls * tmask[:, None], t * tmask[:, None])
            #  total loss = losses * weight
        loss = loss_x * self.lambda_xy + loss_y * self.lambda_xy + \
                   loss_w * self.lambda_wh + loss_h * self.lambda_wh + \
                   loss_conf * self.lambda_conf + loss_cls * self.lambda_cls

        return loss, ops.stop_gradient(ops.stack((loss_x, loss_y, loss_w, loss_h, loss_conf, loss_cls, loss)))

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        targets = targets.view(-1, 6)
        mask_t = targets[:, 1] >= 0
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tmasks, indices, noobj_masks, tx, ty, tw, th, tcls, anch = (), (), (), (), (), (), (), (), ()
        gain = ops.ones(7, ms.int32) # normalized to gridspace gain
        ai = ops.tile(mnp.arange(na).view(-1, 1), (1, nt)) # shape: (na, nt)
        ai = ops.cast(ai, targets.dtype)
        targets = ops.concat((ops.tile(targets, (na, 1, 1)), ai[:, :, None]), 2) # append anchor indices # shape: (na, nt, 7)

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = get_tensor(shape, targets.dtype)[[3, 2, 3, 2]] # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(na,nt,7) # xywhn -> xywh
            # Matches
            gt_box = ops.zeros((na, nt, 4), ms.float32)
            gt_box[..., 2:] = t[..., 4:6]

            anchor_shapes = ops.zeros((na, 1, 4), ms.float32)
            anchor_shapes[..., 2:] = ops.ExpandDims()(anchors, 1)
            anch_ious = bbox_iou(gt_box, anchor_shapes).squeeze()

            j = anch_ious == anch_ious.max(axis=0)
            l = anch_ious > self.ignore_threshold

            mask_m_t = (ops.cast(j, ms.int32) * ops.cast(mask_t[None, :], ms.int32)).view(-1)
            mask_n = (ops.cast(l, ms.int32) * ops.cast(mask_t[None, :], ms.int32)).view(-1)
            t = t.view(-1, 7)

            # Define
            b, c, gxy, gwh, a = ops.cast(t[:, 0], ms.int32), \
                                ops.cast(t[:, 1], ms.int32), \
                                t[:, 2:4], \
                                t[:, 4:6], \
                                ops.cast(t[:, 6], ms.int32)  # (image, class), grid xy, grid wh, anchors
            gij = ops.cast(gxy, ms.int32)
            gij = gij[:]
            gi, gj = gij[:, 0], gij[:, 1]  # grid indices
            gi = gi.clip(0, shape[3] - 1)
            gj = gj.clip(0, shape[2] - 1)

            # Append
            tmasks += (mask_m_t,)
            noobj_masks += (mask_n,)
            indices += (ops.stack((b, a, gj, gi), 0),)  # image, anchor, grid
            anch += (anchors[a],)  # anchors

            tx += (gxy[:, 0] - gi,)
            ty += (gxy[:, 1] - gj,)
            tw += (gwh[:, 0],)
            th += (gwh[:, 1],)
            tcls += (c,)  # class

        return ops.stack(tcls), \
               ops.stack(tx), \
               ops.stack(ty), \
               ops.stack(tw), \
               ops.stack(th), \
               ops.stack(tmasks), \
               ops.stack(noobj_masks), \
               ops.stack(indices), \
               ops.stack(anch)


if __name__ == '__main__':
    # python train.py --workers 8 --device 0 --batch-size 32 --data data/coco.yaml
    #   --img 640 640 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml
    import yaml
    from pathlib import Path
    from mindspore import context
    from network.yolo import Model
    from config.args import get_args
    from utils.general import check_file, increment_path

    opt = get_args()
    # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
    opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
    assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
    opt.name = 'evolve' if opt.evolve else opt.name
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run
    opt.total_batch_size = opt.batch_size

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
    hyp['label_smoothing'] = opt.label_smoothing

    context.set_context(mode=context.PYNATIVE_MODE, pynative_synchronize=True)
    # context.set_context(mode=context.GRAPH_MODE, pynative_synchronize=True)
    cfg = "D:/yolov3_mindspore/config/network/yolov3.yaml"
    model = Model(cfg, ch=3, nc=80, anchors=None)
    model.hyp = hyp
    model.set_train(True)
    compute_loss = ComputeLoss(model)

    x = Tensor(np.random.randint(0, 256, (2, 3, 640, 640)), ms.float32)
    pred = model(x)
    print("pred: ", len(pred))
    # pred, grad = ops.value_and_grad(model, grad_position=0, weights=None)(x)
    # print("pred: ", len(pred), "grad: ", grad.shape)

    # targets = Tensor(np.load("targets_bs2.npy"), ms.float32)
    targets = Tensor(np.random.randn(2, 160, 6), ms.float32)
    # loss = compute_loss(pred, targets)
    (loss, _), grad = ops.value_and_grad(compute_loss, grad_position=0, weights=None, has_aux=True)(pred, targets)
    print("loss: ", loss)