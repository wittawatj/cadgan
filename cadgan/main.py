__author__ = "wittawat"

import datetime
# import matplotlib
# import matplotlib.pyplot as plt
import os
import pprint
from abc import ABCMeta, abstractmethod

# KBRGAN
import cadgan
import cadgan.imutil as imutil
import cadgan.kernel as kernel
import cadgan.log as log
import cadgan.util as util
import numpy as np
import torch
import torchvision
from future.utils import with_metaclass
from torch.nn import functional as F

# import cadgan
# import cadgan.glo as glo
# import cadgan.plot as plot
# import cadgan.util as util


def save_images(imgs, outfile, nrow=8):
    imgs = imgs / 2 + 0.5  # unnormalize
    torchvision.utils.save_image(imgs, outfile, nrow=nrow)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def pixel_values_check(imgs, interval, img_name):
    """
    imgs: a Pytorch stack of images.
    """
    if not (imgs >= interval[0]).all():
        raise ValueError("Pixel values of {} are not >= {}".format(img_name, interval[0]))
    if not (imgs <= interval[1]).all():
        raise ValueError("Pixel values of {} are not <= {}".format(img_name, interval[1]))


class TensorPenalty(torch.nn.Module):
    """
    An abstract class to represent a penalty term to put on a stack of tensors.

    Intended to be used for noise vectors (latent vectors, Z) of a
    PTNoiseTransformer (e.g., GAN model).  Useful for optimization to find Z.
    """

    @abstractmethod
    def forward(self, Z):
        raise NotImplementedError("Subclasses should implement this method")

    def __str__(self):
        return self.__class__.__name__


# end class TensorPenalty


class TPNull(TensorPenalty):
    """
    A penalty that does not actually penalize anything. Always return 0.
    """

    def __init__(self):
        super(TPNull, self).__init__()

    def forward(self, Z):
        return 0.0


# end class TPNull


class TPSymLogBarrier(TensorPenalty):
    """
    Implement a log barrier such that the output is very high when the
    magnitude (absolute value) of each coordinate in the input tensor is above
    the specified bound.
    """

    def __init__(self, bound, scale=1.0):
        """
        bound: a positive constant to soft bound each coordinate in the input
            tensor
        scale: a scaling constant to multiply the computed penalty (i.e.,
            regularization parameter)
            """
        super(TPSymLogBarrier, self).__init__()
        if not bound >= 0:
            raise ValueError("bound must be non-negative. Was {}".format(bound))
        if not scale >= 0:
            raise ValueError("scale must be non-negative. Was {}".format(scale))
        self.bound = bound
        self.scale = scale

    def forward(self, Z):
        # https://www.wolframalpha.com/input/?i=-log(4%5E2-x%5E2)
        penalty = -self.scale * torch.sum(torch.log(self.bound ** 2 - Z ** 2))
        total = Z.numel()
        return penalty / total

    def __str__(self):
        return "{}(b={},s={})".format(self.__class__.__name__, self.bound, self.scale)


# end class TPSymLogBarrier


class TPFunc(TensorPenalty):
    """
    A TensorPenalty directly specified by a function f: Z -> penalty (real number).
    f has to be compatible with Pytorch.
    """

    def __init__(self, f):
        super(TPFunc, self).__init__()
        self.f = f

    def forward(self, Z):
        f = self.f
        return f(Z)


# end class TPFunc

# =================================================================


class ZRefiner(object):
    """
    A base class to implement procedures that refine Z (stack of noise vectors
    of GANs to optimize). Useful for implementing a smart way to initialize Z
    before solving the kernel moment matching for conditional image generation,
    for instance.
    """

    @abstractmethod
    def refine(self, Z):
        """
        Refine Z (stack of GAN's noise vectors).
        Return a new Z of the same shape.
        """
        raise NotImplementedError()

    def __call__(self, Z):
        return self.refine(Z)


# end class ZRefiner


class ZRInvNorm(ZRefiner):
    """
    Find z = arg min_z ||y - g(z)||_p for some p >= 1.
    """

    pass


# end class ZRInvNorm


class ZRMMDIterGreedy(ZRefiner):
    """
    Iteratively draw one z and keep the one which best improves the MMD loss.
    """

    def __init__(
        self,
        g,
        z_sampler,
        k,
        X,
        n_draws,
        n_sample,
        input_weights=None,
        device=torch.device("cpu"),
        tensor_type=torch.FloatTensor,
    ):
        """
        g: a torch.nn.Module representing the generator
        z_sampler: a callable n |-> Z. Should be so that g(Z) is valid.
        k: a kernel to compute MMD^2 between X and g(Z)
        X: a stack of data points
        n_draws: number of times to redraw *each* z_i and evaluate the loss
            So, total number of draws is n_draws*n_sample
        n_sample: number of points (Z's) to generate in total
        input_weights: convex combination weights used to form input mean
            embedding (on X).
        """
        self.g = g
        self.z_sampler = z_sampler
        self.k = k
        self.X = X
        assert n_draws > 0
        assert n_sample > 0
        self.n_draws = n_draws
        self.n_sample = n_sample
        if input_weights is None:
            # None => use uniform weights
            nx = X.shape[0]
            input_weights = torch.ones(nx, device=device).type(tensor_type) / float(nx)
        self.input_weights = input_weights
        self.device = device
        self.tensor_type = tensor_type
        # self.device = device
        # self.tensor_type = tensor_type

    def refine(self, Z):
        # does not really depend on the input Z
        z_sampler = self.z_sampler
        min_loss = None
        best_Z = None
        n_draws = self.n_draws
        g = self.g
        k = self.k
        X = self.X
        # first get best single z
        for i in range(n_draws):
            zi = z_sampler(1)
            # print(zi.shape)
            Yi = g(zi)

            lossi = mmd2(X, Yi, k, weights_X=self.input_weights, device=self.device, tensor_type=self.tensor_type)
            if min_loss is None or lossi < min_loss:
                min_loss = lossi
                best_Z = zi

        # iteratively greedily add more Z's
        for si in range(self.n_sample - 1):
            assert best_Z.shape[0] == si + 1
            min_loss = None
            best_zi = None
            for i in range(n_draws):
                zi = z_sampler(1)
                Z = torch.cat([best_Z, zi])
                Yi = g(Z)
                lossi = mmd2(X, Yi, k, weights_X=self.input_weights, device=self.device, tensor_type=self.tensor_type)
                if min_loss is None or lossi < min_loss:
                    min_loss = lossi
                    best_zi = zi
            best_Z = torch.cat([best_Z, best_zi])
        assert best_Z.shape[0] == self.n_sample
        return best_Z


# end class ZRMMDIterGreedy


class ZRMMDMultipleRestarts(ZRefiner):
    """
    Draw Z from a specified distribution many times and evaluate
    the loss. Return Z which has the lowest loss. Loss is the MMD i.e., same
    loss used in the conditional generation in pt_gkmm().
    """

    def __init__(self, g, z_sampler, k, X, n_restarts, n_sample):
        """
        g: a torch.nn.Module representing the generator
        z_sampler: a callable n |-> Z. Should be so that g(Z) is valid.
        k: a kernel to compute MMD^2 between X and g(Z)
        X: a stack of data points
        n_restarts: number of times to redraw Z and evaluate the loss
        n_sample: number of points (Z's) to generate in each batch
        """
        self.g = g
        self.z_sampler = z_sampler
        self.k = k
        self.X = X
        assert n_restarts > 0
        assert n_sample > 0
        self.n_restarts = n_restarts
        self.n_sample = n_sample

    def refine(self, Z):
        best_Z = Z
        g = self.g
        k = self.k
        X = self.X
        min_loss = mmd2(X, g(Z), k)
        for i in range(self.n_restarts):
            # draw Z
            Zi = self.z_sampler(self.n_sample)
            Yi = g(Zi)
            lossi = mmd2(X, Yi, k)
            if lossi < min_loss:
                min_loss = lossi
                best_Z = Zi
        return best_Z


# end class ZRMMDMultipleRestarts

# ================================================================


def mmd2(X, Y, k, weights_X=None, weights_Y=None, device=torch.device("cpu"), tensor_type=torch.FloatTensor):
    """
    Estimate the squared MMD between X and Y using the kernel k (biased
    estimator).
    * X: stack of data
    * Y: stack of data
    * k: a kernel
    * weights_X: Pytorch 1d tensor for of the same length as X.shape[0].
        Weights which sum to one. None => set to uniform weights.
    * weights_Y: Weights for Y. Same format as weights_X.
    """

    if weights_X is None:
        # None => use uniform weights
        nx = X.shape[0]
        weights_X = torch.ones(nx, device=device).type(tensor_type) / float(nx)

    if weights_Y is None:
        ny = Y.shape[0]
        weights_Y = torch.ones(ny, device=device).type(tensor_type) / float(ny)

    # check the range of weights
    if not ((weights_X >= 0.0).all() and (weights_X <= 1.0).all()):
        raise ValueError(
            '"weights_X" contains at least one weight which is outside [0,1] interval. Was {}'.format(weights_X)
        )

    if not ((weights_Y >= 0.0).all() and (weights_Y <= 1.0).all()):
        raise ValueError(
            '"weights_Y" contains at least one weight which is outside [0,1] interval. Was {}'.format(weights_Y)
        )

    # check that weights sum to 1
    if torch.abs(weights_X.sum() - 1.0) > 1e-3:
        raise ValueError('"weights_X" does not sum to one. Was {}'.format(weights_X.sum()))
    if torch.abs(weights_Y.sum() - 1.0) > 1e-3:
        raise ValueError('"weights_Y" does not sum to one. Was {}'.format(weights_Y.sum()))

    Kxx = k.eval(X, X)
    Kyy = k.eval(Y, Y)
    Kxy = k.eval(X, Y)

    # mmd2 = Kxx.mean() + Kyy.mean() - 2.0*Kxy.mean()
    mmd2 = Kxx.mv(weights_X).dot(weights_X) + Kyy.mv(weights_Y).dot(weights_Y) - 2.0 * Kxy.mv(weights_Y).dot(weights_X)
    return mmd2


def weighing_logits(FX):
    """ Modifying feature extraction such that GAN focus only on a certain region/feature
        of the image"""
    output = F.softmax(FX, 1).data.squeeze()
    top1_idx = torch.argmax(output)
    one_hot = torch.zeros((1, output.size()[-1]))
    one_hot[0][top1_idx] = 1
    return output * one_hot


def pt_gkmm(
    g,
    cond_imgs,
    extractor,
    k,
    Z,
    optimizer,
    sum_writer,
    input_weights=None,
    z_penalty=TPNull(),
    device=torch.device("cpu"),
    tensor_type=torch.FloatTensor,
    n_opt_iter=500,
    seed=1,
    texture=0,
    img_log_steps=10,
    weigh_logits=0,
    log_img_dir="",
):
    """
    Conditionally generate images conditioning on the input images (cond_imgs)
    using kernel moment matching.

    * g: a generator of type torch.nn.Module (forward() takes noise vectors
        and tranforms them into images). Need to be differentiable for the optimization.
    * cond_imgs: a stack of input images to condition on. Pixel value range
        should be [0,1]
    * extractor: an instance of torch.nn.Module representing a
        feature extractor for image input.
    * k: cadgan.kernel.PTKernel representing a kernel on top of the extracted
        features.
    * Z: a stack of noise vectors to be optimized. These are fed to the
        generator g for the optimization.
    * optimizer: a Pytorch optimizer. The list of variables to optimize has to
        contain Z.
    * sum_writer: SummaryWriter for tensorboard.
    * input_weights: a one-dimensional Torch tensor (vector) whose length is the
        same as the number of conditioned images. Specifies weights of the
        conditioned images. 0 <= w_i <= 1 and weights sum to 1.
        If None, automatically set to uniform weight.s
    * z_penalty: a TensorPenalty to penalize Z. Set to TPNull() to set to
        penalty.
    * device: a object constructed from torch.device(..). Likely this might be
        torch.device('cuda') or torch.device('cpu'). Use CPU by default.
    * tensor_type: Default Pytorch tensor type to use e.g., torch.FloatTensor
        or torch.cuda.FloatTensor. Use torch.FloatTensor by default (for cpu)
    * n_opt_iter: number of iterations for the optimization
    * seed: random seed (positive integer)
    * img_log_steps: record generated images once every this many
        optimization steps.
    * weigh_logits: to weight the output logits of feature extactor so that we can
        backpropagate w.r.t certain image feature.

    Write output in a Tensorboard log.
    """
    # Check generator's output range and image pixel range
    # We work with [0, 1]
    pixel_values_check(cond_imgs, (0, 1), "cond_imgs")
    tmp_sam = g.forward(Z)
    pixel_values_check(tmp_sam, (0, 1), "generator's output")

    # number of images to condition on
    n_cond = cond_imgs.shape[0]

    if input_weights is None:
        # None => set to uniform weights.
        input_weights = torch.ones(n_cond, device=device).type(tensor_type) / float(n_cond)

    # Check the rangeo of input_weights. Has to be in [0,1]
    if not ((input_weights >= 0.0).all() and (input_weights <= 1.0).all()):
        raise ValueError(
            '"input_weights" contains at least one weight which is outside [0,1] interval. Was {}'.format(input_weights)
        )
    # Check that the weights sum to 1
    if torch.abs(input_weights.sum() - 1.0) > 1e-3:
        raise ValueError('"input_weights" does not sum to one. Was {}'.format(input_weights.sum()))

    gens_cpu = tmp_sam.to(torch.device("cpu"))
    arranged_init_imgs = torchvision.utils.make_grid(gens_cpu, nrow=2, normalize=True)
    log.l().debug('Adding initial generated images to Tensorboard')
    sum_writer.add_image("Init_Images", arranged_init_imgs)

    del tmp_sam

    # Setting requires_grad=True is very important. We will optimize Z.
    Z.requires_grad = True
    # number of images to generate
    n_sample = Z.shape[0]

    # Put models on gpu if needed
    # with torch.enable_grad():
    #    g = g.to(device)

    # Select a test image from the generated images
    arranged_cond_imgs = torchvision.utils.make_grid(cond_imgs, nrow=2, normalize=True)
    sum_writer.add_image("Cond_Images", arranged_cond_imgs)

    with torch.no_grad():
        FX_ = extractor.forward(cond_imgs)
        FX = FX_
        if weigh_logits:
            FX = weighing_logits(FX)

    # mean_KFX = torch.mean(k.eval(FX, FX))
    kFX = k.eval(FX, FX)
    mean_KFX = kFX.mv(input_weights).dot(input_weights)
    time_per_itr = []
    loss_all = []
    for t in range(n_opt_iter):

        def closure():
            Z.data.clamp_(-3.3, 3.3)
            optimizer.zero_grad()

            gens = g.forward(Z)
            if gens.size()[3] == 1024:
                # Downsample images else it takes a lot of time in optimization
                # TODO: WJ: To downsample, it is better to do it before calling this function.
                # Condiitonal generation function does not need to handle this.
                downsample = torch.nn.AvgPool2d(3, stride=2)
                gens = downsample(downsample(gens))

            if t <= -1 or t % img_log_steps == 0 or t == n_opt_iter - 1:
                gens_cpu = gens.to(torch.device("cpu"))
                imutil.save_images(gens_cpu, os.path.join(log_img_dir, "output_images", str(t)))
                arranged_gens = torchvision.utils.make_grid(gens_cpu, nrow=2, normalize=True)
                log.l().debug('Logging generated images at iteration {}'.format(t+1))
                sum_writer.add_image("Generated_Images", arranged_gens, t)

            F_gz = extractor.forward(gens)
            # import pdb; pdb.set_trace()
            if t <= -1 or t % img_log_steps == 0 or t == n_opt_iter - 1:
                feature_size = int(np.sqrt(F_gz.shape[1]))
                # import pdb; pdb.set_trace()
                try:
                    feat_out = F_gz.view(F_gz.shape[0], 1, feature_size, feature_size)
                    gens_cpu = feat_out.to(torch.device("cpu"))
                    imutil.save_images(gens_cpu, os.path.join(log_img_dir, "feature_images", str(t)))
                    arranged_init_imgs = torchvision.utils.make_grid(gens_cpu, nrow=2, normalize=True)
                    sum_writer.add_image("feature_images", arranged_init_imgs, t)
                except:
                    if t == 0:
                        log.l().debug("Unable to plot features as image. Okay. Will skip plotting features.")

            if weigh_logits:
                # WJ: This option is not really used. Should be removed.
                F_gz = weighing_logits(F_gz)
            KF_gz = k.eval(F_gz, F_gz)

            Z_loss = z_penalty(Z)
            mmd2 = torch.mean(KF_gz) - 2.0 * torch.mean(k.eval(F_gz, FX).mv(input_weights)) + mean_KFX
            loss = mmd2 + Z_loss

            # compute the gradients
            loss.backward(retain_graph=True)

            # record losses
            sum_writer.add_scalar("loss/total", loss.item(), t)
            sum_writer.add_scalar("loss/mmd2", mmd2.item(), t)
            sum_writer.add_scalar("loss/Z_penalty", Z_loss, t)

            # record some statistics
            sum_writer.add_scalar("Z/max_z", torch.max(Z), t)
            sum_writer.add_scalar("Z/min_z", torch.min(Z), t)
            sum_writer.add_scalar("Z/avg_z", torch.mean(Z), t)
            sum_writer.add_scalar("Z/std_z", torch.std(Z), t)
            sum_writer.add_histogram("Z/hist", Z.reshape(-1), t)

            loss_all.append(mmd2.item())

            if t <= 20 or t % 20 == 0:
                log.l().info("Iter [{}], overall_loss: {}".format(t, loss.item()))
            return loss

        #    start_time = datetime.datetime.now()
        optimizer.step(closure)
    #    time_per_itr.append((datetime.datetime.now() - start_time).total_seconds())
    # import time
    # with open(str(time.time())+'.txt','w+') as f:
    #    f.write('Mean:'+ str(np.mean(time_per_itr))+'\n')
    #    f.write('Variance:'+ str(np.var(time_per_itr)))
    #    f.write('Final Loss:'+str(loss_all[-1]))
    # --------- save the generated images ----
    # sample_interval = 20
    # if t%sample_interval==0:
    #     with torch.no_grad():
    #         gens_plot = g(Z.detach().clone())
    #         sample_nrow = 6
    #         img_dir = '.'
    #         save_images(gens_plot, os.path.join(img_dir, 'generated_%03d.png' % t),
    #                   nrow=sample_nrow)
