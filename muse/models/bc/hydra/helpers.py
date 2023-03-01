import numpy as np
import torch

from muse.experiments import logger


def get_mode_smooth_preproc_fn(mode_key, smooth_mode_key, smooth_n=2, use_gaussian=False, gaussian_width=3, debug=False):
    # n on either side of mode=1 segment.
    if use_gaussian:
        from scipy import signal
        K = 2 * smooth_n + 1
        kernel = signal.windows.gaussian(K, std=K / 2 * gaussian_width)
    else:
        # e.g. for sn = 1, weight = [0.5 1 0.5]  (this, next and next-next must be mask=True)
        tail = np.linspace(0, 1, num=smooth_n + 1, endpoint=False)[1:]  # len() = sn
        kernel = np.concatenate([tail, [1.], np.flip(tail)])

    kernel = kernel / np.sum(kernel)  # normalize

    if debug:
        logger.debug(f"Using kernel (gaussian={use_gaussian}): {kernel}")
        logger.debug(f"Smoothing from {mode_key} -> {smooth_mode_key}")

    # default preproc: compute the mask, save to memory.
    def episode_preproc_fn(inputs, onetime_inputs, idx):
        inputs = inputs.leaf_copy()
        mode = inputs[mode_key]  # should be (H, 1) for same initial shape as the key

        with torch.no_grad():
            mode = mode[..., 0].astype(float)  # remove last dim pre conv

            # pad the ends with the same (horizon dim)
            pad_mode = np.concatenate([mode[..., :1]] * smooth_n + [mode] + [mode[..., -1:]] * smooth_n, axis=-1)

            # [H,], [4*sn+1,] -> out shape: [H+4*sn,] -> [H,], float
            new_mode = np.convolve(pad_mode, kernel)[2*smooth_n:-2*smooth_n, None]

            # saturate at 1.
            mode = np.minimum(new_mode, 1)

        # smoothed key
        inputs[smooth_mode_key] = mode

        return inputs, onetime_inputs, [smooth_mode_key]

    return episode_preproc_fn


def get_mode_to_mask_preproc_fn(mode_key, mask_key, meq=None, mlt=None, mgt=None, skip_last_n=None, smooth=False):
    # default preproc: compute the mask, save to memory.
    def episode_preproc_fn(inputs, onetime_inputs, idx):
        inputs = inputs.leaf_copy()
        mode = inputs[mode_key]  # should be (H, 1) for same initial shape as the key
        mask = np.ones_like(mode, dtype=bool)  # (H, 1)

        if meq is not None:
            mask = mask & (mode == meq)
        if mlt is not None:
            mask = mask & (mode < mlt)
        if mgt is not None:
            mask = mask & (mode > mgt)

        # skipping last n
        if skip_last_n is not None:
            with torch.no_grad():
                mask = mask[..., 0].astype(float)  # remove last dim pre conv

                # e.g. for sln = 2, weight = [1 1 1]  (this, next and next-next must be mask=True)
                weight = np.array([1.] * (skip_last_n + 1))

                # [H,], [sln+1,] -> out shape: [H+sln,] -> [H,], but still float...
                new_mask = np.convolve(mask, np.flip(weight))[skip_last_n:]

                if smooth:
                    # ramp up/down within first last skip_last_n, continuous noise
                    mask = new_mask[:, None] / (skip_last_n + 1)
                else:
                    # checking for convolution being satisfied, converting to bool.
                    mask = (new_mask >= skip_last_n + 1)[:, None]  # make back into (H, 1)
        else:
            assert not smooth, "Can only smooth using skip_last_n!"

        inputs[mask_key] = mask

        return inputs, onetime_inputs, [mask_key]

    return episode_preproc_fn