"""
processing_odoc2.py
====================
OD-OC Segmentation model (UNet + FIAM).
Model file : UNet+FIAM_1.2_0back_newformula.h5
Input      : RGB fundus image (numpy uint8, H x W x 3)
Output     : colour-coded RGB numpy array (uint8, H x W x 3)
               - Green  [0, 255, 0]  → Optic Disc  (ch 1)
               - Blue   [0, 0, 255]  → Optic Cup   (ch 2)

Called by app.py as:
    result_rgb = processing(image_cv2, threshold, batch_size)
"""

import os
import numpy as np
import cv2
import tensorflow as tf

# ------------------------------------------------------------------ #
#  Legacy / compatibility patches (must run before model load)        #
#                                                                     #
#  Models saved under TF 2.15 / Python 3.11 embed config keys that   #
#  newer Keras no longer accepts as __init__ arguments:               #
#    SpatialDropout2D : noise_shape, seed, trainable, dtype           #
#    Conv2DTranspose  : groups                                         #
#    Conv2D           : groups                                         #
#    DepthwiseConv2D  : groups                                         #
#  We monkey-patch both __init__ and from_config on every affected    #
#  class so all deserialization paths are covered.                    #
# ------------------------------------------------------------------ #
from tensorflow.keras import layers as _kl

def _make_compat_patches():
    PATCH_MAP = {
        'SpatialDropout2D': ('noise_shape', 'seed', 'trainable', 'dtype'),
        'Conv2DTranspose':  ('groups',),
        'Conv2D':           ('groups',),
        'DepthwiseConv2D':  ('groups',),
        'Conv1D':           ('groups',),
        'Conv3D':           ('groups',),
    }

    for cls_name, bad_keys in PATCH_MAP.items():
        cls = getattr(_kl, cls_name, None)
        if cls is None or getattr(cls, '_compat_patched', False):
            continue

        _orig_init = cls.__init__
        def _make_init(orig, keys):
            def _patched_init(self, *args, **kwargs):
                for k in keys:
                    kwargs.pop(k, None)
                orig(self, *args, **kwargs)
            return _patched_init
        cls.__init__ = _make_init(_orig_init, bad_keys)

        def _make_from_config(keys):
            @classmethod
            def _patched_from_config(cls, config):
                for k in keys:
                    config.pop(k, None)
                return cls(**config)
            return _patched_from_config
        cls.from_config = _make_from_config(bad_keys)

        cls._compat_patched = True

_make_compat_patches()



# ------------------------------------------------------------------ #
#  Custom FIAM layer                                                  #
# ------------------------------------------------------------------ #
class FIAM(tf.keras.layers.Layer):
    def __init__(self, bins=300, **kwargs):
        super().__init__(**kwargs)
        self.bins = bins

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'bins': self.bins})
        return cfg

    # --- helper ops (kept as tf ops so graph export still works) ---
    def fore(self, X, T):
        T = tf.cast(T, tf.float32)
        X = tf.cast(X, tf.float32)
        return tf.where(X >= T, X, tf.zeros_like(X))

    def back(self, X, T):
        T = tf.cast(T, tf.float32)
        X = tf.cast(X, tf.float32)
        return tf.where(X < T, X, tf.zeros_like(X))

    def pearson_r(self, y_true, y_pred):
        epsilon = 1e-5
        x, y    = y_true, y_pred
        mx, my  = tf.reduce_mean(x), tf.reduce_mean(y)
        xm, ym  = x - mx, y - my
        r_num   = tf.reduce_sum(xm * ym)
        r_den   = tf.sqrt(tf.reduce_sum(xm * xm) * tf.reduce_sum(ym * ym))
        return r_num / (r_den + epsilon)

    def call(self, x):
        def process_single_input(x_single):
            Fore = tf.TensorArray(tf.float32, size=x_single.shape[2])
            Back = tf.TensorArray(tf.float32, size=x_single.shape[2])

            for img in range(x_single.shape[2]):
                ch      = x_single[:, :, img]
                ch_min  = tf.reduce_min(ch)
                ch_max  = tf.reduce_max(ch)
                norm    = (ch - ch_min) / (ch_max - ch_min + 1e-10)
                hist_x  = tf.histogram_fixed_width(norm, [0.0, 1.0], nbins=self.bins)
                hist_x  = hist_x / tf.reduce_sum(hist_x)
                temp    = ((hist_x[1:-1] - hist_x[:-2]) ** 2) / (hist_x[:-2] + 1e-10)
                index   = tf.argmin(temp)
                th      = tf.cast(index, tf.float32) / 256.0

                back_v  = self.back(norm, th)
                fore_v  = self.fore(norm, th)
                Fore    = Fore.write(img, tf.expand_dims(fore_v, 0))
                Back    = Back.write(img, tf.expand_dims(back_v, 0))

            Fore = Fore.stack()
            Back = Back.stack()

            f = tf.reshape(Fore, [-1])
            b = tf.reshape(Back, [-1])
            f_new  = tf.boolean_mask(f, f + b != 0)
            b_new  = tf.boolean_mask(b, f + b != 0)
            corr   = self.pearson_r(f_new, b_new)

            f1 = tf.boolean_mask(f, f != 0)
            b1 = tf.boolean_mask(b, b != 0)

            WD = (tf.reduce_mean(f1) - tf.reduce_mean(b1)) ** 2 + \
                 (tf.math.reduce_std(f1) - tf.math.reduce_std(b1)) ** 2 + \
                 2.0 * tf.math.reduce_std(f1) * tf.math.reduce_std(b1) * (1 - corr)
            L  = (tf.reduce_mean(f1) + tf.reduce_mean(b1)) ** 2 + \
                 (tf.math.reduce_std(f1) + tf.math.reduce_std(b1)) ** 2 - \
                 2.0 * tf.math.reduce_std(f1) * tf.math.reduce_std(b1) * (1 - corr)
            B  = (tf.reduce_mean(f1) ** 2 - tf.reduce_mean(b1) ** 2) + \
                 (tf.math.reduce_std(f1) ** 2 - tf.math.reduce_std(b1) ** 2)
            a  = (-B + tf.sqrt(B ** 2 + 1.2 * WD * L)) / L

            new_img = (1 + a) * Fore + (1 - a) * Back
            new_img = tf.squeeze(tf.transpose(new_img, perm=[2, 3, 0, 1]))
            new_img.set_shape(x_single.shape)
            return new_img

        return tf.map_fn(process_single_input, x)


# ------------------------------------------------------------------ #
#  Lazy model loader (singleton so it is loaded only once)           #
# ------------------------------------------------------------------ #
_model = None
MODEL_FILENAME = os.path.join('models', 'UNet+FIAM_1.2_0back_newformula.h5')


def _load_model():
    global _model
    if _model is not None:
        return _model

    if not os.path.exists(MODEL_FILENAME):
        raise FileNotFoundError(
            f"Model file '{MODEL_FILENAME}' not found. "
            "Place it in the same directory as app.py."
        )

    custom_objects = {
        "FIAM": FIAM,
    }
    with tf.keras.utils.custom_object_scope(custom_objects):
        _model = tf.keras.models.load_model(MODEL_FILENAME, compile=False)

    print(f"[ODOC2] Model loaded from {MODEL_FILENAME}")
    return _model


# ------------------------------------------------------------------ #
#  Colour mapping                                                     #
# ------------------------------------------------------------------ #
# Model outputs 3 channels: ch0 = background, ch1 = OD, ch2 = OC
_CHANNEL_COLORS = {
    1: (0, 255,   0),   # Optic Disc → Green
    2: (0,   0, 255),   # Optic Cup  → Blue
}


def _color_map(recon: np.ndarray, threshold: float) -> np.ndarray:
    """Convert (H, W, C) float32 prediction → (H, W, 3) uint8 RGB."""
    h, w    = recon.shape[:2]
    out     = np.zeros((h, w, 3), dtype=np.uint8)
    for ch_idx, color in _CHANNEL_COLORS.items():
        if ch_idx >= recon.shape[2]:
            break
        out[recon[:, :, ch_idx] >= threshold] = color
    return out


# ------------------------------------------------------------------ #
#  Weight logging utility                                             #
# ------------------------------------------------------------------ #
def log_weights(verbose: bool = False) -> None:
    """
    Print a weight summary for the loaded ODOC2 model to the terminal.

    Parameters
    ----------
    verbose : bool
        False (default) — prints one line per layer with shape, mean, std,
                          min, max, and a NaN/Inf flag.
        True            — additionally dumps the first 8 values of every
                          weight tensor so you can spot obvious corruption.

    Usage
    -----
    From a Python shell or a one-off script:
        from processing.processing_odoc2 import log_weights
        log_weights()          # summary
        log_weights(True)      # summary + first-8 values

    Or call it right after the model loads by adding to _load_model():
        log_weights()
    """
    model = _load_model()

    SEP  = "=" * 80
    SEP2 = "-" * 80

    print(f"\n{SEP}")
    print(f"  WEIGHT LOG — ODOC2 model  ({MODEL_FILENAME})")
    print(f"  Total params : {model.count_params():,}")
    print(f"  Layers       : {len(model.layers)}")
    print(SEP)

    total_weights = 0
    bad_layers    = []

    # Header
    print(f"{'Layer':<40} {'Weight':<30} {'Shape':<20} {'Mean':>9} {'Std':>9} "
          f"{'Min':>9} {'Max':>9}  {'Status'}")
    print(SEP2)

    for layer in model.layers:
        weights = layer.weights
        if not weights:
            continue
        for w in weights:
            vals         = w.numpy()
            total_weights += vals.size
            finite       = np.isfinite(vals)
            n_bad        = int(np.sum(~finite))
            status       = "✓ OK" if n_bad == 0 else f"✗ {n_bad} NaN/Inf"
            if n_bad:
                bad_layers.append((layer.name, w.name, n_bad))

            mean_v = float(np.mean(vals[finite])) if finite.any() else float('nan')
            std_v  = float(np.std (vals[finite])) if finite.any() else float('nan')
            min_v  = float(np.min (vals[finite])) if finite.any() else float('nan')
            max_v  = float(np.max (vals[finite])) if finite.any() else float('nan')

            print(f"{layer.name:<40} {w.name:<30} {str(vals.shape):<20} "
                  f"{mean_v:>9.4f} {std_v:>9.4f} {min_v:>9.4f} {max_v:>9.4f}  {status}")

            if verbose:
                flat = vals.flatten()
                snippet = ", ".join(f"{v:.4f}" for v in flat[:8])
                print(f"    first 8 values: [{snippet}{'...' if flat.size > 8 else ''}]")

    print(SEP2)
    print(f"  Total weight values : {total_weights:,}")
    if bad_layers:
        print(f"  ✗ BAD LAYERS ({len(bad_layers)}):")
        for lname, wname, n in bad_layers:
            print(f"      {lname} / {wname} — {n} non-finite values")
    else:
        print("  ✓ All weights are finite.")
    print(SEP)
    print()


# ------------------------------------------------------------------ #
#  Public API expected by app.py                                      #
# ------------------------------------------------------------------ #
def processing(image: np.ndarray, threshold: float = 0.5, batch_size: int = 8) -> np.ndarray:
    """
    Parameters
    ----------
    image      : numpy uint8 RGB image (H x W x 3) — as delivered by app.py
    threshold  : probability cutoff for binarising predictions
    batch_size : inference batch size (all 4 patches are run together by default)

    Returns
    -------
    colour-coded RGB numpy array (uint8, orig_H x orig_W x 3)
    """
    model = _load_model()

    orig_h, orig_w = image.shape[:2]

    # Model was trained on grayscale — convert BGR→gray
    gray      = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    resized   = cv2.resize(gray, (1440, 960))

    # Build 4 patches of shape (480, 720, 1)
    patches = []
    for j in range(0, 960, 480):
        for k in range(0, 1440, 720):
            patch = resized[j:j+480, k:k+720] / 255.0
            patches.append(patch[..., np.newaxis])

    patches = np.array(patches, dtype=np.float32)   # (4, 480, 720, 1)
    preds   = model.predict(patches, batch_size=batch_size, verbose=0)
    preds   = np.nan_to_num(preds, nan=0.0, posinf=1.0, neginf=0.0)

    # Reconstruct full-resolution prediction
    n_ch  = preds.shape[-1]
    recon = np.zeros((960, 1440, n_ch), dtype=np.float32)
    idx   = 0
    for j in range(0, 960, 480):
        for k in range(0, 1440, 720):
            recon[j:j+480, k:k+720] = preds[idx]
            idx += 1

    # Colour-code and resize back to original dimensions
    color_rgb = _color_map(recon, threshold)
    final     = cv2.resize(color_rgb, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return final