"""
processing_lesion.py
=====================
Multi-lesion segmentation model (UNet + FIAM, trained on IDRiD).
Model file : Unet+FIAM_IDriD_1.2_300_cad.h5
Input      : RGB fundus image (numpy uint8, H x W x 3)
Output     : colour-coded RGB numpy array (uint8, H x W x 3)
               ch1 → Hard Exudates   Red     [255,   0,   0]
               ch2 → Hemorrhages     Green   [  0, 255,   0]
               ch3 → Microaneurysms  Blue    [  0,   0, 255]
               ch4 → Soft Exudates   Yellow  [255, 255,   0]

Called by app.py as:
    result_rgb = processing(image_cv2, threshold, batch_size)
"""

import os
import numpy as np
import cv2
import tensorflow as tf

# ------------------------------------------------------------------ #
#  Legacy / compatibility patches                                     #
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
import inspect
from tensorflow.keras import layers as _kl

def _make_compat_patches():
    """
    For each layer class listed below, strip the specified bad keys from
    both __init__ and from_config.  Safe to call multiple times — tracks
    already-patched classes to avoid double-wrapping.
    """
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

        # --- patch __init__ ---
        _orig_init = cls.__init__
        def _make_init(orig, keys):
            def _patched_init(self, *args, **kwargs):
                for k in keys:
                    kwargs.pop(k, None)
                orig(self, *args, **kwargs)
            return _patched_init
        cls.__init__ = _make_init(_orig_init, bad_keys)

        # --- patch from_config ---
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
#  Custom FIAM layer (numpy_function variant — more numerically       #
#  stable than the pure-TF version for this model)                   #
# ------------------------------------------------------------------ #
class FIAM(tf.keras.layers.Layer):
    def __init__(self, bins=300, **kwargs):
        super().__init__(**kwargs)
        self.bins = bins

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'bins': self.bins})
        return cfg

    def call(self, x):
        """
        Run FIAM via tf.py_function which reliably executes numpy code
        at inference time on TF >= 2.16. tf.numpy_function was silently
        producing constant outputs due to graph-tracing issues.
        """
        bins = self.bins

        def _fiam_numpy(x_np):
            result = x_np.copy()
            for img in range(x_np.shape[2]):
                ch    = x_np[:, :, img]
                denom = ch.max() - ch.min()
                if denom < 1e-6:
                    continue
                norm = np.clip((ch - ch.min()) / denom, 0.0, 1.0)
                if not np.isfinite(norm).all():
                    continue
                hist, _ = np.histogram(norm, bins=bins, range=(0.0, 1.0))
                hist    = hist.astype(np.float32)
                hist   /= hist.sum() + 1e-10
                temp    = ((hist[1:-1] - hist[:-2]) ** 2) / (hist[:-2] + 1e-10)
                th      = float(np.argmin(temp)) / 256.0
                fore    = np.where(norm >= th, norm, 0.0)
                back    = np.where(norm <  th, norm, 0.0)
                f1      = fore.flatten(); f1 = f1[f1 != 0]
                b1      = back.flatten(); b1 = b1[b1 != 0]
                if len(f1) == 0 or len(b1) == 0:
                    continue
                mask  = (fore.flatten() + back.flatten()) != 0
                f_new = fore.flatten()[mask]
                b_new = back.flatten()[mask]
                xm    = f_new - f_new.mean()
                ym    = b_new - b_new.mean()
                r_den = np.sqrt((xm**2).sum() * (ym**2).sum())
                corr  = float(np.clip((xm*ym).sum() / (r_den + 1e-5), -1.0, 1.0))
                WD = (f1.mean()-b1.mean())**2 + (f1.std()-b1.std())**2 + \
                     2.0*f1.std()*b1.std()*(1-corr)
                L  = (f1.mean()+b1.mean())**2 + (f1.std()+b1.std())**2 - \
                     2.0*f1.std()*b1.std()*(1-corr)
                B  = (f1.mean()**2-b1.mean()**2) + (f1.std()**2-b1.std()**2)
                if abs(L) < 1e-10:
                    continue
                disc = B**2 + 1.2*WD*L
                if disc < 0:
                    continue
                a = (-B + np.sqrt(disc)) / L
                if not np.isfinite(a):
                    continue
                new_ch = np.clip((1+a)*fore + (1-a)*back, 0.0, 1.0)
                if np.isfinite(new_ch).all():
                    result[:, :, img] = new_ch
            return result.astype(np.float32)

        def process_single(x_single):
            out = tf.py_function(
                func=lambda t: _fiam_numpy(t.numpy()),
                inp=[x_single],
                Tout=tf.float32,
            )
            out.set_shape(x_single.shape)
            return out

        return tf.map_fn(process_single, x, fn_output_signature=tf.float32)


# ------------------------------------------------------------------ #
#  Lazy model loader (singleton)                                     #
# ------------------------------------------------------------------ #
_model = None
MODEL_FILENAME = os.path.join('models', 'Unet+FIAM_IDriD_70epochs_1.2_300.h5')


def _load_model():
    global _model
    if _model is not None:
        return _model

    if not os.path.exists(MODEL_FILENAME):
        raise FileNotFoundError(
            f"Model file '{MODEL_FILENAME}' not found. "
            "Place it in the same directory as app.py."
        )

    custom_objects = {"FIAM": FIAM}
    with tf.keras.utils.custom_object_scope(custom_objects):
        _model = tf.keras.models.load_model(MODEL_FILENAME)

    # --- Patch NaN-producing BatchNorm layers ---
    for layer in _model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            w = layer.get_weights()
            if len(w) == 4:
                gamma, beta, mv_mean, mv_var = w
                mv_var  = np.where(np.isfinite(mv_var)  & (mv_var  > 1e-5), mv_var,  1.0)
                mv_mean = np.where(np.isfinite(mv_mean), mv_mean, 0.0)
                gamma   = np.where(np.isfinite(gamma),   gamma,   1.0)
                beta    = np.where(np.isfinite(beta),    beta,    0.0)
                layer.set_weights([gamma, beta, mv_mean, mv_var])

    # --- Patch any remaining NaN/Inf in all weights ---
    for layer in _model.layers:
        for wt in layer.weights:
            vals = wt.numpy()
            if not np.isfinite(vals).all():
                wt.assign(np.where(np.isfinite(vals), vals, 0.0))

    print(f"[LESION] Model loaded and patched from {MODEL_FILENAME}")
    return _model


# ------------------------------------------------------------------ #
#  Colour mapping                                                     #
# ------------------------------------------------------------------ #
# 5-channel output: ch0 = background, ch1-4 = lesion classes
_CHANNEL_COLORS = {
    1: (255,   0,   0),   # Hard Exudates  → Red
    2: (  0, 255,   0),   # Hemorrhages    → Green
    3: (  0,   0, 255),   # Microaneurysms → Blue
    4: (255, 255,   0),   # Soft Exudates  → Yellow
}


def _color_map(recon: np.ndarray, threshold: float) -> np.ndarray:
    """Convert (H, W, C) float32 prediction → (H, W, 3) uint8 RGB."""
    h, w = recon.shape[:2]
    out  = np.zeros((h, w, 3), dtype=np.uint8)
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
    Print a weight summary for the loaded LESION model to the terminal.

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
        from processing.processing_lesion import log_weights
        log_weights()          # summary
        log_weights(True)      # summary + first-8 values

    Or call it right after the model loads by adding to _load_model():
        log_weights()
    """
    model = _load_model()

    SEP  = "=" * 80
    SEP2 = "-" * 80

    print(f"\n{SEP}")
    print(f"  WEIGHT LOG — LESION model  ({MODEL_FILENAME})")
    print(f"  Total params : {model.count_params():,}")
    print(f"  Layers       : {len(model.layers)}")
    print(SEP)

    total_weights  = 0
    bad_layers     = []

    # Header
    print(f"{'Layer':<40} {'Weight':<30} {'Shape':<20} {'Mean':>9} {'Std':>9} "
          f"{'Min':>9} {'Max':>9}  {'Status'}")
    print(SEP2)

    for layer in model.layers:
        weights = layer.weights
        if not weights:
            continue
        for w in weights:
            vals        = w.numpy()
            total_weights += vals.size
            finite      = np.isfinite(vals)
            n_bad       = int(np.sum(~finite))
            status      = "✓ OK" if n_bad == 0 else f"✗ {n_bad} NaN/Inf"
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
def processing(image: np.ndarray, threshold: float = 0.5, batch_size: int = 1) -> np.ndarray:
    """
    Parameters
    ----------
    image      : numpy uint8 RGB image (H x W x 3) — as delivered by app.py
    threshold  : probability cutoff for binarising each lesion channel
    batch_size : inference batch size (1 recommended for this model)

    Returns
    -------
    colour-coded RGB numpy array (uint8, orig_H x orig_W x 3)
    Colours: Red=Hard Exudates, Green=Hemorrhages, Blue=MAs, Yellow=Soft Exudates
    """
    global _model
    _model = None   # force reload so new FIAM class is used
    model = _load_model()

    orig_h, orig_w = image.shape[:2]

    # Resize to model's expected resolution
    img_resized = cv2.resize(image, (1440, 960))

    # ------------------------------------------------------------------ #
    #  Preprocessing — must match IDRiD training pipeline exactly         #
    #                                                                      #
    #  IDRiD models are trained on the green channel with CLAHE applied.  #
    #  CLAHE (Contrast Limited Adaptive Histogram Equalization) boosts     #
    #  local contrast so lesions become distinguishable from background.   #
    #  Without it the model sees a flat low-contrast image and predicts    #
    #  everything as background (ch0 ~0.97 for all pixels).               #
    # ------------------------------------------------------------------ #

    # Step 1: extract green channel (index 1 in RGB)
    img_green = img_resized[:, :, 1]

    # Step 2: apply CLAHE — clipLimit and tileGridSize match IDRiD convention
    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_green = clahe.apply(img_green)

    print(f"[LESION DEBUG] green+CLAHE  min={img_green.min()}  "
          f"max={img_green.max()}  mean={img_green.mean():.2f}")

    # Build 4 non-overlapping patches of shape (480, 720, 1)
    patches = []
    for j in range(0, 960, 480):
        for k in range(0, 1440, 720):
            patch = img_green[j:j+480, k:k+720] / 255.0
            patches.append(patch[..., np.newaxis])

    patches = np.array(patches, dtype=np.float32)   # (4, 480, 720, 1)

    print(f"\n[LESION DEBUG] input patches  shape={patches.shape}  "
          f"min={patches.min():.4f}  max={patches.max():.4f}  mean={patches.mean():.4f}")

    preds = model.predict(patches, batch_size=batch_size, verbose=0)
    preds = np.nan_to_num(preds, nan=0.0, posinf=1.0, neginf=0.0)

    print(f"[LESION DEBUG] raw preds      shape={preds.shape}  "
          f"min={preds.min():.6f}  max={preds.max():.6f}  mean={preds.mean():.6f}")

    # Per-channel stats — key to diagnosing black output
    # ch0=background, ch1=Hard Exudates, ch2=Hemorrhages, ch3=MAs, ch4=Soft Exudates
    ch_names = {0: "Background", 1: "Hard Exudates", 2: "Hemorrhages",
                3: "Microaneurysms", 4: "Soft Exudates"}
    for c in range(preds.shape[-1]):
        ch = preds[:, :, :, c]
        above = int(np.sum(ch >= threshold))
        label = ch_names.get(c, f"ch{c}")
        print(f"[LESION DEBUG]   ch{c} ({label:<16})  "
              f"min={ch.min():.6f}  max={ch.max():.6f}  "
              f"mean={ch.mean():.6f}  pixels>={threshold}: {above}")

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
    pixels_colored = int(np.any(color_rgb > 0, axis=2).sum())
    print(f"[LESION DEBUG] colored pixels after threshold={threshold}: {pixels_colored}")
    if pixels_colored == 0:
        all_max = preds[:, :, :, 1:].max()
        print(f"[LESION DEBUG] *** BLACK OUTPUT — lesion channel max={all_max:.6f}. "
              f"Try lowering threshold below {all_max:.3f} ***")

    final = cv2.resize(color_rgb, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return final