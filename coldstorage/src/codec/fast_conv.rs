//! Fast Conv2d / ConvTranspose2d using im2col + ndarray GEMM (→ BLAS).
//!
//! Burn's NdArray backend has two performance issues:
//!   1. Conv2d uses a naive rayon-parallelized loop — never calls BLAS.
//!   2. `into_data()` iterates the ndarray element-by-element (~16ns/elem).
//!
//! This module fixes both by:
//!   - Restructuring convolution as im2col → `general_mat_mul` → BLAS GEMM.
//!   - Extracting/creating tensors via direct ndarray access (memcpy, not iterator).
//!   - Using a `ConvDispatch` trait so NdArray gets the fast path while other
//!     backends (Wgpu) fall back to Burn's built-in conv.

use burn::backend::NdArray;
use burn::backend::wgpu::Wgpu;
use burn::nn::conv::{Conv2d, ConvTranspose2d};
use burn::nn::PaddingConfig2d;
use burn::prelude::*;
use burn::tensor::TensorPrimitive;
use burn_ndarray::{NdArrayTensor, NdArrayTensorFloat};
use ndarray::{ArcArray, Array2, ArrayView2, ArrayViewMut2, IxDyn};

// ── ConvDispatch trait ───────────────────────────────────────────────────────

/// Backend-specific convolution dispatch.
pub trait ConvDispatch: Backend {
    fn conv2d_fwd(conv: &Conv2d<Self>, input: Tensor<Self, 4>) -> Tensor<Self, 4>;
    fn conv_t2d_fwd(conv: &ConvTranspose2d<Self>, input: Tensor<Self, 4>) -> Tensor<Self, 4>;
}

impl ConvDispatch for NdArray {
    fn conv2d_fwd(conv: &Conv2d<Self>, input: Tensor<Self, 4>) -> Tensor<Self, 4> {
        conv2d_ndarray(conv, input)
    }
    fn conv_t2d_fwd(conv: &ConvTranspose2d<Self>, input: Tensor<Self, 4>) -> Tensor<Self, 4> {
        conv_t2d_ndarray(conv, input)
    }
}

impl ConvDispatch for Wgpu {
    fn conv2d_fwd(conv: &Conv2d<Self>, input: Tensor<Self, 4>) -> Tensor<Self, 4> {
        conv.forward(input)
    }
    fn conv_t2d_fwd(conv: &ConvTranspose2d<Self>, input: Tensor<Self, 4>) -> Tensor<Self, 4> {
        conv.forward(input)
    }
}

/// Autodiff<NdArray> — uses tensor-based im2col + matmul (BLAS-accelerated AND differentiable).
///
/// The raw ndarray fast path in `conv2d_ndarray` bypasses the autodiff tape.
/// This impl does im2col using Burn tensor slicing, then `Tensor::matmul()` which
/// routes through NdArray → ndarray → BLAS. All operations are tracked by autodiff
/// so gradients flow correctly.
impl ConvDispatch for burn::backend::Autodiff<NdArray> {
    fn conv2d_fwd(conv: &Conv2d<Self>, input: Tensor<Self, 4>) -> Tensor<Self, 4> {
        conv2d_tensor(conv, input)
    }
    fn conv_t2d_fwd(conv: &ConvTranspose2d<Self>, input: Tensor<Self, 4>) -> Tensor<Self, 4> {
        conv_t2d_tensor(conv, input)
    }
}

/// Convenience wrapper: dispatches to fast NdArray path or Burn fallback.
pub fn conv2d_forward<B: ConvDispatch>(conv: &Conv2d<B>, input: Tensor<B, 4>) -> Tensor<B, 4> {
    B::conv2d_fwd(conv, input)
}

/// Convenience wrapper: dispatches to fast NdArray path or Burn fallback.
pub fn conv_transpose2d_forward<B: ConvDispatch>(
    conv: &ConvTranspose2d<B>,
    input: Tensor<B, 4>,
) -> Tensor<B, 4> {
    B::conv_t2d_fwd(conv, input)
}

// ── NdArray fast tensor access ───────────────────────────────────────────────

/// Extract f32 data from a Tensor<NdArray, 4> by direct ndarray access.
/// This is ~100x faster than Burn's `into_data().to_vec()` for large tensors.
fn nd_extract(tensor: Tensor<NdArray, 4>) -> Vec<f32> {
    let float_prim = tensor.into_primitive().tensor();
    match float_prim {
        NdArrayTensorFloat::F32(t) => arc_to_vec_f32(t.array),
        NdArrayTensorFloat::F64(t) => t.array.iter().map(|&v| v as f32).collect(),
    }
}

/// Extract ArcArray<f32> to Vec<f32> via memcpy (not iterator).
fn arc_to_vec_f32(arr: ArcArray<f32, IxDyn>) -> Vec<f32> {
    // Fast path: contiguous array → try zero-copy, else memcpy via as_slice
    if arr.is_standard_layout() {
        match arr.try_into_owned_nocopy() {
            Ok(owned) => owned.into_raw_vec_and_offset().0,
            Err(arr) => arr.as_slice().unwrap().to_vec(),
        }
    } else {
        // Non-contiguous: must iterate (rare for our use case)
        arr.iter().copied().collect()
    }
}

/// Create a Tensor<NdArray, 4> from Vec<f32> without extra copies.
fn nd_create(data: Vec<f32>, shape: [usize; 4]) -> Tensor<NdArray, 4> {
    let array = ndarray::ArrayD::from_shape_vec(IxDyn(&shape), data)
        .unwrap()
        .into_shared();
    let nd_tensor = NdArrayTensor::new(array);
    Tensor::from_primitive(TensorPrimitive::Float(NdArrayTensorFloat::F32(nd_tensor)))
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn resolve_padding(cfg: &PaddingConfig2d, kh: usize, kw: usize) -> (usize, usize) {
    match cfg {
        PaddingConfig2d::Explicit(ph, pw) => (*ph, *pw),
        PaddingConfig2d::Same => (kh / 2, kw / 2),
        PaddingConfig2d::Valid => (0, 0),
    }
}

// ── Conv2d (NdArray) ─────────────────────────────────────────────────────────

fn conv2d_ndarray(conv: &Conv2d<NdArray>, input: Tensor<NdArray, 4>) -> Tensor<NdArray, 4> {
    let [batch, _c_in, h, w] = input.dims();
    let weight = conv.weight.val();
    let [c_out, c_in_k, kh, kw] = weight.dims();
    let [sh, sw] = conv.stride;
    let (ph, pw) = resolve_padding(&conv.padding, kh, kw);

    let out_h = (h + 2 * ph - kh) / sh + 1;
    let out_w = (w + 2 * pw - kw) / sw + 1;
    let col_k = c_in_k * kh * kw;

    // Fast extraction (memcpy, not iterator)
    let weight_vec = nd_extract(weight);
    let weight_nd = Array2::from_shape_vec((c_out, col_k), weight_vec).unwrap();
    let input_data = nd_extract(input);

    let spatial_total = out_h * out_w;
    let out_total = batch * c_out * spatial_total;
    let mut output_flat = vec![0.0f32; out_total];

    // Tile by output rows.  128 MiB per tile.
    let max_floats = 32 * 1024 * 1024;
    let tile_rows = (max_floats / (col_k * out_w)).max(1).min(out_h);
    let max_tile_spatial = tile_rows * out_w;

    let mut col_buf = vec![0.0f32; col_k * max_tile_spatial];
    let mut out_buf = vec![0.0f32; c_out * max_tile_spatial];

    for b_idx in 0..batch {
        let in_off = b_idx * _c_in * h * w;

        for tile_start in (0..out_h).step_by(tile_rows) {
            let tile_end = (tile_start + tile_rows).min(out_h);
            let tile_h = tile_end - tile_start;
            let tile_spatial = tile_h * out_w;

            col_buf[..col_k * tile_spatial].fill(0.0);

            // ── im2col (channel-outer for cache locality) ─────────
            for c in 0..c_in_k {
                let in_c_base = in_off + c * h * w;
                for kkh in 0..kh {
                    let oh_min = if kkh < ph { (ph - kkh).div_ceil(sh) } else { 0 };
                    let oh_max_p1 =
                        if h + ph > kkh { (h - 1 + ph - kkh) / sh + 1 } else { 0 };
                    let th_lo = oh_min.saturating_sub(tile_start).min(tile_h);
                    let th_hi = oh_max_p1.saturating_sub(tile_start).min(tile_h);

                    for kkw in 0..kw {
                        let col_row = (c * kh * kw + kkh * kw + kkw) * tile_spatial;
                        let ow_lo = if kkw < pw { (pw - kkw).div_ceil(sw) } else { 0 };
                        let ow_hi = if w + pw > kkw {
                            ((w - 1 + pw - kkw) / sw + 1).min(out_w)
                        } else {
                            0
                        };
                        let iw_base = ow_lo * sw + kkw - pw;

                        for th in th_lo..th_hi {
                            let oh = tile_start + th;
                            let ih = oh * sh + kkh - ph;
                            let in_row = in_c_base + ih * w;
                            let col_dst = col_row + th * out_w;

                            let mut iw = iw_base;
                            for ow_idx in ow_lo..ow_hi {
                                col_buf[col_dst + ow_idx] = input_data[in_row + iw];
                                iw += sw;
                            }
                        }
                    }
                }
            }

            // ── GEMM via ndarray → BLAS ───────────────────────────
            let col_view = ArrayView2::from_shape(
                (col_k, tile_spatial),
                &col_buf[..col_k * tile_spatial],
            )
            .unwrap();
            let mut out_view = ArrayViewMut2::from_shape(
                (c_out, tile_spatial),
                &mut out_buf[..c_out * tile_spatial],
            )
            .unwrap();
            ndarray::linalg::general_mat_mul(
                1.0f32,
                &weight_nd.view(),
                &col_view,
                0.0f32,
                &mut out_view,
            );

            // ── Scatter tile into output ──────────────────────────
            let out_off = b_idx * c_out * spatial_total;
            for c in 0..c_out {
                let dst = out_off + c * spatial_total + tile_start * out_w;
                let src = c * tile_spatial;
                output_flat[dst..dst + tile_spatial]
                    .copy_from_slice(&out_buf[src..src + tile_spatial]);
            }
        }
    }

    // Fast creation (wraps Vec, no extra copy)
    let mut result = nd_create(output_flat, [batch, c_out, out_h, out_w]);

    if let Some(ref bias) = conv.bias {
        let bias_4d = bias.val().reshape([1, c_out, 1, 1]);
        result = result + bias_4d;
    }

    result
}

// ── ConvTranspose2d (NdArray) ────────────────────────────────────────────────

fn conv_t2d_ndarray(conv: &ConvTranspose2d<NdArray>, input: Tensor<NdArray, 4>) -> Tensor<NdArray, 4> {
    let [batch, c_in, h_in, w_in] = input.dims();
    let weight = conv.weight.val();
    let [_c_in_w, c_out_k, kh, kw] = weight.dims();
    let [sh, sw] = conv.stride;
    let [po_h, po_w] = conv.padding_out;
    let [ph, pw] = conv.padding;

    let out_h = (h_in - 1) * sh - 2 * ph + kh + po_h;
    let out_w = (w_in - 1) * sw - 2 * pw + kw + po_w;
    let col_k = c_out_k * kh * kw;

    let weight_vec = nd_extract(weight);
    let weight_nd = Array2::from_shape_vec((c_in, col_k), weight_vec).unwrap();
    let input_data = nd_extract(input);

    let spatial_in = h_in * w_in;
    let spatial_out = out_h * out_w;
    let out_total = batch * c_out_k * spatial_out;
    let mut output_flat = vec![0.0f32; out_total];

    let max_floats = 32 * 1024 * 1024;
    let tile_rows = (max_floats / (col_k * w_in)).max(1).min(h_in);
    let max_tile_spatial = tile_rows * w_in;

    let mut in_tile = vec![0.0f32; c_in * max_tile_spatial];
    let mut col_buf = vec![0.0f32; col_k * max_tile_spatial];

    for b_idx in 0..batch {
        let in_off = b_idx * c_in * spatial_in;

        for tile_start in (0..h_in).step_by(tile_rows) {
            let tile_end = (tile_start + tile_rows).min(h_in);
            let tile_h = tile_end - tile_start;
            let tile_spatial = tile_h * w_in;

            for c in 0..c_in {
                let src = in_off + c * spatial_in + tile_start * w_in;
                let dst = c * tile_spatial;
                in_tile[dst..dst + tile_spatial]
                    .copy_from_slice(&input_data[src..src + tile_spatial]);
            }

            // ── GEMM ─────────────────────────────────────────────
            let in_view = ArrayView2::from_shape(
                (c_in, tile_spatial),
                &in_tile[..c_in * tile_spatial],
            )
            .unwrap();
            let mut col_view = ArrayViewMut2::from_shape(
                (col_k, tile_spatial),
                &mut col_buf[..col_k * tile_spatial],
            )
            .unwrap();
            ndarray::linalg::general_mat_mul(
                1.0f32,
                &weight_nd.t(),
                &in_view,
                0.0f32,
                &mut col_view,
            );

            // ── col2im (channel-outer for sequential col reads) ───
            let out_off = b_idx * c_out_k * spatial_out;
            for c in 0..c_out_k {
                let out_c_base = out_off + c * spatial_out;
                for kkh in 0..kh {
                    let ih_min = if kkh < ph { (ph - kkh).div_ceil(sh) } else { 0 };
                    let ih_max_p1 = if out_h + ph > kkh {
                        (out_h - 1 + ph - kkh) / sh + 1
                    } else {
                        0
                    };
                    let th_lo = ih_min.saturating_sub(tile_start).min(tile_h);
                    let th_hi = ih_max_p1.saturating_sub(tile_start).min(tile_h);

                    for kkw in 0..kw {
                        let col_row = (c * kh * kw + kkh * kw + kkw) * tile_spatial;
                        let iw_lo = if kkw < pw { (pw - kkw).div_ceil(sw) } else { 0 };
                        let iw_hi = if out_w + pw > kkw {
                            ((out_w - 1 + pw - kkw) / sw + 1).min(w_in)
                        } else {
                            0
                        };

                        for th in th_lo..th_hi {
                            let ih = tile_start + th;
                            let oh = ih * sh + kkh - ph;
                            let out_row = out_c_base + oh * out_w;
                            let col_src = col_row + th * w_in;

                            let mut ow = iw_lo * sw + kkw - pw;
                            for iw in iw_lo..iw_hi {
                                output_flat[out_row + ow] += col_buf[col_src + iw];
                                ow += sw;
                            }
                        }
                    }
                }
            }
        }
    }

    let mut result = nd_create(output_flat, [batch, c_out_k, out_h, out_w]);

    if let Some(ref bias) = conv.bias {
        let bias_4d = bias.val().reshape([1, c_out_k, 1, 1]);
        result = result + bias_4d;
    }

    result
}

// ── Tensor-based im2col Conv2d (any backend, autodiff-compatible) ──────────
//
// Restructures convolution as im2col → matmul.  All operations use Burn tensor
// ops, so the autodiff tape records every step.  The matmul still routes to
// BLAS on NdArray (via Autodiff<NdArray>), giving most of the speed benefit
// of the raw ndarray path while keeping full gradient support.

/// im2col via tensor slicing: for each kernel position (ky, kx), slice the
/// input at the correct stride/padding offsets and stack along the channel dim.
///
/// Input:  [B, C_in, H, W]  (already padded if needed)
/// Output: [B, C_in * kH * kW, out_H * out_W]
fn tensor_im2col<B: Backend>(
    input: &Tensor<B, 4>,
    kh: usize, kw: usize,
    sh: usize, sw: usize,
    out_h: usize, out_w: usize,
) -> Tensor<B, 3> {
    let [batch, c_in, _h, _w] = input.dims();

    // Collect one [B, C_in, out_H, out_W] slice per kernel element
    let mut columns: Vec<Tensor<B, 4>> = Vec::with_capacity(kh * kw);

    for ky in 0..kh {
        for kx in 0..kw {
            // For each output row oh: input row = oh*sh + ky
            // For each output col ow: input col = ow*sw + kx
            // We need out_h rows starting at ky, stepping by sh
            // and out_w cols starting at kx, stepping by sw

            // Build index ranges for the input slice
            let row_end = ky + (out_h - 1) * sh + 1;
            let col_end = kx + (out_w - 1) * sw + 1;

            // Slice the full range then subsample by stride
            let patch = input.clone().slice([0..batch, 0..c_in, ky..row_end, kx..col_end]);

            // Subsample by stride if stride > 1
            let patch = if sh > 1 || sw > 1 {
                // Use narrow slicing: take every sh-th row and sw-th col
                // Burn doesn't have stride-slice, so we gather using Tensor ops.
                // For our model's strides (1 or 2), this is efficient.
                stride_subsample(&patch, sh, sw, out_h, out_w)
            } else {
                patch
            };

            columns.push(patch); // [B, C_in, out_H, out_W]
        }
    }

    // Stack: [kH*kW] × [B, C_in, out_H, out_W] → [B, C_in * kH*kW, out_H * out_W]
    // First cat along dim=1 (channels): [B, C_in * kH * kW, out_H, out_W]
    let stacked = Tensor::cat(columns, 1);
    let total_k = c_in * kh * kw;
    stacked.reshape([batch, total_k, out_h * out_w])
}

/// Subsample a tensor by strides (sh, sw) along the spatial dims.
/// Uses `Tensor::select` — a single indexed-gather per dimension (no per-pixel loops).
///
/// Input: [B, C, H', W'] where H' >= (out_h-1)*sh+1, W' >= (out_w-1)*sw+1
/// Output: [B, C, out_h, out_w]
fn stride_subsample<B: Backend>(
    x: &Tensor<B, 4>,
    sh: usize, sw: usize,
    out_h: usize, out_w: usize,
) -> Tensor<B, 4> {
    if sh == 1 && sw == 1 {
        return x.clone();
    }

    let device = x.device();
    let mut result = x.clone();

    if sh > 1 {
        let indices: Vec<i32> = (0..out_h).map(|i| (i * sh) as i32).collect();
        let idx = Tensor::<B, 1, Int>::from_ints(indices.as_slice(), &device);
        result = result.select(2, idx);
    }

    if sw > 1 {
        let indices: Vec<i32> = (0..out_w).map(|i| (i * sw) as i32).collect();
        let idx = Tensor::<B, 1, Int>::from_ints(indices.as_slice(), &device);
        result = result.select(3, idx);
    }

    result
}

/// Conv2d using tensor-based im2col + matmul.  Works on any backend.
fn conv2d_tensor<B: Backend>(conv: &Conv2d<B>, input: Tensor<B, 4>) -> Tensor<B, 4> {
    let [batch, _c_in, h, w] = input.dims();
    let weight = conv.weight.val();
    let [c_out, c_in_k, kh, kw] = weight.dims();
    let [sh, sw] = conv.stride;
    let (ph, pw) = resolve_padding(&conv.padding, kh, kw);

    let out_h = (h + 2 * ph - kh) / sh + 1;
    let out_w = (w + 2 * pw - kw) / sw + 1;

    // Pad input if needed
    let padded = if ph > 0 || pw > 0 {
        let device = input.device();
        let padded_h = h + 2 * ph;
        let padded_w = w + 2 * pw;
        let mut padded = Tensor::<B, 4>::zeros([batch, _c_in, padded_h, padded_w], &device);
        padded = padded.slice_assign(
            [0..batch, 0.._c_in, ph..ph + h, pw..pw + w],
            input,
        );
        padded
    } else {
        input
    };

    // im2col: [B, C_in*kH*kW, out_H*out_W]
    let col = tensor_im2col(&padded, kh, kw, sh, sw, out_h, out_w);

    // Weight: [C_out, C_in, kH, kW] → permute to [C_out, kH, kW, C_in] → reshape to [C_out, kH*kW*C_in]
    // This matches tensor_im2col's column ordering: (ky, kx, c_in) per patch.
    let col_k = c_in_k * kh * kw;
    let w_perm = weight.swap_dims(1, 2).swap_dims(2, 3); // [C_out, kH, kW, C_in]
    let w_2d = w_perm.reshape([c_out, col_k]);

    // GEMM: [C_out, C_in*kH*kW] × [B, C_in*kH*kW, out_H*out_W]
    // Expand weight to batch dim: [1, C_out, col_k] → broadcast
    let w_3d = w_2d.unsqueeze::<3>(); // [1, C_out, col_k]
    let result = w_3d.matmul(col); // [B, C_out, out_H*out_W]

    // Reshape to [B, C_out, out_H, out_W]
    let mut output = result.reshape([batch, c_out, out_h, out_w]);

    // Add bias
    if let Some(ref bias) = conv.bias {
        let bias_4d = bias.val().reshape([1, c_out, 1, 1]);
        output = output + bias_4d;
    }

    output
}

/// Upsample by inserting (stride-1) zeros between elements along spatial dims.
///
/// Uses interleave-via-reshape: no loops, O(1) tensor ops.
///
/// Input:  [B, C, H, W]
/// Output: [B, C, (H-1)*sh+1, (W-1)*sw+1]
fn upsample_zeros<B: Backend>(input: Tensor<B, 4>, sh: usize, sw: usize) -> Tensor<B, 4> {
    if sh == 1 && sw == 1 {
        return input;
    }

    let [batch, c, h, w] = input.dims();
    let device = input.device();
    let up_h = (h - 1) * sh + 1;
    let up_w = (w - 1) * sw + 1;

    // Upsample columns: interleave each value with (sw-1) zeros
    // [B,C,H,W] → unsqueeze → [B,C,H,W,1] cat zeros → [B,C,H,W,sw] → reshape → [B,C,H,W*sw] → trim
    let x = if sw > 1 {
        let x: Tensor<B, 5> = input.unsqueeze_dim(4);                         // [B,C,H,W,1]
        let z = Tensor::<B, 5>::zeros([batch, c, h, w, sw - 1], &device);     // [B,C,H,W,sw-1]
        let x = Tensor::cat(vec![x, z], 4);                                   // [B,C,H,W,sw]
        let x = x.reshape([batch, c, h, w * sw]);                             // [B,C,H,W*sw]
        x.slice([0..batch, 0..c, 0..h, 0..up_w])                             // [B,C,H,up_w]
    } else {
        input
    };

    // Upsample rows: same trick along dim 2
    if sh > 1 {
        let [_, _, h2, w2] = x.dims();
        let x: Tensor<B, 5> = x.unsqueeze_dim(3);                             // [B,C,H,1,W']
        let z = Tensor::<B, 5>::zeros([batch, c, h2, sh - 1, w2], &device);   // [B,C,H,sh-1,W']
        let x = Tensor::cat(vec![x, z], 3);                                   // [B,C,H,sh,W']
        let x = x.reshape([batch, c, h2 * sh, w2]);                           // [B,C,H*sh,W']
        x.slice([0..batch, 0..c, 0..up_h, 0..w2])                            // [B,C,up_h,W']
    } else {
        x
    }
}

/// ConvTranspose2d using upsample + im2col + matmul.  Works on any backend.
///
/// Strategy: ConvTranspose2d(x, W, stride=S, padding=P) is equivalent to:
///   1. Upsample x by inserting (S-1) zeros between elements
///   2. Conv2d(upsampled, flip(W^T), stride=1, padding=kH-1-P)
///
/// This avoids the problematic col2im scatter loop entirely.
fn conv_t2d_tensor<B: Backend>(conv: &ConvTranspose2d<B>, input: Tensor<B, 4>) -> Tensor<B, 4> {
    let [batch, c_in, h_in, w_in] = input.dims();
    let weight = conv.weight.val(); // [C_in, C_out, kH, kW]
    let [_c_in_w, c_out, kh, kw] = weight.dims();
    let [sh, sw] = conv.stride;
    let [po_h, po_w] = conv.padding_out;
    let [ph, pw] = conv.padding;

    let out_h = (h_in - 1) * sh - 2 * ph + kh + po_h;
    let out_w = (w_in - 1) * sw - 2 * pw + kw + po_w;

    // 1. Upsample: insert (stride-1) zeros between input elements
    let upsampled = upsample_zeros(input, sh, sw);

    // 1b. Add output_padding extra zeros at the end of the upsampled input.
    //     This is the correct way to handle output_padding in the upsample equivalence:
    //     the conv kernel must operate on those positions (not just append zeros to output).
    let upsampled = if po_h > 0 || po_w > 0 {
        let [_, _, uh, uw] = upsampled.dims();
        let ext_h = uh + po_h;
        let ext_w = uw + po_w;
        let device = upsampled.device();
        let mut ext = Tensor::<B, 4>::zeros([batch, c_in, ext_h, ext_w], &device);
        ext = ext.slice_assign([0..batch, 0..c_in, 0..uh, 0..uw], upsampled);
        ext
    } else {
        upsampled
    };

    let [_, _, up_h, up_w] = upsampled.dims();

    // 2. Flip weight spatially and transpose channel dims
    //    [C_in, C_out, kH, kW] → [C_out, C_in, kH, kW] with kernel reversed
    let w_transposed = weight.swap_dims(0, 1);  // [C_out, C_in, kH, kW]
    let w_flipped = w_transposed.flip([2, 3]);   // reverse spatial dims

    // 3. Equivalent conv2d padding: (kH-1-P) for each spatial dim
    let conv_ph = kh - 1 - ph;
    let conv_pw = kw - 1 - pw;

    // 4. Pad upsampled input for the equivalent conv2d
    let padded = if conv_ph > 0 || conv_pw > 0 {
        let device = upsampled.device();
        let padded_h = up_h + 2 * conv_ph;
        let padded_w = up_w + 2 * conv_pw;
        let mut p = Tensor::<B, 4>::zeros([batch, c_in, padded_h, padded_w], &device);
        p = p.slice_assign(
            [0..batch, 0..c_in, conv_ph..conv_ph + up_h, conv_pw..conv_pw + up_w],
            upsampled,
        );
        p
    } else {
        upsampled
    };

    // 5. im2col + matmul (stride=1, since stride is handled by upsampling)
    let col = tensor_im2col(&padded, kh, kw, 1, 1, out_h, out_w);

    // Permute to [C_out, kH, kW, C_in] to match tensor_im2col's (ky, kx, c_in) ordering
    let col_k = c_in * kh * kw;
    let w_perm = w_flipped.swap_dims(1, 2).swap_dims(2, 3); // [C_out, kH, kW, C_in]
    let w_2d = w_perm.reshape([c_out, col_k]);
    let w_3d = w_2d.unsqueeze::<3>(); // [1, C_out, col_k]
    let result = w_3d.matmul(col);     // [B, C_out, out_H * out_W]

    let mut output = result.reshape([batch, c_out, out_h, out_w]);

    // 7. Add bias
    if let Some(ref bias) = conv.bias {
        let bias_4d = bias.val().reshape([1, c_out, 1, 1]);
        output = output + bias_4d;
    }

    output
}
