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
