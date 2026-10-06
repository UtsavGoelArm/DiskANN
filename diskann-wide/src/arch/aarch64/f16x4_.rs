/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::{
    Emulated,
    arch::Scalar,
    constant::Const,
    traits::{SIMDMask, SIMDVector},
};

use half::f16;

// AArch64 masks
use super::{
    Neon, internal,
    macros::{self, AArchLoadStore, AArchSplat},
    masks::mask16x4,
};

// AArch64 intrinsics
use std::arch::aarch64::*;

/////////////////////
// 16-bit floating //
/////////////////////

macros::aarch64_define_register!(f16x4, uint16x4_t, mask16x4, f16, 4, Neon);

impl std::ops::Sub for f16x4 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        if cfg!(miri) {
            let x = self.to_array();
            let y = rhs.to_array();
            Self::from_array(self.arch(), std::array::from_fn(|i| {
                crate::cast_f32_to_f16(crate::cast_f16_to_f32(x[i]) - crate::cast_f16_to_f32(y[i]))
            }))
        } else {
            // SAFETY: The `Neon` architecture requires FHM, which enables FP16.
            Self(unsafe {
                let x = vreinterpret_f16_u16(self.0);
                let y = vreinterpret_f16_u16(rhs.0);
                vreinterpret_u16_f16(vsub_f16(x, y))
            })
        }
    }
}

impl AArchSplat for f16x4 {
    #[inline(always)]
    fn aarch_splat(_: Neon, value: f16) -> Self {
        // SAFETY: Allowed by the `Neon` architecture.
        Self(unsafe { vmov_n_u16(value.to_bits()) })
    }

    #[inline(always)]
    fn aarch_default(arch: Neon) -> Self {
        Self::aarch_splat(arch, f16::default())
    }
}

impl AArchLoadStore for f16x4 {
    #[inline(always)]
    unsafe fn load_simd(_: Neon, ptr: *const f16) -> Self {
        // SAFETY: Allowed by the `Neon` architecture.
        Self(unsafe { vld1_u16(ptr.cast::<u16>()) })
    }

    #[inline(always)]
    unsafe fn load_simd_masked_logical(arch: Neon, ptr: *const f16, mask: Self::Mask) -> Self {
        // SAFETY: Pointer access safety inherited from the caller.
        let e = unsafe {
            Emulated::<f16, 4>::load_simd_masked_logical(Scalar, ptr, mask.bitmask().as_scalar())
        };
        Self::from_array(arch, e.to_array())
    }

    #[inline(always)]
    unsafe fn load_simd_first(arch: Neon, ptr: *const f16, first: usize) -> Self {
        // SAFETY: f16 and u16 share the same 2-byte representation. Pointer access
        // inherited from caller.
        Self(unsafe { internal::load_first::u16x4(arch, ptr.cast::<u16>(), first) })
    }

    #[inline(always)]
    unsafe fn store_simd(self, ptr: *mut <Self as SIMDVector>::Scalar) {
        // SAFETY: Pointer access safety inherited from the caller. Use of the instruction
        // is allowed by the `Neon` architecture.
        unsafe { vst1_u16(ptr.cast::<u16>(), self.0) }
    }

    #[inline(always)]
    unsafe fn store_simd_masked_logical(self, ptr: *mut f16, mask: Self::Mask) {
        let e = Emulated::<f16, 4>::from_array(Scalar, self.to_array());
        // SAFETY: Pointer access safety inherited from the caller.
        unsafe { e.store_simd_masked_logical(ptr, mask.bitmask().as_scalar()) }
    }

    #[inline(always)]
    unsafe fn store_simd_first(self, ptr: *mut f16, first: usize) {
        let e = Emulated::<f16, 4>::from_array(Scalar, self.to_array());
        // SAFETY: Pointer access safety inherited from the caller.
        unsafe { e.store_simd_first(ptr, first) }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{arch::aarch64::test_neon, test_utils};

    #[test]
    fn miri_test_load() {
        if let Some(arch) = test_neon() {
            test_utils::test_load_simd::<f16, 4, f16x4>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = test_neon() {
            test_utils::test_store_simd::<f16, 4, f16x4>(arch);
        }
    }

    #[test]
    fn test_sub() {
        if let Some(arch) = test_neon() {
            let x = f16x4::from_array(arch, [1.0, 2.0, 3.5, -4.0].map(f16::from_f32));
            let y = f16x4::from_array(arch, [0.5, -1.0, 1.0, 2.0].map(f16::from_f32));
            let expected = [0.5, 3.0, 2.5, -6.0].map(f16::from_f32);
            assert_eq!((x - y).to_array(), expected);
        }
    }

    // constructors
    #[test]
    fn test_constructors() {
        if let Some(arch) = test_neon() {
            test_utils::ops::test_splat::<f16, 4, f16x4>(arch);

            assert_eq!(
                f16x4::default(arch).to_array(),
                f16x4::splat(arch, f16::default()).to_array(),
            );
        }
    }
}
