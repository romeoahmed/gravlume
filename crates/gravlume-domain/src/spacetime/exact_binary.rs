use std::cmp::Ordering;

use crate::numerics::{binary_power, binary64_magnitude};

// A finite binary64 magnitude is `significand * 2^exponent` with a 53-bit
// significand and exponent in `[-1074, 971]`. Sixty-six limbs therefore hold
// the exact sum of any two squared magnitudes, including its carry bit.
const LIMBS: usize = 66;
const MINIMUM_SQUARED_EXPONENT: i32 = -2148;
const WORD_BITS: usize = 64;

#[derive(Clone, Eq, PartialEq)]
pub struct ExactBinary {
    limbs: [u64; LIMBS],
}

impl ExactBinary {
    pub fn square(value: f64) -> Self {
        let mut result = Self::default();
        result.add_square(value);
        result
    }

    pub fn sum_of_squares(first: f64, second: f64) -> Self {
        let mut result = Self::default();
        result.add_square(first);
        result.add_square(second);
        result
    }

    fn add_square(&mut self, value: f64) {
        let (significand, exponent) = binary64_magnitude(value.abs());
        if significand == 0 {
            return;
        }
        let coefficient = u128::from(significand) * u128::from(significand);
        let shift = usize::try_from(2 * exponent - MINIMUM_SQUARED_EXPONENT).unwrap_or_default();
        self.add_shifted(coefficient, shift);
    }

    fn add_shifted(&mut self, coefficient: u128, shift: usize) {
        let word = shift / WORD_BITS;
        let bit = shift % WORD_BITS;
        let low = u64::try_from(coefficient & u128::from(u64::MAX)).unwrap_or_default();
        let high = u64::try_from(coefficient >> WORD_BITS).unwrap_or_default();
        if bit == 0 {
            self.add_word(word, low);
            self.add_word(word + 1, high);
        } else {
            self.add_word(word, low << bit);
            self.add_word(word + 1, low >> (WORD_BITS - bit));
            self.add_word(word + 1, high << bit);
            self.add_word(word + 2, high >> (WORD_BITS - bit));
        }
    }

    fn add_word(&mut self, mut index: usize, mut value: u64) {
        while value != 0 {
            let (sum, carry) = self.limbs[index].overflowing_add(value);
            self.limbs[index] = sum;
            value = u64::from(carry);
            index += 1;
        }
    }

    pub fn subtract(&self, smaller: &Self) -> Self {
        let mut result = Self::default();
        let mut borrow = false;
        for index in 0..LIMBS {
            let (difference, first_borrow) =
                self.limbs[index].overflowing_sub(smaller.limbs[index]);
            let (difference, second_borrow) = difference.overflowing_sub(u64::from(borrow));
            result.limbs[index] = difference;
            borrow = first_borrow || second_borrow;
        }
        debug_assert!(!borrow);
        result
    }

    pub fn square_root(&self) -> f64 {
        let highest_bit = self.highest_bit().unwrap_or_default();
        let mut significand = 0_u64;
        for offset in 0..f64::MANTISSA_DIGITS as usize {
            significand <<= 1;
            if highest_bit >= offset && self.bit(highest_bit - offset) {
                significand |= 1;
            }
        }
        let discarded_bits = highest_bit.saturating_sub(f64::MANTISSA_DIGITS as usize - 1);
        if discarded_bits > 0 {
            let guard = self.bit(discarded_bits - 1);
            let sticky = (0..discarded_bits - 1).any(|bit| self.bit(bit));
            if guard && (sticky || significand & 1 == 1) {
                significand += 1;
            }
        }
        let high = u32::try_from(significand >> 32).unwrap_or_default();
        let low = u32::try_from(significand & u64::from(u32::MAX)).unwrap_or_default();
        let normalized =
            f64::from(high).mul_add(2.0_f64.powi(32), f64::from(low)) / 2.0_f64.powi(52);
        let exponent = MINIMUM_SQUARED_EXPONENT + i32::try_from(highest_bit).unwrap_or_default();
        if exponent & 1 == 0 {
            normalized.sqrt() * binary_power(exponent / 2)
        } else {
            (2.0 * normalized).sqrt() * binary_power((exponent - 1) / 2)
        }
    }

    fn highest_bit(&self) -> Option<usize> {
        self.limbs.iter().rposition(|limb| *limb != 0).map(|word| {
            word * WORD_BITS
                + usize::try_from(u64::BITS - 1 - self.limbs[word].leading_zeros())
                    .unwrap_or_default()
        })
    }

    const fn bit(&self, index: usize) -> bool {
        let word = index / WORD_BITS;
        let bit = index % WORD_BITS;
        self.limbs[word] & (1_u64 << bit) != 0
    }
}

impl Default for ExactBinary {
    fn default() -> Self {
        Self { limbs: [0; LIMBS] }
    }
}

impl PartialOrd for ExactBinary {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ExactBinary {
    fn cmp(&self, other: &Self) -> Ordering {
        self.limbs.iter().rev().cmp(other.limbs.iter().rev())
    }
}
