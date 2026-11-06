//! Bounded memo table for interval suffix values. Every probe and allocation is
//! charged to the contraction budget, including collision and rehash work.
use super::charge_work;
use akita_error::AkitaError;

pub(super) struct Memo<F: Copy> {
    slots: Vec<Option<(usize, F)>>,
    used: usize,
    shift: u32,
}
impl<F: Copy> Memo<F> {
    pub(super) fn new() -> Self {
        Self {
            slots: Vec::new(),
            used: 0,
            shift: 0,
        }
    }
    fn bucket(&self, key: usize) -> usize {
        key.wrapping_mul(0x9e3779b97f4a7c15u64 as usize) >> self.shift
    }
    fn insert_free(&mut self, key: usize, value: F, work: &mut usize) -> Result<(), AkitaError> {
        let mut at = self.bucket(key);
        for _ in 0..self.slots.len() {
            charge_work(work, 1)?;
            let slot = self
                .slots
                .get_mut(at)
                .ok_or_else(|| AkitaError::Internal("interval memo slot is absent".into()))?;
            if slot.is_none() {
                *slot = Some((key, value));
                self.used += 1;
                return Ok(());
            }
            if slot.as_ref().is_some_and(|(found, _)| *found == key) {
                *slot = Some((key, value));
                return Ok(());
            }
            at = (at + 1) & (self.slots.len() - 1);
        }
        Err(AkitaError::Internal("interval memo is full".into()))
    }
    pub(super) fn get_or_compute(
        &mut self,
        key: usize,
        work: &mut usize,
        compute: impl FnOnce(&mut usize) -> Result<F, AkitaError>,
    ) -> Result<F, AkitaError> {
        if !self.slots.is_empty() {
            let at = self.find_slot(key, work)?;
            if let Some((_, value)) = self.slots.get(at).copied().flatten() {
                return Ok(value);
            }
            if self.used < self.slots.len() / 2 {
                let value = compute(work)?;
                *self
                    .slots
                    .get_mut(at)
                    .ok_or_else(|| AkitaError::Internal("memo vacant slot is absent".into()))? =
                    Some((key, value));
                self.used += 1;
                return Ok(value);
            }
        }
        self.grow(work)?;
        let at = self.find_slot(key, work)?;
        let value = compute(work)?;
        *self
            .slots
            .get_mut(at)
            .ok_or_else(|| AkitaError::Internal("memo vacant slot is absent".into()))? =
            Some((key, value));
        self.used += 1;
        Ok(value)
    }

    fn find_slot(&self, key: usize, work: &mut usize) -> Result<usize, AkitaError> {
        let mut at = self.bucket(key);
        for _ in 0..self.slots.len() {
            charge_work(work, 1)?;
            match self.slots.get(at).copied().flatten() {
                Some((found, _)) if found == key => return Ok(at),
                None => return Ok(at),
                _ => at = (at + 1) & (self.slots.len() - 1),
            }
        }
        Err(AkitaError::Internal(
            "interval memo has no vacant slot".into(),
        ))
    }

    fn grow(&mut self, work: &mut usize) -> Result<(), AkitaError> {
        let capacity = self
            .slots
            .len()
            .checked_mul(2)
            .map(|n| n.max(8))
            .ok_or_else(|| AkitaError::InvalidInput("interval memo size overflow".into()))?;
        charge_work(work, capacity)?;
        let mut slots = Vec::new();
        slots
            .try_reserve_exact(capacity)
            .map_err(|_| AkitaError::InvalidInput("interval memo allocation failed".into()))?;
        slots.resize(capacity, None);
        let old = std::mem::replace(&mut self.slots, slots);
        self.used = 0;
        self.shift = usize::BITS - capacity.trailing_zeros();
        for (key, value) in old.into_iter().flatten() {
            self.insert_free(key, value, work)?;
        }
        Ok(())
    }
}
