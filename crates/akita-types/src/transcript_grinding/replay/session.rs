//! Private ownership boundary for a grinding replay session.

use std::ops::{Deref, DerefMut};

pub(super) enum ChannelStorage<'a, T: DerefMut> {
    Owned(T),
    Borrowed(&'a mut T::Target),
}

impl<T: DerefMut> Deref for ChannelStorage<'_, T> {
    type Target = T::Target;

    fn deref(&self) -> &T::Target {
        match self {
            Self::Owned(state) => state,
            Self::Borrowed(state) => state,
        }
    }
}

impl<T: DerefMut> DerefMut for ChannelStorage<'_, T> {
    fn deref_mut(&mut self) -> &mut T::Target {
        match self {
            Self::Owned(state) => state,
            Self::Borrowed(state) => state,
        }
    }
}
