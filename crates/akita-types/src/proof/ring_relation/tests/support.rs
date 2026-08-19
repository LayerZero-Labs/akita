pub(super) fn marker<const N: usize>(index: usize) -> [i8; N] {
    let value = (index % 100 + 1) as i8;
    std::array::from_fn(|coefficient| {
        if coefficient.is_multiple_of(2) {
            value
        } else {
            -value
        }
    })
}

pub(super) fn flatten_markers<const N: usize>(
    markers: impl IntoIterator<Item = [i8; N]>,
) -> Vec<i8> {
    markers.into_iter().flatten().collect()
}
