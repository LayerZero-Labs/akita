//! Bounded four-stream squeezing for two rank-two p64 setup columns.

use super::{sample_p64_from, AkitaError, SetupStreamLabel, SetupStreamModulus, TypedSetupStream};

#[cfg(test)]
std::thread_local! {
    static FORCE_SCALAR: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn available() -> bool {
    #[cfg(test)]
    if FORCE_SCALAR.with(std::cell::Cell::get) {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    false
}

fn permute(streams: &mut [TypedSetupStream; 4]) {
    #[cfg(target_arch = "x86_64")]
    if available() {
        let mut state = std::array::from_fn(|word| {
            std::array::from_fn(|lane| streams[lane].squeeze.state[word])
        });
        // SAFETY: runtime detection above includes OS support for AVX2.
        unsafe { crate::keccak_four_way::permute(&mut state) };
        for (lane, stream) in streams.iter_mut().enumerate() {
            stream.squeeze.state = state.map(|word| word[lane]);
            stream.squeeze.refill();
        }
        return;
    }
    for stream in streams {
        stream.squeeze.permute();
    }
}

fn new_four(
    seed: &[u8; 32],
    labels: [SetupStreamLabel; 4],
) -> Result<[TypedSetupStream; 4], AkitaError> {
    let [a, b, c, d] = labels.map(|label| TypedSetupStream::unpermuted(seed, label));
    let mut streams = [a?, b?, c?, d?];
    permute(&mut streams);
    Ok(streams)
}

// Equal-size reads preserve each independent cursor, including partial blocks.
fn read_four(streams: &mut [TypedSetupStream; 4], mut outputs: [&mut [u8]; 4]) {
    loop {
        for (stream, output) in streams.iter_mut().zip(&mut outputs) {
            *output = stream.squeeze.take(std::mem::take(output));
        }
        if outputs.iter().all(|output| output.is_empty()) {
            return;
        }
        if outputs.iter().all(|output| !output.is_empty()) {
            permute(streams);
        } else {
            for (stream, output) in streams.iter_mut().zip(&mut outputs) {
                stream.squeeze.read(std::mem::take(output));
            }
            return;
        }
    }
}

impl TypedSetupStream {
    /// Fill four independent p64 entries, each with 64 coefficients.
    ///
    /// Labels retain their complete typed identities. Initialization and the
    /// minimum output are squeezed together; rejections continue each stream
    /// independently. Non-AVX2 hosts use the scalar reader throughout.
    ///
    /// # Errors
    /// Rejects non-p64 or invalid labels and the same per-coordinate rejection
    /// exhaustion as `sample_p64`. Outputs are local to this call on error.
    pub fn p64_four_entries(
        seed: &[u8; 32],
        labels: [SetupStreamLabel; 4],
    ) -> Result<[[u128; 64]; 4], AkitaError> {
        if labels
            .iter()
            .any(|label| label.modulus != SetupStreamModulus::Prime64Offset59)
        {
            return Err(AkitaError::InvalidSetup(
                "non-p64 setup stream requested as p64".into(),
            ));
        }
        if !available() {
            let mut entries = [[0; 64]; 4];
            for (label, entry) in labels.into_iter().zip(&mut entries) {
                Self::new(seed, label)?.fill_p(entry)?;
            }
            return Ok(entries);
        }
        let mut streams = new_four(seed, labels)?;
        let mut bytes = [[0; 8 * 64]; 4];
        read_four(&mut streams, bytes.each_mut().map(|lane| &mut lane[..]));
        let mut entries = [[0; 64]; 4];
        for ((stream, prefix), entry) in streams.iter_mut().zip(&bytes).zip(&mut entries) {
            fill_from_prefix(stream, prefix, entry)?;
        }
        Ok(entries)
    }
}

fn fill_from_prefix(
    stream: &mut TypedSetupStream,
    mut prefix: &[u8],
    output: &mut [u128],
) -> Result<(), AkitaError> {
    for slot in output {
        *slot = u128::from(sample_p64_from(|bytes| {
            let count = prefix.len().min(bytes.len());
            let (front, rest) = bytes.split_at_mut(count);
            let (head, tail) = prefix.split_at(count);
            front.copy_from_slice(head);
            prefix = tail;
            stream.squeeze.read(rest);
        })?);
    }
    Ok(())
}

#[cfg(test)]
mod tests;
