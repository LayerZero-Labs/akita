//! Bounded overlap of serial rejection and ordered field conversion.

use akita_error::AkitaError;
use std::sync::mpsc::sync_channel;

pub(super) fn convert<E: Send>(
    count: usize,
    mut next_pair: impl FnMut() -> Result<[u64; 2], AkitaError>,
    convert: impl Fn([u64; 2]) -> E + Send,
    mut result: Vec<E>,
) -> Result<Vec<E>, AkitaError> {
    const BUFFER: usize = 1 << 14;
    let mut buffers = Vec::new();
    for _ in 0..2 {
        let mut buffer = Vec::new();
        buffer
            .try_reserve_exact(BUFFER)
            .map_err(|_| AkitaError::InvalidProof)?;
        buffers.push(buffer);
    }
    let (send, receive) = sync_channel::<Vec<[u64; 2]>>(1);
    let (recycle, available) = sync_channel(1);
    let parent = tracing::Span::current();
    std::thread::scope(|scope| {
        // Only conversion owns the output allocation. The caller keeps the
        // channel, logging and rejection cursor, even inside a one-worker pool.
        // A scoped OS worker cannot deadlock waiting on that same Rayon pool.
        let consumer = std::thread::Builder::new()
            .name("p64-conversion".into())
            .spawn_scoped(scope, move || {
                let _phase =
                    tracing::info_span!(parent: &parent, "binary_p64_batch_convert", count)
                        .entered();
                for mut buffer in receive {
                    result.extend(buffer.drain(..).map(&convert));
                    // The producer closes recycling before joining. Continue
                    // draining queued work even when no buffer is wanted back.
                    let _ = recycle.send(buffer);
                }
                result
            })
            .map_err(|_| AkitaError::InvalidInput("p64 conversion worker unavailable".into()))?;
        let produced = (|| {
            let mut remaining = count;
            while remaining != 0 {
                let mut buffer = if let Some(buffer) = buffers.pop() {
                    buffer
                } else {
                    available.recv().map_err(|_| AkitaError::InvalidProof)?
                };
                let len = remaining.min(BUFFER);
                for _ in 0..len {
                    buffer.push(next_pair()?);
                }
                send.send(buffer).map_err(|_| AkitaError::InvalidProof)?;
                remaining -= len;
            }
            Ok::<_, AkitaError>(())
        })();
        drop(send);
        drop(available);
        let converted = consumer.join().map_err(|_| AkitaError::InvalidProof);
        produced?;
        converted
    })
}
