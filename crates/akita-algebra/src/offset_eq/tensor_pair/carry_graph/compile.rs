#[cfg(test)]
use super::Graph;
use super::{Axis, Shape, WeightKind};
use crate::offset_eq::tensor_pair::evaluate::{carry_step, introduction_geometry};
use akita_error::AkitaError;
use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

mod residual;
pub(super) mod shapes;

const MAX_EDGES: usize = 65_536;
const MAX_STATES: usize = 1024;

pub(super) struct OwnedGraph<'a> {
    shape: Shape<'a>,
    layers: Vec<[u32; 3]>,
    introductions: Vec<u16>,
    edges: Vec<u32>,
    peak: usize,
    residual: Option<usize>,
    seeds: usize,
    terminal: Option<usize>,
    work: usize,
}

impl<'a> OwnedGraph<'a> {
    #[cfg(test)]
    pub(super) fn view(&self) -> Graph<'_> {
        Graph {
            shape: self.shape,
            layers: &self.layers,
            introductions: &self.introductions,
            edges: &self.edges,
            peak: self.peak as u16,
            residual: self.residual.map(|axis| axis as u16),
            seeds: self.seeds as u16,
            terminal: self.terminal.map(|slot| slot as u16),
            work: self.work as u32,
        }
    }

    pub(super) fn compile(shape: Shape<'a>) -> Result<Option<Self>, AkitaError> {
        if shape.bits.iter().any(|&bits| bits >= 32)
            || shape.axes.len() > 255
            || shape.axes.iter().any(|axis| axis.len == 0)
        {
            return Ok(None);
        }
        let bits = shape.bits[0].max(shape.bits[1]) as usize;
        let mut at_bit = vec![Vec::new(); bits];
        let mut residual = None;
        for (
            axis_index,
            &Axis {
                len,
                strides: [left, right],
                kind,
            },
        ) in shape.axes.iter().enumerate()
        {
            if kind != WeightKind::Dense && len.is_power_of_two() && (left != 0 || right != 0) {
                for coordinate_bit in 0..len.trailing_zeros() as usize {
                    let (start, pair) =
                        introduction_geometry(left as usize, right as usize, coordinate_bit)?;
                    let Some(bucket) = at_bit.get_mut(start) else {
                        // Forced high-bit branches remain on the online path.
                        return Ok(None);
                    };
                    bucket.push(((axis_index << 8 | coordinate_bit) as u16, pair));
                }
            } else if residual.replace(axis_index).is_some() {
                return Ok(None);
            }
        }
        let seeds = residual.map_or(1, |axis| shape.axes[axis].len as usize);
        if seeds > MAX_STATES {
            return Ok(None);
        }
        let mut states = Vec::with_capacity(seeds);
        for coordinate in 0..seeds {
            let mut pair = (shape.offsets[0] as usize, shape.offsets[1] as usize);
            if let Some(axis) = residual {
                pair.0 = super::super::checked_axis_offset(
                    pair.0,
                    shape.axes[axis].strides[0] as usize,
                    coordinate,
                    "left",
                )?;
                pair.1 = super::super::checked_axis_offset(
                    pair.1,
                    shape.axes[axis].strides[1] as usize,
                    coordinate,
                    "right",
                )?;
            }
            states.push(pair);
        }
        if states.iter().collect::<BTreeSet<_>>().len() != seeds {
            return Ok(None);
        }
        let mut graph = Self {
            shape,
            layers: Vec::with_capacity(bits),
            introductions: Vec::new(),
            edges: Vec::new(),
            peak: seeds,
            residual,
            seeds,
            terminal: None,
            // Include the residual-coordinate visit charge as well as seed emission.
            work: seeds + if residual.is_some() { seeds } else { 0 },
        };
        for (bit, introductions) in at_bit.iter().enumerate() {
            if introductions.len() > 4 {
                return Ok(None);
            }
            let mut choices = vec![(0usize, 0usize)];
            for &(reference, pair) in introductions {
                graph.introductions.push(reference);
                graph.work += choices.len() * 2;
                let mut next = Vec::with_capacity(choices.len() * 2);
                for &previous in &choices {
                    next.push(previous);
                    next.push((
                        previous.0.checked_add(pair.0).ok_or_else(|| {
                            AkitaError::InvalidInput("carry graph choice overflow".into())
                        })?,
                        previous.1.checked_add(pair.1).ok_or_else(|| {
                            AkitaError::InvalidInput("carry graph choice overflow".into())
                        })?,
                    ));
                }
                choices = next;
            }
            // All admitted fixed shapes have distinct choice addresses. A
            // collision needs a richer choice program, not a lost multiplicity.
            if choices.iter().collect::<BTreeSet<_>>().len() != choices.len() {
                return Ok(None);
            }
            let edge_count = states.len() * choices.len();
            if graph.edges.len() + edge_count > MAX_EDGES {
                return Ok(None);
            }
            graph.work += edge_count;
            let mut destinations = BTreeMap::new();
            for (source, &state) in states.iter().enumerate() {
                for (choice, &addition) in choices.iter().enumerate() {
                    let (destination, parity) = carry_step(state, addition)?;
                    if (bit >= shape.bits[0] as usize && parity / 2 != 0)
                        || (bit >= shape.bits[1] as usize && parity % 2 != 0)
                    {
                        continue;
                    }
                    let new_slot = destinations.len();
                    if new_slot == MAX_STATES && !destinations.contains_key(&destination) {
                        return Ok(None);
                    }
                    let (slot, first) = match destinations.entry(destination) {
                        Entry::Vacant(entry) => {
                            entry.insert(new_slot);
                            (new_slot, true)
                        }
                        Entry::Occupied(entry) => (*entry.get(), false),
                    };
                    graph.edges.push(
                        source as u32
                            | (slot as u32) << 10
                            | (choice as u32) << 20
                            | (parity as u32) << 24
                            | u32::from(first) << 26,
                    );
                }
            }
            states = vec![(0, 0); destinations.len()];
            for (pair, slot) in destinations {
                states[slot] = pair;
            }
            graph.peak = graph.peak.max(states.len());
            graph.layers.push([
                graph.edges.len() as u32,
                graph.introductions.len() as u32,
                states.len() as u32,
            ]);
        }
        graph.terminal = states.iter().position(|&pair| pair == (0, 0));
        Ok(Some(graph))
    }

    fn write_source(&self, output: &mut String) -> Result<(), AkitaError> {
        let axes = self
            .shape
            .axes
            .iter()
            .map(|axis| {
                format!(
                    "Axis {{ len: {}, strides: {:?}, kind: WeightKind::{:?} }}",
                    axis.len, axis.strides, axis.kind
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        writeln!(output,
            "Graph {{ shape: Shape {{ bits: {:?}, offsets: {:?}, axes: &[{}] }}, layers: &{:?}, introductions: &{:?}, edges: &{:?}, peak: {}, residual: {:?}, seeds: {}, terminal: {:?}, work: {} }},",
            self.shape.bits, self.shape.offsets, axes, self.layers,
            self.introductions, self.edges, self.peak, self.residual, self.seeds,
            self.terminal, self.work,
        ).map_err(|error| AkitaError::Internal(error.to_string()))
    }
}

/// Emit deterministic Rust constants from public normalized geometry alone.
/// The caller selects the supported fixed shapes; no field or proof is an input.
/// Unsupported or over-budget shapes return an error so an emitter cannot
/// silently claim complete coverage. Runtime misses use the online evaluator.
fn emit_source(shapes: &[Shape<'_>]) -> Result<String, AkitaError> {
    let mut output = String::from("pub(super) static GRAPHS: &[Graph<'static>] = &[\n");
    for &shape in shapes {
        let residual = residual::ResidualShape::new(shape)?;
        let graph = OwnedGraph::compile(residual.view())?
            .ok_or_else(|| AkitaError::InvalidInput("unsupported carry graph shape".into()))?;
        graph.write_source(&mut output)?;
    }
    output.push_str("];\n");
    Ok(output)
}

/// Emit the complete frozen packing census for inner trace exponent 12 or 20.
/// Geometry is pinned in `compile/shapes.rs`; no proof, setup field value, or
/// challenge can affect the generated bytes. Other exponents are unsupported.
pub fn emit_fixed_source(exponent: u32) -> Result<String, AkitaError> {
    let shapes = match exponent {
        12 => shapes::SHAPES12,
        20 => shapes::SHAPES20,
        _ => {
            return Err(AkitaError::InvalidInput(
                "unsupported carry graph exponent".into(),
            ))
        }
    };
    emit_source(shapes)
}

#[cfg(test)]
mod tests;
