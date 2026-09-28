//! The Akita shader library: `jolt-metal`'s field headers, Akita's headers
//! and kernels, and one explicit instantiation per supported shape.

use std::fmt::Write;

use jolt_metal::runtime::{Device, LibrarySpec, Pipeline, ShaderLibrary};
use jolt_metal::shaders::FIELD_HEADERS;

use crate::error::AkitaMetalError;

/// Ring degrees with compiled transform kernels. This covers Q128; Q32 and
/// Q64 also admit degree 2048, which is not compiled here.
pub const RING_DEGREES: [usize; 5] = [64, 128, 256, 512, 1024];

/// Akita's MSL headers, in dependency order. They follow `jolt-metal`'s
/// field headers in a library, since runtime-compiled source cannot
/// `#include` repository paths.
pub const HEADERS: [(&str, &str); 2] = [
    ("akita/mont.h", include_str!("../shaders/akita/mont.h")),
    ("akita/ntt.h", include_str!("../shaders/akita/ntt.h")),
];

/// Kernel sources.
const KERNELS: [(&str, &str); 1] = [(
    "akita/ntt.metal",
    include_str!("../shaders/akita/ntt.metal"),
)];

/// Kernel templates instantiated once per ring degree.
const RING_DEGREE_TEMPLATES: [&str; 2] = ["akita_ntt_forward", "akita_ntt_inverse"];

/// One explicit instantiation `namespace::template<args>` under a host name.
///
/// `jolt-metal` instantiates templates over one field type; Akita's kernels
/// also take shape parameters, so this crate writes its own instantiations
/// and declares each host name as a kernel.
struct Instance {
    template: &'static str,
    args: String,
    host_name: String,
}

/// The host name of a ring-degree template instance.
pub(crate) fn ring_degree_kernel(template: &str, ring_degree: usize) -> String {
    format!("{template}_d{ring_degree}")
}

fn instances() -> Vec<Instance> {
    RING_DEGREE_TEMPLATES
        .iter()
        .flat_map(|&template| {
            RING_DEGREES.iter().map(move |&ring_degree| Instance {
                template,
                args: ring_degree.to_string(),
                host_name: ring_degree_kernel(template, ring_degree),
            })
        })
        .collect()
}

fn library_spec() -> LibrarySpec {
    let instances = instances();
    let mut instantiations = String::new();
    for instance in &instances {
        // Writing to a `String` cannot fail.
        let _ = writeln!(
            instantiations,
            "template [[host_name(\"{host}\")]] [[kernel]] \
             decltype(akita::{template}<{args}>) akita::{template}<{args}>;",
            host = instance.host_name,
            template = instance.template,
            args = instance.args,
        );
    }
    let spec = FIELD_HEADERS
        .iter()
        .chain(&HEADERS)
        .chain(&KERNELS)
        .fold(LibrarySpec::new(), |spec, (name, text)| {
            spec.source(name, text)
        })
        .source("akita instantiations", &instantiations);
    instances
        .iter()
        .fold(spec, |spec, instance| spec.kernel(&instance.host_name))
}

/// A Metal device with every Akita pipeline compiled.
///
/// Construction compiles the whole library, so shader errors surface here
/// and never mid-proof.
pub struct AkitaMetal {
    device: Device,
    library: ShaderLibrary,
}

impl AkitaMetal {
    /// Opens the system default device and compiles every kernel.
    pub fn new() -> Result<Self, AkitaMetalError> {
        let device = Device::system_default()?;
        let library = ShaderLibrary::compile(&device, &library_spec())?;
        Ok(Self { device, library })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub(crate) fn pipeline(&self, host_name: &str) -> Result<&Pipeline, AkitaMetalError> {
        Ok(self.library.pipeline(host_name)?)
    }
}
