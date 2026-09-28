//! The Akita shader library: `jolt-metal`'s field headers, Akita's headers
//! and kernels, and one explicit instantiation per supported shape.

use std::fmt::Write;

use jolt_metal::runtime::{Device, LibrarySpec, Pipeline, ShaderLibrary};
use jolt_metal::shaders::FIELD_HEADERS;

use crate::error::AkitaMetalError;

/// Ring degrees with compiled transform kernels. This covers Q128; Q32 and
/// Q64 also admit degree 2048, which is not compiled here.
pub const RING_DEGREES: [usize; 5] = [64, 128, 256, 512, 1024];

/// The host name of a template instance at one ring degree, with an
/// optional suffix for its remaining arguments.
pub(crate) fn ring_degree_kernel(template: &str, ring_degree: usize) -> String {
    format!("{template}_d{ring_degree}")
}

/// Akita's MSL headers, in dependency order. They follow `jolt-metal`'s
/// field headers in a library, since runtime-compiled source cannot
/// `#include` repository paths.
pub const HEADERS: [(&str, &str); 4] = [
    ("akita/mont.h", include_str!("../shaders/akita/mont.h")),
    ("akita/ntt.h", include_str!("../shaders/akita/ntt.h")),
    ("akita/crt.h", include_str!("../shaders/akita/crt.h")),
    (
        "akita/decompose.h",
        include_str!("../shaders/akita/decompose.h"),
    ),
];

/// Kernel sources.
const KERNELS: [(&str, &str); 5] = [
    (
        "akita/ntt.metal",
        include_str!("../shaders/akita/ntt.metal"),
    ),
    (
        "akita/crt.metal",
        include_str!("../shaders/akita/crt.metal"),
    ),
    (
        "akita/matvec.metal",
        include_str!("../shaders/akita/matvec.metal"),
    ),
    (
        "akita/decompose.metal",
        include_str!("../shaders/akita/decompose.metal"),
    ),
    (
        "akita/onehot.metal",
        include_str!("../shaders/akita/onehot.metal"),
    ),
];

/// One explicit instantiation `namespace::template<args>` under a host name.
///
/// `jolt-metal` instantiates templates over one field type; Akita's kernels
/// also take shape parameters, so this crate writes its own instantiations
/// and declares each host name as a kernel. Each kernel module lists its
/// instances next to the code that dispatches them.
pub(crate) struct Instance {
    pub(crate) template: &'static str,
    pub(crate) args: String,
    pub(crate) host_name: String,
}

fn instances() -> Vec<Instance> {
    crate::ntt::instances()
        .into_iter()
        .chain(crate::crt::instances())
        .chain(crate::matvec::instances())
        .chain(crate::decompose::instances())
        .chain(crate::onehot::instances())
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
        // The compiled library is the capability registry. Generic public
        // entry points may request an uninstantiated field/shape combination;
        // that is a setup limitation, not a device execution fault.
        self.library
            .pipeline(host_name)
            .map_err(|error| match error {
                jolt_metal::MetalError::UnknownPipeline { name } => {
                    AkitaMetalError::Shape(format!("no compiled kernel for {name}"))
                }
                other => other.into(),
            })
    }
}
