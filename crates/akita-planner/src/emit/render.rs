//! Generated-file rendering and module wiring orchestration.

use super::*;

fn emit_mod_wiring(specs: &[EmitSpec]) -> Result<String, String> {
    let mut declarations = String::new();
    let mut accessors = String::new();
    let mut seen = std::collections::BTreeSet::new();
    for spec in specs {
        if !seen.insert(spec.module_name) {
            continue;
        }
        let module_name = spec.module_name;
        let precommitted_module_name = precommitted_profiles_module_name(spec);
        let feat = spec.schedule_feature;
        writeln!(declarations, "#[cfg(feature = \"{feat}\")]").map_err(|e| e.to_string())?;
        writeln!(declarations, "pub mod {module_name};").map_err(|e| e.to_string())?;
        writeln!(declarations, "#[cfg(feature = \"{feat}\")]").map_err(|e| e.to_string())?;
        writeln!(declarations, "pub mod {precommitted_module_name};").map_err(|e| e.to_string())?;
        accessors.push_str(&emit_table_accessor(spec)?);
        accessors.push('\n');
    }
    declarations.push('\n');
    declarations.push_str(&accessors);
    Ok(declarations)
}

fn table_fn_name(module_name: &str) -> String {
    format!("{module_name}_table")
}

fn emit_table_accessor(spec: &EmitSpec) -> Result<String, String> {
    let fn_name = table_fn_name(spec.module_name);
    let feat = spec.schedule_feature;
    let module_name = spec.module_name;
    let precommitted_module_name = precommitted_profiles_module_name(spec);
    let const_name = spec.const_name;
    let precommitted_profiles_const = precommitted_profiles_const_name(spec);
    Ok(format!(
        "#[cfg(feature = \"{feat}\")]\n\
         pub fn {fn_name}() -> GeneratedScheduleTable {{\n    GeneratedScheduleTable {{\n        entries: {module_name}::{const_name},\n        precommitted_profiles: {precommitted_module_name}::{precommitted_profiles_const},\n        identity: {module_name}::CATALOG_IDENTITY,\n    }}\n}}\n"
    ))
}

fn replace_between_markers(
    content: &str,
    begin: &str,
    end: &str,
    replacement: &str,
) -> Result<String, String> {
    let start = content
        .find(begin)
        .ok_or_else(|| format!("missing generated marker `{begin}`"))?
        + begin.len();
    let end_pos = content
        .find(end)
        .ok_or_else(|| format!("missing generated marker `{end}`"))?;
    if end_pos < start {
        return Err(format!(
            "generated markers `{begin}` and `{end}` are out of order"
        ));
    }
    let mut out = String::new();
    out.push_str(&content[..start]);
    out.push('\n');
    out.push_str(replacement.trim_end());
    out.push('\n');
    out.push_str(&content[end_pos..]);
    Ok(out)
}

/// One fully rendered generated file awaiting publication.
#[derive(Debug)]
pub struct GeneratedOutput {
    pub(super) destination: PathBuf,
    pub(super) body: String,
}

fn render_family_outputs(spec: &EmitSpec) -> Result<[GeneratedOutput; 2], String> {
    Ok([
        GeneratedOutput {
            destination: spec.output_dir.join(format!("{}.rs", spec.module_name)),
            body: emit_family_module(spec)?,
        },
        GeneratedOutput {
            destination: spec
                .output_dir
                .join(format!("{}.rs", precommitted_profiles_module_name(spec))),
            body: emit_precommitted_profiles_module(spec)?,
        },
    ])
}

/// Render every family module, precommit registry, and optional wiring update.
///
/// No destination is modified unless the complete batch renders successfully
/// and is later passed to [`publish_generated_outputs`].
pub fn render_generated_outputs(
    specs: &[EmitSpec],
    wiring_specs: &[EmitSpec],
    mod_path: Option<&Path>,
) -> Result<Vec<GeneratedOutput>, String> {
    let workers = schedule_generation_worker_count(specs.len());
    let rendered =
        bounded_parallel_filter_map(specs, workers, |spec| render_family_outputs(spec).map(Some))?;
    let mut outputs = rendered.into_iter().flatten().collect::<Vec<_>>();
    if let Some(mod_path) = mod_path {
        let mod_src = fs::read_to_string(mod_path)
            .map_err(|error| format!("read {}: {error}", mod_path.display()))?;
        let mod_wiring = emit_mod_wiring(wiring_specs)?;
        outputs.push(GeneratedOutput {
            destination: mod_path.to_path_buf(),
            body: replace_between_markers(&mod_src, MOD_WIRING_BEGIN, MOD_WIRING_END, &mod_wiring)?,
        });
    }
    Ok(outputs)
}

#[cfg(test)]
mod tests {
    use super::render_generated_outputs;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_NONCE: AtomicU64 = AtomicU64::new(0);

    fn test_dir(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "akita-generated-publish-{label}-{}-{}",
            std::process::id(),
            TEST_NONCE.fetch_add(1, Ordering::Relaxed)
        ))
    }

    #[test]
    fn render_failure_does_not_touch_existing_wiring() {
        let dir = test_dir("render-failure");
        fs::create_dir_all(&dir).expect("create test directory");
        let mod_path = dir.join("mod.rs");
        let original = "pub mod hand_written;\n";
        fs::write(&mod_path, original).expect("write wiring fixture");

        let error = render_generated_outputs(&[], &[], Some(&mod_path))
            .expect_err("missing wiring markers must fail rendering");
        assert!(error.contains("missing generated marker"));
        assert_eq!(
            fs::read_to_string(&mod_path).expect("read wiring fixture"),
            original
        );
        fs::remove_dir_all(dir).expect("remove test directory");
    }
}
