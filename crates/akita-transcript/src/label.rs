//! Compile-time transcript labels.

#[cfg(feature = "logging-transcript")]
mod imp {
    /// Developer-facing transcript label captured by logging builds.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Label {
        /// Stable semantic label for the transcript event.
        pub tag: &'static str,
        /// Source file where the label macro was expanded.
        pub file: &'static str,
        /// Source line where the label macro was expanded.
        pub line: u32,
    }
}

#[cfg(not(feature = "logging-transcript"))]
mod imp {
    /// Zero-sized transcript label for production and ordinary test builds.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Label;
}

pub use imp::Label;

/// Build a transcript label.
#[cfg(feature = "logging-transcript")]
#[macro_export]
macro_rules! label {
    ($tag:literal) => {
        $crate::Label {
            tag: $tag,
            file: file!(),
            line: line!(),
        }
    };
}

/// Build a transcript label.
#[cfg(not(feature = "logging-transcript"))]
#[macro_export]
macro_rules! label {
    ($tag:literal) => {
        $crate::Label
    };
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "logging-transcript"))]
    use super::Label;

    #[cfg(not(feature = "logging-transcript"))]
    #[test]
    fn label_is_zst_without_logging_feature() {
        assert_eq!(core::mem::size_of::<Label>(), 0);
    }

    #[cfg(feature = "logging-transcript")]
    #[test]
    fn label_captures_source_location_with_logging_feature() {
        let label = crate::label!("test_label_capture");
        assert_eq!(label.tag, "test_label_capture");
        assert!(label.file.ends_with("label.rs"));
        assert!(label.line > 0);
    }
}
