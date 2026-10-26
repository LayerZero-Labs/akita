pub(super) fn json_value_start(bytes: &[u8], key: &[u8], occurrence: usize) -> usize {
    let key_start = bytes
        .windows(key.len())
        .enumerate()
        .filter_map(|(offset, window)| (window == key).then_some(offset))
        .nth(occurrence)
        .expect("artifact key");
    let mut cursor = key_start + key.len();
    while bytes.get(cursor).is_some_and(u8::is_ascii_whitespace) {
        cursor += 1;
    }
    assert_eq!(bytes.get(cursor), Some(&b':'));
    cursor += 1;
    while bytes.get(cursor).is_some_and(u8::is_ascii_whitespace) {
        cursor += 1;
    }
    cursor
}

pub(super) fn mutate_first_json_digit(bytes: &mut [u8], key: &[u8]) {
    let value_start = json_value_start(bytes, key, 0);
    let digit = bytes[value_start..]
        .iter()
        .position(u8::is_ascii_digit)
        .map(|relative| value_start + relative)
        .expect("numeric artifact value");
    bytes[digit] = if bytes[digit] == b'9' {
        b'8'
    } else {
        bytes[digit] + 1
    };
}

pub(super) fn mutate_first_json_number_without_changing_width(bytes: &mut [u8], key: &[u8]) {
    let value_start = json_value_start(bytes, key, 0);
    let number_start = bytes[value_start..]
        .iter()
        .position(u8::is_ascii_digit)
        .map(|relative| value_start + relative)
        .expect("numeric artifact value");
    let number_end = bytes[number_start..]
        .iter()
        .position(|byte| !byte.is_ascii_digit())
        .map(|relative| number_start + relative)
        .unwrap_or(bytes.len());
    let last_digit = bytes
        .get_mut(number_end - 1)
        .expect("nonempty numeric artifact value");
    *last_digit = if *last_digit == b'0' {
        b'1'
    } else {
        *last_digit - 1
    };
}

pub(super) fn json_row_ranges(bytes: &[u8]) -> Vec<std::ops::Range<usize>> {
    let rows_start = json_value_start(bytes, b"\"rows\"", 0);
    assert_eq!(bytes.get(rows_start), Some(&b'['));

    let mut ranges = Vec::new();
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    let mut row_start = None;
    for (relative, &byte) in bytes[rows_start + 1..].iter().enumerate() {
        let offset = rows_start + 1 + relative;
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'{' => {
                if depth == 0 {
                    row_start = Some(offset);
                }
                depth += 1;
            }
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    ranges.push(row_start.take().expect("artifact row start")..offset + 1);
                }
            }
            b']' if depth == 0 => break,
            _ => {}
        }
    }
    ranges
}

pub(super) fn pad_first_row(bytes: &[u8], padding_len: usize) -> Vec<u8> {
    let first_row = json_row_ranges(bytes)
        .into_iter()
        .next()
        .expect("first row");
    let insert_at = first_row.end - 1;
    let field_prefix = b",\"untrusted_padding\":\"";
    let mut padded = Vec::with_capacity(bytes.len() + field_prefix.len() + padding_len + 1);
    padded.extend_from_slice(&bytes[..insert_at]);
    padded.extend_from_slice(field_prefix);
    padded.extend(std::iter::repeat_n(b'x', padding_len));
    padded.push(b'"');
    padded.extend_from_slice(&bytes[insert_at..]);
    padded
}

pub(super) fn duplicate_first_json_row(bytes: &[u8]) -> Vec<u8> {
    let ranges = json_row_ranges(bytes);
    let first = ranges.first().expect("complete first artifact row");
    let insert_at = ranges.last().expect("complete last artifact row").end;
    let separator = &bytes[first.end..ranges.get(1).expect("second artifact row").start];
    let mut duplicated = Vec::with_capacity(bytes.len() + first.len() + separator.len());
    duplicated.extend_from_slice(&bytes[..insert_at]);
    duplicated.extend_from_slice(separator);
    duplicated.extend_from_slice(&bytes[first.clone()]);
    duplicated.extend_from_slice(&bytes[insert_at..]);
    duplicated
}

pub(super) fn swap_first_two_json_rows(bytes: &[u8]) -> Vec<u8> {
    let ranges = json_row_ranges(bytes);
    let first = ranges.first().expect("first artifact row");
    let second = ranges.get(1).expect("second artifact row");
    let separator = &bytes[first.end..second.start];
    let mut swapped = Vec::with_capacity(bytes.len());
    swapped.extend_from_slice(&bytes[..first.start]);
    swapped.extend_from_slice(&bytes[second.clone()]);
    swapped.extend_from_slice(separator);
    swapped.extend_from_slice(&bytes[first.clone()]);
    swapped.extend_from_slice(&bytes[second.end..]);
    swapped
}

pub(super) fn replace_family_name(bytes: &[u8], family_name: &str) -> Vec<u8> {
    let value_start = json_value_start(bytes, b"\"family_name\"", 0);
    assert_eq!(bytes.get(value_start), Some(&b'"'));
    let value_end = bytes[value_start + 1..]
        .iter()
        .position(|byte| *byte == b'"')
        .map(|relative| value_start + 1 + relative)
        .expect("family name terminator");
    let mut replaced = Vec::with_capacity(bytes.len() + family_name.len());
    replaced.extend_from_slice(&bytes[..value_start + 1]);
    replaced.extend_from_slice(family_name.as_bytes());
    replaced.extend_from_slice(&bytes[value_end..]);
    replaced
}

pub(super) fn replace_rows_with_empty_objects(bytes: &[u8], row_count: usize) -> Vec<u8> {
    let rows_start = json_value_start(bytes, b"\"rows\"", 0);
    assert_eq!(bytes.get(rows_start), Some(&b'['));
    let ranges = json_row_ranges(bytes);
    let rows_end = bytes[ranges.last().expect("last artifact row").end..]
        .iter()
        .position(|byte| *byte == b']')
        .map(|relative| ranges.last().expect("last artifact row").end + relative)
        .expect("rows array terminator");
    let mut replaced = Vec::with_capacity(bytes.len() + row_count * 3);
    replaced.extend_from_slice(&bytes[..rows_start + 1]);
    for index in 0..row_count {
        if index != 0 {
            replaced.push(b',');
        }
        replaced.extend_from_slice(b"{}");
    }
    replaced.extend_from_slice(&bytes[rows_end..]);
    replaced
}

pub(super) fn empty_nth_fold_group_list(bytes: &[u8], occurrence: usize) -> Vec<u8> {
    let array_start = json_value_start(bytes, b"\"entries\"", occurrence);
    assert_eq!(bytes.get(array_start), Some(&b'['));

    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    let array_end = bytes[array_start..]
        .iter()
        .enumerate()
        .find_map(|(relative, &byte)| {
            if in_string {
                if escaped {
                    escaped = false;
                } else if byte == b'\\' {
                    escaped = true;
                } else if byte == b'"' {
                    in_string = false;
                }
                return None;
            }
            match byte {
                b'"' => in_string = true,
                b'[' => depth += 1,
                b']' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(array_start + relative);
                    }
                }
                _ => {}
            }
            None
        })
        .expect("complete fold group list");

    let mut malformed = Vec::with_capacity(bytes.len() - (array_end - array_start - 1));
    malformed.extend_from_slice(&bytes[..array_start]);
    malformed.extend_from_slice(b"[]");
    malformed.extend_from_slice(&bytes[array_end + 1..]);
    malformed
}
