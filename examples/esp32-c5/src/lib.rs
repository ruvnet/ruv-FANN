#![no_std]

include!(concat!(env!("OUT_DIR"), "/model.rs"));

pub const Q10_ONE: u16 = 1 << 10;

/// Scores local and adjacent evidence in Q10 units. Values above one are rejected.
pub fn score(local: u16, adjacent: u16) -> Option<u16> {
    if local > Q10_ONE || adjacent > Q10_ONE {
        return None;
    }
    let difference = i32::from(local) - i32::from(adjacent) - i32::from(MARGIN_Q10);
    Some(difference.clamp(0, i32::from(CAP_Q10)) as u16)
}
