#![no_std]
#![no_main]

use core::{hint::black_box, panic::PanicInfo};
use esp_hal::time::Instant;
use esp_println::println;
use ruv_fann_esp32_c5::{score, CAP_Q10, MARGIN_Q10, Q10_ONE};

esp_bootloader_esp_idf::esp_app_desc!();

#[esp_hal::main]
fn main() -> ! {
    esp_hal::init(esp_hal::Config::default());

    let start = Instant::now();
    let mut cases = 0;
    let first_hinge = i32::from(MARGIN_Q10);
    let second_hinge = first_hinge + i32::from(CAP_Q10);
    for local in (0..=Q10_ONE).step_by(32) {
        for adjacent in (0..=Q10_ONE).step_by(32) {
            let difference = i32::from(local) - i32::from(adjacent);
            let expected = (difference - first_hinge).max(0) - (difference - second_hinge).max(0);
            assert_eq!(
                score(black_box(local), black_box(adjacent)),
                Some(expected as u16)
            );
            cases += 1;
        }
    }
    assert_eq!(score(Q10_ONE + 1, 0), None);
    assert_eq!(score(0, Q10_ONE + 1), None);
    let elapsed_us = start.elapsed().as_micros();
    println!(
        "ruvfann-c5 PASS cases={cases} margin_q10={MARGIN_Q10} cap_q10={CAP_Q10} self_test_us={elapsed_us}"
    );
    idle()
}

#[panic_handler]
fn panic(info: &PanicInfo<'_>) -> ! {
    println!("ruvfann-c5 FAIL {info}");
    idle()
}

fn idle() -> ! {
    loop {
        esp_hal::riscv::asm::wfi();
    }
}
